"""Verify saved integration receipts, scientific bytes and matched consumed state.

No model is constructed, restored, trained or sampled. Missing original artifacts
block verification; they never authorize rerunning an experiment.
"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish import ARCHIVE, ROLES, arguments, collect, mechanism_stats, require
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.rng import NamedStreams
from experiments.forge.sources import verify_snapshot
from experiments.forge.state import state_digest

DELTAS = {'constraint_geometry_mode', 'kinetic_transport_weight',
          'kinetic_transport_local_weight', 'kinetic_transport_projections'}


def equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(equal(left[k], right[k]) for k in left))
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def conditions(request):
    return {tid: {key: value for key, value in card.items() if key not in {'field_ownership', 'preflight_blockers'}}
            for tid, card in request['tasks'].items()}


def scientific_source(source):
    """Commit and archive location are provenance, not scientific file identity."""
    return {key: source[key] for key in ('digest', 'files')}


def metadata(entry):
    saved, row = entry['saved'], entry['row']
    actual = saved.get('applied', saved)
    proof = {key: actual.get(key, row.get(key)) for key in ('initializer', 'initialization', 'prior')}
    require(all(proof[key] is not None for key in proof), f'{row["task_id"]}: missing actual initialization/prior proof')
    host = row.get('evidence', {}).get('host', {})
    if isinstance(host, dict):
        proof['initial_host_model_hashes'] = {name: model.get('initial_state_sha256')
            for name, model in host.get('models', {}).items() if 'initial_state_sha256' in model}
        proof['initial_host_prior_hash'] = host.get('initial_prior_sha256')
    return proof


def non_eval(saved):
    streams = saved['streams']
    probe = NamedStreams(streams['manifest']['seed'], version=streams['manifest']['version'])
    probe.validate_state_dict(streams)
    require(streams['manifest']['seed'] == 0, 'Nonzero protocol seed in consumed checkpoint')
    bindings = streams['manifest']['bindings']
    keys = {key for key in streams['states'] if bindings[key]['family'] != 'eval'}
    return {key: bindings[key] for key in keys}, {key: streams['states'][key] for key in keys}


def own_checkpoints(data):
    checks = []
    for entry in data['final']:
        if entry['row']['gate_status'] not in {'PASS', 'FAIL'}:
            continue
        for dependency in entry['task'].get('dependencies', []):
            if not isinstance(dependency, dict) or dependency.get('kind') != 'checkpoint':
                continue
            continuity = entry['row']['evidence'].get('continuity') or {}
            require(continuity.get('restored_exactly') is True and continuity.get('history_reset') is False,
                    f'{entry["item"]["role"]}/{entry["row"]["task_id"]}: missing exact own-checkpoint restore')
            parents = [p for p in data['final'] if p['item']['role'] == entry['item']['role']
                and p['item']['scope'] == entry['item']['scope']
                and p['row']['task_id'] == dependency['task']
                and p['row']['compatibility_key'] == continuity.get('parent_compatibility_key')
                and p['request']['candidate_revision'] == continuity.get('parent_candidate_revision')
                and p['row']['evidence'].get('checkpoint', {}).get('sha256') == continuity.get('parent_checkpoint_sha256')
                and p['row']['evidence'].get('checkpoint', {}).get('state_sha256') == continuity.get('parent_state_sha256')]
            unique = {parent['item']['attempt_id']: parent for parent in parents}
            require(len(unique) == 1, f'{entry["row"]["task_id"]}: unique own certified producer required')
            parent = next(iter(unique.values()))
            require(parent['row']['gate_status'] == 'PASS'
                    and parent['request']['candidate_revision'] == entry['request']['candidate_revision'],
                    f'{entry["row"]["task_id"]}: checkpoint producer is not this candidate passing state')
            evidence = parent['row']['evidence']
            descriptor = evidence['checkpoint']
            root = Path(evidence['artifact_root']).resolve()
            verify_artifacts(root, evidence['artifact_manifest'])
            path = (root / descriptor['path']).resolve()
            require(path.is_relative_to(root) and file_hash(path) == descriptor['sha256']
                    and descriptor['sha256'] == continuity.get('parent_checkpoint_sha256'),
                    f'{entry["row"]["task_id"]}: consumed checkpoint bytes differ from own producer')
            saved = torch.load(path, map_location='cpu', weights_only=False)
            require(state_digest(saved) == descriptor['state_sha256'] == continuity.get('parent_state_sha256'),
                    f'{entry["row"]["task_id"]}: consumed checkpoint state differs')
            checks.append(dict(scope=entry['item']['scope'], role=entry['item']['role'], task_id=entry['row']['task_id'],
                parent_task_id=dependency['task'], parent_attempt_id=parent['item']['attempt_id'],
                parent_scope=parent['item']['scope'], parent_source_commit=parent['request']['source'].get('origin_commit'),
                parent_checkpoint_sha256=descriptor['sha256'], parent_state_sha256=descriptor['state_sha256'],
                prefix_steps=continuity.get('prefix_steps'), restored_exactly=True, history_reset=False))
    return checks


def source_bindings(data, root):
    checked, snapshots, requests = {}, set(), []
    for context in data['scopes']:
        reference = next(iter(context['submissions'].values()))['request']
        for role, sub in context['submissions'].items():
            request = sub['request']
            require(scientific_source(request['source']) == scientific_source(reference['source']),
                    f'{context["scope"]}/{role}: scientific source differs')
            for key in ('runtime', 'protocol', 'compute_profiles'):
                require(request.get(key) == reference.get(key), f'{context["scope"]}/{role}: {key} differs')
            require(conditions(request) == conditions(reference), f'{context["scope"]}/{role}: task conditions differ')
            require(request['protocol']['seed'] == 0, 'Expected protocol seed 0')
            files, source = request['source']['files'], request['source']
            require(stable_hash(files) == source['digest'], f'{role}: source manifest digest differs')
            if source['digest'] not in checked:
                for relative, digest in files.items():
                    require(file_hash(root / relative) == digest, f'Published scientific bytes changed: {relative}')
                checked[source['digest']] = dict(source_digest=source['digest'], source_commits=[],
                    frozen_snapshots=[], snapshot_manifests=[], scientific_files_verified=len(files), published_files_equal=True)
            item = checked[source['digest']]
            origin = source.get('origin_commit')
            if origin is not None and origin not in item['source_commits']:
                item['source_commits'].append(origin)
            snapshot = Path(source['snapshot_path']).resolve()
            identity = (source['digest'], str(snapshot))
            if identity not in snapshots:
                verify_snapshot(snapshot, source)
                stored = read_json(snapshot / 'forge-source.json')
                require(scientific_source(stored) == scientific_source(source),
                        f'{context["scope"]}/{role}: snapshot manifest scientific identity differs')
                snapshots.add(identity)
                item['frozen_snapshots'].append(str(snapshot))
                item['snapshot_manifests'].append(dict(path=str(snapshot / 'forge-source.json'),
                    sha256=file_hash(snapshot / 'forge-source.json'), origin_commit=stored.get('origin_commit')))
            requests.append(dict(scope=context['scope'], role=role, request_id=request['request_id'],
                candidate_revision=request['candidate_revision'], source_digest=source['digest'],
                source_commit=origin, frozen_snapshot=str(snapshot)))
    for item in checked.values():
        item['source_commits'].sort()
        item['frozen_snapshots'].sort()
        item['source_commit'] = item['source_commits'][0] if len(item['source_commits']) == 1 else None
        item['frozen_snapshot'] = item['frozen_snapshots'][0] if len(item['frozen_snapshots']) == 1 else None
    return list(checked.values()), requests


def original_task_contracts(data, root):
    path = root / 'reports/forge/bcap-develop-integration/original-task-contracts.json'
    require(path.is_file(), 'Original full-suite task contract archive is required')
    archive = read_json(path)
    required_ids = {a['task'] for a in data['requirements']}
    for tid in sorted(required_ids):
        archived = subprocess.run(['git', 'show', f'{archive["source_commit"]}:configs/forge/tasks/{tid}.json'],
                                  cwd=root, capture_output=True, text=True, check=True)
        require(json.loads(archived.stdout) == archive['tasks'][tid],
                f'{tid}: original task archive differs from its actual Git source')
    verified = {}
    for context in data['scopes']:
        for sub in context['submissions'].values():
            for tid, card in conditions(sub['request']).items():
                if tid not in required_ids:
                    continue
                require(tid in archive['tasks'], f'{tid}: original task archive is missing')
                original = archive['tasks'][tid]
                restored = deepcopy(card)
                if restored['execution'].get('transport_consumer'):
                    require(restored['execution']['transport_consumer'] == 'output_marginal_v1',
                            f'{tid}: undeclared component transport consumer')
                    contract = restored['execution'].pop('transport_contract')
                    require(contract['conditioning_in_distance'] is False and contract['new_paired_supervision'] is False
                            and contract['new_forward_passes'] is False and contract['new_random_draws'] is False,
                            f'{tid}: original information or sample law changed')
                    restored['execution'].pop('transport_consumer')
                    restored.pop('task_cohort', None)
                    if 'task_cohort' in original:
                        restored['task_cohort'] = original['task_cohort']
                for key in ('sources', 'evaluator_revision'):
                    restored['evaluation'].pop(key, None)
                    if key in original['evaluation']:
                        restored['evaluation'][key] = deepcopy(original['evaluation'][key])
                require(restored == original, f'{tid}: numerical conditions differ from original question')
                for relative, digest in card['evaluation'].get('sources', {}).items():
                    require(sub['request']['source']['files'][relative] == digest, f'{tid}: evaluator source binding differs')
                verified[tid] = stable_hash(original)
    require(len(verified) == 27, 'All 27 original numerical question contracts must be verified')
    return dict(source_commit=archive['source_commit'], contract_archive_sha256=file_hash(path),
                original_task_sha256=verified, original_git_cards_verified=27, numerical_conditions_preserved=True)


def saved_variants(entry):
    """Use actual saved clock branches separately from ordinary final checkpoints."""
    if entry['saved'] is not None:
        return [('final', entry)]
    task, row = entry['task'], entry['row']
    require(task['adapter'] == 'clockfree_audit' and entry['item']['importance'] != 'required',
            f'{row["task_id"]}: complete gate lacks consumed-state checkpoint')
    from experiments.forge.clockfree import _comparisons
    evidence, execution = row['evidence'], task['execution']
    root = Path(evidence['artifact_root']).resolve()
    verify_artifacts(root, evidence['artifact_manifest'])
    path = root / 'comparisons.pt'
    proof = torch.load(path, map_location='cpu', weights_only=True)
    require(proof.get('schema_version') == 1 and proof['execution'] == execution,
            f'{row["task_id"]}: clock artifact execution differs')
    names = {'reference', 'step_label', 'horizon', 'evaluation_cadence', 'restart'}
    require(set(proof['trajectories']) == set(proof['branch_initial']) == names,
            f'{row["task_id"]}: incomplete saved clock comparison branches')
    require(equal(proof['initial'], torch.load(root / 'initial.pt', map_location='cpu', weights_only=True)),
            f'{row["task_id"]}: serialized clock restart differs')
    require(stable_hash(_comparisons(proof)) == stable_hash(evidence['comparisons']),
            f'{row["task_id"]}: certified clock comparisons differ from saved tensors')
    variants = [('warmup', proof['initial'], execution['warmup_steps'])]
    for name in sorted(names):
        require(equal(proof['initial'], proof['branch_initial'][name]),
                f'{row["task_id"]}: {name} did not start from its identical own warmup state')
        trajectory = proof['trajectories'][name]
        require(len(trajectory) == execution['probe_steps'], f'{row["task_id"]}: incomplete {name} trajectory')
        variants.append((name, trajectory[-1], execution['warmup_steps'] + execution['probe_steps']))
    result = []
    for name, saved, updates in variants:
        # This is a reference into the existing certified artifact, not a new
        # checkpoint or an ordinary qualification result.
        item = dict(entry['item'], provenance_checkpoint=dict(completed_steps=updates,
            proof_kind='optional_clock_saved_branch', artifact_path=str(path), artifact_sha256=file_hash(path),
            branch=name, state_sha256=state_digest(saved)), mechanism_stats=mechanism_stats(saved))
        result.append((name, dict(entry, item=item, saved=saved)))
    return result


def matched_state(data):
    grouped = defaultdict(list)
    proofs, unmatched = [], []
    for entry in data['final']:
        if entry['row']['gate_status'] in {'PASS', 'FAIL'}:
            for variant, saved in saved_variants(entry):
                grouped[(entry['item']['scope'], entry['row']['task_id'], variant)].append(saved)
        else:
            unmatched.append(dict(scope=entry['item']['scope'], role=entry['item']['role'], task_id=entry['row']['task_id'],
                reason='Certified incomplete/invalid/blocked attempt cannot establish matched complete consumption',
                gate_status=entry['row']['gate_status']))
    for (scope, tid, variant), group in sorted(grouped.items()):
        if scope == 'research_diagnostic' and len(group) == 1:
            other = [entry for entry in grouped.get(('ordinary', tid, variant), [])
                     if entry['item']['role'] != group[0]['item']['role']]
            if other:
                reference = other[0]
                require(scientific_source(reference['request']['source']) == scientific_source(group[0]['request']['source']),
                        f'{scope}/{tid}: ordinary reference scientific files differ')
                for key in ('runtime', 'protocol'):
                    require(reference['request'][key] == group[0]['request'][key],
                            f'{scope}/{tid}: ordinary reference {key} differs')
                require(conditions(reference['request'])[tid] == conditions(group[0]['request'])[tid],
                        f'{scope}/{tid}: ordinary reference numerical task conditions differ')
                group = [*group, reference]
        require(len(group) <= 2, f'{scope}/{tid}: ambiguous arm comparison')
        first = group[0]
        initial = metadata(first)
        first_bindings, first_states = non_eval(first['saved'])
        proof = dict(scope=scope, task_id=tid, saved_variant=variant,
            proof_kind=first['item']['provenance_checkpoint'].get('proof_kind', 'training_checkpoint'),
            saved_state_references={p['item']['role']: p['item']['provenance_checkpoint'] for p in group},
            original_arm_scopes={p['item']['role']: p['item']['scope'] for p in group},
            original_arm_source_commits={p['item']['role']: p['request']['source'].get('origin_commit') for p in group},
            original_arm_source_digests={p['item']['role']: p['request']['source']['digest'] for p in group},
            completed_roles=[p['item']['role'] for p in group],
            initial_state_proof_sha256=state_digest(initial), non_eval_streams=len(first_states),
            completed_steps={p['item']['role']: p['item']['provenance_checkpoint']['completed_steps'] for p in group},
            all_completed_arms_present=len(group) == 2)
        for entry in group:
            guards = entry['row']['evidence'].get('guards', {})
            require(guards.get('unintended_rng_deviations', 0) == 0,
                    f'{scope}/{tid}: guard reports unintended RNG consumption')
            for audit in entry['row']['evidence'].get('rng_audits', []):
                require(audit.get('unintended_rng_deviations', 0) == 0, f'{scope}/{tid}: unintended RNG deviations')
            if 'applied' in entry['saved']:
                applied = entry['row'].get('applied', entry['row']['evidence'].get('applied'))
                require(applied is not None and stable_hash(applied) == stable_hash(entry['saved']['applied']),
                        f'{scope}/{tid}: applied component receipt differs from consumed state')
            require(equal(initial, metadata(entry)), f'{scope}/{tid}: actual initial models or priors differ')
            stats = entry['item']['mechanism_stats']
            protected = [value for path, value in stats.items() if path.endswith('.constraint_geometry')]
            consumers = [value for path, value in stats.items() if path.endswith('.component_transport')]
            if entry['item']['role'] == 'combined':
                updates = entry['item']['provenance_checkpoint']['completed_steps']
                require(protected and all(value['steps'] == updates for value in protected),
                        f'{scope}/{tid}: enabled projection was not consumed')
                if entry['task']['execution'].get('transport_consumer'):
                    require(consumers and all(value['calls'] == value['active_calls'] == updates for value in consumers),
                            f'{scope}/{tid}: enabled component transport was not consumed')
                    if entry['task']['adapter'] == 'transfer_behavior':
                        require(guards.get('hooks_exercised') is True and guards.get('component_transport_requested') is True
                                and guards.get('component_transport_active_calls') == updates,
                                f'{scope}/{tid}: component transport guards do not bind actual calls')
            else:
                require(not protected and all(value['active_calls'] == 0 for value in consumers),
                        f'{scope}/{tid}: inactive control consumed a new mechanism')
                require(not guards.get('component_transport_requested', False)
                        and guards.get('component_transport_active_calls', 0) == 0,
                        f'{scope}/{tid}: inactive control guard reports transport consumption')
            bindings, states = non_eval(entry['saved'])
            require(equal(first_bindings, bindings), f'{scope}/{tid}: named training-stream bindings differ')
            recipe = entry['row'].get('recipe', entry['saved'].get('applied', {}).get('recipe', {}))
            original = first['row'].get('recipe', first['saved'].get('applied', {}).get('recipe', {}))
            require({k: v for k, v in recipe.items() if k not in DELTAS}
                    == {k: v for k, v in original.items() if k not in DELTAS}, f'{scope}/{tid}: undeclared trainer delta')
            if len(set(proof['completed_steps'].values())) == 1:
                require(equal(first_states, states), f'{scope}/{tid}: consumed non-evaluation stream tensors differ')
                require(first['row']['evidence'].get('data_sha256') == entry['row']['evidence'].get('data_sha256'),
                        f'{scope}/{tid}: actual target batch sequence digest differs')
        proof.update(initialization_and_prior_equal=len(group) == 2,
            named_training_bindings_equal=len(group) == 2,
            consumed_non_eval_stream_tensors_equal=(len(group) == 2 and len(set(proof['completed_steps'].values())) == 1),
            consumption_comparison=('Verified every consumed non-eval stream tensor' if len(group) == 2 and
                len(set(proof['completed_steps'].values())) == 1 else 'Not comparable: missing complete arm or different own confirmed-prefix update counts'))
        proofs.append(proof)
    return proofs, unmatched


def main():
    parser = arguments(__doc__)
    parser.add_argument('--protected', type=Path, default=ARCHIVE / 'qualification-before.json')
    options = parser.parse_args()
    torch.set_num_threads(1)
    data = collect(options)
    root = options.repository.resolve()
    files, requests = source_bindings(data, root)
    original = original_task_contracts(data, root)
    matched, unmatched = matched_state(data)
    dependencies = own_checkpoints(data)
    protected = read_json(options.protected) if options.protected.is_file() else None
    if protected is not None:
        for relative, digest in protected.items():
            require(file_hash(root / relative) == digest, f'Historical qualification/telemetry changed: {relative}')
    result = dict(schema_version=1, qualification_input=False, status='PASS',
        partial=any(context['active'] for context in data['scopes']), requests=requests,
        frozen_source_checks=files, original_question_contracts=original,
        matched_completed_arm_checks=matched, uncomparable_final_attempts=unmatched,
        own_checkpoint_dependencies=dependencies, certified_attempts=len(data['attempts']),
        qualification_preservation=dict(checked=protected is not None, files_verified=len(protected or {}),
            reference=str(options.protected), reason=None if protected is not None else 'No before-snapshot supplied; preservation is not claimed'),
        optimizer_updates_added=0, sampling_draws_added=0,
        interpretation='Audit PASS validates retained evidence, not scientific quality or ordinary qualification.')
    atomic_json(options.output / 'audit.json', result)
    print(json.dumps(dict(event='audit_complete', certified_attempts=len(data['attempts']),
        matched_tasks=len({(p['scope'], p['task_id']) for p in matched}), matched_saved_states=len(matched),
        source_files=sum(p['scientific_files_verified'] for p in files),
        own_checkpoints=len(dependencies), partial=result['partial'])), flush=True)


if __name__ == '__main__':
    main()
