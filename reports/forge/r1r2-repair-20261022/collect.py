"""Collect original Forge certificates and saved-training media; never train.

Different execution sources remain separate cohorts. Existing verdicts and
physical charge ownership are copied, never recomputed or pooled.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.sources import verify_snapshot
from experiments.forge.state import state_digest
from experiments.forge.tier1_media import render
from reports.forge.regenerate_technique_inventory import project_receipt

CAMPAIGN = 'r1r2-repair-20261022'
DEFAULT_QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/queue')
ROLES = {f'r1r2-repair-{role}-20261022': role for role in ('control', 'r1only', 'r1cycle')}


def dictionaries(value, prefix='state'):
    if isinstance(value, dict):
        yield prefix, value
        for key, item in value.items():
            yield from dictionaries(item, f'{prefix}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from dictionaries(item, f'{prefix}.{index}')


def numerical_parts(saved):
    parts = {name: {} for name in ('models', 'role_parameters', 'optimizers', 'stream_states')}
    data_states = {}
    for path, node in dictionaries(saved):
        for name in ('models', 'role_parameters', 'optimizers'):
            if name in node:
                parts[name][f'{path}.{name}'] = state_digest(node[name])
        if isinstance(node.get('states'), dict):
            for name, value in node['states'].items():
                if not isinstance(value, torch.Tensor):
                    continue
                key = f'{path}.states.{name}'
                parts['stream_states'][key] = state_digest(value)
                try:
                    is_data = json.loads(name)[0] == 'data'
                except (ValueError, TypeError, IndexError):
                    is_data = name == 'data' or name.startswith('data:')
                if is_data:
                    data_states[key] = state_digest(value)
    return {name: value for name, value in parts.items() if value}, data_states


def initial_models(row, saved):
    initial = {}
    for path, node in dictionaries(dict(row=row.get('initialization', {}),
        host=row.get('evidence', {}).get('host', {}))):
        if 'initial_state_sha256' in node:
            initial[path] = node['initial_state_sha256']
    initial.update({f'provider.{key}': value for key,value in
                    row.get('evidence', {}).get('initial_models', {}).items()})
    if saved is not None:
        for path, node in dictionaries(saved):
            if isinstance(node.get('initialization'), dict):
                for role, value in node['initialization'].items():
                    if isinstance(value, dict) and 'initial_state_sha256' in value:
                        initial[f'{path}.initialization.{role}'] = value['initial_state_sha256']
    return initial


def checkpoint(row):
    descriptor = row.get('evidence', {}).get('provenance_checkpoint')
    if not descriptor:
        return None, None
    root = Path(descriptor['artifact_root'])
    verify_artifacts(root, descriptor['artifact_manifest'])
    path = root / descriptor['path']
    assert file_hash(path) == descriptor['sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert state_digest(saved) == descriptor['state_sha256']
    parts, data_states = numerical_parts(saved)
    named = descriptor.get('named_stream_state_sha256', {})
    if named:
        assert set(saved['streams']['states']) == set(named)
        assert all(state_digest(saved['streams']['states'][name]) == digest for name,digest in named.items())
    return saved, dict(path=str(path), sha256=descriptor['sha256'],
        state_sha256=descriptor['state_sha256'], artifact_manifest_sha256=descriptor['artifact_manifest']['sha256'],
        completed_steps=descriptor['completed_steps'], purpose=descriptor.get('purpose'),
        numerical_parts=parts, consumed_data_rng_state_sha256=data_states)


def export_media(task, row, local, destination):
    evidence = row.get('evidence', {})
    actual = evidence.get('actual_training_gif')
    if actual:
        source = Path(actual['path'])
        assert file_hash(source) == actual['sha256']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        assert file_hash(destination) == actual['sha256']
        receipt = dict(schema_version=1,kind='retained_original_public_api_actual_training_gif',
            task_id=task['id'],recorded_grade=row['gate_status'],source_path=str(source),
            source_sha256=actual['sha256'],gif_sha256=file_hash(destination),
            optimizer_updates_added=0,sampling_draws_added=0,qualification_input=False)
        atomic_json(destination.with_suffix('.json'), receipt)
        return receipt
    if not evidence.get('observations'):
        return None
    sidecar = destination.with_suffix('.json')
    if destination.exists() and sidecar.exists():
        receipt = read_json(sidecar)
        assert receipt['gif_sha256'] == file_hash(destination)
        assert all(file_hash(Path(path)) == digest for path,digest in receipt.get('source_inputs', {}).items())
        return receipt
    return render(task, row, local, destination)


def collect(options):
    torch.set_num_threads(1)
    state = read_json(options.queue / 'queue/state.json')
    submissions = {rid: value for rid,value in state['submissions'].items()
        if value['request'].get('campaign_id') == CAMPAIGN and value['request']['candidate']['id'] in ROLES}
    if not options.allow_partial:
        active = [rid for rid,sub in submissions.items() if sub['status'] in {'queued','running','paused'}]
        assert not active, f'active or paused requests: {active}'
    source_proofs = {}
    cohorts, attempts, charges = {}, {}, {}
    rows_by_cohort = {}
    for rid, sub in submissions.items():
        request = sub['request']; source = request['source']; digest = source['digest']
        if digest not in source_proofs:
            verify_snapshot(Path(source['snapshot_path']), source)
            source_proofs[digest] = dict(source_digest=digest,origin_commit=source['origin_commit'],
                snapshot_path=source['snapshot_path'],verified_files=len(source['files']),runtime=request['runtime'])
        role = ROLES[request['candidate']['id']]
        cohort = cohorts.setdefault(digest, dict(source=source_proofs[digest],requests=[],roles=[]))
        authorized = [a for a in request['view']['assignments'] if a['qualification_tier'] <= request['through_tier']]
        authorized_names = {a['task'] for a in authorized}
        request_keys = {job['compatibility_key'] for job in request['jobs']
                        if set(job.get('task_ids',[job['task_id']])) <= authorized_names}
        cohort['requests'].append(dict(request_id=rid,role=role,study=request['study']['id'],
            candidate=request['candidate']['id'],candidate_revision=request['candidate_revision'],
            status=sub['status'],reason=sub.get('reason'),view_revision=request['view']['revision'],
            global_recipe=request['candidate']['recipe_overrides']))
        current = {}
        for job in state['jobs'].values():
            result = job.get('result')
            # Cancellation removes live subscriptions, not original evidence.
            if job['definition']['compatibility_key'] not in request_keys or result is None:
                continue
            aid = result['attempt_id']
            summary = project_receipt(ROOT, aid)
            original_dir = ROOT/'reports/forge/attempts'/aid
            envelope = read_json(original_dir/'request.json')
            certificate = read_json(original_dir/'evidence.json')
            local = Path(certificate['local_artifact_root'])
            assert read_json(original_dir/'result.json') == result == read_json(local/'result.json')
            assert read_json(local/'request.json') == envelope
            assert summary['provenance']['source_digest'] == digest
            physical_charge = [c for c in state['charges'] if c['attempt_id'] == aid]
            assert len(physical_charge) == 1
            charges[aid] = dict(**physical_charge[0],full_allowance_seconds=job['definition']['budget_seconds'],source_digest=digest,role=role)
            attempts[aid] = dict(attempt_id=aid,role=role,source_digest=digest,
                provenance=summary['provenance'],original_artifact_root=str(local),
                independent_grading=result['raw'].get('grading',{}).get('grades',{}),
                original_attempt_status=result['raw']['attempt_status'])
            compact = {r['task_id']:r for r in summary['task_results']}
            for row in result['task_results']:
                name = row['task_id'];task = request['tasks'][name]
                saved, cp = checkpoint(row)
                item = dict(**compact[name],attempt_id=aid,role=role,source_digest=digest,
                    task_contract_sha256=stable_hash(task),effective_recipe=row.get('recipe'),
                    initial_prior=row.get('initial_prior'),prior=row.get('prior'),
                    initialization=initial_models(row,saved),checkpoint=cp,
                    batch_sequence_sha256=row.get('evidence',{}).get('batch_sequence_sha256'),
                    observations_sha256=stable_hash(row.get('evidence',{}).get('observations',[])),
                    training_budget=dict(steps=task['execution']['steps'],
                        original_schedule_horizon=task['execution'].get('original_schedule_horizon'),
                        timeout_seconds=task['resources']['timeout_seconds']),
                    qualification_input=False,qualification_reuse=False)
                if options.media:
                    dest=options.output/'media'/f'{digest[:12]}-{role}-{name}-{aid[:8]}.gif'
                    media = export_media(task,row,local,dest)
                    if media:
                        item['actual_training_gif']=dict(path=str(dest.relative_to(ROOT)),
                            sha256=file_hash(dest),sidecar_sha256=file_hash(dest.with_suffix('.json')))
                    else:
                        item['actual_training_gif_gap']='No retained completed training observations; no illustration fabricated.'
                current[name]=item
                rows_by_cohort.setdefault(digest,{})[(role,name)] = item
        cohort['roles'].append(dict(role=role,counts=dict(Counter(current[a['task']]['gate_status'] if a['task'] in current else 'UNKNOWN'
            for a in authorized if a['importance']=='required')),tasks=[dict(**a,result=current.get(a['task'])) for a in authorized]))
    comparisons=[]
    for digest, lookup in rows_by_cohort.items():
        for task in sorted({name for role,name in lookup}):
            available = {role:lookup[(role,task)] for role in ('control','r1only','r1cycle') if (role,task) in lookup}
            initial = {role:row['initialization'] for role,row in available.items() if row['initialization']}
            batch = {role:row['batch_sequence_sha256'] for role,row in available.items() if row['batch_sequence_sha256']}
            data = {role:row['checkpoint']['consumed_data_rng_state_sha256'] for role,row in available.items()
                    if row['checkpoint'] and row['checkpoint']['consumed_data_rng_state_sha256']}
            proof=dict(source_digest=digest,task_id=task,roles_available=sorted(available),
                initial_model_digests=initial,initial_models_match=(len({stable_hash(v) for v in initial.values()})==1 if len(initial)>1 else None),
                actual_batch_digests=batch,actual_batches_match=(len(set(batch.values()))==1 if len(batch)>1 else None),
                consumed_data_rng_digests=data,consumed_data_rng_states_match=(len({stable_hash(v) for v in data.values()})==1 if len(data)>1 else None),
                actual_batch_trace_gap=(None if batch else 'No direct consumed-batch trace is retained by this adapter; named data-stream endpoints are separate evidence.'))
            left,right=available.get('r1only'),available.get('r1cycle')
            if task!='five_word_joint_smoke' and left and right and left['checkpoint'] and right['checkpoint']:
                a,b=left['checkpoint']['numerical_parts'],right['checkpoint']['numerical_parts']
                proof['inactive_joint_inverse_comparison']=dict(
                    numerical_checkpoint_parts_match=a==b,parts_left=a,parts_right=b,
                    observations_match=left['observations_sha256']==right['observations_sha256'],
                    original_grades_match=left['gate_status']==right['gate_status'],
                    excludes='Recipe metadata differs by the declared inapplicable global joint coefficient; original identities and charges remain separate.')
            comparisons.append(proof)
    output=dict(schema_version=1,scope='source_separated_original_campaign_results',campaign=CAMPAIGN,
        qualification_input=False,qualification_reuse=False,training_updates_added=0,sampling_draws_added=0,
        source_cohorts=cohorts,physical_attempts=attempts,unique_paid_costs=charges,
        total_unique_paid_seconds=sum(c['seconds'] for c in charges.values()),
        physical_attempt_count=len(charges),campaign_budget=state['campaigns'][CAMPAIGN]['definition'],
        comparisons=comparisons,generated_leaderboard=False,
        interpretation='Existing certified verdicts only. Sources, recipes, checkpoint parents and physical charges remain distinct. UNKNOWN, cancelled, BLOCKED and INCOMPLETE are not scientific FAILs.')
    options.output.mkdir(parents=True,exist_ok=True)
    atomic_json(options.output/'results.json',output)
    print(json.dumps(dict(source_cohorts=len(cohorts),physical_attempts=len(charges),
        paid_seconds=output['total_unique_paid_seconds'],output=str(options.output/'results.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,default=DEFAULT_QUEUE)
    parser.add_argument('--output',type=Path,default=HERE/'publication')
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--media',action='store_true')
    collect(parser.parse_args())
