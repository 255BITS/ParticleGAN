"""Read-only draft binding review. This program has no enqueue/run command."""
from dataclasses import asdict
import json
import math
from pathlib import Path

from experiments.forge.api import resolve_public_recipe, task_formulation_context
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TASKS = {
    'gaussian1d_smoke', 'two_pole', 'unused_token_hold', 'ae_gan_hold',
    'ring16_acquisition', 'five_word_joint_smoke', 'clockfree_audit_measurement_v1',
}
WIDTHS = (.0625, .125, .25, .5, 1., 2., 4.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def task_hash(task):
    return stable_hash({k: v for k, v in task.items()
                        if k not in {'field_ownership', 'preflight_blockers'}})


def recipes(request):
    return {
        'global': asdict(resolve_public_recipe(request['candidate'])),
        **{task: asdict(task_formulation_context(request['candidate'], request['tasks'][task],
                    request['protocol'], root=ROOT).recipe) for task in sorted(TASKS)},
    }


def main():
    manifest = read_json(OUT / 'manifest.json')
    audit_path = ROOT / 'reports/forge/r1r2-next-repair-research/budget-audit.json'
    audit = read_json(audit_path)
    require(manifest['status'] == 'draft_budget_pending'
            and manifest['budget_change_authorized'] is False
            and manifest['paid_admission_authorized'] is False
            and manifest['current_aggregate_paid_cap_seconds'] == 8000
            and manifest['proposed_aggregate_paid_cap_seconds'] == 12209,
            'This preparation retains pending budget and admission authorization')
    require(manifest['max_candidates'] == 2 and manifest['searches'] == []
            and manifest['campaign'] == dict(id='r1r2-output-mmd-repair-v1',
                budget_seconds=5040, candidate_budget_seconds=2520), 'One finite two-role proposal')
    queue_path = Path(audit['authoritative_queue']['path'])
    original_queue_sha = file_hash(queue_path)
    require(original_queue_sha == audit['authoritative_queue']['sha256'], 'Historical queue changed')
    state = read_json(queue_path)
    charges = {c['attempt_id']: c for c in state['charges']}
    require(len(charges) == len(state['charges']) == 238
            and stable_hash(charges) == audit['authoritative_queue']['aggregate_charge_index_sha256'],
            'Preserve all 238 original charges and owners')
    paid = math.fsum(c['seconds'] for c in charges.values())
    require(paid == audit['authoritative_queue']['paid_seconds']
            and paid == 7168.248061007995
            and not any(j['status'] == 'running' for j in state['jobs'].values()),
            'Preserve paid seconds and inactive queue')
    resolved, projections = {}, {}
    source_files = None
    host_pins = {}
    for card in manifest['candidates']:
        request = resolve_idea(ROOT, card['candidate'], study=card['study'],
                               freeze_source=False)
        role = card['role']
        require(request['study']['status'] == 'draft'
                and request['study_review']['status'] == 'BLOCKED', 'Draft must remain inadmissible')
        expected_blocker = 'study is draft: review actual bindings, prior evidence, finite scope and numerical signatures, then set status=ready'
        require(request['preflight_blockers'] == [expected_blocker],
                f'{role}: unexpected preparation blockers: {request["preflight_blockers"]}')
        jobs = [j for j in request['jobs'] if Queue._authorized(request, j)]
        require(len(jobs) == 7 and {j['task_id'] for j in jobs} == TASKS
                and all(j.get('task_ids', [j['task_id']]) == [j['task_id']]
                        and j['budget_seconds'] == manifest['task_full_allowance_seconds'][j['task_id']]
                        for j in jobs) and sum(j['budget_seconds'] for j in jobs) == 2520,
                'Seven original full allowances, including word900, are required')
        require(request['protocol']['seed'] == 0
                and request['execution_policy']['mode'] == 'complete_current_tier'
                and all(j['compatibility_key'] not in state['jobs'] for j in jobs),
                'Fresh-source seed0 full-tier jobs cannot reuse archived cells')
        for task in TASKS:
            require(not request['tasks'][task]['preflight_blockers'], f'{task}: host blocked')
            for path, digest in request['tasks'][task]['evaluation'].get('sources', {}).items():
                require(file_hash(ROOT / path) == digest
                        and request['source']['files'][path] == digest, f'Original pinned host changed: {path}')
                host_pins[path] = digest
        if source_files is None:
            source_files = request['source']['files']
        require(source_files == request['source']['files'], 'Both roles need the same prospective source')
        projection = dict(candidate=card['candidate'], candidate_revision=request['candidate_revision'],
            source_digest=request['source']['digest'], runtime_sha256=stable_hash(request['runtime']),
            protocol_sha256=stable_hash(request['protocol']), view_sha256=stable_hash(request['view']),
            task_contracts={t: task_hash(request['tasks'][t]) for t in sorted(TASKS)},
            jobs_sha256=stable_hash(jobs), task_keys={j['task_id']: j['compatibility_key'] for j in jobs},
            full_allowance_seconds=2520, draft_blockers=request['preflight_blockers'])
        projections[role] = projection
        resolved[role] = request
        if 'prepared_source_digest' in manifest:
            require(projection['source_digest'] == manifest['prepared_source_digest']
                    and projection['runtime_sha256'] == manifest['prepared_runtime_sha256'],
                    'The reviewed prospective source/runtime must remain frozen')
    require(set(resolved) == {'non_saturating', 'mmd_u'}, 'Two declared roles only')
    parent, child = recipes(resolved['non_saturating']), recipes(resolved['mmd_u'])
    held = dict(loss='non_saturating', energy_distance_weight=0., prior_update='learned',
        prior_reg=0., prior_l2=0., prior_proximal_strength=0., reg_arm='a_r1r2', reg_coeff=.1,
        reg_coeff_end=None, reg_every=1, critic_r1_real=True, amsgrad=True, betas=(.5, .99),
        prior_betas=(0., .99), lr=.0085, lr_schedule='constant', eps=1e-8,
        optimizer_family='adam', adam_variant='pytorch', output_mmd_bandwidths=WIDTHS)
    for scope in parent:
        old, new = parent[scope], child[scope]
        require(all(old[k] == value for k, value in held.items()), f'{scope}: parent changed')
        require(old['output_mmd_weight'] == 0. and new['output_mmd_weight'] == 1.
                and {k for k in old if old[k] != new[k]} == {'output_mmd_weight'},
                f'{scope}: only the MMD weight may change')
    previous = read_json(ROOT / 'reports/forge/r1r2-energy-distance-repair/registration.json')
    control = projections['non_saturating']
    old_control = previous['candidates']['non_saturating']
    require(all(control[key] == old_control[key] for key in
                ['protocol_sha256', 'runtime_sha256', 'view_sha256', 'task_contracts'])
            and control['source_digest'] != old_control['source_digest']
            and all(control['task_keys'][t] != old_control['task_keys'][t] for t in TASKS),
            'Preserve original conditions while requiring fresh source and all fresh control jobs')
    # Declarations/publications under completed registrations stay byte-identical.
    frozen = {}
    for name in ['r1r2-energy-distance-repair', 'r1r2-energy-prior-freeze-repair']:
        for path, digest in read_json(ROOT / f'reports/forge/{name}/registration.json')['declarations'].items():
            require(file_hash(ROOT / path) == digest, f'Closed declaration changed: {path}')
            frozen[path] = digest
    require(file_hash(queue_path) == original_queue_sha, 'Preparation mutated the shared queue')
    receipt = dict(schema_version=1, status='PASS', scope='software_preparation_only',
        paid_admission_authorized=False, budget_change_authorized=False, scientific_success=False,
        qualification_input=False, qualification_reuse=False,
        manifest_sha256=file_hash(OUT / 'manifest.json'), budget_audit_sha256=file_hash(audit_path),
        source_digest=control['source_digest'], source_files_sha256=stable_hash(source_files),
        source_file_count=len(source_files), original_host_pins=host_pins,
        unchanged_closed_declarations=frozen, candidates=projections,
        global_and_host_delta={scope: ['output_mmd_weight'] for scope in parent},
        queue_sha256=original_queue_sha, aggregate238_charge_index_sha256=stable_hash(charges),
        historical_paid_seconds=paid, current_aggregate_cap_seconds=8000,
        proposed_aggregate_cap_seconds=12209, proposed_full_comparison_seconds=5040,
        new_optimizer_updates=0, new_sampling_draws=0, models_constructed=0,
        queue_claims=0, source_snapshots_created=0)
    atomic_json(OUT / 'preparation.json', receipt)
    print(json.dumps({'status': 'PASS', 'scope': receipt['scope'],
        'source_digest': receipt['source_digest'], 'source_file_count': len(source_files),
        'original_host_pins': len(host_pins), 'fresh_cases_required': 14,
        'paid_admission_authorized': False, 'queue_unchanged': True}))


if __name__ == '__main__':
    main()
