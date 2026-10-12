"""One weaker positive paired-penalty contrast, preserving the original Forge execution path."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ORIGINAL = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012')
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-weak-penalty-repair-20261012')
PRIOR_ARCHIVES = (ORIGINAL, *(Path('/mnt/ml7tb/ParticleGAN-forge') / f'{name}-20261012'
    for name in ('r1r2-five-track-repair', 'r1r2-role-repair', 'r1r2-direction-floor-repair',
                 'r1r2-spectral-repair', 'r1r2-objective-memory-repair', 'r1r2-score-law-repair')))
CAMPAIGN = 'r1r2-weak-penalty-repair-v1'
PREDECESSOR = 'reports/forge/r1r2-score-law-repair'
PREVIOUS_CAMPAIGN = 'r1r2-score-law-repair-v1'
HISTORICAL_SECONDS = 6405.0795450999285
SOURCE = '25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507'
RUNTIME = '640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc'
CARDS = {
    'baseline': ('recipe-prior-refactor-r1r2-v1', 'r1r2-five-track-cached-baseline-v1'),
    'non_saturating': ('r1r2-non-saturating-ams-m110-v1', 'r1r2-non-saturating-ams-m110-v1-study'),
    'weak_gamma': ('r1r2-non-saturating-weak-gamma-v1', 'r1r2-non-saturating-weak-gamma-v1-study'),
}
ROLES = set(CARDS)

spec = importlib.util.spec_from_file_location(
    'r1r2_original_weak_penalty_workflow', ROOT / 'reports/forge/r1r2-tier1-search/workflow.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
original_requests = shared.requests
original_storage = shared.storage
original_declarations = shared.declaration_paths


def validate_manifest(value):
    shared.require(value['schema_version'] == 1 and value['round'] == 1
                   and value['id'] == CAMPAIGN, 'Use the separately registered finite weak-penalty campaign')
    shared.require(value['max_candidates'] == 3 and len(value['candidates']) == 3
                   and not value.get('searches'), 'One new whole recipe and two exact cached controls only')
    shared.require({row['role'] for row in value['candidates']} == ROLES
                   and all((row['candidate'], row['study']) == CARDS[row['role']]
                           for row in value['candidates']), 'Preserve the three exact declared cards/studies')
    shared.require(value['baseline_role'] == 'baseline', 'Preserve the scheduled reference')
    shared.require(value['paid_cap_seconds'] == 8000, 'Preserve the shared aggregate paid ceiling')
    shared.require(value['campaign']['id'] == CAMPAIGN
                   and value['campaign']['candidate_budget_seconds'] == shared.ALLOWANCE
                   and value['campaign']['budget_seconds'] == 3 * shared.ALLOWANCE,
                   'Reserve each unchanged full task allowance')
    return value


def predecessor(root):
    publication_path = root / PREDECESSOR / 'publication/results.json'
    publication = shared.read_json(publication_path)
    audit = shared.read_json(root / PREDECESSOR / 'collection-audit.json')
    registration_path = root / PREDECESSOR / 'registration.json'
    registration = shared.read_json(registration_path)
    previous_roles = {'baseline', 'ams_m110', 'non_saturating'}
    expected = {(role, task) for role in previous_roles for task in shared.TASKS}
    shared.require(publication['complete'] and publication['round_id'] == PREVIOUS_CAMPAIGN
                   and all(len(publication[key]) == 21
                           and {(r['role'], r['task_id']) for r in publication[key]} == expected
                           for key in ('certified_results', 'task_cells', 'media'))
                   and all(c['status'] in {'PASS', 'FAIL'} for c in publication['task_cells']),
                   'Complete all 21 predecessor cases/GIFs before weak-penalty preparation')
    shared.require(audit['status'] == 'PASS'
                   and audit['inputs_sha256']['publication'] == shared.file_hash(publication_path)
                   and audit['inputs_sha256']['registration'] == shared.file_hash(registration_path),
                   'Predecessor requires its exact completed independent collection audit')
    density = shared.read_json(root / PREDECESSOR / 'density-audit.json')
    metrics = {'sample_count', 'modes', 'mass_tv', 'hq',
               'component_covariance_error', 'component_min_eigen_ratio'}
    ring = next(r for r in publication['certified_results']
                if r['role'] == 'non_saturating' and r['task_id'] == 'ring16_acquisition')
    shared.require(density['publication_sha256'] == shared.file_hash(publication_path)
                   and density['registration_sha256'] == shared.file_hash(registration_path)
                   and density['source_digest'] == SOURCE and density['runtime_sha256'] == RUNTIME
                   and density['recomputed_saved_observations'] == 96
                   and density['recipe_count'] == 1 and set(density['recomputed_metrics']) == metrics
                   and density['diagnosis_rng_states_unchanged']
                   and density['new_optimizer_updates'] == density['new_sampling_draws'] == 0
                   and len(density['rings']) == 1
                   and all(r['role'] == 'non_saturating' and r['recomputed_observations'] == 96
                           and all(r[field] == ring[field] for field in
                                   ('attempt_id', 'candidate_id', 'candidate_revision'))
                           and set(r['max_absolute_gate_metric_difference']) == metrics
                           and all(v == 0 for v in r['max_absolute_gate_metric_difference'].values())
                           for r in density['rings']),
                   'Predecessor density audit must reproduce the exact saved metrics without training')
    return publication, audit, registration


def public_recipes(request):
    """Resolve actual global and host recipes without models, forwards or draws."""
    from experiments.forge.api import resolve_public_recipe, task_formulation_context
    candidate = request['candidate']
    return asdict(resolve_public_recipe(candidate)), {
        task: asdict(task_formulation_context(candidate, request['tasks'][task], request['protocol'],
                                             root=ROOT).recipe) for task in sorted(shared.TASKS)}


def penalty_delta(resolved):
    parent, parent_tasks = public_recipes(resolved['non_saturating']['request'])
    candidate, candidate_tasks = public_recipes(resolved['weak_gamma']['request'])
    held = dict(loss='non_saturating', reg_arm='a_r1r2', reg_coeff=.1, reg_every=1,
                critic_r1_real=True, optimizer_family='adam', adam_variant='pytorch', amsgrad=True,
                betas=(.5, .99), prior_betas=(0., .99), d_betas=None, lr=.0085,
                d_lr_mult=1., prior_lr_mult=1., eps=1e-8, d_eps=None, prior_eps=None,
                lr_schedule='constant', prior_update='learned', prior_reg=0., prior_l2=0.,
                kinetic_transport_weight=0., kinetic_transport_local_weight=0.)
    shared.require(all(parent.get(k) == v for k, v in held.items()),
                   'Retain exact native AMS m110 optimizer/prior and every-update paired R1/R2')
    for label, old, new in [('global', parent, candidate), *(
            (task, parent_tasks[task], candidate_tasks[task]) for task in sorted(shared.TASKS))]:
        shared.require(old['reg_coeff'] == .1 and new['reg_coeff'] == .01
                       and {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)} == {'reg_coeff'},
                       f'{label}: exactly one actual public positive penalty-field delta is permitted')


def historical_charges(state, publication, audit):
    charges = shared.paid_charge_index(state)
    ids = publication['aggregate_charge_attempt_ids']
    shared.require(len(ids) == len(set(ids)) == publication['aggregate_paid_attempts'] == 210
                   and set(ids) <= charges.keys(), 'Retain every historical210 paid attempt')
    historical = {aid: charges[aid] for aid in ids}
    shared.require(shared.stable_hash(historical) == audit['inputs_sha256']['aggregate210_charge_index']
                   and math.isclose(math.fsum(c['seconds'] for c in historical.values()), HISTORICAL_SECONDS,
                                    abs_tol=1e-7, rel_tol=0)
                   and publication['aggregate_total_paid_seconds'] == HISTORICAL_SECONDS,
                   'Historical charge IDs, owners and exact paid seconds changed')
    jobs = {}
    for job in state['jobs'].values():
        if job.get('result') and job['result']['attempt_id'] in historical:
            aid = job['result']['attempt_id']
            shared.require(aid not in jobs, 'Historical paid attempt appears in multiple jobs')
            jobs[aid] = job
    shared.require(set(jobs) == set(ids), 'Historical paid result disappeared')
    for aid, job in jobs.items():
        charge, result = historical[aid], job['result']
        shared.require(len(job['attempts']) == 1 and job['attempts'][0]['attempt_id'] == aid
                       and job['cost_owner'] == result['cost_owner'] == charge['owner']
                       and math.isclose(math.fsum(r.get('cost', {}).get('wall_seconds', 0)
                                                for r in result['task_results']), charge['seconds'],
                                        abs_tol=1e-7, rel_tol=0),
                       'Historical one-attempt result, original owner or wall time changed')
    return charges


def requests(root, queue_root, manifest):
    publication, audit, previous = predecessor(root)
    resolved, searches = original_requests(root, queue_root, manifest)
    shared.require(set(resolved) == ROLES and not searches, 'Use only the three declared whole recipes')
    shared.require(all(item['projection']['source_digest'] == SOURCE
                       and item['projection']['runtime_sha256'] == RUNTIME for item in resolved.values()),
                   'Preserve the original scientific source and runtime')
    for field in ('protocol_sha256', 'view_sha256', 'task_contracts', 'execution_policy', 'full_allowance_seconds'):
        shared.require(all(item['projection'][field] == resolved['baseline']['projection'][field]
                           for item in resolved.values()), f'Comparison changes a fixed condition: {field}')
    state = shared.read_json(queue_root / 'queue/state.json')
    charges = historical_charges(state, publication, audit)
    for role in ('baseline', 'non_saturating'):
        expected = {k: v for k, v in previous['candidates'][role].items() if k != 'card'}
        shared.require(resolved[role]['projection'] == expected,
                       f'{role}: retain exact measured source/runtime/protocol/task/job identity')
        rows = {r['task_id']: r for r in publication['certified_results'] if r['role'] == role}
        shared.require(set(rows) == set(expected['task_keys']) == shared.TASKS,
                       f'{role}: preserve all six ordinary tasks and own clock proof')
        for task, key in expected['task_keys'].items():
            job = state['jobs'].get(key, {})
            row = rows[task]
            shared.require(job.get('result') is not None and len(job['attempts']) == 1
                           and job['result']['attempt_id'] == row['attempt_id']
                           and row['candidate_id'] == expected['candidate']
                           and row['candidate_revision'] == expected['candidate_revision']
                           and row['source_digest'] == expected['source_digest'],
                           f'{role}: retain the seven exact original one-attempt cached cases')
            paths = root / 'reports/forge/attempts' / row['attempt_id']
            shared.require(all(shared.file_hash(paths / f'{name}.json') == digest
                               for name, digest in publication['exact_certificate_hashes'][row['attempt_id']].items())
                           and shared.read_json(paths / 'result.json') == job['result'],
                           f'{role}: cached certificate/result identity changed')
    penalty_delta(resolved)
    new_keys = set(resolved['weak_gamma']['projection']['task_keys'].values())
    old_keys = {key for p in previous['candidates'].values() for key in p['task_keys'].values()}
    shared.require(len(new_keys) == 7 and not new_keys & old_keys, 'Seven new weak-penalty jobs must be distinct')
    progress_path = queue_root.parent / f'{CAMPAIGN}-progress.json'
    progress = shared.read_json(progress_path) if progress_path.exists() else {}
    for key in new_keys & state['jobs'].keys():
        job = state['jobs'][key]
        rid = progress.get('requests', {}).get('weak_gamma')
        shared.require(rid and rid in job['subscribers'] and len(job['attempts']) <= 1
                       and rid in state['submissions']
                       and shared.request_projection(state['submissions'][rid]['request'])
                       == resolved['weak_gamma']['projection'],
                       'New jobs require the sole frozen admission; no prior substitution or scientific retry')
    new_attempts = {state['jobs'][key]['result']['attempt_id'] for key in new_keys & state['jobs'].keys()
                    if state['jobs'][key].get('result')}
    shared.require(charges.keys() - set(publication['aggregate_charge_attempt_ids']) <= new_attempts,
                   'Only this one registered seven-job contrast may add paid attempts')
    return resolved, searches


def storage(archive):
    shared.require(archive == ARCHIVE and (archive / 'queue').is_symlink()
                   and (archive / 'queue').resolve() == ORIGINAL / 'queue',
                   'Retain the exact original queue, artifacts and every historical charge')
    locations = (archive, *PRIOR_ARCHIVES)
    readings = [original_storage(location) for location in locations]
    shared.require(sum(r['bytes'] for r in readings) < shared.ARCHIVE_CAP,
                   'All eight research archives exceed the combined storage cap')
    return dict(archives={str(p): r for p, r in zip(locations, readings, strict=True)})


shared.validate_manifest = validate_manifest
shared.requests = requests
shared.storage = storage
shared.OUT = OUT

if __name__ == '__main__':
    shared.main()
