"""Two fresh-source energy-U recipes through the original bounded Forge APIs.

This wrapper scopes the shared queue to the two frozen admissions. Historical
jobs, costs and source-bound qualifications remain in their original campaigns.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ORIGINAL = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012')
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-energy-distance-repair-20261012')
PRIOR_ARCHIVES = (ORIGINAL, *(Path('/mnt/ml7tb/ParticleGAN-forge') / f'{name}-20261012'
    for name in ('r1r2-five-track-repair', 'r1r2-role-repair', 'r1r2-direction-floor-repair',
                 'r1r2-spectral-repair', 'r1r2-objective-memory-repair', 'r1r2-score-law-repair',
                 'r1r2-weak-penalty-repair')))
CAMPAIGN = 'r1r2-energy-distance-repair-v1'
PREDECESSOR = 'reports/forge/r1r2-weak-penalty-repair'
PREVIOUS_CAMPAIGN = 'r1r2-weak-penalty-repair-v1'
HISTORICAL_ATTEMPTS = 217
HISTORICAL_SECONDS = 6586.162437112915
OLD_SOURCE = '25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507'
RUNTIME = '640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc'
CARDS = {
    'non_saturating': ('r1r2-non-saturating-ams-m110-v1', 'r1r2-energy-u-fresh-control-v1-study'),
    'energy_u': ('r1r2-non-saturating-energy-u-v1', 'r1r2-non-saturating-energy-u-v1-study'),
}
ROLES = set(CARDS)
CLOCK = 'clockfree_audit_measurement_v1'
BUDGETS = dict(gaussian1d_smoke=120, two_pole=300, unused_token_hold=300,
               ae_gan_hold=300, ring16_acquisition=300, five_word_joint_smoke=900,
               clockfree_audit_measurement_v1=300)

spec = importlib.util.spec_from_file_location(
    'r1r2_original_energy_workflow', ROOT / 'reports/forge/r1r2-tier1-search/workflow.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
original_requests = shared.requests
original_projection = shared.request_projection
original_storage = shared.storage
original_declarations = shared.declaration_paths
original_collect = shared.collect
original_comparison = shared.comparison_audit
original_drain = shared.drain
OriginalBoundedQueue = shared.BoundedQueue


def validate_manifest(value):
    shared.require(value['schema_version'] == 1 and value['round'] == 1 and value['id'] == CAMPAIGN,
                   'Use the separately registered finite energy-distance campaign')
    shared.require(value['max_candidates'] == len(value['candidates']) == 2 and not value.get('searches'),
                   'Exactly two fresh-source whole recipes; no search or cached control role')
    shared.require({r['role'] for r in value['candidates']} == ROLES
                   and all((r['candidate'], r['study']) == CARDS[r['role']] for r in value['candidates']),
                   'Preserve the two exact declared cards and new studies')
    shared.require(value['baseline_role'] == 'non_saturating' and value['paid_cap_seconds'] == 8000,
                   'Retain the fresh NS control and shared aggregate paid ceiling')
    shared.require(value['campaign'] == dict(id=CAMPAIGN, candidate_budget_seconds=2520, budget_seconds=5040),
                   'Reserve both unchanged full task allowances')
    digest = value.get('source_digest', '')
    shared.require(isinstance(digest, str) and len(digest) == 64
                   and all(c in '0123456789abcdef' for c in digest) and digest != OLD_SOURCE,
                   'Explicitly freeze the new scientific source; old-source credit is forbidden')
    return value


def request_projection(request):
    projection = original_projection(request)
    jobs = [j for j in request['jobs'] if shared.Queue._authorized(request, j)]
    shared.require(len(jobs) == 7 and all(j.get('task_ids', [j['task_id']]) == [j['task_id']]
                   and j['task_id'] in BUDGETS and j['budget_seconds'] == BUDGETS[j['task_id']]
                   for j in jobs) and {j['task_id'] for j in jobs} == shared.TASKS,
                   'Seven original full reservations, including word900 and own clock, are mandatory')
    return projection


def predecessor(root):
    path = root / PREDECESSOR / 'publication/results.json'
    registration_path = root / PREDECESSOR / 'registration.json'
    publication = shared.read_json(path)
    audit = shared.read_json(root / PREDECESSOR / 'collection-audit.json')
    registration = shared.read_json(registration_path)
    expected = {(r, t) for r in ('baseline', 'non_saturating', 'weak_gamma') for t in shared.TASKS}
    shared.require(publication['complete'] and publication['round_id'] == PREVIOUS_CAMPAIGN
                   and all(len(publication[k]) == 21 and {(r['role'], r['task_id']) for r in publication[k]} == expected
                           for k in ('certified_results', 'task_cells', 'media'))
                   and all(r['status'] in {'PASS', 'FAIL'} for r in publication['task_cells']),
                   'Complete all 21 predecessor cases and GIFs before the new comparison')
    shared.require(audit['status'] == 'PASS' and audit['inputs_sha256']['publication'] == shared.file_hash(path)
                   and audit['inputs_sha256']['registration'] == shared.file_hash(registration_path),
                   'Predecessor requires its exact completed independent collection audit')
    density = shared.read_json(root / PREDECESSOR / 'density-audit.json')
    metrics = {'sample_count', 'modes', 'mass_tv', 'hq', 'component_covariance_error', 'component_min_eigen_ratio'}
    ring = next(r for r in publication['certified_results'] if r['role'] == 'weak_gamma' and r['task_id'] == 'ring16_acquisition')
    shared.require(density['publication_sha256'] == shared.file_hash(path)
                   and density['registration_sha256'] == shared.file_hash(registration_path)
                   and density['source_digest'] == OLD_SOURCE and density['runtime_sha256'] == RUNTIME
                   and density['recipe_count'] == 1 and density['recomputed_saved_observations'] == 96
                   and set(density['recomputed_metrics']) == metrics and density['diagnosis_rng_states_unchanged']
                   and density['new_optimizer_updates'] == density['new_sampling_draws'] == 0
                   and len(density['rings']) == 1
                   and all(r['role'] == 'weak_gamma' and r['recomputed_observations'] == 96
                           and all(r[k] == ring[k] for k in ('attempt_id', 'candidate_id', 'candidate_revision'))
                           and set(r['max_absolute_gate_metric_difference']) == metrics
                           and all(v == 0 for v in r['max_absolute_gate_metric_difference'].values()) for r in density['rings']),
                   'Predecessor density audit must reproduce the original six metrics without training')
    return publication, audit, registration


def public_recipes(request):
    from experiments.forge.api import resolve_public_recipe, task_formulation_context
    candidate = request['candidate']
    return asdict(resolve_public_recipe(candidate)), {
        t: asdict(task_formulation_context(candidate, request['tasks'][t], request['protocol'], root=ROOT).recipe)
        for t in sorted(shared.TASKS)}


def pinned_host_sources(root, request):
    for task in sorted(shared.TASKS):
        for path, digest in request['tasks'][task]['evaluation'].get('sources', {}).items():
            shared.require(shared.file_hash(root / path) == digest
                           and request['source']['files'].get(path) == digest,
                           f'{task}: preserve the exact pinned original host/evaluator source: {path}')


def source_revision(request):
    from experiments.forge.planning import candidate_revision_for
    return candidate_revision_for(request['source']['digest'], request['candidate'])


def energy_delta(resolved):
    parent, parent_tasks = public_recipes(resolved['non_saturating']['request'])
    candidate, candidate_tasks = public_recipes(resolved['energy_u']['request'])
    held = dict(loss='non_saturating', loss_labels=(0., 1., 1.), reg_arm='a_r1r2', reg_coeff=.1,
                reg_coeff_end=None, reg_every=1, critic_r1_real=True, optimizer_family='adam',
                adam_variant='pytorch', amsgrad=True, betas=(.5, .99), prior_betas=(0., .99),
                d_betas=None, beta2_end=None, lr=.0085, d_lr_mult=1., prior_lr_mult=1., eps=1e-8,
                d_eps=None, prior_eps=None, lr_schedule='constant', prior_update='learned',
                prior_reg=0., prior_l2=0., prior_proximal_strength=0.,
                kinetic_transport_weight=0., kinetic_transport_local_weight=0.,
                input_noise_std=0., output_noise_std=0., ema_decay=0., serve_average=0.,
                critic_step_mode='none', constraint_geometry_mode='none')
    for label, old, new in [('global', parent, candidate), *(
            (t, parent_tasks[t], candidate_tasks[t]) for t in sorted(shared.TASKS))]:
        shared.require(all(old.get(k) == v for k, v in held.items()),
                       f'{label}: retain exact native NS AMS m110 and positive every-update R1/R2')
        shared.require(old.get('energy_distance_weight') == 0. and new.get('energy_distance_weight') == 1.
                       and {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)} == {'energy_distance_weight'},
                       f'{label}: exactly one raw energy-distance weight delta is permitted')


def historical_charges(state, publication, audit):
    charges = shared.paid_charge_index(state)
    ids = publication['aggregate_charge_attempt_ids']
    shared.require(len(ids) == len(set(ids)) == publication['aggregate_paid_attempts'] == HISTORICAL_ATTEMPTS
                   and set(ids) <= charges.keys(), 'Retain every historical217 paid attempt')
    historical = {aid: charges[aid] for aid in ids}
    shared.require(shared.stable_hash(historical) == audit['inputs_sha256']['aggregate217_charge_index']
                   and math.isclose(math.fsum(c['seconds'] for c in historical.values()), HISTORICAL_SECONDS, abs_tol=1e-7, rel_tol=0)
                   and publication['aggregate_total_paid_seconds'] == HISTORICAL_SECONDS,
                   'Historical charge IDs, owners and exact paid seconds changed')
    jobs = {}
    for key, job in state['jobs'].items():
        if job.get('result') and job['result']['attempt_id'] in historical:
            aid = job['result']['attempt_id']
            shared.require(aid not in jobs, 'Historical paid attempt appears in multiple jobs')
            jobs[aid] = (key, job)
    shared.require(set(jobs) == set(ids), 'Historical paid result disappeared')
    for aid, (_, job) in jobs.items():
        charge, result = historical[aid], job['result']
        shared.require(len(job['attempts']) == 1 and job['attempts'][0]['attempt_id'] == aid
                       and job['cost_owner'] == result['cost_owner'] == charge['owner']
                       and math.isclose(math.fsum(r.get('cost', {}).get('wall_seconds', 0) for r in result['task_results']),
                                        charge['seconds'], abs_tol=1e-7, rel_tol=0),
                       'Historical one-attempt result, original owner or wall time changed')
    return charges, {key for key, _ in jobs.values()}


def study_campaigns(resolved, manifest):
    shared.require(all(item['request']['study']['campaign'] == manifest['campaign']
                       for item in resolved.values()),
                   'Both actual studies must share the exact declared 5040/2520 campaign')


def requests(root, queue_root, manifest):
    publication, audit, previous = predecessor(root)
    resolved, searches = original_requests(root, queue_root, manifest)
    shared.require(set(resolved) == ROLES and not searches, 'Use only the two declared fresh recipes')
    study_campaigns(resolved, manifest)
    parent = previous['candidates']['non_saturating']
    for role, item in resolved.items():
        p = item['projection']
        shared.require(p['source_digest'] == manifest['source_digest'] != OLD_SOURCE
                       and p['runtime_sha256'] == RUNTIME, 'Bind both measurements to the new source and original runtime')
        for field in ('protocol_sha256', 'view_sha256', 'task_contracts', 'execution_policy', 'full_allowance_seconds'):
            shared.require(p[field] == parent[field], f'{role}: comparison changes an original fixed condition: {field}')
        pinned_host_sources(root, item['request'])
        shared.require(p['candidate_revision'] == source_revision(item['request'])
                       and p['candidate_revision'] != parent['candidate_revision'],
                       'Every new measurement requires its actual fresh-source candidate revision')
    shared.require(resolved['non_saturating']['projection']['candidate'] == parent['candidate']
                   and shared.file_hash(root / f"configs/forge/ideas/{parent['candidate']}.json") ==
                       previous['declarations'][f"configs/forge/ideas/{parent['candidate']}.json"],
                   'Fresh-source NS control must retain the exact measured parent card bytes and law')
    energy_delta(resolved)
    state = shared.read_json(queue_root / 'queue/state.json')
    charges, historical_keys = historical_charges(state, publication, audit)
    keys = {key for item in resolved.values() for key in item['projection']['task_keys'].values()}
    shared.require(len(keys) == 14 and not keys & historical_keys,
                   'Fourteen fresh-source jobs must be distinct; historical cached grades cannot fill them')
    progress_path = queue_root.parent / f'{CAMPAIGN}-progress.json'
    progress = shared.read_json(progress_path) if progress_path.exists() else {}
    new_attempts = set()
    for role, item in resolved.items():
        rid = progress.get('requests', {}).get(role)
        for key in set(item['projection']['task_keys'].values()) & state['jobs'].keys():
            job = state['jobs'][key]
            shared.require(rid and job['subscribers'] == [rid] and len(job['attempts']) <= 1
                           and rid in state['submissions'] and shared.request_projection(state['submissions'][rid]['request']) == item['projection'],
                           'Fresh jobs require the sole frozen admission; no cache substitution, retry or foreign subscriber')
            if job.get('result'):
                aid = job['result']['attempt_id']
                shared.require(aid not in publication['aggregate_charge_attempt_ids'] and aid in charges
                               and charges[aid]['owner']['request'] == rid
                               and job['cost_owner'] == job['result']['cost_owner'] == charges[aid]['owner'],
                               'Every fresh result requires its own original new paid owner')
                new_attempts.add(aid)
    shared.require(charges.keys() - set(publication['aggregate_charge_attempt_ids']) <= new_attempts,
                   'Only this registered fourteen-job contrast may add paid attempts')
    return resolved, searches


def storage(archive):
    shared.require(archive == ARCHIVE and (archive / 'queue').is_symlink()
                   and (archive / 'queue').resolve() == ORIGINAL / 'queue',
                   'Retain the exact original queue and every historical artifact and charge')
    locations = (archive, *PRIOR_ARCHIVES)
    readings = [original_storage(p) for p in locations]
    shared.require(sum(r['bytes'] for r in readings) < shared.ARCHIVE_CAP,
                   'All nine research archives exceed the combined storage cap')
    return dict(archives={str(p): r for p, r in zip(locations, readings, strict=True)})


class ScopedQueue(OriginalBoundedQueue):
    """Restrict claims/collection to new request IDs while charging the full queue."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scope = threading.local()

    def admissions(self):
        path = ARCHIVE / f'{CAMPAIGN}-progress.json'
        progress = shared.read_json(path)
        shared.require(progress['phase'] == 'admitted' and set(progress['requests']) == ROLES
                       and len(set(progress['requests'].values())) == 2,
                       'Drain requires both immutable new request admissions')
        return progress['requests']

    @contextmanager
    def scoped(self, mode):
        previous = getattr(self._scope, 'mode', None)
        self._scope.mode = mode
        try:
            yield
        finally:
            self._scope.mode = previous

    @contextmanager
    def state(self):
        with super().state() as raw:
            mode = getattr(self._scope, 'mode', None)
            if mode is None:
                yield raw
                return
            admitted = self.admissions()
            selected = {rid: raw['submissions'][rid] for rid in admitted.values()}
            registration_path = OUT / 'registration.json'
            registration = shared.read_json(registration_path)
            progress = shared.read_json(ARCHIVE / f'{CAMPAIGN}-progress.json')
            shared.require(progress['registration_sha256'] == shared.file_hash(registration_path),
                           'Immutable admitted registration changed during execution')
            for role, rid in admitted.items():
                request = selected[rid]['request']
                shared.require(request['request_id'] == rid and request['campaign_id'] == CAMPAIGN
                               and request['candidate']['id'] == CARDS[role][0]
                               and request['study']['id'] == CARDS[role][1], 'Foreign request cannot enter the frozen campaign')
                shared.require(shared.request_projection(request) == {
                    k: v for k, v in registration['candidates'][role].items() if k != 'card'},
                    'Frozen request cannot change its source, recipe, tasks or protocol during drain')
            scoped = {**raw, 'submissions': selected}
            if mode == 'collect':
                keys = {j['compatibility_key'] for s in selected.values() for j in s['request']['jobs']
                        if shared.Queue._authorized(s['request'], j)}
                scoped['jobs'] = {k: raw['jobs'][k] for k in keys}
            yield scoped

    def claim(self, slots, *, campaign_filter=None, goal_filter=None):
        shared.require(campaign_filter in (None, CAMPAIGN), 'Drain cannot select a historical campaign')
        with self.scoped('claim'):
            return super().claim(slots, campaign_filter=CAMPAIGN, goal_filter=goal_filter)

    def collect(self, *, only_orphans=False):
        with self.scoped('collect'):
            return super().collect(only_orphans=only_orphans)

    def pause(self, campaign_id, paused):
        # The shared storage watcher visits campaigns; preserve every old one.
        if campaign_id == CAMPAIGN:
            return super().pause(campaign_id, paused)


def drain(queue, devices, **kwargs):
    shared.require(isinstance(queue, ScopedQueue) and kwargs.get('campaign') in (None, CAMPAIGN),
                   'Only the scoped new campaign queue can drain')
    kwargs['campaign'] = CAMPAIGN
    return original_drain(queue, devices, **kwargs)


def comparison_audit(rows):
    for row in rows:
        if row['task_id'] != CLOCK:
            shared.require(row.get('comparison_protocol') and row['comparison_protocol']['rng_checkpoints']['all_named_streams_checkpointed'],
                           'Every measured native host requires its actual initialization, ordered batches and consumed streams')
    audits = original_comparison(rows)
    for audit in audits:
        if audit['task_id'] != CLOCK and set(audit.get('roles', [])) != ROLES:
            audit.update(status='INCOMPLETE_MATCHED_COMPARISON', reason='One fresh-source peer remains unmeasured')
    return audits


def clock_comparison(rows, exports):
    """Inspect saved own-clock tensors/streams; construct no model or RNG draw."""
    import torch
    from experiments.forge.state import state_digest
    measured = {r['role'] for r in rows if r['task_id'] == CLOCK}
    snapshots, inputs = {}, {}
    branches = {'reference', 'step_label', 'horizon', 'evaluation_cadence', 'restart'}
    expected_streams = {('data', 'clock_probe', 'training'), ('data', 'target', 'training'),
        ('eval', 'clock_probe', 'samples'), ('eval', 'sampler', 'samples'), ('eval', 'target', 'reference'),
        ('init', 'discriminator', 'construction'), ('init', 'generator', 'construction'),
        ('init', 'prior', 'locations'), ('noise', 'critic', 'input'), ('noise', 'generator', 'output'),
        ('noise', 'models', 'stochastic_layers'), ('noise', 'penalty', 'training'),
        ('noise', 'prior', 'gaussian'), ('prior', 'latent', 'indices')}
    expected_trainer_streams = {'latent_generator', 'penalty_generator', 'eval_generator', 'noise_generator',
                               'input_noise_generator', 'prior_noise_generator', 'model_generator'}
    for role in sorted(measured):
        media = next(r for r in exports if r['role'] == role and r['task_id'] == CLOCK)
        sources = media['receipt']['source_inputs']
        shared.require(len(sources) == 1, 'Own-clock GIF requires its exact saved comparisons proof')
        path, digest = next(iter(sources.items()))
        shared.require(Path(path).name == 'comparisons.pt' and shared.file_hash(Path(path)) == digest,
                       'Own-clock saved proof identity changed')
        proof = torch.load(path, map_location='cpu', weights_only=True)
        initial = proof['initial']
        shared.require(set(proof['trajectories']) == set(proof['branch_initial']) == branches
                       and all(len(v) == 4 for v in proof['trajectories'].values()),
                       'Retain all five original own-clock branches and four probe updates')
        shared.require(initial['initializer'] == 'deterministic_orthogonal'
                       and set(initial['initialization']) == {'generator', 'discriminator', 'prior'}
                       and all(isinstance(v.get('initial_state_sha256'), str) and len(v['initial_state_sha256']) == 64
                               for v in initial['initialization'].values()),
                       'Own-clock constructor requires its original captured initialization receipts')
        for branch in proof['branch_initial'].values():
            shared.require(state_digest(branch['trainer']['models']) == state_digest(initial['trainer']['models']),
                           'Each own-clock branch must start from its own actual warmed tensors')
        stream_keys = set(initial['streams']['states'])
        trainer_keys = set(initial['trainer']['streams'])
        shared.require({tuple(json.loads(k)[:3]) for k in stream_keys} == expected_streams
                       and trainer_keys == expected_trainer_streams
                       and set(initial['streams']['manifest']['bindings']) == stream_keys
                       and initial['streams']['manifest']['seed'] == 0,
                       'Own-clock initial proof must retain the original fourteen seed0 streams')
        all_states = [initial, *proof['branch_initial'].values(),
                      *(s for trajectory in proof['trajectories'].values() for s in trajectory)]
        for saved in all_states:
            shared.require(set(saved['streams']['states']) == stream_keys
                           and set(saved['trainer']['streams']) == trainer_keys
                           and all(isinstance(v, torch.Tensor) and v.dtype == torch.uint8
                                   for v in saved['streams']['states'].values())
                           and all(v is None or isinstance(v, torch.Tensor) and v.dtype == torch.uint8
                                   for v in saved['trainer']['streams'].values()),
                           'Own-clock proof must checkpoint every consumed stream at every saved state')
        snapshots[role] = dict(constructor_initialization_sha256=state_digest(initial['initialization']),
            own_warmed_tensors_sha256=state_digest(initial['trainer']['models']),
            pre_warmup_tensor_copy_available=False,
            # Device labels may differ across workers; compare the actual state
            # bytes under the same family/component/purpose and declared seed.
            initial_streams_sha256=state_digest({json.dumps(json.loads(k)[:3]):
                dict(state=v, seed=initial['streams']['manifest']['bindings'][k]['seed'])
                for k, v in initial['streams']['states'].items()}),
            initial_trainer_streams_sha256=state_digest(initial['trainer']['streams']),
            saved_states_checked=len(all_states), all_named_streams_checkpointed=True)
        inputs[path] = digest
    if measured == ROLES:
        for field in ('constructor_initialization_sha256', 'initial_streams_sha256', 'initial_trainer_streams_sha256'):
            shared.require(len({s[field] for s in snapshots.values()}) == 1,
                           f'Fresh own-clock comparison changed constructor receipts or matched stream states: {field}')
    return dict(task_id=CLOCK, status='PASS' if measured == ROLES else 'INCOMPLETE_MATCHED_COMPARISON',
                roles=sorted(measured), saved_inputs=inputs, own_clock_saved_proofs=snapshots,
                optimizer_updates_added=0, sampling_draws_added=0)


def collect(args):
    shared.require(args.media, 'Every measured fresh-source task requires its actual-training GIF')
    resolved, _ = requests(args.repository, args.artifacts / 'queue', shared.read_json(args.registration)['manifest'])
    original_collect(args)
    publication = shared.read_json(args.output / 'results.json')
    measured = {(r['role'], r['task_id']) for r in publication['certified_results']}
    shared.require(len(measured) == len(publication['certified_results'])
                   and {(r['role'], r['task_id']) for r in publication['media']} == measured
                   and len(publication['media']) == len(measured)
                   and all(not c['reused'] and c['original_charge_owner']['request'] in
                           shared.read_json(args.artifacts / f'{CAMPAIGN}-progress.json')['requests'].values()
                           for c in publication['costs'].values()),
                   'Each measured fresh-source case requires its own new charge and exact saved GIF')
    if any(task == CLOCK for _, task in measured):
        clock = clock_comparison(publication['certified_results'], publication['media'])
        publication['comparison_audit'] = [a for a in publication['comparison_audit'] if a['task_id'] != CLOCK] + [clock]
    if publication['complete']:
        shared.require(len(measured) == 14 and len(publication['costs']) == 14
                       and len(publication['comparison_audit']) == 7
                       and all(r['status'] == 'PASS' for r in publication['comparison_audit']),
                       'A complete comparison requires all fourteen new cases and actual matched conditions')
    shared.require(publication['aggregate_total_paid_seconds'] <= 8000
                   and publication['source_digest'] == next(iter(resolved.values()))['projection']['source_digest'],
                   'Collection cannot exceed the original paid ceiling or change the fresh source')
    shared.atomic_json(args.output / 'results.json', publication)


shared.validate_manifest = validate_manifest
shared.request_projection = request_projection
shared.requests = requests
shared.storage = storage
shared.BoundedQueue = ScopedQueue
shared.drain = drain
shared.comparison_audit = comparison_audit
shared.collect = collect
shared.OUT = OUT

if __name__ == '__main__':
    shared.main()
