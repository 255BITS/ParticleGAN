"""One prior-policy contrast on the unchanged energy-U scientific source.

Only the frozen-prior recipe adds paid work. The learned-prior control reuses
its exact source-bound evidence and charge owners through a new cached study.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ORIGINAL = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012')
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-energy-prior-freeze-repair-20261012')
CAMPAIGN = 'r1r2-energy-prior-freeze-repair-v1'
PREDECESSOR = 'reports/forge/r1r2-energy-distance-repair'
SOURCE = '2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12'
RUNTIME = '640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc'
HISTORICAL_ATTEMPTS = 231
HISTORICAL_SECONDS = 6970.914169566939
CARDS = {
    'energy_u': ('r1r2-non-saturating-energy-u-v1', 'r1r2-energy-u-cached-prior-control-v1-study'),
    'frozen_prior': ('r1r2-non-saturating-energy-u-frozen-prior-v1',
                     'r1r2-non-saturating-energy-u-frozen-prior-v1-study'),
}
ROLES = set(CARDS)
PINNED_HELPERS = {
    'reports/forge/r1r2-tier1-search/workflow.py':
        '6357353a6494f1a66409e3f0f8093bfe2e776f57afb9f21416aafa16d56cf066',
    'reports/forge/r1r2-energy-distance-repair/workflow.py':
        '991e573c137327b8dd00c129f8ac916dff1f95f790171ea78bc77ee94b189b96',
}
for path, digest in PINNED_HELPERS.items():
    if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
        raise ValueError(f'Frozen workflow helper changed: {path}')
spec = importlib.util.spec_from_file_location('prior_freeze_energy_helpers', ROOT / PREDECESSOR / 'workflow.py')
energy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(energy)
shared = energy.shared
BUDGETS = energy.BUDGETS
CLOCK = energy.CLOCK
original_requests = energy.original_requests
original_collect = energy.original_collect
original_storage = energy.original_storage
public_recipes = energy.public_recipes
pinned_host_sources = energy.pinned_host_sources
source_revision = energy.source_revision
request_projection = energy.request_projection
PRIOR_ARCHIVES = (*energy.PRIOR_ARCHIVES, energy.ARCHIVE)

# These are isolated imported module instances, never predecessor file edits.
# Frozen protocol/clock equations read their campaign scope from that instance.
energy.OUT, energy.ARCHIVE, energy.CAMPAIGN = OUT, ARCHIVE, CAMPAIGN
energy.CARDS, energy.ROLES = CARDS, ROLES
energy.HISTORICAL_ATTEMPTS, energy.HISTORICAL_SECONDS = HISTORICAL_ATTEMPTS, HISTORICAL_SECONDS
comparison_audit = energy.comparison_audit
clock_comparison = energy.clock_comparison


def validate_manifest(value):
    shared.require(value['schema_version'] == 1 and value['round'] == 1 and value['id'] == CAMPAIGN,
                   'Use the sole separately reviewed prior-freeze comparison')
    shared.require(value['max_candidates'] == len(value['candidates']) == 2 and not value.get('searches')
                   and {r['role'] for r in value['candidates']} == ROLES
                   and all((r['candidate'], r['study']) == CARDS[r['role']] for r in value['candidates']),
                   'One new whole recipe and its exact cached energy parent only')
    shared.require(value['baseline_role'] == 'energy_u' and value['source_digest'] == SOURCE
                   and value['paid_cap_seconds'] == 8000,
                   'Preserve the cached parent, exact scientific source and aggregate paid cap')
    shared.require(value['campaign'] == dict(id=CAMPAIGN, budget_seconds=5040, candidate_budget_seconds=2520),
                   'Retain both unchanged full allowances, including original word900')
    return value


def predecessor(root):
    publication_path, registration_path = root / PREDECESSOR / 'publication/results.json', root / PREDECESSOR / 'registration.json'
    publication, registration = shared.read_json(publication_path), shared.read_json(registration_path)
    audit = shared.read_json(root / PREDECESSOR / 'collection-audit.json')
    expected = {(role, task) for role in ('energy_u', 'non_saturating') for task in shared.TASKS}
    shared.require(publication['complete'] and publication['round_id'] == 'r1r2-energy-distance-repair-v1'
                   and publication['source_digest'] == SOURCE
                   and all(len(publication[k]) == 14 and {(r['role'], r['task_id']) for r in publication[k]} == expected
                           for k in ('certified_results', 'task_cells', 'media'))
                   and all(r['status'] in {'PASS', 'FAIL'} for r in publication['task_cells']),
                   'Finish all fourteen original energy cases and GIFs before this contrast')
    shared.require(audit['status'] == 'PASS' and audit['inputs_sha256']['publication'] == shared.file_hash(publication_path)
                   and audit['inputs_sha256']['registration'] == shared.file_hash(registration_path),
                   'Require the exact closed independent energy collection audit')
    density = shared.read_json(root / PREDECESSOR / 'density-audit.json')
    metrics = {'sample_count', 'modes', 'mass_tv', 'hq', 'component_covariance_error', 'component_min_eigen_ratio'}
    shared.require(density['publication_sha256'] == shared.file_hash(publication_path)
                   and density['registration_sha256'] == shared.file_hash(registration_path)
                   and density['source_digest'] == SOURCE and density['runtime_sha256'] == RUNTIME
                   and density['recomputed_saved_observations'] == 192 and density['recipe_count'] == 2
                   and set(density['recomputed_metrics']) == metrics and density['diagnosis_rng_states_unchanged']
                   and density['new_optimizer_updates'] == density['new_sampling_draws'] == 0
                   and len(density['rings']) == 2 and {r['role'] for r in density['rings']} == {'energy_u', 'non_saturating'}
                   and all(r['recomputed_observations'] == 96 and set(r['max_absolute_gate_metric_difference']) == metrics
                           and all(v == 0 for v in r['max_absolute_gate_metric_difference'].values()) for r in density['rings']),
                   'Require exact saved-only six-metric density evidence for both closed recipes')
    return publication, audit, registration


def prior_delta(resolved):
    parent, parent_tasks = public_recipes(resolved['energy_u']['request'])
    child, child_tasks = public_recipes(resolved['frozen_prior']['request'])
    held = dict(loss='non_saturating', loss_labels=(0., 1., 1.), energy_distance_weight=1.,
                reg_arm='a_r1r2', reg_coeff=.1, reg_coeff_end=None, reg_every=1, critic_r1_real=True,
                optimizer_family='adam', adam_variant='pytorch', amsgrad=True, betas=(.5, .99),
                prior_betas=(0., .99), d_betas=None, beta2_end=None, lr=.0085,
                d_lr_mult=1., prior_lr_mult=1., eps=1e-8, d_eps=None, prior_eps=None,
                lr_schedule='constant', prior_reg=0., prior_l2=0., prior_proximal_strength=0.,
                kinetic_transport_weight=0., kinetic_transport_local_weight=0.,
                input_noise_std=0., output_noise_std=0., ema_decay=0., serve_average=0.,
                critic_step_mode='none', constraint_geometry_mode='none')
    for label, old, new in [('global', parent, child), *(
            (task, parent_tasks[task], child_tasks[task]) for task in sorted(shared.TASKS))]:
        shared.require(all(old.get(k) == v for k, v in held.items()), f'{label}: exact energy NS AMS m110 parent changed')
        shared.require(old['prior_update'] == 'learned' and new['prior_update'] == 'frozen'
                       and {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)} == {'prior_update'},
                       f'{label}: prior learned-to-frozen is the only actual recipe delta')
    required = resolved['frozen_prior']['request']['candidate'].get('requires_capabilities', [])
    shared.require('learned_locations' not in required and 'named_rng' in required,
                   'Frozen-prior declaration cannot claim a learned-location requirement')


def historical_charges(state, publication, audit):
    charges = shared.paid_charge_index(state)
    ids = publication['aggregate_charge_attempt_ids']
    shared.require(len(ids) == len(set(ids)) == publication['aggregate_paid_attempts'] == HISTORICAL_ATTEMPTS
                   and set(ids) <= charges.keys(), 'Retain every historical231 paid attempt')
    historical = {aid: charges[aid] for aid in ids}
    shared.require(shared.stable_hash(historical) == audit['inputs_sha256']['aggregate231_charge_index']
                   and math.isclose(math.fsum(c['seconds'] for c in historical.values()), HISTORICAL_SECONDS, abs_tol=1e-7, rel_tol=0)
                   and publication['aggregate_total_paid_seconds'] == HISTORICAL_SECONDS,
                   'Historical231 charge IDs, owners and exact paid seconds changed')
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


def cached_parent(root, state, publication, registration, resolved):
    expected = {k: v for k, v in registration['candidates']['energy_u'].items() if k != 'card'}
    shared.require(resolved['energy_u']['projection'] == expected,
                   'Cached energy parent must retain its exact source/runtime/protocol/task/job projection')
    card_path = f"configs/forge/ideas/{CARDS['energy_u'][0]}.json"
    shared.require(shared.file_hash(root / card_path) == registration['declarations'][card_path],
                   'The cached parent card bytes cannot change')
    rows = {r['task_id']: r for r in publication['certified_results'] if r['role'] == 'energy_u'}
    shared.require(set(rows) == shared.TASKS, 'Retain all seven exact cached parent cases')
    for task, key in expected['task_keys'].items():
        job, row = state['jobs'].get(key, {}), rows[task]
        shared.require(job.get('result') is not None and len(job['attempts']) == 1
                       and job['result']['attempt_id'] == row['attempt_id']
                       and row['candidate_id'] == expected['candidate'] and row['candidate_revision'] == expected['candidate_revision']
                       and row['source_digest'] == SOURCE, f'{task}: exact cached parent result changed')
        paths = root / 'reports/forge/attempts' / row['attempt_id']
        shared.require(all(shared.file_hash(paths / f'{name}.json') == digest
                           for name, digest in publication['exact_certificate_hashes'][row['attempt_id']].items())
                       and shared.read_json(paths / 'result.json') == job['result'],
                       f'{task}: cached certificate/result identity changed')


def requests(root, queue_root, manifest):
    publication, audit, previous = predecessor(root)
    resolved, searches = original_requests(root, queue_root, manifest)
    shared.require(set(resolved) == ROLES and not searches, 'Only the two fixed prior-policy roles may execute')
    parent = previous['candidates']['energy_u']
    for role, item in resolved.items():
        p, request = item['projection'], item['request']
        shared.require(request['study']['campaign'] == manifest['campaign'], 'Both new studies need the exact5040/2520 campaign')
        shared.require(p['source_digest'] == SOURCE and p['runtime_sha256'] == RUNTIME
                       and p['candidate_revision'] == source_revision(request), 'Bind actual source, runtime and recipe revision')
        for field in ('protocol_sha256', 'view_sha256', 'task_contracts', 'execution_policy', 'full_allowance_seconds'):
            shared.require(p[field] == parent[field], f'{role}: original fixed condition changed: {field}')
        pinned_host_sources(root, request)
    prior_delta(resolved)
    state = shared.read_json(queue_root / 'queue/state.json')
    charges, historical_keys = historical_charges(state, publication, audit)
    cached_parent(root, state, publication, previous, resolved)
    new_keys = set(resolved['frozen_prior']['projection']['task_keys'].values())
    shared.require(len(new_keys) == 7 and not new_keys & historical_keys, 'Exactly seven distinct frozen-prior jobs are permitted')
    progress_path = queue_root.parent / f'{CAMPAIGN}-progress.json'
    progress = shared.read_json(progress_path) if progress_path.exists() else {}
    new_attempts = set()
    for key in new_keys & state['jobs'].keys():
        job, rid = state['jobs'][key], progress.get('requests', {}).get('frozen_prior')
        shared.require(rid and job['subscribers'] == [rid] and len(job['attempts']) <= 1
                       and rid in state['submissions']
                       and shared.request_projection(state['submissions'][rid]['request']) == resolved['frozen_prior']['projection'],
                       'New jobs require the sole frozen admission; no cached substitution or scientific retry')
        if job.get('result'):
            aid = job['result']['attempt_id']
            shared.require(aid not in publication['aggregate_charge_attempt_ids'] and aid in charges
                           and charges[aid]['owner']['request'] == rid
                           and job['cost_owner'] == job['result']['cost_owner'] == charges[aid]['owner'],
                           'Every new frozen-prior result requires its own new paid owner')
            new_attempts.add(aid)
    shared.require(charges.keys() - set(publication['aggregate_charge_attempt_ids']) <= new_attempts,
                   'Only seven new frozen-prior jobs may add paid attempts')
    paid = math.fsum(c['seconds'] for c in charges.values())
    reserved = math.fsum(j.get('reserved_seconds', 0) for j in state['jobs'].values())
    shared.require(paid + reserved <= 8000, 'Preserve all global historical charges and full pending reservations')
    if not progress:
        shared.require(paid + reserved + BUDGETS['five_word_joint_smoke'] <= 8000,
                       'Admission cannot fit the original full900 word reservation')
    return resolved, searches


def storage(archive):
    shared.require(archive == ARCHIVE and (archive / 'queue').is_symlink()
                   and (archive / 'queue').resolve() == ORIGINAL / 'queue', 'Retain the original shared queue and artifacts')
    locations = (archive, *PRIOR_ARCHIVES)
    readings = [original_storage(path) for path in locations]
    shared.require(sum(r['bytes'] for r in readings) < shared.ARCHIVE_CAP,
                   'All ten research archives exceed their combined8GiB ceiling')
    return dict(archives={str(p): r for p, r in zip(locations, readings, strict=True)})


class ScopedQueue(energy.ScopedQueue):
    def _refresh(self, state):
        # Submission lifecycle and collection retain the original eligibility
        # semantics; only the actual claim decision defers the word resource.
        with self.scoped('refresh'):
            return super()._refresh(state)

    def _eligible(self, state, submission):
        eligible, reason, running = super()._eligible(state, submission)
        request = submission['request']
        if (getattr(self._scope, 'mode', None) == 'claim' and request.get('campaign_id') == CAMPAIGN
                and request['candidate']['id'] == CARDS['frozen_prior'][0]):
            definitions = [j for j in request['jobs'] if shared.Queue._authorized(request, j)]
            peers = [j for j in definitions if j['task_id'] != 'five_word_joint_smoke']
            if not all(state['jobs'][j['compatibility_key']]['status'] == 'terminal' for j in peers):
                word_keys = {j['compatibility_key'] for j in definitions if j['task_id'] == 'five_word_joint_smoke'}
                eligible = [key for key in eligible if key not in word_keys]
        return eligible, reason, running

    @contextmanager
    def state(self):
        if getattr(self._scope, 'mode', None) != 'collect':
            with super().state() as raw:
                yield raw
            return
        # Queue._results consumes all29 planned keys per admitted request,
        # including22 unsubscribed higher-tier bookkeeping keys. Claim scope
        # remains unchanged and retains every raw global paid reservation.
        with self.scoped('all_planned_collect'), super().state() as scoped:
            keys = {j['compatibility_key'] for s in scoped['submissions'].values() for j in s['request']['jobs']}
            authorized = {j['compatibility_key'] for s in scoped['submissions'].values()
                          for j in s['request']['jobs'] if shared.Queue._authorized(s['request'], j)}
            shared.require(len(keys) == 58 and len(authorized) == 14,
                           'Exactly58 own planned keys and14 authorized Tier1 keys are required')
            for key in keys - authorized:
                job = scoped['jobs'][key]
                shared.require(not job['attempts'] and job.get('result') is None and not job['subscribers']
                               and job.get('reserved_seconds', 0) == 0,
                               'Higher-tier bookkeeping must remain unclaimed and unpaid')
            yield {**scoped, 'jobs': {key: scoped['jobs'][key] for key in keys}}


def drain(queue, devices, **kwargs):
    shared.require(isinstance(queue, ScopedQueue) and kwargs.get('campaign') in (None, CAMPAIGN),
                   'Only this scoped single comparison may drain')
    kwargs['campaign'] = CAMPAIGN
    return energy.original_drain(queue, devices, **kwargs)


def collect(args):
    shared.require(args.media, 'Every measured task needs its saved actual-training GIF')
    manifest = shared.read_json(args.registration)['manifest']
    resolved, _ = requests(args.repository, args.artifacts / 'queue', manifest)
    previous, _, _ = predecessor(args.repository)
    cached_ids = {r['attempt_id'] for r in previous['certified_results'] if r['role'] == 'energy_u'}
    original_collect(args)
    publication = shared.read_json(args.output / 'results.json')
    rows = publication['certified_results']
    measured = {(r['role'], r['task_id']) for r in rows}
    shared.require(len(measured) == len(rows) and {(r['role'], r['task_id']) for r in publication['media']} == measured
                   and len(publication['media']) == len(measured), 'One exact saved GIF per measured cell is required')
    new_rid = shared.read_json(args.artifacts / f'{CAMPAIGN}-progress.json')['requests']['frozen_prior']
    for row in rows:
        aid, cost = row['attempt_id'], publication['costs'][row['attempt_id']]
        if row['role'] == 'energy_u':
            shared.require(aid in cached_ids and cost['reused'] and cost['seconds'] == 0,
                           'Cached parent keeps its original certificate and charge owner')
        else:
            shared.require(row['role'] == 'frozen_prior' and aid not in previous['aggregate_charge_attempt_ids']
                           and not cost['reused'] and cost['original_charge_owner']['request'] == new_rid,
                           'Only frozen-prior cases add new original paid charges')
    if any(t == CLOCK for _, t in measured):
        clock = clock_comparison(rows, publication['media'])
        publication['comparison_audit'] = [a for a in publication['comparison_audit'] if a['task_id'] != CLOCK] + [clock]
    if publication['complete']:
        shared.require(len(rows) == len(publication['costs']) == 14 and len(publication['comparison_audit']) == 7
                       and all(a['status'] == 'PASS' for a in publication['comparison_audit'])
                       and publication['aggregate_paid_attempts'] == 238,
                       'Complete requires14 cases/GIFs, exact cached7/new7 and all matched protocol proofs')
    shared.require(publication['source_digest'] == SOURCE and publication['aggregate_total_paid_seconds'] <= 8000,
                   'Retain actual fixed scientific source and shared paid ceiling')
    shared.atomic_json(args.output / 'results.json', publication)


shared.validate_manifest, shared.request_projection, shared.requests = validate_manifest, request_projection, requests
shared.storage, shared.BoundedQueue, shared.drain = storage, ScopedQueue, drain
shared.comparison_audit, shared.collect, shared.OUT = comparison_audit, collect, OUT

if __name__ == '__main__':
    shared.main()
