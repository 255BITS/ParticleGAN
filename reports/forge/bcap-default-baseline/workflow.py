"""Explicit ordinary preparation/execution and saved-only publication.

One fixed paired configuration, original revision-8 tasks and ordinary tier
vetoes. No diagnostic import, tuning, seed variation or compile callbacks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-default-adoption-20261010')
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash, utc_now
from experiments.forge.planning import plan_summary, resolve_idea
from experiments.forge.studies import validate_study
from experiments.forge.views import load_view
from particlegan import get_recipe

CAMPAIGN = 'bcap-default-baseline-v1'
ROLES = ('control', 'candidate')
IDS = dict(control='bcap-default-baseline-control-v1', candidate='bcap-default-baseline-direction-v1')
MODES = dict(control='none', candidate='direction_blend')
PER_ARM, TOTAL, SOFTWARE = 46620, 96000, 1800


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


shared = module('default_baseline_phase2', ROOT / 'reports/forge/bcap-three-phase/phase2.py')
require, head = shared.require, shared.head


def originals(root):
    view = load_view(root, shared.VIEW)
    require(view['revision'] == 8 and view.get('evidence_scope', 'ordinary') == 'ordinary', 'Retain ordinary revision8.')
    required = [a for a in view['assignments'] if a['importance'] == 'required']
    require(Counter(a['qualification_tier'] for a in required) == {1: 6, 2: 21, 3: 2}, 'Retain all6/21/2 required cells.')
    require(len(view['assignments']) == 30, 'Retain the optional clock diagnostic separately.')
    earlier = read_json(root / shared.ORIGINALS)['tasks']
    tasks = {a['task']: read_json(root / f'configs/forge/tasks/{a["task"]}.json') for a in view['assignments']}
    for name, task in tasks.items():
        if name in earlier:
            require(shared.original_question(task) == shared.original_question(earlier[name]), f'{name}: original question changed.')
        for path, digest in task['evaluation'].get('sources', {}).items():
            require(file_hash(root / path) == digest, f'{name}: evaluator source binding changed: {path}')
    groups = {}
    for name, task in tasks.items():
        key = task['execution'].get('execution_group', name)
        groups[key] = max(groups.get(key, 0), task['resources']['timeout_seconds'])
    require(sum(groups.values()) == PER_ARM, 'Full execution-group reservation must be46620 seconds; ring endurance is one job.')
    return view, tasks


def summary(request):
    plan = plan_summary(request)
    active = [t for t in plan['tasks'] if t['permitted_by_tier_cap']]
    require(request['through_tier'] == 3 and request['view']['revision'] == 8
            and request['view'].get('evidence_scope', 'ordinary') == 'ordinary', 'Use original ordinary throughTier3.')
    require(request['execution_policy']['mode'] == 'complete_current_tier', 'Finish independent current-tier peers before veto.')
    require(request['protocol']['seed'] == 0 and request['study_review']['status'] == 'READY'
            and not request['preflight_blockers'], f'Admission blocked: {request.get("preflight_blockers")}')
    require(len(active) == 30 and not any(t['blockers'] or t['execution_group_blockers'] for t in active), 'All original hosts must preflight.')
    require(plan['worst_case_seconds'] == PER_ARM, 'Reserve the complete46620-second per-arm ladder.')
    return dict(candidate_id=request['candidate']['id'], candidate_revision=request['candidate_revision'],
        study_id=request['study']['id'], study_admission=request['study_admission'], admission='READY',
        source_digest=request['source']['digest'], full_reservation_seconds=PER_ARM, active_tasks=active,
        protocol_sha256=stable_hash(request['protocol']), runtime_sha256=stable_hash(request['runtime']),
        task_keys={m: j['compatibility_key'] for j in request['jobs'] for m in j.get('task_ids', [j['task_id']])})


def prepare(options):
    root = options.repository.resolve()
    view, tasks = originals(root)
    archived = read_json(root / 'configs/forge/ideas/bcap-projection-baseline-review-baseline-v1.json')
    fixed = deepcopy(archived['recipe_overrides'])
    require(len(fixed) == 13 and 'constraint_geometry_mode' not in fixed, 'Freeze the exact13 incumbent overrides.')
    recipes = {}
    for role in ROLES:
        arm = dict(parent='bcap-three-phase-incumbent-v1', candidate_id=IDS[role],
            recipe_overrides={**fixed, 'constraint_geometry_mode': MODES[role]},
            changed_factors=[f'Explicit geometry mode {MODES[role]}; all13 incumbent overrides fixed'],
            mechanism_class='structural', mechanism_rationale='Direction blend protects existing loss gradients; control explicitly restores the incumbent geometry mode on the new common preset source.')
        card = shared.successor(root, arm)
        card.update(trainer_family='bcap-dualnorm', guide='reports/forge/bcap-default-baseline/spec.json')
        atomic_json(root / f'configs/forge/ideas/{IDS[role]}.json', card)
        # Check the complete effective law. Compatibility serialization omits
        # optional defaults (including mode='none') from historical packets.
        recipes[role] = asdict(get_recipe(card['recipe_preset'], **card['recipe_overrides']))
        other = 'candidate' if role == 'control' else 'control'
        study = dict(schema_version=1, id=f'{IDS[role]}-study', status='ready', candidate=IDS[role],
            control=dict(candidate_id=IDS[other], task_map={}), max_rounds=1,
            hypothesis='One common-source ordinary direction-only recipe preserves6/6 Tier1, all control Tier2 passes, and adds a complete Tier2 pass across the full21-task tier.',
            competing_explanation='Earlier diagnostic repairs may not transfer to original full-view consumers, own-prefix continuity or the new common preset source; no historical grade substitutes for these ordinary results.',
            scope=dict(view=shared.VIEW, through_tier=3, execution_backend='cuda', cuda_model='NVIDIA RTX A6000'),
            campaign=dict(id=CAMPAIGN, budget_seconds=TOTAL, candidate_budget_seconds=48000),
            prior_evidence=[dict(path='reports/forge/bcap-projection-baseline-review/phase3-results.json', selector=[],
                identity=dict(source_commit='442dc7ef52658a727cad6014e2e2d5fbcb1787a5', scope='phase3_paired_research_diagnostic'), use='motivation_only')],
            prediction=dict(task_id='trajectory', metric='identity_mse', op='<=' if role == 'candidate' else '>', threshold=.02, phase='final'),
            falsifier=dict(task_id='trajectory', metric='identity_mse', op='>' if role == 'candidate' else '<=', threshold=.02, phase='final'),
            terminal_rules=shared.OUTCOMES)
        validate_study(study)
        atomic_json(root / f'configs/forge/studies/{study["id"]}.json', study)
    delta = {k: dict(control=recipes['control'][k], candidate=recipes['candidate'][k])
             for k in recipes['control'] if recipes['control'][k] != recipes['candidate'][k]}
    require(delta == dict(constraint_geometry_mode=dict(control='none', candidate='direction_blend')), 'No second global trainer delta is permitted.')
    requests = {role: resolve_idea(root, IDS[role], study=f'{IDS[role]}-study', queue_root=options.artifacts / 'queue') for role in ROLES}
    plans = {role: summary(request) for role, request in requests.items()}
    require(len({p['source_digest'] for p in plans.values()}) == 1, 'Both arms need exact scientific source bytes.')
    require(requests['control']['runtime'] == requests['candidate']['runtime']
            and requests['control']['protocol'] == requests['candidate']['protocol'], 'Exact runtime/protocol pairing required.')
    spec = dict(schema_version=1, scope='ordinary_best_observed_bcap_preset_comparison', qualification_claim=False,
        campaign_id=CAMPAIGN, view=shared.VIEW, view_revision=8, through_tier=3, protocol_seed=0,
        full_reservation_seconds=2 * PER_ARM, per_arm_full_reservation_seconds=PER_ARM,
        paid_ceiling_seconds=TOTAL, candidate_budget_seconds=48000, software_allowance_seconds=SOFTWARE,
        recipe_delta=delta, resolved_recipes=recipes, fixed13_overrides=fixed,
        requirements=view['assignments'], historical_evidence_use='Motivation only; no diagnostic reuse or historical7/21 causal baseline.',
        selection_rule=dict(both_roles_tier1_passes=6, complete_tier2_tasks_per_role=21,
            preserve_all_matched_control_tier2_passes=True, minimum_additional_tier2_passes=1,
            exact_source_task_runtime_initialization_rng_proof=True, own_prefixes='Own exact producers; unequal prefixes stay unverified, never silently matched.',
            tier3='Ordinary prerequisite veto; unavailable cells remain BLOCKED and do not claim endurance.'),
        stopping='Complete independent current-tier peers; required non-PASS blocks higher tiers. No tuning, seed change, scientific retry or deeper diagnostic bypass.',
        adoption_scope='Best-observed named bcap preset and benchmark family standard only. View calibration remains provisional; no scientific calibrated-default or robustness claim.')
    atomic_json(OUT / 'spec.json', spec)
    registration = dict(schema_version=1, prepared_commit=head(root), source_digest=next(iter(plans.values()))['source_digest'],
        spec_sha256=file_hash(OUT / 'spec.json'), arms=plans, campaign_id=CAMPAIGN,
        full_reservation_seconds=2 * PER_ARM, required_counts=dict(tier1=6, tier2=21, tier3=2),
        task_contracts={k: stable_hash(t) for k, t in tasks.items()}, original_requirements=view['assignments'])
    atomic_json(options.registration, registration)
    print(json.dumps(dict(event='ordinary_prepared', plans=plans, registration=str(options.registration))), flush=True)


def resolved(options):
    root = options.repository.resolve()
    registration = read_json(options.registration)
    require(file_hash(OUT / 'spec.json') == registration['spec_sha256'], 'Frozen spec differs.')
    _, tasks = originals(root)
    require({k: stable_hash(t) for k, t in tasks.items()} == registration['task_contracts'], 'Frozen task declarations differ.')
    requests = {role: resolve_idea(root, IDS[role], study=f'{IDS[role]}-study', queue_root=options.artifacts / 'queue') for role in ROLES}
    require({role: summary(r) for role, r in requests.items()} == registration['arms'], 'Bindings changed after preparation; regenerate before admission.')
    return registration, requests


def run(options):
    root, artifacts = options.repository.resolve(), options.artifacts.resolve()
    require(head(root) == options.source_commit and not artifacts.is_relative_to(root), 'Use exact committed source and external artifacts.')
    registration, requests = resolved(options)
    declarations = {f'configs/forge/ideas/{IDS[r]}.json' for r in ROLES}
    declarations |= {f'configs/forge/studies/{IDS[r]}-study.json' for r in ROLES}
    declarations |= {f'configs/forge/tasks/{t}.json' for t in registration['task_contracts']}
    declarations.add('configs/forge/views/discriminator_stability.json')
    shared.committed_source(root, requests['control']['source'], declarations)
    from experiments.forge.queue import Queue, drain
    queue = Queue(artifacts / 'queue', report_root=root / 'reports/forge', on_completion=None)
    path = artifacts / 'progress.json'
    if options.submit:
        require(not path.exists(), 'One immutable admission; no duplicate or quality rerun.')
        progress = dict(phase='submitting', requests={}, source_commit=options.source_commit,
            source_digest=registration['source_digest'], registration_sha256=file_hash(options.registration),
            qualification_scope='ordinary', log=str(artifacts / 'logs/driver.log'))
        for role in ROLES:
            request = resolve_idea(root, IDS[role], study=f'{IDS[role]}-study', queue_root=artifacts / 'queue', freeze_source=True)
            require(summary(request) == registration['arms'][role], 'Source freezing changed scientific bindings.')
            receipt = queue.submit(request, request['study']['campaign'])
            progress['requests'][role] = receipt['request']['request_id']
            progress.update(updated_at=utc_now())
            atomic_json(path, progress)
            print(json.dumps(dict(event='ordinary_submitted', role=role, request=progress['requests'][role])), flush=True)
        progress.update(phase='admitted', updated_at=utc_now())
        atomic_json(path, progress)
    if options.drain:
        require(path.is_file(), 'Drain requires explicit prior admission.')
        progress = read_json(path)
        require(progress['phase'] in {'admitted', 'running'} and progress['source_commit'] == options.source_commit
                and progress['registration_sha256'] == file_hash(options.registration), 'Existing admission differs.')
        progress.update(phase='running', updated_at=utc_now())
        atomic_json(path, progress)
        drain(queue, options.gpus.split(','), workers_per_gpu=1, allow_sharing=True, watch=False, campaign=CAMPAIGN)
        progress.update(phase='ordinary_complete', updated_at=utc_now())
        atomic_json(path, progress)
        print(json.dumps(dict(event='ordinary_complete', requests=progress['requests'])), flush=True)


def own_producers(collection, audit_module):
    """Keep uninterrupted group evidence distinct from checkpoint restores."""
    filtered, uninterrupted = [], []
    for entry in collection['final']:
        task = dict(entry['task'])
        dependencies = []
        for dependency in task.get('dependencies', []):
            group = task['execution'].get('execution_group')
            if (not isinstance(dependency, dict) or dependency.get('kind') != 'checkpoint'
                    or not group or not task['execution'].get('uninterrupted')):
                dependencies.append(dependency)
                continue
            parents = [p for p in collection['final'] if p['item']['role'] == entry['item']['role']
                and p['item']['scope'] == entry['item']['scope']
                and p['row']['task_id'] == dependency['task']]
            require(len(parents) == 1, f'{task["id"]}: one own uninterrupted parent required.')
            parent = parents[0]
            a, b = (e['row']['evidence'].get('continuity', {}) for e in (entry, parent))
            require(parent['task']['execution'].get('execution_group') == group
                and parent['item']['attempt_id'] == entry['item']['attempt_id']
                and parent['request']['candidate_revision'] == entry['request']['candidate_revision']
                and a.get('mode') == b.get('mode') == 'uninterrupted'
                and a.get('run_id') and a['run_id'] == b.get('run_id')
                and a.get('resume_count') == b.get('resume_count') == 0
                and a.get('original_schedule_horizon') == b.get('original_schedule_horizon')
                    == task['execution']['original_schedule_horizon']
                and a.get('max_total_steps') == b.get('max_total_steps')
                    == task['execution']['max_total_steps'], f'{task["id"]}: uninterrupted own-group proof differs.')
            require(entry['row']['gate_status'] != 'PASS' or parent['row']['gate_status'] == 'PASS',
                f'{task["id"]}: passing extension requires its own passing hold.')
            uninterrupted.append(dict(scope=entry['item']['scope'], role=entry['item']['role'],
                task_id=task['id'], parent_task_id=dependency['task'],
                parent_attempt_id=parent['item']['attempt_id'], own_attempt_id=entry['item']['attempt_id'],
                continuity_mode='uninterrupted', resume_count=0, run_id=a['run_id'],
                parent_gate_status=parent['row']['gate_status'], gate_status=entry['row']['gate_status']))
        task['dependencies'] = dependencies
        filtered.append(dict(entry, task=task))
    restored = audit_module.own_checkpoints(dict(collection, final=filtered))
    return restored, uninterrupted


def publish(options):
    root, artifacts, output = options.repository.resolve(), options.artifacts.resolve(), options.output.resolve()
    registration, _ = resolved(options)
    progress = read_json(artifacts / 'progress.json')
    require(progress['phase'] == 'ordinary_complete' and head(root) == progress['source_commit'] == options.source_commit
            and progress['registration_sha256'] == file_hash(options.registration), 'Publish only the exact completed ordinary campaign.')
    saved = shared.publisher(root)
    saved.ROLES = ROLES
    saved.required_questions = lambda _root: registration['original_requirements']
    collection = saved.collect(argparse.Namespace(repository=root, queue=artifacts / 'queue', progress=artifacts / 'progress.json',
        diagnostic_queue=None, diagnostic_progress=None, allow_partial=False))
    for context in collection['scopes']:
        for role, submission in context['submissions'].items():
            require(summary(submission['request']) == registration['arms'][role],
                f'{role}: executed declaration differs from frozen registration.')
    audit = shared.audit_saved_comparison(root, collection, ROLES)
    own = module('ordinary_default_own_producers', root / 'reports/forge/bcap-develop-integration/audit.py')
    restored, uninterrupted = own_producers(collection, own)
    audit.update(scope='ordinary_bcap_default_baseline_audit', own_checkpoint_producers=restored,
        own_uninterrupted_groups=uninterrupted,
        source_commit=progress['source_commit'], source_digest=progress['source_digest'])
    renderer = saved._saved_renderer(root)
    media = []
    from PIL import Image
    for entry in collection['final']:
        item = entry['item']
        if item['gate_status'] not in {'PASS', 'FAIL'}:
            continue
        gif = output / 'media' / f'{item["role"]}-{item["task_id"]}.gif'
        display_entry = entry
        if entry['task']['adapter'] == 'ring_endurance':
            # Its certified curve is named dense, while the shared plotting
            # interface accepts observations. Alias in a transient display
            # packet only; never edit the grading row or its retained bytes.
            evidence = entry['row']['evidence']
            require(len(evidence.get('dense', [])) >= 2, 'Ring media requires its actual certified training curve.')
            display_entry = dict(entry, row=dict(entry['row'], evidence=dict(evidence,
                observations=evidence['dense'])))
        with renderer.forbid_live_execution():
            receipt = saved.render_saved(display_entry, gif, renderer)
        if display_entry is not entry:
            receipt.update(display_curve='Exact original certified dense trajectory',
                original_dense_sha256=stable_hash(entry['row']['evidence']['dense']),
                observations_added=0, grading_row_mutated=False)
            atomic_json(gif.with_suffix('.json'), receipt)
        with Image.open(gif) as image:
            frames = image.n_frames
        require(frames >= 2, 'Every completed toy needs multiple saved actual-training states.')
        media.append(dict(receipt, role=item['role'], task_id=item['task_id'], gif=str(gif.relative_to(output)), frames=frames))
        print(json.dumps(dict(event='ordinary_saved_gif', role=item['role'], task=item['task_id'])), flush=True)
    result = dict(schema_version=1, scope='ordinary_bcap_default_baseline_pair',
        source_commit=progress['source_commit'], source_digest=progress['source_digest'], protocol_seed=0,
        registration_sha256=file_hash(options.registration), original_placement_counts=dict(tier1=6, tier2=21, tier3=2),
        task_cells=collection['cells'], task_results=[e['item'] for e in collection['final']], accounting=collection['accounting'],
        paid_attempt_history=[dict(a['compact']) for a in collection['attempts'].values()], actual_training_gifs=len(media),
        optimizer_updates_added=0, sampling_draws_added=0, calibrated_default_adoption=False,
        selection_evidence='Independent full paired audit and select.py decide the named best-observed preset; no automatic mutation.')
    atomic_json(output / 'results.json', result)
    audit.update(task_results_sha256=stable_hash(result['task_results']), results_file_sha256=file_hash(output / 'results.json'),
        actual_training_gifs=len(media), paid_seconds=sum(a['selected_paid_seconds'] for a in collection['accounting']))
    atomic_json(output / 'audit.json', audit)
    atomic_json(output / 'media/index.json', dict(schema_version=1, media=media, optimizer_updates_added=0, sampling_draws_added=0))
    print(json.dumps(dict(event='ordinary_saved_publication_complete', results=str(output / 'results.json'), audit=str(output / 'audit.json'))), flush=True)


def main(action=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--artifacts', type=Path, default=ARCHIVE)
    parser.add_argument('--registration', type=Path, default=OUT / 'registration.json')
    parser.add_argument('--source-commit')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--drain', action='store_true')
    options = parser.parse_args()
    require(action == 'prepare' or options.source_commit, 'Execution/publication require an exact source commit.')
    require(action != 'run' or options.submit or options.drain, 'Choose explicit submit and/or drain.')
    globals()[action](options)

