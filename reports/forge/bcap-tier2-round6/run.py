"""Five fixed global recipes, ordinary seed-0 Tier 1/2 comparison.

Preparation is read-only with respect to execution. Run admits one immutable
campaign; publish reads certified saved states without training or sampling.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.forge.api import resolve_public_recipe
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash, utc_now
from experiments.forge.planning import plan_summary, resolve_idea
from experiments.forge.studies import validate_study
from experiments.forge.views import load_view

CAMPAIGN = 'bcap-tier2-round6-v1'
PARENT = 'recipe-prior-refactor-bcap-pure-v1'
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-tier2-round6-20261011')
PER_ARM, CEILING = 43020, 220000
DELTAS = {
    'control': {},
    'late_anneal': dict(lr_floor=.1, network_lr_floor=.2),
    'half_network': dict(lr=.006, d_lr_mult=3., prior_lr_mult=5.),
    'prior_smoothing': dict(optimizer_prior_smoothing=.01),
    'zero_damping': dict(reg_zero_coeff=.05),
}
ROLES = tuple(DELTAS)
IDS = {role: f'bcap-tier2-round6-{role.replace("_", "-")}-v1' for role in ROLES}
RATIONALES = {
    'control': 'Fresh matched direction-blend reference under current recipe-owned priors; historical results are motivation only.',
    'late_anneal': 'Hold rates through 60% of the original horizon, then decay networks to .2 and learned-prior motion to .1 to test acquired-fit retention with nonzero adaptation.',
    'half_network': 'Halve G/E and direct generator motion while preserving absolute D .018 and prior .030 rates, testing saved overshoot evidence without weakening the critic.',
    'prior_smoothing': 'Keep network smoothing .001; prior row denominator .01 makes weak prior forces yield smaller actual travel while retaining learned locations and strong-force motion.',
    'zero_damping': 'Retain BCAP and add .05/2 times real plus fake squared input slopes; tests damping in the unpenalized below-cap regime.',
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


shared = module('round6_shared', ROOT / 'reports/forge/bcap-three-phase/phase2.py')
require, head = shared.require, shared.head


def task_contract(root):
    view = load_view(root, 'discriminator_stability')
    require(view['revision'] == 9 and view.get('evidence_scope', 'ordinary') == 'ordinary', 'Use current ordinary revision9.')
    active = [a for a in view['assignments'] if a['qualification_tier'] <= 2]
    require(Counter(a['qualification_tier'] for a in active if a['importance'] == 'required') == {1: 6, 2: 21}, 'Retain six/21 required gates.')
    tasks = {a['task']: read_json(root / f'configs/forge/tasks/{a["task"]}.json') for a in active}
    require(sum(t['resources']['timeout_seconds'] for t in tasks.values()) == PER_ARM, 'Retain full43020-second reservation.')
    for name, task in tasks.items():
        require(task['execution']['prior_contract'] == 'recipe_owned_v1', f'{name}: recipe-owned prior required.')
        for path, digest in task['evaluation'].get('sources', {}).items():
            require(file_hash(root / path) == digest, f'{name}: evaluator source differs: {path}')
    return view, active, tasks


def summary(request):
    plan = plan_summary(request)
    active = [t for t in plan['tasks'] if t['permitted_by_tier_cap']]
    require(request['through_tier'] == 2 and request['view']['revision'] == 9, 'Freeze ordinary Tier2 scope.')
    require(request['protocol']['seed'] == 0 and request['execution_policy']['mode'] == 'complete_current_tier', 'Keep seed0 and ordinary tier veto.')
    require(request['study_review']['status'] == 'READY' and not request['preflight_blockers'], f'Admission blocked: {request["preflight_blockers"]}')
    require(len(active) == 28 and not any(t['blockers'] or t['execution_group_blockers'] for t in active), 'All ordinary hosts must preflight.')
    require(plan['worst_case_seconds'] == PER_ARM, 'Full per-arm allowance changed.')
    return dict(candidate_id=request['candidate']['id'], candidate_revision=request['candidate_revision'],
        study_id=request['study']['id'], study_admission=request['study_admission'], source_digest=request['source']['digest'],
        full_reservation_seconds=PER_ARM, protocol_sha256=stable_hash(request['protocol']),
        runtime_sha256=stable_hash(request['runtime']), tasks=active,
        task_keys={n: j['compatibility_key'] for j in request['jobs'] for n in j.get('task_ids', [j['task_id']])
                   if n in {t['task'] for t in active}})


def requests(root, artifacts):
    return {role: resolve_idea(root, IDS[role], study=f'{IDS[role]}-study', queue_root=artifacts / 'queue') for role in ROLES}


def prepare(options):
    root = options.repository.resolve()
    view, active, tasks = task_contract(root)
    parent = read_json(root / f'configs/forge/ideas/{PARENT}.json')
    recipes = {}
    for role in ROLES:
        card = shared.successor(root, dict(parent=PARENT, candidate_id=IDS[role], recipe_overrides=DELTAS[role],
            changed_factors=[RATIONALES[role]], mechanism_class='structural' if role in {'control', 'zero_damping'} else 'floor_constant',
            mechanism_rationale=RATIONALES[role]))
        # Explicit .001 on every arm is numerically the parent's inherited
        # prior smoothing and enables identical force/motion counters.
        card['recipe_overrides'].setdefault('optimizer_prior_smoothing', .001)
        if role == 'zero_damping':
            card['trainer_family'] = 'bcap-zero-damping'
        card['guide'] = 'reports/forge/bcap-tier2-round6/STUDY.md'
        atomic_json(root / f'configs/forge/ideas/{IDS[role]}.json', card)
        recipes[role] = asdict(resolve_public_recipe(card))
        target, metric, threshold = ('grid100', 'precision', .97) if role == 'half_network' else ('mode_hold', 'hq', .9)
        study = dict(schema_version=1, id=f'{IDS[role]}-study', status='ready', candidate=IDS[role],
            control=dict(candidate_id=IDS['late_anneal'] if role == 'control' else IDS['control'], task_map={}), max_rounds=1,
            hypothesis=RATIONALES[role] + ' Global success requires all six Tier1 gates, all21 eligible Tier2 measurements, preservation of every matched control pass and at least one new sustained Tier2 pass.',
            competing_explanation='Smaller motion or critic damping may improve a final endpoint while impairing acquisition, rare mass, local density, conditional identity or shifted reacquisition; all original sustained gates remain binding.',
            scope=dict(view='discriminator_stability', through_tier=2, execution_backend='cuda', cuda_model='NVIDIA RTX A6000'),
            campaign=dict(id=CAMPAIGN, budget_seconds=CEILING, candidate_budget_seconds=44000),
            prior_evidence=[dict(path='reports/forge/recipe-prior-refactor/publication/results.json', selector=[],
                identity=dict(scope='recipe_prior_refactor_selected_leader_measurement'), use='motivation_only'),
                dict(path='reports/forge/bcap-default-baseline/results.json', selector=[],
                identity=dict(source_commit='d378734f40b09ce223a389e8f54a9783ec6a0c75'), use='motivation_only')],
            prediction=dict(task_id=target, metric=metric, op='>=', threshold=threshold, phase='final'),
            falsifier=dict(task_id=target, metric=metric, op='<', threshold=threshold, phase='final'), terminal_rules=shared.OUTCOMES)
        validate_study(study)
        atomic_json(root / f'configs/forge/studies/{study["id"]}.json', study)
    actual_delta = {r: {k: dict(control=recipes['control'][k], candidate=v) for k, v in recipes[r].items()
                       if v != recipes['control'][k]} for r in ROLES}
    require({r: {k: v['candidate'] for k, v in d.items()} for r, d in actual_delta.items()} == DELTAS, 'Undeclared recipe delta.')
    plans = {role: summary(r) for role, r in requests(root, options.artifacts).items()}
    require(len({p['source_digest'] for p in plans.values()}) == len({p['protocol_sha256'] for p in plans.values()}) == len({p['runtime_sha256'] for p in plans.values()}) == 1, 'Shared source/runtime/protocol required.')
    spec = dict(schema_version=1, campaign_id=CAMPAIGN, scope='ordinary_bcap_tier2_fixed_five_arm_comparison',
        view='discriminator_stability', view_revision=9, through_tier=2, protocol_seed=0,
        per_arm_reservation_seconds=PER_ARM, full_reservation_seconds=5 * PER_ARM, paid_ceiling_seconds=CEILING,
        recipe_deltas=actual_delta, resolved_recipes=recipes, requirements=active,
        task_contracts={k: stable_hash(v) for k, v in tasks.items()},
        selection_rule=dict(all_tier1_pass=True, complete_eligible_tier2=21, preserve_every_matched_control_tier2_pass=True,
            minimum_additional_tier2_passes=1, historical_control_passes=9, historical_control_is_motivation_only=True),
        prior_policy='Learned locations, VICReg selected with zero weight, prior_l2 zero; initial distributions/capacities and sampling remain task-owned.',
        common_monitoring='Explicit optimizer_prior_smoothing .001 equals the inherited baseline law and records the same cumulative gradient and motion counters on every arm; only the prior-smoothing candidate changes it to .01.',
        stopping='One configuration per idea; no retries, tuning, new seed, Tier3 spend or diagnostic bypass. Finish independent current-tier peers, then apply ordinary prerequisite veto.',
        scientific_default_adoption=False, calibration='provisional')
    atomic_json(OUT / 'spec.json', spec)
    atomic_json(options.registration, dict(schema_version=1, prepared_commit=head(root), source_digest=plans['control']['source_digest'],
        spec_sha256=file_hash(OUT / 'spec.json'), arms=plans, original_requirements=active,
        task_contracts=spec['task_contracts'], campaign_id=CAMPAIGN, full_reservation_seconds=5 * PER_ARM))
    print(json.dumps(dict(event='prepared', arms={r: dict(candidate=p['candidate_id'], seconds=p['full_reservation_seconds']) for r, p in plans.items()})), flush=True)


def resolved(options):
    registration = read_json(options.registration)
    require(file_hash(OUT / 'spec.json') == registration['spec_sha256'], 'Frozen spec changed.')
    _, _, tasks = task_contract(options.repository)
    require({k: stable_hash(v) for k, v in tasks.items()} == registration['task_contracts'], 'Frozen task changed.')
    selected = requests(options.repository, options.artifacts)
    require({role: summary(r) for role, r in selected.items()} == registration['arms'], 'Frozen execution binding changed.')
    return registration, selected


def run(options):
    root, artifacts = options.repository.resolve(), options.artifacts.resolve()
    require(head(root) == options.source_commit and not artifacts.is_relative_to(root), 'Use exact committed source and external artifacts.')
    registration, selected = resolved(options)
    shared.committed_source(root, selected['control']['source'],
        [f'configs/forge/{kind}/{IDS[r]}{suffix}.json' for r in ROLES for kind, suffix in [('ideas', ''), ('studies', '-study')]])
    from experiments.forge.queue import Queue, drain
    queue = Queue(artifacts / 'queue', report_root=root / 'reports/forge', on_completion=None)
    progress_path = artifacts / 'progress.json'
    if options.submit:
        require(not progress_path.exists(), 'Already admitted; resume with --drain only.')
        require(shutil.disk_usage(artifacts.parent).free >= 18 * 1024**3, 'Require18GiB free for bounded artifacts.')
        progress = dict(phase='submitting', requests={}, source_commit=options.source_commit,
            source_digest=registration['source_digest'], registration_sha256=file_hash(options.registration))
        for role in ROLES:
            request = resolve_idea(root, IDS[role], study=f'{IDS[role]}-study', queue_root=artifacts / 'queue', freeze_source=True)
            require(summary(request) == registration['arms'][role], 'Freezing changed binding.')
            receipt = queue.submit(request, request['study']['campaign'])
            progress['requests'][role] = receipt['request']['request_id']
            atomic_json(progress_path, progress)
            print(json.dumps(dict(event='submitted', role=role, request=progress['requests'][role])), flush=True)
        progress.update(phase='admitted', updated_at=utc_now())
        atomic_json(progress_path, progress)
    if options.drain:
        progress = read_json(progress_path)
        require(progress['phase'] in {'admitted', 'running'} and progress['source_commit'] == options.source_commit
                and progress['registration_sha256'] == file_hash(options.registration), 'Admission changed.')
        progress.update(phase='running', updated_at=utc_now())
        atomic_json(progress_path, progress)
        drain(queue, options.gpus.split(','), workers_per_gpu=1, allow_sharing=False, watch=False, campaign=CAMPAIGN)
        progress.update(phase='complete', updated_at=utc_now())
        atomic_json(progress_path, progress)
        print(json.dumps(dict(event='complete', requests=progress['requests'])), flush=True)


def publish(options):
    root, artifacts, output = options.repository.resolve(), options.artifacts.resolve(), options.output.resolve()
    registration, _ = resolved(options)
    progress = read_json(artifacts / 'progress.json')
    require(progress['phase'] == 'complete' and progress['source_commit'] == options.source_commit == head(root)
            and progress['registration_sha256'] == file_hash(options.registration), 'Publish only complete frozen work.')
    saved = shared.publisher(root)
    saved.ROLES = ROLES
    saved.required_questions = lambda _: registration['original_requirements']
    collection = saved.collect(argparse.Namespace(repository=root, queue=artifacts / 'queue', progress=artifacts / 'progress.json',
        diagnostic_queue=None, diagnostic_progress=None, allow_partial=False))
    for context in collection['scopes']:
        for role, submission in context['submissions'].items():
            require(summary(submission['request']) == registration['arms'][role], 'Executed submission differs from frozen registration.')
    audit = shared.audit_saved_comparison(root, collection, ROLES)
    ordinary = module('round6_own', root / 'reports/forge/bcap-default-baseline/workflow.py')
    own = module('round6_own_audit', root / 'reports/forge/bcap-develop-integration/audit.py')
    restored, uninterrupted = ordinary.own_producers(collection, own)
    audit.update(own_checkpoint_producers=restored, own_uninterrupted_groups=uninterrupted)
    renderer, media = saved._saved_renderer(root), []
    for entry in collection['final']:
        item = entry['item']
        if item['gate_status'] not in {'PASS', 'FAIL'}:
            continue
        gif = output / 'media' / f'{item["role"]}-{item["task_id"]}.gif'
        with renderer.forbid_live_execution():
            receipt = saved.render_saved(entry, gif, renderer)
        media.append(dict(role=item['role'], task_id=item['task_id'], gif=str(gif.relative_to(output)), receipt=receipt))
        print(json.dumps(dict(event='saved_training_gif', role=item['role'], task=item['task_id'])), flush=True)
    cells = collection['cells']
    counts = {r: {str(t): dict(Counter(c['gate_status'] for c in cells if c['role'] == r and c['tier'] == t
        and next(a['importance'] for a in registration['original_requirements'] if a['task'] == c['task_id']) == 'required')) for t in (1, 2)} for r in ROLES}
    baseline_pass = {c['task_id'] for c in cells if c['role'] == 'control' and c['tier'] == 2 and c['gate_status'] == 'PASS'}
    decisions = {}
    control_complete = counts['control']['1'].get('PASS', 0) == 6 and sum(
        counts['control']['2'].get(s, 0) for s in ('PASS', 'FAIL')) == 21
    for role in ROLES[1:]:
        passed = {c['task_id'] for c in cells if c['role'] == role and c['tier'] == 2 and c['gate_status'] == 'PASS'}
        eligible = counts[role]['1'].get('PASS', 0) == 6
        complete = sum(counts[role]['2'].get(s, 0) for s in ('PASS', 'FAIL')) == 21
        decisions[role] = dict(eligible=eligible, tier2_complete=complete, lost_passes=sorted(baseline_pass - passed),
            added_passes=sorted(passed - baseline_pass), matched_control_complete=control_complete,
            winner=control_complete and eligible and complete and baseline_pass < passed)
    result = dict(schema_version=1, scope='ordinary_bcap_tier2_round6', source_commit=options.source_commit,
        source_digest=progress['source_digest'], required_counts=dict(tier1=6, tier2=21), arm_counts=counts,
        decisions=decisions, task_cells=cells, task_results=[e['item'] for e in collection['final']],
        accounting=collection['accounting'], paid_attempt_history=[dict(a['compact']) for a in collection['attempts'].values()],
        actual_training_gifs=len(media), optimizer_updates_added=0, sampling_draws_added=0, scientific_default_adoption=False)
    atomic_json(output / 'results.json', result)
    atomic_json(output / 'audit.json', dict(audit, results_sha256=file_hash(output / 'results.json')))
    atomic_json(output / 'media/index.json', dict(schema_version=1, media=media, optimizer_updates_added=0, sampling_draws_added=0))
    print(json.dumps(dict(event='published', counts=counts, decisions=decisions)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'publish'))
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--artifacts', type=Path, default=ARCHIVE)
    parser.add_argument('--registration', type=Path, default=OUT / 'registration.json')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--source-commit')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--drain', action='store_true')
    options = parser.parse_args()
    require(options.action == 'prepare' or options.source_commit, 'Exact source commit required.')
    require(options.action != 'run' or options.submit or options.drain, 'Choose explicit admission/drain.')
    globals()[options.action](options)


if __name__ == '__main__':
    main()
