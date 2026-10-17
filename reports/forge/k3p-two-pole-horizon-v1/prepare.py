"""Freeze four diagnostic measurements of two unchanged global K3P recipes.

Preparation is read-only with respect to the queue and every existing solution
card. The study's decision contracts change task conditions, never the recipes.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.decision_contracts import scaffold
from experiments.forge.planning import load_idea, resolve_idea
from experiments.forge.queue import Queue

REPORT = ROOT / 'reports/forge/k3p-two-pole-horizon-v1'
STUDY = 'k3p-two-pole-horizon-v1'
VIEW = 'k3p_two_pole_horizon'
TASK_CAP = 300
CANDIDATE_CAP = 600
CAMPAIGN_CAP = 1200
CONTROL = 'k3p--01eca360219ea5225a6e30a31800c7bbe70ca08c3800e259c17a67f0ea528ab5'
CANDIDATES = [('word_positive', 'k3p-global-repair-v1'), ('movement_control', CONTROL)]
TASKS = [('schedule80', 80), ('schedule800', 800)]


def evidence(relative):
    value = read_json(ROOT / relative)
    identity = {key: value[key] for key in ('id', 'study', 'candidate_revision', 'source_digest') if key in value}
    return {'path': relative, 'sha256': file_hash(ROOT / relative), 'selector': [],
            'identity': identity, 'use': 'motivation_only'}


def main():
    queue_root = ROOT / 'runs/forge' / STUDY / 'queue'
    state = Queue(queue_root, on_completion=None).inspect()
    if state.get('campaigns', {}).get(STUDY) or (REPORT / 'summary.json').exists():
        raise ValueError('Admitted/concluded study is immutable; do not prepare or rerun it')
    REPORT.mkdir(parents=True, exist_ok=True)
    base = read_json(ROOT / 'configs/forge/tasks/two_pole.json')
    original_task_hash = file_hash(ROOT / 'configs/forge/tasks/two_pole.json')
    task_ids = []
    for suffix, horizon in TASKS:
        task = deepcopy(base)
        name = 'two_pole_800_' + suffix + '_diagnostic_v1'
        task_ids.append(name)
        task.update(id=name, description=f'Does an unchanged global recipe move from zero by 800 updates with schedule horizon {horizon}? Movement and bounded slope are the declared question; two-mode fidelity is diagnostic only.',
                    retained_question_ids=['two_pole'])
        task['execution'].update(steps=800, original_schedule_horizon=horizon,
            schedule_policy='public_recipe_with_explicit_diagnostic_horizon',
            horizon_diagnostic={'schema_version': 1, 'kind': 'two_pole_force_v1',
                'checkpoints': [80, 200, 400, 800], 'prefix_horizon': 80})
        task['execution']['host_definition'].update(steps=800, schedule_horizon=horizon)
        task['research_artifacts'] = {'readout': 'reports/forge/k3p-two-pole-horizon-v1/README.md'}
        # Preserve all original scorer bindings. Reproduction sources travel
        # with the executed source snapshot as explicit support inputs.
        task['evaluation']['sources'].update({str(path.relative_to(ROOT)): file_hash(path)
            for path in (REPORT / 'prepare.py', REPORT / 'run.py')})
        atomic_json(ROOT / 'configs/forge/tasks' / (name + '.json'), task)
    view = {'schema_version': 1, 'id': VIEW, 'goal': 'discriminator_stability', 'revision': 1,
        'evidence_scope': 'research_diagnostic',
        'assignments': [{'task': name, 'qualification_tier': 1, 'importance': 'diagnostic', 'order': i}
                        for i, name in enumerate(task_ids)],
        'eligibility': {}, 'ranking': {'policy': 'raw_metrics_only', 'compare_compatible_cohorts': True, 'cost_separate': True},
        'calibration': {'status': 'provisional', 'adoption_blocker': 'Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.'},
        'policy_change_reason': 'Explicitly scoped two-pole budget and schedule diagnostic; continue both registered tasks after numerical failure. Original80-step main-view gate and recipe selections remain unchanged.'}
    atomic_json(ROOT / 'configs/forge/views' / (VIEW + '.json'), view)
    campaign = {'schema_version': 1, 'id': STUDY, 'budget_seconds': CAMPAIGN_CAP,
        'candidate_budget_seconds': CANDIDATE_CAP, 'accept_shared_cost_transfer': False,
        'purpose': 'Four fixed-seed substantive budget/schedule diagnostics; two unchanged global recipes, two800-step tasks, sequentialCPU execution, no ordinary qualification.'}
    atomic_json(ROOT / 'configs/forge/campaigns' / (STUDY + '.json'), campaign)
    motivations = [evidence('reports/forge/word-root-cause/receipts/k3p-coeff170-cap1.json'),
                   evidence('reports/forge/k3p-global-tier1-v3/summary.json'),
                   evidence('reports/forge/k3p-global-tier1-v3/direct-rate-analysis.json')]
    plans, arms, requests = [], [], []
    for label, candidate_id in CANDIDATES:
        original = load_idea(ROOT, candidate_id)
        original_path = ROOT / ('configs/forge/configurations' if 'configuration_id' in original else 'configs/forge/ideas') / (candidate_id + '.json')
        original_hash = file_hash(original_path)
        candidate = deepcopy(original)
        contract = scaffold(candidate_id, VIEW)
        contract['control']['task_map'] = {name: 'two_pole' for name in task_ids}
        candidate['decision_contract'] = contract
        request = resolve_idea(ROOT, candidate_id, declaration=candidate, view_id=VIEW, through_tier=1, execution_backend='cpu')
        expected = request['decision_review']['expected']
        contract.update(status='ready', prior_evidence=motivations,
            candidate_binding_sha256=expected['candidate_binding_sha256'],
            substantive_delta=expected['substantive_delta'],
            prediction={'task_id': task_ids[1], 'metric': 'mean_abs', 'op': '>=', 'threshold': .3, 'phase': 'final'},
            falsifier={'task_id': task_ids[1], 'metric': 'mean_abs', 'op': '<', 'threshold': .3, 'phase': 'final'},
            competing_explanation='Additional execution may only extend a low-force plateau. Stretching the whole declared schedule changes LR/noise windows and LR-dependent critic behavior; longer execution can activate the existing guard after200. No single-factor or full-family qualification claim follows.')
        contract['control'].update(binding_sha256=expected['control_binding_sha256'], task_map=expected['task_map'])
        contract['scope'].update(view=VIEW, through_tier=1, task_ids=expected['task_ids'], max_rounds=1,
            candidate_budget_seconds=CANDIDATE_CAP, campaign_budget_seconds=CAMPAIGN_CAP,
            **{key: expected[key] for key in ('protocol_sha256', 'source_digest', 'execution_backend', 'runtime_cohort_sha256', 'jobs_sha256')})
        declaration = REPORT / 'declarations' / (label + '.json')
        atomic_json(declaration, candidate)
        request = resolve_idea(ROOT, candidate_id, declaration=candidate, view_id=VIEW, through_tier=1, execution_backend='cpu')
        assert request['decision_review']['status'] == 'READY', request['decision_review']
        assert not request['preflight_blockers'], request['preflight_blockers']
        assert all(not task['preflight_blockers'] for task in request['tasks'].values()), request['tasks']
        assert sum(j['budget_seconds'] for j in request['jobs']) == CANDIDATE_CAP
        assert file_hash(original_path) == original_hash
        assert {k: v for k, v in candidate.items() if k != 'decision_contract'} == {k: v for k, v in original.items() if k != 'decision_contract'}
        requests.append(request)
        plans.append({'recipe_label': label, 'candidate_id': candidate_id,
            'candidate_revision': request['candidate_revision'], 'declaration': str(declaration.relative_to(ROOT)),
            'declaration_sha256': file_hash(declaration), 'original_declaration': str(original_path.relative_to(ROOT)),
            'original_declaration_sha256': original_hash, 'decision_status': 'READY',
            'decision_contract_sha256': stable_hash(contract), 'jobs_sha256': expected['jobs_sha256']})
        for task_id, (_, horizon) in zip(task_ids, TASKS):
            job = next(j for j in request['jobs'] if j['task_id'] == task_id)
            arms.append({'id': label + '_schedule' + str(horizon), 'candidate_id': candidate_id,
                'candidate_revision': request['candidate_revision'], 'task_id': task_id,
                'schedule_horizon': horizon, 'recipe_label': label,
                'declaration': str(declaration.relative_to(ROOT)), 'compatibility_key': job['compatibility_key']})
    assert len({r['source']['digest'] for r in requests}) == 1
    assert file_hash(ROOT / 'configs/forge/tasks/two_pole.json') == original_task_hash
    plan = {'schema_version': 1, 'study': STUDY, 'campaign_id': STUDY, 'view': VIEW,
        'source_digest': requests[0]['source']['digest'], 'source_file_count': len(requests[0]['source']['files']),
        'protocol_sha256': stable_hash(requests[0]['protocol']), 'runtime': requests[0]['runtime'],
        'compute': requests[0]['compute_profiles'], 'scientific_python_executable': sys.executable,
        'seed': 0, 'through_tier': 1, 'qualification_input': False,
        'task_ceiling_seconds': TASK_CAP, 'candidate_ceiling_seconds': CANDIDATE_CAP,
        'campaign_ceiling_seconds': CAMPAIGN_CAP, 'maximum_workers': 1,
        'execution_updates_per_arm': 800, 'milestones': [80, 200, 400, 800],
        'original_two_pole_task_sha256': original_task_hash, 'candidate_plans': plans,
        'arms': arms, 'prior_evidence': motivations,
        'scope': 'One bounded diagnostic round. Existing global recipes only; no seeds/new techniques/task-specific recipes or original gate changes. Standard24checks/fivepassingterminalchecks per800-step diagnostic; extra milestones/prefix do not enter its gate. No follow-on search or default adoption.'}
    atomic_json(REPORT / 'plans.json', plan)
    print(json.dumps({'event': 'diagnostic_prepared', 'study': STUDY, 'arms': 4,
        'declared_maximum_seconds': CAMPAIGN_CAP, 'source_digest': plan['source_digest']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
