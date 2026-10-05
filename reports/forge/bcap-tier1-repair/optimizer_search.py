"""Run the final four shorter-moment-memory contrasts inside the repair cap."""
import argparse
from copy import deepcopy
from pathlib import Path
from prepare import BASE, QUEUE, REPORT, ROOT, ROUND_CAP, bind_contract, emit
from run import verify_source
from experiments.forge.configuration_search import materialize_search, plan_search, enqueue_search, report_search
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue

STUDY = 'bcap-tier1-repair-moments-v1'
PLAN = REPORT / 'moment-plans.json'


def prepare():
    if (ROOT / f'reports/forge/configuration-search/{STUDY}.json').exists():
        raise ValueError('registered study is immutable')
    state = Queue(QUEUE, on_completion=None).inspect()
    if STUDY in state['campaigns']:
        raise ValueError('admitted study is immutable')
    results = read_json(REPORT / 'results.json')
    from publish import attempts_and_rows
    _, _, actual_rows = attempts_and_rows()
    if actual_rows != results['candidates']:
        raise ValueError('readout differs from the actual queue receipts')
    if any(abs(v['reserved_seconds']) > 1e-6 for v in state['campaigns'].values()):
        raise ValueError('conclude existing reservations before another contrast')
    if {r['campaign'] for r in actual_rows} != {
            'bcap-tier1-repair-rates-v1', 'bcap-original-horizon-diagnostics-v1'}:
        raise ValueError('first-round campaign identity differs')
    if len(results['candidates']) != 9 or any(r['submission_status'] in {'queued', 'running', 'paused'} for r in results['candidates']):
        raise ValueError('conclude the eight rate recipes and two duration tasks first')
    if any(t['status'] not in {'PASS', 'FAIL'} for r in results['candidates'] for t in r['tasks']):
        raise ValueError('all original-round tasks must have certified numerical outcomes')
    first_round = REPORT / 'first-round.json'
    if first_round.exists() and read_json(first_round) != results:
        raise ValueError('retain the exact concluded first-round evidence')
    atomic_json(first_round, results)
    cap = read_json(REPORT / 'plans.json')['candidate_cap_seconds']
    prior_reserved = sum(v['definition']['budget_seconds'] for v in state['campaigns'].values())
    if prior_reserved + 4 * cap + cap > ROUND_CAP:
        raise ValueError('four contrasts plus one whole-recipe allowance exceed repair-round cap')
    spec = deepcopy(read_json(ROOT / 'configs/forge/searches/bcap-tier1-repair-rates-v1.json'))
    spec.update(id=STUDY, grid={'lr': [.00425], 'd_lr_mult': [2.], 'prior_lr_mult': [1., 2.],
        'reg_coeff': [1.], 'betas': [[0., .9], [0., .99]]},
        campaign={'id': STUDY, 'budget_seconds': 4 * cap, 'candidate_budget_seconds': cap, 'accept_shared_cost_transfer': False},
        hypothesis='Shorter Adam second-moment memory permits ring acquisition at the movement-compatible global LR while slower prior transport limits Gaussian drift.',
        rationale='Lower global rates produced sustained Gaussian passes but failed direct movement and ring400 acquisition. Restore LR.00425/D2, retain cap1/coefficient1/noise/all task budgets, compare beta2.9/.99 and prior1/2. This tests a hypothesis, not a demonstrated stale-moment bug: shorter memory can increase late normalized steps and worsen drift/tails. Do not rerun an unchanged beta2.999 control. Complete all seven independent Tier1 jobs and report one whole recipe; no seed study or higher tiers.')
    path = ROOT / f'configs/forge/searches/{STUDY}.json'
    atomic_json(path, spec)
    evidence = [{'path': str(first_round.relative_to(ROOT)), 'sha256': file_hash(first_round),
        'selector': [], 'identity': {'scope': results['scope']}, 'use': 'motivation_only'}]
    reviews = []
    for declaration in materialize_search(ROOT, spec):
        idea = read_json(declaration)
        if idea['search_study_id'] != STUDY:
            raise ValueError('would repeat an existing configuration')
        request = bind_contract(declaration, view=spec['view'], candidate_cap=cap, campaign_cap=4 * cap,
            control=BASE, evidence=evidence,
            competing_explanation='Shorter moment memory can increase late normalized steps and worsen Gaussian drift/radial tails; a final ring improvement must retain every sustained acquisition and movement gate.',
            prediction={'task_id': 'ring16_acquisition', 'metric': 'component_covariance_error', 'op': '<=', 'threshold': .85, 'phase': 'final'},
            falsifier={'task_id': 'ring16_acquisition', 'metric': 'component_covariance_error', 'op': '>', 'threshold': .85, 'phase': 'final'})
        reviews.append({'candidate': idea['id'], 'declaration': str(declaration.relative_to(ROOT)),
            'declaration_sha256': file_hash(declaration), 'decision_status': request['decision_review']['status']})
    plan = plan_search(ROOT, QUEUE, spec)
    if any(t['submission_status'] != 'READY' for t in plan['trials']):
        raise ValueError('all four contrasts must be ready')
    atomic_json(PLAN, {'schema_version': 1, 'study': STUDY, 'spec': str(path.relative_to(ROOT)),
        'spec_sha256': file_hash(path), 'source_digest': plan['source_digest'], 'trials': reviews,
        'prior_declared_allowance_seconds': prior_reserved, 'campaign_cap_seconds': 4 * cap,
        'remaining_whole_recipe_allowance_seconds': cap, 'round_cap_seconds': ROUND_CAP,
        'qualification_input': False})
    emit('moment_contrasts_prepared', configurations=4, campaign_cap_seconds=4 * cap)


def enqueue():
    plan = read_json(PLAN)
    if file_hash(ROOT / plan['spec']) != plan['spec_sha256']:
        raise ValueError('search specification changed since preparation')
    for trial in plan['trials']:
        if file_hash(ROOT / trial['declaration']) != trial['declaration_sha256']:
            raise ValueError('declaration changed since preparation')
        request = resolve_idea(ROOT, trial['candidate'], through_tier=1, execution_backend='cuda',
            cuda_model='NVIDIA RTX A6000', queue_root=QUEUE)
        verify_source(request['source'])
        if request['source']['digest'] != plan['source_digest']:
            raise ValueError('source changed since preparation')
    result = enqueue_search(ROOT, QUEUE, STUDY, queue=Queue(QUEUE, report_root=ROOT / 'reports/forge', on_completion=None))
    if result['blocked_count']:
        raise ValueError('moment contrast admission blocked')
    emit('moment_contrasts_enqueued', configurations=result['submitted_count'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'enqueue', 'report'])
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'enqueue': enqueue()
    else:
        report = report_search(ROOT, QUEUE, STUDY, queue=Queue(QUEUE, report_root=ROOT / 'reports/forge', on_completion=None))
        emit('moment_contrasts_complete', selection=report['selection'], cost=report['cost'])
