"""Admit explicit deeper diagnostics after a completed failed ordinary screen.

Full original Tier2 gates remain fixed. A hold with no fully passing own
producer remains BLOCKED; it is not run from an attractive partial checkpoint.
Passing producers are replayed in the isolated diagnostic scope because Forge
deliberately prevents diagnostic jobs from aliasing ordinary qualification jobs.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

from experiments.forge.contracts import atomic_json, read_json, utc_now
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue, drain

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next')
PREFIX = 'bcap-develop-integration'


def main():
    state = read_json(ARCHIVE/'queue/queue/state.json')
    progress = read_json(ARCHIVE/'progress.json')
    request = state['submissions'][progress['requests']['combined']]['request']
    tier1 = [a['task'] for a in request['view']['assignments']
             if a['qualification_tier']==1 and a['importance']=='required']
    grades = {}
    for job in state['jobs'].values():
        result = job.get('result') or {}
        if result.get('candidate_revision') == request['candidate_revision']:
            for row in result.get('task_results', []):
                grades[row['task_id']] = row['gate_status']
    assert all(t in grades for t in tier1), 'Complete all runnable ordinary Tier1 cells first'
    assert any(grades[t] != 'PASS' for t in tier1), 'Ordinary Tier2 eligible; diagnostics unnecessary'
    required = [a['task'] for a in request['view']['assignments']
                if a['qualification_tier']==2 and a['importance']=='required']
    selected, producers, blocked = [], [], []
    for name in required:
        task = request['tasks'][name]
        dependencies = [d['task'] for d in task['dependencies']]
        bad = [p for p in dependencies if grades.get(p) != 'PASS']
        if bad:
            blocked.append(dict(task_id=name, reason='Own required producer did not completely PASS',
                                producer_grades={p:grades.get(p, 'NOT_RUN') for p in bad}))
            continue
        producers.extend(p for p in dependencies if p not in producers)
        selected.append(name)
    selected = producers + selected
    view = deepcopy(request['view'])
    view.update(id=f'{PREFIX}-deeper-diagnostic-v1', revision=1, evidence_scope='research_diagnostic',
        policy_change_reason='Complete remaining full-gate measurements after a failed integrated ordinary screen; separate diagnostic evidence and own checkpoint dependencies, no qualification credit.',
        assignments=[dict(task=t, qualification_tier=1, importance='diagnostic', order=i) for i,t in enumerate(selected)])
    atomic_json(ROOT/f'configs/forge/views/{view["id"]}.json', view)
    study = deepcopy(read_json(ROOT/f'configs/forge/studies/{PREFIX}-combined-study-v1.json'))
    study.update(id=f'{PREFIX}-combined-deeper-study-v1',
        hypothesis='Characterize whether the same global combined recipe retains its measured conditional/rare repairs and incumbent full-gate passes on the remaining Tier2 tasks despite its failed ordinary Tier1 screen.')
    study['scope'].update(view=view['id'], through_tier=1)
    study['campaign'].update(id=f'{PREFIX}-deeper-v1', budget_seconds=45000, candidate_budget_seconds=45000)
    atomic_json(ROOT/f'configs/forge/studies/{study["id"]}.json', study)
    diagnostic = resolve_idea(ROOT, f'{PREFIX}-combined-v1', study=study['id'],
                              queue_root=ARCHIVE/'diagnostic-queue', freeze_source=False)
    assert diagnostic['source']['digest'] == request['source']['digest'], 'Do not change measured scientific bytes'
    assert diagnostic['candidate_revision'] == request['candidate_revision']
    assert diagnostic['study_review']['status'] == 'READY', diagnostic['study_review']
    assert not diagnostic['preflight_blockers'], diagnostic['preflight_blockers']
    reservation = sum(j['budget_seconds'] for j in diagnostic['jobs'])
    assert reservation <= 45000
    atomic_json(OUT/'deeper-preregistration.json', dict(schema_version=1, qualification_input=False,
        evidence_scope='research_diagnostic', ordinary_request=progress['requests']['combined'],
        ordinary_tier1=grades, required_tier2=required, measured_tasks=selected,
        passing_producers_replayed=producers, blocked_holds=blocked,
        source_digest=request['source']['digest'], candidate_revision=request['candidate_revision'],
        admission=diagnostic['study_review']['status'], full_reservation_seconds=reservation,
        campaign_ceiling_seconds=45000, total_main_and_diagnostic_ceiling_seconds=135000,
        software_allowance_seconds=3600, stopping_rule='One unchanged global revision, full gates, no tuning or seeds; no ordinary qualification credit.'))
    print(json.dumps(dict(event='diagnostic_preregistered', selected=selected, blocked=blocked,
                          reservation=reservation)), flush=True)
    if '--run' not in sys.argv:
        return
    diagnostic = resolve_idea(ROOT, f'{PREFIX}-combined-v1', study=study['id'],
                              queue_root=ARCHIVE/'diagnostic-queue', freeze_source=True)
    queue = Queue(ARCHIVE/'diagnostic-queue', report_root=ROOT/'reports/forge', on_completion=None)
    receipt = queue.submit(diagnostic, diagnostic['study']['campaign'])
    details = dict(phase='running', updated_at=utc_now(), requests=dict(combined=receipt['request']['request_id']),
                   source=diagnostic['source']['digest'], ordinary_requests=progress['requests'])
    atomic_json(ARCHIVE/'diagnostic-progress.json', details)
    print(json.dumps(dict(event='diagnostic_submitted', time=utc_now(), request=details['requests'])), flush=True)
    drain(queue, ['0','1'], workers_per_gpu=1, allow_sharing=True, watch=False, campaign=f'{PREFIX}-deeper-v1')
    details.update(phase='complete', updated_at=utc_now())
    atomic_json(ARCHIVE/'diagnostic-progress.json', details)
    print(json.dumps(dict(event='diagnostic_complete', time=utc_now())), flush=True)


if __name__ == '__main__':
    main()
