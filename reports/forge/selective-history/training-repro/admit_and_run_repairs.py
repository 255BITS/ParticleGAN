"""Execute the root-selected two-candidate repair campaign, preserving initial evidence."""
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from experiments.forge.planning import resolve_idea
from experiments.forge.studies import validate_admission
from experiments.forge.queue import Queue, drain

root = Path('/ml2/hypergan/particlegan-selective-history-20261015')
out = Path('/ml2/hypergan/combination-training-20261015')
repair = out / 'repairs'
repair.mkdir(exist_ok=True)
campaign = 'bcap-selective-history-repair-screen-v1'
python = '/ml2/hypergan/.venvs/particlegan-selective-history-20261015/bin/python'
initial = json.loads((out / 'completion.json').read_text())
assert initial['status'] == 'COMPLETE' and initial['completed_attempts'] == 8
assert initial['scientific_retries'] == 0 and initial['reserved_seconds'] == 0
assert not Path('/proc/1014142').exists(), 'initial coordinator must have finished'
assert not (repair / 'launch-manifest.json').exists(), 'repair launch already attempted; inspect state before resuming'

evidence = copy.deepcopy(initial)
evidence['transport_attempts'] = {j['task']: j['attempt_id'] for j in initial['jobs']
                                if j['candidate'] == 'bcap-selective-history-transport-v1'}
evidence['selective_attempts'] = {j['task']: j['attempt_id'] for j in initial['jobs']
                                if j['candidate'] == 'bcap-selective-history-adversarial-v1'}
for job in evidence['jobs']:
    certified = json.loads((Path(job['external_artifact_directory']) / 'graded-result.json').read_text())
    job['convergence'] = certified['grades'][job['task']]['evaluator_result']['convergence']
    job['durable_receipt_directory'] = str(Path(job['durable_receipt_directory']).relative_to(root))
evidence_path = root / 'reports/forge/selective-history/initial-training-v2.json'
evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
evidence_ref = {'path': str(evidence_path.relative_to(root)), 'selector': [], 'use': 'motivation_only',
                'identity': {key: evidence[key] for key in ['campaign', 'source_digest', 'transport_attempts', 'selective_attempts']}}

t = json.loads((root / 'configs/forge/ideas/bcap-selective-history-transport-v1.json').read_text())
s = json.loads((root / 'configs/forge/ideas/bcap-selective-history-adversarial-v1.json').read_text())
primary = copy.deepcopy(t)
primary.update(id='bcap-transport-penalty2-v1', parent=t['id'], mechanism_class='floor_constant',
               changed_factors=['Double only the existing BCAP penalty coefficient from 1 to 2; preserve T transport, history off, all role rates and original task conditions.'],
               mechanism_rationale='T passed the complete Ring gate but failed sustained pole slopes. Strengthen the existing critic restoring penalty while retaining the measured history-off Ring behavior. This soft penalty cannot guarantee a finite cap or prevent first crossings; both complete gates remain required.')
primary['recipe_overrides']['reg_coeff'] = 2.0
assert primary['requires_capabilities'] == t['requires_capabilities']
assert 'generator_component_history' not in primary['requires_capabilities']
secondary = json.loads((out / 'repair-drafts/bcap-selective-history-halfjoint-v1.json').read_text())
expected = copy.deepcopy(s['recipe_overrides'])
expected.update(lr=.006, d_lr_mult=3.0)
assert secondary['recipe_overrides'] == expected
expected = copy.deepcopy(t['recipe_overrides'])
expected['reg_coeff'] = 2.0
assert primary['recipe_overrides'] == expected

primary_study = json.loads((out / 'repair-drafts/bcap-selective-history-penalty2-v1-pole-ring-study-v1.json').read_text())
primary_study.update(id='bcap-transport-penalty2-pole-ring-study-v1', candidate=primary['id'],
                     control={'candidate_id': t['id'], 'task_map': {}},
                     hypothesis='Fresh complete T passes Ring with terminal suffix 6 but fails pole with terminal suffix 1 and 11 slope violations. Test only reg_coeff 1 to 2 on T, with generator history off. The stronger soft critic penalty may reduce pole overshoot while retaining Ring. Both unchanged full gates must pass; final scalar signatures alone are insufficient.',
                     competing_explanation='A stronger coefficient can change critic response without bounding the actual finite update, and may damage T Ring behavior. Unlike the withdrawn S+penalty2 draft, this primary starts from T, avoiding the additional measured S Ring deficit. Complete both independent tasks and retain every failure.')
secondary_study = json.loads((out / 'repair-drafts/bcap-selective-history-halfjoint-v1-pole-ring-study-v1.json').read_text())
secondary_study['hypothesis'] = ('Fresh S fails pole and Ring; its Ring retains all 16 modes but final HQ and covariance fail. '
    'Test half G/E/prior finite motion with D rate held at .018: lr .006, d_lr_mult 3, prior_lr_mult unchanged at 2.5. '
    'This may reduce overshoot, but slower acquisition and weaker pole travel are risks. Both original complete 80/1600-update gates must pass.')
studies = [primary_study, secondary_study]
for card, study in zip([primary, secondary], studies):
    study.update(status='ready', prior_evidence=[evidence_ref])
    assert study['campaign'] == {'id': campaign, 'budget_seconds': 1200, 'candidate_budget_seconds': 600}
    (root / f'configs/forge/ideas/{card["id"]}.json').write_text(json.dumps(card, indent=2, sort_keys=True) + '\n')
    (root / f'configs/forge/studies/{study["id"]}.json').write_text(json.dumps(study, indent=2, sort_keys=True) + '\n')

selection = {'authorization': 'Root explicitly selected T+reg_coeff2 and S+halfjoint after certified initial eight completed, under the user instruction train it make it work.',
             'campaign': campaign, 'selected': [primary['id'], secondary['id']],
             'withdrawn_unsubmitted_draft': 'bcap-selective-history-penalty2-v1',
             'reason': 'T already passes Ring, whereas S additionally fails Ring; primary targets T pole overshoot without adding history.',
             'initial_completion_sha256': hashlib.sha256((out / 'completion.json').read_bytes()).hexdigest(),
             'initial_evidence_sha256': hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
             'candidate_budget_seconds': 600, 'campaign_budget_seconds': 1200,
             'diagnostics_profile': 'same default as all initial arms; no added optimizer diagnostics',
             'source_algorithm_changes': False, 'protocol_seed': 0}
(repair / 'selection.json').write_text(json.dumps(selection, indent=2, sort_keys=True) + '\n')

base = [python, '-m', 'experiments.forge', '--root', str(root), '--queue-root', str(out / 'queue')]
validation = subprocess.run(base + ['validate'], cwd=root, capture_output=True, text=True)
(repair / 'validate.stdout.json').write_text(validation.stdout)
(repair / 'validate.stderr.log').write_text(validation.stderr)
assert validation.returncode == 0, validation.stderr
requests = []
for study in studies:
    request = resolve_idea(root, study['candidate'], study=study['id'], queue_root=out / 'queue')
    assert not request['preflight_blockers'], request['preflight_blockers']
    assert all(not t['preflight_blockers'] for t in request['tasks'].values())
    assert len(request['jobs']) == 2 and sum(j['budget_seconds'] for j in request['jobs']) == 600
    assert request['protocol']['seed'] == 0
    assert request['source']['digest'] == initial['source_digest'], 'scientific source changed'
    validate_admission(root, request, study['campaign'])
    if requests:
        assert request['source'] == requests[0]['source'] and request['runtime'] == requests[0]['runtime']
    (repair / f'{study["candidate"]}-plan.json').write_text(json.dumps(request, indent=2, sort_keys=True) + '\n')
    requests.append(request)
manifest = {**selection, 'source_digest': requests[0]['source']['digest'], 'runtime': requests[0]['runtime'],
            'time_unix': time.time(), 'commands': [], 'submissions': [], 'driver_pid': os.getpid(),
            'roster': [{'candidate': r['candidate']['id'], 'study': r['study']['id'], 'candidate_revision': r['candidate_revision'],
                        'jobs': [{'task': j['task_id'], 'key': j['compatibility_key'], 'budget_seconds': j['budget_seconds']} for j in r['jobs']]}
                       for r in requests]}
def save():
    (repair / 'launch-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
for study in studies:
    command = base + ['enqueue', study['candidate'], '--study', study['id']]
    manifest['commands'].append(command)
    save()
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    (repair / f'{study["candidate"]}-enqueue.stdout.json').write_text(result.stdout)
    (repair / f'{study["candidate"]}-enqueue.stderr.log').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    manifest['submissions'].append(receipt)
    save()
    print('ENQUEUED', study['candidate'], json.dumps(receipt), flush=True)
manifest['drain_started_unix'] = time.time()
save()
print('DRAIN START', campaign, os.getpid(), flush=True)
queue = Queue(out / 'queue', report_root=root / 'reports/forge', on_completion=None)
drain(queue, ['0', '1'], workers_per_gpu=1, allow_sharing=False, campaign=campaign)
manifest['drain_finished_unix'] = time.time()
save()
print('DRAIN COMPLETE', flush=True)
