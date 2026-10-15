"""Validate, enqueue and drain explicitly selected existing READY studies."""
import argparse, json, os, subprocess, time
from pathlib import Path
from experiments.forge.planning import resolve_idea
from experiments.forge.studies import validate_admission
from experiments.forge.queue import Queue, drain

parser = argparse.ArgumentParser()
parser.add_argument('--campaign', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--entry', action='append', nargs=2, required=True)
args = parser.parse_args()
root = Path('/ml2/hypergan/particlegan-selective-history-20261015')
out = Path('/ml2/hypergan/combination-training-20261015')
stage = Path(args.output)
stage.mkdir(exist_ok=True)
assert not (stage / 'launch-manifest.json').exists(), 'inspect prior launch before resuming'
initial = json.loads((out / 'b-planned-request.json').read_text())
python = '/ml2/hypergan/.venvs/particlegan-selective-history-20261015/bin/python'
base = [python, '-m', 'experiments.forge', '--root', str(root), '--queue-root', str(out / 'queue')]
validation = subprocess.run(base + ['validate'], cwd=root, capture_output=True, text=True)
(stage / 'validate.stdout.json').write_text(validation.stdout)
(stage / 'validate.stderr.log').write_text(validation.stderr)
assert validation.returncode == 0, validation.stderr
requests = []
def permitted_jobs(request):
    names = {a['task'] for a in request['view']['assignments']
             if a['qualification_tier'] <= request['through_tier']}
    return [job for job in request['jobs']
            if names.intersection(job.get('task_ids', [job['task_id']]))]
for candidate, study in args.entry:
    request = resolve_idea(root, candidate, study=study, queue_root=out / 'queue')
    assert not request['preflight_blockers'], request['preflight_blockers']
    active_jobs = permitted_jobs(request)
    assert all(not request['tasks'][name]['preflight_blockers'] for job in active_jobs
               for name in job.get('task_ids', [job['task_id']]))
    assert request['study']['campaign']['id'] == args.campaign
    assert sum(j['budget_seconds'] for j in active_jobs) <= request['study']['campaign']['candidate_budget_seconds']
    old_files, new_files = initial['source']['files'], request['source']['files']
    assert set(old_files) <= set(new_files)
    assert all(new_files[name] == value for name, value in old_files.items())
    declared_inputs = {name for task in request['tasks'].values()
                       for name in task.get('evaluation', {}).get('sources', {})}
    assert set(new_files) - set(old_files) <= declared_inputs
    assert request['runtime'] == initial['runtime'] and request['protocol'] == initial['protocol']
    if requests:
        assert request['source'] == requests[0]['source'] and request['compute_profiles'] == requests[0]['compute_profiles']
    validate_admission(root, request, request['study']['campaign'])
    (stage / f'{candidate}-plan.json').write_text(json.dumps(request, indent=2, sort_keys=True) + '\n')
    requests.append(request)
manifest = {'campaign': args.campaign, 'source_digest': requests[0]['source']['digest'],
            'runtime': requests[0]['runtime'], 'driver_pid': os.getpid(), 'time_unix': time.time(),
            'commands': [], 'submissions': [], 'roster': [
                {'candidate': r['candidate']['id'], 'study': r['study']['id'], 'candidate_revision': r['candidate_revision'],
                 'jobs': [{'task': j['task_id'], 'key': j['compatibility_key'], 'budget_seconds': j['budget_seconds']}
                          for j in permitted_jobs(r)],
                 'through_tier': r['through_tier'],
                 'deferred_job_descriptors': len(r['jobs']) - len(permitted_jobs(r))} for r in requests]}
def save():
    (stage / 'launch-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
for candidate, study in args.entry:
    command = base + ['enqueue', candidate, '--study', study]
    manifest['commands'].append(command)
    save()
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    (stage / f'{candidate}-enqueue.stdout.json').write_text(result.stdout)
    (stage / f'{candidate}-enqueue.stderr.log').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    manifest['submissions'].append(receipt)
    save()
    print('ENQUEUED', candidate, json.dumps(receipt), flush=True)
manifest['drain_started_unix'] = time.time()
save()
print('DRAIN START', args.campaign, os.getpid(), flush=True)
drain(Queue(out / 'queue', report_root=root / 'reports/forge', on_completion=None),
      ['0', '1'], workers_per_gpu=1, allow_sharing=False, campaign=args.campaign)
manifest['drain_finished_unix'] = time.time()
save()
print('DRAIN COMPLETE', flush=True)
