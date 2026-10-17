"""Run the prescribed CUDA jobs once, serially, with immutable source guards."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parent
PYTHON = '/tmp/pr38-default-env/bin/python'
TASKS = ('mode_hold', 'img_blobs4', 'vector_unequal_mass', 'ring_shift', 'grid100', 'rotated100', 'staggered100', 'stationary', 'img_intensity2', 'img_stripes2', 'img_bars4', 'vector_two_broad', 'vector_unequal_width', 'vector_anisotropic', 'vector_overlap', 'vector_spiral')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def event(event_type, **values):
    row = dict(event=event_type, utc=datetime.now(timezone.utc).isoformat(), **values)
    with (ROOT / 'run.log').open('a') as f:
        f.write(json.dumps(row, allow_nan=True) + '\n')
    print(json.dumps(row, allow_nan=True), flush=True)

def check_sources(frozen):
    failures = [str(ROOT / name) for name, expected in frozen['local_sources'].items() if sha(ROOT / name) != expected]
    failures += [name for name, expected in frozen['external_sources'].items() if sha(name) != expected]
    if failures:
        event('source_integrity_failure', failures=failures)
        raise SystemExit(2)

def jobs():

    def learned(problem):
        return dict(name=f'learned-{problem}-CB64-RA8', threads=2, command=[PYTHON, '-u', str(ROOT / 'learned/run_training.py'), '--problem', problem, '--variant', 'CB64-RA8'], result=ROOT / 'learned/training' / problem / 'CB64-RA8' / 'result.json')
    yield learned('toy')
    yield dict(name='screen-grid100', threads=1, command=[PYTHON, '-u', str(ROOT / 'screens/run_screen.py'), '--task', 'grid100'], result=ROOT / 'screens/runs/grid100/result.json')
    yield learned('mnist')
    yield dict(name='replay-CB64-RA8', threads=2, command=[PYTHON, '-u', str(ROOT / 'learned/replay.py'), 'CB64-RA8'], result=ROOT / 'learned/replay-CB64-RA8.json')
    for task in TASKS:
        if task == 'grid100':
            continue
        yield dict(name=f'screen-{task}', threads=1, command=[PYTHON, '-u', str(ROOT / 'screens/run_screen.py'), '--task', task], result=ROOT / 'screens/runs' / task / 'result.json')

def main():
    frozen = json.loads((ROOT / 'source-freeze.json').read_text())
    check_sources(frozen)
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--through', type=int, default=19)
    args = parser.parse_args()
    lock = (ROOT / '.execution.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs_file = ROOT / 'jobs.json'
    plan = [{**j, 'result': str(j['result']), 'log': str(logs / (j['name'] + '.log'))} for j in jobs()]
    if not 1 <= args.through <= len(plan):
        raise SystemExit('Invalid phase boundary')
    if jobs_file.exists():
        if json.loads(jobs_file.read_text()) != plan:
            raise SystemExit('Frozen job plan mismatch')
        started = json.loads((ROOT / 'execution-started.json').read_text())
        if started['source_freeze_sha256'] != sha(ROOT / 'source-freeze.json'):
            raise SystemExit('Source freeze changed since first phase')
        completed = json.loads((ROOT / 'execution-results.json').read_text())
        for index, receipt in enumerate(completed):
            job = plan[index]
            if receipt['name'] != job['name'] or receipt['result'] != job['result'] or receipt['returncode'] != 0 or (receipt['status'] == 'ERROR') or (sha(job['result']) != receipt['result_sha256']):
                raise SystemExit('Previous completed-job integrity failure')
        event('queue_resume', completed=len(completed), through=args.through)
    else:
        if (ROOT / 'execution-started.json').exists():
            raise SystemExit('Incomplete execution metadata')
        jobs_file.write_text(json.dumps(plan, indent=2) + '\n')
        (ROOT / 'execution-started.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(), jobs=len(plan), source_freeze_sha256=sha(ROOT / 'source-freeze.json'), numerical_gpu_parallelism=1, physical_gpu=0), indent=2) + '\n')
        event('queue_start', jobs=len(plan), physical_gpu=0, serial=True)
        completed = []
    for job in plan[len(completed):args.through]:
        check_sources(frozen)
        output = Path(job['result'])
        if output.exists():
            raise SystemExit(f'Result already exists before its job: {output}')
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = str(job['threads'])
        for name in ('ABSENT', 'ABSENT_START', 'ABSENT_END', 'LRFREE_NATIVE_TEST_STEPS'):
            env.pop(name, None)
        event('job_start', name=job['name'], command=job['command'], log=job['log'])
        start = time.monotonic()
        with Path(job['log']).open('x') as log:
            process = subprocess.run(job['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        check_sources(frozen)
        record = json.loads(output.read_text()) if output.exists() else None
        if job['name'].startswith('replay-'):
            status = 'PASS' if record and all((r.get('status') == 'PASS' for r in record.values())) else 'ERROR' if record is None else 'FAIL'
        else:
            status = record.get('status', 'COMPLETE') if record else 'ERROR'
        receipt = dict(name=job['name'], returncode=process.returncode, status=status, wall_seconds=time.monotonic() - start, result=str(output), log=job['log'], result_sha256=sha(output) if output.exists() else None)
        completed.append(receipt)
        (ROOT / 'execution-results.json').write_text(json.dumps(completed, indent=2) + '\n')
        event('job_complete', **receipt)
        if process.returncode != 0 or status == 'ERROR':
            event('queue_aborted', name=job['name'], reason='runtime or evidence error')
            raise SystemExit(1)
    event('queue_complete' if len(completed) == len(plan) else 'phase_complete', jobs=len(completed), statuses={s: sum((r['status'] == s for r in completed)) for s in ('COMPLETE', 'PASS', 'FAIL', 'ERROR')})
if __name__ == '__main__':
    main()
