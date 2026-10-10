"""Recover supervision without modifying frozen numerical sources or jobs."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
QUEUE = HERE.parent / 'validation'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return fields[0], fields[19]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orphan-pid', type=int, required=True)
    parser.add_argument('--orphan-start-ticks', required=True)
    args = parser.parse_args()
    lock = (HERE / '.supervisor.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (HERE / 'STARTED.json').exists():
        raise SystemExit('Recovery has already started; refusing duplicate supervision.')
    spec = importlib.util.spec_from_file_location('frozen_queue', QUEUE / 'launch.py')
    frozen_queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen_queue)
    frozen = read(QUEUE / 'source-freeze.json')
    freeze_sha = sha(QUEUE / 'source-freeze.json')
    supervisor_sha = sha(__file__)

    def guard():
        if sha(__file__) != supervisor_sha or sha(QUEUE / 'source-freeze.json') != freeze_sha:
            raise RuntimeError('Recovery or original source-freeze changed.')
        frozen_queue.check_sources(frozen)

    guard()
    plan = read(QUEUE / 'jobs.json')
    expected = [{**j, 'result': str(j['result']),
                 'log': str(QUEUE / 'logs' / (j['name'] + '.log'))}
                for j in frozen_queue.jobs()]
    if plan != expected:
        raise RuntimeError('Saved plan differs from frozen launch.py.')
    completed = read(QUEUE / 'execution-results.json')
    for index, row in enumerate(completed):
        job = plan[index]
        if (row['name'] != job['name'] or row['result'] != job['result']
                or row['returncode'] != 0 or row['status'] == 'ERROR'
                or sha(row['result']) != row['result_sha256']):
            raise RuntimeError('Completed-job identity or evidence changed.')
    job = plan[len(completed)]
    if job['name'] != 'screen-grid100':
        raise RuntimeError('This recovery is for the interrupted grid100 supervisor only.')
    state, stamp = identity(args.orphan_pid)
    argv = Path(f'/proc/{args.orphan_pid}/cmdline').read_bytes().decode().split('\0')
    if stamp != args.orphan_start_ticks or not all(arg in argv for arg in job['command'][2:]):
        raise RuntimeError('Orphan process identity differs from frozen active job.')
    write(HERE / 'STARTED.json', dict(
        pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        original_validation=str(QUEUE), completed_jobs=len(completed),
        adopted_job=job['name'], orphan_pid=args.orphan_pid,
        orphan_start_ticks=stamp, source_freeze_sha256=freeze_sha,
        supervisor_sha256=supervisor_sha,
        policy='same frozen numerical jobs; no completed job repeated; no budget reset'))
    frozen_queue.event('queue_supervisor_recovery', supervisor_pid=os.getpid(),
                       adopted_job=job['name'], orphan_pid=args.orphan_pid,
                       supervisor_sha256=supervisor_sha)
    last_notice = 0.
    while True:
        try:
            state, current_stamp = identity(args.orphan_pid)
        except FileNotFoundError:
            break
        if current_stamp != stamp:
            raise RuntimeError('Orphan PID was reused before finalization.')
        if state in ('Z', 'X'):
            break
        if time.monotonic() - last_notice > 30:
            print(json.dumps(dict(event='waiting_for_adopted_job', name=job['name'],
                                  pid=args.orphan_pid)), flush=True)
            last_notice = time.monotonic()
        time.sleep(.5)
    guard()
    execution = read(QUEUE / 'screens/runs/grid100/execution-receipt.json')
    result = read(job['result'])
    code = execution.get('process_exit_code')
    if (not isinstance(code, int) or execution.get('status') != 'COMPLETE'
            or execution['result_sha256'] != sha(job['result'])
            or execution['source_integrity_after']['status'] != 'VALID'):
        raise RuntimeError('Adopted wrapper did not produce complete valid final evidence.')
    receipt = dict(name=job['name'], returncode=code, status=result['status'],
                   wall_seconds=execution['wall_seconds'], result=job['result'],
                   log=job['log'], result_sha256=sha(job['result']),
                   returncode_source='frozen wrapper finalized process_exit_code',
                   os_wait_status_observed=False, supervisor_recovered=True)

    def save(row):
        completed.append(row)
        write(QUEUE / 'execution-results.json', completed)
        frozen_queue.event('job_complete', **row)
        if row['returncode'] != 0 or row['status'] == 'ERROR':
            frozen_queue.event('queue_aborted', name=row['name'], reason='runtime or evidence error')
            raise SystemExit(1)

    save(receipt)
    for job in plan[len(completed):]:
        guard()
        if Path(job['result']).exists() or Path(job['log']).exists():
            raise RuntimeError('Uncompleted job has existing output; refusing overwrite.')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONUNBUFFERED='1')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = str(job['threads'])
        for key in ('ABSENT', 'ABSENT_START', 'ABSENT_END', 'LRFREE_NATIVE_TEST_STEPS'):
            env.pop(key, None)
        frozen_queue.event('job_start', name=job['name'], command=job['command'], log=job['log'])
        start = time.monotonic()
        with Path(job['log']).open('x') as output:
            process = subprocess.run(job['command'], cwd=QUEUE, env=env,
                                     stdout=output, stderr=subprocess.STDOUT)
        guard()
        result = read(job['result']) if Path(job['result']).exists() else None
        save(dict(name=job['name'], returncode=process.returncode,
                  status=result.get('status', 'COMPLETE') if result else 'ERROR',
                  wall_seconds=time.monotonic()-start, result=job['result'],
                  log=job['log'], result_sha256=sha(job['result']) if result else None,
                  os_wait_status_observed=True, supervisor_recovered=True))
    frozen_queue.event('queue_complete', jobs=len(completed),
                       statuses={s: sum(row['status'] == s for row in completed)
                                 for s in ('COMPLETE', 'PASS', 'FAIL', 'ERROR')})
    write(HERE / 'COMPLETE.json', dict(status='COMPLETE', jobs=len(completed),
          source_freeze_sha256=freeze_sha, supervisor_sha256=supervisor_sha,
          execution_results_sha256=sha(QUEUE / 'execution-results.json')))


if __name__ == '__main__':
    main()
