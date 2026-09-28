#!/usr/bin/env python
"""Screening pool daemon: claims queue/*.json jobs and runs screen.py with N slots per GPU.

Start:  nohup /tmp/pr38-default-env/bin/python harness/pool.py >> pool.log 2>&1 &
Config: pool-config.json (re-read every poll; edit slots live).  Stop: touch queue/STOP
"""
from pathlib import Path
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lrlib  # noqa: E402

STOP = lrlib.QUEUE / 'STOP'


def log(*parts):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *parts, flush=True)


def load_config():
    config = json.loads(json.dumps(lrlib.DEFAULT_CONFIG))
    if lrlib.CONFIG.exists():
        try:
            user = json.loads(lrlib.CONFIG.read_text())
            for key, value in user.items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
        except json.JSONDecodeError as error:
            log('bad pool-config.json, using defaults:', error)
    else:
        lrlib.CONFIG.write_text(json.dumps(lrlib.DEFAULT_CONFIG, indent=1) + '\n')
    return config


def gpu_free_mib():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
        return {line.split(',')[0].strip(): int(line.split(',')[1]) for line in out.strip().splitlines()}
    except Exception:
        return {}


def proc_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(')')[-1].split()[19]
    except OSError:
        return None


def alive(entry):
    return proc_start(entry['pid']) == entry.get('pid_start')


def task_kind(task):
    if task in lrlib.IMAGE_TASKS:
        return 'image'
    if task in lrlib.VECTOR_TASKS:
        return 'vector'
    return task


def summarize(job, entry, result, run_dir):
    keep = ('status', 'passing_checks', 'observations', 'first_arrival', 'final_streak', 'seconds',
            'train_seconds', 'stream_deviations', 'completed_steps', 'max_gpu_mib', 'thresholds', 'segments', 'error')
    row = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S'), cand=job['cand'], task=job['task'])
    row.update({k: result.get(k) for k in keep if k in result})
    final = result.get('final') or {}
    row['final'] = {k: v for k, v in final.items() if not isinstance(v, (list, dict)) and k not in ('pass', 'seconds')}
    ema = result.get('ema_final') or {}
    row['ema_final'] = {k: v for k, v in ema.items() if not isinstance(v, (list, dict))}
    header = result.get('header') or {}
    row.update(gpu=entry.get('gpu'), config_hash=job.get('config_hash'), run_dir=str(run_dir),
               package_sha256=header.get('package_sha256'), options=header.get('options'),
               wall_seconds=round(time.time() - entry.get('started', time.time()), 1))
    if header.get('package_sha256') and job.get('package_sha256') and header['package_sha256'] != job['package_sha256']:
        row['warning'] = 'package changed between submit and run'
    return row


def finalize(entry, killed=None):
    job = entry['job']
    run_dir = Path(entry['run_dir'])
    result_path = run_dir / 'result.json'
    result = {}
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            result = {}
    if not result or killed:
        tail = ''
        log_path = run_dir / 'log.txt'
        if log_path.exists():
            tail = log_path.read_text()[-1500:]
        result = dict(result or {}, status='ERROR', error=killed or f'no result.json; log tail: {tail}')
        (run_dir / 'result.json').write_text(json.dumps(dict(task=job['task'], cand=job['cand'], **result), indent=1))
    row = summarize(job, entry, result, run_dir)
    with open(lrlib.LEDGER, 'a') as handle:
        handle.write(json.dumps(row, default=str) + '\n')
    try:
        os.remove(entry['marker'])
    except FileNotFoundError:
        pass
    log('DONE', job['cand'], job['task'], row.get('status'), f"{row.get('passing_checks')}/{row.get('observations')}",
        f"@{row.get('first_arrival')}", f"gpu{entry.get('gpu')}", f"{row.get('wall_seconds')}s")
    try:
        importlib.reload(lrlib)  # pick up leaderboard-format edits without restarting the pool
        lrlib.rebuild_leaderboard()
    except Exception:
        log('leaderboard rebuild failed', traceback.format_exc())


def launch(path, gpu, config):
    """Claim queue file `path` for `gpu` (atomic rename) and start screen.py."""
    marker = lrlib.RUNNING / path.name
    try:
        os.rename(path, marker)
    except OSError:
        return None  # claimed by someone else
    job = json.loads(marker.read_text())
    run_dir = lrlib.RUNS / job['cand'] / job['task']
    if run_dir.exists() and any(run_dir.iterdir()):
        os.rename(run_dir, run_dir.with_name(f"{job['task']}.prev-{time.strftime('%Y%m%dT%H%M%S')}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'job.json').write_text(json.dumps(job, indent=1) + '\n')
    (run_dir / 'overrides.json').write_text(json.dumps(job['overrides'], indent=1) + '\n')
    (run_dir / 'options.json').write_text(json.dumps(job.get('candidate_options') or {}, indent=1) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONDONTWRITEBYTECODE='1')
    command = [lrlib.PYTHON, str(lrlib.HARNESS / 'screen.py'), '--package-root', job['package_root'],
               '--overrides', str(run_dir / 'overrides.json'), '--candidate-options', str(run_dir / 'options.json'),
               '--task', job['task'], '--output', str(run_dir), '--device', 'cuda:0', '--cand', job['cand']]
    with open(run_dir / 'log.txt', 'w') as out:
        proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=str(lrlib.BASE),
                                start_new_session=True)
    entry = dict(job=job, gpu=str(gpu), pid=proc.pid, pid_start=proc_start(proc.pid), started=time.time(),
                 run_dir=str(run_dir), marker=str(marker), kind=task_kind(job['task']))
    marker.write_text(json.dumps(dict(job, _running=dict(gpu=str(gpu), pid=proc.pid, pid_start=entry['pid_start'],
                                                             started=entry['started'], run_dir=str(run_dir))),
                                 indent=1))
    log('START', job['cand'], job['task'], f'gpu{gpu}', f'pid{proc.pid}')
    return entry, proc


def adopt_or_requeue():
    """After a pool restart: adopt live children, requeue dead ones without result.json."""
    entries = []
    for marker in sorted(lrlib.RUNNING.glob('*.json')):
        data = json.loads(marker.read_text())
        info = data.pop('_running', None)
        if info is None:
            os.rename(marker, lrlib.QUEUE / marker.name)
            continue
        entry = dict(job=data, gpu=info['gpu'], pid=info['pid'], pid_start=info['pid_start'], started=info['started'],
                     run_dir=info['run_dir'], marker=str(marker), kind=task_kind(data['task']))
        if alive(entry):
            log('ADOPT', data['cand'], data['task'], f"pid{info['pid']}")
            entries.append((entry, None))
        elif (Path(info['run_dir']) / 'result.json').exists():
            finalize(entry)
        else:
            log('REQUEUE (died with pool)', data['cand'], data['task'])
            marker.write_text(json.dumps(data, indent=1))
            os.rename(marker, lrlib.QUEUE / marker.name)
    return entries


def queued_jobs():
    jobs = []
    for path in lrlib.QUEUE.glob('*.json'):
        try:
            job = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        jobs.append((-int(job.get('priority', 0)), job.get('submitted', ''), path.name, path, job))
    jobs.sort(key=lambda x: x[:3])
    return [(p, j) for _, _, _, p, j in jobs]


def main():
    lrlib.RUNNING.mkdir(parents=True, exist_ok=True)
    lrlib.RUNS.mkdir(parents=True, exist_ok=True)
    pidfile = lrlib.BASE / 'pool.pid'
    if pidfile.exists():
        try:
            old = json.loads(pidfile.read_text())
            if proc_start(old['pid']) == old['pid_start']:
                sys.exit(f"pool already running (pid {old['pid']})")
        except (json.JSONDecodeError, KeyError):
            pass
    pidfile.write_text(json.dumps(dict(pid=os.getpid(), pid_start=proc_start(os.getpid()))))
    log('pool starting pid', os.getpid())
    running = adopt_or_requeue()
    lrlib.rebuild_leaderboard()
    stopping = False

    def handle(signum, frame):
        nonlocal stopping
        stopping = True
        log('signal', signum, '-> drain: no new launches; children keep running and are adopted on restart')
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    last_free = {}
    last_free_time = 0
    while True:
        config = load_config()
        # Reap
        still = []
        for entry, proc in running:
            done = (proc.poll() is not None) if proc is not None else not alive(entry)
            limit = config['timeouts'].get(entry['kind'], 3600)
            if not done and time.time() - entry['started'] > limit:
                log('TIMEOUT', entry['job']['cand'], entry['job']['task'], f'{limit}s')
                try:
                    os.killpg(entry['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if proc is not None:
                    proc.wait()
                finalize(entry, killed=f'timeout after {limit}s')
                continue
            if done:
                finalize(entry)
            else:
                still.append((entry, proc))
        running = still
        if STOP.exists() or stopping:
            if not running or stopping:
                log('stopping; running children:', len(running))
                break
        else:
            # Launch
            queue = queued_jobs()
            if queue:
                if time.time() - last_free_time > 10:
                    last_free, last_free_time = gpu_free_mib(), time.time()
                heavy = config.get('heavy_tasks', {})
                launched = 0
                for path, job in queue:
                    load = {g: 0 for g in config['slots']}
                    for entry, _ in running:
                        load[entry['gpu']] = load.get(entry['gpu'], 0) + heavy.get(entry['job']['task'], 1)
                    cost = heavy.get(job['task'], 1)
                    options = [g for g, n in config['slots'].items()
                               if load.get(g, 0) + cost <= int(n) and last_free.get(g, 10 ** 6) >= config['min_free_mib']]
                    if not options:
                        if cost > 1:
                            continue  # a lighter job behind it may still fit
                        break
                    gpu = min(options, key=lambda g: (load.get(g, 0) / max(int(config['slots'][g]), 1), g))
                    started = launch(path, gpu, config)
                    if started:
                        running.append(started)
                        launched += 1
                        last_free[gpu] = last_free.get(gpu, 10 ** 6) - 450
                if launched:
                    lrlib.rebuild_leaderboard()
        time.sleep(float(config.get('poll_seconds', 2)))
    try:
        pidfile.unlink()
    except FileNotFoundError:
        pass
    if STOP.exists():
        STOP.unlink()


if __name__ == '__main__':
    main()
