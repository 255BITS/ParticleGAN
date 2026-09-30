"""Reserve one diagnostic GPU slot after an owned queue child finishes."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / 'feature-cells-cuda-retest-20260929'
LAUNCHER = str(BASELINE / 'launch.py')


def emit(event, **values):
    row = dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **values)
    with (ROOT / 'gpu-slots.jsonl').open('a') as out:
        out.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def identity(pid):
    stat = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return stat[0], stat[19]


def find_launcher(launcher):
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode().split('\0')
            cwd = (entry / 'cwd').resolve(strict=True)
            launcher_args = [arg for arg in argv if arg.endswith('launch.py')]
            if any((Path(arg) if Path(arg).is_absolute() else cwd / arg).resolve()
                   == launcher for arg in launcher_args):
                matches.append(int(entry.name))
        except (OSError, UnicodeError):
            pass
    if len(matches) > 1:
        raise RuntimeError('More than one baseline launcher exists.')
    return matches[0] if matches else None


def active_children(pid):
    path = Path(f'/proc/{pid}/task/{pid}/children')
    active = []
    for child in path.read_text().split():
        try:
            if identity(int(child))[0] not in ('Z', 'X'):
                active.append(int(child))
        except OSError:
            pass
    return active


def owned_numerical_processes():
    """Include an adopted GPU job whose original supervisor exited."""
    scripts = set()
    plans = [BASELINE / 'jobs.json', *ROOT.rglob('jobs.json')]
    for plan in plans:
        if not plan.exists():
            continue
        for job in json.loads(plan.read_text()):
            script = next((arg for arg in job['command'] if arg.endswith('.py')), None)
            if script:
                scripts.add(Path(script).resolve())
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode().split('\0')
            script = next((arg for arg in argv if arg.endswith('.py')), None)
            cwd = (entry / 'cwd').resolve(strict=True)
            if script and (Path(script) if Path(script).is_absolute() else cwd / script).resolve() in scripts:
                if identity(int(entry.name))[0] not in ('Z', 'X'):
                    active.append(int(entry.name))
        except (OSError, UnicodeError):
            pass
    return active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reason', required=True)
    parser.add_argument('--queue', type=Path, default=BASELINE,
                        help='Owned frozen queue whose launcher must be parked.')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A diagnostic command is required.')
    with (ROOT / '.gpu-slot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = args.queue.resolve()
        launcher = queue / 'launch.py'
        owned_launchers = {BASELINE / 'launch.py', *ROOT.rglob('launch.py')}
        other_active = [str(path.parent) for path in owned_launchers
                        if path != launcher and find_launcher(path) is not None]
        if other_active:
            raise RuntimeError('Another owned GPU queue is active; select it with --queue: '
                               + ', '.join(other_active))
        pid = find_launcher(launcher)
        stamp = None
        try:
            if pid is not None:
                stamp = identity(pid)[1]
                os.kill(pid, signal.SIGSTOP)
                emit('baseline_launcher_parked', pid=pid, start_ticks=stamp, queue=str(queue),
                     reason=args.reason, active_children=active_children(pid))
                last_notice = 0.
                # A restarted session can leave the numerical process adopted
                # by systemd. It still owns GPU0 even if it is no longer a
                # direct child of the recovered supervisor.
                while (waiting := sorted(set(active_children(pid) + owned_numerical_processes()))):
                    if time.monotonic() - last_notice > 30:
                        emit('waiting_for_original_child', pid=pid, children=waiting)
                        last_notice = time.monotonic()
                    time.sleep(.25)
            else:
                recovered = queue / 'RECOVERY.json'
                original = Path(json.loads(recovered.read_text())['original_validation']) if recovered.exists() else queue
                rows = (original / 'run.log').read_text().splitlines()
                if not rows or json.loads(rows[-1])['event'] != 'queue_complete':
                    raise RuntimeError('No launcher and no complete baseline receipt.')
                if owned_numerical_processes():
                    raise RuntimeError('An owned numerical GPU job is still active.')
            emit('diagnostic_gpu_slot_start', command=command, physical_gpu=0, reason=args.reason)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       NUMEXPR_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            result = subprocess.run(command, cwd=ROOT, env=env)
            emit('diagnostic_gpu_slot_complete', returncode=result.returncode, command=command)
        finally:
            if pid is not None and stamp is not None:
                if identity(pid)[1] != stamp:
                    raise RuntimeError('Launcher process identity changed.')
                os.kill(pid, signal.SIGCONT)
                emit('baseline_launcher_resumed', pid=pid)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
