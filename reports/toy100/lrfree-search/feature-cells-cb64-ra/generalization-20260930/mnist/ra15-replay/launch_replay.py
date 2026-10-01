"""Root-authorized original40update CUDA replay under the shared serial lock."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def parked_owner():
    for pid, ticks, stopped in ((384331, '163720702', True), (383348, '163716372', False)):
        stat = Path(f'/proc/{pid}/stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        assert fields[19] == ticks, 'original processidentity changed'
        if stopped:
            assert fields[0] == 'T', 'original numerical supervisor no longer parked'


def verify():
    # common import sets only original environment and uses stdlib; no Torch.
    import common
    inputs = common.verify_inputs()
    cpu = json.loads((ROOT / 'CPU-CLOSED.json').read_text())
    assert cpu['status'] == 'PASS_SAVED_CHECKPOINT_SOURCE_ALIAS_PREFLIGHT'
    assert not cpu['cuda_context_initialized'] and cpu['model_calls'] == cpu['training_updates'] == 0
    assert cpu['source_freeze_sha256'] == sha(ROOT / 'SOURCE-FREEZE.json')
    assert cpu['inputs_sha256'] == sha(ROOT / 'INPUTS.json')
    return dict(status='VALID', source_freeze_sha256=sha(ROOT / 'SOURCE-FREEZE.json'),
        inputs_sha256=sha(ROOT / 'INPUTS.json'), cpu_closed_sha256=sha(ROOT / 'CPU-CLOSED.json'),
        package_sha256=inputs['variants']['RA15-partial-recovery']['package_sha256'],
        read_only_files=len(inputs['read_only_file_sha256']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    before = verify()
    if args.check_only:
        print(json.dumps(before, sort_keys=True), flush=True)
        return
    assert not (ROOT / 'LAUNCH-replay.json').exists(), 'retain every replay attempt'
    command = [sys.executable, '-u', '-B', str(ROOT / 'replay.py'), 'RA15-partial-recovery']
    log = ROOT / 'logs/RA15-partial-recovery-replay.log'
    write(ROOT / 'LAUNCH-replay.json', dict(status='WAITING_FOR_SERIAL_LOCK', start=time.time(),
        command=command, source_integrity_before=before, log=str(log), fresh_training_planned=False,
        original_updates_per_branch=10, original_branches_per_fixture=2, fixtures=2, total_updates=40))
    print(json.dumps(dict(event='waiting_for_serial_lock', log=str(log))), flush=True)
    with (OLD / 'quality/.serial-phase.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        parked_owner()
        verify()
        start = time.time()
        with log.open('x') as out:
            code = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                stdout=out, stderr=subprocess.STDOUT, close_fds=True).returncode
        after = verify()
        parked_owner()
        aggregate = ROOT / 'replay-RA15-partial-recovery.json'
        result = json.loads(aggregate.read_text()) if aggregate.exists() else None
        status = 'COMPLETE' if code == 0 and result is not None and all(v['status'] == 'PASS' for v in result.values()) else 'ERROR'
        write(ROOT / 'COMPLETION-replay.json', dict(status=status, returncode=code, command=command,
            completed=time.time(), wall_seconds=time.time() - start, source_integrity_after=after,
            result=str(aggregate), result_sha256=sha(aggregate) if aggregate.exists() else None,
            log=str(log), log_sha256=sha(log), fresh_training_updates=0,
            fresh_replay_updates_total=40 if status == 'COMPLETE' else None, signaled_processes=[]))
        print(json.dumps(dict(event='replay_complete', status=status, returncode=code,
            result=str(aggregate), log=str(log))), flush=True)
    raise SystemExit(code or (0 if status == 'COMPLETE' else 1))


if __name__ == '__main__':
    main()
