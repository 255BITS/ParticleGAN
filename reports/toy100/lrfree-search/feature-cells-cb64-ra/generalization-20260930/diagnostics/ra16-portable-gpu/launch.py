"""Run the single frozen CUDA-default portability regression under the shared lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
LOCK = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/quality/.serial-phase.lock')
GPU_UUID = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
ENV = dict(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, exclusive=True):
    with Path(path).open('x' if exclusive else 'w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def parked_owner():
    for pid, ticks, stopped in ((384331, '163720702', True), (383348, '163716372', False)):
        record = Path(f'/proc/{pid}/stat').read_text()
        fields = record[record.rfind(')') + 2:].split()
        assert fields[19] == ticks, 'original parked process identity changed'
        if stopped:
            assert fields[0] == 'T', 'original numerical supervisor no longer parked'


def verify():
    inputs = json.loads((ROOT / 'INPUTS.json').read_text())
    frozen = json.loads((ROOT / 'SOURCE-FREEZE.json').read_text())
    for name, value in frozen['local_source_sha256'].items():
        assert sha(ROOT / name) == value, name
    for path, value in inputs['read_only_file_sha256'].items():
        assert sha(path) == value, path
    return inputs, dict(status='VALID', files=len(inputs['read_only_file_sha256']),
        package_sha256=inputs['package_sha256'], source_freeze_sha256=sha(ROOT / 'SOURCE-FREEZE.json'),
        inputs_sha256=sha(ROOT / 'INPUTS.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    inputs, before = verify()
    if args.check_only:
        print(json.dumps(before, sort_keys=True), flush=True)
        return
    attempt = ROOT / 'attempt-1'
    assert not attempt.exists(), 'retain every GPU regression attempt'
    attempt.mkdir()
    log = attempt / 'run.log'
    receipt = dict(status='WAITING_FOR_SHARED_SERIAL_LOCK', created=time.time(),
        test_node=inputs['test_node'], source_integrity_before=before, log=str(log),
        resources=dict(physical_gpu=0, gpu_uuid=GPU_UUID, memory_fraction=.2),
        original_fixed_seed=1234, new_quality_gate=False, fresh_full_quality_training_updates=0,
        signaled_processes=[])
    write(attempt / 'LAUNCH.json', receipt)
    print(json.dumps(dict(event='portability_waiting', log=str(log))), flush=True)
    code = 1
    try:
        with LOCK.open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            parked_owner()
            verify()
            command = [sys.executable, '-u', '-B', str(ROOT / 'worker.py'), str(attempt), str(lock.fileno())]
            receipt.update(status='RUNNING', command=command, numerical_started=time.time())
            write(attempt / 'EXECUTION.json', receipt)
            with log.open('x') as output:
                code = subprocess.run(command, cwd=ROOT, env=dict(os.environ, **ENV),
                    stdout=output, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),)).returncode
            _, after = verify()
            parked_owner()
            result_path = attempt / 'TEST-RESULT.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            passed = code == 0 and result is not None and result.get('status') == 'PASS'
            receipt.update(status='PASS' if passed else 'ERROR', returncode=code,
                completed=time.time(), numerical_wall_seconds=time.time() - receipt['numerical_started'],
                source_integrity_after=after, result=str(result_path),
                result_sha256=sha(result_path) if result_path.exists() else None,
                log_sha256=sha(log))
    except Exception as error:
        receipt.update(status='ERROR', completed=time.time(), error=repr(error), traceback=traceback.format_exc())
    write(attempt / 'COMPLETION.json', receipt)
    print(json.dumps(dict(status=receipt['status'], returncode=code, log=str(log))), flush=True)
    raise SystemExit(0 if receipt['status'] == 'PASS' else (code or 1))


if __name__ == '__main__':
    main()
