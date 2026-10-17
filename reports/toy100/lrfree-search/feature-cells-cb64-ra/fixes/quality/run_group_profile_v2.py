"""Root-owned serial microprofile after the active quality phase completes."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

from nested_slot import ROOT, check_owned_slot

AREA = ROOT / 'quality' / 'group-profile-phase-v2'
OWNER = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra4-quality/anchor-profile'
OUTPUT = ROOT / 'integration/review/group-count-gpu'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(frozen):
    for name, expected in frozen['source_sha256'].items():
        assert sha(Path(name)) == expected, f'frozen input changed: {name}'


def main():
    frozen = json.loads((AREA / 'READY.json').read_text())
    verify(frozen)
    assert frozen['status'] == 'FROZEN_ROOT_GPU_PENDING'
    owner = json.loads((OWNER / 'READY.json').read_text())
    expected = owner['gpu_command']
    assert expected[-1] == str(OUTPUT / 'result.json')
    assert not OUTPUT.exists(), 'profile phase already started; retain its evidence'
    with (ROOT / 'quality/.serial-phase.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_owned_slot()
        # Import the original read-only job-plan process scanner.
        import importlib.util
        spec = importlib.util.spec_from_file_location('profile_slot_helpers', ROOT / 'gpu_slot.py')
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        assert not helpers.owned_numerical_processes(), 'an owned numerical child is active'
        gpu = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
            '--format=csv,noheader'], check=True, capture_output=True, text=True).stdout.strip()
        assert gpu == 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47', 'physical GPU0 changed'
        verify(frozen)
        OUTPUT.mkdir(parents=True)
        started = datetime.now(timezone.utc).isoformat()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with (OUTPUT / 'profile.log').open('w') as log:
            child = subprocess.Popen(expected, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                pass_fds=(lock.fileno(),))
            fields = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()
            (OUTPUT / 'LAUNCH.json').write_text(json.dumps(dict(command=expected,
                pid=child.pid, startticks=fields[19], started_utc=started,
                gpu_uuid=gpu, phase_ready_sha256=sha(AREA / 'READY.json'),
                source_integrity='VALID', numerical_parallelism=1), indent=2) + '\n')
            returncode = child.wait()
        verify(frozen)
        check_owned_slot()
        assert not helpers.owned_numerical_processes()
        result_path = OUTPUT / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        status = 'PASS' if returncode == 0 and result and result['status'] == 'PASS' else 'ERROR'
        (OUTPUT / 'PHASE-RESULT.json').write_text(json.dumps(dict(status=status,
            returncode=returncode, started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            phase_ready_sha256=sha(AREA / 'READY.json'), gpu_uuid=gpu,
            result_sha256=sha(result_path) if result_path.exists() else None,
            original_supervisor_still_parked=True, numerical_parallelism=1,
            quality_verdict=None, source_integrity='VALID'), indent=2) + '\n')
    raise SystemExit(returncode or (0 if status == 'PASS' else 1))


if __name__ == '__main__':
    main()
