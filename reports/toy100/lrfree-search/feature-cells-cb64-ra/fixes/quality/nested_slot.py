"""Serial candidate phase inside the already held, root-owned RA4 GPU slot."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = 384331
SUPERVISOR_TICKS = '163720702'
OUTER = 383348
OUTER_TICKS = '163716372'


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return fields[0],fields[19]


def check_owned_slot():
    assert identity(SUPERVISOR) == ('T',SUPERVISOR_TICKS), 'RA4 supervisor is not parked or PID reused'
    assert identity(OUTER)[1] == OUTER_TICKS, 'outer GPU slot identity changed'
    assert str(ROOT/'gpu_slot.py').encode() in Path(f'/proc/{OUTER}/cmdline').read_bytes()
    assert any(p.resolve() == ROOT/'.gpu-slot.lock' for p in Path(f'/proc/{OUTER}/fd').iterdir())
    with (ROOT/'.gpu-slot.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        else:
            fcntl.flock(lock,fcntl.LOCK_UN)
            raise RuntimeError('original outer GPU lock is not held')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane',type=Path,required=True)
    parser.add_argument('--through',type=int,required=True)
    args = parser.parse_args()
    lane = args.lane.resolve()
    assert lane.is_relative_to(ROOT) and (lane/'source-freeze.json').exists()
    output = lane/f'owned-slot-through-{args.through}.json'
    assert not output.exists(), 'candidate phase slot receipt already exists'
    spec = importlib.util.spec_from_file_location('original_slot_helpers',ROOT/'gpu_slot.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    with (ROOT/'quality/.serial-phase.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_owned_slot()
        last = 0.
        while waiting := helpers.owned_numerical_processes():
            check_owned_slot()
            if time.monotonic()-last>30:
                print(json.dumps(dict(event='waiting_for_untouched_original_child',pids=waiting)),flush=True)
                last = time.monotonic()
            time.sleep(.25)
        check_owned_slot()
        started = datetime.now(timezone.utc).isoformat()
        command = ['/tmp/pr38-default-env/bin/python','-u','-B',str(lane/'launch.py'),'--through',str(args.through)]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
        print(json.dumps(dict(event='owned_serial_candidate_start',command=command,utc=started)),flush=True)
        result = subprocess.run(command,cwd=ROOT,env=env)
        check_owned_slot()
        assert not helpers.owned_numerical_processes(), 'owned numerical child remains active'
        output.write_text(json.dumps(dict(command=command,started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,
            original_outer_pid=OUTER,original_outer_startticks=OUTER_TICKS,
            original_supervisor_pid=SUPERVISOR,original_supervisor_startticks=SUPERVISOR_TICKS,
            original_supervisor_still_parked=True,owned_numerical_parallelism=1,
            prior_active_child_untouched=True),indent=2)+'\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
