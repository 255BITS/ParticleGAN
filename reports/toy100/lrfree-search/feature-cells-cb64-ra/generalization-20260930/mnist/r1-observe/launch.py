"""Run the root-authorized 200-update observation inside the owned outer slot."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2)+'\n')

def main():
    assert not (ROOT/'LAUNCH.json').exists(), 'attempt already launched'
    freeze = json.loads((ROOT/'SOURCE-FREEZE.json').read_text())
    for name, expected in freeze['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    inputs = json.loads((ROOT/'INPUTS.json').read_text())
    for path, expected in inputs['read_only_sha256'].items():
        assert sha(path) == expected, path
    assert json.loads((ROOT/'PREFLIGHT.json').read_text())['status'] == 'PASS_ZERO_UPDATE_SOURCE_CONTRACT'
    spec = importlib.util.spec_from_file_location('original_owned_slot', ORIGINAL/'gpu_slot.py')
    slot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slot)
    identities = {str(pid): slot.identity(pid) for pid in (384331, 383348)}
    assert identities['384331'] == ('T', '163720702'), identities
    assert identities['383348'][1] == '163716372', identities
    assert not slot.owned_numerical_processes()
    with (ORIGINAL/'.gpu-slot.lock').open('r') as outer:
        try:
            fcntl.flock(outer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(outer, fcntl.LOCK_UN)
            raise RuntimeError('Existing owned outer GPU slot is not held')
    with (ORIGINAL/'quality/.serial-phase.lock').open('r') as serial:
        fcntl.flock(serial, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_sha, inputs_sha = sha(ROOT/'SOURCE-FREEZE.json'), sha(ROOT/'INPUTS.json')
        command = ['/tmp/pr38-default-env/bin/python', '-u', '-B', str(ROOT/'run_observe.py')]
        logfile = ROOT/'observe.log'
        start, started = time.perf_counter(), utc()
        with logfile.open('x') as output:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ,
                PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'), stdout=output, stderr=subprocess.STDOUT)
            launch = dict(status='RUNNING', started_utc=started, pid=child.pid,
                startticks=slot.identity(child.pid)[1], command=command, log=str(logfile),
                source_freeze_sha256=source_sha, inputs_sha256=inputs_sha,
                authorization='Root explicitly authorized frozen RA12 Toy750->835 and MNIST100->215 observation-only replay',
                known_original_processes=identities, original_outer_slot_held=True,
                original_owned_numerical_processes_at_launch=[], signaled_processes=[])
            write(ROOT/'LAUNCH.json', launch)
            print(json.dumps(dict(event='observation_launch', **launch)), flush=True)
            code = child.wait()
        assert sha(ROOT/'SOURCE-FREEZE.json') == source_sha
        assert sha(ROOT/'INPUTS.json') == inputs_sha
        for name, expected in freeze['source_sha256'].items():
            assert sha(ROOT/name) == expected, name
        for path, expected in inputs['read_only_sha256'].items():
            assert sha(path) == expected, path
        assert slot.identity(384331) == ('T', '163720702')
        assert slot.identity(383348)[1] == '163716372'
        result = ROOT/'result.json'
        completion = dict(status='COMPLETE' if code == 0 and result.exists() else 'ERROR',
            started_utc=started, completed_utc=utc(), returncode=code, wall_seconds=time.perf_counter()-start,
            command=command, result=str(result), result_sha256=sha(result) if result.exists() else None,
            log=str(logfile), log_sha256=sha(logfile), source_freeze_sha256=source_sha,
            inputs_sha256=inputs_sha, original_supervisor_still_parked=True, signaled_processes=[])
        write(ROOT/'COMPLETION.json', completion)
        print(json.dumps(dict(event='observation_complete', **completion)), flush=True)
    raise SystemExit(code)

if __name__ == '__main__':
    main()
