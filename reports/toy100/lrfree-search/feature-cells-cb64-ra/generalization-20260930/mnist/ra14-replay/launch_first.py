"""Launch root-authorized original learned training/replay inside owned slot."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
NAME = 'RA14-replay'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=('toy', 'mnist', 'replay'), required=True)
    args = parser.parse_args()
    phase = args.problem
    assert not (ROOT / f'LAUNCH-{phase}.json').exists(), 'owned attempt already exists'
    cpu = json.loads((ROOT / 'CPU-CLOSED.json').read_text())
    assert cpu['status'] == 'PASS_ORIGINAL_FIXTURE_CURRENT_API_PREFLIGHT'
    for name, expected in cpu['file_sha256'].items():
        assert sha(ROOT / name) == expected, name
    spec = importlib.util.spec_from_file_location('original_owned_slot', ORIGINAL / 'gpu_slot.py')
    slot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slot)
    identities = {str(pid): slot.identity(pid) for pid in (384331, 383348)}
    assert identities['384331'] == ('T', '163720702'), identities
    assert identities['383348'][1] == '163716372', identities
    assert not slot.owned_numerical_processes()
    with (ORIGINAL / '.gpu-slot.lock').open('r') as outer:
        try:
            fcntl.flock(outer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(outer, fcntl.LOCK_UN)
            raise RuntimeError('Existing owned outer GPU slot is not held')
    with (ORIGINAL / 'quality/.serial-phase.lock').open('r') as serial:
        fcntl.flock(serial, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze, inputs = sha(ROOT / 'SOURCE-FREEZE.json'), sha(ROOT / 'INPUTS.json')
        command = ['/tmp/pr38-default-env/bin/python', '-u', '-B', str(ROOT / ('replay.py' if phase == 'replay' else 'run_training.py'))]
        command += [NAME] if phase == 'replay' else ['--variant', NAME, '--problem', phase]
        log_path = ROOT / 'logs' / f'{NAME}-{phase}.log'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        start, started = time.perf_counter(), utc()
        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            launch = dict(status='RUNNING', started_utc=started, pid=process.pid,
                          startticks=slot.identity(process.pid)[1], command=command,
                          log=str(log_path), source_freeze_sha256=freeze, inputs_sha256=inputs,
                          root_authorization='Root review/freeze and GO required for this exact actual-candidate phase',
                          known_original_processes=identities, original_outer_slot_held=True,
                          original_owned_numerical_processes_at_launch=[], signaled_processes=[])
            write(ROOT / f'LAUNCH-{phase}.json', launch)
            print(json.dumps(dict(event='candidate_launch', **launch)), flush=True)
            code = process.wait()
        assert sha(ROOT / 'SOURCE-FREEZE.json') == freeze
        assert sha(ROOT / 'INPUTS.json') == inputs
        assert slot.identity(384331) == ('T', '163720702')
        assert slot.identity(383348)[1] == '163716372'
        path = ROOT / f'replay-{NAME}.json' if phase == 'replay' else ROOT / 'training' / phase / NAME / 'result.json'
        completion = dict(status='COMPLETE' if code == 0 and path.exists() else 'ERROR',
                          started_utc=started, completed_utc=utc(), returncode=code,
                          wall_seconds=time.perf_counter() - start, command=command,
                          result=str(path), result_sha256=sha(path) if path.exists() else None,
                          log=str(log_path), log_sha256=sha(log_path),
                          source_freeze_sha256=freeze, inputs_sha256=inputs,
                          original_supervisor_still_parked=True, signaled_processes=[])
        write(ROOT / f'COMPLETION-{phase}.json', completion)
        print(json.dumps(dict(event='candidate_complete', **completion)), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
