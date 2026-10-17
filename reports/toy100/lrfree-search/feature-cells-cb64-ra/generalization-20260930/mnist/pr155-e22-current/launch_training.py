"""Root-only fresh current E22 learned baselines under the shared GPU mutex."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
NAME = 'PR155-E22-cabe2084'


def write(path, value):
    with Path(path).open('x') as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def parked_owner():
    for pid, ticks, stopped in ((384331, '163720702', True), (383348, '163716372', False)):
        stat = Path(f'/proc/{pid}/stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        assert fields[19] == ticks, 'original process identity changed'
        if stopped:
            assert fields[0] == 'T', 'original numerical supervisor no longer parked'


def verify():
    import common
    inputs = common.verify_inputs()
    cpu = json.loads((ROOT / 'CPU-CLOSED.json').read_text())
    assert cpu['status'] == 'PASS_CURRENT_PR155_E22_ORIGINAL_FIXTURE_PREFLIGHT'
    assert not cpu['cuda_context_initialized'] and cpu['model_forwards'] == cpu['training_updates'] == 0
    assert cpu['source_freeze_sha256'] == common.sha(ROOT / 'SOURCE-FREEZE.json')
    assert cpu['inputs_sha256'] == common.sha(ROOT / 'INPUTS.json')
    return dict(status='VALID', source_freeze_sha256=common.sha(ROOT / 'SOURCE-FREEZE.json'),
        inputs_sha256=common.sha(ROOT / 'INPUTS.json'), cpu_closed_sha256=common.sha(ROOT / 'CPU-CLOSED.json'),
        package_sha256=inputs['variants'][NAME]['package_sha256'], read_only_files=len(inputs['read_only_file_sha256']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--problem', choices=('toy', 'mnist', 'all'), default='all')
    args = parser.parse_args()
    before = verify()
    if args.check_only:
        parked_owner()
        print(json.dumps(before, sort_keys=True), flush=True)
        return
    problems = ('toy', 'mnist') if args.problem == 'all' else (args.problem,)
    launch = ROOT / f'LAUNCH-{args.problem}.json'
    assert not launch.exists(), 'retain every launch attempt'
    write(launch, dict(status='WAITING_FOR_SERIAL_LOCK', start=time.time(), problems=problems,
        source_integrity_before=before, baseline_label=NAME, fresh_training_updates_per_fixture=2000,
        fresh_training_planned=True, shared_mutex=str(OLD / 'quality/.serial-phase.lock'), signaled_processes=[]))
    print(json.dumps(dict(event='waiting_for_serial_lock', problems=problems,
                         log_pattern=str(ROOT / f'logs/{NAME}-PROBLEM.log'))), flush=True)
    with (OLD / 'quality/.serial-phase.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        parked_owner()
        verify()
        for problem in problems:
            command = [sys.executable, '-u', '-B', str(ROOT / 'run_training.py'), '--variant', NAME, '--problem', problem]
            log = ROOT / f'logs/{NAME}-{problem}.log'
            completion = ROOT / f'COMPLETION-{problem}.json'
            assert not completion.exists() and not log.exists(), 'retain every actual run'
            start = time.time()
            with log.open('x') as out:
                code = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                    stdout=out, stderr=subprocess.STDOUT, close_fds=True).returncode
            after = verify()
            parked_owner()
            import common
            result_path = ROOT / f'training/{problem}/{NAME}/result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            status = 'COMPLETE' if code == 0 and result is not None and result['status'] == 'COMPLETE' and result['steps'] == 2000 else 'ERROR'
            write(completion, dict(status=status, returncode=code, command=command,
                completed=time.time(), wall_seconds=time.time() - start, source_integrity_after=after,
                result=str(result_path), result_sha256=common.sha(result_path) if result_path.exists() else None,
                log=str(log), log_sha256=common.sha(log), fresh_training_updates=2000 if status == 'COMPLETE' else None,
                numerical_quality_gate=result.get('original_quality_gate') if result else None, signaled_processes=[]))
            print(json.dumps(dict(event='baseline_training_complete', problem=problem, status=status,
                returncode=code, result=str(result_path), log=str(log))), flush=True)
            if status != 'COMPLETE':
                raise SystemExit(code or 1)


if __name__ == '__main__':
    main()
