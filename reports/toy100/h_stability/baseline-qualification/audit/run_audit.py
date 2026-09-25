"""One authorized eps_net_1m audit, using existing per-host runners."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'reports/toy100/h_stability'
PYTHON = '/tmp/pr38-default-env/bin/python'
HOSTS = ['two_pole', 'mode_hold', 'unipolar', 'mid_scale_identity', 'cover_leftover',
         'trajectory', 'residual_student', 'img_stripes2', 'img_bars4', 'vector_overlap',
         'img_blobs4', 'img_intensity2', 'vector_unequal_mass', 'vector_unequal_width',
         'vector_two_broad', 'vector_anisotropic', 'vector_spiral', 'ae_gan_hold', 'unused_token_hold']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=ROOT.parent)
    args = parser.parse_args()
    out = args.output.resolve(); runtime = args.runtime.resolve()
    supervisor = runtime / 'supervisor.md'
    steering = supervisor.read_text() if supervisor.exists() else '(absent)'
    print('SUPERVISOR ' + steering.strip(), flush=True)
    if 'STOP' in steering:
        raise SystemExit('Supervisor STOP')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'supervisor-before-batch.md').write_text(steering)
    shutil.copy2(__file__, out / 'run_audit.py')
    shutil.copy2(BASE / 'eps-net-base/declaration.json', out / 'declaration.json')
    shutil.copy2(BASE / 'current-base.json', out / 'current-base.json')
    events = {}; commands = []
    retained = json.loads((BASE / 'eps-net-base/independent-cold/eps_net_1m/status.json').read_text())
    for row in retained['stages']:
        task = row['gate']
        local = BASE / 'eps-net-base/independent-cold/eps_net_1m' / task
        row = dict(row, candidate='regression', recipe='eps_net_1m', evidence='retained independent run; not rerun',
                   artifact=str(local / 'index.json'))
        events[task] = row

    def publish():
        lines = ['# eps_net_1m baseline qualification audit', '',
                 'Diagnostic only; no promotion. Fixed G/D/prior rates .001125/.0015/.00225; epsilon .001/.001/1e-8.',
                 'Ring and two_pole use retained independent evidence. Remaining hosts run individually with the complete selected policy.', '',
                 '| Host | Status | Terminal metrics | Evidence |', '|---|---|---|---|']
        for task in HOSTS:
            row = events.get(task)
            if row is None:
                lines.append(f'| {task} | PENDING | | |'); continue
            live = row['metrics'].get('live', {})
            metrics = ', '.join(f'{k}={v:.9g}' if isinstance(v, (int, float)) else f'{k}={v}' for k, v in live.items())
            lines.append(f"| {task} | {row['status']} | {metrics} | {row.get('evidence', 'executed this audit')} |")
        hold = events.get('own_state_diagnostic_1200')
        lines += ['', 'Own-state hold: ' + (json.dumps({k: hold['metrics'].get(k) for k in ('status', 'window', 'final', 'full_budget')}) if hold else 'PENDING'), '',
                  'All executed audit/regrade rows use candidate=regression in tests.jsonl. Native100 SKIPPED; known older failure.',
                  'New proposals executed: 0. Unit tests: pending.', '',
                  f'Artifacts: `{out}`. Commands: `{out / "commands.json"}`.',
                  f'Logs: `tail -F {out}/*.log {runtime / "tests.jsonl"}`', '',
                  'Replay into a new output directory:', '```bash',
                  f'{PYTHON} {Path(__file__).relative_to(ROOT)} --output reports/toy100/h_stability/baseline-qualification/replay --runtime {runtime}', '```']
        report = '\n'.join(lines) + '\n'
        (runtime / 'result.md').write_text(report)
        (BASE / 'RESULTS.md').write_text(report)
        dump(out / 'audit-results.json', events)

    def execute(task):
        start = time.perf_counter(); destination = out / task
        if task == 'own_state_diagnostic_1200':
            command = [PYTHON, '-u', str(BASE / 'selected_base_probe.py'), '--output', str(destination), '--steps', '1200']
        else:
            command = [PYTHON, '-u', str(BASE / 'adam_response_cold.py'), '--declaration', str(out / 'declaration.json'),
                       '--output', str(destination), '--ledger', str(out / (task + '.raw.jsonl')), '--workers', '1', '--tasks', task]
        log_path = out / (task + '.log')
        print(json.dumps(dict(event='START', gate=task, log=str(log_path))), flush=True)
        with log_path.open('w', buffering=1) as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if task == 'own_state_diagnostic_1200' and (destination / 'summary.json').exists():
            metrics = json.loads((destination / 'summary.json').read_text())
            row = dict(status=metrics['status'], metrics=metrics, artifact=str(destination / 'summary.json'))
        elif task != 'own_state_diagnostic_1200' and (out / (task + '.raw.jsonl')).exists():
            row = json.loads((out / (task + '.raw.jsonl')).read_text().splitlines()[-1])
        else:
            row = dict(status='ERROR', metrics={}, artifact=str(log_path), error=f'runner exit {completed.returncode}')
        row.update(candidate='regression', recipe='eps_net_1m', gate=task,
                   seconds=time.perf_counter()-start, diagnostic_only=True)
        if completed.returncode:
            row.update(status='ERROR', error=f'runner exit {completed.returncode}; see log')
        return row, shlex.join(command)

    publish()
    jobs = iter(['own_state_diagnostic_1200'] + HOSTS[2:])
    started = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(execute, next(jobs)): None for _ in range(3)}
        halted = False
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                row, command = future.result()
                events[row['gate']] = row; commands.append(command)
                with (runtime / 'tests.jsonl').open('a') as ledger:
                    ledger.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                dump(out / 'commands.json', commands); publish()
                print(json.dumps(dict(event='DONE', gate=row['gate'], status=row['status'], seconds=round(row['seconds'], 3),
                                      live=row['metrics'].get('live'), final=row['metrics'].get('final'))), flush=True)
                halted |= row['status'] == 'ERROR'
                del pending[future]
            if supervisor.exists() and 'STOP' in supervisor.read_text():
                halted = True
            while not halted and len(pending) < 3:
                task = next(jobs, None)
                if task is None:
                    break
                pending[pool.submit(execute, task)] = None
    dump(out / 'batch-summary.json', dict(started=started, finished=datetime.now(timezone.utc).isoformat(),
         executed=len(commands), halted=halted, workers=3, threads_per_worker=1, candidates=0))
    print(json.dumps(dict(event='AUDIT_COMPLETE', executed=len(commands), halted=halted)), flush=True)


if __name__ == '__main__':
    main()
