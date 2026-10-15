"""Read already-recorded observations; never import or execute a trainer.

The derived gate summary is diagnostic, not a replacement for Forge certificates.
Run repeatedly as files arrive. Partial final JSONL lines are ignored.
"""
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get('PARTICLEGAN_COMBINATION_ARTIFACT_ROOT', Path(__file__).resolve().parent))
CAMPAIGN = ROOT / 'queue' / 'bcap-selective-history-pole-ring-v2'
CAMPAIGNS = [CAMPAIGN, ROOT/'queue'/'bcap-selective-history-repair-screen-v1',
             ROOT/'queue'/'bcap-critic-rate-final-screen-v1']
ARMS = {
    'recipe-prior-refactor-bcap-pure-v1': 'B',
    'bcap-selective-history-transport-v1': 'T',
    'bcap-selective-history-whole-v1': 'W',
    'bcap-selective-history-adversarial-v1': 'S',
    'bcap-transport-penalty2-v1': 'TP',
    'bcap-selective-history-halfjoint-v1': 'SH',
    'bcap-transport-halfcritic-v1': 'TD',
    'bcap-selective-history-halfjoint-halfcritic-v1': 'SHD',
}
ORDER = ['B','T','W','S','TP','SH','TD','SHD']


def attempt_directories():
    for campaign in CAMPAIGNS:
        if campaign.exists():
            yield from sorted(campaign.iterdir())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failures(metrics, thresholds):
    bad = []
    for name, op, bound in thresholds:
        v = metrics.get(name)
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            bad.append(name)
        elif not {'<=': v <= bound, '>=': v >= bound, '==': v == bound}[op]:
            bad.append(name)
    return bad


def main():
    rows, overview = [], []
    for directory in attempt_directories():
        request_path = directory / 'request.json'
        if not directory.is_dir() or not request_path.exists():
            continue
        request = json.loads(request_path.read_text())
        candidate = request['request']['candidate']['id']
        if candidate not in ARMS:
            continue
        arm, task_id = ARMS[candidate], request['job']['task_id']
        task = request['request']['tasks'][task_id]
        thresholds = task['evaluation']['thresholds']
        events = []
        log = directory / 'run.log'
        for line in log.read_text().splitlines() if log.exists() else []:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('event') == 'observation' and event.get('task') == task_id:
                events.append(event)
        # A log can contain a repeated print; identical steps must agree.
        unique = {}
        for event in events:
            step = event['step']
            if step in unique and event['metrics'] != unique[step]['metrics']:
                raise ValueError(f'disagreeing observations at {directory.name}:{step}')
            unique[step] = event
        events = [unique[step] for step in sorted(unique)]
        failcounts, suffix, passes = Counter(), 0, 0
        details = []
        for event in events:
            bad = failures(event['metrics'], thresholds)
            passed = not bad
            passes += passed
            suffix = suffix + 1 if passed else 0
            failcounts.update(bad)
            row = dict(arm=arm, task=task_id, attempt=directory.name,
                       step=event['step'], passed=passed, failed_bounds=bad,
                       metrics=event['metrics'])
            details.append(row)
            rows.append(row)
        complete = (len(events) == task['evaluation'].get('observations', 24)
                    and bool(events) and events[-1]['step'] == task['execution']['steps'])
        result_path = directory / 'raw-result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        overview.append(dict(
            arm=arm, task=task_id, attempt=directory.name, candidate=candidate,
            campaign=directory.parent.name,
            request_sha256=digest(request_path), raw_result_path=str(result_path) if result else None,
            raw_result_sha256=digest(result_path) if result else None,
            observations=len(events), completed_full_schedule=complete,
            passing_observations=passes, terminal_suffix=suffix,
            diagnostic_gate=('PASS' if suffix >= task['evaluation'].get('minimum_stable_checks', 5)
                             else 'FAIL') if complete else 'RUNNING_OR_INCOMPLETE',
            failed_bound_counts=dict(failcounts),
            terminal_rows=details[-5:],
            first_passing_step=next((r['step'] for r in details if r['passed']), None),
            last_failing_step=next((r['step'] for r in reversed(details) if not r['passed']), None),
            declared_thresholds=thresholds,
            guards=result['evidence'].get('guards') if result else None,
            cost=result.get('cost') if result else None,
        ))
    overview.sort(key=lambda x:(x['task'], ORDER.index(x['arm'])))
    report = dict(schema=1, kind='read_only_saved_curve_analysis',
                  script_sha256=digest(Path(__file__)),
                  training_updates=0, new_model_evaluations=0,
                  note='Derived numerical gates require full schedule; Forge certification remains authoritative.',
                  attempts=overview)
    (ROOT / 'curve-analysis.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    (ROOT / 'curve-observations.json').write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')
    with (ROOT / 'curve-observations.csv').open('w') as handle:
        names = ['arm','task','attempt','step','passed','failed_bounds','mean_abs','grad_med',
                 'modes','mass_tv','hq','component_covariance_error','component_min_eigen_ratio',
                 'component_core_covariance_error','component_core_min_eigen_ratio']
        writer=csv.DictWriter(handle, fieldnames=names); writer.writeheader()
        for row in rows:
            values={k:row[k] for k in names if k in row}
            values['failed_bounds']=';'.join(row['failed_bounds'])
            values.update({k:row['metrics'][k] for k in names if k in row['metrics']})
            writer.writerow(values)
    for row in overview:
        print(row['arm'], row['task'], row['diagnostic_gate'],
              f"{row['passing_observations']}/{row['observations']}",
              'suffix', row['terminal_suffix'], 'failed', row['failed_bound_counts'])


if __name__ == '__main__':
    main()
