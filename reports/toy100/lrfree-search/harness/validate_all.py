#!/usr/bin/env python
"""Compare every finished run that has archived PR155 new-init evidence; print a markdown table."""
from pathlib import Path
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lrlib  # noqa: E402

R = Path('/ml2/hypergan/ParticleGAN-k3p-continuous-search/reports/toy100/deterministic-init-retest')
PY = lrlib.PYTHON


def evidence(cand, task):
    c = cand.lower()
    if task == 'mode_hold':
        name = c.replace('api-', 'api-') + '-new-init'
        d = R / 'evidence' / name
    elif task.startswith('img_'):
        d = R / 'followup-evidence' / f"{c}-img-{task[4:]}-new-init"
    elif task == 'vector_unequal_mass' and c == 'api-dv12':
        d = R / 'followup-evidence' / 'api-dv12-unequal-mass-new-init'
    elif task == 'ring_shift' and c in ('api-dv2', 'api-dv3'):
        d = R / 'supplementary-evidence' / f'{c}-single-shift-new-init'
    else:
        return None
    for name in ('metrics.jsonl.gz', 'metrics.jsonl'):
        if (d / name).exists():
            return d, d / name
    return None


def compare(run, metrics, rates=None):
    cmd = [PY, str(lrlib.HARNESS / 'compare.py'), str(run), str(metrics)]
    if rates and rates.exists():
        cmd += ['--rates', str(rates)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def summary(result):
    if result.get('task') in lrlib.RING_TASKS:
        return lrlib.cell(dict(result, task=result['task']))
    return lrlib.cell(result)


def main():
    lines = ['| candidate | task | evidence result | harness result | observations exact | LR steps exact |',
             '|---|---|---|---|---|---|']
    for run in sorted(lrlib.RUNS.glob('*/*/result.json')):
        cand, task = run.parent.parent.name, run.parent.name
        found = evidence(cand, task)
        if not found:
            continue
        d, metrics = found
        mine = json.loads(run.read_text())
        if mine.get('status') not in ('PASS', 'FAIL'):
            continue
        res = compare(run.parent, metrics, next((p for p in (d / 'learning-rates.jsonl.gz',) if p.exists()), None))
        ev = json.loads((d / 'result.json').read_text())
        if task == 'mode_hold':
            conv = ev['metrics']['verdict']['convergence']
            evs = f"{ev['status']} {conv['passing_observations']}/24 @{conv['first_pass_step']} final {ev['metrics']['final']['modes']}/8 hq {ev['metrics']['final']['hq']:.4f}"
        elif task.startswith('img_'):
            conv = ev['convergence']
            evs = f"{ev['status']} {conv['passing_observations']}/24 @{conv['first_pass_step']}"
        elif task == 'ring_shift':
            s = json.loads((d / 'summary.json').read_text())['segments']
            evs = ' / '.join(f"arr {x['first_arrival']} ret {x['passing_since_arrival']}/{x['checks_since_arrival']}" for x in s)
        else:
            conv = ev['verdict']['convergence']
            f = ev['final']
            evs = f"{ev['status']} {conv['passing_observations']}/24 cov {f['component_covariance_error']:.4f} mmr {f['min_mass_ratio']:.4f}"
        if task == 'mode_hold':
            ms = f"{summary(mine)} final {mine['final']['modes']}/8 hq {mine['final']['hq']:.4f}"
        elif task == 'vector_unequal_mass':
            ms = f"{summary(mine)} cov {mine['final']['component_covariance_error']:.4f} mmr {mine['final']['min_mass_ratio']:.4f}"
        else:
            ms = summary(mine)
        lr = f"{res.get('lr_steps_exact')}/{res.get('lr_steps_exact', 0) + res.get('lr_steps_diff', 0)}" if 'lr_steps_exact' in res else '-'
        lines.append(f"| {cand} | {task} | {evs} | {ms} | {res['values_exact']}/{res['values_compared']} "
                     f"({'bitwise' if res['bitwise'] else 'DIFF max ' + str(res['max_abs_diff'])}) | {lr} |")
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
