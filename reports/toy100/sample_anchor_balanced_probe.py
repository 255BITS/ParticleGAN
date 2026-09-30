"""Cold trajectory, ring, and fidelity receipt for balanced-mass anchors.

Fail-fast: a trajectory miss or a ring that is not eight modes at HQ>=.9
stops before any hold. Fidelity is measured on the realized 12 particles
against the pinned pre-start receipt, not on the free-output target.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.sample_anchor_balanced_candidate import METHOD, sample_anchor_balanced_candidate
from reports.toy100.sample_anchor_mass_balance import SIGMA_OUT

PINNED_COUNTS = [1, 1, 1, 1, 1, 1, 1, 5]
PINNED_TV = 0.2917
PINNED_SPREAD = 0.029
PINNED_TRAJ_MSE = 0.000942662


def fidelity(points, means):
    y = points.detach().double()
    which = torch.cdist(y, means.double()).argmin(1)
    counts = torch.bincount(which, minlength=len(means))
    masses = counts.double() / len(y)
    tv = float((masses - 1 / len(means)).abs().sum() / 2)
    spreads = []
    for group in range(len(means)):
        own = y[which == group]
        if len(own) == 0:
            spreads.append(None)
            continue
        var = (own - own.mean(0)).square().mean() + SIGMA_OUT ** 2
        spreads.append(float(var.sqrt()))
    present = [value for value in spreads if value is not None]
    return dict(nearest_mode_counts=counts.tolist(), mode_mass_tv=tv,
                per_mode_emitted_spread=spreads,
                mean_emitted_spread=sum(present) / len(present) if present else None,
                pinned_counts=PINNED_COUNTS, pinned_tv=PINNED_TV,
                pinned_spread=PINNED_SPREAD,
                tv_delta=tv - PINNED_TV,
                spread_delta=(sum(present) / len(present) - PINNED_SPREAD) if present else None)


def realized_points(recorder):
    local = recorder._local
    generator = getattr(local['generator'], 'model', local['generator'])
    with torch.no_grad():
        return generator(local['prior'].z).detach()


def run_task(task, steps, recipe, noise, config, output):
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy
    spec = next(job['spec'] for job in plan() if job['spec']['name'] == task)
    with sample_anchor_balanced_candidate(task=task, correction=True) as (recorder, _):
        def progress(calls, outer):
            if outer % 100 == 0:
                print(json.dumps(dict(event='COLD_PROGRESS', task=task, update=outer)), flush=True)
        recorder.accounting = progress
        result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    verdict = test_verdict(spec, result)
    row = dict(task=task, verdict=verdict, live=result.get('live'), seconds=result.get('seconds'),
               steps=steps, method=METHOD)
    if task == 'mode_hold':
        from benchmarks.locked_shared import mode_hold
        row['fidelity'] = fidelity(realized_points(recorder), mode_hold.ring_means())
        selected = [item['selected'] for item in recorder.corrections]
        row['selected_counts'] = {name: selected.count(name) for name in sorted(set(selected))}
        row['corrections'] = len(selected)
    (output / f'{task}.json').write_text(json.dumps(dict(
        method=METHOD, result_live=result.get('live'), seconds=result.get('seconds'),
        verdict=verdict, fidelity=row.get('fidelity'),
        selected_counts=row.get('selected_counts')), allow_nan=False) + '\n')
    print(json.dumps(dict(event='STAGE_DONE', **{k: row[k] for k in row if k != 'live'},
                          live_modes=(result.get('live') or {}).get('modes'),
                          live_hq=(result.get('live') or {}).get('hq'))), flush=True)
    return row, recorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name=METHOD, lr_floor=1., lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap', None)
    config.pop('network_lr_floor', None)
    recipe, noise, _ = declared_recipe(config)
    print(json.dumps(dict(event='DECLARED', method=METHOD, lane='C1',
                          pinned_tv=PINNED_TV, pinned_spread=PINNED_SPREAD,
                          pinned_traj_mse=PINNED_TRAJ_MSE)), flush=True)
    stages = []
    traj, _ = run_task('trajectory', 400, recipe, noise, config, args.output)
    stages.append(traj)
    passed_traj = bool(traj['verdict']['passed'])
    ring = None
    if passed_traj:
        ring, _ = run_task('mode_hold', 1200, recipe, noise, config, args.output)
        stages.append(ring)
    ring_ok = bool(ring and ring['verdict']['passed'] and (ring['live'] or {}).get('modes') == 8
                   and (ring['live'] or {}).get('hq', 0) >= .9)
    fidelity_ok = bool(ring and ring.get('fidelity') and ring['fidelity']['mode_mass_tv'] < PINNED_TV - 1e-4)
    status = 'PASS' if passed_traj and ring_ok and fidelity_ok else 'FAIL'
    summary = dict(method=METHOD, lane='C1', host='neural', status=status,
                   trajectory_passed=passed_traj, ring_eight=ring_ok,
                   fidelity_improved=fidelity_ok, stages=stages,
                   pinned=dict(traj_mse=PINNED_TRAJ_MSE, counts=PINNED_COUNTS,
                               tv=PINNED_TV, spread=PINNED_SPREAD))
    (args.output / 'summary.json').write_text(json.dumps(summary, allow_nan=False) + '\n')
    print(json.dumps(dict(event='DONE', status=status, trajectory_passed=passed_traj,
                          ring_eight=ring_ok, fidelity_improved=fidelity_ok,
                          fidelity=None if not ring else ring.get('fidelity'))), flush=True)


if __name__ == '__main__':
    main()
