"""Completed-only recurrence and optional clock-rate interventions, evaluation only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as handoff
from experiments import memory_core_scout as core


def recurrence(paths):
    x = np.asarray(paths, dtype=np.float64)[:, -512:]
    scale = ((x-x.mean(1, keepdims=True))**2).sum(2).mean(1).clip(1e-12)
    lags = np.arange(8, 251)
    errors = np.stack([((x[:, lag:]-x[:, :-lag])**2).sum(2).mean(1)/scale
                       for lag in lags], axis=1)
    best = lags[errors.argmin(1)]
    return {'median_best_lag': float(np.median(best)),
            'fraction_best_lag_199_203': float(((best >= 199) & (best <= 203)).mean()),
            'median_error_lag201': float(np.median(errors[:, 201-8])),
            'median_best_error': float(np.median(errors.min(1))),
            'best_lag_by_particle': best.tolist()}


def history_retention(cold, warm, prefix):
    """Compare different starts at the same absolute clock time and fixed z."""
    end = min(cold.shape[1]-prefix, warm.shape[1])
    a = np.asarray(cold[:, prefix:prefix+end], dtype=np.float64)[:, -256:]
    b = np.asarray(warm[:, :end], dtype=np.float64)[:, -256:]
    scale = ((a-a.mean(1, keepdims=True))**2).mean((1, 2)).clip(1e-12)
    error = ((a-b)**2).mean((1, 2))/scale
    return {'normalized_same_particle_cold_warm_error_median': float(np.median(error)),
            'fraction_below_1e_3': float((error < 1e-3).mean())}


@torch.no_grad()
def diagnose(path, device, clock_rates):
    summary = json.loads((path/'summary.json').read_text())
    with np.load(path/'trajectories.npz') as arrays:
        panels = {k: arrays[k].copy() for k in ('generated', 'prefix8', 'prefix32')}
        observed = arrays['observed_prefix32'].copy()
        reference = arrays['continuation_reference'].copy()
    result = {'name': summary['name'], 'source': str(path),
              'saved': {k: recurrence(v) for k, v in panels.items()}, 'clock_interventions': {},
              'history_retention': {str(n): history_retention(panels['generated'], panels[f'prefix{n}'], n)
                                    for n in (8, 32)}}
    if clock_rates:
        saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
        cfg = handoff.Config(**saved['config'])
        assert cfg.clock_bands, 'Clock intervention requires a clock-enabled model'
        g, d, prior, _ = handoff.build(cfg, device)
        for name, module in [('generator', g), ('critic', d), ('prior', prior)]:
            module.load_state_dict(saved[name])
            module.eval()
        z = prior(torch.arange(cfg.eval_batch, device=device))
        initial = core.context(d.writer, torch.as_tensor(observed, device=device))
        for rate in clock_rates:
            memory, points = initial.clone(), []
            for step in range(cfg.eval_steps):
                point, _ = handoff.local_point(g, z, memory, 32+rate*step)
                points.append(point)
                memory = d.writer.write(memory, point)
            paths = torch.stack(points, 1).cpu().numpy()
            if rate == 1:
                np.testing.assert_allclose(paths[:, 0], panels['prefix32'][:, 0], atol=1e-5, rtol=1e-5)
            result['clock_interventions'][str(rate)] = {
                'expected_clock_period': 2*np.pi/(cfg.clock_frequency*cfg.clock_rate*rate),
                'recurrence': recurrence(paths), 'fidelity': core.fidelity(paths, reference, 32)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--clock-rates', type=float, nargs='*', default=[])
    args = parser.parse_args()
    assert all(rate > 0 for rate in args.clock_rates)
    torch.set_num_threads(1)
    results = [diagnose(p, args.device, args.clock_rates) for p in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'evaluation_only': True,
        'definition': 'Minimum normalized position recurrence over integer lags8..250 in the last512 points; not necessarily fundamental period. Normalize by each path centered position variance. Clock interventions preserve t=32 initially and alter only subsequent clock advancement. History retention compares last256 overlapping absolute-clock times of warm versus cold for the same particle, normalized by centered cold variance. Small error indicates convergence of outputs from different histories, not absence of memory use.',
        'results': results}, indent=2, allow_nan=False)+'\n')
    for row in results:
        print(row['name'], 'saved', row['saved']['prefix32']['median_best_lag'],
              'clock rates', {k: v['recurrence']['median_best_lag'] for k, v in row['clock_interventions'].items()})
