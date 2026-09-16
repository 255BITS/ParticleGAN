"""Matched real-history interventions on completed models; evaluation only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as handoff
from experiments import memory_core_scout as core
from experiments.memory_orbit_metrics import fit_circle, angles, orbit_progress


def counterfactuals(clean, observed, prefix=32):
    """Keep center, handoff phase, particle and observation-noise realization fixed."""
    clean = np.asarray(clean, dtype=np.float64)
    center, radius = fit_circle(clean[:, :32])
    offsets = clean-center[:, None]
    omega = angles(offsets[:, 0], offsets[:, 1])
    phase = np.arctan2(offsets[:, prefix, 1], offsets[:, prefix, 0])
    noise = observed-clean[:, :prefix]
    variants = {'original': (radius, omega),
                'radius_low': (np.full_like(radius, .65), omega),
                'radius_high': (np.full_like(radius, 1.35), omega),
                'speed_low': (radius, np.sign(omega)*.14),
                'speed_high': (radius, np.sign(omega)*.36),
                'direction_flipped': (radius, -omega)}
    result = {}
    for name, (r, w) in variants.items():
        phase_path = phase[:, None]+w[:, None]*(np.arange(clean.shape[1])-prefix)
        reference = center[:, None]+r[:, None, None]*np.stack((np.cos(phase_path), np.sin(phase_path)), -1)
        history = reference[:, :prefix]+noise
        if name == 'original':
            reference, history = clean, observed
        result[name] = (reference.astype(np.float32), history.astype(np.float32), center, r, w)
    return result


def measured_process(path, center, omega):
    offsets = np.asarray(path, dtype=np.float64)-center[:, None]
    radius = np.linalg.norm(offsets[:, -256:], axis=-1).mean(1)
    angular = angles(offsets[:, :-1], offsets[:, 1:])[:, -256:]
    return {'radius': radius, 'signed_speed': angular.mean(1),
            'direction': (angular*np.sign(omega[:, None]) > 0).mean(1)}


@torch.no_grad()
def diagnose(path, device):
    json.loads((path/'summary.json').read_text())
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = handoff.Config(**saved['config'])
    g, d, prior, _ = handoff.build(cfg, device)
    for key, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[key])
        module.eval()
    z = prior(torch.arange(cfg.eval_batch, device=device))
    with np.load(path/'trajectories.npz') as arrays:
        clean = arrays['continuation_reference'].copy()
        observed = arrays['observed_prefix32'].copy()
        original_path = arrays['prefix32'].copy()
    results, measured = {}, {}
    for name, (reference, history, center, radius, omega) in counterfactuals(clean, observed).items():
        if name == 'original':
            generated = original_path
        else:
            generated, _ = core.continuation(g, d.writer, z,
                torch.as_tensor(history, device=device), cfg.eval_steps)
            generated = generated.cpu().numpy()
        m = measured_process(generated, center, omega)
        measured[name] = m
        results[name] = {'progress': orbit_progress(generated, reference, 32),
                         'fidelity': core.fidelity(generated, reference, 32),
                         'late_radius_mae': float(np.abs(m['radius']-radius).mean()),
                         'late_signed_speed_mae': float(np.abs(m['signed_speed']-omega).mean()),
                         'late_direction_agreement': float(m['direction'].mean())}
    original_omega = counterfactuals(clean, observed)['original'][-1]
    radius_response = (measured['radius_high']['radius']-measured['radius_low']['radius'])/.7
    speed_response = (measured['speed_high']['signed_speed']-measured['speed_low']['signed_speed'])*np.sign(original_omega)/.22
    flipped = measured['direction_flipped']['signed_speed']
    original = measured['original']['signed_speed']
    return {'name': cfg.name, 'source': str(path), 'steps': cfg.steps,
            'variants': results,
            'response': {
                'radius_response_mean_ideal1': float(radius_response.mean()),
                'radius_response_median_ideal1': float(np.median(radius_response)),
                'radius_correct_change_fraction': float((radius_response > 0).mean()),
                'speed_response_mean_ideal1': float(speed_response.mean()),
                'speed_response_median_ideal1': float(np.median(speed_response)),
                'speed_correct_change_fraction': float((speed_response > 0).mean()),
                'direction_correct_in_both_fraction': float(((original*original_omega > 0) & (flipped*original_omega < 0)).mean()),
            }}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for path in args.runs:
        row = diagnose(path, args.device)
        results.append(row)
        print(row['name'], row['response'], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'evaluation_only': True,
        'definition': 'Real prefixes vary one process property, keeping center, phase at handoff, noise, z and clock fixed. Radius .65/1.35 and absolute speed .14/.36 are inside training support. Direction is flipped. Evaluate1024 expert-free points after prefix32; measure process response in last256. Ideal normalized radius/speed response is1. Sensitivity alone does not establish correctness: inspect fidelity and direction, too. No training uses these trajectories.',
        'results': results}, indent=2, allow_nan=False)+'\n')
