#!/usr/bin/env python
"""Probe a frozen transition encoder's response to physical action perturbations.

This leaves the training route manifold. It is a post-hoc response/stability
check, not a causal-identification test or part of the generation leaderboard.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_transition import training_recipe
from lib.transition import (Transitions, TransitionEncoder, TransitionGenerator, TransitionScaler,
                            encoded_transition)


@torch.no_grad()
def audit(path, device, delta):
    path = Path(path)
    saved = torch.load(path/'final.pt', weights_only=True, map_location=device)
    cfg = saved['config']
    g = TransitionGenerator(cfg['z_dim'], cfg['architecture'], cfg['width'], cfg['g_class_scale'],
                            cfg['g_context_scale']).to(device).eval()
    g.load_state_dict(saved['G'])
    e = TransitionEncoder(cfg['z_dim'], cfg['encoder_width'], cfg['g_class_scale'],
                          cfg['g_context_scale']).to(device).eval()
    e.load_state_dict(saved['E'])
    prior = training_recipe(cfg).make_prior(device=device)
    prior.load_state_dict(saved['prior'])
    scaler = TransitionScaler(**saved['scaler'])
    toy = Transitions(cfg['length'], device, cfg['geometry_mode'])
    result = dict(run=path.name, delta=delta, splits={}, description=__doc__)
    for split in ('train', 'test'):
        with np.load(path/f'{split}_samples.npz') as raw:
            ids = np.concatenate([np.flatnonzero(raw['group'] == group)[:32]
                                  for group in np.unique(raw['group'])])
            real = torch.as_tensor(raw['real'][ids], device=device)
            c = torch.as_tensor(raw['c'][ids], device=device)
            context = toy.condition(torch.as_tensor(raw['geom'][ids], device=device),
                                    torch.as_tensor(raw['tick'][ids], device=device))
        normalized = scaler(real)
        jacobians, switches, offsets, indices = [], [], [], []
        for start in range(0, len(c), 256):
            sl = slice(start, start+256)
            obs = normalized[sl, :4]
            _, enc = encoded_transition(e, g, prior, obs, c[sl], context[sl])
            indices.append(enc.indices[:, 0])
            offsets.append((enc.codes[:, 0]-prior.means()[enc.indices[:, 0]])/prior.sigma)
            axes, axis_switches = [], []
            for axis in (0, 1):
                predictions, routings = [], []
                for sign in (-1, 1):
                    changed = obs.clone()
                    changed[:, 2+axis] += sign*delta/scaler.scale[2+axis]
                    decoded, pert = encoded_transition(e, g, prior, changed, c[sl], context[sl])
                    predictions.append(scaler.inverse(decoded)[:, 4:])
                    routings.append(pert.indices[:, 0])
                axes.append((predictions[1]-predictions[0])/(2*delta))
                axis_switches.append(routings[1] != routings[0])
            jacobians.append(torch.stack(axes, dim=2))
            switches.append(torch.stack(axis_switches, dim=1))
        j, switch, off, idx = map(torch.cat, (jacobians, switches, offsets, indices))
        error = (j-torch.eye(2, device=device)).square().sum((1, 2)).sqrt()
        stable = ~switch.any(dim=1)
        result['splits'][split] = dict(
            count=len(c), mean_response_matrix=j.mean(0).cpu().tolist(),
            mean_identity_error=float(error.mean()), median_identity_error=float(error.median()),
            p95_identity_error=float(error.quantile(.95)),
            stable_route_mean_identity_error=float(error[stable].mean()) if stable.any() else None,
            mean_response_norm=float(j.square().sum((1, 2)).sqrt().mean()),
            route_switch_fraction=float(switch.float().mean()),
            offset_rms_in_sigma=float(off.square().mean().sqrt()),
            offset_saturation_fraction=float((off.abs() > 2.9).float().mean()),
            component_counts={str(int(i)): int((idx == i).sum()) for i in idx.unique()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--delta', type=float, default=.0001)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if not np.isfinite(args.delta) or args.delta <= 0:
        parser.error('delta must be positive and finite')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = audit(args.run, args.device, args.delta)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    Path(args.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
