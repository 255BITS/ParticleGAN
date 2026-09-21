#!/usr/bin/env python
"""Replay saved EMA transition paths and independently check reported metrics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_transition import training_recipe
from experiments.analyze_transition import preference_audit
from lib.transition import (Transitions, TransitionGenerator, TransitionEncoder, TransitionScaler,
                            encoded_transition, composed_transition, metrics)


def check_metrics(measured, reported):
    for key, value in measured.items():
        if key == 'contexts':
            for a, b in zip(value, reported[key], strict=True):
                check_metrics(a, b)
        elif abs(value-reported[key]) > 1e-6:
            raise AssertionError((key, value, reported[key]))


@torch.no_grad()
def verify(path, device):
    path = Path(path)
    run = json.loads((path/'summary.json').read_text())
    saved = torch.load(path/'final.pt', map_location=device, weights_only=True)
    cfg = saved['config']
    g = TransitionGenerator(cfg['z_dim'], cfg['architecture'], cfg['width'],
                            class_scale=cfg.get('g_class_scale', 1.),
                            context_scale=cfg.get('g_context_scale', 1.)).to(device).eval()
    g.load_state_dict(saved['G'])
    prior = training_recipe(cfg).make_prior(device=device)
    prior.load_state_dict(saved['prior'])
    assert float(prior.sigma) == json.loads((path/'prior.json').read_text())['sigma']
    scaler = TransitionScaler(**saved['scaler'])
    toy = Transitions(cfg['length'], device, cfg['geometry_mode'])
    e = None
    if cfg.get('encoder', False):
        e = TransitionEncoder(cfg['z_dim'], cfg['encoder_width'], cfg['g_class_scale'],
                              cfg['g_context_scale']).to(device).eval()
        e.load_state_dict(saved['E'])
    result = dict(run=path.name, fixed_sigma=float(prior.sigma), splits={}, preference=preference_audit(path, run))
    for split in ('train', 'test'):
        with np.load(path/f'{split}_samples.npz') as arrays:
            data = {key: torch.as_tensor(arrays[key], device=device) for key in arrays.files}
        count = min(256, len(data['c']))
        sl = slice(0, count)
        z, _ = prior.sample(count, torch.Generator(device=device).manual_seed(99000))
        context = toy.condition(data['geom'][sl], data['tick'][sl])
        replay = scaler.inverse(g(z, data['c'][sl], context))
        torch.testing.assert_close(replay, data['x'][sl], rtol=0, atol=0)
        measured = metrics(data['x'], data['real'], scaler, data['group'])
        check_metrics(measured, run['final'][split])
        classes = {}
        for c in (0, 1):
            mask = data['c'] == c
            classes[str(c)] = {k: v for k, v in metrics(data['x'][mask], data['real'][mask], scaler,
                                                        data['group'][mask]).items() if k != 'contexts'}
        checks = dict(prior_replay='first 256 exact', metrics='all contexts within 1e-6', classes=classes)
        if e is not None:
            with np.load(path/f'{split}_inference.npz') as arrays:
                inf = {key: torch.as_tensor(arrays[key], device=device) for key in arrays.files}
            real, fake = scaler(data['real'][sl]), scaler(data['x'][sl])
            decoded, encoding = encoded_transition(e, g, prior, real[:, :4], data['c'][sl], context)
            synthetic, _, senc = composed_transition(e, g, prior, fake, data['c'][sl], context)
            torch.testing.assert_close(scaler.inverse(decoded), inf['reconstruction'][sl], atol=0, rtol=0)
            torch.testing.assert_close(scaler.inverse(synthetic), inf['synthetic'][sl], atol=0, rtol=0)
            torch.testing.assert_close(encoding.indices[:, 0], inf['real_ids'][sl], atol=0, rtol=0)
            torch.testing.assert_close(senc.indices[:, 0], inf['synthetic_ids'][sl], atol=0, rtol=0)
            assert torch.isfinite(inf['prediction']).all() and torch.isfinite(inf['synthetic']).all()
            torch.testing.assert_close(inf['prediction'][:, :4], data['real'][:, :4], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(inf['synthetic'][:, :4], data['x'][:, :4], atol=1e-6, rtol=1e-6)
            for key, array in (('conditional', 'prediction'), ('synthetic', 'synthetic')):
                check_metrics(metrics(inf[array], data['real'], scaler, data['group']), run['inference'][split][key])
            errors = (inf['prediction'][:, 4:]-data['real'][:, 4:]).norm(dim=1)
            assert abs(float(errors.mean())-run['inference'][split]['paired']['next_l2']) < 1e-6
            checks['encoder_replay'] = 'first 256 reconstructions, compositions, routing IDs exact'
        result['splits'][split] = checks
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = [verify(path, args.device) for path in args.runs]
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
