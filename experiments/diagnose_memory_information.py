"""Evaluation-only held-out probes of original-process information in frozen M."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as handoff
from experiments.memory_core_scout import context


def episodes(count, length, rng):
    """Same process distribution as training, explicit evaluation-only labels."""
    center = (torch.rand(count, 1, 2, generator=rng)-.5)*1.5
    radius = .6+.8*torch.rand(count, 1, generator=rng)
    phase = 2*math.pi*torch.rand(count, 1, generator=rng)
    speed = (.12+.28*torch.rand(count, 1, generator=rng))*(2*(torch.rand(count, 1, generator=rng)>.5)-1)
    theta = phase+speed*torch.arange(length)
    clean = center+radius[..., None]*torch.stack((theta.cos(), theta.sin()), -1)
    observed = clean+.03*torch.randn(clean.shape, generator=rng)
    ids = torch.randint(512, (count,), generator=rng)
    return observed, torch.cat((radius, speed), -1), ids


def make_splits(sizes, length):
    # Single diagnostic RNG, successive independent episodes; no reseeding experiments.
    rng = torch.Generator().manual_seed(9174301)
    return {name: episodes(n, length, rng) for name, n in zip(('train', 'validation', 'test'), sizes)}


@torch.no_grad()
def collect(g, d, prior, observed, ids, depths, prefix=32, batch=512, memory_kind='M'):
    device = next(g.parameters()).device
    out = {kind: {depth: [] for depth in depths} for kind in ('generated', 'real')}
    particles = []
    for start in range(0, len(ids), batch):
        x = observed[start:start+batch].to(device)
        z = prior(ids[start:start+batch].to(device))
        particles.append(z.cpu())
        real = context(d.writer, x[:, :prefix])
        generated = real.clone()
        recurrent = bool(getattr(g, 'g_state_dim', 0))
        if memory_kind == 'Mg' and not recurrent:
            raise ValueError('Mg probes require a recurrent generator')
        if recurrent:
            state = g.initial_state(z)
            prefix_memory = d.writer.initial(z)
            for point in x[:, :prefix].unbind(1):
                state = g.write_state(state, point, prefix_memory)
                prefix_memory = d.writer.write(prefix_memory, point)
            real_state = state.clone()
        for depth in range(max(depths)+1):
            if depth in depths:
                out['generated'][depth].append((state if memory_kind == 'Mg' else generated).flatten(1).cpu())
                out['real'][depth].append((real_state if memory_kind == 'Mg' else real).flatten(1).cpu())
            if depth < max(depths):
                if recurrent:
                    point, features = g(z, generated, state, time_index=prefix+depth)
                    state = g.generated_state(state, point, generated, features)
                    real_state = g.write_state(real_state, x[:, prefix+depth], real)
                else:
                    point, _ = g(z, generated, time_index=prefix+depth)
                generated = d.writer.write(generated, point)
                real = d.writer.write(real, x[:, prefix+depth])
    return {kind: {n: torch.cat(rows) for n, rows in by_depth.items()}
            for kind, by_depth in out.items()}, torch.cat(particles)


def normalization(train):
    return train.mean(0), train.std(0, unbiased=False).clamp_min(1e-6)


def metrics(pred, truth):
    pred, truth = pred.double(), truth.double()
    result = {}
    for i, key in enumerate(('radius', 'signed_speed')):
        err = pred[:, i]-truth[:, i]
        result[key+'_r2'] = float(1-err.square().sum()/(truth[:, i]-truth[:, i].mean()).square().sum())
        result[key+'_mae'] = float(err.abs().mean())
    result['direction_accuracy'] = float(((pred[:, 1]>0)==(truth[:, 1]>0)).double().mean())
    result['speed_magnitude_mae'] = float((pred[:, 1].abs()-truth[:, 1].abs()).abs().mean())
    return result


def fit_probe(features, targets, kind, device, epochs=250):
    """Normalize only on training examples, select on validation, never test."""
    xmean, xstd = normalization(features['train'])
    ymean, ystd = normalization(targets['train'])
    xs = {k: ((x-xmean)/xstd).to(device) for k, x in features.items()}
    ys = {k: ((y-ymean)/ystd).to(device) for k, y in targets.items()}
    if kind == 'ridge':
        xa = torch.cat((xs['train'].double(), torch.ones(len(xs['train']), 1, device=device)), -1)
        xv = torch.cat((xs['validation'].double(), torch.ones(len(xs['validation']), 1, device=device)), -1)
        gram, cross = xa.T@xa/len(xa), xa.T@ys['train'].double()/len(xa)
        penalty = torch.eye(xa.shape[1], dtype=torch.double, device=device)
        penalty[-1, -1] = 0
        best, chosen = float('inf'), None
        for alpha in (1e-6, 1e-4, .001, .01, .1, 1., 10.):
            weights = torch.linalg.solve(gram+alpha*penalty, cross)
            score = float((xv@weights-ys['validation']).square().mean())
            if score < best:
                best, chosen, model = score, alpha, weights
        predict_normalized = lambda x: torch.cat((x.double(), torch.ones(len(x), 1, device=device)), -1)@model
        selection = {'ridge_alpha': chosen, 'validation_standardized_mse': best}
    else:
        # Fixed architecture and initialization protocol across domains; no test tuning.
        with torch.random.fork_rng(devices=[torch.device(device).index] if str(device).startswith('cuda') else []):
            torch.manual_seed(83941)
            model = nn.Sequential(nn.Linear(xs['train'].shape[1], 64), nn.SiLU(),
                                  nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
        best, best_epoch, state = float('inf'), 0, None
        for epoch in range(1, epochs+1):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(xs['train'])-ys['train']).square().mean()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 or epoch == epochs:
                with torch.no_grad():
                    score = float((model(xs['validation'])-ys['validation']).square().mean())
                if score < best:
                    best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        model.load_state_dict(state)
        model.eval()
        predict_normalized = model
        selection = {'epoch': best_epoch, 'validation_standardized_mse': best}

    @torch.no_grad()
    def predict(x):
        xn = ((x-xmean)/xstd).to(device)
        return predict_normalized(xn).cpu()*ystd+ymean
    return predict, selection


def diagnose(path, splits, depths, device, epochs, memory_kind='M'):
    json.loads((path/'summary.json').read_text())  # completed models only
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = handoff.Config(**saved['config'])
    g, d, prior, _ = handoff.build(cfg, device)
    assert memory_kind != 'Mg' or cfg.g_state_dim
    for key, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[key])
        module.eval().requires_grad_(False)
    states, zs = {}, {}
    for split, (observed, _, ids) in splits.items():
        states[split], zs[split] = collect(g, d, prior, observed, ids, depths, memory_kind=memory_kind)
    targets = {k: row[1] for k, row in splits.items()}
    rows = []
    for depth in depths:
        for domain in ('real', 'generated'):
            for kind, add_z in (('ridge', False), ('mlp', False), ('mlp', True)):
                features = {k: torch.cat((s[domain][depth], zs[k]), -1) if add_z else s[domain][depth]
                            for k, s in states.items()}
                predictor, selection = fit_probe(features, targets, kind, device, epochs)
                row = {'depth': depth, 'training_domain': domain, 'probe': kind,
                       'features': memory_kind+'+z' if add_z else memory_kind, 'selection': selection,
                       'test': metrics(predictor(features['test']), targets['test'])}
                if domain == 'real':
                    test = states['test']['generated'][depth]
                    if add_z:
                        test = torch.cat((test, zs['test']), -1)
                    row['transfer_generated_test'] = metrics(predictor(test), targets['test'])
                rows.append(row)
        print(json.dumps({'event': 'depth_completed', 'name': cfg.name, 'depth': depth,
                          'results': rows[-6:]}), flush=True)
    controls = {}
    for feature_name in ('z', 'shuffled_M', 'oracle_labels'):
        if feature_name == 'z':
            features = zs
        elif feature_name == 'oracle_labels':
            features = targets
        else:
            rng = torch.Generator().manual_seed(817326)
            features = {k: s['real'][0][torch.randperm(len(s['real'][0]), generator=rng)] for k, s in states.items()}
        predictor, selection = fit_probe(features, targets, 'ridge' if feature_name == 'oracle_labels' else 'mlp', device, epochs)
        controls[feature_name] = {'selection': selection, 'test': metrics(predictor(features['test']), targets['test'])}
    controls['train_mean'] = metrics(targets['train'].mean(0).expand_as(targets['test']), targets['test'])
    return {'name': cfg.name, 'source': str(path), 'checkpoint_sha256': hashlib.sha256((path/'model.pt').read_bytes()).hexdigest(),
            'rows': rows, 'controls': controls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--sizes', nargs=3, type=int, default=[2048, 512, 1024])
    parser.add_argument('--depths', nargs='+', type=int, default=[0, 1, 8, 32, 128])
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--memory', choices=['M', 'Mg'], default='M')
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.time()
    splits = make_splits(args.sizes, 32+max(args.depths))
    report = {'evaluation_only': True, 'protocol': {
        'split_sizes': dict(zip(splits, args.sizes)), 'prefix': 32, 'depths': args.depths, 'memory': args.memory,
        'targets': ['radius', 'signed_angular_speed'], 'distribution': 'Independent center uniform[-.75,.75]^2, radius uniform[.6,1.4], phase uniform[0,2pi], signed speed magnitude uniform[.12,.40], observation noise std .03.',
        'episodes': 'Successive fresh independent episodes from one diagnostic RNG seed 9174301, disjoint across train/validation/test; same panels across checkpoints. No training histories reused. Not a GAN seed experiment.',
        'particles': 'Uniform independent draws from saved learned 512-particle table; fixed z per episode. Probe episodes are held out, particles are not.',
        'normalization': 'Probe train only. Ridge regularization and nonlinear checkpoint selected using validation standardized MSE. Test used only for final reporting.',
        'nonlinear': f'64-SiLU-64-SiLU-2, AdamW lr .003 decay .001, full batch, {args.epochs} epochs, validation every10.',
        'teacher_control': 'Real observations continue through matching depths; all process parameters remain in training support, but histories after63 exceed training length.',
        'interpretation': 'Depth n is after n generated writes. Separate per-depth/domain probes assess decodability; real-trained transfer assesses representation compatibility. Probe failure does not prove information absence. M+z tests simple entanglement. Probe capacity is not exhaustive; compare clean baseline calibration. Regression losses belong only to evaluation probes, never GAN training.'},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'split_hashes': {k: hashlib.sha256(row[0].numpy().tobytes()).hexdigest() for k, row in splits.items()}, 'results': []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path in args.runs:
        report['results'].append(diagnose(path, splits, args.depths, args.device, args.epochs, args.memory))
        report['wall_seconds'] = time.time()-started
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'event': 'model_completed', 'name': report['results'][-1]['name']}), flush=True)


if __name__ == '__main__':
    main()
