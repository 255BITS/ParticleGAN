#!/usr/bin/env python
"""Matched D-only interventions and held-out branch/gradient diagnostics.

The architecture trainer is pinned so the grid source certificate transitively
covers it. No historical source is modified. D-only artifacts are explicit D
state deltas, never mislabeled as joint-training checkpoints.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'experiments/train_cifar_ae_transgan.py'
BASE_SHA = '76a481fe19b79450f89dd6ec2c592ddeb3f8a3095d6cddc35c8f26bc4609248e'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
from experiments import train_cifar_ae_transgan as base
from experiments.run_grid import code_provenance
from particlegan import GANLoss, GradientPenalty, MoGParticlePrior

DEFAULTS = {
    'checkpoint': '', 'checkpoint_sha256': '', 'out_dir': '',
    'steps': 2048, 'reg_every': 8, 'penalty_coeff': 1.,
    'eval_every': 512, 'log_every': 128, 'samples': 2048,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_parent(path, expected):
    assert digest(path) == expected, 'parent digest mismatch'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    for name, value in ck['sources'].items():
        assert digest(ROOT / name) == value, name
    cfg = {**base.DEFAULTS, **ck['config']}
    g, d, e = [m.cuda() for m in base.build_models(cfg)]
    prior = MoGParticlePrior(cfg['num_particles'], cfg['z_dim'], sigma_rel=cfg['sigma_rel']).cuda()
    for m, key in ((g, 'G'), (d, 'D'), (e, 'E'), (prior, 'prior')):
        m.load_state_dict(ck[key], strict=True)
        m.eval().requires_grad_(False)
    return ck, cfg, g, d, e, prior


def branch_scores(d, x):
    """Use original forward, capturing exact weighted additive components."""
    captured = {}
    def hook(name, pixel=False):
        def save(_m, _args, output):
            captured[name] = (output[1] if pixel else output).squeeze(1)
        return save
    handles = [d.critic.pixel.register_forward_hook(hook('pixel', True))]
    handles += [head.register_forward_hook(hook(f'feature{i}')) for i, head in enumerate(d.critic.project)]
    try:
        total = d(x)
    finally:
        for handle in handles:
            handle.remove()
    result = {'pixel': captured['pixel'] / math.sqrt(2)}
    result.update({f'feature{i}': captured[f'feature{i}'] / math.sqrt(6) for i in range(3)})
    result['features'] = sum(result[f'feature{i}'] for i in range(3))
    torch.testing.assert_close(total, result['pixel'] + result['features'], rtol=2e-5, atol=2e-5)
    return {'total': total, **result}


def auc(real, fake):
    ordered = fake.sort().values.contiguous()
    return float(((torch.searchsorted(ordered, real, right=False) +
                   torch.searchsorted(ordered, real, right=True)).float() / (2 * len(fake))).mean())


def probe(g, d, prior, images, samples, seed, parameter_gradients=True):
    """Fixed independent held-out draws; never advances training RNGs."""
    stream = base.rng(seed)
    ids = torch.randperm(len(images), device='cuda', generator=stream)[:samples]
    g.eval(); d.eval(); prior.eval()
    d.requires_grad_(False)
    g.requires_grad_(parameter_gradients); prior.requires_grad_(parameter_gradients)
    scores, norms, extras = {}, {}, []
    try:
        for lo in range(0, samples, 64):
            real = (images[ids[lo:lo + 64]].float() / 127.5 - 1).requires_grad_(True)
            z, _ = prior.sample(len(real), stream)
            fake = g(z)
            if not parameter_gradients:
                fake = fake.detach().requires_grad_(True)
            rs, fs = branch_scores(d, real), branch_scores(d, fake)
            grads = {}
            for key in rs:
                scores.setdefault(key, [[], []])
                norms.setdefault(key, [[], []])
                for index, (values, image) in enumerate(((rs, real), (fs, fake))):
                    grad, = torch.autograd.grad(values[key].sum(), image, retain_graph=True)
                    scores[key][index].append(values[key].detach().cpu())
                    norms[key][index].append(grad.detach().flatten(1).norm(dim=1).cpu())
                    if index == 1:
                        grads[key] = grad.detach().flatten(1)
            ng = {k: v.norm(dim=1) for k, v in grads.items()}
            row = {
                'pixel_features_cosine': float(F.cosine_similarity(grads['pixel'], grads['features'], dim=1).mean()),
                'total_over_sum_branch_grad_norm': float((ng['total'] / (ng['pixel'] + ng['features']).clamp_min(1e-20)).mean()),
                'g_loss': float(GANLoss().g_loss(fs['total'], rs['total'].detach()).detach()),
            }
            if parameter_gradients:
                gp, pp = list(g.parameters()), list(prior.parameters())
                grad = torch.autograd.grad(GANLoss().g_loss(fs['total'], rs['total'].detach()), gp + pp)
                row['adv_G_norm'] = float(torch.stack([v.square().sum() for v in grad[:len(gp)]]).sum().sqrt())
                row['adv_prior_norm'] = float(torch.stack([v.square().sum() for v in grad[len(gp):]]).sum().sqrt())
            extras.append(row)
        result = {'samples_per_distribution': samples, 'seed': seed, 'branches': {}}
        for key, pair in scores.items():
            r, f = [torch.cat(v) for v in pair]
            nr, nf = [torch.cat(v) for v in norms[key]]
            result['branches'][key] = {
                'auc': auc(r, f), 'paired_real_above_fake': float((r > f).float().mean()),
                'real_minus_fake': float(r.mean() - f.mean()),
                'real_score_mean': float(r.mean()), 'fake_score_mean': float(f.mean()),
                'real_score_std': float(r.std()), 'fake_score_std': float(f.std()),
                **{f'{side}_grad_{stat}': float(value) for side, n in [('real', nr), ('fake', nf)]
                   for stat, value in [('mean', n.mean()), ('median', n.median()), ('p90', n.quantile(.9)),
                                       ('above_cap_fraction', (n > 1).float().mean())]},
            }
        result.update({key: sum(row[key] for row in extras) / len(extras) for key in extras[0]})
        return result
    finally:
        g.requires_grad_(False); prior.requires_grad_(False)


def train(cfg):
    assert set(cfg) == set(DEFAULTS)
    assert cfg['steps'] >= 0 and 0 < cfg['samples'] <= 10000
    assert cfg['samples'] % 64 == 0 and cfg['eval_every'] > 0 and cfg['log_every'] > 0
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / cfg['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'metrics.jsonl').exists(), 'fresh directory required'
    started = time.perf_counter()
    ck, parent_cfg, g, d, e, prior = load_parent(ROOT / cfg['checkpoint'], cfg['checkpoint_sha256'])
    provenance = code_provenance(__file__, sys.executable)
    provenance['sources'][str(BASE.relative_to(ROOT))] = BASE_SHA
    base.write_json(out / 'provenance.json', provenance)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg))
    with zipfile.ZipFile(out / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in provenance['sources'].items():
            assert digest(ROOT / name) == expected
            archive.write(ROOT / name, name)
    def data(train):
        array = CIFAR10(parent_cfg['data_dir'], train=train, download=False).data
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().cuda()
    train_images, test_images = data(True), data(False)
    frozen_hash = base.state_hash([g, e, prior, d.critic.features])
    d.requires_grad_(True)
    od = torch.optim.Adam([p for p in d.parameters() if p.requires_grad], lr=parent_cfg['d_lr'], betas=(0., .999), fused=True)
    od.load_state_dict(ck['optimizer_d'])
    initial_d = base.state_hash([d])
    initial_counter = max(float(state['step']) for state in od.state.values())
    streams = {name: base.rng(parent_cfg['seed']) for name in ('data', 'prior')}
    for name, stream in streams.items():
        stream.set_state(ck['rng'][name])
    torch.set_rng_state(ck['torch_rng'])
    torch.cuda.set_rng_state_all(ck['cuda_rng'])
    penalty = GradientPenalty(coeff=cfg['penalty_coeff'], lazy_k=cfg['reg_every'])
    seed = parent_cfg['seed'] + 80000
    train_seconds = 0.
    with (out / 'metrics.jsonl').open('w', buffering=1) as log:
        def evaluate(step):
            before = base.state_hash([g, d, e, prior])
            states = {k: v.get_state().clone() for k, v in streams.items()}
            result = {'step': step, 'train_seconds': train_seconds,
                      'test': probe(g, d, prior, test_images, cfg['samples'], seed),
                      'train': probe(g, d, prior, train_images, cfg['samples'], seed + 1, False)}
            assert before == base.state_hash([g, d, e, prior]), 'probe mutated models'
            assert all(torch.equal(states[k], v.get_state()) for k, v in streams.items())
            log.write(json.dumps(result, allow_nan=False) + '\n')
            t = result['test']; b = t['branches']['total']
            print(f"EVAL d_steps={step} test_auc={b['auc']:.4f} train_auc={result['train']['branches']['total']['auc']:.4f} "
                  f"input_grad={b['fake_grad_mean']:.5f} adv_G={t['adv_G_norm']:.5f} cancellation={t['total_over_sum_branch_grad_norm']:.4f}", flush=True)
            return result
        baseline = evaluate(0)
        final = baseline
        window = []
        block = time.perf_counter()
        for step in range(1, cfg['steps'] + 1):
            d.train().requires_grad_(True)
            ids = torch.randint(len(train_images), (parent_cfg['batch_size'],), device='cuda', generator=streams['data'])
            real = train_images[ids].float() / 127.5 - 1
            flip = torch.rand((len(real), 1, 1, 1), device='cuda', generator=streams['data']) < .5
            real = torch.where(flip, real.flip(-1), real)
            with torch.no_grad():
                z, _ = prior.sample(len(real), streams['prior'])
                fake = g(z)
            od.zero_grad(set_to_none=True)
            dp = penalty(d, real, fake, ck['step'] + step)
            loss = GANLoss().d_loss(d(real), d(fake))
            (loss + dp).backward()
            od.step()
            window.append(torch.stack([loss.detach(), dp.detach()]))
            if step % cfg['log_every'] == 0 or step == cfg['steps']:
                values = torch.stack(window).mean(0).tolist(); window = []
                assert all(math.isfinite(v) for v in values)
                print(f"step={step}/{cfg['steps']} D_adversarial={values[0]:.6f} penalty_window_mean={values[1]:.6f}", flush=True)
            if step % cfg['eval_every'] == 0 or step == cfg['steps']:
                torch.cuda.synchronize()
                train_seconds += time.perf_counter() - block
                final = evaluate(step)
                block = time.perf_counter()
    assert frozen_hash == base.state_hash([g, e, prior, d.critic.features]), 'frozen state changed'
    assert digest(ROOT / cfg['checkpoint']) == cfg['checkpoint_sha256']
    final_counter = max(float(state['step']) for state in od.state.values())
    assert final_counter == initial_counter + cfg['steps']
    if cfg['steps']:
        assert initial_d != base.state_hash([d])
        torch.save({'artifact_type': 'D_only_delta', 'D': d.state_dict(), 'optimizer_d': od.state_dict(),
                    'parent': cfg['checkpoint'], 'parent_sha256': cfg['checkpoint_sha256'],
                    'd_only_steps': cfg['steps'], 'config': cfg, 'sources': provenance['sources'],
                    'rng': {k: v.get_state() for k, v in streams.items()}}, out / 'D_delta.pt')
    summary = {'config': cfg, 'baseline': baseline, 'final': final, 'train_seconds': train_seconds,
               'total_seconds': time.perf_counter() - started, 'frozen_state_unchanged': True,
               'parent_unchanged': True, 'optimizer_d_steps_before': initial_counter,
               'optimizer_d_steps_after': final_counter, 'initial_D_sha256': initial_d,
               'final_D_sha256': base.state_hash([d]),
               'note': 'Live G/prior fixed. Test split never trained; train split diagnostic may include training examples. AUC is not FID.'}
    base.write_json(out / 'summary.json', summary)
    print(f"COMPLETE d_steps={cfg['steps']} train_s={train_seconds:.1f} total_s={summary['total_seconds']:.1f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    user = base.read_config(args.config)
    assert not set(user) - set(DEFAULTS)
    train({**DEFAULTS, **user})
