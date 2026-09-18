#!/usr/bin/env python
"""Lower-LR particle VAE queue, including a constant-KL hard posterior.

Copied from train_mog_vae.py to preserve previous queue provenance. Soft arms
use unbiased two-draw routing; hard routing uses a biased ST query gradient."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import merge_config, read_config
from experiments.run_grid import code_provenance
from lib.mog_metrics import evaluate as generation_metrics, sample_metrics
from lib.toy_metrics import sliced_w1
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, sample_100gaussians
from particlegan import GANLoss, GradientPenalty, MoGParticlePrior, ParticleRegularizer

DEFAULTS = {
    'posterior': 'categorical', 'gan_weight': 1., 'kl_weight': 1.,
    'obs_sigma': .03, 'temperature': .0025, 'seed': 24002,
    'steps': 6000, 'batch_size': 256, 'width': 128, 'num_particles': 400,
    'sigma_rel': .025, 'lr': .0003, 'd_lr': .00045, 'prior_lr': .003,
    'spread_weight': 1., 'log_interval': 250, 'eval_interval': 2000,
    'eval_samples': 100000, 'final_samples': 100000, 'recon_samples': 8192,
    'max_train_seconds': 300., 'out_dir': 'runs/mog_vae/stability/default',
}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(modules):
    h = hashlib.sha256()
    for module in modules:
        for k, v in module.state_dict().items():
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes()
                     if isinstance(v, torch.Tensor) else json.dumps(v, sort_keys=True).encode())
    return h.hexdigest()


def rng(seed):
    return torch.Generator(device='cuda').manual_seed(seed)


class Encoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = SimpleMLPGenerator(2, width, n_hidden=2, out_dim=6)
        nn.init.zeros_(self.net.net[-1].weight)
        nn.init.zeros_(self.net.net[-1].bias)

    def forward(self, x, means, temperature):
        scaled = x / math.sqrt(8.25 + .03**2)
        h = self.net(scaled)
        query = scaled + h[:, :2]
        logq = (-(query[:, None] - means[None]).square().sum(-1) / temperature).log_softmax(1)
        u = 3 * torch.tanh(h[:, 2:4] / 3)
        # Posterior std is relative to fixed prior sigma; starts at one.
        log_s = 3 * torch.tanh(h[:, 4:] / 3)
        return logq, u, log_s


def posterior_kl(logq, u, log_s, local):
    categorical = (logq.exp() * (logq + math.log(logq.shape[1]))).sum(1)
    continuous = .5 * (u.square() + (2 * log_s).exp() - 1 - 2 * log_s).sum(1)
    return categorical, continuous if local else torch.zeros_like(categorical)


def two_draw_loss(cost, selected_logq):
    """Cost [B,2]; other independent draw is a leave-one-out baseline.

    Value equals mean cost. Backward adds unbiased categorical score gradient;
    pathwise gradients of cost are retained. No straight-through categorical VAE.
    """
    advantage = (cost - cost.flip(1)).detach()
    score = (advantage * selected_logq).mean()
    return cost.mean() + score - score.detach()


def deterministic_code(e, x, prior, temperature):
    means = prior.means()
    logq, u, _ = e(x, means.detach(), temperature)
    ids = logq.argmax(1)
    proxy = logq.exp() @ means.detach()
    return means[ids] + proxy - proxy.detach() + prior.sigma * u, ids, u


def sampled_codes(e, x, prior, cfg, generator, draws):
    means = prior.means()
    hard = cfg['posterior'] == 'hard'
    logq, u, log_s = e(x, means.detach() if hard else means, cfg['temperature'])
    if hard:
        chosen = logq.argmax(1)
        proxy = logq.exp() @ means.detach()
        # Exact hard forward; only the query receives this biased surrogate.
        center = means[chosen] + (proxy - proxy.detach())
        ids = chosen[:, None].expand(-1, draws)
        centers = center[:, None]
        # Return the TRUE posterior distribution, not soft backward weights.
        logq = torch.nn.functional.one_hot(chosen, len(means)).to(x.dtype).log()
    else:
        ids = torch.multinomial(logq.exp(), draws, replacement=True, generator=generator)
        centers = means[ids]
    noise = torch.randn((len(x), draws, 2), device=x.device, dtype=x.dtype, generator=generator)
    local = cfg['posterior'] == 'local'
    offsets = u[:, None] + log_s.exp()[:, None] * noise if local else noise
    return centers + prior.sigma * offsets, ids, logq, u, log_s


def variational_kl(logq, u, log_s, posterior):
    if posterior == 'hard':
        categorical = torch.full_like(u[:, 0], math.log(logq.shape[1]))
        return categorical, torch.zeros_like(categorical)
    return posterior_kl(logq, u, log_s, posterior == 'local')


@torch.no_grad()
def evaluate(g, e, prior, cfg, out, step):
    n = cfg['final_samples'] if step == cfg['steps'] else cfg['eval_samples']
    m, _, fake, real = generation_metrics(g, prior, n, cfg['seed'])
    # Sparse pilot samples can leave the core-width statistic undefined.
    m = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in m.items()}
    m['sample_sw1'] = sliced_w1(fake[:8192], real[:8192], seed=cfg['seed'] + 10001)
    m['reconstruction'] = None
    if cfg['posterior'] == 'none':
        return m
    generator = rng(cfg['seed'] + 20000)
    x = sample_100gaussians(cfg['recon_samples'], torch.device('cuda'), generator=generator)
    errors, map_errors, shuffle_errors, center_errors = [], [], [], []
    qs, ids_all, kls, locals_, scales, offsets, variations, same = [], [], [], [], [], [], [], []
    all_outputs = []
    mode = lambda v: (v + 4.5).round().clamp(0, 9).long()
    for chunk in x.split(512):
        if cfg['posterior'] == 'ae':
            z, ids, u = deterministic_code(e, chunk, prior, cfg['temperature'])
            logq, _, log_s = e(chunk, prior.means(), cfg['temperature'])
            y = g(z)[:, None].expand(-1, 8, -1)
            map_y = y[:, 0]
            kc = kl = torch.zeros(len(chunk), device='cuda')
        else:
            z, ids, logq, u, log_s = sampled_codes(e, chunk, prior, cfg, generator, 8)
            y = g(z.flatten(0, 1)).reshape(len(chunk), 8, 2)
            kc, kl = variational_kl(logq, u, log_s, cfg['posterior'])
            center = prior.means()[logq.argmax(1)]
            if cfg['posterior'] == 'local':
                center = center + prior.sigma * u
            map_y = g(center)
        center = prior.means()[logq.argmax(1)]
        errors.append((y - chunk[:, None]).square().mean((1, 2)))
        map_errors.append((map_y - chunk).square().mean(1))
        shuffle_errors.append((y.roll(1, 0) - chunk[:, None]).square().mean((1, 2)))
        center_errors.append((g(center) - chunk).square().mean(1))
        qs.append(logq.exp()); ids_all.append(ids.reshape(-1)); kls.append(kc); locals_.append(kl)
        offsets.append(u if cfg['posterior'] in ('local', 'ae') else torch.zeros_like(u))
        scales.append(log_s.exp() if cfg['posterior'] == 'local' else torch.ones_like(log_s))
        # Average squared Euclidean distance across the 28 unordered draw pairs.
        variations.append(2 * y.var(1, unbiased=True).sum(1))
        same.append((mode(y) == mode(chunk)[:, None]).all(-1).float().mean(1))
        all_outputs.append(y[:min(len(chunk), 64)].cpu())
    errors, map_errors, shuffle_errors, center_errors, qs, ids_all, kls, locals_, scales, offsets, variations, same = [torch.cat(v) for v in
        (errors, map_errors, shuffle_errors, center_errors, qs, ids_all, kls, locals_, scales, offsets, variations, same)]
    agg = qs.mean(0)
    counts = torch.bincount(ids_all, minlength=prior.num_particles).float()
    freq = counts / counts.sum()
    aggregate_kl = float((agg * (agg.clamp_min(1e-30).log() + math.log(len(agg)))).sum())
    r = dict(recon_mse=float(errors.mean()), map_mse=float(map_errors.mean()),
             recon_p99=float(errors.quantile(.99)), shuffled_mse=float(shuffle_errors.mean()),
             zero_offset_mse=float(center_errors.mean()), same_mode=float(same.mean()),
             pair_rms=float(variations.mean().sqrt()), categorical_kl=float(kls.mean()),
             local_kl=float(locals_.mean()), aggregate_categorical_kl=aggregate_kl,
             categorical_mutual_information=float(kls.mean()) - aggregate_kl if cfg['posterior'] != 'ae' else None,
             posterior_effective_particles=float((-(qs * qs.clamp_min(1e-30).log()).sum(1)).mean().exp()),
             used_particles=int((counts > 0).sum()), effective_particles=float((-(freq * freq.clamp_min(1e-30).log()).sum()).exp()),
             aggregate_usage_tv=float((agg - 1 / len(agg)).abs().sum() / 2),
             posterior_std_ratio=float(scales.mean()) if cfg['posterior'] != 'ae' else 0.,
             offset_rms=float(offsets.square().mean().sqrt()), samples=len(x), draws=8)
    if cfg['posterior'] != 'ae':
        # Gaussian decoder likelihood in 2D: E[MSE]/tau^2 + log(2*pi*tau^2).
        r['negative_elbo_nats'] = r['recon_mse'] / cfg['obs_sigma']**2 + math.log(2 * math.pi * cfg['obs_sigma']**2) + r['categorical_kl'] + r['local_kl']
        # Distinguish G(z) benchmark samples from the actual Gaussian likelihood.
        noisy = fake + cfg['obs_sigma'] * torch.randn(fake.shape, device='cuda', generator=generator)
        pred, _, _ = sample_metrics(noisy)
        m['likelihood_predictive'] = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in pred.items()}
    else:
        r['negative_elbo_nats'] = None
    m['reconstruction'] = r
    np.savez_compressed(out / f'reconstruction_{step:06d}.npz',
        errors=errors.cpu().numpy(), map_errors=map_errors.cpu().numpy(),
        shuffle_errors=shuffle_errors.cpu().numpy(), center_errors=center_errors.cpu().numpy(),
        categorical_kl=kls.cpu().numpy(), local_kl=locals_.cpu().numpy(),
        pair_squared=variations.cpu().numpy(), same_mode=same.cpu().numpy(),
        counts=counts.cpu().numpy(), aggregate_q=agg.cpu().numpy(),
        inputs=x.cpu().numpy(), output_subset=torch.cat(all_outputs).numpy())
    return m


def validate(cfg):
    if set(cfg) != set(DEFAULTS) or cfg['posterior'] not in ('none', 'ae', 'categorical', 'local', 'hard'):
        raise ValueError('unknown configuration')
    for key in ('steps', 'batch_size', 'width', 'num_particles', 'log_interval', 'eval_interval', 'eval_samples', 'final_samples', 'recon_samples'):
        if type(cfg[key]) is not int or cfg[key] < 2:
            raise ValueError(f'invalid {key}')
    for key in ('obs_sigma', 'temperature', 'sigma_rel', 'lr', 'd_lr', 'prior_lr', 'max_train_seconds'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'invalid {key}')
    for key in ('gan_weight', 'kl_weight', 'spread_weight'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'invalid {key}')


def train(cfg):
    validate(cfg)
    started_all = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(cfg['out_dir']); out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise RuntimeError('refusing overwrite')
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True))
    provenance = code_provenance(__file__, sys.executable)
    write_json(out / 'provenance.json', provenance)
    with zipfile.ZipFile(out / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in provenance['sources']:
            archive.write(ROOT / name, arcname=name)
    g = SimpleMLPGenerator(2, cfg['width']).cuda()
    d = SimpleMLPDiscriminator(hidden_dim=cfg['width']).cuda()
    prior = MoGParticlePrior(num_particles=cfg['num_particles'], z_dim=2,
                            sigma_rel=cfg['sigma_rel'], generator=rng(cfg['seed'] + 1), device='cuda').cuda()
    e = Encoder(cfg['width']).cuda()
    initial_sigma = prior.sigma.detach().clone()
    metadata = dict(initialization_sha256=state_hash([g, d, prior, e]), sigma=float(prior.sigma),
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    evaluation_weights='final online weights; no checkpoint selection',
                    parameters={k: sum(p.numel() for p in m.parameters()) for k, m in [('G', g), ('D', d), ('E', e), ('prior', prior)]})
    write_json(out / 'metadata.json', metadata)
    og = torch.optim.Adam([{'params': g.parameters(), 'lr': cfg['lr']},
                          {'params': e.parameters(), 'lr': cfg['lr']},
                          {'params': prior.parameters(), 'lr': cfg['prior_lr'], 'betas': (.5, .999)}], betas=(0., .999))
    od = torch.optim.Adam(d.parameters(), lr=cfg['d_lr'], betas=(0., .999))
    adversarial, penalty, spread = GANLoss(), GradientPenalty(lazy_k=4), ParticleRegularizer()
    streams = {k: rng(cfg['seed'] + v) for k, v in [('data', 2), ('prior', 3), ('posterior', 4)]}
    print(f"START posterior={cfg['posterior']} steps={cfg['steps']} sigma={float(prior.sigma):.8g} init={metadata['initialization_sha256']}", flush=True)
    train_seconds = 0.
    torch.cuda.synchronize(); block_start = time.perf_counter()
    with (out / 'metrics.jsonl').open('w', buffering=1) as history:
        for step in range(1, cfg['steps'] + 1):
            x = sample_100gaussians(cfg['batch_size'], torch.device('cuda'), generator=streams['data'])
            # Always consume the same two prior draws, including VAE without GAN.
            z_d, _ = prior.sample(len(x), generator=streams['prior'])
            z_g, _ = prior.sample(len(x), generator=streams['prior'])
            dl = x.new_zeros(())
            if cfg['gan_weight']:
                d.requires_grad_(True); od.zero_grad(set_to_none=True)
                fake = g(z_d.detach()).detach()
                dl = adversarial.d_loss(d(x), d(fake)) + penalty(d, x, fake, step)
                dl.backward(); od.step()
            d.requires_grad_(False); og.zero_grad(set_to_none=True)
            gl = adversarial.g_loss(d(g(z_g)), d(x).detach()) if cfg['gan_weight'] else x.new_zeros(())
            rec = kc = kl = x.new_zeros(())
            reconstruction_loss = x.new_zeros(())
            if cfg['posterior'] == 'ae':
                z, _, _ = deterministic_code(e, x, prior, cfg['temperature'])
                rec = reconstruction_loss = (g(z) - x).square().mean()
            elif cfg['posterior'] != 'none':
                z, ids, logq, u, log_s = sampled_codes(e, x, prior, cfg, streams['posterior'], 2)
                cost = (g(z.flatten(0, 1)).reshape(len(x), 2, 2) - x[:, None]).square().mean(-1)
                reconstruction_loss = cost.mean() if cfg['posterior'] == 'hard' else two_draw_loss(cost, logq.gather(1, ids))
                rec = cost.mean()
                categorical, local = variational_kl(logq, u, log_s, cfg['posterior'])
                kc, kl = categorical.mean(), local.mean()
            # Hard posterior KL is log(K), independent of all learned parameters.
            # Omit it from optimization; retain it in the evaluated ELBO.
            kl_loss = x.new_zeros(()) if cfg['posterior'] == 'hard' else cfg['obs_sigma']**2 * cfg['kl_weight'] * (kc + kl)
            loss = cfg['gan_weight'] * gl + reconstruction_loss + kl_loss + cfg['spread_weight'] * spread(prior.z)
            loss.backward(); og.step()
            if step == 1 or step % cfg['log_interval'] == 0:
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                print(f"TRAIN step={step}/{cfg['steps']} d={float(dl.detach()):.4f} g={float(gl.detach()):.4f} mse={float(rec.detach()):.6f} kl_cat={float(kc.detach()):.4f} kl_local={float(kl.detach()):.4f}", flush=True)
                if train_seconds + time.perf_counter() - block_start > cfg['max_train_seconds']:
                    raise RuntimeError('training time budget exceeded')
            if step % cfg['eval_interval'] == 0 or step == cfg['steps']:
                torch.cuda.synchronize(); train_seconds += time.perf_counter() - block_start
                assert torch.equal(initial_sigma, prior.sigma)
                metrics = evaluate(g, e, prior, cfg, out, step)
                metrics.update(step=step, train_seconds=train_seconds)
                history.write(json.dumps(metrics, allow_nan=False) + '\n')
                print(f"EVAL step={step} modes={metrics['modes']} hq={metrics['hq']:.4f} sw1={metrics['sample_sw1']:.5f} recon={metrics['reconstruction']} train_s={train_seconds:.1f}", flush=True)
                torch.save(dict(config=cfg, step=step, G=g.state_dict(), D=d.state_dict(), E=e.state_dict(), prior=prior.state_dict(),
                                optimizer_g=og.state_dict(), optimizer_d=od.state_dict(), rng={k: v.get_state() for k, v in streams.items()}), out / f'checkpoint_{step:06d}.pt')
                torch.cuda.synchronize(); block_start = time.perf_counter()
    write_json(out / 'summary.json', dict(config=cfg, final=metrics, metadata=metadata,
        train_seconds=train_seconds, total_seconds=time.perf_counter() - started_all,
        peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30, sigma_unchanged=True,
        rng_sha256={k: digest_tensor(v.get_state()) for k, v in streams.items()},
        checkpoints={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('checkpoint_*.pt')}))
    print(f"COMPLETE {out} train_s={train_seconds:.1f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    user = read_config(args.config)
    if not isinstance(user, dict) or set(user) - set(DEFAULTS):
        raise ValueError('unknown config keys')
    train(merge_config(DEFAULTS, user))
