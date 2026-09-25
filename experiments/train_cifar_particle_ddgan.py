#!/usr/bin/env python
"""Matched direct/DDGAN fixed-sigma particle autoencoder queue entry point."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.config import merge_config, read_config
from experiments.run_grid import code_provenance
from lib.image_ddgan import update_ema
from lib.image_particle_autoencoder import DirectGenerator, DirectDiscriminator, ImageRoutingEncoder
from lib.image_particle_ddgan import ParticleDDGenerator, ParticleDDDiscriminator
from particlegan import calibrate_mog_sigma, GANLoss, GradientPenalty, MoGParticlePrior, ParticleRegularizer

DEFAULTS = {
    'arm': 'gan', 'model': 'direct',
    'alpha_bar': [1., .9, .5, .05, .0001], 'seed': 24002, 'steps': 10000, 'batch_size': 64,
    'z_dim': 64, 'num_particles': 1024, 'width': 32, 'sigma_rel': .025,
    'temperature': .125, 'recon_weight': 1., 'lr': .0003, 'prior_lr': .003,
    'd_lr': .00045, 'ema': .995, 'log_interval': 100, 'eval_interval': 2500,
    'eval_samples': 5000, 'final_samples': 50000, 'recon_samples': 10000,
    'eval_batch_size': 128, 'max_train_seconds': 1800.,
    'data_dir': '/home/martyn/dev/ParticleGAN/data',
    'fid_cache': '/home/martyn/dev/ParticleGAN/results/cifar_ddgan/fid_cache',
    'out_dir': 'runs/cifar_particle_ddgan/default',
}


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def state_hash(modules):
    digest = hashlib.sha256()
    for i, module in enumerate(modules):
        for key, value in module.state_dict().items():
            digest.update(f'{i}:{key}'.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes()
                          if isinstance(value, torch.Tensor) else json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def rng(seed, device='cuda'):
    return torch.Generator(device=device).manual_seed(seed)


def validate(cfg):
    if set(cfg) != set(DEFAULTS) or cfg['arm'] not in ('gan', 'bounded'):
        raise ValueError('unknown configuration or arm')
    if cfg['model'] not in ('direct', 'ddgan'):
        raise ValueError('unknown model')
    from particlegan import DDGAN
    DDGAN(cfg['alpha_bar'])
    for key in ('steps', 'batch_size', 'z_dim', 'num_particles', 'width', 'log_interval',
                'eval_interval', 'recon_samples', 'eval_batch_size'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['width'] % 8 or cfg['z_dim'] < 2 or cfg['num_particles'] < 2 or cfg['recon_samples'] > 10000:
        raise ValueError('invalid model dimensions or held-out sample count')
    for key in ('eval_samples', 'final_samples'):
        if type(cfg[key]) is not int or cfg[key] < 0 or cfg[key] == 1:
            raise ValueError('FID sample counts must be zero (pilot) or at least two')
    for key in ('temperature', 'recon_weight', 'sigma_rel', 'lr', 'prior_lr', 'd_lr', 'max_train_seconds'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0 <= cfg['ema'] < 1:
        raise ValueError('invalid EMA decay')


@torch.no_grad()
def reconstruction(g, e, prior, images, cfg, out, step):
    """Test-split diagnostics with independent RNG, including ablations and tails."""
    from torchvision.utils import save_image
    batch = cfg['eval_batch_size']
    means = prior.means()
    codes, selected, offsets, soft_sum = [], [], [], torch.zeros(len(means), device='cuda')
    for raw in images.split(batch):
        x = raw.cuda().float() / 127.5 - 1
        z, ids, u, soft = e(x, means, prior.sigma, cfg['temperature'])
        codes.append(z); selected.append(ids); offsets.append(u)
        soft_sum += soft.sum(0)
    codes, selected, offsets = torch.cat(codes), torch.cat(selected), torch.cat(offsets)
    generator = rng(cfg['seed'] + 20000)
    noise = torch.randn(codes.shape, device='cuda', generator=generator)
    perm = torch.randperm(len(codes), device='cuda', generator=generator)
    variants = {'recon': codes, 'zero_offset': means[selected],
                'random_offset': means[selected] + prior.sigma * noise,
                'shuffled_particle': means[selected[perm]] + prior.sigma * offsets}
    errors, display = {}, []
    for name, values in variants.items():
        chunks = []
        for lo in range(0, len(images), batch):
            x = images[lo:lo + batch].cuda().float() / 127.5 - 1
            y = g(values[lo:lo + batch])
            chunks.append((y - x).square().flatten(1).mean(1).cpu())
            if lo == 0:
                display.append(y[:16].cpu())
        errors[name] = torch.cat(chunks)
    counts = torch.bincount(selected, minlength=len(means)).float()
    p = counts / counts.sum()
    ps = soft_sum / len(images)
    metrics = {f'{name}_mse': float(error.mean()) for name, error in errors.items()}
    metrics.update(recon_psnr=10 * math.log10(4 / metrics['recon_mse']),
                   recon_p90=float(errors['recon'].quantile(.9)),
                   recon_p99=float(errors['recon'].quantile(.99)),
                   used_particles=int((counts > 0).sum()),
                   effective_particles=float((-(p * p.clamp_min(1e-30).log()).sum()).exp()),
                   hard_usage_tv=float((p - 1 / len(p)).abs().sum() / 2),
                   hard_soft_usage_tv=float((p - ps).abs().sum() / 2),
                   offset_rms=float(offsets.square().mean().sqrt()),
                   offset_saturation=float((offsets.abs() > 2.9).float().mean()),
                   samples=len(images))
    sums = torch.zeros_like(means).index_add_(0, selected, offsets)
    conditional = sums / counts.clamp_min(1)[:, None]
    metrics['conditional_offset_mean_rms'] = float((conditional.square().mean(1) * p).sum().sqrt())
    reference = images[:16].float() / 127.5 - 1
    # Columns are input, reconstruction, zero, random, shuffled; rows are examples.
    grid = torch.stack([reference, *display], 1).flatten(0, 1)
    save_image(grid, out / f'recon_{step:06d}.png', nrow=5, normalize=True, value_range=(-1, 1))
    np.savez_compressed(out / f'recon_{step:06d}.npz',
                        **{k: v.numpy() for k, v in errors.items()},
                        ids=selected.cpu().numpy(), counts=counts.cpu().numpy())
    return metrics


@torch.no_grad()
def diffusion_reconstruction(g, e, prior, images, cfg, out, step):
    """One-step clean prediction, with identical noisy inputs across ablations.

    This is denoising with a side input, not direct latent-only reconstruction.
    Report each noise level separately; never pool it into direct AE MSE.
    """
    batch = cfg['eval_batch_size']
    means = prior.means()
    codes, selected, offsets = [], [], []
    if cfg['arm'] == 'bounded':
        for raw in images.split(batch):
            z, ids, u, _ = e(raw.cuda().float() / 127.5 - 1, means, prior.sigma, cfg['temperature'])
            codes.append(z); selected.append(ids); offsets.append(u)
        codes, selected, offsets = torch.cat(codes), torch.cat(selected), torch.cat(offsets)
    generator = rng(cfg['seed'] + 20000)
    perm = torch.randperm(len(images), device='cuda', generator=generator)
    random_z, _ = prior.sample(len(images), generator)
    variants = {'prior': random_z}
    result = {'samples': len(images), 'per_t': {}}
    arrays = {}
    if cfg['arm'] == 'bounded':
        variants.update(recon=codes, shuffled_code=codes[perm], zero_offset=means[selected],
                        shuffled_particle=means[selected[perm]] + prior.sigma * offsets)
        counts = torch.bincount(selected, minlength=len(means)).float()
        p = counts / counts.sum()
        result.update(used_particles=int((counts > 0).sum()),
                      effective_particles=float((-(p * p.clamp_min(1e-30).log()).sum()).exp()),
                      offset_rms=float(offsets.square().mean().sqrt()),
                      offset_saturation=float((offsets.abs() > 2.9).float().mean()))
        arrays.update(ids=selected.cpu().numpy(), counts=counts.cpu().numpy())
    for time_index in range(1, g.schedule.steps + 1):
        errors = {name: [] for name in variants}
        for lo in range(0, len(images), batch):
            x = images[lo:lo + batch].cuda().float() / 127.5 - 1
            t = torch.full((len(x),), time_index, device='cuda', dtype=torch.long)
            _, xt = g.schedule.forward_pair(x, t, generator)
            for name, z in variants.items():
                y = g(z[lo:lo + batch], xt, t)
                errors[name].append((y - x).square().flatten(1).mean(1).cpu())
        errors = {k: torch.cat(v) for k, v in errors.items()}
        result['per_t'][str(time_index)] = {f'{k}_mse': float(v.mean()) for k, v in errors.items()}
        arrays.update({f't{time_index}_{k}': v.numpy() for k, v in errors.items()})
    np.savez_compressed(out / f'recon_{step:06d}.npz', **arrays)
    return result


@torch.no_grad()
def generation(g, prior, evaluator, cfg, n, out, step):
    from lib.cifar_metrics import uint8_images
    from torch_fidelity.metric_fid import fid_statistics_to_metric
    from torchvision.utils import save_image
    generator = rng(cfg['seed'] + 10000)
    chunks = []
    count = n or 100
    for lo in range(0, count, cfg['eval_batch_size']):
        size = min(cfg['eval_batch_size'], count - lo)
        if cfg['model'] == 'ddgan':
            x = g.sample(prior, size, generator)
        else:
            z, _ = prior.sample(size, generator)
            x = g(z)
        chunks.append(uint8_images(x).cpu())
    images = torch.cat(chunks)
    save_image(images[:100].float() / 255, out / f'samples_{step:06d}.png', nrow=10)
    if not n:
        return {'samples': 0, 'fid': None, 'feature_variance_ratio': None}
    stats = evaluator.statistics(images)
    fid5k = None
    if n >= 5000:
        small = evaluator.statistics(images[:5000]) if n != 5000 else stats
        fid5k = float(fid_statistics_to_metric(small, evaluator.real, verbose=False)['frechet_inception_distance'])
    return {'fid5k': fid5k, 'samples': n, 'fid': float(fid_statistics_to_metric(stats, evaluator.real, verbose=False)['frechet_inception_distance']),
            'feature_variance_ratio': float(np.trace(stats['sigma']) / np.trace(evaluator.real['sigma']))}


def train(cfg):
    validate(cfg)
    from torchvision.datasets import CIFAR10
    from lib.cifar_metrics import FIDEvaluator, PROTOCOL
    started_all = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    out = ROOT / cfg['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.jsonl').exists():
        raise FileExistsError('use a fresh output directory; the grid runner archives previous attempts')
    provenance = code_provenance(__file__, sys.executable)
    write_json(out / 'provenance.json', provenance)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg))
    with zipfile.ZipFile(out / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, digest in provenance['sources'].items():
            content = (ROOT / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == digest
            archive.writestr(name, content)
    train_data = CIFAR10(cfg['data_dir'], train=True, download=False)
    test_data = CIFAR10(cfg['data_dir'], train=False, download=False)
    images = torch.from_numpy(train_data.data).permute(0, 3, 1, 2).contiguous()
    test_images = torch.from_numpy(test_data.data[:cfg['recon_samples']]).permute(0, 3, 1, 2).contiguous()
    evaluator = FIDEvaluator(images, cfg['fid_cache'], cfg['eval_batch_size']) if cfg['final_samples'] or cfg['eval_samples'] else None
    images = images.cuda()
    # Build all modules in both arms, in the same order and on CPU before transfer.
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    if cfg['model'] == 'ddgan':
        g = ParticleDDGenerator(cfg['z_dim'], cfg['width'], cfg['alpha_bar']).cuda()
        d = ParticleDDDiscriminator(cfg['width'], cfg['z_dim'], cfg['alpha_bar']).cuda()
    else:
        g = DirectGenerator(cfg['z_dim'], cfg['width']).cuda()
        d = DirectDiscriminator(cfg['width']).cuda()
    e = ImageRoutingEncoder(cfg['z_dim'], cfg['width']).cuda()
    prior = MoGParticlePrior(num_particles=cfg['num_particles'], z_dim=cfg['z_dim'],
                             sigma=0, generator=rng(cfg['seed'] + 1, 'cpu'))
    sigma, d0 = calibrate_mog_sigma(prior.means(), cfg['sigma_rel'])
    prior.set_sigma(sigma)
    prior.d0.copy_(d0)
    prior.sigma_rel = cfg['sigma_rel']
    prior = prior.cuda()
    initial_hash = state_hash([g, d, e, prior])
    initial_sigma = prior.sigma.clone()
    initial_prior = prior.z.detach().clone()
    initial_features = state_hash([d.critic.features])
    eg, ee, ep = [copy.deepcopy(m).eval().requires_grad_(False) for m in (g, e, prior)]
    og = torch.optim.Adam([
        {'params': g.parameters(), 'lr': cfg['lr']},
        {'params': e.parameters(), 'lr': cfg['lr']},
        {'params': prior.parameters(), 'lr': cfg['prior_lr'], 'betas': (.5, .999)},
    ], betas=(0., .999), fused=True)
    od = torch.optim.Adam([p for p in d.parameters() if p.requires_grad], lr=cfg['d_lr'], betas=(0., .999), fused=True)
    adversarial, penalty, spread = GANLoss(), GradientPenalty(arm="b_cap", lazy_k=4), ParticleRegularizer()
    streams = {name: rng(cfg['seed'] + offset) for name, offset in [('data', 2), ('prior', 3), ('time', 4), ('corruption', 5), ('reverse', 6)]}
    metadata = {'initialization_sha256': initial_hash, 'sigma': float(prior.sigma),
                'initial_nearest_neighbor_median': float(prior.d0), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'python': platform.python_version(),
                'gpu': torch.cuda.get_device_name(), 'pretrained': d.critic.pretrained_metadata,
                'parameters': {name: sum(p.numel() for p in m.parameters()) for name, m in [('G', g), ('D', d), ('E', e), ('prior', prior)]},
                'fid_protocol': PROTOCOL, 'reconstruction_split': 'CIFAR-10 test, unaugmented',
                'evaluation_weights': 'EMA G/E/prior; final checkpoint, no best selection'}
    write_json(out / 'metadata.json', metadata)
    print(f"START arm={cfg['arm']} steps={cfg['steps']} sigma={float(prior.sigma):.6f} init={initial_hash}", flush=True)

    def batch():
        ids = torch.randint(len(images), (cfg['batch_size'],), device='cuda', generator=streams['data'])
        x = images[ids].float() / 127.5 - 1
        flip = torch.rand((len(x), 1, 1, 1), device='cuda', generator=streams['data']) < .5
        return torch.where(flip, x.flip(-1), x)

    def context(x):
        if cfg['model'] == 'direct':
            return x, None, None
        t = torch.randint(1, g.schedule.steps + 1, (len(x),), device='cuda', generator=streams['time'])
        real, xt = g.schedule.forward_pair(x, t, streams['corruption'])
        return real, xt, t

    def decode(z, xt, t):
        return g(z) if xt is None else g(z, xt, t)

    def transition(clean, xt, t):
        if xt is None:
            return clean
        eta = torch.randn(xt.shape, device='cuda', generator=streams['reverse'])
        return g.schedule.reverse(clean, xt, t, eta)

    train_seconds = 0.
    torch.cuda.synchronize()
    block_start = time.perf_counter()
    with (out / 'metrics.jsonl').open('w', buffering=1) as log:
        for step in range(1, cfg['steps'] + 1):
            d.requires_grad_(True)
            x = batch()
            real, xt, t = context(x)
            critic = d if xt is None else d.conditioned(xt, t)
            with torch.no_grad():
                z, _ = prior.sample(cfg['batch_size'], streams['prior'])
                fake = transition(decode(z, xt, t), xt, t)
            od.zero_grad(set_to_none=True)
            dp = penalty(critic, real, fake, step)
            dl = adversarial.d_loss(critic(real), critic(fake)) + dp
            dl.backward()
            od.step()
            d.requires_grad_(False)
            x = batch()
            real, xt, t = context(x)
            critic = d if xt is None else d.conditioned(xt, t)
            og.zero_grad(set_to_none=True)
            z, _ = prior.sample(cfg['batch_size'], streams['prior'])
            with torch.no_grad():
                dr = critic(real)
            gl = adversarial.g_loss(critic(transition(decode(z, xt, t), xt, t)), dr)
            rec = gl.new_zeros(())
            if cfg['arm'] == 'bounded':
                encoded, _, _, _ = e(x, prior.means(), prior.sigma, cfg['temperature'])
                rec = (decode(encoded, xt, t) - x).square().mean()
            loss = gl + cfg['recon_weight'] * rec + spread(prior.z)
            loss.backward()
            og.step()
            for target, source in ((eg, g), (ee, e), (ep, prior)):
                update_ema(target, source, cfg['ema'])
            report = step % cfg['log_interval'] == 0 or step == 1 or step == cfg['steps']
            evaluate = step % cfg['eval_interval'] == 0 or step == cfg['steps']
            if report or evaluate:
                torch.cuda.synchronize()
                train_seconds += time.perf_counter() - block_start
                row = {'step': step, 'd_loss': float(dl.detach()), 'g_loss': float(gl.detach()),
                       'recon_train_mse': float(rec.detach()), 'penalty': float(dp.detach()),
                       'train_seconds': train_seconds, 'steps_per_second': step / train_seconds,
                       'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30}
                print(f"step={step}/{cfg['steps']} D={row['d_loss']:.4f} G={row['g_loss']:.4f} recon={row['recon_train_mse']:.5f} train_s={train_seconds:.1f} steps/s={row['steps_per_second']:.2f} peak_GB={row['peak_memory_gb']:.2f}", flush=True)
                if not all(math.isfinite(row[k]) for k in ('d_loss', 'g_loss', 'recon_train_mse')):
                    raise RuntimeError('nonfinite training loss')
                if train_seconds > cfg['max_train_seconds']:
                    raise RuntimeError('training time cap reached; no completed-run summary will be emitted')
                if evaluate:
                    assert torch.equal(prior.sigma, initial_sigma)
                    n = cfg['final_samples'] if step == cfg['steps'] else cfg['eval_samples']
                    print(f'EVAL step={step} generated_samples={n}', flush=True)
                    row['generation'] = generation(eg, ep, evaluator, cfg, n, out, step)
                    row['reconstruction'] = (diffusion_reconstruction(eg, ee, ep, test_images, cfg, out, step)
                                             if cfg['model'] == 'ddgan' else
                                             reconstruction(eg, ee, ep, test_images, cfg, out, step)
                                             if cfg['arm'] == 'bounded' else None)
                    print(json.dumps({'evaluation': row}, allow_nan=False), flush=True)
                    torch.save({'config': cfg, 'sources': provenance['sources'], 'step': step,
                                'G': g.state_dict(), 'D': d.state_dict(), 'E': e.state_dict(),
                                'prior': prior.state_dict(), 'ema_G': eg.state_dict(), 'ema_E': ee.state_dict(),
                                'ema_prior': ep.state_dict(), 'optimizer_g': og.state_dict(), 'optimizer_d': od.state_dict(),
                                'rng': {k: v.get_state() for k, v in streams.items()},
                                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                                'initialization_sha256': initial_hash, 'train_seconds': train_seconds}, out / f'checkpoint_{step:06d}.pt')
                    latest = out / 'checkpoint.pt'
                    latest.unlink(missing_ok=True)
                    latest.symlink_to(f'checkpoint_{step:06d}.pt')
                log.write(json.dumps(row, allow_nan=False) + '\n')
                torch.cuda.synchronize()
                block_start = time.perf_counter()
    assert initial_features == state_hash([d.critic.features]), 'frozen feature extractor changed'
    assert torch.equal(prior.sigma, initial_sigma) and torch.equal(ep.sigma, initial_sigma)
    summary = {'config': cfg, 'final': {'step': step, **row['generation'], 'reconstruction': row['reconstruction']},
               'train_seconds': train_seconds, 'total_seconds': time.perf_counter() - started_all,
               'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30,
               'prior_rms_movement': float((prior.z.detach() - initial_prior).square().mean().sqrt()),
               'rng_sha256': {k: hashlib.sha256(v.get_state().cpu().numpy().tobytes()).hexdigest() for k, v in streams.items()},
               'metadata': metadata, 'frozen_features_unchanged': True, 'sigma_unchanged': True}
    write_json(out / 'summary.json', summary)
    print(f"COMPLETE arm={cfg['arm']} fid={summary['final']['fid']} train_s={train_seconds:.1f} total_s={summary['total_seconds']:.1f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    user = read_config(args.config)
    if not isinstance(user, dict) or set(user) - set(DEFAULTS):
        raise ValueError('unknown config keys')
    train(merge_config(DEFAULTS, user))
