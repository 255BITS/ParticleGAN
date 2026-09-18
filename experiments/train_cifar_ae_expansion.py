#!/usr/bin/env python
"""Checkpoint continuations testing additional independently trainable centers.

Standalone G-only learning-rate intervention preserves historical certificates and the joint
update ordering. New interventions are explicit, independently selectable and
recorded with parent/delta hashes. No architecture change.
"""
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
from lib.image_ddgan import ResBlock
from lib.image_particle_autoencoder import DirectGenerator, DirectDiscriminator, build_encoder
from particlegan import GANLoss, GradientPenalty, MoGParticlePrior, ParticleRegularizer

class ExpandedPrior(MoGParticlePrior):
    """Cloned rows with reference-count normalization and separate child RNG.

    Construct a normal prior first, then switch class and resize after calibration.
    num_particles in config remains the reference initialization count.
    """
    def means(self):
        n, m = self.reference_count, self.num_particles
        if n == m:
            return super().means()
        correction = math.sqrt(n * (m - 1) / (m * (n - 1)))
        return (self.z - self.z.mean(0)) / (self.z.std(0) * correction + 1e-6)

    def sample(self, batch_size, generator=None, *, fixed_first_n=False, offset=0, eps=None):
        if fixed_first_n:
            return super().sample(batch_size, generator, fixed_first_n=True, offset=offset, eps=eps)
        parents = torch.randint(self.reference_count, (batch_size,), device=self.z.device, generator=generator)
        children = torch.randint(self.num_particles // self.reference_count, (batch_size,), device=self.z.device, generator=self.clone_rng)
        ids = parents * (self.num_particles // self.reference_count) + children
        if self.track_exposure:
            self.exposure.index_add_(0, ids, torch.ones_like(ids))
        return self(ids, generator=generator, eps=eps), ids

    def get_extra_state(self):
        return {**super().get_extra_state(), 'reference_count': self.reference_count,
                'clone_rng': self.clone_rng.get_state().tolist(), 'exposure': self.exposure.cpu().tolist()}

    def set_extra_state(self, state):
        super().set_extra_state(state)
        self.reference_count = state['reference_count']
        self.clone_rng.set_state(torch.tensor(state['clone_rng'], dtype=torch.uint8))
        self.exposure.copy_(torch.tensor(state['exposure'], device=self.z.device))


class ReferenceRegularizer(ParticleRegularizer):
    def __init__(self, reference_count):
        super().__init__()
        self.reference_count = reference_count

    def forward(self, z):
        n, m = self.reference_count, len(z)
        if n == m:
            return super().forward(z)
        # Correct both sample variance and covariance before epsilon/hinge.
        correction = n * (m - 1) / (m * (n - 1))
        std = (z.var(0) * correction + self.eps).sqrt()
        centered = z - z.mean(0)
        cov = (centered.T @ centered) / (m - 1) * correction
        d = z.shape[1]
        off = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        return self.weight * ((self.target_std - std).relu().mean() + off.square().sum() / d)


def activate_expansion(prior, factor, seed, track):
    prior.__class__ = ExpandedPrior
    prior.reference_count = prior.num_particles
    prior.z.data = prior.z.data.repeat_interleave(factor, 0)
    prior.clone_rng = torch.Generator(device=prior.z.device).manual_seed(seed)
    prior.exposure = torch.zeros(prior.num_particles, dtype=torch.long, device=prior.z.device)
    prior.track_exposure = track


def prepare_expansion(ck, prior, ema_prior, cfg):
    factor = cfg['expansion_factor']
    old_factor = ck['config'].get('expansion_factor', 1)
    if factor == 1:
        assert old_factor == 1
        return {'factor': 1, 'intervention': False}
    assert old_factor in (1, factor), 'only fresh expansion or unchanged expanded resume supported'
    audit = {'factor': factor, 'reference_count': cfg['num_particles'], 'intervention': old_factor == 1,
             'noise_recalibrated': False, 'adam_lr_compensation': False}
    for name, module in [('prior', prior), ('ema_prior', ema_prior)]:
        original = copy.deepcopy(module)
        if old_factor == 1:
            original.load_state_dict(ck[name])
        activate_expansion(module, factor, cfg['seed'] + 90000, name == 'prior')
        if old_factor == 1:
            ck[name]['z'] = ck[name]['z'].repeat_interleave(factor, 0)
            ck[name]['_extra_state'] = {**original.get_extra_state(),
                                      **{k:v for k,v in module.get_extra_state().items() if k not in ('sigma_rel', 'standardize')}}
            module.load_state_dict(ck[name])
            error = float((module.means() - original.means().repeat_interleave(factor, 0)).abs().max())
            assert error < 2e-6, error
            assert torch.equal(module.sigma, original.sigma) and torch.equal(module.d0, original.d0)
            z = original.z.detach().clone().requires_grad_(True)
            expanded = z.detach().repeat_interleave(factor, 0).requires_grad_(True)
            a, b = ParticleRegularizer()(z), ReferenceRegularizer(len(z))(expanded)
            ga, gb = torch.autograd.grad(a, z)[0], torch.autograd.grad(b, expanded)[0].reshape(len(z), factor, -1).sum(1)
            audit[name] = {'max_center_error': error, 'regularizer_original': float(a), 'regularizer_expanded': float(b),
                           'regularizer_gradient_max_error': float((ga-gb).abs().max()),
                           'sigma': float(module.sigma), 'd0': float(module.d0)}
            assert abs(float(a-b)) < 2e-6 and float((ga-gb).abs().max()) < 2e-6
    if old_factor == 1:
        group = ck['optimizer_g']['param_groups'][2]
        assert len(group['params']) == 1
        state = ck['optimizer_g']['state'][group['params'][0]]
        for key in ('exp_avg', 'exp_avg_sq'):
            state[key] = state[key].repeat_interleave(factor, 0)
        audit['prior_adam_step'] = float(state['step'])
    return audit


def exposure_summary(prior):
    if not hasattr(prior, 'exposure'):
        return {'tracked': False}
    x = prior.exposure.float()
    return {'tracked': True, 'scope': 'since expansion for split; this continuation for control', 'total': int(x.sum()), 'min': int(x.min()), 'max': int(x.max()),
            'mean': float(x.mean()), 'zero': int((x == 0).sum())}


@torch.no_grad()
def descendant_panel(g, prior, cfg, out, step, evaluator):
    from torchvision.utils import save_image
    from lib.cifar_metrics import uint8_images
    factor = cfg['expansion_factor']
    stream = rng(cfg['seed'] + 82000)
    panel_count = min(32, cfg['num_particles'])
    parents = torch.randperm(cfg['num_particles'], device='cuda', generator=stream)[:panel_count]
    means = prior.means().reshape(cfg['num_particles'], factor, cfg['z_dim'])[parents]
    means = means.expand(-1, 4, -1) if factor == 1 else means
    # Coupled noise across siblings isolates center effects. Four independent
    # draws per child separately measure local noise variability.
    eps = torch.randn(panel_count, 1, 4, cfg['z_dim'], device='cuda', generator=stream)
    z = means[:, :, None] + prior.sigma * eps
    pixels = torch.cat([uint8_images(g(x)).cpu() for x in z.flatten(0, 2).split(cfg['eval_batch_size'])])
    shaped = pixels.reshape(panel_count, 4, 4, 3, 32, 32)
    save_image(shaped[:8].flatten(0, 2).float()/255, out/f'siblings_{step:06d}.png', nrow=16)
    latent = means - means.mean(1, keepdim=True)
    result = {'parent_count': panel_count, 'children': 4, 'draws_per_child': 4,
              'sibling_latent_rms': float(latent.square().mean().sqrt()),
              'sigma': float(prior.sigma), 'note': 'Coupled noise across siblings; differences are not semantic coverage.'}
    def variation(x):
        x = x.double().reshape(panel_count, 4, 4, -1)
        return {'between_sibling_mse_coupled_noise': float((x-x.mean(1, keepdim=True)).square().mean()),
                'within_child_noise_mse': float((x-x.mean(2, keepdim=True)).square().mean())}
    result['pixel'] = variation(pixels.float()/255)
    if evaluator is not None:
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            features = torch.cat([evaluator.model(x.cuda())[0].cpu() for x in pixels.split(cfg['eval_batch_size'])])
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn
        result['inception'] = variation(features)
    return result


DEFAULTS = {
    'arm': 'gan', 'seed': 24002, 'steps': 10000, 'batch_size': 64,
    'z_dim': 64, 'num_particles': 1024, 'width': 32, 'sigma_rel': .025,
    'temperature': .125, 'recon_weight': 1., 'lr': .0006, 'prior_lr': .006,
    'd_lr': .0009, 'ema': .995, 'log_interval': 100, 'eval_interval': 2500,
    'eval_samples': 5000, 'final_samples': 50000, 'recon_samples': 10000,
    'eval_batch_size': 128, 'max_train_seconds': 1800.,
    'data_dir': '/home/martyn/dev/ParticleGAN/data',
    'fid_cache': '/home/martyn/dev/ParticleGAN/results/cifar_ddgan/fid_cache',
    'out_dir': 'runs/cifar_particle_ae/default',
    'encoder_backbone': 'scratch', 'keep_checkpoints': False, 'reg_every': 4,
    'g_width': 0, 'g_depth': 1, 'resume_checkpoint': '', 'resume_sha256': '',
    'expansion_factor': 1, 'initial_eval_samples': 0,
    'g_lr_scale': 1., 'lr_scale': 1., 'lr_final_ratio': 1., 'lr_decay_start': 0, 'lr_decay_end': 200000,
    'd_updates': 1, 'generator_arch': 'transgan', 'recon_grad': 'all',
    'transgan_dim': 512, 'transgan_depths': [5, 4, 2],
    'transgan_heads': 4, 'transgan_mlp_ratio': 4.,
    'transgan_rgb_gain': .1,
    'penalty_coeff': 1., 'd_warmstart': '', 'd_warmstart_sha256': '',
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
    if type(cfg['expansion_factor']) is not int or cfg['expansion_factor'] not in (1, 4):
        raise ValueError('expansion_factor must be 1 or 4')
    if type(cfg['initial_eval_samples']) is not int or cfg['initial_eval_samples'] < 0:
        raise ValueError('invalid initial_eval_samples')
    if cfg['expansion_factor'] > 1 and (not cfg['resume_checkpoint'] or cfg['recon_grad'] != 'encoder_only'):
        raise ValueError('expansion requires checkpoint and E-only reconstruction')
    if set(cfg) != set(DEFAULTS) or cfg['arm'] not in ('gan', 'bounded'):
        raise ValueError('unknown configuration or arm')
    for key in ('steps', 'batch_size', 'z_dim', 'num_particles', 'width', 'log_interval',
                'eval_interval', 'recon_samples', 'eval_batch_size', 'reg_every', 'g_depth', 'd_updates', 'lr_decay_end'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['g_width']) is not int or cfg['g_width'] < 0 or cfg['g_width'] % 8:
        raise ValueError('g_width must be zero (inherit width) or a positive multiple of 8')
    if type(cfg['resume_checkpoint']) is not str or type(cfg['resume_sha256']) is not str:
        raise ValueError('resume path/hash must be strings')
    if bool(cfg['resume_checkpoint']) != bool(cfg['resume_sha256']):
        raise ValueError('resume_checkpoint and resume_sha256 must be provided together')
    if cfg['width'] % 8 or cfg['z_dim'] < 2 or cfg['num_particles'] < 2 or cfg['recon_samples'] > 10000:
        raise ValueError('invalid model dimensions or held-out sample count')
    for key in ('eval_samples', 'final_samples'):
        if type(cfg[key]) is not int or cfg[key] < 0 or cfg[key] == 1:
            raise ValueError('FID sample counts must be zero (pilot) or at least two')
    for key in ('temperature', 'sigma_rel', 'lr', 'prior_lr', 'd_lr', 'max_train_seconds'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('recon_weight', 'penalty_coeff'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('g_lr_scale', 'lr_scale', 'lr_final_ratio'):
        if not math.isfinite(cfg[key]) or not 0 < cfg[key] <= 1:
            raise ValueError(f'{key} must be in (0, 1]')
    if type(cfg['lr_decay_start']) is not int or not 0 <= cfg['lr_decay_start'] < cfg['lr_decay_end']:
        raise ValueError('invalid global learning-rate schedule')
    if cfg['generator_arch'] not in ('cnn', 'transgan'):
        raise ValueError('generator_arch must be cnn or transgan')
    if cfg['recon_grad'] not in ('all', 'encoder_only'):
        raise ValueError('recon_grad must be all or encoder_only')
    for key in ('transgan_dim', 'transgan_heads'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['transgan_dim'] % (16 * cfg['transgan_heads']):
        raise ValueError('transgan_dim must be divisible by 16 * transgan_heads')
    depths = cfg['transgan_depths']
    if type(depths) is not list or len(depths) != 3 or any(type(d) is not int or d <= 0 for d in depths):
        raise ValueError('transgan_depths must contain three positive integers')
    if not math.isfinite(cfg['transgan_mlp_ratio']) or cfg['transgan_mlp_ratio'] < 1:
        raise ValueError('transgan_mlp_ratio must be finite and at least one')
    if not math.isfinite(cfg['transgan_rgb_gain']) or cfg['transgan_rgb_gain'] <= 0:
        raise ValueError('transgan_rgb_gain must be finite and positive')
    if cfg['generator_arch'] == 'transgan' and (cfg['g_width'] != 0 or cfg['g_depth'] != 1):
        raise ValueError('CNN g_width/g_depth overrides do not apply to transgan')
    if not 0 <= cfg['ema'] < 1:
        raise ValueError('invalid EMA decay')
    if cfg['encoder_backbone'] not in ('scratch', 'pretrained_resnet18'):
        raise ValueError('invalid encoder_backbone')
    if cfg['d_warmstart'] and not cfg['resume_checkpoint']:
        raise ValueError('D warmstart requires a parent checkpoint')
    if bool(cfg['d_warmstart']) != bool(cfg['d_warmstart_sha256']):
        raise ValueError('warmstart path and hash must be paired')
    if type(cfg['keep_checkpoints']) is not bool:
        raise ValueError('keep_checkpoints must be boolean')


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
def generation(g, prior, evaluator, cfg, n, out, step):
    from lib.cifar_metrics import uint8_images
    from torch_fidelity.metric_fid import fid_statistics_to_metric
    from torchvision.utils import save_image
    clone_state = prior.clone_rng.get_state() if hasattr(prior, 'clone_rng') else None
    if clone_state is not None: prior.clone_rng.manual_seed(cfg['seed'] + 91000)
    generator = rng(cfg['seed'] + 10000)
    chunks = []
    count = n or 100
    for lo in range(0, count, cfg['eval_batch_size']):
        z, _ = prior.sample(min(cfg['eval_batch_size'], count - lo), generator)
        chunks.append(uint8_images(g(z)).cpu())
    if clone_state is not None: prior.clone_rng.set_state(clone_state)
    images = torch.cat(chunks)
    save_image(images[:100].float() / 255, out / f'samples_{step:06d}.png', nrow=10)
    if not n:
        return {'samples': 0, 'fid': None, 'feature_variance_ratio': None}
    stats = evaluator.statistics(images)
    return {'samples': n, 'fid': float(fid_statistics_to_metric(stats, evaluator.real, verbose=False)['frechet_inception_distance']),
            'feature_variance_ratio': float(np.trace(stats['sigma']) / np.trace(evaluator.real['sigma']))}


class SpatialAttention(torch.nn.Module):
    """Global attention over a square grid with learned 2D relative bias."""
    def __init__(self, dim, heads, side):
        super().__init__()
        self.heads = heads
        self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.project = torch.nn.Linear(dim, dim)
        self.relative_bias = torch.nn.Parameter(torch.empty((2 * side - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing='ij')).flatten(1)
        delta = coords[:, :, None] - coords[:, None, :] + side - 1
        self.register_buffer('relative_index', delta[0] * (2 * side - 1) + delta[1])
        torch.nn.init.trunc_normal_(self.relative_bias, std=.02)

    def forward(self, x):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1).unsqueeze(0)
        h = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.)
        return self.project(h.transpose(1, 2).reshape(batch, tokens, dim))


class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, heads, side, mlp_ratio):
        super().__init__()
        self.norm_attention = torch.nn.LayerNorm(dim)
        self.attention = SpatialAttention(dim, heads, side)
        self.norm_mlp = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, int(dim * mlp_ratio)),
                                       torch.nn.GELU(), torch.nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class TransGANGenerator(torch.nn.Module):
    """Three transformer stages and two channel-to-space upsampling operations."""
    def __init__(self, z_dim, dim=512, depths=(5, 4, 2), heads=4, mlp_ratio=4., rgb_gain=.1):
        super().__init__()
        self.dim = dim
        self.input = torch.nn.Linear(z_dim, 8 * 8 * dim)
        self.positions = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(1, side * side, channels))
            for side, channels in zip((8, 16, 32), (dim, dim // 4, dim // 16))])
        self.stages = torch.nn.ModuleList([
            torch.nn.Sequential(*[TransformerBlock(channels, heads, side, mlp_ratio) for _ in range(depth)])
            for side, channels, depth in zip((8, 16, 32), (dim, dim // 4, dim // 16), depths)])
        # Upstream emits unbounded RGB. With AE-GAN's tanh image range, a final
        # token norm prevents residual-stream growth from killing its gradients.
        self.output_norm = torch.nn.LayerNorm(dim // 16)
        self.output = torch.nn.Conv2d(dim // 16, 3, 1)
        # Match upstream train_derived.py: Xavier applies only to Conv2d weights.
        # Its Linear override is commented out: retain constructor fan-in uniform
        # weights/biases. Blanket Xavier amplifies the deep residual stages and
        # can drive this bounded RGB head into tanh saturation immediately.
        torch.nn.init.xavier_uniform_(self.output.weight, gain=rgb_gain)
        for position in self.positions:
            torch.nn.init.trunc_normal_(position, std=.02)

    def forward(self, z):
        x = self.input(z).reshape(len(z), 64, self.dim)
        for index, (position, stage) in enumerate(zip(self.positions, self.stages)):
            if index:
                side = 8 * 2 ** (index - 1)
                h = x.transpose(1, 2).reshape(len(z), -1, side, side)
                x = torch.nn.functional.pixel_shuffle(h, 2).flatten(2).transpose(1, 2)
            x = stage(x + position)
        x = self.output_norm(x)
        return self.output(x.transpose(1, 2).reshape(len(z), -1, 32, 32)).tanh()


class DeeperGenerator(DirectGenerator):
    """Additional residual blocks at each output resolution; no image side inputs."""
    def __init__(self, z_dim, width, depth):
        super().__init__(z_dim, width)
        self.refine = torch.nn.ModuleList([
            torch.nn.Sequential(*[ResBlock(ch, ch, 4 * width) for _ in range(depth - 1)])
            for ch in (4 * width, 2 * width, width)])

    def forward(self, z):
        from torch.nn import functional as F
        h = self.input(z).reshape(-1, 4 * self.width, 4, 4)
        e = self.embed(z)
        for block, refinement in zip(self.blocks, self.refine):
            h = block(F.interpolate(h, scale_factor=2, mode='nearest'), e)
            for extra in refinement:
                h = extra(h, e)
        return self.output(F.leaky_relu(h, .2)).tanh()


def build_models(cfg):
    # Advance the shared initialization stream exactly as the historical G did.
    # Alternate G construction then leaves D/E initialization untouched.
    g = DirectGenerator(cfg['z_dim'], cfg['width'])
    width = cfg['g_width'] or cfg['width']
    if cfg['generator_arch'] == 'transgan':
        with torch.random.fork_rng(devices=[]):
            g = TransGANGenerator(cfg['z_dim'], cfg['transgan_dim'], cfg['transgan_depths'],
                                 cfg['transgan_heads'], cfg['transgan_mlp_ratio'], cfg['transgan_rgb_gain'])
    elif width != cfg['width'] or cfg['g_depth'] != 1:
        with torch.random.fork_rng(devices=[]):
            g = (DirectGenerator(cfg['z_dim'], width) if cfg['g_depth'] == 1
                 else DeeperGenerator(cfg['z_dim'], width, cfg['g_depth']))
    return g, DirectDiscriminator(cfg['width']), build_encoder(cfg)


def load_resume(cfg):
    if not cfg['resume_checkpoint']:
        return None, None
    path = (ROOT / cfg['resume_checkpoint']).resolve()
    out = (ROOT / cfg['out_dir']).resolve()
    if path.is_relative_to(out):
        raise ValueError('resume requires a new output directory outside the source checkpoint')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != cfg['resume_sha256']:
        raise ValueError('resume checkpoint digest mismatch')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    required = {'config', 'sources', 'step', 'G', 'D', 'E', 'prior', 'ema_G', 'ema_E',
                'ema_prior', 'optimizer_g', 'optimizer_d', 'rng', 'torch_rng', 'cuda_rng',
                'initialization_sha256', 'train_seconds'}
    if not required <= set(ck):
        raise ValueError('checkpoint lacks full training state')
    for name, expected in ck['sources'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'checkpoint source changed: {name}')
    allowed = {'expansion_factor', 'initial_eval_samples', 'steps', 'out_dir', 'log_interval', 'eval_interval', 'eval_samples', 'final_samples',
               'recon_samples', 'eval_batch_size', 'max_train_seconds', 'keep_checkpoints',
               'resume_checkpoint', 'resume_sha256', 'recon_weight', 'g_lr_scale', 'lr_scale',
               'lr_final_ratio', 'lr_decay_start', 'lr_decay_end', 'd_updates',
               'recon_grad', 'reg_every', 'penalty_coeff', 'd_warmstart', 'd_warmstart_sha256'}
    old = {**DEFAULTS, **ck['config']}
    if 'generator_arch' not in ck['config']:
        raise ValueError('resume requires a checkpoint from this architecture-aware trainer')
    def immutable(c):
        values = {k: v for k, v in c.items() if k not in allowed}
        values['g_width'] = c['g_width'] or c['width']
        return values
    if immutable(cfg) != immutable(old):
        raise ValueError('resume cannot change architecture, objective, optimizer or data configuration')
    if not 0 < ck['step'] < cfg['steps']:
        raise ValueError('requested steps must exceed saved checkpoint step')
    return ck, {'path': str(path), 'sha256': digest, 'step': ck['step'],
                'previous_train_seconds': ck['train_seconds'], 'sources': ck['sources'],
                'interventions': {k: {'before': old[k], 'after': cfg[k]} for k in
                    ('expansion_factor', 'recon_weight', 'g_lr_scale', 'lr_scale', 'lr_final_ratio', 'lr_decay_start', 'lr_decay_end', 'd_updates', 'recon_grad', 'reg_every', 'penalty_coeff', 'd_warmstart', 'd_warmstart_sha256')
                    if old[k] != cfg[k]}}


def restore_checkpoint(ck, modules, optimizers, streams):
    for module, name in zip(modules, ('G', 'D', 'E', 'prior', 'ema_G', 'ema_E', 'ema_prior')):
        module.load_state_dict(ck[name], strict=True)
    for optimizer, name in zip(optimizers, ('optimizer_g', 'optimizer_d')):
        optimizer.load_state_dict(ck[name])
    if set(streams) != set(ck['rng']):
        raise ValueError('checkpoint training RNG streams differ')
    for name, generator in streams.items():
        generator.set_state(ck['rng'][name])
    if len(ck['cuda_rng']) != torch.cuda.device_count():
        raise ValueError('resume requires the same number of visible CUDA devices')
    torch.set_rng_state(ck['torch_rng'])
    torch.cuda.set_rng_state_all(ck['cuda_rng'])


def reconstruction_loss(g, e, prior, real, cfg):
    means = prior.means()
    if cfg['recon_grad'] == 'encoder_only':
        means = means.detach()
    encoded, _, _, _ = e(real, means, prior.sigma, cfg['temperature'])
    # Freeze G parameters only for this forward: gradients still pass through G
    # into E. Restore flags before the combined adversarial/reconstruction backward.
    flags = [p.requires_grad for p in g.parameters()]
    if cfg['recon_grad'] == 'encoder_only':
        g.requires_grad_(False)
    try:
        return (g(encoded) - real).square().mean()
    finally:
        for p, enabled in zip(g.parameters(), flags):
            p.requires_grad_(enabled)


def set_learning_rates(cfg, step, og, od):
    fraction = min(1., max(0., (step - cfg['lr_decay_start']) /
                           (cfg['lr_decay_end'] - cfg['lr_decay_start'])))
    scale = cfg['lr_scale'] * (1 + fraction * (cfg['lr_final_ratio'] - 1))
    for group, base in zip(og.param_groups, (cfg['lr'] * cfg['g_lr_scale'], cfg['lr'], cfg['prior_lr'])):
        group['lr'] = base * scale
    od.param_groups[0]['lr'] = cfg['d_lr'] * scale
    return scale


def train(cfg):
    validate(cfg)
    checkpoint, resume_info = load_resume(cfg)
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
    g, d, e = [m.cuda() for m in build_models(cfg)]
    prior = MoGParticlePrior(num_particles=cfg['num_particles'], z_dim=cfg['z_dim'],
                             sigma_rel=cfg['sigma_rel'], generator=rng(cfg['seed'] + 1, 'cpu')).cuda()
    initial_hash = state_hash([g, d, e, prior])
    initial_sigma = prior.sigma.clone()
    initial_prior = prior.z.detach().clone()
    initial_features = state_hash([d.critic.features])
    initial_encoder_features = state_hash([e.features])
    eg, ee, ep = [copy.deepcopy(m).eval().requires_grad_(False) for m in (g, e, prior)]
    og = torch.optim.Adam([
        {'params': g.parameters(), 'lr': cfg['lr'] * cfg['g_lr_scale']},
        {'params': [p for p in e.parameters() if p.requires_grad], 'lr': cfg['lr']},
        {'params': prior.parameters(), 'lr': cfg['prior_lr'], 'betas': (.5, .999)},
    ], betas=(0., .999), fused=True)
    od = torch.optim.Adam([p for p in d.parameters() if p.requires_grad], lr=cfg['d_lr'], betas=(0., .999), fused=True)
    adversarial, penalty, spread = GANLoss(), GradientPenalty(coeff=cfg['penalty_coeff'], lazy_k=cfg['reg_every']), ParticleRegularizer()
    streams = {name: rng(cfg['seed'] + offset) for name, offset in [('data', 2), ('prior', 3)]}
    start_step, previous_train_seconds = 0, 0.
    if checkpoint is not None:
        if checkpoint['initialization_sha256'] != initial_hash:
            raise ValueError('checkpoint initialization differs from this model/configuration')
        expansion_audit = prepare_expansion(checkpoint, prior, ep, cfg)
        initial_prior = initial_prior.repeat_interleave(cfg['expansion_factor'], 0)
        write_json(out / 'expansion_audit.json', expansion_audit)
        restore_checkpoint(checkpoint, (g, d, e, prior, eg, ee, ep), (og, od), streams)
        if cfg['d_warmstart']:
            path = ROOT / cfg['d_warmstart']
            if hashlib.sha256(path.read_bytes()).hexdigest() != cfg['d_warmstart_sha256']:
                raise ValueError('D warmstart hash mismatch')
            delta = torch.load(path, map_location='cpu', weights_only=False)
            if delta['artifact_type'] != 'D_only_delta' or delta['parent_sha256'] != cfg['resume_sha256']:
                raise ValueError('D warmstart must descend from this exact parent')
            for name, expected in delta['sources'].items():
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
            untouched = state_hash([g, e, prior, eg, ee, ep])
            d.load_state_dict(delta['D'], strict=True)
            od.load_state_dict(delta['optimizer_d'])
            assert untouched == state_hash([g, e, prior, eg, ee, ep])
            assert initial_features == state_hash([d.critic.features])
            resume_info['D_warmstart'] = {'path': str(path), 'sha256': cfg['d_warmstart_sha256'],
                                         'd_only_steps': delta['d_only_steps'], 'config': delta['config']}
            del delta
        start_step, previous_train_seconds = checkpoint['step'], checkpoint['train_seconds']
        resume_info['restored_state_sha256'] = state_hash([g, d, e, prior, eg, ee, ep])
        write_json(out / 'resume.json', resume_info)
        print(f"RESUME step={start_step} parent_sha256={resume_info['sha256']}", flush=True)
        del checkpoint
    if cfg['expansion_factor'] == 1:
        prior.exposure = torch.zeros(prior.num_particles, dtype=torch.long, device='cuda')
    if cfg['expansion_factor'] > 1:
        spread = ReferenceRegularizer(cfg['num_particles'])
    if cfg['initial_eval_samples']:
        initial_eval = generation(eg, ep, evaluator, cfg, cfg['initial_eval_samples'], out, start_step)
        write_json(out / 'initial_evaluation.json', initial_eval)
        print('INITIAL_EVAL ' + json.dumps(initial_eval), flush=True)
    write_json(out / f'descendants_{start_step:06d}.json', descendant_panel(eg, ep, cfg, out, start_step, evaluator))
    metadata = {'initialization_sha256': initial_hash, 'sigma': float(prior.sigma),
                'initial_nearest_neighbor_median': float(prior.d0), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'python': platform.python_version(),
                'gpu': torch.cuda.get_device_name(), 'pretrained': d.critic.pretrained_metadata,
                'parameters': {name: sum(p.numel() for p in m.parameters()) for name, m in [('G', g), ('D', d), ('E', e), ('prior', prior)]},
                'fid_protocol': PROTOCOL, 'reconstruction_split': 'CIFAR-10 test, unaugmented',
                'evaluation_weights': 'EMA G/E/prior; final checkpoint, no best selection'}
    metadata['encoder'] = {'backbone': cfg['encoder_backbone'],
                           'pretrained': getattr(e, 'pretrained_metadata', None),
                           'initial_feature_sha256': initial_encoder_features,
                           'trainable_parameters': sum(p.numel() for p in e.parameters() if p.requires_grad)}
    metadata['generator'] = {'architecture': cfg['generator_arch'],
                             'cnn_width': cfg['g_width'] or cfg['width'], 'cnn_depth': cfg['g_depth'],
                             'transformer_dim': cfg['transgan_dim'], 'transformer_depths': cfg['transgan_depths'],
                             'transformer_heads': cfg['transgan_heads'], 'transformer_mlp_ratio': cfg['transgan_mlp_ratio'],
                             'transformer_rgb_gain': cfg['transgan_rgb_gain'], 'transformer_output_norm': 'LayerNorm',
                             'reference': 'VITA-Group/TransGAN@6b85440ca56716fd7a60bac964466cc0296ce663',
                             'implementation': 'TransGAN-style G with existing AE-GAN recipe; not full TransGAN'}
    metadata['reconstruction_routing'] = {'mode': cfg['recon_grad'], 'E': True,
                                         'G': cfg['recon_grad'] == 'all', 'prior': cfg['recon_grad'] == 'all'}
    metadata['resume'] = resume_info
    metadata['D_E_initialization_sha256'] = state_hash([d, e]) if start_step == 0 else None
    metadata['regularizer'] = {'arm': penalty.arm, 'method': penalty.method,
                              'coeff': penalty.coeff, 'kappa': penalty.kappa,
                              'every': penalty.lazy_k, 'applied_coeff': penalty.coeff * penalty.lazy_k}
    write_json(out / 'metadata.json', metadata)
    print(f"START architecture={cfg['generator_arch']} recon_grad={cfg['recon_grad']} arm={cfg['arm']} encoder={cfg['encoder_backbone']} reg_every={cfg['reg_every']} g_width={cfg['g_width'] or cfg['width']} g_depth={cfg['g_depth']} steps={cfg['steps']} sigma={float(prior.sigma):.6f} init={initial_hash}", flush=True)

    def batch():
        ids = torch.randint(len(images), (cfg['batch_size'],), device='cuda', generator=streams['data'])
        x = images[ids].float() / 127.5 - 1
        flip = torch.rand((len(x), 1, 1, 1), device='cuda', generator=streams['data']) < .5
        return torch.where(flip, x.flip(-1), x)

    train_seconds = 0.
    torch.cuda.synchronize()
    block_start = time.perf_counter()
    with (out / 'metrics.jsonl').open('w', buffering=1) as log:
        for step in range(start_step + 1, cfg['steps'] + 1):
            lr_scale = set_learning_rates(cfg, step, og, od)
            for d_index in range(cfg['d_updates']):
                d.requires_grad_(True)
                real = batch()
                with torch.no_grad():
                    z, sampled_ids = prior.sample(cfg['batch_size'], streams['prior'])
                    if cfg['expansion_factor'] == 1: prior.exposure.index_add_(0, sampled_ids, torch.ones_like(sampled_ids))
                    fake = g(z)
                od.zero_grad(set_to_none=True)
                dp = penalty(d, real, fake, (step - 1) * cfg['d_updates'] + d_index + 1)
                dl = adversarial.d_loss(d(real), d(fake)) + dp
                dl.backward()
                od.step()
            d.requires_grad_(False)
            real = batch()
            og.zero_grad(set_to_none=True)
            z, sampled_ids = prior.sample(cfg['batch_size'], streams['prior'])
            if cfg['expansion_factor'] == 1: prior.exposure.index_add_(0, sampled_ids, torch.ones_like(sampled_ids))
            with torch.no_grad():
                dr = d(real)
            gl = adversarial.g_loss(d(g(z)), dr)
            rec = gl.new_zeros(())
            if cfg['arm'] == 'bounded' and cfg['recon_weight'] > 0:
                rec = reconstruction_loss(g, e, prior, real, cfg)
            loss = gl + cfg['recon_weight'] * rec + spread(prior.z)
            loss.backward()
            if step == start_step + 1 or step % cfg['log_interval'] == 0 or step == cfg['steps']:
                grad_norms = {name: float(sum((p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None), gl.new_zeros(())).sqrt()) for name, module in [('G', g), ('E', e), ('prior', prior)]}
                prior_before = prior.z.detach().clone()
            og.step()
            if step == start_step + 1 or step % cfg['log_interval'] == 0 or step == cfg['steps']:
                prior_update_rms = float((prior.z.detach() - prior_before).square().mean().sqrt())
            for target, source in ((eg, g), (ee, e), (ep, prior)):
                update_ema(target, source, cfg['ema'])
            report = step % cfg['log_interval'] == 0 or step == start_step + 1 or step == cfg['steps']
            evaluate = step % cfg['eval_interval'] == 0 or step == cfg['steps']
            if report or evaluate:
                torch.cuda.synchronize()
                train_seconds += time.perf_counter() - block_start
                row = {'step': step, 'gradient_norms': grad_norms, 'prior_update_rms': prior_update_rms, 'learning_rates': {'G': og.param_groups[0]['lr'], 'E': og.param_groups[1]['lr'], 'prior': og.param_groups[2]['lr'], 'D': od.param_groups[0]['lr']}, 'lr_scale': lr_scale, 'd_updates': cfg['d_updates'], 'd_loss': float(dl.detach()), 'g_loss': float(gl.detach()),
                       'recon_train_mse': float(rec.detach()), 'penalty': float(dp.detach()),
                       'train_seconds': train_seconds, 'steps_per_second': (step - start_step) / train_seconds,
                       'start_step': start_step, 'cumulative_train_seconds': previous_train_seconds + train_seconds,
                       'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30}
                print(f"step={step}/{cfg['steps']} D={row['d_loss']:.4f} G={row['g_loss']:.4f} recon={row['recon_train_mse']:.5f} train_s={train_seconds:.1f} steps/s={row['steps_per_second']:.2f} lr_scale={lr_scale:.4f} peak_GB={row['peak_memory_gb']:.2f}", flush=True)
                if not all(math.isfinite(row[k]) for k in ('d_loss', 'g_loss', 'recon_train_mse')):
                    raise RuntimeError('nonfinite training loss')
                if train_seconds > cfg['max_train_seconds']:
                    raise RuntimeError('training time cap reached; no completed-run summary will be emitted')
                if evaluate:
                    assert torch.equal(prior.sigma, initial_sigma)
                    n = cfg['final_samples'] if step == cfg['steps'] else cfg['eval_samples']
                    print(f'EVAL step={step} generated_samples={n}', flush=True)
                    eval_started = time.perf_counter()
                    row['descendants'] = descendant_panel(eg, ep, cfg, out, step, evaluator)
                    row['exposure'] = exposure_summary(prior)
                    write_json(out / f'descendants_{step:06d}.json', row['descendants'])
                    row['generation'] = generation(eg, ep, evaluator, cfg, n, out, step)
                    torch.cuda.synchronize()
                    row['generation_seconds'] = time.perf_counter() - eval_started
                    recon_started = time.perf_counter()
                    row['reconstruction'] = (reconstruction(eg, ee, ep, test_images, cfg, out, step)
                                             if cfg['arm'] == 'bounded' else None)
                    torch.cuda.synchronize()
                    row['reconstruction_seconds'] = time.perf_counter() - recon_started
                    print(json.dumps({'evaluation': row}, allow_nan=False), flush=True)
                    torch.save({'config': cfg, 'sources': provenance['sources'], 'step': step,
                                'G': g.state_dict(), 'D': d.state_dict(), 'E': e.state_dict(),
                                'prior': prior.state_dict(), 'ema_G': eg.state_dict(), 'ema_E': ee.state_dict(),
                                'ema_prior': ep.state_dict(), 'optimizer_g': og.state_dict(), 'optimizer_d': od.state_dict(),
                                'rng': {k: v.get_state() for k, v in streams.items()},
                                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                                'initialization_sha256': initial_hash, 'train_seconds': previous_train_seconds + train_seconds,
                                'resume': resume_info}, out / 'checkpoint.pt')
                    if cfg['keep_checkpoints']:
                        import shutil
                        shutil.copy2(out / 'checkpoint.pt', out / f'checkpoint_{step:06d}.pt')
                log.write(json.dumps(row, allow_nan=False) + '\n')
                torch.cuda.synchronize()
                block_start = time.perf_counter()
    assert initial_features == state_hash([d.critic.features]), 'frozen feature extractor changed'
    encoder_frozen_unchanged = None
    if cfg['encoder_backbone'] == 'pretrained_resnet18':
        encoder_frozen_unchanged = initial_encoder_features == state_hash([e.features]) == state_hash([ee.features])
        assert encoder_frozen_unchanged, 'frozen encoder feature extractor changed'
    assert torch.equal(prior.sigma, initial_sigma) and torch.equal(ep.sigma, initial_sigma)
    if resume_info is not None:
        assert hashlib.sha256(Path(resume_info['path']).read_bytes()).hexdigest() == resume_info['sha256'], 'parent checkpoint changed'
    summary = {'config': cfg, 'final': {'step': step, **row['generation'], 'reconstruction': row['reconstruction']},
               'train_seconds': train_seconds, 'start_step': start_step,
               'cumulative_train_seconds': previous_train_seconds + train_seconds, 'total_seconds': time.perf_counter() - started_all,
               'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30,
               'prior_rms_movement': float((prior.z.detach() - initial_prior).square().mean().sqrt()),
               'rng_sha256': {k: hashlib.sha256(v.get_state().cpu().numpy().tobytes()).hexdigest() for k, v in streams.items()},
               'metadata': metadata, 'frozen_features_unchanged': True, 'sigma_unchanged': True,
               'frozen_encoder_features_unchanged': encoder_frozen_unchanged}
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
