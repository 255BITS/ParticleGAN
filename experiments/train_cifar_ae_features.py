#!/usr/bin/env python
"""Pretrained discriminator replacement and selective G-growth reconstruction gradients.

Separate entry point preserves the source of earlier certified experiments.
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
    'lr_scale': 1., 'lr_final_ratio': 1., 'lr_decay_start': 0, 'lr_decay_end': 200000,
    'd_updates': 1, 'grow_g': False, 'grow_d_heads': False,
    'd_backbone': 'pretrained_resnet18', 'recon_growth_grad': True,
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
    for key in ('recon_weight',):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('lr_scale', 'lr_final_ratio'):
        if not math.isfinite(cfg[key]) or not 0 < cfg[key] <= 1:
            raise ValueError(f'{key} must be in (0, 1]')
    if type(cfg['lr_decay_start']) is not int or not 0 <= cfg['lr_decay_start'] < cfg['lr_decay_end']:
        raise ValueError('invalid global learning-rate schedule')
    if not 0 <= cfg['ema'] < 1:
        raise ValueError('invalid EMA decay')
    if cfg['encoder_backbone'] not in ('scratch', 'pretrained_resnet18'):
        raise ValueError('invalid encoder_backbone')
    if type(cfg['keep_checkpoints']) is not bool:
        raise ValueError('keep_checkpoints must be boolean')
    for key in ('grow_g', 'grow_d_heads', 'recon_growth_grad'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if cfg['d_backbone'] not in ('pretrained_resnet18', 'pretrained_resnet34'):
        raise ValueError('invalid discriminator backbone')
    if not cfg['recon_growth_grad'] and not cfg['grow_g']:
        raise ValueError('selective reconstruction routing requires G growth')


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
    generator = rng(cfg['seed'] + 10000)
    chunks = []
    count = n or 100
    for lo in range(0, count, cfg['eval_batch_size']):
        z, _ = prior.sample(min(cfg['eval_batch_size'], count - lo), generator)
        chunks.append(uint8_images(g(z)).cpu())
    images = torch.cat(chunks)
    save_image(images[:100].float() / 255, out / f'samples_{step:06d}.png', nrow=10)
    if not n:
        return {'samples': 0, 'fid': None, 'feature_variance_ratio': None}
    stats = evaluator.statistics(images)
    return {'samples': n, 'fid': float(fid_statistics_to_metric(stats, evaluator.real, verbose=False)['frechet_inception_distance']),
            'feature_variance_ratio': float(np.trace(stats['sigma']) / np.trace(evaluator.real['sigma']))}


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


class IdentityGeneratorRefinement(ResBlock):
    """Latent-conditioned residual branch; zero last convolution makes x + f(x) = x."""
    def __init__(self, channels, embedding):
        super().__init__(channels, channels, embedding)
        torch.nn.init.zeros_(self.c2.weight)
        torch.nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        from torch.nn import functional as F
        h = self.c1(F.leaky_relu(self.n1(x), .2))
        scale, shift = self.cond(F.leaky_relu(e, .2))[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + scale) + shift
        return x + self.c2(F.leaky_relu(h, .2))


class GrownGenerator(DirectGenerator):
    def forward(self, z):
        from torch.nn import functional as F
        h = self.input(z).reshape(-1, 4 * self.width, 4, 4)
        e = self.embed(z)
        for i, block in enumerate(self.blocks):
            h = block(F.interpolate(h, scale_factor=2, mode='nearest'), e)
            if hasattr(self, 'refine'):
                for extra in self.refine[i]:
                    h = extra(h, e)
            h = self.growth[i](h, e)
        return self.output(F.leaky_relu(h, .2)).tanh()


class IdentityHeadRefinement(torch.nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.n1 = torch.nn.GroupNorm(8, channels)
        self.c1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.n2 = torch.nn.GroupNorm(8, channels)
        self.c2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        torch.nn.init.zeros_(self.c2.weight)
        torch.nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        from torch.nn import functional as F
        h = self.c1(F.leaky_relu(self.n1(x), .2))
        return x + self.c2(F.leaky_relu(self.n2(h), .2))


class GrownFeatureHead(torch.nn.Sequential):
    """Retain original numeric parameter keys; insert named growth before pooling."""
    def forward(self, x):
        for name, module in self._modules.items():
            if name == 'growth':
                continue
            if name == '5':
                x = self.growth(x)
            x = module(x)
        return x


def install_growth(cfg, g, d, eg, og=None, od=None):
    """Append new parameters to the existing first Adam groups without touching old state.

    Build base models first. For a grown parent, call with the parent's config before
    restore_checkpoint; then call with the requested config after strict restoration.
    Separate forked CPU seeds keep new branches identical across factorial arms.
    """
    added = {'G': 0, 'D': 0}
    if cfg.get('grow_g', False) and not hasattr(g, 'growth'):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg['seed'] + 60000)
            growth = torch.nn.ModuleList([
                IdentityGeneratorRefinement(ch, 4 * g.width)
                for ch in (4 * g.width, 2 * g.width, g.width)])
        g.__class__ = GrownGenerator
        g.growth = growth.to(next(g.parameters()))
        eg.__class__ = GrownGenerator
        eg.growth = copy.deepcopy(g.growth).eval().requires_grad_(False)
        parameters = list(g.growth.parameters())
        if og is not None:
            og.param_groups[0]['params'].extend(parameters)
        added['G'] = sum(p.numel() for p in parameters)
    if cfg.get('grow_d_heads', False) and not hasattr(d.critic.project[0], 'growth'):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg['seed'] + 70000)
            refinements = [IdentityHeadRefinement() for _ in d.critic.project]
        parameters = []
        for head, refinement in zip(d.critic.project, refinements):
            head.__class__ = GrownFeatureHead
            head.growth = refinement.to(next(head.parameters()))
            parameters.extend(head.growth.parameters())
        if od is not None:
            od.param_groups[0]['params'].extend(parameters)
        added['D'] = sum(p.numel() for p in parameters)
    return added


def _growth_probe(g, d, eg, cfg):
    """Small fixed probes, independent of all sampling and global random streams."""
    device = next(g.parameters()).device
    z = torch.linspace(-1.5, 1.5, 2 * cfg['z_dim'], device=device).reshape(2, -1)
    x = torch.linspace(-.95, .95, 2 * 3 * 32 * 32, device=device).reshape(2, 3, 32, 32)
    result = {}
    for name, module, value in (('G', g, z), ('ema_G', eg, z), ('D', d, x)):
        value = value.detach().requires_grad_(True)
        output = module(value)
        derivative, = torch.autograd.grad(output.sum(), value)
        result[name] = output.detach()
        result[name + '_input_gradient'] = derivative.detach()
    return result


def expand_with_audit(cfg, g, d, eg, og=None, od=None):
    """Audit with reproducible full-precision kernels, then restore training settings."""
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_context = d._context_features
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            return _expand_with_audit(cfg, g, d, eg, og, od)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        # A fresh production context must be computed using production precision.
        d._context_features = previous_context


def _expand_with_audit(cfg, g, d, eg, og=None, od=None):
    """Verify exact initial functions, D input gradients and trainable double backward."""
    before = _growth_probe(g, d, eg, cfg)
    added = install_growth(cfg, g, d, eg, og, od)
    after = _growth_probe(g, d, eg, cfg)
    errors = {}
    for name in before:
        torch.testing.assert_close(after[name], before[name], rtol=1e-5, atol=1e-6)
        errors[name] = float((after[name] - before[name]).abs().max())
    device = next(d.parameters()).device
    x = torch.linspace(-.9, .9, 2 * 3 * 32 * 32, device=device).reshape(2, 3, 32, 32).requires_grad_(True)
    gradient, = torch.autograd.grad(d(x).sum(), x, create_graph=True)
    parameters = [p for p in d.parameters() if p.requires_grad]
    second = torch.autograd.grad(gradient.square().sum(), parameters, allow_unused=True)
    if not torch.isfinite(gradient).all() or not all(torch.isfinite(v).all() for v in second if v is not None):
        raise RuntimeError('nonfinite discriminator double-backward growth audit')
    flags = [(p, p.requires_grad) for p in d.parameters()]
    try:
        d.requires_grad_(False)
        frozen_input = x.detach().requires_grad_(True)
        frozen_gradient, = torch.autograd.grad(d(frozen_input).sum(), frozen_input)
        torch.testing.assert_close(frozen_gradient, gradient.detach(), rtol=1e-5, atol=1e-6)
    finally:
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)
    if any(p.requires_grad for p in d.critic.features.parameters()):
        raise RuntimeError('pretrained discriminator features must remain frozen')
    return {'added_parameters': added, 'maximum_absolute_error': errors,
            'double_backward_finite': True, 'frozen_D_input_gradient_preserved': True,
            'new_G_EMA_equal': state_hash([g.growth]) == state_hash([eg.growth]) if added['G'] else None}


def install_backbone(d, kind):
    """Replace only frozen features; inherited trainable heads/Adam keep their identities.

    The two pretrained networks expose the same layer1/2/3 channel dimensions.
    Their coordinates differ: this is explicitly NOT a function-preserving D edit.
    """
    current = ('pretrained_resnet34' if 'ResNet34' in d.critic.pretrained_metadata['weights']
               else 'pretrained_resnet18')
    if current == kind:
        return False
    from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
    factory, weights = ((resnet34, ResNet34_Weights.IMAGENET1K_V1) if kind == 'pretrained_resnet34'
                        else (resnet18, ResNet18_Weights.IMAGENET1K_V1))
    with torch.random.fork_rng(devices=[]):
        net = factory(weights=weights)
        features = torch.nn.ModuleList([
            torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1),
            net.layer2, net.layer3])
    for module in features.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    d.critic.features = features.to(next(d.parameters())).eval().requires_grad_(False)
    d._context_features = None
    digest = hashlib.sha256()
    for name, value in features.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    d.critic.pretrained_metadata = {'weights': str(weights), 'feature_state_sha256': digest.hexdigest(),
                                    'input_size': 64, 'stages': ['layer1', 'layer2', 'layer3']}
    return True


def replace_backbone_with_audit(cfg, g, d, eg):
    """Record the critic discontinuity and verify unchanged generator functions."""
    previous_tf32, previous_context = torch.backends.cuda.matmul.allow_tf32, d._context_features
    old_weights = d.critic.pretrained_metadata['weights']
    old_head_hash = state_hash([d.critic.pixel, d.critic.project])
    changed = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        d._context_features = None
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            before = _growth_probe(g, d, eg, cfg)
            changed = install_backbone(d, cfg['d_backbone'])
            after = _growth_probe(g, d, eg, cfg)
            errors = {k: float((after[k] - before[k]).abs().max()) for k in before}
            for key in ('G', 'G_input_gradient', 'ema_G', 'ema_G_input_gradient'):
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
            assert old_head_hash == state_hash([d.critic.pixel, d.critic.project])
            return {'changed': changed, 'before': old_weights,
                    'after': d.critic.pretrained_metadata['weights'],
                    'maximum_absolute_change': errors, 'trainable_D_unchanged_at_swap': True,
                    'D_function_preservation_expected': not changed,
                    'adaptation': 'joint training; no extra D warmup or optimizer reset'}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        d._context_features = None if changed else previous_context


def reconstruction_loss(g, e, prior, real, cfg):
    encoded, _, _, _ = e(real, prior.means(), prior.sigma, cfg['temperature'])
    parameters = list(g.growth.parameters()) if not cfg['recon_growth_grad'] else []
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        # Parameter freezing applies only to this forward. Gradients still flow
        # through each refinement into old G, E, and prior, and the previously
        # built adversarial graph still updates the new parameters.
        return (g(encoded) - real).square().mean()
    finally:
        for p, enabled in zip(parameters, flags):
            p.requires_grad_(enabled)


def build_models(cfg):
    # Advance the shared initialization stream exactly as the historical G did.
    # Alternate G construction then leaves D/E initialization untouched.
    g = DirectGenerator(cfg['z_dim'], cfg['width'])
    width = cfg['g_width'] or cfg['width']
    if width != cfg['width'] or cfg['g_depth'] != 1:
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
    allowed = {'steps', 'out_dir', 'log_interval', 'eval_interval', 'eval_samples', 'final_samples',
               'recon_samples', 'eval_batch_size', 'max_train_seconds', 'keep_checkpoints',
               'resume_checkpoint', 'resume_sha256', 'recon_weight', 'lr_scale',
               'lr_final_ratio', 'lr_decay_start', 'lr_decay_end', 'd_updates',
               'grow_g', 'grow_d_heads', 'd_backbone', 'recon_growth_grad'}
    old = {**DEFAULTS, **ck['config']}
    for key in ('grow_g', 'grow_d_heads'):
        if old[key] and not cfg[key]:
            raise ValueError(f'resume cannot remove existing {key} capacity')
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
                    ('recon_weight', 'lr_scale', 'lr_final_ratio', 'lr_decay_start', 'lr_decay_end', 'd_updates', 'grow_g', 'grow_d_heads', 'd_backbone', 'recon_growth_grad')
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


def set_learning_rates(cfg, step, og, od):
    fraction = min(1., max(0., (step - cfg['lr_decay_start']) /
                           (cfg['lr_decay_end'] - cfg['lr_decay_start'])))
    scale = cfg['lr_scale'] * (1 + fraction * (cfg['lr_final_ratio'] - 1))
    for group, base in zip(og.param_groups, (cfg['lr'], cfg['lr'], cfg['prior_lr'])):
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
        {'params': g.parameters(), 'lr': cfg['lr']},
        {'params': [p for p in e.parameters() if p.requires_grad], 'lr': cfg['lr']},
        {'params': prior.parameters(), 'lr': cfg['prior_lr'], 'betas': (.5, .999)},
    ], betas=(0., .999), fused=True)
    od = torch.optim.Adam([p for p in d.parameters() if p.requires_grad], lr=cfg['d_lr'], betas=(0., .999), fused=True)
    adversarial, penalty, spread = GANLoss(), GradientPenalty(lazy_k=cfg['reg_every']), ParticleRegularizer()
    streams = {name: rng(cfg['seed'] + offset) for name, offset in [('data', 2), ('prior', 3)]}
    start_step, previous_train_seconds = 0, 0.
    if checkpoint is not None:
        if checkpoint['initialization_sha256'] != initial_hash:
            raise ValueError('checkpoint initialization differs from this model/configuration')
        install_backbone(d, checkpoint['config'].get('d_backbone', 'pretrained_resnet18'))
        install_growth({**DEFAULTS, **checkpoint['config']}, g, d, eg, og, od)
        restore_checkpoint(checkpoint, (g, d, e, prior, eg, ee, ep), (og, od), streams)
        start_step, previous_train_seconds = checkpoint['step'], checkpoint['train_seconds']
        resume_info['restored_state_sha256'] = state_hash([g, d, e, prior, eg, ee, ep])
        write_json(out / 'resume.json', resume_info)
        print(f"RESUME step={start_step} parent_sha256={resume_info['sha256']}", flush=True)
        del checkpoint
    backbone_audit = replace_backbone_with_audit(cfg, g, d, eg)
    initial_features = state_hash([d.critic.features])
    write_json(out / 'backbone_audit.json', backbone_audit)
    print('BACKBONE ' + json.dumps(backbone_audit, allow_nan=False), flush=True)
    growth_audit = expand_with_audit(cfg, g, d, eg, og, od)
    write_json(out / 'growth_audit.json', growth_audit)
    print('GROWTH ' + json.dumps(growth_audit, allow_nan=False), flush=True)
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
    metadata['generator'] = {'width': cfg['g_width'] or cfg['width'], 'depth_per_resolution': cfg['g_depth']}
    metadata['growth'] = {'grow_g': cfg['grow_g'], 'grow_d_heads': cfg['grow_d_heads'], 'audit': growth_audit}
    metadata['backbone_intervention'] = backbone_audit
    metadata['reconstruction_routing'] = {'old_G': True, 'new_G': cfg['recon_growth_grad'], 'E': True, 'prior': True}
    metadata['resume'] = resume_info
    metadata['D_E_initialization_sha256'] = state_hash([d, e]) if start_step == 0 else None
    metadata['regularizer'] = {'arm': penalty.arm, 'method': penalty.method,
                              'coeff': penalty.coeff, 'kappa': penalty.kappa,
                              'every': penalty.lazy_k, 'applied_coeff': penalty.coeff * penalty.lazy_k}
    write_json(out / 'metadata.json', metadata)
    print(f"START arm={cfg['arm']} encoder={cfg['encoder_backbone']} reg_every={cfg['reg_every']} g_width={cfg['g_width'] or cfg['width']} g_depth={cfg['g_depth']} steps={cfg['steps']} sigma={float(prior.sigma):.6f} init={initial_hash}", flush=True)

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
                    z, _ = prior.sample(cfg['batch_size'], streams['prior'])
                    fake = g(z)
                od.zero_grad(set_to_none=True)
                dp = penalty(d, real, fake, (step - 1) * cfg['d_updates'] + d_index + 1)
                dl = adversarial.d_loss(d(real), d(fake)) + dp
                dl.backward()
                od.step()
            d.requires_grad_(False)
            real = batch()
            og.zero_grad(set_to_none=True)
            z, _ = prior.sample(cfg['batch_size'], streams['prior'])
            with torch.no_grad():
                dr = d(real)
            gl = adversarial.g_loss(d(g(z)), dr)
            rec = gl.new_zeros(())
            if cfg['arm'] == 'bounded' and cfg['recon_weight'] > 0:
                rec = reconstruction_loss(g, e, prior, real, cfg)
            loss = gl + cfg['recon_weight'] * rec + spread(prior.z)
            loss.backward()
            og.step()
            for target, source in ((eg, g), (ee, e), (ep, prior)):
                update_ema(target, source, cfg['ema'])
            report = step % cfg['log_interval'] == 0 or step == start_step + 1 or step == cfg['steps']
            evaluate = step % cfg['eval_interval'] == 0 or step == cfg['steps']
            if report or evaluate:
                torch.cuda.synchronize()
                train_seconds += time.perf_counter() - block_start
                row = {'step': step, 'lr_scale': lr_scale, 'd_updates': cfg['d_updates'], 'd_loss': float(dl.detach()), 'g_loss': float(gl.detach()),
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
