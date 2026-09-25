"""Anima-style paired-error supervision for the previous-action three-G GAN."""
import json
from pathlib import Path

import torch
from torch import nn

from lib.gym_previous_gan import build_models as build_previous, hashes as previous_hashes
from lib.gym_previous_gan import task_loss as l2_task_loss
from lib.gym_state_control import parameter_hashes
from lib.gym_transition import GymTransitionScaler, contact_record
from lib.vendor.concept_slider_core.reference import (GlobalMixErrorCritic,
    register_paired_error_norm, noise_std, rp_d_loss, rp_g_loss)

MODULE_KEYS = ('G', 'E', 'prior', 'D', 'R')


class PairedErrorCritic(nn.Module):
    def __init__(self, targets, cfg):
        super().__init__()
        self.scope = cfg['slider_scope']
        self.horizon = cfg['steps']
        # Constant training-mean baseline replaces the frozen neutral teacher.
        # It cancels from prediction-minus-target, but defines scale calibration.
        register_paired_error_norm(self, targets, targets.mean(0).expand_as(targets))
        self.critic = GlobalMixErrorCritic(torch.stack([torch.zeros(targets.shape[1]), torch.ones(targets.shape[1])]),
            tokens=cfg['error_tokens'], width=cfg['error_width'], layers=1,
            heads=cfg['error_heads'], score_bound=8.)

    def sigma(self, step):
        # Absolute hold 1.0 from the Anima release, not the older ratio default.
        return noise_std(step - 1, start=float(self.edit_rms) / .28, decay_steps=self.horizon, hold=1.)

    def forward(self, coordinates):
        return self.critic(coordinates)

    def residual(self, decoded, real):
        if self.scope == 'all':
            predicted = contact_record(decoded, mode='probability')
            target = real
        else:
            predicted, target = decoded[:, 8:10], real[:, 8:10]
        return (predicted - target.detach()) / self.target_std


def build_models(cfg, scaler, targets, device='cpu'):
    bundle = build_previous(cfg, scaler, device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg['seed'] + 104)
        selected = targets[:, 8:10] if cfg['slider_scope'] == 'action' else targets
        bundle['R'] = PairedErrorCritic(selected.cpu(), cfg).to(device)
    return bundle


def hashes(bundle):
    return {**previous_hashes(bundle), 'R': parameter_hashes(
        dict(G=bundle['R'], E=bundle['R'], prior=bundle['R']))['G']}


def error_loss(critic, decoded, real, step, rng, *, capper=None):
    residual = critic.residual(decoded, real)
    noise = torch.randn(residual.shape, generator=rng, device=residual.device) * critic.sigma(step)
    fake = noise + (residual.detach() if capper is not None else residual)
    if capper is not None:
        adversarial = rp_d_loss(critic(noise), critic(fake))
        penalty, _ = capper.penalty(critic, noise, fake, step=step, collect_stats=False)
        return adversarial + penalty, dict(error_d_adversarial=adversarial, error_cap=penalty)
    with torch.no_grad():
        real_score = critic(noise)
    loss = rp_g_loss(real_score, critic(fake))
    return loss, dict(error_g_adversarial=loss)


def paired_loss(bundle, decoded, real, step, rng):
    cfg = bundle['config']
    learned, terms = error_loss(bundle['R'], decoded, real, step, rng)
    # Diagnostics only in all-heads mode: no MSE or BCE can enter its graph.
    with torch.no_grad():
        _, diagnostics = l2_task_loss(decoded.detach(), real.detach(), cfg)
    result = cfg['paired_error_weight'] * learned
    if cfg['slider_scope'] == 'action':
        _, aux = l2_task_loss(decoded, real, cfg)
        result = result + cfg['lambda_state'] * aux['state_loss'] + cfg['lambda_next'] * aux['next_loss']
    return result, {**diagnostics, **terms}


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('format') != 'gym_slider_gan_v1' or saved.get('gan_steps') != saved['step'] or saved['step'] < 1:
        raise ValueError('Expected a slider-error GAN checkpoint')
    cfg = saved['config']
    if cfg['slider_scope'] not in ('all', 'action') or min(cfg['paired_error_weight'], cfg['adversarial_weight'], cfg['marginal_weight']) <= 0:
        raise ValueError('All configured adversarial losses must remain active')
    # Normalization buffers below restore training-only calibration exactly.
    fixture = torch.stack([torch.zeros(18), torch.ones(18)])
    bundle = build_models(cfg, GymTransitionScaler(**saved['scaler']), fixture, device)
    for key in MODULE_KEYS:
        bundle[key].load_state_dict(saved[key])
        bundle[key].eval().requires_grad_(False)
    bundle.update(step=saved['step'], gan_steps=saved['gan_steps'], provenance=saved['provenance'])
    summary = Path(path).parent/'summary.json'
    if summary.exists():
        bundle['training_summary'] = json.loads(summary.read_text())
    return bundle
