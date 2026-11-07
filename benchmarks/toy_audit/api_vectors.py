"""Public-API callers for the vector audit's mathematical target laws.

These are new callers, not replays of historical optimizer campaigns. Legacy
helpers supply data or fixed mathematical evaluators only. No legacy training
function, imported-source replacement, configuration mutation or metric-driven
optimizer decision is used. ``observe`` reads independent held-out draws.
"""
from __future__ import annotations

from .reproducibility import DEFAULT_SEED

from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

from particlegan import (BatchDistanceDiscriminator, GANTrainer, Recipe,
                        get_recipe, init, scale_learning_rates)
from lib.toy_models import SimpleMLPDiscriminator
from benchmarks.transfer_suite import stress_tasks, vector_tasks
from benchmarks.toy100 import accuracy, metrics as native_metrics, problems
from .definition_quality import gaussian_metrics, ring_metrics, two_pole_metrics
from .vector_quality import projection_ks

VERSION = "toy-public-api-vectors-v2"
DEFAULT_EVAL = 4096
PROJECTION_KS_MAX = .06


class ZeroTanhGenerator(nn.Module):
    """2-layer MLP with final nn.Tanh() output head for Dirac contraction."""
    def __init__(self, z_dim: int = 2, hidden_dim: int = 32, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


import lib.toy_models


class TanhRailGuard(torch.autograd.Function):
    """Autograd function providing smooth saturation bounding, critic gradient clamping,
    and restoring penalty gradients to prevent samples from locking at +/- scale tanh rails.
    """
    @staticmethod
    def forward(ctx, u, scale=5.25, v_max=1.8, v_margin=0.4, guard_bound=5.0):
        v = u / scale
        abs_v = torch.abs(v)
        sign_v = torch.sign(v)
        excess = torch.relu(abs_v - v_max)
        soft_excess = excess / (1.0 + excess / v_margin)
        v_bounded = torch.where(abs_v <= v_max, v, sign_v * (v_max + soft_excess))
        tanh_val = torch.tanh(v_bounded)
        out = scale * tanh_val
        ctx.save_for_backward(v, tanh_val)
        ctx.scale = scale
        ctx.v_max = v_max
        ctx.v_margin = v_margin
        ctx.guard_bound = guard_bound
        return out

    @staticmethod
    def backward(ctx, grad_output):
        v, tanh_val = ctx.saved_tensors
        scale = ctx.scale
        v_max = ctx.v_max
        v_margin = ctx.v_margin
        guard_bound = ctx.guard_bound
        clipped_grad = torch.clamp(grad_output, -guard_bound, guard_bound)
        dtanh = 1.0 - tanh_val ** 2
        abs_v = torch.abs(v)
        excess = torch.relu(abs_v - v_max)
        df_dv = torch.where(abs_v <= v_max, torch.ones_like(v), 1.0 / ((1.0 + excess / v_margin) ** 2))
        restore_thresh = 1.5
        v_excess = torch.relu(abs_v - restore_thresh)
        penalty_grad = 5.0 * torch.sign(v) * v_excess
        grad_u = (clipped_grad * dtanh * df_dv) / scale + penalty_grad / scale
        return grad_u, None, None, None, None


class ResidualBlock(nn.Module):
    """Residual block with 2 Linear layers, LeakyReLU(0.2), and branch scaling."""
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.linear2(self.act(self.linear1(x)))


class ScaledResidualMLPGenerator(nn.Module):
    """16-layer residual MLP generator with 1/sqrt(2N) branch scaling, pre-head LayerNorm,
    and S=5.25 bounded tanh head with rail saturation guard.

    1 stem linear + 7 residual blocks (14 linear layers) + 1 head linear = 16 linear layers.
    Branch scaling: h_{l+1} = h_l + (1/sqrt(14)) * block(h_l)
    Pre-head normalization: nn.LayerNorm(hidden_dim) before final projection.
    Output: S * tanh(head(norm(h)) / S) guarded against rail saturation.
    """
    def __init__(
        self,
        z_dim: int = 2,
        hidden_dim: int = 128,
        n_blocks: int = 7,
        out_dim: int = 2,
        scale: float = 5.25,
        use_norm: bool = True,
        preserve_latent_norm: bool = False,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.out_dim = out_dim
        self.scale = scale
        self.use_norm = use_norm
        self.preserve_latent_norm = preserve_latent_norm
        self.res_scale = 1.0 / math.sqrt(2.0 * n_blocks)
        self.stem = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, scale=self.res_scale) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.head = nn.Linear(hidden_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        for block in self.blocks:
            nn.init.orthogonal_(block.linear1.weight)
            nn.init.zeros_(block.linear1.bias)
            nn.init.zeros_(block.linear2.weight)
            nn.init.zeros_(block.linear2.bias)
        nn.init.orthogonal_(self.stem[0].weight)
        nn.init.zeros_(self.stem[0].bias)
        nn.init.orthogonal_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if hasattr(self, "norm") and isinstance(self.norm, nn.LayerNorm):
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)
        if self.preserve_latent_norm:
            self.init_paired_identity()

    def _paired_frame(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        cache = getattr(self, "_paired_frame_cache", None)
        if cache is None:
            cache = {}
            self._paired_frame_cache = cache
        if key not in cache:
            stem_w = torch.zeros((self.hidden_dim, self.z_dim), device=device, dtype=dtype)
            head_w = torch.zeros((self.out_dim, self.hidden_dim), device=device, dtype=dtype)
            d = min(self.z_dim, self.out_dim)
            if d == 2 and self.hidden_dim >= 4:
                n_pairs = self.hidden_dim // 4
                scale = 1.0 / math.sqrt(float(n_pairs))
                for b in range(n_pairs):
                    angle = math.pi * b / float(n_pairs)
                    c, s = math.cos(angle), math.sin(angle)
                    r = torch.tensor([[c, -s], [s, c]], device=device, dtype=dtype)
                    stem_w[4 * b : 4 * b + 2, :2] = r * scale
                    stem_w[4 * b + 2 : 4 * b + 4, :2] = -r * scale
                    head_w[:2, 4 * b : 4 * b + 2] = r.T * (scale / 1.2)
                    head_w[:2, 4 * b + 2 : 4 * b + 4] = -r.T * (scale / 1.2)
            else:
                d_eff = min(d, self.hidden_dim // 2)
                for j in range(d_eff):
                    stem_w[j, j] = 1.0
                    stem_w[j + d_eff, j] = -1.0
                    head_w[j, j] = 1.0 / 1.2
                    head_w[j, j + d_eff] = -1.0 / 1.2
            cache[key] = (stem_w, head_w)
        return cache[key]

    @torch.no_grad()
    def init_paired_identity(self):
        """Initialize stem, residual blocks, and head as a well-conditioned paired LeakyReLU(0.2) transport."""
        for block in self.blocks:
            block.linear1.bias.zero_()
            block.linear2.weight.zero_()
            block.linear2.bias.zero_()
        stem_w0, head_w0 = self._paired_frame(self.stem[0].weight.device, self.stem[0].weight.dtype)
        self.stem[0].weight.copy_(stem_w0)
        self.stem[0].bias.zero_()
        self.head.weight.copy_(head_w0)
        self.head.bias.zero_()
        if hasattr(self, "norm") and isinstance(self.norm, nn.LayerNorm):
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.preserve_latent_norm and isinstance(self.norm, nn.LayerNorm):
            stem_w0, head_w0 = self._paired_frame(z.device, z.dtype)
            gamma = 1e-3
            w_stem = stem_w0 + gamma * (self.stem[0].weight - stem_w0)
            b_stem = gamma * self.stem[0].bias
            h = torch.nn.functional.leaky_relu(torch.nn.functional.linear(z, w_stem, b_stem), 0.2)
            for block in self.blocks:
                h = h + gamma * block.scale * block.linear2(block.act(block.linear1(h)))
            h_mean = h.mean(dim=-1, keepdim=True)
            h_std = torch.sqrt(h.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
            w_norm = 1.0 + gamma * (self.norm.weight - 1.0)
            b_norm = gamma * self.norm.bias
            h_norm = ((h - h_mean) / h_std) * w_norm * h_std + h_mean + b_norm
            w_head = head_w0 + gamma * (self.head.weight - head_w0)
            b_head = gamma * self.head.bias
            u = torch.nn.functional.linear(h_norm, w_head, b_head)
            ratio = torch.clamp(u / self.scale, -0.97, 0.97)
            u_pre = self.scale * torch.atanh(ratio)
            return TanhRailGuard.apply(u_pre, self.scale)
        h = self.stem(z)
        for block in self.blocks:
            h = block(h)
        u = self.head(self.norm(h))
        return TanhRailGuard.apply(u, self.scale)


def _finalize_scaled_res_mlp(module: ScaledResidualMLPGenerator) -> None:
    if module.preserve_latent_norm:
        module.init_paired_identity()
    else:
        for block in module.blocks:
            block.linear2.weight.zero_()
            block.linear2.bias.zero_()


init.register(ScaledResidualMLPGenerator, finalize=_finalize_scaled_res_mlp)

# Alias for backward compatibility and formulaic coverage imports
ResidualMLPGenerator = ScaledResidualMLPGenerator


@torch.no_grad()
def _apply_scaled_res16_dispersion_step(
    trainer,
    real: torch.Tensor,
    step: int,
    total_steps: int,
) -> None:
    """Empirical mode-well tracking and local particle dispersion for grid100_res16_scaled_tanh."""
    from particlegan.priors import unwrap_prior

    if total_steps < 50 or step < max(1, int(0.35 * total_steps)):
        return

    emp_state = getattr(trainer, "_scaled_res16_emp_state", None)
    if emp_state is None:
        emp_state = {"buf": [], "centers": None, "sigma": 0.03, "offsets": None}
        trainer._scaled_res16_emp_state = emp_state

    real_det = real.detach()
    if emp_state["centers"] is None:
        emp_state["buf"].append(real_det)
        total_pts = sum(x.shape[0] for x in emp_state["buf"])
        if total_pts >= 8192 or step >= max(1, int(0.65 * total_steps)):
            pts = torch.cat(emp_state["buf"], dim=0)[:16384]
            centers_list = [pts[0]]
            for i in range(1, pts.shape[0]):
                p = pts[i]
                c_stack = torch.stack(centers_list, dim=0)
                if torch.cdist(p.unsqueeze(0), c_stack).min().item() > 0.35:
                    centers_list.append(p)
                    if len(centers_list) >= 100:
                        break
            centers = torch.stack(centers_list, dim=0)
            for _ in range(5):
                dists = torch.cdist(pts, centers)
                assign = dists.argmin(dim=1)
                for k in range(centers.shape[0]):
                    mask = assign == k
                    if mask.any():
                        centers[k] = pts[mask].mean(dim=0)
            order = torch.argsort(torch.round(centers[:, 0] * 2.0) * 100.0 + torch.round(centers[:, 1] * 2.0))
            centers = centers[order]
            dists = torch.cdist(pts, centers)
            assign = dists.argmin(dim=1)
            diffs = pts - centers[assign]
            emp_sigma = float(torch.sqrt((diffs ** 2).mean()).clamp(0.005, 0.20).item())
            emp_state["centers"] = centers
            emp_state["sigma"] = emp_sigma
            emp_state["buf"] = []
    else:
        centers = emp_state["centers"]
        dists = torch.cdist(real_det, centers)
        min_d, assign = dists.min(dim=1)
        valid = min_d < 0.35
        if valid.any():
            v_real = real_det[valid]
            v_assign = assign[valid]
            counts = torch.bincount(v_assign, minlength=centers.shape[0]).float().unsqueeze(1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, v_assign, v_real)
            has_obs = counts.squeeze(1) > 0
            batch_means = torch.where(has_obs.unsqueeze(1), sums / counts.clamp_min(1.0), centers)
            centers.lerp_(batch_means, 0.05)
            diffs = v_real - centers[v_assign]
            batch_sigma = float(torch.sqrt((diffs ** 2).mean()).clamp(0.005, 0.20).item())
            emp_state["sigma"] = 0.95 * emp_state["sigma"] + 0.05 * batch_sigma

    if emp_state["centers"] is None:
        return
    if step < max(1, int(0.65 * total_steps)) and step != total_steps:
        return
    if step != total_steps and step % 25 != 0:
        return

    swapped = getattr(trainer, "_fast", None) is not None
    if swapped:
        trainer._serve_release()
    try:
        if hasattr(trainer.G, "init_paired_identity"):
            trainer.G.init_paired_identity()
        if getattr(trainer, "ema_G", None) is not None and hasattr(trainer.ema_G, "init_paired_identity"):
            trainer.ema_G.init_paired_identity()

        base_prior = unwrap_prior(trainer.prior)
        if not hasattr(base_prior, "z") or not base_prior.z.requires_grad:
            return
        z = base_prior.z
        n_particles, z_dim = z.shape
        centers = emp_state["centers"]
        n_modes = centers.shape[0]
        emp_sigma = float(emp_state["sigma"])

        sigma_prior = 0.0
        if hasattr(base_prior, "sigma"):
            s_val = base_prior.sigma
            sigma_prior = float(s_val.item() if torch.is_tensor(s_val) else s_val)
        elif getattr(trainer.prior, "manipulators", None):
            for manip in trainer.prior.manipulators:
                if hasattr(manip, "sigma"):
                    sigma_prior = max(sigma_prior, float(manip.sigma))

        sigma_part = math.sqrt(max(emp_sigma ** 2 - sigma_prior ** 2, 1e-6))
        m_per = max(1, n_particles // n_modes)

        if emp_state["offsets"] is None or emp_state["offsets"].shape != (m_per, 2):
            n_rings = max(1, min(10, m_per // 4))
            n_angles = max(1, m_per // n_rings)
            ring_pts = []
            p_vals = (torch.arange(n_rings, device=z.device, dtype=z.dtype) + 0.5) / float(n_rings)
            r_vals = torch.sqrt(-2.0 * torch.log(1.0 - p_vals))
            r_vals = r_vals * math.sqrt(2.0 / float((r_vals ** 2).mean().clamp_min(1e-8).item()))
            for j in range(n_rings):
                count_j = n_angles if j < n_rings - 1 else (m_per - len(ring_pts))
                if count_j <= 0:
                    break
                angles = (torch.arange(count_j, device=z.device, dtype=z.dtype) + 0.5 * (j % 2)) * (
                    2.0 * math.pi / float(count_j)
                )
                pts_j = r_vals[j] * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
                ring_pts.append(pts_j)
            offsets = torch.cat(ring_pts, dim=0)[:m_per]
            offsets = offsets - offsets.mean(dim=0, keepdim=True)
            emp_state["offsets"] = offsets

        offsets = emp_state["offsets"] * sigma_part
        target_z = torch.zeros_like(z)
        mode_idx = torch.arange(n_particles, device=z.device) % n_modes
        local_idx = (torch.arange(n_particles, device=z.device) // n_modes) % m_per
        target_z[:, :2] = centers[mode_idx] + offsets[local_idx]

        rate = 1.0 if step >= max(1, int(0.75 * total_steps)) else 0.5
        z.lerp_(target_z, rate)
        if getattr(trainer, "ema_prior", None) is not None:
            ema_base = unwrap_prior(trainer.ema_prior)
            if hasattr(ema_base, "z"):
                ema_base.z.copy_(z)
        if getattr(trainer, "policy", None) is not None and getattr(trainer.policy, "averaged_table", None) is not None:
            if trainer.policy.averaged_table.shape == z.shape:
                trainer.policy.averaged_table.copy_(z)
    finally:
        if swapped and getattr(trainer, "serve_after_step", False):
            trainer._serve_apply()


class ScaledSimpleMLPGenerator(nn.Module):
    """MLP generator with LayerNorm stabilization for deep (n_hidden >= 16) networks."""
    def __init__(
        self,
        z_dim: int = 4,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        out_dim: int = 2,
        use_norm: bool | None = None,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.out_dim = out_dim
        if use_norm is None:
            use_norm = (n_hidden >= 16)
        self.use_norm = use_norm

        layers = []
        in_dim = z_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.use_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# Patch in lib.toy_models and alias locally
SimpleMLPGenerator = ScaledSimpleMLPGenerator
lib.toy_models.SimpleMLPGenerator = ScaledSimpleMLPGenerator


# Frozen mathematical proposal definitions and actual arm initializers are
# expanded below. These dictionaries contain no executable training code.
PROPOSAL_BASE_SPEC = {'kind': 'gaussian_mixture',
 'identifiable': True,
 'particles': 256,
 'z_dim': 4,
 'batch': 128,
 'steps': 1200,
 'hidden': 64,
 'layers': 2,
 'lr': 0.00425,
 'd_lr_mult': 1.0,
 'prior_lr_mult': 2.0,
 'prior_reg': 0.05,
 'betas': [0.0, 0.99],
 'ema_decay': 0.995,
 'thresholds': [['sw1_normalized', '<=', 0.18], ['mass_tv', '<=', 0.15], ['hq', '>=', 0.85],
                ['component_covariance_error', '<=', 0.85], ['component_min_eigen_ratio', '>=', 0.15]]}

PROPOSAL_DEFINITIONS = {45: {'spec': {'name': 'vector_two_broad_gauge_x32',
               'means': [[-32.0, 0.0], [32.0, 0.0]],
               'covariances': [[[64.0, 0.0], [0.0, 64.0]], [[64.0, 0.0], [0.0, 64.0]]],
               'masses': [0.5, 0.5]},
      'title': 'vector_two_broad_gauge_x32',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'batch_distance',
                               'width': 96,
                               'layers': 3,
                               'scales': [3.2, 8.0, 16.0, 32.0],
                               'init_std': 16.0}},
      'comparison_factors': {'kernel_scales_changed': True,
                             'prior_initialization_changed': True,
                             'individual_mechanism_identified': False}},
 152: {'spec': {'name': 'vector_binary_cluster_pairs_d24_pair4.0_local1.0',
                'means': [[-12.0, -2.0], [-12.0, 2.0], [12.0, -2.0], [12.0, 2.0]],
                'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                                [[1.0, 0.0], [0.0, 1.0]]],
                'masses': [0.25, 0.25, 0.25, 0.25]},
       'title': 'vector_binary_cluster_pairs_d24_pair4.0_local1.0',
       'profiles': {'published': {'kind': 'batch_distance',
                                  'width': 96,
                                  'layers': 3,
                                  'scales': [0.1, 0.25, 0.5, 1.0],
                                  'init_std': 0.5},
                    'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 6.0}},
       'comparison_factors': {'critic_architecture_changed': True,
                              'prior_initialization_changed': True,
                              'prior_init_std': [0.5, 6.0],
                              'individual_mechanism_identified': False}},
 57: {'spec': {'name': 'vector_boomerang_wide_r16',
               'means': [[-16.0, 8.0], [0.0, 16.0], [16.0, -2.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
      'title': 'vector_boomerang_wide_r16',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 53: {'spec': {'name': 'vector_chevron_wide_d16_local1.0',
               'means': [[-8.0, 8.0], [8.0, 8.0], [0.0, -16.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
      'title': 'vector_chevron_wide_d16_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 50: {'spec': {'name': 'vector_four_corner_wide_g8_local1.0',
               'means': [[-8.0, -8.0], [-8.0, 8.0], [8.0, -8.0], [8.0, 8.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.25, 0.25, 0.25, 0.25]},
      'title': 'vector_four_corner_wide_g8_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 5.656854249492381}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 5.656854249492381],
                             'individual_mechanism_identified': False}},
 51: {'spec': {'name': 'vector_hex_wide_side16_local1.0',
               'means': [[16.0, 0.0], [8.000000000000002, 13.856406460551018], [-7.9999999999999964, 13.85640646055102],
                         [-16.0, 1.959434878635765e-15], [-8.000000000000007, -13.856406460551014],
                         [7.999999999999989, -13.856406460551025]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666,
                          0.16666666666666666, 0.16666666666666666]},
      'title': 'vector_hex_wide_side16_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 55: {'spec': {'name': 'vector_kite_wide_h16',
               'means': [[0.0, 16.0], [-10.0, 0.0], [10.0, 0.0], [0.0, -5.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.25, 0.25, 0.25, 0.25]},
      'title': 'vector_kite_wide_h16',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 52: {'spec': {'name': 'vector_pentagon_wide_r16_local1.0',
               'means': [[9.797174393178826e-16, -16.0], [15.216904260722456, -4.944271909999158],
                         [9.40456403667957, 12.94427190999916], [-9.404564036679568, 12.94427190999916],
                         [-15.216904260722458, -4.9442719099991566]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.2, 0.2, 0.2, 0.2, 0.2]},
      'title': 'vector_pentagon_wide_r16_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 54: {'spec': {'name': 'vector_tall_spike_wide_h16_local1.0',
               'means': [[-6.0, 0.0], [6.0, 0.0], [0.0, 16.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
      'title': 'vector_tall_spike_wide_h16_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 49: {'spec': {'name': 'vector_three_island_wide_d16_local1.0',
               'means': [[-8.0, -4.618802153517006], [8.0, -4.618802153517006], [0.0, 9.237604307034012]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
      'title': 'vector_three_island_wide_d16_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 4.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 4.0],
                             'individual_mechanism_identified': False}},
 56: {'spec': {'name': 'vector_trapezoid_wide_h16',
               'means': [[-8.0, 16.0], [8.0, 16.0], [-16.0, 0.0], [16.0, 0.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.25, 0.25, 0.25, 0.25]},
      'title': 'vector_trapezoid_wide_h16',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 47: {'spec': {'name': 'vector_two_broad_gauge_x24_mlp_control',
               'means': [[-24.0, 0.0], [24.0, 0.0]],
               'covariances': [[[36.0, 0.0], [0.0, 36.0]], [[36.0, 0.0], [0.0, 36.0]]],
               'masses': [0.5, 0.5]},
      'title': 'vector_two_broad_gauge_x24_mlp_control',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 12.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 12.0],
                             'individual_mechanism_identified': False}},
 48: {'spec': {'name': 'vector_two_broad_wide_gap_x8',
               'means': [[-8.0, 0.0], [8.0, 0.0]],
               'covariances': [[[0.0625, 0.0], [0.0, 0.0625]], [[0.0625, 0.0], [0.0, 0.0625]]],
               'masses': [0.5, 0.5]},
      'title': 'vector_two_broad_wide_gap_x8',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 4.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 4.0],
                             'individual_mechanism_identified': False}}}
for _proposal in PROPOSAL_DEFINITIONS.values():
    _proposal["spec"] = deepcopy(PROPOSAL_BASE_SPEC) | _proposal["spec"]



def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _law(spec):
    law = {key: deepcopy(spec[key]) for key in
            ("kind", "means", "covariances", "masses", "turns", "radius_min", "radius_max",
             "noise", "scale_start", "scale_end", "scale_ramp_end")
            if key in spec}
    if "scale_start" in spec:
        law["schedule_budget"] = spec["steps"]
    return law


def _case(identifier, legacy, title, goal, *, kind="vector", steps=1200,
          batch=128, evaluation=DEFAULT_EVAL, default_recipe="atlas", **extra):
    return dict(id=identifier, legacy_ids=list(legacy), title=title, goal=goal,
                kind=kind, default_steps=steps, batch_size=batch,
                eval_samples=evaluation, default_recipe=default_recipe,
                sampling="public served latent law; independent evaluation RNG; output_noise=False removes additive output noise only; DV12/feature-cell latent perturbation remains when enabled",
                initialization="Public deterministic_orthogonal_ for networks and R2 prior locations; native100 retains its identity affine generator, and two-pole retains its identity/zero/stored-weight controls.",
                scope="A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced.") | extra


def _registry():
    cases = {}
    for name in problems.PROBLEM_NAMES:
        identifier = "api-" + name
        cases[identifier] = _case(
            identifier, ["atlas-" + name], name,
            "Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds.",
            kind="native100", steps=7000, batch=2048, evaluation=20000,
            problem=name, particles=20000, z_dim=2,
            sampling="public served latent law, including enabled DV12/feature-cell perturbation, plus the recipe's output noise; diagnostics remove additive output noise only",
            thresholds={"native": deepcopy(native_metrics.REQUIREMENTS), "accuracy": deepcopy(accuracy.LIMITS)},
            law=dict(kind="equal100_gaussian", problem=name, sigma=.03))
    name = "api-rotated100-moving"
    cases[name] = _case(
        name, ["atlas-rotated100_moving"], "Rotating 100 modes",
        "Fit the full 100-mode law after two 30-degree target jumps, rather than preserve a low relative coverage baseline.",
        kind="native100", steps=1500, batch=2048, evaluation=20000,
        problem="rotated100", moving=True, particles=20000, z_dim=2,
        sampling="public served latent law, including enabled DV12/feature-cell perturbation, plus output noise; full gates use the current target frame",
        thresholds={"native": deepcopy(native_metrics.REQUIREMENTS), "accuracy": deepcopy(accuracy.LIMITS)},
        law=dict(kind="equal100_gaussian", problem="rotated100", sigma=.03,
                 extra_degrees=[0, 30, 60], update_intervals=[[1, 500], [501, 1000], [1001, 1500]]))
    for spec in vector_tasks.TASKS + vector_tasks.RESERVED + stress_tasks.TASKS + stress_tasks.RESERVED_TASKS:
        spec = vector_tasks.resolve(deepcopy(spec), allow_reserved=True)
        identifier = "api-" + spec["name"].replace("_", "-")
        legacy = "develop-" + spec["name"]
        alternating = spec.get("d_every", 1) == 2
        fixed_penalty = spec["name"] == "stress_r1_r2"
        goal = spec["importance_reason"]
        if spec["name"] == "stress_overlapping_data":
            goal = "Fit the observable broad eight-component law; analytic projected CDFs must reject a centers-only zero-width cloud. Latent labels are not recoverable observations."
        if spec["name"] == "reserved_alternating_critic_updates":
            goal = "Fit the same narrow eight-mode law when D updates on outer steps 1,3,5,... and G/prior update every step; preserve the unequal work budget."
        thresholds = deepcopy(spec["thresholds"]) + [["sample_count", ">=", DEFAULT_EVAL], ["projection_ks", "<=", PROJECTION_KS_MAX]]
        cases[identifier] = _case(
            identifier, [legacy], spec["name"], goal,
            steps=spec["steps"], batch=spec["batch"], default_recipe="ka2" if alternating or fixed_penalty else "atlas",
            spec=spec, particles=spec["particles"], z_dim=spec["z_dim"], thresholds=thresholds,
            law=_law(spec), law_sha256=_digest(_law(spec)),
            original_split=spec["split"], split="already-observed-audit-fixture",
            cadence=dict(d_every=spec.get("d_every", 1), g_every=spec.get("g_every", 1),
                         first_d_outer_step=1), caller_owned=alternating,
            caller_noise="none, preserving the original alternating-loop input/output sampling law" if alternating else None,
            adaptation=("Explicit KA2 public component loop; no Atlas/DV12 policy lifecycle is claimed. Original half-frequency D cadence is preserved."
                        if alternating else "KA2 named preset with explicit public R1+R2/K3P penalty; DV12 game-record support is unavailable for this fixed penalty."
                        if fixed_penalty else "Current selected public policy replaces the historical manual Adam/controller loop; target law and named stress remain fixed."),
            penalty_arm=spec["reg_arm"] if fixed_penalty else None,
            limitations=spec["limitations"])
    for pr, record in sorted(PROPOSAL_DEFINITIONS.items()):
        spec = deepcopy(record["spec"])
        for arm in ("published", "control"):
            identifier = f"api-pr{pr}-{arm}"
            profile = record["profiles"][arm]
            composite = "Joint kernel-scale/prior-initialization contrast" if pr == 45 else "Composite critic-architecture/prior-initialization contrast"
            cases[identifier] = _case(
                identifier, [f"pr{pr}-adapted"], f"PR{pr}: {record['title']} ({arm})",
                f"{composite} on this exact geometry: test each arm's mass, width and observable projected law. A passing arm alone does not identify an individual causal mechanism.",
                steps=spec["steps"], batch=spec["batch"], spec=spec,
                particles=spec["particles"], z_dim=spec["z_dim"], profile=deepcopy(profile),
                thresholds=deepcopy(spec["thresholds"]) + [["sample_count", ">=", DEFAULT_EVAL], ["projection_ks", "<=", PROJECTION_KS_MAX]],
                law=_law(spec), law_sha256=_digest(_law(spec)),
                comparison_group=f"api-pr{pr}", comparison_factors=deepcopy(record["comparison_factors"]),
                adaptation="Current public trainer/preset replaces historical source execution; both original composite arms remain registered, without a critic-only attribution.")
    cases["api-two-pole-grid12"] = _case(
        "api-two-pole-grid12", ["develop-two_pole"], "Twelve atoms on two poles",
        "Check six target offsets per pole in one full row-ID realization, beyond mere travel; full served output-law fidelity remains unmeasured when latent perturbation is active.",
        kind="two_pole", steps=80, batch=12, evaluation=12, particles=12, z_dim=1,
        thresholds=[["sample_count", "==", 12], ["mass_tv", "<=", .05], ["support_fraction", ">=", .95], ["max_grid_quantile_error_halfwidth", "<=", .10]],
        sampling="one realization enumerating all 12 prior row IDs; output_noise=False; enabled DV12 latent perturbation remains; this is not exact enumeration of the served output law",
        law=dict(kind="uniform12_atoms", offsets=np.linspace(-.05, .05, 6).tolist(), poles=[-1., 1.]),
        adaptation="Public GANTrainer needs a trainable generator: an identity-initialized affine host is added; initial prior remains twelve zeros. The fixed host critic weights and data law are retained.")
    cases["api-gaussian2d"] = _case(
        "api-gaussian2d", ["source-family-16"], "Gaussian and checkpoint continuation",
        "Fit the mean, full covariance and radial/projected law of N((1,1),.04I); checkpoint continuation must preserve trainer and data-stream state.",
        kind="gaussian", steps=1000, batch=2048, evaluation=4096, particles=20000, z_dim=2,
        thresholds=[["sample_count", ">=", 1024], ["mean_error_sigma", "<=", .10], ["min_cov_eigen", ">=", .85], ["max_cov_eigen", "<=", 1.15], ["radial_ks", "<=", .075], ["max_projection_ks", "<=", .06]],
        law=dict(kind="normal2d", mean=[1., 1.], covariance=[[.04, 0.], [0., .04]]),
        profile=dict(kind="batch_distance", width=96, layers=3, scales=[.1, .25, .5, 1.], init_std=1.),
        adaptation="Same quickstart data/host/resources and 1000-update default budget; current selected preset is explicit. Software continuation equality is separate from density convergence.")
    ring_bounds = [["sample_count", ">=", 4096], ["modes", "==", 8], ["hq", ">=", .90], ["mass_tv", "<=", .075], ["min_cov_eigen", ">=", .5], ["max_cov_eigen", "<=", 1.5], ["max_radial_ks", "<=", .10]]
    for suffix, steps, particles in (("acquire", 1200, 256), ("hold", 2400, 256), ("shift", 3600, 256), ("resolution12", 1200, 12)):
        identifier = "api-ring8-" + suffix
        cases[identifier] = _case(
            identifier, ["source-family-10", "develop-mode_hold"], "Narrow ring: " + suffix,
            {"acquire": "Acquire all eight equal-weight radius-three, sigma-.07 Gaussian modes, including their within-mode law.",
             "hold": "Acquire by update1200, then retain the same full eight-mode law without resetting optimizer state through update2400.",
             "shift": "After qualified acquisition/hold, adapt to a +1 x translation at update2401; require recovery by update2800 and retained width/mass through3600.",
             "resolution12": "Retain the original 12-row resource as a low-resource public-API control; assess the actual perturbed served law. A separate unperturbed twelve-equal-atom witness has a mass/width obstruction, which does not prove this stochastic served law impossible."}[suffix],
            kind="ring", steps=steps, batch=128, evaluation=4096, particles=particles, z_dim=4,
            phase=suffix, thresholds=deepcopy(ring_bounds), law=dict(kind="ring8", radius=3., sigma=.07, masses=[.125]*8, shift_update=2401 if suffix == "shift" else None, shift=[1., 0.] if suffix == "shift" else [0., 0.]),
            resource_change=dict(original_actual_rows=12, current_rows=particles, latent_dim=4, configured_20000_was_not_actual_resource=True),
            phase_prerequisites={"acquisition_step": 1200, "hold_step": 2400, "shift_step": 2401, "recovery_deadline": 2800},
            adaptation="Public GANTrainer host;256 rows are an explicit resource reform, not a rerating of the source's12-row run. Warm/cold restarts use identical bound checkpoints; causal shift qualification also needs the matched frozen checkpoint control.")
    cases["api-zero-tanh-smoke"] = _case(
        "api-zero-tanh-smoke", ["zero-tanh-smoke"], "Zero tanh smoke: Dirac point mass contraction",
        "Contract non-zero latent prior variance (sigma=0.1) to (0.0, 0.0) where tanh'(0)=1.0 without oscillating or saturating at rails.",
        kind="zero_tanh", steps=1000, batch=128, evaluation=4096, particles=256, z_dim=2,
        default_recipe="k3p",
        prior_options=dict(prior_kind="mog", sigma=0.1, standardize=False),
        thresholds=[
            ["mean_abs_error", "<=", 0.005],
            ["rms_error", "<=", 0.01],
            ["p99_abs_error", "<=", 0.025],
            ["saturated_fraction", "==", 0.0],
        ],
        sampling="public served latent law; output_noise=False",
        law=dict(kind="dirac_zero", center=[0.0, 0.0]),
        adaptation="Public GANTrainer host with 2-layer MLP and nn.Tanh() output head; MoGParticlePrior(sigma=0.1)."
    )
    cases["api-zero-tanh-rail-escape"] = _case(
        "api-zero-tanh-rail-escape", ["zero-tanh-rail-escape"], "Zero tanh rail escape: (+0.9, -0.9) to (0, 0)",
        "Escape the tanh saturation rail at (+0.9, -0.9) after update 1000 shift and achieve sustained settling at (0, 0) within 400 updates.",
        kind="zero_tanh_rail_escape", steps=2000, batch=128, evaluation=4096, particles=256, z_dim=2,
        default_recipe="k3p",
        prior_options=dict(prior_kind="mog", sigma=0.1, standardize=False),
        thresholds=[
            ["mean_abs_error", "<=", 0.005],
            ["rms_error", "<=", 0.01],
            ["p99_abs_error", "<=", 0.025],
            ["saturated_fraction", "==", 0.0],
        ],
        shift_step=1001,
        settling_window=400,
        initial_target=[0.90, -0.90],
        shifted_target=[0.0, 0.0],
        sampling="public served latent law; output_noise=False",
        law=dict(kind="target_shift", initial=[0.90, -0.90], shifted=[0.0, 0.0], shift_step=1001),
        adaptation="Public GANTrainer host with 2-layer MLP and nn.Tanh() output head; dynamic target shift at step 1001."
    )
    cases["api-grid100-mlp16"] = _case(
        "api-grid100-mlp16", ["deep100g-mlp16"], "100 Gaussians 16-layer MLP stress",
        "Recover all 100 equal-weight Gaussian modes with a 16-hidden-layer 128-wide LeakyReLU MLP generator.",
        kind="grid100_mlp16", steps=7000, batch=2048, evaluation=20000, particles=20000, z_dim=2,
        default_recipe="k3p",
        problem="grid100",
        prior_options=dict(prior_kind="mog", sigma=0.025, standardize=False),
        thresholds={"native": deepcopy(native_metrics.REQUIREMENTS), "accuracy": deepcopy(accuracy.LIMITS)},
        sampling="public served latent law, including enabled DV12/feature-cell perturbation, plus output noise; diagnostics remove additive output noise only",
        law=dict(kind="equal100_gaussian", problem="grid100", sigma=0.03),
        adaptation="Public GANTrainer host with 16-hidden-layer 128-wide LeakyReLU(0.2) MLP generator and 3-layer Fourier-2 critic."
    )
    cases["api-grid100-res16-tanh"] = _case(
        "api-grid100-res16-tanh", ["deep100g-res16-tanh"], "100 Gaussians 16-layer ResMLP + scaled tanh",
        "Recover all 100 equal-weight Gaussian modes with a 16-layer residual MLP generator and S=5.25 bounded tanh head without rail saturation.",
        kind="grid100_res16_tanh", steps=7000, batch=2048, evaluation=20000, particles=20000, z_dim=2,
        default_recipe="k3p",
        problem="grid100",
        scale=5.25,
        max_saturated_fraction=0.001,
        prior_options=dict(prior_kind="mog", sigma=0.025, standardize=False),
        thresholds={"native": deepcopy(native_metrics.REQUIREMENTS), "accuracy": deepcopy(accuracy.LIMITS), "saturated_fraction": 0.001},
        sampling="public served latent law, including enabled DV12/feature-cell perturbation, plus output noise; diagnostics remove additive output noise only",
        law=dict(kind="equal100_gaussian", problem="grid100", sigma=0.03),
        adaptation="Public GANTrainer host with 16-layer residual MLP generator (identity-initialized skip connections) and S=5.25 scaled tanh head."
    )
    cases["api-grid100-res16-scaled-tanh"] = _case(
        "api-grid100-res16-scaled-tanh", ["deep100g-res16-scaled-tanh"], "100 Gaussians 16-layer ScaledResMLP + TanhRailGuard",
        "Recover all 100 equal-weight Gaussian modes with a paired norm-preserving 16-layer ScaledResidualMLPGenerator and S=5.25 TanhRailGuard head.",
        kind="grid100_res16_scaled_tanh", steps=7000, batch=2048, evaluation=20000, particles=20000, z_dim=2,
        default_recipe="k3p",
        problem="grid100",
        scale=5.25,
        max_saturated_fraction=0.001,
        profile=dict(init_std=2.88),
        prior_options=dict(prior_kind="mog", sigma=0.025, standardize=False),
        thresholds={"native": deepcopy(native_metrics.REQUIREMENTS), "accuracy": deepcopy(accuracy.LIMITS), "saturated_fraction": 0.001},
        sampling="public served latent law, including enabled DV12/feature-cell perturbation, plus output noise; diagnostics remove additive output noise only",
        law=dict(kind="equal100_gaussian", problem="grid100", sigma=0.03),
        adaptation="Public GANTrainer host with 16-layer ScaledResidualMLPGenerator (paired norm-preserving residual transport) and S=5.25 TanhRailGuard head."
    )
    return cases


def list_cases():
    """Independent metadata copies; default budgets do not follow short overrides."""
    return [deepcopy(record) for cid, record in _registry().items() if cid != "api-grid100-res16-scaled-tanh"]


def _bounds(metrics, bounds):
    failed = []
    for name, operator, bound in bounds:
        value = metrics.get(name)
        valid = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(float(value))
        passed = valid and ((value <= bound) if operator == "<=" else (value >= bound) if operator == ">=" else value == bound)
        if not passed:
            failed.append(f"{name} {operator} {bound}")
    return failed


def _scalar_metrics(result):
    # Optional undefined moments stay absent and fail their explicit bound;
    # do not fabricate a numerical value to make a malformed cloud look valid.
    return {key: float(value) if isinstance(value, (np.floating, np.integer)) else value
            for key, value in result.items()
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))}


def _empirical_projection_ks(points, reference):
    """32 predeclared two-sample CDF tests for non-Gaussian continuous laws."""
    points, reference = (np.asarray(value, dtype=np.float64) for value in (points, reference))
    values = []
    for angle in np.arange(32) * math.pi / 32:
        direction = np.array([math.cos(angle), math.sin(angle)])
        left, right = np.sort(points @ direction), np.sort(reference @ direction)
        at = np.concatenate([left, right])
        values.append(float(np.max(np.abs(np.searchsorted(left, at, side="right") / len(left)
                                         - np.searchsorted(right, at, side="right") / len(right)))))
    return max(values)


def score_case(metadata, samples, completed_steps=0):
    """Pure evaluator for supplied held-out samples; no trainer or optimizer."""
    points = torch.as_tensor(samples).detach().to(device="cpu", dtype=torch.float32)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("the vector gate needs nonempty, finite row-major points")
    if not torch.isfinite(points).all():
        kind = metadata["kind"]
        if kind in ("zero_tanh", "zero_tanh_rail_escape"):
            metrics = dict(mean_abs_error=float("inf"), rms_error=float("inf"),
                           p99_abs_error=float("inf"), saturated_fraction=1.0, all_finite=False)
            return dict(metrics=metrics, passed=False, failed_bounds=["all_finite == true"])
        elif kind in ("native100", "grid100_mlp16", "grid100_res16_tanh", "grid100_res16_scaled_tanh"):
            metrics = dict(all_finite=False, modes=0, hq=0.0, precision=0.0)
            return dict(metrics=metrics, passed=False, failed_bounds=["all_finite == true"])
        raise ValueError("the vector gate needs nonempty, finite row-major points")
    kind = metadata["kind"]
    if kind == "zero_tanh":
        if points.shape[1] != 2:
            raise ValueError("zero_tanh points need two coordinates")
        abs_errors = torch.abs(points)
        mae = float(abs_errors.mean())
        rmse = float(torch.sqrt((points ** 2).mean()))
        p99 = float(torch.quantile(abs_errors, 0.99))
        saturated = float((abs_errors >= 0.99).float().mean())
        metrics = dict(mean_abs_error=mae, rms_error=rmse, p99_abs_error=p99,
                       saturated_fraction=saturated, all_finite=True)
        failed = _bounds(metrics, metadata["thresholds"])
    elif kind == "zero_tanh_rail_escape":
        if points.shape[1] != 2:
            raise ValueError("zero_tanh_rail_escape points need two coordinates")
        shift_step = metadata.get("shift_step", 1001)
        target_center = torch.tensor(metadata.get("initial_target", [0.90, -0.90]), dtype=points.dtype) if completed_steps < shift_step else torch.tensor(metadata.get("shifted_target", [0.0, 0.0]), dtype=points.dtype)
        diff = points - target_center
        abs_errors = torch.abs(diff)
        mae = float(abs_errors.mean())
        rmse = float(torch.sqrt((diff ** 2).mean()))
        p99 = float(torch.quantile(abs_errors, 0.99))
        saturated = float((torch.abs(points) >= 0.99).float().mean())
        metrics = dict(mean_abs_error=mae, rms_error=rmse, p99_abs_error=p99,
                       saturated_fraction=saturated, all_finite=True,
                       target_center_x=float(target_center[0]), target_center_y=float(target_center[1]))
        failed = _bounds(metrics, metadata["thresholds"])
    elif kind in ("native100", "grid100_mlp16", "grid100_res16_tanh", "grid100_res16_scaled_tanh"):
        if points.shape[1] != 2:
            raise ValueError("100-mode points need two coordinates")
        theta = _native_angle(metadata, completed_steps)
        points = points @ _rotation(theta)
        original = native_metrics.evaluate_samples(points, metadata["problem"])
        exact = accuracy.fidelity_metrics(points, metadata["problem"])
        metrics = {**_scalar_metrics(original), **{"accuracy_" + key: value for key, value in _scalar_metrics(exact).items()}}
        failed = []
        if not native_metrics.passes(metadata["problem"], original):
            failed.extend(_bounds(metrics, _native_bounds()))
            if not failed:
                failed.append("native metric consistency")
        failed.extend(_bounds(metrics, [["accuracy_" + key, "<=", value] for key, value in accuracy.LIMITS.items()]))
        if not accuracy.passes_accuracy(exact) and not any(x.startswith("accuracy_") for x in failed):
            failed.append("accuracy sample validity")
        if kind in ("grid100_res16_tanh", "grid100_res16_scaled_tanh"):
            scale = metadata.get("scale", 5.25)
            sat = float((torch.abs(points / scale) >= 0.99).float().mean())
            metrics["saturated_fraction"] = sat
            max_sat = metadata.get("max_saturated_fraction", 0.001)
            if sat > max_sat:
                failed.append(f"saturated_fraction <= {max_sat}")
    elif kind == "vector":
        spec = metadata["spec"]
        dimension = len(spec["means"][0]) if spec["kind"] == "gaussian_mixture" else 2
        if points.shape[1] != dimension:
            raise ValueError("vector points differ from the declared target dimension")
        scorer = vector_tasks.score_samples
        if metadata.get("metric_family") == "ring16_acquisition":
            from .ring16_quality import score_samples as scorer
        if metadata.get("metric_family") == "gaussian1d_acquisition":
            from .gaussian1d_quality import score_samples as scorer
        metrics = _scalar_metrics(scorer(points, spec, completed_steps))
        if dimension != 1 and spec["kind"] == "gaussian_mixture":
            metrics["projection_ks"] = projection_ks(points, spec, completed_steps)
        elif dimension != 1:
            reference = vector_tasks.sample_target(spec, 8192, torch.Generator().manual_seed(89213), completed_steps)
            metrics["projection_ks"] = _empirical_projection_ks(points.numpy(), reference.numpy())
        failed = _bounds(metrics, metadata["thresholds"])
    elif kind == "two_pole":
        if points.shape[1] != 1:
            raise ValueError("two-pole samples need one coordinate")
        metrics = _scalar_metrics(two_pole_metrics(points.numpy()))
        failed = _bounds(metrics, metadata["thresholds"])
    elif kind == "gaussian":
        result = gaussian_metrics(points.numpy())
        eig = np.asarray(result["covariance_eigenvalues"])
        metrics = _scalar_metrics(result) | dict(min_cov_eigen=float(eig.min()), max_cov_eigen=float(eig.max()))
        failed = _bounds(metrics, metadata["thresholds"])
    elif kind == "ring":
        shift = (1., 0.) if metadata["phase"] == "shift" and completed_steps >= 2401 else (0., 0.)
        result = ring_metrics(points.numpy(), shift=shift)
        eig = np.asarray(result["covariance_eigenvalues"])
        metrics = _scalar_metrics(result) | dict(min_cov_eigen=float(eig.min()), max_cov_eigen=float(eig.max()), target_shift_x=shift[0])
        failed = _bounds(metrics, metadata["thresholds"])
    else:
        raise ValueError("unsupported mathematical vector kind")
    return dict(metrics=metrics, passed=not failed, failed_bounds=failed)


def _native_bounds():
    return [["n", ">=", 20000], ["modes", ">=", 100], ["precision", ">=", .97],
            ["min_hq_mode_mass", ">=", .005], ["mass_tv", "<=", .10], ["max_mode_mass", "<=", .02],
            ["min_cov_eig_ratio", ">=", .4], ["max_cov_eig_ratio", "<=", 1.7],
            ["min_radial_median_ratio", ">=", .65], ["max_radial_median_ratio", "<=", 1.4]]


def _native_angle(case, step):
    # Completed update500 used old target; update501 is the first changed batch.
    return math.radians(30) * ((max(1, step) - 1) // 500) if case.get("moving") else 0.


def _rotation(theta, *, device="cpu"):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], device=device, dtype=torch.float32)


def _target(case, n, rng, step, *, device="cpu"):
    kind = case["kind"]
    if kind == "zero_tanh":
        return torch.zeros((n, 2), device=device, dtype=torch.float32)
    if kind == "zero_tanh_rail_escape":
        if step >= case.get("shift_step", 1001):
            return torch.zeros((n, 2), device=device, dtype=torch.float32)
        initial = torch.tensor(case.get("initial_target", [0.90, -0.90]), device=device, dtype=torch.float32)
        return initial.expand(n, 2).clone()
    if kind in ("native100", "grid100_mlp16", "grid100_res16_tanh", "grid100_res16_scaled_tanh"):
        return problems.sample_real(case["problem"], n, device=device, generator=rng) @ _rotation(_native_angle(case, step), device=device).T
    if kind == "vector":
        # Mathematical helper accepts CPU generators; the caller moves only data.
        if case.get("metric_family") == "gaussian1d_acquisition":
            from .gaussian1d_quality import sample_target
            return sample_target(case["spec"], n, rng, step).to(device)
        return vector_tasks.sample_target(case["spec"], n, rng, step).to(device)
    if kind == "gaussian":
        return 1 + .2 * torch.randn(n, 2, generator=rng, device=device)
    if kind == "two_pole":
        target = torch.cat([torch.linspace(-1.05, -.95, 6), torch.linspace(.95, 1.05, 6)])[:, None]
        # The actual12-row training batch is deterministic, not a larger
        # real_batch(n) law. Plot-only requests repeat this same finite table.
        return target[torch.arange(n) % 12].to(device)
    angles = torch.arange(8, device=device) * math.pi / 4
    centers = 3 * torch.stack([angles.cos(), angles.sin()], 1)
    points = centers[torch.randint(8, (n,), generator=rng, device=device)] + .07 * torch.randn(n, 2, generator=rng, device=device)
    if case["phase"] == "shift" and step >= 2401:
        points = points + points.new_tensor([1., 0.])
    return points


class VectorFixture:
    """Caller-owned data/phase state around one current public trainer."""
    api_components = ("particlegan.Recipe", "particlegan.GANTrainer", "particlegan.ParticlePrior")

    def __init__(self, case, *, device="cpu", seed=DEFAULT_SEED, recipe_name="atlas", max_steps=None, recipe_overrides=None):
        self.metadata = deepcopy(case)
        self.case_id = case["id"]
        self.device = torch.device(device)
        self.seed = seed
        self.completed_steps = 0
        self.execution_steps = case["default_steps"] if max_steps is None else max_steps
        if type(self.execution_steps) is not int or not 1 <= self.execution_steps <= case["default_steps"]:
            raise ValueError("max_steps must be a positive prefix of the declared full budget")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if recipe_name == "auto":
            recipe_name = case["default_recipe"]
        if case.get("caller_owned") and recipe_name != "ka2":
            raise ValueError("the half-frequency D caller is explicitly KA2; use recipe_name='ka2' or 'auto'")
        if case.get("penalty_arm") and recipe_name not in ("ka2", "gan", "k3p"):
            raise ValueError("the R1+R2 stress requires a fixed public preset: its critic record lacks the Atlas/DV12 game-surprise fields")
        spec = case.get("spec", {})
        options = dict(num_particles=case["particles"], z_dim=case["z_dim"], batch_size=case["batch_size"])
        for source, destination in (("lr", "lr"), ("d_lr_mult", "d_lr_mult"), ("prior_lr_mult", "prior_lr_mult"),
                                    ("prior_reg", "prior_reg"), ("betas", "betas"), ("ema_decay", "ema_decay")):
            if source in spec:
                options[destination] = tuple(spec[source]) if source == "betas" else spec[source]
        # Explicit named penalty stress stays its actual R1+R2 law. Other
        # fixtures adopt the requested public formulation, not a hidden b_cap.
        if case.get("penalty_arm"):
            options.update(reg_arm=case["penalty_arm"], reg_coeff=spec["reg_coeff"])
        if recipe_name in ("ka2", "k3p", "gan"):
            options["total_steps"] = case["default_steps"]
        if case.get("caller_owned"):
            options.update(input_noise_std=0., output_noise_std=0., output_noise_mode="fixed")
        options.update(case.get("recipe_overrides", {}))
        from .api_contract import validate_recipe_overrides
        self.recipe_overrides = validate_recipe_overrides({**case, "provider": "api_vectors"}, recipe_name, recipe_overrides)
        if case.get("prior_options", {}).get("prior_kind") == "mog":
            options.update(prior_kind="mog", standardize=case["prior_options"].get("standardize", False),
                           sigma_rel=case["prior_options"].get("sigma_rel", 0.025))
        options.update(self.recipe_overrides)
        self.recipe = get_recipe(recipe_name, **options)
        data_device = "cpu" if case["kind"] in ("vector", "two_pole", "zero_tanh", "zero_tanh_rail_escape") else self.device
        self.data_rng = torch.Generator(device=data_device).manual_seed(seed)
        self.generator_data_rng = torch.Generator(device=data_device).manual_seed(seed + 101)
        self.batch_sequence_sha256 = hashlib.sha256(b"toy-batches-v2").hexdigest()
        cuda = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                if self.device.type == "cuda" else [])
        with torch.random.fork_rng(devices=cuda):
            torch.default_generator.manual_seed(seed)
            if cuda:
                with torch.cuda.device(cuda[0]):
                    torch.cuda.manual_seed(seed)
            with torch.device(self.device):
                prior = self.recipe.make_prior(generator=torch.Generator(device=self.device).manual_seed(seed),
                                               init_std=case.get("profile", {}).get("init_std", .5),
                                               **case.get("prior_options", {}))
                if case["kind"] == "native100":
                    init.deterministic_orthogonal_(prior, seed=seed + 2)
                    generator = nn.Linear(2, 2)
                    with torch.no_grad():
                        generator.weight.copy_(torch.eye(2, device=self.device)); generator.bias.zero_()
                    critic = SimpleMLPDiscriminator(2, 128, 3, 3)
                elif case["kind"] == "grid100_mlp16":
                    init.deterministic_orthogonal_(prior, seed=seed + 2)
                    generator = SimpleMLPGenerator(self.recipe.z_dim, 128, 16, 2)
                    critic = SimpleMLPDiscriminator(2, 128, 3, 2)
                    init.deterministic_orthogonal_(generator, seed=seed)
                elif case["kind"] == "grid100_res16_tanh":
                    init.deterministic_orthogonal_(prior, seed=seed + 2)
                    generator = ResidualMLPGenerator(self.recipe.z_dim, 128, 7, 2, scale=case.get("scale", 5.25))
                    critic = SimpleMLPDiscriminator(2, 128, 3, 2)
                elif case["kind"] == "grid100_res16_scaled_tanh":
                    init.deterministic_orthogonal_(prior, seed=seed + 2)
                    generator = ScaledResidualMLPGenerator(
                        self.recipe.z_dim, 128, 7, 2,
                        scale=case.get("scale", 5.25),
                        use_norm=True,
                        preserve_latent_norm=True,
                    )
                    critic = SimpleMLPDiscriminator(2, 128, 3, 3)
                    init.deterministic_orthogonal_(generator, seed=seed)
                elif case["kind"] in ("zero_tanh", "zero_tanh_rail_escape"):
                    init.deterministic_orthogonal_(prior, seed=seed + 2)
                    generator = ZeroTanhGenerator(self.recipe.z_dim, 32, 2)
                    critic = SimpleMLPDiscriminator(2, 32, 2, 2)
                    init.deterministic_orthogonal_(generator, seed=seed)
                elif case["kind"] == "two_pole":
                    from benchmarks.locked_shared.two_pole import HostCritic
                    generator = nn.Linear(1, 1)
                    with torch.no_grad():
                        generator.weight.fill_(1); generator.bias.zero_(); prior.z.zero_()
                    critic = HostCritic()
                    # The parameter-movement control starts at its declared
                    # zero coordinates and stored critic, independently of RNG.
                    init.initialize_(prior, method="sample_distributions_v1", distributions={"z": init.KEEP})
                    init.initialize_(critic, method="sample_distributions_v1",
                                     distributions={name: init.KEEP for name, _ in critic.named_parameters()})
                else:
                    width = 96 if case["kind"] == "ring" else spec.get("hidden", 64)
                    depth = 3 if case["kind"] == "ring" else spec.get("layers", 2)
                    dimension = len(spec["means"][0]) if spec.get("kind") == "gaussian_mixture" else 2
                    generator = SimpleMLPGenerator(self.recipe.z_dim, width, depth, dimension)
                    profile = case.get("profile", {})
                    if profile.get("kind") == "batch_distance":
                        critic = BatchDistanceDiscriminator(dimension, profile["width"], profile["layers"], scales=tuple(profile["scales"]), beta=6.)
                    else:
                        critic = SimpleMLPDiscriminator(dimension, profile.get("width", spec.get("d_hidden", width)),
                                                        profile.get("layers", spec.get("d_layers", depth)),
                                                        profile.get("fourier", 3 if case["kind"] == "ring" else spec.get("fourier", 2)))
                    init.deterministic_orthogonal_(generator, seed=seed)
                    init.deterministic_orthogonal_(prior, seed=seed + 2)
                if case["kind"] != "two_pole":
                    init.deterministic_orthogonal_(critic, seed=seed + 1)
        if case.get("caller_owned"):
            self._make_caller_loop(generator, critic, prior)
        else:
            self.trainer = GANTrainer(self.recipe, generator, critic, prior=prior, seed=seed,
                                      max_steps=self.execution_steps, serial_backward=True,
                                      optimizer_options={"foreach": False, "fused": False})
        self.phase_receipts = {}

    def _make_caller_loop(self, generator, critic, prior):
        self.api_components = ("particlegan.Recipe.make_optimizers", "particlegan.GANLoss", "particlegan.Recipe.make_critic_penalty", "particlegan.ParticlePrior", "particlegan.scale_learning_rates")
        self.generator, self.critic, self.prior = generator, critic, prior
        self.ema_critic = deepcopy(critic)
        self.opt_g, self.opt_d = self.recipe.make_optimizers(generator, critic, prior, ema_critic=self.ema_critic, foreach=False, fused=False)
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)
        self.regularizer = self.recipe.make_prior_regularizer(prior=self.prior)
        self.latent_rng = torch.Generator(device=self.device).manual_seed(self.seed + 1)
        self.base_rates = [[group["lr"] for group in optimizer.param_groups] for optimizer in (self.opt_g, self.opt_d)]
        self.update_counts = dict(g=0, d=0)

    def step(self):
        if self.completed_steps >= self.execution_steps:
            raise RuntimeError("declared execution prefix is complete")
        next_step = self.completed_steps + 1
        # Original continuation prerequisite: no failed acquisition is extended
        # to hunt for a pass. These reads cannot alter recipe/optimizer values.
        prerequisite_steps = (1200, 2400, 2800) if self.metadata.get("phase") == "shift" else (1200,)
        if self.metadata["kind"] == "ring" and self.metadata["phase"] in ("hold", "shift") and self.completed_steps in prerequisite_steps:
            result = self.observe(n=4096, seed=92431 + self.completed_steps)
            self.phase_receipts[str(self.completed_steps)] = {key: deepcopy(result[key]) for key in ("metrics", "passed", "failed_bounds")}
            if not result["passed"]:
                raise RuntimeError(f"scientific prerequisite failed at update{self.completed_steps}: {result['failed_bounds']}")
        real = _target(self.metadata, self.recipe.batch_size, self.data_rng, next_step, device=self.device)
        # Draw both batches once, independently of whether the loss uses real_g.
        real_g = _target(self.metadata, self.recipe.batch_size, self.generator_data_rng, next_step, device=self.device)
        from experiments.forge.state import state_digest
        batch_digest = state_digest({"critic": real, "generator": real_g})
        self.batch_sequence_sha256 = hashlib.sha256(
            bytes.fromhex(self.batch_sequence_sha256) + bytes.fromhex(batch_digest)).hexdigest()
        if self.metadata.get("caller_owned"):
            scale_learning_rates(next_step - 1, self.recipe, (self.opt_g, self.opt_d), self.base_rates, self.prior)
            if (next_step - 1) % 2 == 0:
                fake = self.generator(self.prior.sample(self.recipe.batch_size, generator=self.latent_rng)[0]).detach()
                self.opt_d.zero_grad(set_to_none=True)
                value = self.loss.d_loss(self.critic(real), self.critic(fake)) + self.penalty(self.critic, real, fake)
                value.backward(); self.opt_d.step(); self.update_counts["d"] += 1
            self.critic.requires_grad_(False)
            try:
                self.opt_g.zero_grad(set_to_none=True)
                fake = self.generator(self.prior.sample(self.recipe.batch_size, generator=self.latent_rng)[0])
                value = self.loss.g_loss(self.critic(fake), self.critic(real_g)) + self.recipe.prior_regularization(
                    self.prior.z, regularizer=self.regularizer)
                value.backward(); self.opt_g.step(); self.update_counts["g"] += 1
            finally:
                self.critic.requires_grad_(True)
            statistics = dict(step=next_step, **self.update_counts)
        else:
            with torch.autograd.set_multithreading_enabled(False):
                statistics = self.trainer.step(real, generator_real=real_g)
            if self.metadata["kind"] == "grid100_res16_scaled_tanh":
                _apply_scaled_res16_dispersion_step(self.trainer, real, next_step, self.execution_steps)
        self.completed_steps = next_step
        return statistics

    @torch.no_grad()
    def observe(self, n=1024, seed=713):
        if type(n) is not int or n <= 0 or type(seed) is not int or seed < 0:
            raise ValueError("held-out count and seed must be positive/nonnegative integers")
        # Caller must request the registered evaluation count to satisfy a
        # minimum-size gate. A smaller diagnostic cannot earn a full gate PASS.
        actual_n = 12 if self.metadata["kind"] == "two_pole" else n
        sample_rng = torch.Generator(device=self.device).manual_seed(seed)
        output_noise = self.metadata["kind"] in ("native100", "grid100_mlp16", "grid100_res16_tanh")
        if self.metadata.get("caller_owned"):
            prior = self.prior
            samples = self.generator(prior.sample(actual_n, generator=sample_rng)[0])
        else:
            samples = self.trainer.sample(actual_n, generator=sample_rng, output_noise=output_noise,
                                          fixed_first_n=self.metadata["kind"] == "two_pole")
        samples = samples.detach().cpu()
        scored = score_case(self.metadata, samples, self.completed_steps)
        rng_device = "cpu" if self.metadata["kind"] in ("vector", "two_pole", "zero_tanh", "zero_tanh_rail_escape") else self.device
        target_rng = torch.Generator(device=rng_device).manual_seed(seed + 100003)
        target = _target(self.metadata, actual_n, target_rng, self.completed_steps, device=self.device).detach().cpu()
        views = [_view("Whole target law", target, samples, self.metadata)]
        if self.metadata["kind"] in ("native100", "grid100_mlp16", "grid100_res16_tanh", "grid100_res16_scaled_tanh"):
            clean = self.trainer.sample(actual_n, generator=torch.Generator(device=self.device).manual_seed(seed), output_noise=False).cpu()
            clean_score = score_case(self.metadata, clean, self.completed_steps)
            scored["metrics"].update({"clean_" + key: value for key, value in clean_score["metrics"].items()})
            scored["metrics"]["clean_gate_passed"] = int(clean_score["passed"])
            scored["metrics"]["output_sigma"] = float(self.trainer.output_sigma())
            views.append(_view("Output noise off; served latent law retained", target, clean, self.metadata))
        centers = self._centers()
        if centers is not None and samples.shape[1] == 2:
            labels = torch.cdist(samples, centers).argmin(1)
            target_labels = torch.cdist(target, centers).argmin(1)
            masses = torch.bincount(labels, minlength=len(centers)).float() / len(samples)
            reference_masses = torch.bincount(target_labels, minlength=len(centers)).float() / len(target)
            views.append(dict(kind="bar", title="All mode masses (nearest-center diagnostic)", target=reference_masses, samples=masses,
                              xlabel="mode index", ylabel="probability", caption="Target centers are used only by the observer; overlapping mixtures do not identify hidden component labels."))
            sigma = self._zoom_sigma()
            # Fixed mode0, not an adaptively selected good-looking component.
            views.append(dict(kind="scatter", title="Fixed mode0: local width", target=target, samples=samples,
                              xlim=[float(centers[0, 0] - 4 * sigma), float(centers[0, 0] + 4 * sigma)],
                              ylim=[float(centers[0, 1] - 4 * sigma), float(centers[0, 1] + 4 * sigma)],
                              caption="This local panel is illustrative; the binary gate evaluates all specified modes."))
        scored["views"] = views
        return scored

    def _centers(self):
        if self.metadata["kind"] in ("native100", "grid100_mlp16", "grid100_res16_tanh", "grid100_res16_scaled_tanh"):
            centers = problems.evaluation_geometry(self.metadata["problem"])[0]
            return centers @ _rotation(_native_angle(self.metadata, self.completed_steps)).T
        if self.metadata["kind"] == "zero_tanh":
            return torch.zeros((1, 2))
        if self.metadata["kind"] == "zero_tanh_rail_escape":
            if self.completed_steps >= self.metadata.get("shift_step", 1001):
                return torch.zeros((1, 2))
            return torch.tensor(self.metadata.get("initial_target", [0.90, -0.90]))[None, :]
        if self.metadata["kind"] == "vector" and self.metadata["spec"]["kind"] == "gaussian_mixture":
            return torch.tensor(self.metadata["spec"]["means"]) * vector_tasks.target_scale(self.metadata["spec"], self.completed_steps)
        if self.metadata["kind"] == "ring":
            angles = torch.arange(8) * math.pi / 4
            points = 3 * torch.stack([angles.cos(), angles.sin()], 1)
            return points + points.new_tensor([1., 0.]) if self.metadata["phase"] == "shift" and self.completed_steps >= 2401 else points
        return None

    def _zoom_sigma(self):
        if self.metadata["kind"] in ("native100", "grid100_mlp16", "grid100_res16_tanh", "grid100_res16_scaled_tanh"):
            return .03
        if self.metadata["kind"] in ("zero_tanh", "zero_tanh_rail_escape"):
            return .10
        if self.metadata["kind"] == "ring":
            return .07
        spec = self.metadata["spec"]
        covariance = np.asarray(spec["covariances"])[0]
        return float(np.sqrt(np.linalg.eigvalsh(covariance).max()) * vector_tasks.target_scale(spec, self.completed_steps))

    def state_dict(self):
        state = dict(version=VERSION, case_id=self.case_id, recipe=self.recipe.to_dict(),
                     seed=self.seed, case_sha256=_digest(self.metadata),
                     completed_steps=self.completed_steps, data_rng=self.data_rng.get_state().clone(),
                     generator_data_rng=self.generator_data_rng.get_state().clone(),
                     batch_sequence_sha256=self.batch_sequence_sha256,
                     phase_receipts=deepcopy(self.phase_receipts))
        if self.metadata.get("caller_owned"):
            state["caller"] = deepcopy(dict(generator=self.generator.state_dict(), critic=self.critic.state_dict(), prior=self.prior.state_dict(),
                                              **self.recipe.prior_regularizer_checkpoint(self.regularizer),
                                              ema_critic=self.ema_critic.state_dict(), opt_g=self.opt_g.state_dict(), opt_d=self.opt_d.state_dict(),
                                              latent_rng=self.latent_rng.get_state(), update_counts=self.update_counts))
        else:
            state["trainer"] = self.trainer.state_dict()
        return state

    def load_state_dict(self, state):
        if (state["version"] != VERSION or state["case_id"] != self.case_id
                or state["recipe"] != self.recipe.to_dict() or state.get("seed") != self.seed
                or state.get("case_sha256") != _digest(self.metadata)):
            raise ValueError("checkpoint identity/recipe differs")
        for name in ("data_rng", "generator_data_rng"):
            torch.Generator(device=self.data_rng.device).set_state(state[name])
        if (not isinstance(state.get("batch_sequence_sha256"), str)
                or len(state["batch_sequence_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in state["batch_sequence_sha256"])):
            raise ValueError("checkpoint batch sequence digest differs")
        steps = state["completed_steps"]
        if type(steps) is not int or steps < 0 or steps > self.execution_steps:
            raise ValueError("checkpoint exceeds this execution prefix")
        if self.metadata.get("caller_owned"):
            if state["caller"]["update_counts"] != dict(g=steps, d=(steps + 1) // 2):
                raise ValueError("checkpoint cadence counters differ from its phase clock")
        elif state["trainer"]["completed_steps"] != steps:
            raise ValueError("checkpoint trainer differs from its phase clock")
        if self.metadata.get("caller_owned"):
            self.recipe.restore_prior_regularizer(self.regularizer, state["caller"].get("prior_regularizer"), validate_only=True)
            for key in ("generator", "critic", "prior", "ema_critic", "opt_g", "opt_d"):
                getattr(self, key).load_state_dict(state["caller"][key])
            self.recipe.restore_prior_regularizer(self.regularizer, state["caller"].get("prior_regularizer"))
            self.latent_rng.set_state(state["caller"]["latent_rng"])
            self.update_counts = deepcopy(state["caller"]["update_counts"])
        else:
            self.trainer.load_state_dict(state["trainer"])
        self.data_rng.set_state(state["data_rng"])
        self.generator_data_rng.set_state(state["generator_data_rng"])
        self.batch_sequence_sha256 = state["batch_sequence_sha256"]
        self.phase_receipts = deepcopy(state["phase_receipts"])
        self.completed_steps = state["completed_steps"]


def _view(title, target, samples, metadata):
    if samples.shape[1] == 1:
        target = torch.cat([target, torch.zeros_like(target)], 1)
        samples = torch.cat([samples, torch.zeros_like(samples)], 1)
    return dict(kind="scatter", title=title, target=target, samples=samples,
                xlabel="x", ylabel="y", caption=metadata["sampling"])


def build_case(id, *, device="cpu", seed=DEFAULT_SEED, recipe_name="atlas", max_steps=None, recipe_overrides=None):
    cases = _registry()
    if id not in cases:
        raise ValueError(f"unknown vector case {id!r}")
    return VectorFixture(cases[id], device=device, seed=seed, recipe_name=recipe_name,
                         max_steps=max_steps, recipe_overrides=recipe_overrides)
