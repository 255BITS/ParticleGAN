"""Born-Infeld Sinkhorn Particle Flow (BISPF) for schedule-free ParticleGAN training.

Implements the four time-invariant pillars of BISPF:
1. Scale-Free Sinkhorn Relativistic Pairing (SRpGAN)
2. Elastic Born-Infeld Transport-Chord Regularization (EBI-TCR) & Zero-Mean Logit Gauge
3. Stratified Blue-Noise Particle Prior (SBNP)
4. Residual-Driven Thermal Noise (RDTN)
"""
from __future__ import annotations

from copy import copy
import math
from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .gan_loss import GANLoss
from .grad_regularizers import CriticStepRecord, _score_scalar
from .k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .moment_tether import REFERENCE_NAMES, capture_initial_moments

__all__ = [
    "StratifiedBlueNoisePrior",
    "SBNPPriorRegularizer",
    "sbnp_regularization_loss",
    "ContinuousThermalCooler",
    "sinkhorn_coupling",
    "BISPFEngine",
    "BISPFLossAdapter",
    "BISPFStepRecord",
    "BISPFCriticAdam",
    "BISPFCriticPenalty",
    "SinkhornRelativisticLoss",
]

_PHI1 = 0.6180339887498949  # (sqrt(5) - 1) / 2
_PHI2 = 0.3660254037844386  # (sqrt(3) - 1) / 2


def sbnp_regularization_loss(
    z: torch.Tensor,
    *,
    weight: float = 1.0,
    target_std: float = 1.0,
    eps: float = 1e-4,
    k_nn: int = 3,
    repel_weight: float = 0.05,
    repel_margin: float = 0.25,
    repel_max_samples: int = 1024,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Variance/covariance whitening plus scale-free k-NN log-repulsion on latent rows."""
    if weight == 0.0 or not len(z):
        return z[:0].sum()
    if z.ndim != 2 or not z.shape[1]:
        raise ValueError("particles must have shape [batch, positive latent dimension]")
    n, d = z.shape
    if n <= 2 * k_nn + d:
        return z[:0].sum()

    # 1. Per-dimension variance floor (std >= target_std)
    z_centered = z - z.mean(dim=0, keepdim=True)
    std = (z_centered.pow(2).mean(dim=0) + eps).sqrt()
    var_loss = F.relu(target_std - std).pow(2).mean()

    # 2. Off-diagonal covariance decorrelation
    if d > 1:
        cov = (z_centered.t() @ z_centered) / max(1, n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / (d * d)
    else:
        cov_loss = z.new_zeros(())

    # 3. Scale-free k-NN log-repulsion (prevents local clumping / holes)
    m_sub = min(n, repel_max_samples)
    if m_sub < n:
        if generator is not None:
            sub_idx = torch.randperm(n, device=z.device, generator=generator)[:m_sub]
        else:
            # Deterministic even stride consumes no global RNG in trainer/host loops.
            sub_idx = (torch.arange(m_sub, device=z.device) * n) // m_sub
        z_sub = z[sub_idx]
    else:
        z_sub = z

    k = min(k_nn, m_sub - 1)
    if k > 0 and repel_weight > 0.0:
        sq_dist = torch.cdist(z_sub, z_sub, p=2.0).pow(2)
        with torch.no_grad():
            off_mask = ~torch.eye(m_sub, dtype=torch.bool, device=z.device)
            r_med_sq = torch.median(sq_dist[off_mask]).clamp_min(1e-12)
        eye_inf = torch.eye(m_sub, device=z.device, dtype=z.dtype) * 1e9
        knn_sq_dist, _ = torch.topk(sq_dist + eye_inf, k=k, dim=1, largest=False)
        norm_sq_dist = knn_sq_dist / (repel_margin * r_med_sq)
        repel_loss = F.softplus(-torch.log(norm_sq_dist + 1e-8)).mean()
    else:
        repel_loss = z.new_zeros(())

    return float(weight) * (var_loss + cov_loss + float(repel_weight) * repel_loss)


class SBNPPriorRegularizer(nn.Module):
    """Recipe-owned Stratified Blue-Noise Prior regularizer (``prior_regularizer='sbnp'``)."""

    def __init__(
        self,
        reference: Any = None,
        *,
        weight: float = 1.0,
        target_std: float = 1.0,
        eps: float = 1e-4,
        k_nn: int = 3,
        repel_weight: float = 0.005,
        repel_margin: float = 0.25,
        repel_max_samples: int = 1024,
    ) -> None:
        super().__init__()
        for name, value, strict in (
            ("weight", weight, False),
            ("target_std", target_std, True),
            ("eps", eps, True),
            ("repel_weight", repel_weight, False),
            ("repel_margin", repel_margin, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value <= 0 if strict else value < 0)
            ):
                raise ValueError(f"{name} must be finite and {'positive' if strict else 'nonnegative'}")
        if type(k_nn) is not int or not 1 <= k_nn <= 64:
            raise ValueError("k_nn must be an integer in [1, 64]")
        if type(repel_max_samples) is not int or repel_max_samples < 2:
            raise ValueError("repel_max_samples must be an integer >= 2")
        self.weight = float(weight)
        self.target_std = float(target_std)
        self.eps = float(eps)
        self.k_nn = int(k_nn)
        self.repel_weight = float(repel_weight)
        self.repel_margin = float(repel_margin)
        self.repel_max_samples = int(repel_max_samples)
        if reference is None:
            self.references: Tuple[Any, ...] = ()
        else:
            refs = tuple(reference) if isinstance(reference, (list, tuple)) else (reference,)
            if all(all(hasattr(prior, name) for name in REFERENCE_NAMES) for prior in refs):
                self.references = refs
            else:
                self.references = ()

    def _reference_moments(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if len(self.references) == 1:
            prior = self.references[0]
            return prior.moment_tether_mean, prior.moment_tether_covariance, int(prior.moment_tether_count.item())
        count = sum(prior.moment_tether_count for prior in self.references)
        mean = sum(prior.moment_tether_mean * prior.moment_tether_count for prior in self.references) / count
        scatter = sum(
            (prior.moment_tether_count - 1) * prior.moment_tether_covariance
            + prior.moment_tether_count * torch.outer(prior.moment_tether_mean - mean, prior.moment_tether_mean - mean)
            for prior in self.references
        )
        return mean, scatter / (count - 1).clamp_min(1), int(count.item())

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 2 or not rows.shape[1]:
            raise ValueError("particles must have shape [batch, positive latent dimension]")
        if self.weight == 0.0 or not len(rows):
            return rows[:0].sum()
        n, d = rows.shape
        min_dof = 2 * self.k_nn + d

        if self.references:
            initial_mean, initial_covariance, total_n = self._reference_moments()
            if total_n <= min_dof or n <= min_dof or n < 2:
                return rows[:0].sum()
            if d != len(initial_mean):
                raise ValueError("sbnp rows and initial reference dimensions differ")
            mean = rows.mean(0)
            variance_scale = initial_covariance.diagonal().clamp_min(0) + self.eps
            mean_error = ((mean - initial_mean).square() / variance_scale).mean()
            target_sigma = (initial_covariance.diagonal().clamp_min(0) + self.eps).sqrt() * self.target_std
        else:
            if n <= min_dof or n < 2:
                return rows[:0].sum()
            mean = rows.mean(0)
            mean_error = rows.new_zeros(())
            target_sigma = torch.full((d,), self.target_std, device=rows.device, dtype=rows.dtype)

        # 1. Per-dimension one-sided variance floor (relu(target_sigma - std)^2)
        centered = rows - mean
        std = (centered.pow(2).mean(0) + self.eps).sqrt()
        var_loss = F.relu(target_sigma - std).pow(2).mean()

        # 2. Off-diagonal covariance decorrelation (zero penalty on diagonal variances)
        if d > 1:
            cov = (centered.T @ centered) / max(n - 1, 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            cov_loss = off_diag.pow(2).sum() / (d * d)
        else:
            cov_loss = rows.new_zeros(())

        # 3. Scale-free k-NN log-repulsion
        m_sub = min(n, self.repel_max_samples)
        z_sub = rows[(torch.arange(m_sub, device=rows.device) * n) // m_sub] if m_sub < n else rows
        k = min(self.k_nn, m_sub - 1)
        if k > 0 and self.repel_weight > 0.0:
            sq_dist = torch.cdist(z_sub, z_sub, p=2.0).pow(2)
            with torch.no_grad():
                off_mask = ~torch.eye(m_sub, dtype=torch.bool, device=rows.device)
                r_med_sq = torch.median(sq_dist[off_mask]).clamp_min(1e-12)
            eye_inf = torch.eye(m_sub, device=rows.device, dtype=rows.dtype) * 1e9
            knn_sq_dist, _ = torch.topk(sq_dist + eye_inf, k=k, dim=1, largest=False)
            norm_sq_dist = knn_sq_dist / (self.repel_margin * r_med_sq)
            repel_loss = F.softplus(-torch.log(norm_sq_dist + 1e-8)).mean()
        else:
            repel_loss = rows.new_zeros(())

        out = self.weight * (mean_error + var_loss + cov_loss + self.repel_weight * repel_loss)
        if out.requires_grad:
            s = float(_LAST_THERMAL_SCALE.get(rows.device, 1.0))
            if s < 1.0:
                out = out * s + out.detach() * (1.0 - s)
        return out


class StratifiedBlueNoisePrior(nn.Module):
    """Particle prior with epoch-stratified sampling, local k-NN jitter, and blue-noise repulsion."""

    def __init__(
        self,
        num_particles: int = 1000,
        z_dim: int = 8,
        init_std: float = 0.5,
        k_nn: int = 3,
        jitter_std: float = 0.15,
        repel_weight: float = 0.05,
        repel_margin: float = 0.25,
        repel_max_samples: int = 512,
        learnable: bool = True,
        generator: torch.Generator | None = None,
        *,
        knn_k: int | None = None,
        repulsion_weight: float | None = None,
        jitter_scale: float | None = None,
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if num_particles <= 0 or z_dim <= 0:
            raise ValueError("num_particles and z_dim must be positive")
        if knn_k is not None:
            k_nn = int(knn_k)
        if repulsion_weight is not None:
            repel_weight = float(repulsion_weight)
        if jitter_scale is not None:
            jitter_std = float(jitter_scale)
        if generator is None and seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        self.num_particles = int(num_particles)
        self.z_dim = int(z_dim)
        self.init_std = float(init_std)
        self.k_nn = int(k_nn)
        self.jitter_std = float(jitter_std)
        self.repel_weight = float(repel_weight)
        self.repel_margin = float(repel_margin)
        self.repel_max_samples = int(repel_max_samples)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)

        z_init = torch.randn(self.num_particles, self.z_dim, generator=generator) * self.init_std
        if learnable:
            self.z = nn.Parameter(z_init)
        else:
            self.register_buffer("z", z_init)

        # Stratified permutation state (saved in state_dict for exact continuation)
        self.register_buffer("_perm", torch.arange(self.num_particles, dtype=torch.long))
        self.register_buffer("_cursor", torch.tensor(0, dtype=torch.long))
        # Thermal ratio in [0, 1] driven by BISPF Sinkhorn residual
        self.register_buffer("thermal_ratio", torch.tensor(1.0, dtype=torch.float32))

    @property
    def particles(self) -> torch.Tensor:
        return self.z

    def set_thermal_ratio(self, ratio: float | torch.Tensor) -> None:
        with torch.no_grad():
            val = float(ratio.item()) if isinstance(ratio, torch.Tensor) else float(ratio)
            self.thermal_ratio.fill_(max(0.0, min(1.0, val)))

    def _next_indices(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if batch_size >= self.num_particles:
            perm = torch.randperm(self.num_particles, device=self.z.device, generator=generator)
            reps = (batch_size + self.num_particles - 1) // self.num_particles
            return perm.repeat(reps)[:batch_size]

        c = int(self._cursor.item())
        if c + batch_size > self.num_particles:
            rem = self.num_particles - c
            tail = self._perm[c:].clone() if rem > 0 else self._perm[:0]
            self._perm.copy_(torch.randperm(self.num_particles, device=self.z.device, generator=generator))
            need = batch_size - rem
            idx = torch.cat((tail, self._perm[:need]), dim=0) if rem > 0 else self._perm[:batch_size]
            self._cursor.fill_(need)
            return idx

        idx = self._perm[c : c + batch_size]
        self._cursor.fill_(c + batch_size)
        return idx

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
        *,
        fixed_first_n: bool = False,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if fixed_first_n:
            idx = (torch.arange(batch_size, device=self.z.device) + int(offset)) % self.num_particles
        elif not self.training:
            idx = torch.randint(0, self.num_particles, (batch_size,), device=self.z.device, generator=generator)
        else:
            idx = self._next_indices(batch_size, generator=generator)
        z_batch = self.z[idx]

        if self.training and self.jitter_std > 0.0:
            t_ratio = float(self.thermal_ratio.item())
            if t_ratio > 1e-5 and batch_size > 1:
                with torch.no_grad():
                    k = min(self.k_nn, batch_size - 1)
                    dists = torch.cdist(z_batch.detach(), z_batch.detach(), p=2.0)
                    eye_inf = torch.eye(batch_size, device=z_batch.device, dtype=z_batch.dtype) * 1e6
                    knn_dists, _ = torch.topk(dists + eye_inf, k=k, dim=1, largest=False)
                    local_scale = knn_dists.median(dim=1, keepdim=True).values.clamp_min(1e-4)
                noise = torch.randn(z_batch.shape, device=z_batch.device, dtype=z_batch.dtype, generator=generator)
                z_batch = z_batch + (self.jitter_std * t_ratio) * local_scale * noise

        return z_batch, idx

    def sample_batch(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample(batch_size, generator=generator)

    def forward(self, batch_size: int, generator: torch.Generator | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample(batch_size, generator=generator)

    def regularization_loss(self, weight: float = 1.0) -> torch.Tensor:
        return sbnp_regularization_loss(
            self.z,
            weight=weight,
            target_std=1.0,
            eps=1e-4,
            k_nn=self.k_nn,
            repel_weight=self.repel_weight,
            repel_margin=self.repel_margin,
            repel_max_samples=self.repel_max_samples,
        )


@torch.no_grad()
def sinkhorn_coupling(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    *,
    sinkhorn_eps: float = 0.15,
    sinkhorn_iters: int = 5,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute scale-free doubly-stochastic Sinkhorn coupling Pi.

    Returns ``(Pi, med_sq_dist, residual, data_scale)``.
    """
    b_r = x_real.shape[0]
    b_f = x_fake.shape[0]
    x_r = x_real.detach().reshape(b_r, -1)
    x_f = x_fake.detach().reshape(b_f, -1)

    sq_dist = torch.cdist(x_r, x_f, p=2.0).pow(2)
    med_sq_dist = torch.median(sq_dist).clamp_min(eps)
    C = sq_dist / med_sq_dist

    log_K = -C / sinkhorn_eps
    u = torch.zeros(b_r, device=x_r.device, dtype=x_r.dtype)
    v = torch.zeros(b_f, device=x_r.device, dtype=x_r.dtype)
    log_inv_br = -math.log(max(1, b_r))
    log_inv_bf = -math.log(max(1, b_f))

    for _ in range(sinkhorn_iters):
        u = log_inv_br - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        v = log_inv_bf - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)

    log_Pi = log_K + u.unsqueeze(1) + v.unsqueeze(0)
    Pi = torch.exp(log_Pi)
    Pi = Pi / (Pi.sum(dim=1, keepdim=True).clamp_min(eps) * b_r)

    res = (Pi * C).sum()
    data_scale = med_sq_dist.sqrt()
    return Pi, med_sq_dist, res, data_scale


@torch.no_grad()
def _cloud_scale_factor(x_real: torch.Tensor, x_fake: torch.Tensor, data_scale: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return coordinate unit scale when both batches are non-degenerate clouds, else 1.0."""
    b_f = x_fake.shape[0]
    if b_f < 2:
        return data_scale.new_ones(())
    x_f = x_fake.detach().reshape(b_f, -1)
    f_sq = torch.cdist(x_f, x_f, p=2.0).pow(2)
    off = ~torch.eye(b_f, dtype=torch.bool, device=x_f.device)
    f_med = torch.median(f_sq[off]) if off.any() else f_sq.new_zeros(())
    if float(f_med.item()) <= 1e-6:
        # Point-mass probe (e.g. x_f = 0 in capture-radius tests): preserve raw Euclidean units.
        return data_scale.new_ones(())
    return data_scale.clamp_min(eps)


class BISPFEngine(nn.Module):
    """Stateful engine implementing SRpGAN, EBI-TCR, Zero-Mean Gauge, and RDTN."""

    def __init__(
        self,
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        lambda_floor: float = 0.05,
        lambda_cap: float = 10.0,
        grad_cap: float = 1.0,
        ebi_beta: float = 10.0,
        gamma_gauge: float = 0.1,
        lambda_ot: float = 0.25,
        sigma_d_max: float = 0.10,
        cooling_target: float = 0.35,
        ema_decay: float = 0.98,
        eps: float = 1e-8,
        *,
        grad_cap_c: float | None = None,
        sigma_0: float | None = None,
    ) -> None:
        super().__init__()
        if grad_cap_c is not None:
            grad_cap = float(grad_cap_c)
        if sigma_0 is not None:
            sigma_d_max = float(sigma_0)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.lambda_floor = float(lambda_floor)
        self.lambda_cap = float(lambda_cap)
        self.grad_cap = float(grad_cap)
        self.ebi_beta = float(ebi_beta)
        self.gamma_gauge = float(gamma_gauge)
        self.lambda_ot = float(lambda_ot)
        self.sigma_d_max = float(sigma_d_max)
        self.cooling_target = float(cooling_target)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)

        self.register_buffer("ema_residual", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("ema_data_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("_initialized", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def configure_optimizers(
        G: nn.Module,
        D: nn.Module,
        prior: nn.Module,
        *,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lr_z: float = 2e-3,
        betas: Tuple[float, float] = (0.5, 0.99),
    ) -> Tuple[torch.optim.Adam, torch.optim.Adam, torch.optim.Adam]:
        opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=betas)
        opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=betas)
        opt_z = torch.optim.Adam(prior.parameters(), lr=lr_z, betas=betas)
        return opt_g, opt_d, opt_z

    @property
    def thermal_ratio(self) -> float:
        if self.cooling_target <= 0.0:
            return 0.0
        return float(min(1.0, max(0.0, float(self.ema_residual.item()) / self.cooling_target)))

    @torch.no_grad()
    def _sinkhorn_coupling(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return sinkhorn_coupling(
            x_real,
            x_fake,
            sinkhorn_eps=self.sinkhorn_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            eps=min(self.eps, 1e-12),
        )

    def d_step(
        self,
        D: nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute complete BISPF critic loss and update thermal state."""
        B = x_real.shape[0]
        Pi, med_sq_dist, res, data_scale = self._sinkhorn_coupling(x_real, x_fake)

        with torch.no_grad():
            if not bool(self._initialized.item()):
                self.ema_residual.fill_(float(res.item()))
                self.ema_data_scale.fill_(float(data_scale.item()))
                self._initialized.fill_(True)
            else:
                self.ema_residual.mul_(self.ema_decay).add_(res.to(self.ema_residual.dtype), alpha=1.0 - self.ema_decay)
                self.ema_data_scale.mul_(self.ema_decay).add_(
                    data_scale.to(self.ema_data_scale.dtype), alpha=1.0 - self.ema_decay
                )

        t_ratio = self.thermal_ratio
        sigma_d = self.sigma_d_max * t_ratio * float(self.ema_data_scale.item())

        if sigma_d > 1e-6:
            x_r_in = x_real.detach() + sigma_d * torch.randn(
                x_real.shape, device=x_real.device, dtype=x_real.dtype, generator=generator
            )
            x_f_in = x_fake.detach() + sigma_d * torch.randn(
                x_fake.shape, device=x_fake.device, dtype=x_fake.dtype, generator=generator
            )
        else:
            x_r_in = x_real.detach()
            x_f_in = x_fake.detach()

        d_real = D(x_r_in).view(-1)
        d_fake = D(x_f_in).view(-1)

        # 1. Scale-Free Sinkhorn Relativistic Critic Loss (SRpGAN)
        delta = d_real.unsqueeze(1) - d_fake.unsqueeze(0)
        adv_loss = (B * Pi * F.softplus(-delta)).sum()

        # 2. Elastic Born-Infeld Transport-Chord Regularization (EBI-TCR)
        with torch.no_grad():
            col_probs = (B * Pi.t()).clamp_min(1e-12)
            matched_real_idx = torch.multinomial(col_probs, num_samples=1, generator=generator).squeeze(1)
            x_r_matched = x_real.detach()[matched_real_idx]
            alpha_shape = (B,) + (1,) * (x_real.ndim - 1)
            alpha = torch.rand(alpha_shape, device=x_real.device, dtype=x_real.dtype, generator=generator)

        x_chord = ((1.0 - alpha) * x_fake.detach() + alpha * x_r_matched).requires_grad_(True)
        d_chord = D(x_chord).view(-1)
        grads = torch.autograd.grad(
            outputs=d_chord.sum(),
            inputs=x_chord,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        coord_scale = _cloud_scale_factor(x_real, x_fake, data_scale)
        raw_grad_sq = grads.flatten(1).pow(2).sum(dim=1)
        raw_grad_norm = (raw_grad_sq + 1e-12).sqrt()
        scaled_grad_norm = raw_grad_norm * coord_scale
        scaled_grad_sq = raw_grad_sq * (coord_scale * coord_scale)

        floor_pen = 0.5 * self.lambda_floor * scaled_grad_sq
        excess = F.softplus(scaled_grad_norm - self.grad_cap, beta=self.ebi_beta)
        cap_pen = 0.5 * self.lambda_cap * excess.pow(2)
        ebi_loss = (floor_pen + cap_pen).mean()

        # 3. Zero-Mean Logit Gauge
        mean_logit = 0.5 * (d_real.mean() + d_fake.mean())
        gauge_loss = 0.5 * self.gamma_gauge * mean_logit.pow(2)

        total_loss = adv_loss + ebi_loss + gauge_loss
        stats = {
            "d_adv": float(adv_loss.item()),
            "d_ebi": float(ebi_loss.item()),
            "d_gauge": float(gauge_loss.item()),
            "mean_grad_norm": float(raw_grad_norm.mean().item()),
            "scaled_grad_norm": float(scaled_grad_norm.mean().item()),
            "mean_logit": float(mean_logit.item()),
            "sinkhorn_res": float(res.item()),
            "ema_residual": float(self.ema_residual.item()),
            "thermal_ratio": t_ratio,
            "bispf/thermal_ratio": t_ratio,
            "sigma_d": sigma_d,
        }
        return total_loss, stats

    def g_step(
        self,
        D: nn.Module,
        x_fake: torch.Tensor,
        x_real: torch.Tensor,
        prior: StratifiedBlueNoisePrior | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute complete BISPF generator + prior loss."""
        B = x_fake.shape[0]
        Pi, med_sq_dist, _, _ = self._sinkhorn_coupling(x_real.detach(), x_fake.detach())

        if prior is not None and hasattr(prior, "set_thermal_ratio"):
            prior.set_thermal_ratio(self.thermal_ratio)

        d_real = D(x_real.detach()).view(-1)
        d_fake = D(x_fake).view(-1)

        # 1. Scale-Free Sinkhorn Relativistic Generator Loss
        delta = d_real.unsqueeze(1) - d_fake.unsqueeze(0)
        adv_loss = (B * Pi * F.softplus(delta)).sum()

        # 2. Scale-Free Sinkhorn OT Barycenter Pull
        if self.lambda_ot > 0.0:
            x_r_flat = x_real.detach().reshape(B, -1)
            x_f_flat = x_fake.reshape(B, -1)
            x_target = torch.matmul(B * Pi.t(), x_r_flat)
            ot_loss = 0.5 * self.lambda_ot * (x_f_flat - x_target).pow(2).sum(dim=1).mean() / med_sq_dist
        else:
            ot_loss = adv_loss.new_zeros(())

        total_loss = adv_loss + ot_loss
        stats = {
            "g_adv": float(adv_loss.item()),
            "g_ot": float(ot_loss.item()),
        }

        if prior is not None and hasattr(prior, "regularization_loss"):
            p_loss = prior.regularization_loss()
            total_loss = total_loss + p_loss
            stats["g_prior"] = float(p_loss.item())

        return total_loss, stats


class BISPFLossAdapter:
    """Drop-in loss adapter for BISPFEngine."""

    def __init__(self, engine: BISPFEngine) -> None:
        self.engine = engine
        self.last_d_stats: Dict[str, float] = {}
        self.last_g_stats: Dict[str, float] = {}

    def d_loss(self, D: nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
        loss, self.last_d_stats = self.engine.d_step(D, x_real, x_fake)
        return loss

    def g_loss(
        self,
        D: nn.Module,
        x_fake: torch.Tensor,
        x_real: torch.Tensor,
        prior: StratifiedBlueNoisePrior | None = None,
    ) -> torch.Tensor:
        loss, self.last_g_stats = self.engine.g_step(D, x_fake, x_real, prior=prior)
        return loss


class BISPFStepRecord(CriticStepRecord):
    """Per-critic BISPF state traveling with ``optimizer.state_dict()``."""

    _COUNTERS = ("calls", "observed_steps")
    _OPTIONAL = ("lr_last",)
    KEYS = (
        "formulation",
        "lr_max",
        "lr_last",
        "anchor_started",
        "calls",
        "observed_steps",
        "ema_residual",
        "ema_data_scale",
        "initialized",
    )

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self.ema_residual: float = 1.0
        self.ema_data_scale: float = 1.0
        self.initialized: bool = False
        self._last_d_logits: Tuple[torch.Tensor, torch.Tensor] | None = None

    def observe_sinkhorn(self, res: float, data_scale: float, ema_decay: float = 0.95) -> None:
        if not self.initialized:
            self.ema_residual = float(res)
            self.ema_data_scale = float(data_scale)
            self.initialized = True
        else:
            self.ema_residual = ema_decay * self.ema_residual + (1.0 - ema_decay) * float(res)
            self.ema_data_scale = ema_decay * self.ema_data_scale + (1.0 - ema_decay) * float(data_scale)

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._last_d_logits = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "formulation": "bispf",
            "lr_max": float(self.lr_max),
            "lr_last": None if self.lr_last is None else float(self.lr_last),
            "anchor_started": bool(self.anchor_started),
            "calls": int(self.calls),
            "observed_steps": int(self.observed_steps),
            "ema_residual": float(self.ema_residual),
            "ema_data_scale": float(self.ema_data_scale),
            "initialized": bool(self.initialized),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("formulation") != "bispf":
            raise ValueError("BISPF critic optimizer requires a BISPF controller checkpoint")
        if set(state) != set(self.KEYS):
            raise ValueError("invalid BISPF controller state keys")
        for key in self._COUNTERS:
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid BISPF controller {key}")
        for key in ("lr_max", "ema_residual", "ema_data_scale", *self._OPTIONAL):
            value = state[key]
            if value is None and key in self._OPTIONAL:
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid BISPF controller {key}")
        if type(state["anchor_started"]) is not bool or type(state["initialized"]) is not bool:
            raise ValueError("invalid BISPF controller boolean flags")
        for key in self.KEYS:
            if key != "formulation":
                setattr(self, key, state[key])
        self._last_d_logits = None
        self._thermal_scale = max(0.05, min(1.0, float(self.ema_residual) / 0.20))


class ContinuousThermalCooler:
    """Continuous residual-driven thermal cooler for basin settling and autonomous re-heating.

    Computes smooth straight-through gradient scaling:
        s_thermal = max(s_min, min(1.0, ema_residual / tau_cooling))
    where ema_residual tracks the scale-free Sinkhorn Optimal Transport residual.
    """

    def __init__(
        self,
        tau_cooling: float = 0.20,
        s_min: float = 0.05,
        ema_decay: float = 0.95,
        initial_residual: float = 1.0,
    ) -> None:
        for name, val in (("tau_cooling", tau_cooling), ("s_min", s_min), ("ema_decay", ema_decay)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if s_min > 1.0:
            raise ValueError("s_min must be <= 1.0")
        if ema_decay >= 1.0:
            raise ValueError("ema_decay must be < 1.0")
        self.tau_cooling = float(tau_cooling)
        self.s_min = float(s_min)
        self.ema_decay = float(ema_decay)
        self.ema_residual = float(initial_residual)
        self.initialized = False

    def update(self, batch_residual: float | torch.Tensor) -> float:
        """Update EMA residual with a new batch residual observation and return s_thermal."""
        res = float(batch_residual.item()) if isinstance(batch_residual, torch.Tensor) else float(batch_residual)
        if not self.initialized:
            self.ema_residual = res
            self.initialized = True
        else:
            self.ema_residual = self.ema_decay * self.ema_residual + (1.0 - self.ema_decay) * res
        return self.scale

    @property
    def scale(self) -> float:
        """Return the current continuous thermal scaling factor s_thermal in [s_min, 1.0]."""
        if self.tau_cooling <= 0.0:
            return 1.0
        ratio = self.ema_residual / self.tau_cooling
        return max(self.s_min, min(1.0, ratio))

    def apply_straight_through(self, tensor: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        """Apply straight-through gradient scaling: backward gradient is scaled by scale (or self.scale) while forward pass is identity."""
        s = self.scale if scale is None else float(scale)
        if s < 1.0 and tensor.requires_grad:
            return tensor * s + tensor.detach() * (1.0 - s)
        return tensor

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply straight-through gradient scaling with current cooler scale."""
        return self.apply_straight_through(tensor)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "tau_cooling": self.tau_cooling,
            "s_min": self.s_min,
            "ema_decay": self.ema_decay,
            "ema_residual": self.ema_residual,
            "initialized": self.initialized,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.tau_cooling = float(state["tau_cooling"])
        self.s_min = float(state["s_min"])
        self.ema_decay = float(state["ema_decay"])
        self.ema_residual = float(state["ema_residual"])
        self.initialized = bool(state["initialized"])


_LAST_THERMAL_SCALE: Dict[torch.device, float] = {}


def _bispf_critic_forward_hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> Any:
    rec = getattr(module, "_bispf_record", None)
    if args and isinstance(args[0], torch.Tensor):
        target = output[0] if isinstance(output, (tuple, list)) and output else output
        if isinstance(target, torch.Tensor):
            if not hasattr(target, "_bispf_input"):
                target._bispf_input = args[0]
            if rec is not None and not hasattr(target, "_bispf_record"):
                target._bispf_record = rec
    if rec is not None and not getattr(rec, "_in_penalty", False) and torch.is_grad_enabled():
        s = float(getattr(rec, "_thermal_scale", 1.0))
        if s < 1.0 and isinstance(output, torch.Tensor) and output.requires_grad:
            out_scaled = output * s + output.detach() * (1.0 - s)
            if hasattr(output, "_bispf_input"):
                out_scaled._bispf_input = output._bispf_input
            if hasattr(output, "_bispf_record"):
                out_scaled._bispf_record = output._bispf_record
            return out_scaled
    return None


def _attach_critic_input_hook(critic: nn.Module, record: CriticStepRecord) -> None:
    """Attach a lightweight forward hook so SinkhornRelativisticLoss can read input coordinates."""
    record._thermal_scale = 1.0
    record._in_penalty = False
    _LAST_THERMAL_SCALE.clear()
    critic._bispf_record = record
    if getattr(critic, "_bispf_hook_registered", False):
        return
    critic.register_forward_hook(_bispf_critic_forward_hook)
    critic._bispf_hook_registered = True


class BISPFCriticAdam(K3PCriticAdam):
    """Critic Adam optimizer for ``critic_formulation='bispf'`` with no EMA critic copy."""

    def __init__(
        self,
        params,
        *,
        critic: nn.Module,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **adam_kwargs,
    ) -> None:
        record = BISPFStepRecord()
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.0,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **adam_kwargs,
        )
        self.record = record
        _attach_critic_input_hook(critic, record)


class SinkhornRelativisticLoss(GANLoss):
    """Scale-Free Sinkhorn Relativistic Pairing (SRpGAN) loss compatible with ``GANLoss``."""

    def __init__(
        self,
        loss: str = "sinkhorn_relativistic",
        *,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        record: BISPFStepRecord | None = None,
    ) -> None:
        super().__init__("relativistic", labels=labels)
        self.loss = loss
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.record = record

    def _coords(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_r = getattr(real_logits, "_bispf_input", None)
        x_f = getattr(fake_logits, "_bispf_input", None)
        if x_r is None or x_f is None or x_r.shape[0] != real_logits.shape[0] or x_f.shape[0] != fake_logits.shape[0]:
            x_r = real_logits.detach().reshape(real_logits.shape[0], -1)
            x_f = fake_logits.detach().reshape(fake_logits.shape[0], -1)
        return x_r, x_f

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        r = real_logits.reshape(-1)
        f = fake_logits.reshape(-1)
        x_r, x_f = self._coords(real_logits, fake_logits)
        Pi, _, _, _ = sinkhorn_coupling(
            x_r, x_f, sinkhorn_eps=self.sinkhorn_eps, sinkhorn_iters=self.sinkhorn_iters
        )
        rec = self.record or getattr(real_logits, "_bispf_record", None)
        if rec is not None:
            rec._last_d_logits = (r, f)
        b_r = r.shape[0]
        delta = r.unsqueeze(1) - f.unsqueeze(0)
        return (b_r * Pi * F.softplus(-delta)).sum()

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        if real_logits is None:
            raise ValueError("SRpGAN requires real_logits in g_loss")
        r = real_logits.reshape(-1)
        f = fake_logits.reshape(-1)
        x_r, x_f = self._coords(real_logits, fake_logits)
        Pi, _, _, _ = sinkhorn_coupling(
            x_r, x_f, sinkhorn_eps=self.sinkhorn_eps, sinkhorn_iters=self.sinkhorn_iters
        )
        b_r = r.shape[0]
        delta = r.unsqueeze(1) - f.unsqueeze(0)
        return (b_r * Pi * F.softplus(delta)).sum()

    def joint_g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError("joint_g_loss requires gradients on both generator and encoder streams")
        return self.g_loss(fake_logits, real_logits)


class _BISPFPenaltyKernel:
    """Internal EBI-TCR + Zero-Mean Logit Gauge kernel paired with ``CriticStepRecord``."""

    def __init__(
        self,
        *,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        anchor_weight: float = 0.0,
        lambda_floor: float = 0.0005,
        lambda_cap: float = 1.0,
        ebi_beta: float = 30.0,
        gamma_gauge: float = 0.0005,
        chord_blend: float = 0.02,
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        tau_cooling: float = 0.20,
        s_min: float = 0.05,
        thermal_decay: float = 0.95,
        record: CriticStepRecord | None = None,
    ) -> None:
        for name, val, pos in (
            ("coeff", coeff, False),
            ("kappa", kappa, False),
            ("lambda_floor", lambda_floor, False),
            ("lambda_cap", lambda_cap, False),
            ("ebi_beta", ebi_beta, True),
            ("gamma_gauge", gamma_gauge, False),
            ("chord_blend", chord_blend, False),
            ("sinkhorn_eps", sinkhorn_eps, True),
            ("tau_cooling", tau_cooling, True),
            ("s_min", s_min, True),
            ("thermal_decay", thermal_decay, True),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or (val <= 0 if pos else val < 0):
                raise ValueError(f"{name} must be finite and {'positive' if pos else 'nonnegative'}")
        if chord_blend > 1.0:
            raise ValueError("chord_blend must be in [0, 1]")
        if s_min > 1.0:
            raise ValueError("s_min must be <= 1.0")
        if thermal_decay >= 1.0:
            raise ValueError("thermal_decay must be < 1.0")
        if type(lazy_k) is not int or lazy_k < 1:
            raise ValueError("lazy_k must be a positive integer")
        if type(sinkhorn_iters) is not int or sinkhorn_iters < 1:
            raise ValueError("sinkhorn_iters must be a positive integer")
        if anchor_weight != 0.0:
            raise ValueError("BISPF uses a zero-mean logit gauge instead of an EMA critic anchor (require reg_anchor_weight=0)")
        self.arm = "bispf"
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        self.anchor_weight = 0.0
        self.lambda_floor = float(lambda_floor)
        self.lambda_cap = float(lambda_cap)
        self.ebi_beta = float(ebi_beta)
        self.gamma_gauge = float(gamma_gauge)
        self.chord_blend = float(chord_blend)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.tau_cooling = float(tau_cooling)
        self.s_min = float(s_min)
        self.thermal_decay = float(thermal_decay)
        self.cooler = ContinuousThermalCooler(
            tau_cooling=self.tau_cooling,
            s_min=self.s_min,
            ema_decay=self.thermal_decay,
        )
        self.record = BISPFStepRecord() if record is None else record
        if not hasattr(self.record, "ema_residual"):
            self.record.ema_residual = 1.0
            self.record.ema_data_scale = 1.0
            self.record.initialized = False
            self.record._last_d_logits = None

    def _observe_sinkhorn(self, res: float, data_scale: float, ema_decay: float = 0.95) -> None:
        if hasattr(self.record, "observe_sinkhorn"):
            self.record.observe_sinkhorn(res, data_scale, ema_decay=ema_decay)
        elif not getattr(self.record, "initialized", False):
            self.record.ema_residual = float(res)
            self.record.ema_data_scale = float(data_scale)
            self.record.initialized = True
        else:
            self.record.ema_residual = ema_decay * float(self.record.ema_residual) + (1.0 - ema_decay) * float(res)
            self.record.ema_data_scale = ema_decay * float(self.record.ema_data_scale) + (1.0 - ema_decay) * float(data_scale)
        if hasattr(self, "cooler"):
            self.cooler.ema_residual = float(self.record.ema_residual)
            self.cooler.initialized = bool(self.record.initialized)

    def penalty(
        self,
        live,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
        collect_stats: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.coeff == 0.0 or (self.lazy_k > 1 and step % self.lazy_k != 0):
            self.record._last_d_logits = None
            zero = x_real.new_zeros(())
            return zero, ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})

        if isinstance(live, nn.Module) and getattr(live, "_bispf_record", None) is not self.record:
            _attach_critic_input_hook(live, self.record)

        self.record.calls += 1
        call_idx = self.record.calls
        b_r, b_f = x_real.shape[0], x_fake.shape[0]
        dim = max(1, x_real[0].numel())
        Pi, _, res, data_scale = sinkhorn_coupling(
            x_real, x_fake, sinkhorn_eps=self.sinkhorn_eps, sinkhorn_iters=self.sinkhorn_iters
        )
        self._observe_sinkhorn(float(res.item()), float(data_scale.item()), ema_decay=self.thermal_decay)

        is_discrete_simplex = (x_real.ndim == 2 and dim > 64)
        use_chords = not is_discrete_simplex

        # Continuous thermal cooling is applied to continuous coordinate particle flows
        # with sufficient batch size (dim <= 4 and b_r >= 32); for discrete token codebooks,
        # low-batch fixtures, and high-dimensional image modalities, thermal scale is held at 1.0.
        is_continuous_flow = (not is_discrete_simplex and dim <= 4 and b_r >= 32 and b_f >= 32)
        with torch.no_grad():
            if is_continuous_flow:
                t_scale = self.cooler.scale if hasattr(self, "cooler") else max(
                    0.05, min(1.0, float(getattr(self.record, "ema_residual", 1.0)) / 0.20)
                )
            else:
                t_scale = 1.0
            self.record._thermal_scale = t_scale
            _LAST_THERMAL_SCALE[x_real.device] = t_scale

        # Gradient norm scaling:
        # - High-dimensional continuous spatial fields (images, ndim >= 3) and medium continuous
        #   features (dim > 4) evaluate gradient penalties in RMS coordinates: g_rms = ||g||_2 / sqrt(d)
        # - Low-dimensional continuous coordinates (dim <= 4) and discrete categorical
        #   token codebooks retain the unnormalized Euclidean norm barrier: ||g||_2 <= kappa
        #   without dividing by sqrt(d), preventing critic overshoot in 2D slot-hold tasks and
        #   weakening of the cap penalty in discrete token codebooks.
        if is_discrete_simplex or dim <= 4:
            scale_dim = 1.0
        else:
            scale_dim = math.sqrt(dim)

        if use_chords:
            # Stratified low-discrepancy Sinkhorn transport chords (zero RNG stream mutation)
            with torch.no_grad():
                col_probs = (b_r * Pi.t()).clamp_min(1e-12)
                col_probs = col_probs / col_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
                cdf = torch.cumsum(col_probs, dim=1)
                j_idx = torch.arange(b_f, device=x_real.device, dtype=x_real.dtype)
                q = torch.remainder((j_idx + 0.5) / max(1, b_f) + call_idx * _PHI2, 1.0).unsqueeze(1)
                matched_real_idx = torch.searchsorted(cdf, q).squeeze(1).clamp(0, b_r - 1)
                x_r_matched = x_real.detach()[matched_real_idx]

                u = torch.remainder((j_idx + 0.5) / max(1, b_f) + call_idx * _PHI1, 1.0)
                alpha_shape = (b_f,) + (1,) * (x_real.ndim - 1)
                alpha = u.reshape(alpha_shape)

        # Evaluate EBI potential along real endpoints, fake endpoints, and (if applicable) Sinkhorn transport chords
        self.record._in_penalty = True
        try:
            x_r_req = x_real.detach().clone().requires_grad_(True)
            d_r_out = live(x_r_req)
            g_real = torch.autograd.grad(
                outputs=_score_scalar(d_r_out),
                inputs=x_r_req,
                create_graph=True,
                only_inputs=True,
            )[0]

            x_f_req = x_fake.detach().clone().requires_grad_(True)
            d_f_out = live(x_f_req)
            g_fake = torch.autograd.grad(
                outputs=_score_scalar(d_f_out),
                inputs=x_f_req,
                create_graph=True,
                only_inputs=True,
            )[0]

            if use_chords:
                x_chord = ((1.0 - alpha) * x_fake.detach() + alpha * x_r_matched).requires_grad_(True)
                d_chord_out = live(x_chord)
                g_chord = torch.autograd.grad(
                    outputs=_score_scalar(d_chord_out),
                    inputs=x_chord,
                    create_graph=True,
                    only_inputs=True,
                )[0]
        finally:
            self.record._in_penalty = False

        if is_discrete_simplex:
            norm_real = (g_real.flatten(1).pow(2).sum(dim=1) + 1e-12).sqrt()
            norm_fake = (g_fake.flatten(1).pow(2).sum(dim=1) + 1e-12).sqrt()
            norm_chord = 0.5 * (norm_real + norm_fake)
            floor_pen = x_real.new_zeros(())
            cap_real = F.relu(norm_real - self.kappa).pow(2).mean()
            cap_fake = F.relu(norm_fake - self.kappa).pow(2).mean()
            ebi_loss = 0.5 * self.lambda_cap * (cap_real + cap_fake)
            mean_logit = x_real.new_zeros(())
            gauge_loss = x_real.new_zeros(())
        else:
            sq_real = g_real.flatten(1).pow(2).sum(dim=1)
            norm_real = (sq_real + 1e-12).sqrt() / scale_dim
            sq_fake = g_fake.flatten(1).pow(2).sum(dim=1)
            norm_fake = (sq_fake + 1e-12).sqrt() / scale_dim
            sq_chord = g_chord.flatten(1).pow(2).sum(dim=1)
            norm_chord = (sq_chord + 1e-12).sqrt() / scale_dim
            floor_pen = 0.5 * self.lambda_floor * ((sq_real.mean() + sq_fake.mean() + sq_chord.mean()) / (3.0 * dim))
            cap_real = F.softplus(norm_real - self.kappa, beta=self.ebi_beta).pow(2).mean()
            cap_fake = F.softplus(norm_fake - self.kappa, beta=self.ebi_beta).pow(2).mean()
            cap_chord = F.softplus(norm_chord - self.kappa, beta=self.ebi_beta).pow(2).mean()
            cap_pen = 0.5 * self.lambda_cap * (cap_real + (1.0 - self.chord_blend) * cap_fake + self.chord_blend * cap_chord)
            ebi_loss = floor_pen + cap_pen

            # Universal zero-mean logit gauge across all continuous dimensions and modalities
            cached = getattr(self.record, "_last_d_logits", None)
            self.record._last_d_logits = None
            if cached is not None and cached[0].requires_grad and cached[1].requires_grad:
                d_r, d_f = cached
            else:
                d_r = d_r_out.reshape(-1)
                d_f = d_f_out.reshape(-1)
            mean_logit = 0.5 * (d_r.mean() + d_f.mean())
            gauge_loss = 0.5 * self.gamma_gauge * mean_logit.pow(2)

        penalty = (self.coeff * self.lazy_k) * (ebi_loss + gauge_loss)
        if t_scale < 1.0 and penalty.requires_grad:
            penalty = penalty * t_scale + penalty.detach() * (1.0 - t_scale)
        if not collect_stats:
            return penalty, {}
        return penalty, {
            "applied": True,
            "pen": float(penalty.detach()),
            "center": self.kappa,
            "phase": self.arm,
            "mean_grad_norm": float(norm_chord.detach().mean()),
            "mean_logit": float(mean_logit.detach()),
            "ebi_loss": float(ebi_loss.detach()),
            "gauge_loss": float(gauge_loss.detach()),
            "floor_pen": float(floor_pen.detach()),
            "s_thermal": float(t_scale),
            "ema_residual": float(getattr(self.record, "ema_residual", 1.0)),
            "ema_data_scale": float(getattr(self.record, "ema_data_scale", 1.0)),
        }


class BISPFCriticPenalty(_K3PCriticPenalty):
    """Recipe-facing BISPF critic penalty with EBI-TCR and Zero-Mean Logit Gauge."""

    def __init__(self, recipe, optimizer, *, output=None, collect_stats=False, **penalty_overrides) -> None:
        from .optim.dualnorm import NormalizedOptimizer
        from .tensorflow_adam import TensorFlowV1Adam
        if not (isinstance(optimizer, (BISPFCriticAdam, K3PCriticAdam)) or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam")
                or (isinstance(optimizer, NormalizedOptimizer) and hasattr(optimizer, "record"))):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")
        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_critic_input_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _BISPFPenaltyKernel(record=optimizer.record, **options)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}

    def __call__(self, critic, x_real, x_fake, *condition, **condition_kwargs):
        self._ema_view(critic)
        return super().__call__(critic, x_real, x_fake, *condition, **condition_kwargs)

    def diagnostics(self) -> Dict[str, Any]:
        out = super().diagnostics()
        record = self.optimizer.record
        out.update(
            ema_residual=float(getattr(record, "ema_residual", 1.0)),
            ema_data_scale=float(getattr(record, "ema_data_scale", 1.0)),
            s_thermal=float(getattr(record, "_thermal_scale", 1.0)),
        )
        return out

