"""Stratified Blue-Noise Prior (SBNP): its regularizer and the standalone particle prior."""
from __future__ import annotations

import math
from typing import Any, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .._stats import knn_excluding_self, knn_sq_and_median_bandwidth, pairwise_sq_dist
from ..moment_tether import REFERENCE_NAMES, pooled_reference_moments
from .gates import BISPFGates, _DEFAULT_GATES
from .thermal import FROZEN_THERMAL_SCALE, _read_thermal

# StratifiedBlueNoisePrior jitter: plain (not squared) distances; the
# diagonal fill that excludes a row from its own neighbours.
_JITTER_SELF_DISTANCE_FILL = 1e6
# Jitter is skipped at or below this thermal ratio.
_JITTER_MIN_THERMAL_RATIO = 1e-5


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
    dof_low_dim_max: int = 4,
) -> torch.Tensor:
    """Variance/covariance whitening plus scale-free k-NN log-repulsion on latent rows.

    Rows with ``n <= 2 * k_nn`` (plus ``d`` when ``d <= dof_low_dim_max``) return
    zero. The default 4 is the historical hard-coded gate; 0 disables the
    low-dimension term.
    """
    if weight == 0.0 or not len(z):
        return z[:0].sum()
    if z.ndim != 2 or not z.shape[1]:
        raise ValueError("particles must have shape [batch, positive latent dimension]")
    n, d = z.shape
    if type(dof_low_dim_max) is not int or dof_low_dim_max < 0:
        raise ValueError("dof_low_dim_max must be a non-negative integer")
    min_dof = 2 * k_nn + (d if d <= dof_low_dim_max else 0)
    if n <= min_dof:
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
        knn_sq_dist, r_med_sq = knn_sq_and_median_bandwidth(z_sub, k)
        norm_sq_dist = knn_sq_dist / (repel_margin * r_med_sq)
        repel_loss = F.softplus(-torch.log(norm_sq_dist + 1e-8)).mean()
    else:
        repel_loss = z.new_zeros(())

    return float(weight) * (var_loss + cov_loss + float(repel_weight) * repel_loss)


class SBNPPriorRegularizer(nn.Module):
    """Recipe-owned Stratified Blue-Noise Prior regularizer (``prior_regularizer='sbnp'``).

    The loss is ``w * (mean/variance/covariance moments + r * repulsion
    + isotropy terms)``. ``w`` is the declared ``weight`` unless one of
    these declared shape regimes in ``gates`` (``BISPFGates``) applies:

    * small latent (``sbnp_small_latent_*``): ``w = sbnp_small_latent_weight``
      (default 1.0) when that value is not None. Repulsion is multiplied by
      ``sbnp_small_latent_repel_mult`` (25) and uses ``k_nn >= 10`` and
      margin 1.
    * unanchored (no initial-moment reference, ``sbnp_unanchored_weight_*``):
      a weight in ``(0, 0.1)`` becomes ``sbnp_unanchored_weight`` (1.0).

    These defaults reproduce the historical behavior. With
    ``BISPFGates(shape_gates=False)``, or with these regimes disabled, the
    declared ``weight`` is the weight applied. ``effective_weight`` reports the
    weight applied to a row batch.

    ``thermal_source`` (usually the critic optimizer's ``record``, bound by
    ``bind_thermal_source``) supplies the critic's thermal scale. When it is
    unbound, the scale comes from the legacy process channel.
    """

    def __init__(
        self,
        reference: Any = None,
        *,
        weight: float = 1.0,
        target_std: float = 1.0,
        eps: float = 1e-4,
        k_nn: int = 5,
        repel_weight: float = 0.05,
        repel_margin: float = 0.50,
        repel_max_samples: int = 1024,
        gates: BISPFGates | None = None,
        thermal_source: Any = None,
    ) -> None:
        super().__init__()
        if gates is not None and not isinstance(gates, BISPFGates):
            raise TypeError("gates must be a BISPFGates instance")
        self.gates = _DEFAULT_GATES if gates is None else gates
        self.thermal_source = thermal_source
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
        mean, covariance, count = pooled_reference_moments(self.references)
        return mean, covariance, int(count.item())

    def _min_dof(self, d: int) -> int:
        gates = self.gates
        low_dim = gates.active("sbnp_dof_low_dim") and d <= gates.sbnp_dof_low_dim_max
        return 2 * self.k_nn + (d if low_dim else 0)

    def _is_small_latent(self, n: int, d: int) -> bool:
        gates = self.gates
        return (gates.active("sbnp_small_latent") and n <= gates.sbnp_small_latent_max_rows
                and d >= gates.sbnp_small_latent_min_dim
                and (not self.references or sum(getattr(r, "moment_tether_count", 0) for r in self.references)
                     <= gates.sbnp_small_latent_max_reference))

    def _applied_weight(self, small_latent: bool) -> float:
        gates = self.gates
        if small_latent and gates.sbnp_small_latent_weight is not None:
            return float(gates.sbnp_small_latent_weight)
        if (not self.references and gates.active("sbnp_unanchored_weight")
                and 0.0 < self.weight < gates.sbnp_unanchored_weight_below):
            return float(gates.sbnp_unanchored_weight)
        return self.weight

    def effective_weight(self, rows: torch.Tensor) -> float:
        """The loss weight ``forward(rows)`` applies (declared weight or a declared regime weight)."""
        if self.weight == 0.0:
            return 0.0
        n, d = rows.shape
        return self._applied_weight(self._is_small_latent(n, d))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 2 or not rows.shape[1]:
            raise ValueError("particles must have shape [batch, positive latent dimension]")
        if self.weight == 0.0 or not len(rows):
            return rows[:0].sum()
        gates = self.gates
        n, d = rows.shape
        min_dof = self._min_dof(d)

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
            target_cov = initial_covariance
            target_var = variance_scale
        else:
            if n <= min_dof or n < 2:
                return rows[:0].sum()
            mean = rows.mean(0)
            mean_error = rows.new_zeros(())
            target_sigma = torch.full((d,), self.target_std, device=rows.device, dtype=rows.dtype)
            target_var = target_sigma.pow(2) + self.eps
            target_cov = torch.diag(target_var)

        # 1. Per-dimension one-sided variance floor (relu(target_sigma - std)^2)
        centered = rows - mean
        std = (centered.pow(2).mean(0) + self.eps).sqrt()
        var_loss = F.relu(target_sigma - std).pow(2).mean()

        # 2. Covariance loss: full covariance in the declared small anchored
        # regime (sbnp_full_cov_*), otherwise off-diagonal decorrelation.
        if d > 1:
            cov = (centered.T @ centered) / max(n - 1, 1)
            if (self.references and gates.active("sbnp_full_cov")
                    and d <= gates.sbnp_full_cov_max_dim and n <= gates.sbnp_full_cov_max_rows):
                cov_diff = cov - target_cov
                cov_loss = (cov_diff.pow(2) / (target_var.unsqueeze(0) * target_var.unsqueeze(1) + 1e-6)).sum()
            else:
                off_diag = cov - torch.diag(torch.diag(cov))
                cov_loss = (off_diag.pow(2) / (target_var.unsqueeze(0) * target_var.unsqueeze(1) + 1e-6)).sum() / (d * d)
        else:
            cov_loss = rows.new_zeros(())

        # 3. Scale-free k-NN log-repulsion
        m_sub = min(n, self.repel_max_samples)
        z_sub = rows[(torch.arange(m_sub, device=rows.device) * n) // m_sub] if m_sub < n else rows

        # 4. Scale-free Local Covariance Circularization (Isotropy), declared
        # by sbnp_local_iso_*: circularizes local cluster covariance without
        # imposing global centripetal forces.
        local_iso_loss = rows.new_zeros(())
        if (gates.active("sbnp_local_iso") and d == gates.sbnp_local_iso_dim and not self.references
                and m_sub > gates.sbnp_local_iso_above):
            with torch.no_grad():
                k_iso = min(max(self.k_nn, d + 1), m_sub - 1)
                _, iso_knn_idx = knn_excluding_self(pairwise_sq_dist(z_sub), k_iso)
            delta_z = z_sub[iso_knn_idx] - z_sub.unsqueeze(1)  # (m_sub, k_iso, d)
            M = torch.einsum("nkd,nke->nde", delta_z, delta_z) / max(1, k_iso)
            tr_M = torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
            M_iso = (tr_M / d) * torch.eye(d, device=rows.device, dtype=rows.dtype).unsqueeze(0)
            A = M - M_iso
            local_iso_loss = (A.pow(2).sum(dim=(-2, -1)) / ((tr_M.squeeze(-1).squeeze(-1) / d) + 1e-6).pow(2)).mean()

        # 5. Active 2D plane circularization on top-2 eigenvalues (sbnp_active_iso_*)
        active_iso_loss = rows.new_zeros(())
        if gates.active("sbnp_active_iso") and d >= 2 and m_sub > gates.sbnp_active_iso_above:
            row_centered = z_sub - z_sub.mean(dim=0, keepdim=True)
            cov_mat = (row_centered.T @ row_centered) / max(1, m_sub - 1)
            eigs = torch.linalg.eigvalsh(cov_mat)
            l1, l2 = eigs[-1], eigs[-2]
            active_iso_loss = gates.sbnp_active_iso_weight * (l1 - l2).pow(2) / (l1 + l2 + 1e-6).pow(2)

        is_small_latent = self._is_small_latent(n, d)
        if is_small_latent:
            eff_k_nn = min(m_sub - 1, max(self.k_nn, gates.sbnp_small_latent_min_k_nn))
            eff_repel_weight = self.repel_weight * gates.sbnp_small_latent_repel_mult
            effective_margin = gates.sbnp_small_latent_margin
        else:
            eff_k_nn = self.k_nn
            eff_repel_weight = self.repel_weight
            small_margin = gates.active("sbnp_small_margin") and m_sub <= gates.sbnp_small_margin_max_rows
            effective_margin = gates.sbnp_small_margin if small_margin else self.repel_margin
        k = min(eff_k_nn, m_sub - 1)
        if k > 0 and eff_repel_weight > 0.0:
            knn_sq_dist, r_med_sq = knn_sq_and_median_bandwidth(z_sub, k)
            norm_sq_dist = knn_sq_dist / (effective_margin * r_med_sq)
            repel_loss = F.softplus(-torch.log(norm_sq_dist.clamp_min(1e-12))).mean()
        else:
            repel_loss = rows.new_zeros(())

        global_moments = mean_error + var_loss + cov_loss
        scale_w = self._applied_weight(is_small_latent)
        s = _read_thermal(self.thermal_source, rows.device)
        local_iso_weight = gates.sbnp_local_iso_weight
        if s <= FROZEN_THERMAL_SCALE:
            # Frozen thermal state: low-dim rows (sbnp_thermal_full_*) pass the
            # whole loss through with gradient scale s; others are unscaled.
            base_loss = global_moments + eff_repel_weight * repel_loss + local_iso_weight * local_iso_loss + active_iso_loss
            out = scale_w * base_loss
            if (gates.active("sbnp_thermal_full") and d <= gates.sbnp_thermal_full_max_dim
                    and out.requires_grad):
                out = out * s + out.detach() * (1.0 - s)
            return out
        if global_moments.requires_grad and s < 1.0:
            global_moments = global_moments * s + global_moments.detach() * (1.0 - s)
        base_loss = global_moments + eff_repel_weight * repel_loss + local_iso_weight * local_iso_loss + active_iso_loss
        return scale_w * base_loss


class StratifiedBlueNoisePrior(nn.Module):
    """Particle prior with epoch-stratified sampling, local k-NN jitter, and blue-noise repulsion.

    Random draws use the caller's ``generator`` (construction, permutation,
    evaluation indices and jitter). ``generator=None`` draws from the global
    RNG; pass explicit streams for isolated training. Recipes do not build this
    prior.
    """

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
        thermal_source: Any = None,
    ) -> None:
        super().__init__()
        # Critic thermal state (bind_thermal_source); None reads the legacy channel.
        self.thermal_source = thermal_source
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
            t_scale = _read_thermal(self.thermal_source, self.z.device, float(self.thermal_ratio.item()))
            t_ratio = min(float(self.thermal_ratio.item()), t_scale)
            if t_ratio > _JITTER_MIN_THERMAL_RATIO and batch_size > 1:
                with torch.no_grad():
                    k = min(self.k_nn, batch_size - 1)
                    dists = torch.cdist(z_batch.detach(), z_batch.detach(), p=2.0)
                    knn_dists, _ = knn_excluding_self(dists, k, _JITTER_SELF_DISTANCE_FILL)
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
