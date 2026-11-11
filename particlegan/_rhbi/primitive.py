"""Relativistic Hellinger-Born-Infeld (RHBI) slack-screened transport operator."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RHBILoss(nn.Module):
    """Relativistic Hellinger-Born-Infeld (RHBI) slack-screened transport primitive.

    Unifies nonlocal Born-Infeld optimal transport, hollow self-coupling entropic
    debiasing, and local relativistic critic log-odds ascent through a single
    ``(B + 1) x (B + 1)`` Kantorovich-Hellinger slack coupling with zero RNG draws
    and zero schedule annealing.
    """

    def __init__(
        self,
        weight: float = 1.0,
        rho: float = 2.0,
        tau: float = 2.0,
        eps: float = 0.1,
        iters: int = 8,
        c_max: float = 20.0,
        beta_ell: float = 0.1,
        nn_quantile: float = 0.25,
        *,
        rapidity_screening: bool = True,
        self_coupling: str = "hollow",
        relativistic_weight: float = 0.0,
    ) -> None:
        super().__init__()
        for name, val, positive in (
            ("weight", weight, False),
            ("rho", rho, True),
            ("tau", tau, False),
            ("eps", eps, True),
            ("c_max", c_max, True),
            ("beta_ell", beta_ell, True),
            ("relativistic_weight", relativistic_weight, False),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or (
                val <= 0.0 if positive else val < 0.0
            ):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        if beta_ell > 1.0:
            raise ValueError("beta_ell must be in (0, 1]")
        if isinstance(nn_quantile, bool) or not isinstance(nn_quantile, (int, float)) or not (0.0 < nn_quantile < 1.0):
            raise ValueError("nn_quantile must be in (0, 1)")
        if type(iters) is not int or iters < 1:
            raise ValueError("iters must be a positive integer")
        if type(rapidity_screening) is not bool:
            raise ValueError("rapidity_screening must be a boolean")
        if type(self_coupling) is not str or self_coupling not in ("hollow", "feydy"):
            raise ValueError("self_coupling must be 'hollow' or 'feydy'")

        self.weight = float(weight)
        self.rho = float(rho)
        self.tau = float(tau)
        self.eps = float(eps)
        self.iters = int(iters)
        self.c_max = float(c_max)
        self.beta_ell = float(beta_ell)
        self.nn_quantile = float(nn_quantile)
        self.rapidity_screening = rapidity_screening
        self.self_coupling = self_coupling
        self.relativistic_weight = float(relativistic_weight)
        self.register_buffer("ell_ema", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("ell_initialized", torch.tensor(False, dtype=torch.bool))
        self._ell_init_py: bool = False
        self.last_p_xy_slack: Optional[torch.Tensor] = None

    @staticmethod
    def _sq_euclidean_f64(x_flat: torch.Tensor, y_flat: torch.Tensor) -> torch.Tensor:
        """Exact-gradient, O(B_x * B_y)-memory squared Euclidean matrix in float64.

        In float64, every finite float32 input up to ``FLT_MAX`` (~3.4e38) squares
        without overflow (``(3.4e38)^2 ~ 1.15e77 << DBL_MAX ~ 1.8e308``). Adding the
        detached floor correction keeps forward values strictly ``>= 0`` while
        preserving exact analytical gradients ``2 * (x_i - y_j)`` everywhere,
        including coincident points ``x_i == y_j``.
        """
        x64 = x_flat.double()
        y64 = y_flat.double()
        sq_raw = (
            x64.square().sum(dim=-1, keepdim=True)
            + y64.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * (x64 @ y64.T)
        )
        return sq_raw + (sq_raw.detach().clamp_min(0.0) - sq_raw.detach())

    @staticmethod
    def _born_infeld_pair(
        x: torch.Tensor,
        y: torch.Tensor,
        ell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Overflow-safe, cancellation-free C^infinity Born-Infeld pairwise costs.

        Returns ``(c_phys, c_hat)`` of shape ``[B_x, B_y]`` where
        ``c_phys = sqrt(ell^2 + ||x - y||_2^2) - ell`` gives exact intra-mode W_2
        quadratic moments ``||x - y||_2^2 / (2 * ell)`` and unit Euclidean velocity
        cap ``<= 1`` in the far tail across all coordinate scales, while
        ``c_hat = c_phys / ell`` measures dimensionless local-horizon distance for
        Sinkhorn slack screening.
        """
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        dim = max(1, x_flat.shape[-1])
        ell_safe = ell.clamp_min(1e-6)

        sq64 = RHBILoss._sq_euclidean_f64(x_flat, y_flat)
        ell64 = ell_safe.detach().double()
        rad_phys = torch.sqrt(ell64.square() + sq64)
        val_phys = sq64 / (rad_phys + ell64)
        # Cancellation-free forward value paired with single-sqrt backward derivative.
        c_phys64 = (rad_phys - ell64) + (val_phys - (rad_phys - ell64)).detach()

        fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))
        c_phys = c_phys64.clamp_max(fmax).to(x_flat.dtype)

        with torch.no_grad():
            val_hat = val_phys.detach() / ell64
            ratio_max = torch.finfo(torch.float32).max * (0.25 / math.sqrt(float(dim) + 1.0))
            c_hat = val_hat.clamp_max(ratio_max).to(x_flat.dtype)
        return c_phys, c_hat

    @staticmethod
    @torch.no_grad()
    def _born_infeld_c_hat_detached(
        x: torch.Tensor,
        y: torch.Tensor,
        ell: torch.Tensor,
    ) -> torch.Tensor:
        """Fast pure-float32 overflow-safe detached dimensionless cost ``c_hat``."""
        x_flat = x.detach().reshape(x.shape[0], -1).float()
        y_flat = y.detach().reshape(y.shape[0], -1).float()
        dim = max(1, x_flat.shape[-1])
        ell_f = ell.detach().float().clamp_min(1e-6)
        scale = torch.maximum(
            torch.maximum(x_flat.abs().amax(), y_flat.abs().amax()),
            ell_f,
        ).clamp_min(1.0)
        dist_scaled = torch.cdist(x_flat / scale, y_flat / scale, p=2.0)
        inv_u = ell_f / scale
        # c_hat = sqrt(1 + (dist_scaled / inv_u)^2) - 1 = dist_scaled^2 / (inv_u * (sqrt(inv_u^2 + dist_scaled^2) + inv_u))
        rad = torch.sqrt(inv_u.square() + dist_scaled.square())
        val_hat = dist_scaled.square() / ((rad + inv_u) * inv_u).clamp_min(1e-38)
        ratio_max = torch.finfo(torch.float32).max * (0.25 / math.sqrt(float(dim) + 1.0))
        return val_hat.clamp_max(ratio_max).to(x.dtype)

    @staticmethod
    def _born_infeld_cost(x: torch.Tensor, y: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        """Dimensionless Born-Infeld cost ``sqrt(1 + ||(x - y)/ell||_2^2) - 1``."""
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        dim = max(1, x_flat.shape[-1])
        ell64 = ell.clamp_min(1e-6).double()
        sq64 = RHBILoss._sq_euclidean_f64(x_flat, y_flat)
        u_sq = sq64 / ell64.square()
        rad = torch.sqrt(1.0 + u_sq)
        val = u_sq / (rad + 1.0)
        c64 = (rad - 1.0) + (val - (rad - 1.0)).detach()
        fmax = torch.finfo(x_flat.dtype).max * (0.25 / math.sqrt(float(dim) + 1.0))
        return c64.clamp_max(fmax).to(x_flat.dtype)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._ell_init_py = bool(self.ell_initialized.item())

    @staticmethod
    @torch.no_grad()
    def _batch_ell_quantile_tensor(y_real: torch.Tensor, nn_quantile: float = 0.25) -> torch.Tensor:
        """Zero-sync scale-equivariant nearest-neighbor horizon quantile on device."""
        b_y = y_real.shape[0]
        if b_y <= 1:
            return y_real.new_tensor(1.0, dtype=torch.float32)
        y_flat = y_real.detach().reshape(b_y, -1).float()
        scale = y_flat.abs().amax().clamp_min(1.0)
        y_unit = y_flat / scale
        dist_unit = torch.cdist(y_unit, y_unit, p=2.0)
        dist_unit.fill_diagonal_(0.0)
        d_max = dist_unit.amax()
        thresh = 1e-5 * d_max
        pos_min = torch.where(dist_unit > thresh, dist_unit, float("inf")).amin()

        dist_unit.fill_diagonal_(float("inf"))
        k_nn = min(b_y - 1, max(1, b_y // 200))
        if k_nn == 1:
            nn_dist = dist_unit.amin(dim=-1)
        else:
            nn_dist = torch.topk(dist_unit, k=k_nn, dim=-1, largest=False).values[:, -1]
        q_raw = torch.quantile(nn_dist, nn_quantile)

        rms_unit = y_unit.square().mean().sqrt()

        tie_base = torch.where(torch.isfinite(pos_min), 0.15 * pos_min, 0.05 * d_max)
        nonzero_val = scale * torch.where(q_raw <= thresh, tie_base, q_raw) + 1e-6
        dirac_val = 0.05 * torch.clamp_min(scale * rms_unit, 1.0) + 1e-6
        return torch.where(d_max == 0.0, dirac_val, nonzero_val)

    def _get_ell_tensor(self, device: torch.device, record: Any = None) -> Tuple[torch.Tensor, bool]:
        if record is not None and hasattr(record, "rhbi_ell_initialized"):
            init_cur = bool(record.rhbi_ell_initialized)
            ell_t = getattr(record, "_rhbi_ell_t", None)
            if ell_t is not None and ell_t.device == device:
                return ell_t, init_cur
            val = float(record._rhbi_ell_scalar) if hasattr(record, "_rhbi_ell_scalar") else float(record.rhbi_ell_ema)
            t = torch.tensor(val, dtype=torch.float32, device=device)
            if hasattr(record, "_rhbi_ell_t"):
                record._rhbi_ell_t = t
            return t, init_cur
        return self.ell_ema.to(device=device, dtype=torch.float32), self._ell_init_py

    def _set_ell_tensor(self, ell_f32: torch.Tensor, record: Any = None) -> None:
        if self.ell_ema.device != ell_f32.device:
            self.ell_ema = ell_f32.detach().clone()
            self.ell_initialized = torch.tensor(True, dtype=torch.bool, device=ell_f32.device)
        else:
            self.ell_ema.copy_(ell_f32.detach())
            if not self._ell_init_py:
                self.ell_initialized.fill_(True)
        self._ell_init_py = True
        if record is not None and hasattr(record, "rhbi_ell_initialized"):
            record.rhbi_ell_initialized = True
            if hasattr(record, "_rhbi_ell_t"):
                record._rhbi_ell_t = ell_f32.detach()
            else:
                record.rhbi_ell_ema = float(ell_f32.item())

    @torch.no_grad()
    def _update_ell(
        self,
        y_real: torch.Tensor,
        *,
        record: Any = None,
        allow_update: bool = True,
    ) -> torch.Tensor:
        b_y = y_real.shape[0]
        ell_cur_f32, init_cur = self._get_ell_tensor(y_real.device, record)
        if not allow_update:
            out = ell_cur_f32.clamp_min(1e-6) if init_cur else y_real.new_tensor(1.0, dtype=torch.float32)
            return out.to(dtype=y_real.dtype)
        if b_y <= 1:
            out = ell_cur_f32.clamp_min(1.0) if init_cur else y_real.new_tensor(1.0, dtype=torch.float32)
            return out.to(dtype=y_real.dtype)

        q_f32 = self._batch_ell_quantile_tensor(y_real, self.nn_quantile)
        if not init_cur:
            new_ell_f32 = q_f32
        else:
            new_ell_f32 = (1.0 - self.beta_ell) * ell_cur_f32 + self.beta_ell * q_f32
        self._set_ell_tensor(new_ell_f32, record)
        return new_ell_f32.clamp_min(1e-6).to(dtype=y_real.dtype)

    @torch.no_grad()
    def _screening_energy(self, c_hat: torch.Tensor) -> torch.Tensor:
        c_f = c_hat.float().clamp_min(0.0)
        if self.rapidity_screening:
            two_rho = 2.0 * self.rho
            return two_rho * torch.log1p(c_f / two_rho)
        return c_f.clamp_max(1e6)

    @torch.no_grad()
    def _sinkhorn_slack(
        self,
        c_hat: torch.Tensor,
        d_row: torch.Tensor,
        d_col: torch.Tensor,
        hollow: bool = False,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b_r, b_c = c_hat.shape
        energy = self._screening_energy(c_hat)
        L = c_hat.new_zeros((b_r + 1, b_c + 1), dtype=torch.float32)
        active = -energy / self.eps
        if pair_mask is not None:
            active = active.masked_fill(~pair_mask, float("-inf"))
        if hollow and b_r == b_c:
            active.fill_diagonal_(float("-inf"))
        L[:b_r, :b_c] = active
        L[:b_r, b_c] = (-0.5 * self.rho + 0.5 * self.tau * d_row.float()) / self.eps
        L[b_r, :b_c] = (-0.5 * self.rho - 0.5 * self.tau * d_col.float()) / self.eps

        # Slack reservoir marginal S = 2.0 ensures P[b_r, b_c] = 1 + alpha in [1, 2]
        # stays strictly inside the Birkhoff polytope even when active mass alpha -> 0
        # (disjoint classes, 100% singleton batches, or B = 1 hollow self-coupling),
        # preserving exponential Sinkhorn convergence and pair-local cross-ratios.
        log_row = L.new_full((b_r + 1,), -math.log(max(1, b_r)))
        log_row[b_r] = math.log(2.0)
        log_col = L.new_full((b_c + 1,), -math.log(max(1, b_c)))
        log_col[b_c] = math.log(2.0)

        u = torch.zeros_like(log_row)
        v = torch.zeros_like(log_col)
        for _ in range(self.iters):
            v = log_col - torch.logsumexp(L + u.unsqueeze(1), dim=0)
            u = log_row - torch.logsumexp(L + v.unsqueeze(0), dim=1)
        return torch.exp(L + u.unsqueeze(1) + v.unsqueeze(0)).to(c_hat.dtype)

    @staticmethod
    def _discrete_pair_mask(
        cond_row: Optional[torch.Tensor],
        cond_col: Optional[torch.Tensor],
        b_r: int,
        b_c: int,
    ) -> Optional[torch.Tensor]:
        if cond_row is None or cond_col is None:
            return None
        if cond_row.shape[0] != b_r or cond_col.shape[0] != b_c:
            return None
        r_flat = cond_row.detach().reshape(b_r, -1)
        c_flat = cond_col.detach().reshape(b_c, -1)
        if r_flat.shape[1] != c_flat.shape[1]:
            return None
        return (r_flat.unsqueeze(1) == c_flat.unsqueeze(0)).all(dim=-1)

    def _centered_log_odds(
        self,
        d_fake: torch.Tensor,
        d_real: torch.Tensor,
        *,
        detach_real: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_f = d_fake.shape[0]
        b_r = d_real.shape[0]
        f_flat = d_fake.reshape(b_f, -1).mean(dim=1) if d_fake.ndim > 1 else d_fake.reshape(b_f)
        r_src = d_real.detach() if detach_real else d_real
        r_flat = r_src.reshape(b_r, -1).mean(dim=1) if r_src.ndim > 1 else r_src.reshape(b_r)
        # Compute centered difference in float64 so large constant shifts (+1000.0) cancel cleanly.
        d_bar = 0.5 * (f_flat.detach().double().mean() + r_flat.detach().double().mean())
        f_centered = (f_flat.double() - d_bar).to(f_flat.dtype)
        r_centered = (r_flat.double() - d_bar).to(r_flat.dtype)
        d_fake_tilde = self.c_max * torch.tanh(f_centered / self.c_max)
        d_real_tilde = self.c_max * torch.tanh(r_centered / self.c_max)
        return d_fake_tilde, d_real_tilde

    def critic_slack_loss(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        d_fake: torch.Tensor,
        d_real: torch.Tensor,
        *,
        critic_weight: float = 0.0,
        record: Any = None,
        cond_fake: Optional[torch.Tensor] = None,
        cond_real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optional critic slack-focusing regularizer ``L_{D,slack}``."""
        if critic_weight == 0.0:
            return d_real.new_zeros(())
        b_f = x_fake.shape[0]
        b_r = y_real.shape[0]
        ell = self._update_ell(y_real, record=record, allow_update=False)
        c_hat_xy = self._born_infeld_c_hat_detached(x_fake, y_real, ell)
        d_fake_tilde, d_real_tilde = self._centered_log_odds(d_fake, d_real, detach_real=False)
        mask_xy = self._discrete_pair_mask(cond_fake, cond_real, b_f, b_r)
        P_xy = self._sinkhorn_slack(
            c_hat_xy, d_fake_tilde.detach(), d_real_tilde.detach(), hollow=False, pair_mask=mask_xy
        )
        p_row_slack = P_xy[:b_f, b_r]
        p_col_slack = P_xy[b_f, :b_r]
        return float(critic_weight) * (
            -(p_col_slack * (0.5 * self.tau * d_real_tilde)).sum()
            + (p_row_slack * (0.5 * self.tau * d_fake_tilde)).sum()
        )

    def forward(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        d_fake: torch.Tensor,
        d_real: torch.Tensor,
        feat_fake: Optional[torch.Tensor] = None,
        feat_real: Optional[torch.Tensor] = None,
        *,
        record: Any = None,
        symmetric: bool = False,
        allow_ell_update: bool = True,
        cond_fake: Optional[torch.Tensor] = None,
        cond_real: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Protocol Invariant 1: Exact zero-strength bypass (no buffer mutation or RNG draw)
        if self.weight == 0.0:
            zero = torch.zeros((), device=x_fake.device, dtype=x_fake.dtype)
            if not return_metrics:
                return zero, {}
            ell_cur = (
                float(record.rhbi_ell_ema)
                if record is not None and hasattr(record, "rhbi_ell_initialized")
                else float(self.ell_ema.item())
            )
            return zero, {"rhbi_active_rate": 0.0, "rhbi_ell": ell_cur}

        b_f = x_fake.shape[0]
        b_r = y_real.shape[0]
        ell = self._update_ell(y_real, record=record, allow_update=allow_ell_update)

        y_target = y_real if symmetric else y_real.detach()
        if feat_fake is not None and feat_real is not None:
            f_real = feat_real if symmetric else feat_real.detach()
            x_aug = torch.cat([x_fake.reshape(b_f, -1), feat_fake.reshape(b_f, -1) * ell], dim=-1)
            y_aug = torch.cat([y_target.reshape(b_r, -1), f_real.reshape(b_r, -1) * ell], dim=-1)
        else:
            x_aug = x_fake.reshape(b_f, -1)
            y_aug = y_target.reshape(b_r, -1)

        c_phys_xy, c_hat_xy = self._born_infeld_pair(x_aug, y_aug, ell)
        c_phys_xx, c_hat_xx = self._born_infeld_pair(x_aug, x_aug, ell)

        d_fake_tilde, d_real_tilde = self._centered_log_odds(
            d_fake, d_real, detach_real=not symmetric
        )

        mask_xy = self._discrete_pair_mask(cond_fake, cond_real, b_f, b_r)
        mask_xx = self._discrete_pair_mask(cond_fake, cond_fake, b_f, b_f)
        P_xy = self._sinkhorn_slack(
            c_hat_xy, d_fake_tilde.detach(), d_real_tilde.detach(), hollow=False, pair_mask=mask_xy
        )
        use_hollow = self.self_coupling == "hollow"
        zeros_f = d_fake_tilde.new_zeros(b_f)
        P_xx = self._sinkhorn_slack(
            c_hat_xx, zeros_f, zeros_f, hollow=use_hollow, pair_mask=mask_xx
        )

        P_xy_act = P_xy[:b_f, :b_r]
        P_xy_slack = P_xy[:b_f, b_r]
        self.last_p_xy_slack = P_xy_slack.detach()
        P_xx_act = P_xx[:b_f, :b_f].clone()
        P_xx_act.fill_diagonal_(0.0)

        # Match self-coupling debiasing to each row's active OT transport
        # fraction so slack-routed singletons and empty-batch modes never suffer
        # uncompensated intra-mode repulsion.
        a_f = (float(b_f) * P_xy_act.sum(dim=1)).clamp(0.0, 1.0)
        P_xx_eff = P_xx_act * torch.sqrt(a_f.unsqueeze(1) * a_f.unsqueeze(0))

        transport_xy = (P_xy_act * c_phys_xy).sum()
        if self.relativistic_weight > 0.0:
            active_rel = (
                P_xy_act * F.softplus(d_real_tilde.unsqueeze(0) - d_fake_tilde.unsqueeze(1))
            ).sum()
            slack_rel = (P_xy_slack * F.softplus(-d_fake_tilde)).sum()
            local_critic_slack = self.relativistic_weight * (active_rel + slack_rel)
        else:
            local_critic_slack = -(P_xy_slack * (0.5 * self.tau * d_fake_tilde)).sum()
        debias_xx = -0.5 * (P_xx_eff * c_phys_xx).sum()

        total = transport_xy + local_critic_slack + debias_xx
        if symmetric:
            c_phys_yy, c_hat_yy = self._born_infeld_pair(y_aug, y_aug, ell)
            mask_yy = self._discrete_pair_mask(cond_real, cond_real, b_r, b_r)
            zeros_r = d_real_tilde.new_zeros(b_r)
            P_yy = self._sinkhorn_slack(
                c_hat_yy, zeros_r, zeros_r, hollow=use_hollow, pair_mask=mask_yy
            )
            P_yy_act = P_yy[:b_r, :b_r].clone()
            P_yy_act.fill_diagonal_(0.0)
            a_r = (float(b_r) * P_xy_act.sum(dim=0)).clamp(0.0, 1.0)
            P_yy_eff = P_yy_act * torch.sqrt(a_r.unsqueeze(1) * a_r.unsqueeze(0))
            P_xy_col_slack = P_xy[b_f, :b_r]
            if self.relativistic_weight > 0.0:
                local_real_slack = self.relativistic_weight * (
                    P_xy_col_slack * F.softplus(d_real_tilde)
                ).sum()
            else:
                local_real_slack = (P_xy_col_slack * (0.5 * self.tau * d_real_tilde)).sum()
            debias_yy = -0.5 * (P_yy_eff * c_phys_yy).sum()
            total = total + local_real_slack + debias_yy

        loss = self.weight * total
        if not return_metrics:
            return loss, {}

        with torch.no_grad():
            m_vals = torch.stack([
                P_xy_act.float().sum(),
                (P_xy_act * (c_hat_xy > self.rho).to(P_xy_act.dtype)).float().sum(),
                ell.float(),
            ]).tolist()
        active_rate = float(m_vals[0])

        metrics = {
            "rhbi_active_rate": active_rate,
            "rhbi_slack_rate": 1.0 - active_rate,
            "rhbi_tunnel_flux": float(m_vals[1]),
            "rhbi_ell": float(m_vals[2]),
        }
        return loss, metrics

    def joint_relativistic_loss(
        self,
        x_fake: torch.Tensor,
        y_real: torch.Tensor,
        d_fake: torch.Tensor,
        d_real: torch.Tensor,
        *,
        record: Any = None,
        allow_ell_update: bool = True,
        cond_fake: Optional[torch.Tensor] = None,
        cond_real: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Slack-screened relativistic joint BiGAN loss on ``(G(z), z)`` vs ``(x, E(x))``."""
        if self.weight == 0.0:
            zero = torch.zeros((), device=d_fake.device, dtype=d_fake.dtype)
            if not return_metrics:
                return zero, {}
            ell_cur = (
                float(record.rhbi_ell_ema)
                if record is not None and hasattr(record, "rhbi_ell_initialized")
                else float(self.ell_ema.item())
            )
            return zero, {"rhbi_active_rate": 0.0, "rhbi_ell": ell_cur}

        b_f = d_fake.shape[0]
        b_r = d_real.shape[0]
        ell = self._update_ell(y_real, record=record, allow_update=allow_ell_update)

        with torch.no_grad():
            c_hat_xy = self._born_infeld_c_hat_detached(x_fake, y_real, ell)
            df_sg, dr_sg = self._centered_log_odds(d_fake.detach(), d_real.detach(), detach_real=True)
            mask_xy = self._discrete_pair_mask(cond_fake, cond_real, b_f, b_r)
            P_xy = self._sinkhorn_slack(c_hat_xy, df_sg, dr_sg, hollow=False, pair_mask=mask_xy)
            P_xy_act = P_xy[:b_f, :b_r]
            P_xy_row_slack = P_xy[:b_f, b_r]
            P_xy_col_slack = P_xy[b_f, :b_r]
            self.last_p_xy_slack = P_xy_row_slack

        f_flat = d_fake.reshape(b_f, -1).mean(dim=1) if d_fake.ndim > 1 else d_fake.reshape(b_f)
        r_flat = d_real.reshape(b_r, -1).mean(dim=1) if d_real.ndim > 1 else d_real.reshape(b_r)
        d_bar = 0.5 * (f_flat.detach().mean() + r_flat.detach().mean())

        active_rel = (P_xy_act * F.softplus(r_flat.unsqueeze(0) - f_flat.unsqueeze(1))).sum()
        slack_fake = (P_xy_row_slack * F.softplus(d_bar - f_flat)).sum()
        slack_real = (P_xy_col_slack * F.softplus(r_flat - d_bar)).sum()
        loss = self.weight * (active_rel + slack_fake + slack_real)

        if not return_metrics:
            return loss, {}

        with torch.no_grad():
            m_vals = torch.stack([
                P_xy_act.float().sum(),
                (P_xy_act * (c_hat_xy > self.rho).to(P_xy_act.dtype)).float().sum(),
                ell.float(),
            ]).tolist()
        active_rate = float(m_vals[0])
        metrics = {
            "rhbi_active_rate": active_rate,
            "rhbi_slack_rate": 1.0 - active_rate,
            "rhbi_tunnel_flux": float(m_vals[1]),
            "rhbi_ell": float(m_vals[2]),
        }
        return loss, metrics
