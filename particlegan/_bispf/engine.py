"""Standalone BISPF engine (SRpGAN + EBI-TCR + gauge + RDTN) and its loss adapter."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from . import transport as _transport
from .prior import StratifiedBlueNoisePrior

# Instance noise is skipped at or below this standard deviation.
_MIN_INSTANCE_NOISE = 1e-6


class BISPFEngine(nn.Module):
    """Stateful engine implementing SRpGAN, EBI-TCR, Zero-Mean Gauge, and RDTN.

    ``ot_target_column_normalize`` selects the OT barycentric target in
    ``g_step``. False (legacy) uses ``B * Pi^T @ x_real``. Its row weights are
    Sinkhorn column masses, about 0.94-1.08 after 5 iterations rather than
    exactly 1, so the target is pulled toward the origin and is not
    translation-equivariant. True divides each row by its mass, giving the
    barycenter ``Pi^T @ x_real / Pi.sum(0)``, which is translation-equivariant.

    ``d_step`` draws instance noise, chord matches and chord positions from its
    ``generator`` argument. ``BISPFLossAdapter`` supplies a declared stream.
    """

    def __init__(
        self,
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        lambda_floor: float = 0.01,
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
        ot_target_column_normalize: bool = False,
    ) -> None:
        super().__init__()
        if type(ot_target_column_normalize) is not bool:
            raise TypeError("ot_target_column_normalize must be a boolean")
        self.ot_target_column_normalize = ot_target_column_normalize
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
        return _transport.sinkhorn_coupling(
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

        if sigma_d > _MIN_INSTANCE_NOISE:
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

        coord_scale = _transport._cloud_scale_factor(x_real, x_fake, data_scale)
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

    def _barycentric_target(self, Pi: torch.Tensor, x_r_flat: torch.Tensor) -> torch.Tensor:
        weights = Pi.shape[1] * Pi.t()
        if self.ot_target_column_normalize:
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.matmul(weights, x_r_flat)

    @torch.no_grad()
    def ot_target(self, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
        """The flattened OT barycentric target that ``g_step`` pulls ``x_fake`` toward."""
        Pi = self._sinkhorn_coupling(x_real.detach(), x_fake.detach())[0]
        return self._barycentric_target(Pi, x_real.detach().reshape(x_real.shape[0], -1))

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
            x_target = self._barycentric_target(Pi, x_r_flat)
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
    """Drop-in loss adapter for BISPFEngine.

    ``d_loss`` passes a training-noise generator to ``BISPFEngine.d_step``, so
    instance noise and chord draws never use the global RNG. The stream is the
    ``generator`` argument of ``d_loss``, else the adapter's ``generator``.
    Without either, the adapter creates its own stream on the batch device,
    seeded with ``seed``. Checkpoint it with ``state_dict()``.
    """

    def __init__(self, engine: BISPFEngine, *, generator: torch.Generator | None = None, seed: int = 0) -> None:
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        self.engine = engine
        self.generator = generator
        self.seed = seed
        self.last_d_stats: Dict[str, float] = {}
        self.last_g_stats: Dict[str, float] = {}

    def _training_generator(self, device: torch.device) -> torch.Generator:
        if self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(self.seed)
        return self.generator

    def d_loss(
        self,
        D: nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        stream = self._training_generator(x_real.device) if generator is None else generator
        loss, self.last_d_stats = self.engine.d_step(D, x_real, x_fake, generator=stream)
        return loss

    def state_dict(self) -> Dict[str, Any]:
        if self.generator is None:
            return {"seed": self.seed, "device": None, "generator": None}
        return {"seed": self.seed, "device": str(self.generator.device), "generator": self.generator.get_state()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if (not isinstance(state, dict) or set(state) != {"seed", "device", "generator"}
                or type(state["seed"]) is not int):
            raise ValueError("invalid BISPFLossAdapter state")
        self.seed = state["seed"]
        if state["generator"] is None:
            self.generator = None
            return
        device = torch.device(state["device"])
        if self.generator is None or self.generator.device != device:
            self.generator = torch.Generator(device=device)
        self.generator.set_state(state["generator"])

    def g_loss(
        self,
        D: nn.Module,
        x_fake: torch.Tensor,
        x_real: torch.Tensor,
        prior: StratifiedBlueNoisePrior | None = None,
    ) -> torch.Tensor:
        loss, self.last_g_stats = self.engine.g_step(D, x_fake, x_real, prior=prior)
        return loss
