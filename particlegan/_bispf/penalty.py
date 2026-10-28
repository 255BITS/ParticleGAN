"""BISPF critic penalty: EBI-TCR transport chords and the zero-mean logit gauge."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from ..grad_regularizers import CriticStepRecord, _score_scalar
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .._stats import pairwise_sq_dist
from . import transport as _transport
from .critic import BISPFCriticAdam, BISPFStepRecord, _attach_critic_input_hook
from .gates import BISPFGates, _DEFAULT_GATES
from .thermal import FROZEN_THERMAL_SCALE, ContinuousThermalCooler, _publish_thermal

_PHI1 = 0.6180339887498949  # (sqrt(5) - 1) / 2
_PHI2 = 0.3660254037844386  # (sqrt(3) - 1) / 2

class _BISPFPenaltyKernel:
    """Internal EBI-TCR + Zero-Mean Logit Gauge kernel paired with ``CriticStepRecord``.

    Shape- and step-conditioned branches (simplex/image inputs, chord bypass,
    continuous-flow and dense cooling queries, the late 2-D thermal freeze,
    gauge caps and ramps, the 1-D curvature term and cooler batch scaling) are
    declared by ``gates`` (``BISPFGates``). Its defaults reproduce the
    historical behavior.
    """

    def __init__(
        self,
        *,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        anchor_weight: float = 0.0,
        lambda_floor: float = 0.01,
        lambda_cap: float = 1.0,
        ebi_beta: float = 30.0,
        gamma_gauge: float = 0.050,
        chord_blend: float = 0.02,
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        tau_cooling: float = 0.020,
        s_min: float = 1e-5,
        thermal_decay: float = 0.95,
        record: CriticStepRecord | None = None,
        gates: BISPFGates | None = None,
    ) -> None:
        if gates is not None and not isinstance(gates, BISPFGates):
            raise TypeError("gates must be a BISPFGates instance")
        self.gates = _DEFAULT_GATES if gates is None else gates
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
            tau_batch_scaling=self.gates.active("cooler_tau_scaling"),
            reference_batch=self.gates.cooler_reference_batch,
            min_batch=self.gates.cooler_min_batch,
            batch_numerator=self.gates.cooler_batch_numerator,
            max_dim=self.gates.cooler_max_dim,
            default_batch_size=self.gates.cooler_default_query_batch,
            default_dim=self.gates.cooler_default_query_dim,
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
        elif float(res) > float(self.record.ema_residual):
            self.record.ema_residual = float(res)
            self.record.ema_data_scale = float(data_scale)
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
        Pi, _, res, data_scale = _transport.sinkhorn_coupling(
            x_real, x_fake, sinkhorn_eps=self.sinkhorn_eps, sinkhorn_iters=self.sinkhorn_iters
        )
        self._observe_sinkhorn(float(res.item()), float(data_scale.item()), ema_decay=self.thermal_decay)

        gates = self.gates
        is_image_tensor = gates.active("image") and x_real.ndim >= 3
        is_discrete_simplex = gates.active("simplex") and x_real.ndim == 2 and dim >= gates.simplex_min_dim
        # Evaluate transport chords for continuous low-dimensional vector flows (dim <= chord_max_dim).
        # The declared chord bypass skips chords for discrete categorical token codebooks and
        # high-dimensional images when chord_blend is standard (<= chord_blend_override),
        # preventing manifold-blur distortion across non-convex visual boundaries while
        # allowing explicit chord tests (chord_blend > chord_blend_override).
        use_chords = not is_discrete_simplex and (
            not gates.active("chord_bypass")
            or x_real.ndim == 2 and dim <= gates.chord_max_dim
            or self.chord_blend > gates.chord_blend_override)

        is_continuous_flow = (
            gates.active("continuous_flow")
            and not is_discrete_simplex
            and not is_image_tensor
            and 1 <= dim <= gates.continuous_flow_max_dim
            and b_r >= gates.continuous_flow_min_batch
            and b_f >= gates.continuous_flow_min_batch
            and call_idx >= gates.continuous_flow_min_call
        )
        is_dense_multimode = (
            gates.active("dense_multimode")
            and is_continuous_flow
            and b_r >= gates.dense_multimode_min_batch
            and b_f >= gates.dense_multimode_min_batch
            and gates.dense_multimode_min_dim <= dim <= gates.dense_multimode_max_dim
            and call_idx >= gates.dense_multimode_min_call
        )
        # Scale-invariant thermal cooler query
        if is_discrete_simplex or is_image_tensor:
            t_scale = 1.0
        elif is_dense_multimode and hasattr(self, "cooler"):
            t_scale = self.cooler.scale_for(batch_size=gates.dense_multimode_query_batch, dim=dim)
        elif is_continuous_flow:
            t_scale = self.cooler.scale_for(batch_size=b_r, dim=dim) if hasattr(self, "cooler") else max(
                1e-5, min(1.0, float(getattr(self.record, "ema_residual", 1.0)) / self.tau_cooling)
            )
        else:
            t_scale = self.cooler.scale if hasattr(self, "cooler") else 1.0

        if (gates.active("late_thermal") and call_idx >= gates.late_thermal_min_call
                and dim == gates.late_thermal_dim
                and gates.late_thermal_min_real_batch <= b_r <= gates.late_thermal_max_real_batch
                and b_f >= gates.late_thermal_min_fake_batch):
            with torch.no_grad():
                d2 = pairwise_sq_dist(x_real.detach().flatten(1), x_fake.detach().flatten(1))
                norm_d = (d2 / d2.median().clamp_min(1e-12)).sqrt()
                r2f_q = float(norm_d.min(dim=1).values.quantile(gates.late_thermal_quantile).item())
                f2r_q = float(norm_d.min(dim=0).values.quantile(gates.late_thermal_quantile).item())
                prev_scale = float(getattr(self.record, "_thermal_scale", 1.0))
                thresh = gates.late_thermal_threshold_cold if prev_scale < 1.0 else gates.late_thermal_threshold_hot
                if max(r2f_q, f2r_q) < thresh:
                    t_scale = gates.late_thermal_scale

        _publish_thermal(self.record, x_real.device, t_scale)

        # Dimension normalization for Born-Infeld gradient norm (raw norms in the declared
        # rms_exempt low-dimensional regime)
        rms_exempt = gates.active("rms_exempt") and dim <= gates.rms_exempt_max_dim
        scale_dim = 1.0 if (is_discrete_simplex or is_image_tensor or rms_exempt) else math.sqrt(dim)

        if use_chords:
            # Stratified low-discrepancy Sinkhorn transport chords + Density-Directed Transport Bridge
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

        res_ratio = min(1.0, max(0.0, float(getattr(self.record, "ema_residual", 1.0))))
        if is_image_tensor:
            ref_scale = gates.image_scale_ref
            img_scale = min(1.0, max(gates.image_scale_min,
                                     float(getattr(self.record, "ema_data_scale", ref_scale)) / ref_scale))
            kappa_eff = (self.kappa / img_scale) * (gates.image_kappa_base + gates.image_kappa_gain * (1.0 - res_ratio))
        else:
            kappa_eff = self.kappa

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
            d_r = d_r_out.reshape(-1)
            d_f = d_f_out.reshape(-1)
            self.record._last_d_logits = (d_r.detach(), d_f.detach())
        else:
            sq_real = g_real.flatten(1).pow(2).sum(dim=1)
            norm_real = (sq_real + 1e-12).sqrt() / scale_dim
            sq_fake = g_fake.flatten(1).pow(2).sum(dim=1)
            norm_fake = (sq_fake + 1e-12).sqrt() / scale_dim
            cap_real = F.softplus(norm_real - kappa_eff, beta=self.ebi_beta).pow(2).mean()
            cap_fake = F.softplus(norm_fake - kappa_eff, beta=self.ebi_beta).pow(2).mean()

            eff_floor = gates.image_lambda_floor if is_image_tensor else self.lambda_floor
            curv_pen = x_real.new_zeros(())

            if use_chords:
                sq_chord = g_chord.flatten(1).pow(2).sum(dim=1)
                norm_chord = (sq_chord + 1e-12).sqrt() / scale_dim
                floor_pen = 0.5 * eff_floor * ((sq_real.mean() + sq_fake.mean() + sq_chord.mean()) / (3.0 * dim))
                cap_chord = F.softplus(norm_chord - kappa_eff, beta=self.ebi_beta).pow(2).mean()
                cap_pen = 0.5 * self.lambda_cap * (cap_real + (1.0 - self.chord_blend) * cap_fake + self.chord_blend * cap_chord)
                if gates.active("curvature") and is_continuous_flow and dim == gates.curvature_dim:
                    g_r_matched = g_real.detach()[matched_real_idx]
                    g_target = (1.0 - alpha) * g_fake + alpha * g_r_matched
                    curv_pen = gates.curvature_weight * (g_chord - g_target).flatten(1).pow(2).sum(dim=1).mean()
                ebi_loss = floor_pen + cap_pen + curv_pen
            else:
                norm_chord = 0.5 * (norm_real + norm_fake)
                floor_pen = 0.5 * eff_floor * ((sq_real.mean() + sq_fake.mean()) / (2.0 * dim))
                cap_pen = 0.5 * self.lambda_cap * (cap_real + cap_fake)
                ebi_loss = floor_pen + cap_pen

            d_r = d_r_out.reshape(-1)
            d_f = d_f_out.reshape(-1)
            self.record._last_d_logits = (d_r.detach(), d_f.detach())
            mean_logit = 0.5 * (d_r.mean() + d_f.mean())
            if is_image_tensor:
                eff_gamma_gauge = gates.image_gamma_gauge
            elif gates.active("small_batch_gauge") and b_r < gates.small_batch_gauge_below:
                eff_gamma_gauge = min(self.gamma_gauge, gates.small_batch_gauge_cap)
            elif gates.active("gauge_ramp") and is_continuous_flow and call_idx >= gates.gauge_ramp_start_call:
                ramp = min(1.0, max(0.0, (call_idx - float(gates.gauge_ramp_start_call))
                                    / float(gates.gauge_ramp_calls)))
                eff_gamma_gauge = self.gamma_gauge + gates.gauge_ramp_extra * ramp
            else:
                eff_gamma_gauge = self.gamma_gauge
            gauge_loss = 0.5 * eff_gamma_gauge * mean_logit.pow(2)

        penalty = (self.coeff * self.lazy_k) * (ebi_loss + gauge_loss)
        if t_scale <= FROZEN_THERMAL_SCALE and penalty.requires_grad:
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
        from ..optim.dualnorm import NormalizedOptimizer
        from ..tensorflow_adam import TensorFlowV1Adam
        if not (isinstance(optimizer, (BISPFCriticAdam, K3PCriticAdam)) or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam")
                or (isinstance(optimizer, NormalizedOptimizer) and hasattr(optimizer, "record"))):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")
        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_critic_input_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        options.setdefault("gates", recipe.bispf_gates())
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
