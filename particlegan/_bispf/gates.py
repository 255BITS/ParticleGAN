"""Declared task-shape and step-count gates of the BISPF components."""
from __future__ import annotations

from dataclasses import dataclass, fields as _dataclass_fields
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class BISPFGates:
    """Every task-shape and step-count gate used by the BISPF components.

    BISPF selects some of its behavior from the shapes it sees (input rank and
    dimension, batch sizes, latent-table size) and from the critic-penalty call
    count. Each regime is declared here as an ``<name>_enabled`` switch plus
    its thresholds and constants. The defaults reproduce the historical,
    previously hard-coded behavior bit for bit. ``Recipe.bispf_shape_gates`` and
    ``Recipe.bispf_gate_overrides`` declare a recipe's values.

    ``shape_gates=False`` is the master switch for a pure global recipe. It
    disables every regime below, so each component takes the branch it takes
    for shapes outside all special regimes. In that branch:

    * the critic penalty uses chords, RMS (``sqrt(dim)``) gradient-norm units,
      the generic cooler query, the declared ``lambda_floor``, ``kappa`` and
      ``gamma_gauge``, and no curvature term;
    * the generator loss has no dense rescale;
    * the SBNP regularizer applies its declared ``weight``, ``k_nn``,
      ``repel_weight`` and ``repel_margin``, has no isotropy terms, and
      matches only off-diagonal covariance.

    Gates that do not depend on shape or step, such as the thermal freeze
    threshold, are unaffected.
    """

    shape_gates: bool = True

    # Critic penalty: rank-2 inputs with dim >= simplex_min_dim are treated as
    # discrete simplex codes (endpoint hard caps, no gauge, no chords, no cooling).
    simplex_enabled: bool = True
    simplex_min_dim: int = 170
    # Critic penalty: inputs of rank >= 3 are treated as images (adaptive
    # kappa, image floor/gauge, no cooling, no chords unless chord_blend is large).
    image_enabled: bool = True
    image_scale_ref: float = 4.899
    image_scale_min: float = 0.15
    image_kappa_base: float = 0.6
    image_kappa_gain: float = 1.10
    image_lambda_floor: float = 0.0005
    image_gamma_gauge: float = 0.005
    # Critic penalty: skip transport chords unless the input is rank 2 with
    # dim <= chord_max_dim or chord_blend > chord_blend_override.
    chord_bypass_enabled: bool = True
    chord_max_dim: int = 64
    chord_blend_override: float = 0.10
    # Critic penalty: dim <= rms_exempt_max_dim uses raw (not sqrt(dim)-normalized)
    # input-gradient norms.
    rms_exempt_enabled: bool = True
    rms_exempt_max_dim: int = 4
    # Critic penalty: "continuous flow" regime (low-dim vectors, both batches
    # >= min_batch, after min_call penalty calls) queries the cooler with the
    # real batch size and its dimension.
    continuous_flow_enabled: bool = True
    continuous_flow_max_dim: int = 4
    continuous_flow_min_batch: int = 32
    continuous_flow_min_call: int = 500
    # Critic penalty: within continuous flow, large batches query the cooler at
    # dense_multimode_query_batch instead of the real batch size.
    dense_multimode_enabled: bool = True
    dense_multimode_min_batch: int = 1024
    dense_multimode_min_dim: int = 2
    dense_multimode_max_dim: int = 4
    dense_multimode_min_call: int = 1000
    dense_multimode_query_batch: int = 128
    # Critic penalty: after late_thermal_min_call calls on dim == late_thermal_dim
    # with the real batch in [min_real, max_real] and fake batch >= min_fake,
    # force the thermal scale to late_thermal_scale when the quantile of both
    # median-normalized nearest-neighbor distances is below the threshold
    # (threshold_cold when the previous scale was < 1, else threshold_hot).
    late_thermal_enabled: bool = True
    late_thermal_min_call: int = 1650
    late_thermal_dim: int = 2
    late_thermal_min_real_batch: int = 64
    late_thermal_max_real_batch: int = 512
    late_thermal_min_fake_batch: int = 64
    late_thermal_quantile: float = 0.95
    late_thermal_threshold_hot: float = 0.050
    late_thermal_threshold_cold: float = 0.065
    late_thermal_scale: float = 1e-5
    # Critic penalty: real batches smaller than small_batch_gauge_below cap the
    # logit gauge at small_batch_gauge_cap.
    small_batch_gauge_enabled: bool = True
    small_batch_gauge_below: int = 16
    small_batch_gauge_cap: float = 0.020
    # Critic penalty: in continuous flow, ramp the gauge by gauge_ramp_extra over
    # gauge_ramp_calls calls starting at gauge_ramp_start_call.
    gauge_ramp_enabled: bool = True
    gauge_ramp_start_call: int = 1000
    gauge_ramp_calls: int = 200
    gauge_ramp_extra: float = 0.150
    # Critic penalty: in continuous flow with dim == curvature_dim, add a
    # chord-gradient curvature penalty with this weight.
    curvature_enabled: bool = True
    curvature_dim: int = 1
    curvature_weight: float = 0.25
    # Thermal cooler: batches below cooler_reference_batch raise the cooling
    # threshold by (cooler_batch_numerator / max(cooler_min_batch, b)) ** (1 / d),
    # with d = clamp(dim, 1, cooler_max_dim).
    cooler_tau_scaling_enabled: bool = True
    cooler_reference_batch: int = 128
    cooler_min_batch: int = 16
    cooler_batch_numerator: float = 1024.0
    cooler_max_dim: int = 4
    # Thermal cooler: branches without a batch-specific query (not dense
    # multimode, not continuous flow) read the scale at this batch size and dim.
    cooler_default_query_batch: int = 128
    cooler_default_query_dim: int = 2
    # Generator loss: fake batches >= min_batch with a thermal scale s below
    # max_scale scale the generator-loss gradient by
    # clamp(s / sqrt(1 - s^2) * eps_smooth, 1e-6, 1).
    dense_g_rescale_enabled: bool = True
    dense_g_rescale_min_batch: int = 1024
    dense_g_rescale_max_scale: float = 0.999
    dense_g_rescale_eps_smooth: float = 0.001
    # SBNP: rows with d <= max add d degrees of freedom to the minimum row count.
    sbnp_dof_low_dim_enabled: bool = True
    sbnp_dof_low_dim_max: int = 4
    # SBNP: with a reference and d <= max_dim, n <= max_rows, match the full
    # covariance (not only its off-diagonal).
    sbnp_full_cov_enabled: bool = True
    sbnp_full_cov_max_dim: int = 4
    sbnp_full_cov_max_rows: int = 64
    # SBNP: unanchored rows of this dim with more than `above` repulsion rows add
    # a local k-NN covariance circularization term with this weight.
    sbnp_local_iso_enabled: bool = True
    sbnp_local_iso_dim: int = 2
    sbnp_local_iso_above: int = 64
    sbnp_local_iso_weight: float = 0.10
    # SBNP: more than `above` repulsion rows (d >= 2) add a top-2 eigenvalue
    # circularization term with this weight.
    sbnp_active_iso_enabled: bool = True
    sbnp_active_iso_above: int = 64
    sbnp_active_iso_weight: float = 0.05
    # SBNP: at most max_rows repulsion rows use this margin instead of repel_margin.
    sbnp_small_margin_enabled: bool = True
    sbnp_small_margin_max_rows: int = 64
    sbnp_small_margin: float = 1.0
    # SBNP "small latent" regime: n <= max_rows, d >= min_dim and total
    # reference count <= max_reference (or no reference). Repulsion uses
    # repel_weight * repel_mult, k_nn >= min_k_nn and this margin. When
    # small_latent_weight is not None it is the weight applied instead of
    # the regularizer's ``weight``.
    sbnp_small_latent_enabled: bool = True
    sbnp_small_latent_max_rows: int = 64
    sbnp_small_latent_min_dim: int = 8
    sbnp_small_latent_max_reference: int = 64
    sbnp_small_latent_repel_mult: float = 25.0
    sbnp_small_latent_min_k_nn: int = 10
    sbnp_small_latent_margin: float = 1.0
    sbnp_small_latent_weight: float | None = 1.0
    # SBNP without a reference: a weight in (0, below) is replaced by this
    # value. When disabled, the weight is applied as declared.
    sbnp_unanchored_weight_enabled: bool = True
    sbnp_unanchored_weight_below: float = 0.1
    sbnp_unanchored_weight: float = 1.0
    # SBNP: at a frozen thermal scale (<= 1e-4), d <= max passes the whole loss
    # through with gradient scale s. Higher dimensions leave it unscaled.
    sbnp_thermal_full_enabled: bool = True
    sbnp_thermal_full_max_dim: int = 4

    _NULLABLE = ("sbnp_small_latent_weight",)
    _POSITIVE = ("gauge_ramp_calls", "cooler_min_batch", "cooler_reference_batch",
                 "cooler_batch_numerator", "cooler_max_dim", "image_scale_ref",
                 "image_scale_min", "dense_multimode_query_batch",
                 "cooler_default_query_batch", "cooler_default_query_dim")

    def __post_init__(self):
        for spec in _dataclass_fields(self):
            value, default = getattr(self, spec.name), spec.default
            if isinstance(default, bool):
                valid = type(value) is bool
            elif spec.name in self._NULLABLE and value is None:
                valid = True
            elif isinstance(default, int):
                valid = type(value) is int and value >= 0
            else:
                valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                         and math.isfinite(value) and value >= 0)
            if valid and spec.name in self._POSITIVE:
                valid = value > 0
            if not valid:
                raise ValueError(f"invalid BISPF gate {spec.name}={value!r}")
        if not 0 <= self.late_thermal_quantile <= 1:
            raise ValueError("BISPF gate late_thermal_quantile must be in [0, 1]")

    @classmethod
    def from_overrides(cls, shape_gates: bool = True, overrides: Any = ()) -> "BISPFGates":
        """Build gates from a master switch and ``{name: value}`` or ``(name, value)`` pairs."""
        values = dict(overrides.items() if isinstance(overrides, Mapping) else overrides)
        known = {spec.name for spec in _dataclass_fields(cls)} - {"shape_gates"}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown BISPF gates: {unknown}")
        return cls(shape_gates=shape_gates, **values)

    def active(self, name: str) -> bool:
        """Whether regime ``name`` (its ``<name>_enabled`` switch) is on under the master switch."""
        return self.shape_gates and getattr(self, f"{name}_enabled")


_DEFAULT_GATES = BISPFGates()
