"""Magnitude-sensitive restoration to the initial learned-prior locations.

This is a final optimizer update, not a loss penalty. It deliberately runs
after direction blend; protection diagnostics therefore describe the preceding
base displacement, not the final displacement with this spring.
"""
from copy import deepcopy
import math

import torch

from .direction_blend import DirectionBlendOptimizer
from .dualnorm import NormalizedOptimizer


class _PriorProximal:
    def __init__(self, *args, prior_proximal_strength, **kwargs):
        if (type(prior_proximal_strength) not in (int, float)
                or not math.isfinite(prior_proximal_strength) or prior_proximal_strength <= 0):
            raise ValueError("prior proximal strength must be finite and positive")
        super().__init__(*args, **kwargs)
        if self.family != "dualnorm" or self.momentum:
            raise ValueError("prior proximal update requires zero-momentum full DualNorm")
        self.prior_proximal_strength = float(prior_proximal_strength)
        self._prior_initial = {}
        for group in self.param_groups:
            if group["role"] in ("prior", "table"):
                if group["algorithm"] != "rownorm" or not group["sampled_rows_required"]:
                    raise ValueError("prior proximal update requires sampled latent-row ownership")
                table = group["params"][0]
                if not bool(torch.isfinite(table).all()):
                    raise ValueError("prior proximal initial locations must be finite")
                self._prior_initial[table] = table.detach().clone()
        self.prior_proximal_stats = dict(steps=0, rows=0, max_restoration=0.,
                                        sum_squared_restoration=0.)

    def add_param_group(self, group):
        if (hasattr(self, "_prior_initial")
                and group.get("role", group.get("forge_role")) in ("prior", "table")):
            raise ValueError("construct prior proximal optimizer after all latent tables are initialized")
        super().add_param_group(group)

    def _proximal_pending(self, closure):
        """Validate the step and return its (table, rows, fraction) restorations; mutates nothing."""
        if closure is not None:
            raise ValueError("prior proximal update requires explicit backward and row ownership")
        pending = []
        for group in self.param_groups:
            if group["role"] not in ("prior", "table"):
                continue
            table = group["params"][0]
            if table not in self._prior_initial:
                raise ValueError("missing prior proximal initial reference")
            if table.grad is None:
                continue
            rows = self.sampled_rows_for(table)
            if rows is None:
                raise ValueError("prior proximal update requires actual sampled rows")
            scaled_strength = group["lr"] * self.prior_proximal_strength
            if not math.isfinite(scaled_strength):
                raise ValueError("prior proximal learning-rate times strength must be finite")
            if scaled_strength and rows.numel():
                pending.append((table, rows, scaled_strength / (1. + scaled_strength)))
        return pending

    @torch.no_grad()
    def step(self, closure=None):
        # A rejected precheck is a step that will not happen: discard the caller-bound inputs
        # (sampled rows, protected losses of a direction-blend base) so the next step binds afresh.
        try:
            pending = self._proximal_pending(closure)
        except BaseException:
            self.abort_pending()
            raise
        result = super().step()
        # lerp is the stable equivalent of (z_base + lr*lambda*z_initial) /
        # (1 + lr*lambda), with no work/rounding when the scheduled rate is zero.
        for table, rows, fraction in pending:
            before = table[rows]
            restored = torch.lerp(before, self._prior_initial[table][rows], fraction)
            delta = restored - before
            table.index_copy_(0, rows, restored)
            stats = self.prior_proximal_stats
            stats["rows"] += len(rows)
            stats["max_restoration"] = max(stats["max_restoration"], float(delta.norm(dim=1).max()))
            stats["sum_squared_restoration"] += float(delta.double().square().sum())
        if pending:
            self.prior_proximal_stats["steps"] += 1
        return result

    def state_dict(self):
        saved = super().state_dict()
        saved["prior_proximal"] = dict(schema=1, strength=self.prior_proximal_strength,
            order="after_base_update", stats=deepcopy(self.prior_proximal_stats),
            initial_locations=[self._prior_initial[group["params"][0]].clone()
                               if group["role"] in ("prior", "table") else None
                               for group in self.param_groups])
        return saved

    def validate_state_dict(self, saved):
        if not isinstance(saved, dict):
            raise ValueError("invalid prior proximal checkpoint")
        base = dict(saved)
        meta = base.pop("prior_proximal", None)
        if (not isinstance(meta, dict)
                or set(meta) != {"schema", "strength", "order", "stats", "initial_locations"}
                or meta["schema"] != 1 or meta["strength"] != self.prior_proximal_strength
                or meta["order"] != "after_base_update"
                or not isinstance(meta["stats"], dict)
                or set(meta["stats"]) != set(self.prior_proximal_stats)
                or not isinstance(meta["initial_locations"], list)
                or len(meta["initial_locations"]) != len(self.param_groups)):
            raise ValueError("invalid prior proximal checkpoint settings")
        for key, value in meta["stats"].items():
            if key in ("steps", "rows"):
                valid = type(value) is int and value >= 0
            else:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            if not valid:
                raise ValueError("invalid prior proximal checkpoint statistics")
        for group, reference in zip(self.param_groups, meta["initial_locations"]):
            if group["role"] not in ("prior", "table"):
                if reference is not None:
                    raise ValueError("unexpected prior proximal reference for network group")
                continue
            table = group["params"][0]
            if (not isinstance(reference, torch.Tensor) or reference.shape != table.shape
                    or reference.dtype != table.dtype or not bool(torch.isfinite(reference).all())):
                raise ValueError("invalid prior proximal initial reference")
        super().validate_state_dict(base)

    def load_state_dict(self, saved):
        # The base loader calls this complete validator before any mutation.
        super().load_state_dict(saved)
        meta = saved["prior_proximal"]
        self._prior_initial = {group["params"][0]: reference.detach().to(group["params"][0]).clone()
                              for group, reference in zip(self.param_groups, meta["initial_locations"])
                              if reference is not None}
        self.prior_proximal_stats = deepcopy(meta["stats"])


class PriorProximalOptimizer(_PriorProximal, NormalizedOptimizer):
    """Full DualNorm followed by the initial-position spring."""


class PriorProximalDirectionBlendOptimizer(_PriorProximal, DirectionBlendOptimizer):
    """BCAP leader direction blend followed by the initial-position spring."""


def proximal_optimizer_class(base_class):
    if base_class is NormalizedOptimizer:
        return PriorProximalOptimizer
    if base_class is DirectionBlendOptimizer:
        return PriorProximalDirectionBlendOptimizer
    raise ValueError("unsupported prior proximal optimizer composition")
