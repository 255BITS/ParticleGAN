"""Signed fixed Gaussian-mixture MMD U on caller-owned output panels.

No labels, sampling, normalization, jitter or nonnegative projection is used.
Bandwidths are standard deviations in the original output coordinates.
"""
import math

import torch


DEFAULT_OUTPUT_MMD_BANDWIDTHS = (.0625, .125, .25, .5, 1., 2., 4.)
__all__ = ["DEFAULT_OUTPUT_MMD_BANDWIDTHS", "kernel_mmd_u", "OutputMMDU"]


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_bandwidths(bandwidths):
    """Normalize a nonempty explicit sequence without accepting bools."""
    if (not isinstance(bandwidths, (list, tuple)) or not bandwidths
            or any(not _finite_number(value) or value <= 0 for value in bandwidths)):
        raise ValueError("output_mmd_bandwidths must be nonempty finite positive numbers")
    return tuple(float(value) for value in bandwidths)


def _kernel(distance, widths):
    # Scale before squaring; never form a silently rounded Python sigma**2.
    if not bool(torch.isfinite(distance).all()):
        raise ValueError("MMD distances must be representable in the panel dtype")
    scaled = distance.unsqueeze(0) / widths[:, None, None]
    squared = scaled.square()
    if not bool(torch.isfinite(squared).all()):
        raise ValueError("MMD scaled distances must be representable in the panel dtype")
    return torch.exp(-.5 * squared).mean(0)


def kernel_mmd_u(fake, real, *, bandwidths=DEFAULT_OUTPUT_MMD_BANDWIDTHS):
    """Ordered off-diagonal within means minus twice the cross mean.

    Both fake arguments retain gradients; target panels are detached. Only
    diagonal indices are excluded, so duplicate-valued rows remain in sums.
    Negative finite-sample estimates are deliberate. Unequal panel sizes and
    arbitrary matching nonempty floating sample spaces are supported.
    """
    bandwidths = validate_bandwidths(bandwidths)
    if (not isinstance(fake, torch.Tensor) or not isinstance(real, torch.Tensor)
            or fake.ndim < 2 or real.ndim < 2 or len(fake) < 2 or len(real) < 2
            or fake.shape[1:] != real.shape[1:] or not fake.is_floating_point()
            or fake.dtype != real.dtype or fake.device != real.device
            or fake.flatten(1).shape[1] == 0
            or not bool(torch.isfinite(fake).all()) or not bool(torch.isfinite(real).all())):
        raise ValueError("MMD U requires finite matching floating sample spaces and at least two rows each")
    widths = fake.new_tensor(bandwidths)
    squared = widths.square()
    if (not bool(torch.isfinite(squared).all()) or not bool((squared > 0).all())):
        raise ValueError("output_mmd_bandwidths squared must be representable in the panel dtype")
    x, y = fake.flatten(1), real.detach().flatten(1)
    def distances(a, b):
        return torch.cdist(a, b, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    cross, ff = _kernel(distances(x, y), widths), _kernel(distances(x, x), widths)
    with torch.no_grad():
        rr = _kernel(distances(y, y), widths)
    n, m = len(x), len(y)
    return ((ff.sum() - ff.diagonal().sum()) / (n * (n - 1))
            + (rr.sum() - rr.diagonal().sum()) / (m * (m - 1)) - 2 * cross.mean())


class OutputMMDU:
    """Separate finite signed accounting, bound to weight and kernel widths.

    Conditioning is checked but never added to the output coordinates. This
    diagnostic state has no influence on training or sampling.
    """
    def __init__(self, recipe):
        self.recipe = recipe
        self.calls = self.active_calls = 0
        self.loss_sum = self.minimum_loss = self.maximum_loss = 0.0

    def add(self, total, fake, real, *, conditioning=None):
        if not self.recipe.output_mmd_weight:
            return total
        if (not isinstance(fake, torch.Tensor) or not isinstance(real, torch.Tensor)
                or fake.ndim != 2 or fake.shape != real.shape
                or (conditioning is not None and (not isinstance(conditioning, torch.Tensor)
                    or conditioning.ndim != 2 or len(conditioning) != len(fake)
                    or conditioning.device != fake.device
                    or not bool(torch.isfinite(conditioning).all())))):
            raise ValueError("output MMD requires matched panels and conditioning")
        term = self.recipe.output_mmd_loss(fake, real)
        value = float(term.detach())
        if not math.isfinite(value) or not math.isfinite(self.loss_sum + value):
            raise ValueError("nonfinite output MMD accounting")
        self.minimum_loss = min(self.minimum_loss, value) if self.calls else value
        self.maximum_loss = max(self.maximum_loss, value) if self.calls else value
        self.calls += 1
        self.active_calls += 1
        self.loss_sum += value
        return total + term

    def state_dict(self):
        return dict(schema_version=1, consumer="output_mmd_u_v1",
                    weight=self.recipe.output_mmd_weight,
                    bandwidths=self.recipe.output_mmd_bandwidths, calls=self.calls,
                    active_calls=self.active_calls, loss_sum=self.loss_sum,
                    minimum_loss=self.minimum_loss, maximum_loss=self.maximum_loss)

    def validate_state_dict(self, state):
        expected = self.state_dict()
        if isinstance(state, dict) and "bandwidths" in state:
            try:
                widths = validate_bandwidths(state["bandwidths"])
            except ValueError as error:
                raise ValueError("invalid output MMD checkpoint bandwidths") from error
        else:
            widths = None
        if (not isinstance(state, dict) or set(state) != set(expected)
                or type(state.get("schema_version")) is not int or state["schema_version"] != 1
                or state["consumer"] != expected["consumer"]
                or type(state["weight"]) not in (int, float) or state["weight"] != expected["weight"]
                or widths != expected["bandwidths"]
                or any(type(state[key]) is not int or state[key] < 0 for key in ("calls", "active_calls"))
                or state["active_calls"] != state["calls"]
                or (not expected["weight"] and state["calls"] != 0)
                or any(not _finite_number(state[key])
                       for key in ("loss_sum", "minimum_loss", "maximum_loss"))
                or state["minimum_loss"] > state["maximum_loss"]
                or (state["calls"] == 0 and any(state[key] != 0
                    for key in ("loss_sum", "minimum_loss", "maximum_loss")))):
            raise ValueError("invalid output MMD checkpoint")
        try:
            lower = state["minimum_loss"] * state["calls"]
            upper = state["maximum_loss"] * state["calls"]
        except OverflowError as error:
            raise ValueError("invalid output MMD checkpoint accounting") from error
        value = state["loss_sum"]
        if (not math.isfinite(lower) or not math.isfinite(upper)
                or (value < lower and not math.isclose(value, lower, rel_tol=1e-12, abs_tol=1e-12))
                or (value > upper and not math.isclose(value, upper, rel_tol=1e-12, abs_tol=1e-12))):
            raise ValueError("invalid output MMD checkpoint accounting")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        for key in ("calls", "active_calls", "loss_sum", "minimum_loss", "maximum_loss"):
            setattr(self, key, state[key])
