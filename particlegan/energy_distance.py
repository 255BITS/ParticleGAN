"""Raw signed energy-distance U-statistic on caller-owned output panels.

No labels, normalizers, sampling or learned features enter this objective.
The norm has the explicitly chosen zero subgradient at exact collisions.
"""
import math

import torch


def _distance(x, y):
    distance = torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    return torch.where(distance > 0, distance, distance.detach())


def energy_distance_u(fake, real):
    """Signed 2 cross mean minus two ordered off-diagonal within means.

    Targets are detached. Both fake arguments retain gradients; duplicate-valued
    off-diagonal observations remain. This is nonsquared Euclidean distance,
    and finite-sample negative values are deliberately not clamped.
    """
    if (not isinstance(fake, torch.Tensor) or not isinstance(real, torch.Tensor)
            or fake.ndim < 2 or real.ndim < 2 or len(fake) < 2 or len(real) < 2
            or fake.shape[1:] != real.shape[1:] or not fake.is_floating_point()
            or fake.dtype != real.dtype or fake.device != real.device
            or fake.flatten(1).shape[1] == 0
            or not bool(torch.isfinite(fake).all()) or not bool(torch.isfinite(real).all())):
        raise ValueError("energy U requires finite matching floating sample spaces and at least two rows each")
    x, y = fake.flatten(1), real.detach().flatten(1)
    cross, ff = _distance(x, y), _distance(x, x)
    with torch.no_grad():
        rr = _distance(y, y)
    n, m = len(x), len(y)
    return (2 * cross.mean() - (ff.sum() - ff.diagonal().sum()) / (n * (n - 1))
            - (rr.sum() - rr.diagonal().sum()) / (m * (m - 1)))


class OutputEnergyU:
    """A separate signed output-marginal accounting/checkpoint contract.

    Conditioning is validated but never becomes a distance coordinate. The
    counters and scalar summaries are observations, not inputs to training.
    """
    def __init__(self, recipe):
        self.recipe = recipe
        self.calls = self.active_calls = 0
        self.loss_sum = self.minimum_loss = self.maximum_loss = 0.0

    def add(self, total, fake, real, *, conditioning=None):
        if not self.recipe.energy_distance_weight:
            return total
        if (not isinstance(fake, torch.Tensor) or not isinstance(real, torch.Tensor)
                or fake.ndim != 2 or fake.shape != real.shape
                or (conditioning is not None and (not isinstance(conditioning, torch.Tensor) or conditioning.ndim != 2
                    or len(conditioning) != len(fake) or conditioning.device != fake.device
                    or not bool(torch.isfinite(conditioning).all())))):
            raise ValueError("output energy requires matched panels and conditioning")
        term = self.recipe.energy_distance_loss(fake, real)
        value = float(term.detach())
        if not math.isfinite(value) or not math.isfinite(self.loss_sum + value):
            raise ValueError("nonfinite output energy accounting")
        self.minimum_loss = min(self.minimum_loss, value) if self.calls else value
        self.maximum_loss = max(self.maximum_loss, value) if self.calls else value
        self.calls += 1
        self.active_calls += 1
        self.loss_sum += value
        return total + term

    def state_dict(self):
        return dict(schema_version=1, consumer="output_energy_u_v1",
                    weight=self.recipe.energy_distance_weight, calls=self.calls,
                    active_calls=self.active_calls, loss_sum=self.loss_sum,
                    minimum_loss=self.minimum_loss, maximum_loss=self.maximum_loss)

    def validate_state_dict(self, state):
        expected = self.state_dict()
        if (not isinstance(state, dict) or set(state) != set(expected)
                or type(state.get("schema_version")) is not int or state["schema_version"] != 1
                or state["consumer"] != expected["consumer"]
                or type(state["weight"]) not in (int, float) or state["weight"] != expected["weight"]
                or any(type(state[key]) is not int or state[key] < 0 for key in ("calls", "active_calls"))
                or state["active_calls"] != state["calls"]
                or (not expected["weight"] and state["calls"] != 0)
                or any(type(state[key]) not in (int, float) or not math.isfinite(state[key])
                       for key in ("loss_sum", "minimum_loss", "maximum_loss"))
                or state["minimum_loss"] > state["maximum_loss"]
                or (state["calls"] == 0 and any(state[key] != 0
                    for key in ("loss_sum", "minimum_loss", "maximum_loss")))):
            raise ValueError("invalid output energy checkpoint")
        for bound, lower in ((state["minimum_loss"] * state["calls"], True),
                             (state["maximum_loss"] * state["calls"], False)):
            value = state["loss_sum"]
            if ((value < bound if lower else value > bound)
                    and not math.isclose(value, bound, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("invalid output energy checkpoint accounting")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        for key in ("calls", "active_calls", "loss_sum", "minimum_loss", "maximum_loss"):
            setattr(self, key, state[key])
