# Pure sampler/oracle excerpt from c85af40f1e30339702dcb6143830fca15d5334f2:lib/circle_transition.py
# Original Git blob 1d662989605d9e5aa856e51dd174685f179fb481. Training builders intentionally excluded.
"""Fully observed circle task for a feedforward transition controller.

Training rows are independent local samples. The analytic successor is a label
and an evaluation oracle; playback never snaps onto it. E_control sees the
current position and the circle context only.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F



CENTER_LIMIT = 0.75
RADIUS_RANGE = (0.6, 1.4)
SPEED_RANGE = (0.12, 0.40)
RHO_RANGE = (0.8, 1.2)
RECOVERY_RATE = 0.25
CENTER_BINS = 5
RADIUS_BINS = 4
SPEED_BINS = 4
SPLIT_MOD = 20
SPLIT_BUCKETS = {"train": (0, 14), "val": (14, 17), "test": (17, 20)}
HORIZONS = (256, 1024)
RECOVERY_WINDOW = 64
EVAL_EPISODES = 128
PANEL_SEEDS = {
    ("val", "main"): 51001,
    ("test", "main"): 51002,
    ("val", "recovery"): 51011,
    ("test", "recovery"): 51012,
}
SUCCESS_RADIAL_RMSE = 0.1
SUCCESS_SPEED_ERROR = 0.03
SUCCESS_DIRECTION = 0.95
# Frozen #22 cartesian paired-error run (f8a5a4a), test main, 1024 steps.
# Direction and turns were solved; radius was not. The fidelity score still passed.
SHIPPED_RADIAL_RMSE_1024 = 0.2478
SHIPPED_WORST_DIRECTION_1024 = 0.203125
SHIPPED_CLEAR_MARGIN = 0.15
# Radial critic noise must sit under the one-step restore of half the RMSE bar.
# Capping only at the expert radial std still left that restore inside the hold.
RADIAL_NOISE_FRACTION = 0.5
# Second half of a tangent-refine run fits a tangent residual on detached
# features while E_control and G2 keep training the wide-radius restore.
# The tolerance is half the speed bar at the smallest radius, so the
# paired-noise floor sits under the signed-step success line.
TANGENT_REFINE_FRACTION = 0.5
TANGENT_NOISE_FRACTION = 0.5
WIDE_RHO_RANGE = (0.5, 1.5)
CONTEXT_DIM = 4
STATE_DIM = 2
ACTION_DIM = 2

_FIELDS = ("position", "center", "radius", "angular_step", "action", "next_position", "rho")


def protocol():
    """Concrete v1 evaluation contract. Thresholds match the circle handoff."""
    return dict(
        version=1,
        horizons=list(HORIZONS),
        episodes=EVAL_EPISODES,
        recovery_window=RECOVERY_WINDOW,
        radial_rmse_max=SUCCESS_RADIAL_RMSE,
        signed_speed_error_max=SUCCESS_SPEED_ERROR,
        direction_agreement_min=SUCCESS_DIRECTION,
        min_turns=1.0,
        center_limit=CENTER_LIMIT,
        radius_range=list(RADIUS_RANGE),
        speed_range=list(SPEED_RANGE),
        rho_range=list(RHO_RANGE),
        recovery_rate=RECOVERY_RATE,
        bins=dict(center=CENTER_BINS, radius=RADIUS_BINS, speed=SPEED_BINS),
        split_buckets={name: list(bounds) for name, bounds in SPLIT_BUCKETS.items()},
        panel_seeds={f"{split}_{kind}": seed for (split, kind), seed in PANEL_SEEDS.items()},
        positive_angular_step="counterclockwise",
    )


class CircleBatch:
    """One independent row per index. Columns stay aligned under indexing."""

    def __init__(self, position, center, radius, angular_step, action, next_position, rho):
        tensors = (position, center, radius, angular_step, action, next_position, rho)
        n = position.shape[0]
        if any(tensor.shape[0] != n for tensor in tensors):
            raise ValueError("circle batch columns must share a row count")
        self.position, self.center, self.radius = position, center, radius
        self.angular_step, self.action = angular_step, action
        self.next_position, self.rho = next_position, rho

    def __len__(self):
        return self.position.shape[0]

    def index(self, ids):
        return CircleBatch(*(getattr(self, name)[ids] for name in _FIELDS))

    def context(self):
        """Center, target radius, signed angular step. No phase clock and no action."""
        return torch.cat([self.center, self.radius[:, None], self.angular_step[:, None]], 1)


def _bin(value, low, high, bins):
    scaled = (value - low) / (high - low)
    return scaled.clamp(0, 1 - 1e-6).mul(bins).long()


def parameter_cell(center, radius, angular_step):
    """Disjoint geometry/speed cell. Direction is not part of the split key."""
    ix = _bin(center[:, 0], -CENTER_LIMIT, CENTER_LIMIT, CENTER_BINS)
    iy = _bin(center[:, 1], -CENTER_LIMIT, CENTER_LIMIT, CENTER_BINS)
    ir = _bin(radius, RADIUS_RANGE[0], RADIUS_RANGE[1], RADIUS_BINS)
    isp = _bin(angular_step.abs(), SPEED_RANGE[0], SPEED_RANGE[1], SPEED_BINS)
    return ((ix * CENTER_BINS + iy) * RADIUS_BINS + ir) * SPEED_BINS + isp


def in_split(center, radius, angular_step, split):
    if split not in SPLIT_BUCKETS:
        raise ValueError(split)
    low, high = SPLIT_BUCKETS[split]
    bucket = torch.remainder(parameter_cell(center, radius, angular_step), SPLIT_MOD)
    return (bucket >= low) & (bucket < high)


def expert_transition(position, center, radius, angular_step):
    """Analytic label. Rotate, then move normalized radius a quarter of the way toward 1."""
    q = (position - center) / radius[:, None]
    rho = q.norm(dim=1).clamp_min(1e-8)
    rho_next = rho + RECOVERY_RATE * (1 - rho)
    cosine, sine = angular_step.cos(), angular_step.sin()
    unit_x, unit_y = q[:, 0] / rho, q[:, 1] / rho
    rotated = torch.stack((cosine * unit_x - sine * unit_y, sine * unit_x + cosine * unit_y), 1)
    q_next = rotated * rho_next[:, None]
    action = radius[:, None] * (q_next - q)
    return action, position + action


def assert_aligned(batch, atol=1e-5):
    if not torch.allclose(batch.position + batch.action, batch.next_position, atol=atol, rtol=1e-5):
        raise AssertionError("state, action, and next state are not aligned")
    action, nxt = expert_transition(batch.position, batch.center, batch.radius, batch.angular_step)
    if not torch.allclose(action, batch.action, atol=atol, rtol=1e-5):
        raise AssertionError("expert action does not match the analytic successor")
    if not torch.allclose(nxt, batch.next_position, atol=atol, rtol=1e-5):
        raise AssertionError("next state does not match the analytic successor")


def _sample_signed(count, rng, split, sign, rho_mode, device, rho_range, on_circle_rate):
    centers, radii, speeds, phases = [], [], [], []
    have = 0
    while have < count:
        draw = max((count - have) * 8, 64)
        center = torch.empty(draw, 2, device=device)
        center[:, 0] = -CENTER_LIMIT + 2 * CENTER_LIMIT * torch.rand(draw, device=device, generator=rng)
        center[:, 1] = -CENTER_LIMIT + 2 * CENTER_LIMIT * torch.rand(draw, device=device, generator=rng)
        radius = RADIUS_RANGE[0] + (RADIUS_RANGE[1] - RADIUS_RANGE[0]) * torch.rand(draw, device=device, generator=rng)
        speed = SPEED_RANGE[0] + (SPEED_RANGE[1] - SPEED_RANGE[0]) * torch.rand(draw, device=device, generator=rng)
        phase = 2 * math.pi * torch.rand(draw, device=device, generator=rng)
        omega = speed * sign
        keep = in_split(center, radius, omega, split)
        if not bool(keep.any()):
            continue
        centers.append(center[keep])
        radii.append(radius[keep])
        speeds.append(omega[keep])
        phases.append(phase[keep])
        have += int(keep.sum())
    center = torch.cat(centers)[:count]
    radius = torch.cat(radii)[:count]
    omega = torch.cat(speeds)[:count]
    phase = torch.cat(phases)[:count]
    if rho_mode == "on":
        rho = torch.ones(count, device=device)
    elif rho_mode == "off":
        low, high = rho_range
        rho = low + (high - low) * torch.rand(count, device=device, generator=rng)
    elif rho_mode == "mixed":
        low, high = rho_range
        rho = low + (high - low) * torch.rand(count, device=device, generator=rng)
        on_circle = torch.rand(count, device=device, generator=rng) < on_circle_rate
        rho = torch.where(on_circle, torch.ones_like(rho), rho)
    elif rho_mode == "recovery":
        rho = torch.empty(count, device=device)
        mid = count // 2
        rho[:mid] = RHO_RANGE[0]
        rho[mid:] = RHO_RANGE[1]
    else:
        raise ValueError(rho_mode)
    direction = torch.stack((phase.cos(), phase.sin()), 1)
    position = center + radius[:, None] * direction * rho[:, None]
    action, nxt = expert_transition(position, center, radius, omega)
    return CircleBatch(position, center, radius, omega, action, nxt, rho)


def sample_rows(n, rng, split="train", rho_mode="mixed", device="cpu", rho_range=None, on_circle_rate=0.5):
    """Independent local rows. Shuffled. Not a collected trajectory.

    ``rho_range`` and ``on_circle_rate`` change only ``mixed`` and, for the range,
    ``off``. Recovery panels stay on the protocol radii 0.8 and 1.2.
    """
    if type(n) is not int or n < 2:
        raise ValueError("n must be an integer >= 2")
    if rho_range is None:
        rho_range = RHO_RANGE
    low, high = float(rho_range[0]), float(rho_range[1])
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0 or low >= high:
        raise ValueError("rho_range must be a positive finite interval")
    if type(on_circle_rate) is not float or not 0 <= on_circle_rate <= 1:
        raise ValueError("on_circle_rate must be a float in [0, 1]")
    rho_range = (low, high)
    device = torch.device(device)
    half = n // 2
    positive = _sample_signed(half, rng, split, 1., rho_mode, device, rho_range, on_circle_rate)
    negative = _sample_signed(n - half, rng, split, -1., rho_mode, device, rho_range, on_circle_rate)
    batch = CircleBatch(*(torch.cat([getattr(positive, name), getattr(negative, name)], 0) for name in _FIELDS))
    order = torch.randperm(n, device=device, generator=rng)
    shuffled = batch.index(order)
    assert_aligned(shuffled)
    return shuffled


def evaluation_panel(split, kind, episodes=EVAL_EPISODES, device="cpu"):
    """Fixed held-out panel. Main starts on the circle; recovery starts at 0.8 and 1.2."""
    if kind not in ("main", "recovery") or (split, kind) not in PANEL_SEEDS:
        raise ValueError(f"unknown panel {split}/{kind}")
    if type(episodes) is not int or episodes < 4 or episodes % 4:
        raise ValueError("episodes must be a multiple of 4 so both directions and recovery radii are covered")
    rng = torch.Generator(device=device).manual_seed(PANEL_SEEDS[(split, kind)])
    mode = "on" if kind == "main" else "recovery"
    return sample_rows(episodes, rng, split=split, rho_mode=mode, device=device)
