"""Fully observed circle task for a feedforward transition controller.

Training rows are independent local samples. The analytic successor is a label
and an evaluation oracle; playback never snaps onto it. E_control sees the
current position and the circle context only.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from particlegan.autoencoder import particle_ae


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
# Last quarter of a tangent-refine run freezes the radius pathway and fits a
# tangent residual. The tolerance is half the speed bar at the smallest radius,
# so the paired-noise floor sits under the signed-step success line.
TANGENT_REFINE_FRACTION = 0.75
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


class CircleScaler(nn.Module):
    """Frozen per-coordinate statistics. Current and next position share one scale."""

    def __init__(self, position_mean, position_scale, action_mean, action_scale, context_mean, context_scale):
        super().__init__()
        buffers = dict(position_mean=position_mean, position_scale=position_scale, action_mean=action_mean,
                       action_scale=action_scale, context_mean=context_mean, context_scale=context_scale)
        for name, value in buffers.items():
            tensor = torch.as_tensor(value, dtype=torch.float32).detach().clone()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
            self.register_buffer(name, tensor)

    @classmethod
    def fit(cls, batch):
        position = torch.cat([batch.position, batch.next_position], 0)
        context = batch.context()
        def stats(value):
            return value.mean(0), value.std(0, unbiased=False).clamp_min(1e-6)
        return cls(*stats(position), *stats(batch.action), *stats(context))

    def state(self, position):
        return (position - self.position_mean) / self.position_scale

    def inverse_state(self, position):
        return position * self.position_scale + self.position_mean

    def action(self, action):
        return (action - self.action_mean) / self.action_scale

    def inverse_action(self, action):
        return action * self.action_scale + self.action_mean

    def context(self, context):
        return (context - self.context_mean) / self.context_scale

    def triple(self, position, action, next_position):
        return torch.cat([self.state(position), self.action(action), self.state(next_position)], 1)


def mlp(inp, width, out):
    return nn.Sequential(nn.Linear(inp, width), nn.LeakyReLU(.2),
                         nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, out))


class CircleGenerator(nn.Module):
    """Independent G1/G2/G3 heads on one code and the circle context."""

    def __init__(self, z_dim=8, width=128, context_dim=CONTEXT_DIM):
        super().__init__()
        self.z_dim, self.context_dim = z_dim, context_dim
        self.branches = nn.ModuleList([mlp(z_dim + context_dim, width, 2) for _ in range(3)])

    def forward(self, z, context):
        if z.ndim != 2 or context.shape != (z.shape[0], self.context_dim):
            raise ValueError("G expects [batch, z] and a matching circle context")
        shared = torch.cat([z, context], 1)
        return torch.cat([branch(shared) for branch in self.branches], 1)


class CircleEncoder(nn.Module):
    """Stateless particle encoder. Each call depends only on its argument."""

    def __init__(self, in_dim, z_dim=8, width=128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.features = mlp(self.in_dim, width, width)
        self.query = nn.Linear(width, z_dim)
        self.offset = nn.Linear(width, z_dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, features, prior):
        if features.ndim != 2 or features.shape[1] != self.in_dim:
            raise ValueError(f"encoder expects [batch, {self.in_dim}] with no carried state")
        hidden = self.features(features)
        query = F.layer_norm(self.query(hidden), (self.query.out_features,))
        return particle_ae(query, self.offset(hidden), prior, temperature=.25,
                           distance_reduction="sum", offset_bound=3.)


class CircleDiscriminator(nn.Module):
    def __init__(self, width=128, triple_dim=6, context_dim=CONTEXT_DIM):
        super().__init__()
        self.triple_dim, self.context_dim = triple_dim, context_dim
        self.net = nn.Sequential(
            nn.Linear(triple_dim + context_dim, width), nn.LeakyReLU(.2),
            nn.Linear(width, width), nn.LeakyReLU(.2),
            nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, 1))

    def forward(self, triple, context):
        if triple.shape[-1] != self.triple_dim or context.shape[-1] != self.context_dim:
            raise ValueError("D expects a normalized transition triple and circle context")
        return self.net(torch.cat([triple, context], 1)).squeeze(1)


def assert_feedforward(*modules):
    banned = (nn.GRU, nn.LSTM, nn.RNN, nn.GRUCell, nn.LSTMCell, nn.RNNCell)
    found = [type(module).__name__ for root in modules for module in root.modules() if isinstance(module, banned)]
    if found:
        raise ValueError(f"explicit memory is outside this toy: {found}")


def encode_pair(encoder, generator, prior, state_action, context):
    encoding = encoder(torch.cat([state_action, context], 1), prior)
    return generator(encoding.codes[:, 0], context), encoding


def compose_pair(encoder, generator, prior, fake, context):
    decoded, encoding = encode_pair(encoder, generator, prior, fake[:, :4], context)
    return torch.cat([fake[:, :4], decoded[:, 4:]], 1), decoded, encoding


class TangentResidual(nn.Module):
    """Zero-init correction added only along the requested tangent.

    The base action is detached. Radius stays on the frozen paired-error policy.
    """

    def __init__(self, width, hidden=32):
        super().__init__()
        if type(width) is not int or width < 1 or type(hidden) is not int or hidden < 1:
            raise ValueError("tangent residual widths must be positive integers")
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden):
        return self.net(hidden)


def normalized_control_action(bundle, position, context):
    """E_control(position, context) -> z -> G2. No action, phase, or next state.

    When the tangent residual is active, its output is added in physical
    tangent units and the base displacement is detached.
    """
    scaler = bundle["scaler"]
    features = torch.cat([scaler.state(position), scaler.context(context)], 1)
    encoded = bundle["E_control"](features, bundle["prior"])
    branch = bundle["G"].branches[1]
    incoming = torch.cat([encoded.codes[:, 0], scaler.context(context)], 1)
    head = bundle.get("tangent_head")
    if head is None or not bundle.get("tangent_head_active", False):
        return branch(incoming)
    hidden = incoming
    for layer in list(branch)[:-1]:
        hidden = layer(hidden)
    normalized = branch[-1](hidden)
    center, angular_step = context[:, :2], context[:, 3]
    physical = scaler.inverse_action(normalized).detach()
    tangent, _ = _radial_axis(position, center, angular_step)
    corrected = physical + head(hidden.detach()) * tangent
    return scaler.action(corrected)


def physical_control_action(bundle, position, center, radius, angular_step):
    context = torch.cat([center, radius[:, None], angular_step[:, None]], 1)
    return bundle["scaler"].inverse_action(normalized_control_action(bundle, position, context))


def expert_policy(position, center, radius, angular_step):
    action, _ = expert_transition(position, center, radius, angular_step)
    return action


def zero_policy(position, center, radius, angular_step):
    return torch.zeros_like(position)


def reversed_policy(position, center, radius, angular_step):
    """Analytic successor for the opposite signed step. The request stays unchanged."""
    action, _ = expert_transition(position, center, radius, -angular_step)
    return action


def _radial_axis(position, center, angular_step):
    relative = position - center
    rho = relative.norm(dim=1, keepdim=True).clamp_min(1e-8)
    radial = relative / rho
    sign = torch.where(angular_step >= 0, torch.ones_like(angular_step), -torch.ones_like(angular_step))
    tangent = torch.stack((-radial[:, 1], radial[:, 0]), 1) * sign[:, None]
    return tangent, radial


def radial_tangent(action, position, center, angular_step):
    """Action in the requested-tangent, outward-radial frame. Linear in the action."""
    tangent, radial = _radial_axis(position, center, angular_step)
    return torch.stack(((action * tangent).sum(1), (action * radial).sum(1)), 1)


def missing_restore_policy(position, center, radius, angular_step):
    """Expert tangent plus the on-circle radial chord. Off-circle restore is dropped.

    This is the closed-loop radius walk of the cartesian paired-error baseline:
    direction stays correct while radius is not pulled back to 1.
    """
    action, _ = expert_transition(position, center, radius, angular_step)
    _, radial = _radial_axis(position, center, angular_step)
    on_circle = center + radius[:, None] * radial
    on_action, _ = expert_transition(on_circle, center, radius, angular_step)
    radial_command = (on_action * radial).sum(1, keepdim=True)
    tangent_command = action - (action * radial).sum(1, keepdim=True) * radial
    return tangent_command + radial_command * radial


def paired_edit_actions(bundle, normalized_action, batch):
    """Controller critic coordinates. Cartesian keeps the #22 recipe."""
    frame = bundle["edit_frame"]
    if frame == "cartesian":
        return normalized_action, bundle["scaler"].action(batch.action)
    if frame == "radial_tangent":
        physical = bundle["scaler"].inverse_action(normalized_action)
        return (radial_tangent(physical, batch.position, batch.center, batch.angular_step),
                radial_tangent(batch.action, batch.position, batch.center, batch.angular_step))
    raise ValueError(frame)


def cap_radial_edit_scale(critic, targets, noise_hold=None):
    """Keep the radial divisor from hiding the restore.

    ``noise_hold=None`` caps at the expert radial spread. A positive hold also
    caps at ``tolerance / (noise_hold * edit_rms)``. Tangent scale, edit RMS,
    and noise start stay on the paired edit.
    """
    if noise_hold is not None and (not math.isfinite(noise_hold) or noise_hold <= 0):
        raise ValueError("noise_hold must be finite and positive")
    signal = float(targets[:, -1].detach().std(unbiased=False).clamp_min(1e-4))
    tolerance = RECOVERY_RATE * (SUCCESS_RADIAL_RMSE * RADIAL_NOISE_FRACTION)
    if noise_hold is None:
        hold_sigma = None
        noise_limited = signal
    else:
        hold_sigma = float(noise_hold) * float(critic.edit_rms)
        if hold_sigma <= 1e-8:
            noise_limited = signal
        else:
            noise_limited = max(tolerance / hold_sigma, 1e-4)
    cap = min(signal, noise_limited)
    before = float(critic.target_std[-1])
    info = dict(capped=False, radial_scale=before, signal_scale=signal, previous_scale=before,
                noise_limited=noise_limited, tolerance=tolerance, hold_sigma=hold_sigma)
    if before <= cap:
        return info
    with torch.no_grad():
        critic.target_std[-1].copy_(torch.tensor(cap, dtype=critic.target_std.dtype, device=critic.target_std.device))
    info.update(capped=True, radial_scale=cap)
    return info


def tangent_hold_tolerance():
    """Physical tangent noise aimed under the signed-step bar on the smallest circle."""
    return SUCCESS_SPEED_ERROR * RADIUS_RANGE[0] * TANGENT_NOISE_FRACTION


def cap_tangent_edit_scale(critic, noise_hold, tolerance=None):
    """Shrink the tangent divisor so a speed-bar residual clears the paired-noise hold.

    Radial scale, edit RMS, and noise start stay put. The radius pathway is
    frozen while this cap is in effect.
    """
    if not math.isfinite(noise_hold) or noise_hold <= 0:
        raise ValueError("noise_hold must be finite and positive")
    if tolerance is None:
        tolerance = tangent_hold_tolerance()
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    hold_sigma = float(noise_hold) * float(critic.edit_rms)
    if hold_sigma <= 1e-8:
        raise ValueError("edit RMS is too small to set a tangent noise cap")
    cap = max(float(tolerance) / hold_sigma, 1e-4)
    before = float(critic.target_std[0])
    info = dict(capped=False, tangent_scale=before, previous_scale=before, noise_limited=cap,
                tolerance=float(tolerance), hold_sigma=hold_sigma)
    if before <= cap:
        return info
    with torch.no_grad():
        critic.target_std[0].copy_(torch.tensor(cap, dtype=critic.target_std.dtype, device=critic.target_std.device))
    info.update(capped=True, tangent_scale=cap)
    return info


def isolate_tangent_pair(predicted, target):
    """Paired-error coordinates whose radial residual is identically zero.

    The tangent column keeps its graph. Used after the radius pathway is frozen.
    """
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("tangent isolation expects matching [rows, 2] actions")
    return torch.cat([predicted[:, :1], target[:, 1:].detach()], 1), target


def radial_channel_diagnostics(physical_action, batch):
    """Outward-radial absolute error and its correlation with (1 - rho).

    Logged only. This tensor is not part of the controller loss.
    """
    action = physical_action.detach()
    frame = radial_tangent(action, batch.position, batch.center, batch.angular_step)
    expert = radial_tangent(batch.action, batch.position, batch.center, batch.angular_step)
    radial_l1 = (frame[:, 1] - expert[:, 1]).abs().mean()
    tangent_l1 = (frame[:, 0] - expert[:, 0]).abs().mean()
    restore = 1 - batch.rho
    centered_action = frame[:, 1] - frame[:, 1].mean()
    centered_restore = restore - restore.mean()
    corr = centered_action.dot(centered_restore) / (centered_action.norm() * centered_restore.norm()).clamp_min(1e-8)
    return radial_l1, corr, tangent_l1


def learned_policy(bundle):
    def policy(position, center, radius, angular_step):
        return physical_control_action(bundle, position, center, radius, angular_step)
    return policy


def rollout(policy, batch, horizon):
    """Displacement environment. One forward state, no recurrent buffer."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    position = batch.position.clone()
    frames = [position]
    for _ in range(horizon):
        action = policy(position, batch.center, batch.radius, batch.angular_step)
        if action.shape != position.shape:
            raise ValueError("action must be a 2D displacement")
        position = position + action
        frames.append(position)
    return torch.stack(frames, 0)


def _turn_angle(before, after):
    cross = before[..., 0] * after[..., 1] - before[..., 1] * after[..., 0]
    dot = (before * after).sum(-1)
    return torch.atan2(cross, dot)


def trace_metrics(positions, center, radius, angular_step):
    relative = positions - center[None]
    radial = relative.norm(dim=-1) / radius[None].clamp_min(1e-8) - 1
    angle = _turn_angle(relative[:-1], relative[1:])
    finite = torch.isfinite(positions).all(dim=(0, 2)) & torch.isfinite(angle).all(0)
    radial = torch.where(torch.isfinite(radial), radial, torch.zeros_like(radial))
    angle = torch.where(torch.isfinite(angle), angle, torch.zeros_like(angle))
    error = angle - angular_step[None]
    agreement = angle * angular_step[None] > 0
    progress = (angle * angular_step.sign()[None]).sum(0)
    late = max(1, positions.shape[0] // 4)
    step_norm = (positions[1:] - positions[:-1]).norm(dim=-1)
    return dict(
        finite=finite,
        radial=radial,
        rmse=radial.square().mean(0).sqrt(),
        max_radial=radial.abs().amax(0),
        late_rmse=radial[-late:].square().mean(0).sqrt(),
        speed_err=error.abs().mean(0),
        agreement=agreement.float().mean(0),
        turns=progress / (2 * math.pi),
        stopping=(step_norm < 1e-3).float().mean(0),
        drift=(radial[-1] - radial[0]).abs(),
    )


def _masked_mean(values, mask=None):
    chosen = values if mask is None else values[mask]
    if chosen.numel() == 0:
        raise ValueError("metric mask is empty")
    return float(chosen.float().mean())


def summarize_trace(stats, angular_step):
    success = (stats["finite"] & (stats["rmse"] < SUCCESS_RADIAL_RMSE)
               & (stats["speed_err"] < SUCCESS_SPEED_ERROR)
               & (stats["agreement"] > SUCCESS_DIRECTION) & (stats["turns"] >= 1.))
    positive, negative = angular_step > 0, angular_step < 0
    return dict(
        episodes=int(angular_step.shape[0]),
        success=_masked_mean(success),
        success_positive=_masked_mean(success, positive),
        success_negative=_masked_mean(success, negative),
        worst_direction_success=min(_masked_mean(success, positive), _masked_mean(success, negative)),
        radial_rmse=float(stats["radial"].square().mean().sqrt()),
        max_radial_error=float(stats["max_radial"].max()),
        late_radial_rmse=_masked_mean(stats["late_rmse"]),
        signed_speed_error=_masked_mean(stats["speed_err"]),
        direction_agreement=_masked_mean(stats["agreement"]),
        completed_turns=_masked_mean(stats["turns"]),
        stopping_fraction=_masked_mean(stats["stopping"]),
        drift=_masked_mean(stats["drift"]),
        nonfinite=int((~stats["finite"]).sum()),
    )


def evaluate_panel(policy, batch, horizons=HORIZONS, recovery_window=0):
    positions = rollout(policy, batch, max(horizons))
    report = {}
    for horizon in horizons:
        window = positions[:horizon + 1]
        summary = summarize_trace(trace_metrics(window, batch.center, batch.radius, batch.angular_step),
                                  batch.angular_step)
        if recovery_window:
            if horizon <= recovery_window:
                raise ValueError("recovery window must be shorter than the horizon")
            suffix = window[recovery_window:]
            summary["after_recovery_window"] = summarize_trace(
                trace_metrics(suffix, batch.center, batch.radius, batch.angular_step), batch.angular_step)
        report[int(horizon)] = summary
    return report


def closed_loop_fidelity(summary):
    """Legacy joint score. A solved direction can still pass with radial RMSE 0.25."""
    turns = min(1., max(0., summary["completed_turns"]))
    return summary["direction_agreement"] * math.exp(-summary["radial_rmse"] / SUCCESS_RADIAL_RMSE) * turns


def radius_hold_gate(summary, baseline_worst=SHIPPED_WORST_DIRECTION_1024, margin=SHIPPED_CLEAR_MARGIN):
    """Protocol bars plus a worst-direction gain over the cartesian #22 run.

    Fails when direction and turn count look solved but the radius walks.
    The #22 test row is the reference miss: radial RMSE 0.248, worst-direction
    success 0.203, while legacy fidelity still cleared zero and reversed motion.
    """
    checks = dict(
        radial_rmse=summary["radial_rmse"] < SUCCESS_RADIAL_RMSE,
        signed_speed_error=summary["signed_speed_error"] < SUCCESS_SPEED_ERROR,
        direction_agreement=summary["direction_agreement"] > SUCCESS_DIRECTION,
        worst_direction_success=summary["worst_direction_success"] > baseline_worst + margin,
        majority_success=summary["success"] >= 0.5,
        finite=summary["nonfinite"] == 0,
    )
    return dict(passed=all(checks.values()), checks=checks,
                baseline_worst=baseline_worst, margin=margin)


@torch.no_grad()
def local_errors(bundle, batch):
    """Held-out action and G3 errors. Separate from closed-loop rank."""
    predicted = physical_control_action(bundle, batch.position, batch.center, batch.radius, batch.angular_step)
    action_l2 = (predicted - batch.action).norm(dim=1).mean()
    context = batch.context()
    scaled_context = bundle["scaler"].context(context)
    control = bundle["E_control"](torch.cat([bundle["scaler"].state(batch.position), scaled_context], 1),
                                  bundle["prior"])
    next_hat = bundle["scaler"].inverse_state(
        bundle["G"].branches[2](torch.cat([control.codes[:, 0], scaled_context], 1)))
    paired, _ = encode_pair(bundle["E"], bundle["G"], bundle["prior"],
                            torch.cat([bundle["scaler"].state(batch.position), bundle["scaler"].action(batch.action)], 1),
                            scaled_context)
    paired_next = bundle["scaler"].inverse_state(paired[:, 4:])
    return dict(
        action_l2=float(action_l2),
        g3_next_l2=float((next_hat - batch.next_position).norm(dim=1).mean()),
        paired_g3_next_l2=float((paired_next - batch.next_position).norm(dim=1).mean()),
        persistence_next_l2=float((batch.position - batch.next_position).norm(dim=1).mean()))
