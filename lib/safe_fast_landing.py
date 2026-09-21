"""Safe-and-fast landing term on top of YuE2 paired-error RpGAN.

The plant is a 2D pad: lateral position, altitude, and both velocities. A
slow expert sinks gently, so many starts are still airborne when the horizon
ends. Matching that expert with paired-error RpGAN (`adv_weight=1`) lands
late and misses the rest. The combined arm keeps that GAN and adds a
differentiable cost: time aloft, a crash penalty, and a bonus for a soft
on-pad contact. A hover that never descends and a sink that hits too fast
both score below a quick soft landing.

`adv_weight=0` is the supervised ablation. It can land and is not an
accepted controller step. Gym uses the same weights on a short kinematic
unroll of the physical action; `particle.yaml` leaves the term off.
"""
import torch
from torch import nn

from lib.vendor.concept_slider_core.reference import (noise_std, register_paired_error_norm,
    rp_d_loss, rp_g_loss)
from particlegan.grad_regularizers import GradientPenalty

# One gate seed. Not a sweep.
SEED = 0
GAIN = 0.25
GRAVITY = 0.04
TRACK = 0.5
HORIZON = 48
PAD = 0.35
V_LIMIT = 0.62
X_LIMIT = 1.6
Y_LIMIT = 2.4
SLOW = 0.10
RANGE = 0.85
TEMP = 0.04
TIME_SCORE = 0.5
STEPS = 250
BETA_LR = 0.15
ADV_WEIGHT = 1.0
SAFE_FAST_WEIGHT = 1.0
TIME_WEIGHT = 0.15
CRASH_WEIGHT = 4.0
SUCCESS_BONUS = 2.0
QUICK_SINK = 0.55
CRASH_SINK = 0.90
EVAL_ROWS = 200

# Combined must clear these. They sit inside the gap measured for seed 0.
BASELINE_LAND_MAX = 0.25
BASELINE_STEPS_MIN = 40.0
COMBINED_LAND_MIN = 0.95
COMBINED_STEPS_MAX = 28.0
LAND_GAP_MIN = 0.50
STEP_GAP_MIN = 12.0
GAN_GRAD_MIN = 1.0
LATE_GAN_MIN = 0.01
COMBINED_LATE_GAN_MIN = 0.2

MAPPING = (
    dict(toy="Paired-error RpGAN on the action residual, adv_weight 1, lazy b_cap. "
             "No safe-fast term. Matches the slow expert and lands late.",
         gym="particle.yaml. safe_fast_weight 0. controller_objective is Rp logistic only."),
    dict(toy="Same GAN plus safe_fast_weight * (time aloft + crash − soft success).",
         gym="particle_safe_fast.yaml. adv_weight stays 1. "
             "safe_fast_time_weight, safe_fast_crash_weight, safe_fast_success_bonus, "
             "safe_fast_speed_limit, safe_fast_pad_half, safe_fast_horizon."),
    dict(toy="Supervised safe-fast only. adv_weight 0. Landings may pass. The arm is rejected.",
         gym="require_live_adversary rejects adv_weight 0 before any step."),
    dict(toy="Score = landing rate − 0.5 * (mean steps among successes / horizon). "
             "Hover-forever and a too-fast crash both lose to a quick soft landing.",
         gym="The gym term is this cost on a kinematic unroll of (x, y, vx, vy). Not a Lunar score."),
)


def require_live_adversary(adv_weight):
    """Reject a configured GAN that the controller step does not apply."""
    if adv_weight == 0:
        raise ValueError("adv_weight=0 leaves RpGAN and b_cap configured but not applied")
    if adv_weight != 1.:
        raise ValueError("adv_weight stays 1 so the controller step is the adversarial loss")


def sink_of(beta):
    """Descent speed from the slow expert toward, and past, the crash limit."""
    return SLOW + RANGE * torch.sigmoid(beta)


def pd_action(state, sink):
    """Track the pad and a sink rate. The vertical command stays inside (-1, 1).

    Equilibrium vertical speed is ``-sink``. A hard clamp would zero the sink
    gradient once thrust saturates, which drops the GAN signal. Gains are set
    so the command does not saturate on this plant.
    """
    x, _, vx, vy = state.unbind(-1)
    ax = (-1.4 * x - 1.1 * vx).clamp(-1, 1)
    ay = GRAVITY / GAIN - TRACK * (vy + sink)
    return torch.stack([ax, ay.clamp(-1, 1)], -1)


def expert_action(state):
    return pd_action(state, SLOW)


def kinematic_step(state, action):
    """Semi-implicit step. Column 0 is lateral, column 1 is altitude."""
    x, y, vx, vy = state.unbind(-1)
    ax, ay = action.unbind(-1)
    vx = vx + GAIN * ax
    vy = vy + GAIN * ay - GRAVITY
    x = x + GAIN * vx
    y = y + GAIN * vy
    return torch.stack([x, y, vx, vy], -1)


def initial_states(n, seed):
    generator = torch.Generator().manual_seed(int(seed))
    x = torch.rand(n, generator=generator) * 0.8 - 0.4
    y = torch.rand(n, generator=generator) * 1.3 + 1.05
    vx = torch.rand(n, generator=generator) * 0.3 - 0.15
    vy = torch.rand(n, generator=generator) * 0.2 - 0.1
    return torch.stack([x, y, vx, vy], -1)


def rollout_cost(state, transition, horizon, time_weight, crash_weight, success_bonus,
                 speed_limit, pad_half, x_limit=X_LIMIT, y_limit=Y_LIMIT, temp=TEMP):
    """Differentiable safe-fast cost. Lower is better. `transition` maps a 4-vector."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    kin = state
    flying = torch.ones(state.shape[0], device=state.device, dtype=state.dtype)
    cost = state.new_zeros(())
    for _ in range(int(horizon)):
        kin = transition(kin)
        x, y, _, vy = kin.unbind(-1)
        contact = torch.sigmoid(-y / temp)
        event = flying * contact
        too_fast = torch.sigmoid((vy.abs() - speed_limit) / temp)
        off_pad = torch.sigmoid((x.abs() - pad_half) / temp)
        oob = (torch.sigmoid((x.abs() - x_limit) / (2 * temp))
               + torch.sigmoid((y - y_limit) / (2 * temp))).clamp(0, 1)
        crash = event * (1 - (1 - too_fast) * (1 - off_pad)) + flying * oob
        success = event * (1 - too_fast) * (1 - off_pad)
        cost = cost + (time_weight * flying.mean() + crash_weight * crash.mean()
                       - success_bonus * success.mean())
        flying = flying * (1 - contact)
    return cost


def gym_shaping_cost(states, previous, terrain, predict_physical, horizon, time_weight,
                     crash_weight, success_bonus, speed_limit, pad_half):
    """Same cost as the toy, on a kinematic unroll of the physical 2-vector action.

    `states` is the 8-wide Lunar record. Only (x, y, vx, vy) move. Angle and
    contacts stay at the batch values. `predict_physical` reads the full state
    and returns the physical action. This is not Box2D.
    """
    if states.ndim != 2 or states.shape[1] < 4:
        raise ValueError("Expected a Lunar-style state with x, y, vx, vy in front")
    rest = states[:, 4:]
    previous_action = previous

    def transition(kin):
        nonlocal previous_action
        full = torch.cat([kin, rest], 1) if rest.shape[1] else kin
        physical = predict_physical(full, previous_action, terrain).clamp(-1, 1)
        previous_action = physical
        return kinematic_step(kin, physical)

    return rollout_cost(states[:, :4], transition, horizon, time_weight, crash_weight,
                        success_bonus, speed_limit, pad_half)


def _hard_rollout(policy, states):
    state = states.clone()
    landed = torch.zeros(len(state), dtype=torch.bool)
    crashed = torch.zeros(len(state), dtype=torch.bool)
    steps = torch.full((len(state),), float(HORIZON))
    done = torch.zeros(len(state), dtype=torch.bool)
    with torch.no_grad():
        for t in range(1, HORIZON + 1):
            state = kinematic_step(state, policy(state))
            x, y, vx, vy = state.unbind(-1)
            contact = (y <= 0) & ~done
            safe = contact & (x.abs() <= PAD) & (vx.abs() <= V_LIMIT) & (vy.abs() <= V_LIMIT)
            oob = (~done) & ~contact & ((x.abs() > X_LIMIT) | (y > Y_LIMIT))
            landed = landed | safe
            crashed = crashed | (contact & ~safe) | oob
            steps = torch.where(safe, torch.full_like(steps, float(t)), steps)
            done = done | contact | oob
    return landed, crashed, steps


def evaluate_policy(policy, states=None):
    """Landing rate and mean steps among successes. Timeouts use the horizon."""
    if states is None:
        states = initial_states(EVAL_ROWS, 1000)
    landed, crashed, steps = _hard_rollout(policy, states)
    landings = float(landed.float().mean())
    crash_rate = float(crashed.float().mean())
    mean_steps = float(steps[landed].mean()) if bool(landed.any()) else float(HORIZON)
    return dict(landings=landings, mean_steps=mean_steps, crash_rate=crash_rate,
                score=landings - TIME_SCORE * (mean_steps / HORIZON))


def _sink_policy(sink):
    sink = float(sink)
    return lambda state: pd_action(state, sink)


def reference_policies():
    """Fixed policies that define the score. No training."""
    states = initial_states(EVAL_ROWS, 1000)

    def hover(state):
        # Sink command 0 holds altitude. It never meets the pad.
        return pd_action(state, 0.)

    rows = {}
    for name, policy in (("hover", hover), ("crash_sink", _sink_policy(CRASH_SINK)),
                         ("quick_soft", _sink_policy(QUICK_SINK))):
        row = evaluate_policy(policy, states)
        row.update(arm=name, adv_weight=None, safe_fast_weight=None)
        rows[name] = row
    return rows


class _Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

    def forward(self, coordinates):
        return self.net(coordinates).squeeze(-1)


def _edit_scale():
    module = nn.Module()
    pool = initial_states(512, 3)
    targets = expert_action(pool)
    register_paired_error_norm(module, targets, torch.zeros_like(targets))
    return module


def _cap():
    return GradientPenalty(arm="b_cap", coeff=1., kappa=1., lazy_k=4, norm="l2",
                           method="autograd", target_anneal="none")


def _safe_fast(beta, states):
    sink = sink_of(beta)

    def transition(state):
        return kinematic_step(state, pd_action(state, sink))

    return rollout_cost(states, transition, HORIZON, TIME_WEIGHT, CRASH_WEIGHT, SUCCESS_BONUS,
                        V_LIMIT, PAD)


def train_arm(mode, steps=STEPS, adv_weight=ADV_WEIGHT, safe_fast_weight=SAFE_FAST_WEIGHT):
    """Train one arm. `baseline` is GAN only. `combined` adds the safe-fast cost.

    `supervised` is safe-fast with `adv_weight=0` and is not an accepted arm.
    """
    if mode not in ("baseline", "combined", "supervised"):
        raise ValueError(f"Unknown arm {mode}")
    if mode == "supervised":
        if adv_weight != 0:
            raise ValueError("The supervised ablation is adv_weight 0")
    else:
        require_live_adversary(adv_weight)
    torch.manual_seed(SEED)
    beta = nn.Parameter(torch.tensor(-4.0))
    generator = torch.Generator().manual_seed(SEED + 2)
    gan_grad_abs = 0.
    safe_fast_grad_abs = 0.
    late_gan_grad = 0.
    late_count = 0
    applications = 0
    if mode == "supervised":
        opt = torch.optim.SGD([beta], lr=BETA_LR)
        for step in range(1, steps + 1):
            loss = _safe_fast(beta, initial_states(32, 2 + step))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            safe_fast_grad_abs += abs(float(beta.grad.detach()))
            opt.step()
        result = evaluate_policy(_sink_policy(sink_of(beta.detach())))
        result.update(arm="supervised_only", adv_weight=0., safe_fast_weight=SAFE_FAST_WEIGHT,
                      accepted=False, beta=float(beta.detach()),
                      sink=float(sink_of(beta.detach())), gan_grad_abs=0., late_gan_grad=0.,
                      safe_fast_grad_abs=safe_fast_grad_abs, b_cap_applications=0,
                      reason="adv_weight=0 leaves RpGAN and b_cap configured but not applied")
        return result

    norm = _edit_scale()
    critic = _Critic()
    cap = _cap()
    opt = torch.optim.SGD([beta], lr=BETA_LR)
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.0, 0.999))
    hold = 1.3 * float(norm.edit_rms)
    for step in range(1, steps + 1):
        state = initial_states(64, 10000 + step)
        target = expert_action(state)
        pred = pd_action(state, sink_of(beta.detach()))
        residual = (pred - target) / norm.target_std
        sigma = noise_std(step - 1, start=norm.noise_start, decay_steps=steps, hold=hold)
        noise = torch.randn(residual.shape, generator=generator) * sigma
        penalty = cap(critic, noise.detach(), (noise + residual).detach(), step=step)
        loss_d = rp_d_loss(critic(noise), critic(noise + residual)) + penalty
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
        applications += int(step % cap.lazy_k == 0)
        pred = pd_action(state, sink_of(beta))
        residual = (pred - target) / norm.target_std
        noise = torch.randn(residual.shape, generator=generator) * sigma
        loss_g = adv_weight * rp_g_loss(critic(noise).detach(), critic(noise + residual))
        opt.zero_grad(set_to_none=True)
        loss_g.backward()
        gan_now = abs(float(beta.grad.detach()))
        gan_grad_abs += gan_now
        if step > steps * 0.6:
            late_gan_grad += gan_now
            late_count += 1
        if mode == "combined":
            before = beta.grad.detach().clone()
            (_safe_fast(beta, initial_states(24, 20000 + step)) * safe_fast_weight).backward()
            safe_fast_grad_abs += float((beta.grad.detach() - before).abs())
        opt.step()
    result = evaluate_policy(_sink_policy(sink_of(beta.detach())))
    accepted = mode == "combined"
    reason = ("RpGAN weight 1 plus the safe-fast cost" if accepted
              else "RpGAN weight 1 only; slow expert match")
    result.update(arm="combined" if accepted else "baseline_rpgan", adv_weight=float(adv_weight),
                  safe_fast_weight=float(safe_fast_weight) if accepted else 0., accepted=accepted,
                  beta=float(beta.detach()), sink=float(sink_of(beta.detach())),
                  gan_grad_abs=gan_grad_abs, safe_fast_grad_abs=safe_fast_grad_abs,
                  late_gan_grad=late_gan_grad / max(late_count, 1),
                  b_cap_applications=applications, reason=reason)
    return result


def _beats(combined, baseline):
    return (combined["landings"] >= COMBINED_LAND_MIN
            and combined["mean_steps"] <= COMBINED_STEPS_MAX
            and baseline["landings"] <= BASELINE_LAND_MAX
            and baseline["mean_steps"] >= BASELINE_STEPS_MIN
            and combined["landings"] >= baseline["landings"] + LAND_GAP_MIN
            and combined["mean_steps"] <= baseline["mean_steps"] - STEP_GAP_MIN
            and combined["crash_rate"] <= 0.02 and baseline["crash_rate"] <= 0.02
            and combined["sink"] < V_LIMIT)


def run_gate():
    """Pass when the combined arm beats GAN-only on rate and steps, and the GAN stays on."""
    torch.set_num_threads(1)
    refs = reference_policies()
    baseline = train_arm("baseline")
    combined = train_arm("combined")
    supervised = train_arm("supervised", adv_weight=0)
    metric_ok = (refs["hover"]["landings"] == 0
                 and refs["crash_sink"]["landings"] == 0
                 and refs["quick_soft"]["landings"] >= COMBINED_LAND_MIN
                 and refs["hover"]["score"] < refs["quick_soft"]["score"]
                 and refs["crash_sink"]["score"] < refs["quick_soft"]["score"])
    gan_ok = (baseline["adv_weight"] == 1. and baseline["safe_fast_weight"] == 0.
              and baseline["gan_grad_abs"] > GAN_GRAD_MIN and baseline["late_gan_grad"] > LATE_GAN_MIN
              and baseline["b_cap_applications"] > 0 and baseline["safe_fast_grad_abs"] == 0.
              and combined["adv_weight"] == 1. and combined["safe_fast_weight"] == SAFE_FAST_WEIGHT
              and combined["gan_grad_abs"] > GAN_GRAD_MIN
              and combined["late_gan_grad"] > COMBINED_LATE_GAN_MIN
              and combined["b_cap_applications"] > 0
              and combined["safe_fast_grad_abs"] > GAN_GRAD_MIN and combined["accepted"] is True)
    rejected = supervised["adv_weight"] == 0 and supervised["accepted"] is False and supervised["landings"] >= COMBINED_LAND_MIN
    passed = bool(metric_ok and gan_ok and rejected and _beats(combined, baseline))
    return dict(passed=passed, hover=refs["hover"], crash_sink=refs["crash_sink"],
                quick_soft=refs["quick_soft"], baseline=baseline, combined=combined,
                supervised=supervised, mapping=MAPPING,
                thresholds=dict(baseline_land_max=BASELINE_LAND_MAX,
                                baseline_steps_min=BASELINE_STEPS_MIN,
                                combined_land_min=COMBINED_LAND_MIN,
                                combined_steps_max=COMBINED_STEPS_MAX,
                                land_gap_min=LAND_GAP_MIN, step_gap_min=STEP_GAP_MIN,
                                gan_grad_min=GAN_GRAD_MIN, late_gan_min=LATE_GAN_MIN,
                                combined_late_gan_min=COMBINED_LATE_GAN_MIN, adv_weight=ADV_WEIGHT,
                                safe_fast_weight=SAFE_FAST_WEIGHT, time_weight=TIME_WEIGHT,
                                crash_weight=CRASH_WEIGHT, success_bonus=SUCCESS_BONUS,
                                speed_limit=V_LIMIT, pad_half=PAD, horizon=HORIZON))


def _fmt_arm(row):
    extra = ""
    if row.get("sink") is not None:
        extra += f" sink={row['sink']:.3f}"
    if row.get("adv_weight") is not None:
        extra += f" adv_weight={row['adv_weight']} safe_fast_weight={row['safe_fast_weight']}"
        extra += (f" gan_grad_abs={row['gan_grad_abs']:.3f} late_gan_grad={row['late_gan_grad']:.5f} "
                  f"safe_fast_grad_abs={row['safe_fast_grad_abs']:.3f}")
        extra += f" b_cap_applications={row['b_cap_applications']} accepted={row['accepted']} reason={row['reason']}"
    return (f"[safe-fast-2d] {row['arm']} landings={row['landings']:.3f} "
            f"steps={row['mean_steps']:.2f} crash={row['crash_rate']:.3f} "
            f"score={row['score']:.3f}{extra}")


def format_report(result):
    lines = [f"[safe-fast-2d] GATE {'PASS' if result['passed'] else 'FAIL'}"]
    for key in ("hover", "crash_sink", "quick_soft", "baseline", "combined", "supervised"):
        lines.append(_fmt_arm(result[key]))
    limits = result["thresholds"]
    lines.append(
        "[safe-fast-2d] thresholds "
        f"baseline_land<={limits['baseline_land_max']} baseline_steps>={limits['baseline_steps_min']} "
        f"combined_land>={limits['combined_land_min']} combined_steps<={limits['combined_steps_max']} "
        f"land_gap>={limits['land_gap_min']} step_gap>={limits['step_gap_min']} "
        f"late_gan>={limits['late_gan_min']} combined_late_gan>={limits['combined_late_gan_min']} "
        f"adv_weight={limits['adv_weight']} safe_fast_weight={limits['safe_fast_weight']} "
        f"time={limits['time_weight']} crash={limits['crash_weight']} "
        f"success={limits['success_bonus']} speed_limit={limits['speed_limit']} "
        f"pad_half={limits['pad_half']} horizon={limits['horizon']}")
    lines.append("[safe-fast-2d] mapping")
    for row in result["mapping"]:
        lines.append(f"[safe-fast-2d] toy: {row['toy']}")
        lines.append(f"[safe-fast-2d] gym: {row['gym']}")
    return "\n".join(lines)
