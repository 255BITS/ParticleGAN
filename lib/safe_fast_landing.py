"""Safe-fast term that does not erase a closed-loop landing.

PR #21 trained YuE2 RpGAN plus a kinematic cost and the toy PASSed. The same
yaml then scored 0/20 and 0/50 on Lunar, mean return about -407, against the
YuE2 #18 controller at 20/20 and 50/50. Two wiring choices caused that:

* `gym_shaping_cost` fed the physical action ``[main, side]`` into
  ``kinematic_step``, which reads column 0 as lateral acceleration and column
  1 as bipolar vertical acceleration. Lunar's main engine is up-only throttle.
  The cost therefore drove the side engine negative to "descend" in the wrong
  plant, and the craft left the pad.
* ``safe_fast_speed_limit: 0.62`` paid the success bonus for an impact the
  closed-loop pad counts as a crash. A correct channel map with that limit
  still dives through the soft-landing threshold.

The gate scores a closed-loop pad (main engine up only, soft speed 0.18).
The frozen #21 loss and the loose-limit ablation must both fail that score.
The revised loss uses the throttle map and the soft limit, keeps
``adv_weight=1``, and must land more often and sooner than GAN-only.
``adv_weight=0`` is still rejected. This is not a Lunar landing result.

The same module refuses a late corpse. The #23 Lunar train of the throttle-up
yaml was healthy at step 1000 and dead at step 2500. ``select_shipped`` keeps
the earlier checkpoint. A CPU walk-off with ``adv_weight=1`` fails the same
way if the export is the last step, and passes if the export is the selection.
"""
import torch
from torch import nn

from lib.vendor.concept_slider_core.reference import (noise_std, register_paired_error_norm,
    rp_d_loss, rp_g_loss)
from particlegan.grad_regularizers import GradientPenalty

# One gate seed. Not a sweep.
SEED = 0
GAIN = 0.25
GRAVITY = 0.05
TRACK = 0.25
HORIZON = 40
PAD = 0.35
SOFT_SPEED_LIMIT = 0.18
# Shipped in particle_safe_fast.yaml on #21. An impact this fast is a crash
# on the closed-loop pad and was paid as a success.
V21_SPEED_LIMIT = 0.62
MAX_SPEED_LIMIT = 0.25
X_LIMIT = 1.6
Y_LIMIT = 2.4
SLOW = 0.08
SINK_SPAN = 0.50
TEMP = 0.04
TIME_SCORE = 0.5
STEPS = 200
BETA_LR = 0.12
ADV_WEIGHT = 1.0
SAFE_FAST_WEIGHT = 1.0
TIME_WEIGHT = 0.15
CRASH_WEIGHT = 4.0
SUCCESS_BONUS = 2.0
QUICK_SINK = 0.16
CRASH_SINK = 0.40
EVAL_ROWS = 200
ACTION_MAP = "throttle_up"

V21_LAND_MAX = 0.05
V21_CRASH_MIN = 0.40
LOOSE_LAND_MAX = 0.05
BASELINE_LAND_MIN = 0.60
BASELINE_STEPS_MIN = 30.0
FIXED_LAND_MIN = 0.95
FIXED_STEPS_MAX = 28.0
LAND_GAP_MIN = 0.15
STEP_GAP_MIN = 5.0
GAN_GRAD_MIN = 1.0
FIXED_LATE_GAN_MIN = 0.01

MAPPING = (
    dict(toy="Closed-loop score: up-only main engine, |speed| <= 0.18, on the pad. "
             "Hover and a 0.40 sink both lose to a 0.16 soft sink.",
         gym="Not the #21 toy, which scored the same bipolar plant it trained on."),
    dict(toy="Frozen #21 loss: raw [main, side] goes through kinematic_step as "
             "[lateral, bipolar vertical], success if |speed| <= 0.62. "
             "Side engine is driven off the pad. Closed-loop landings collapse.",
         gym="The #21 gym_shaping_cost wiring and safe_fast_speed_limit 0.62. "
             "Rejected by validate when the term is on."),
    dict(toy="Loose-limit ablation: throttle map is correct, speed limit stays 0.62. "
             "The dive is a success in the cost and a crash on the closed-loop pad.",
         gym="Do not put 0.62 back. Max accepted limit is 0.25; the yaml uses 0.18."),
    dict(toy="Fixed loss: throttle_layout then the same integrator, speed limit 0.18, "
             "adv_weight 1. Lands more often and in fewer steps than GAN-only.",
         gym="particle_safe_fast.yaml. safe_fast_action_map throttle_up. "
             "particle.yaml still has safe_fast_weight 0."),
    dict(toy="Supervised safe-fast, adv_weight 0, can land. Rejected.",
         gym="require_live_adversary rejects adv_weight 0."),
)


def require_live_adversary(adv_weight):
    """Reject a configured GAN that the controller step does not apply."""
    if adv_weight == 0:
        raise ValueError("adv_weight=0 leaves RpGAN and b_cap configured but not applied")
    if adv_weight != 1.:
        raise ValueError("adv_weight stays 1 so the controller step is the adversarial loss")


def throttle_layout(action):
    """Map a Lunar ``[main, side]`` command onto ``[lateral, up-only thrust]``.

    Main ``-1`` is engine off. It is not downward thrust. Column 0 of the
    integrator is lateral, so the side engine stays in column 0.
    """
    if action.ndim != 2 or action.shape[1] != 2:
        raise ValueError("Expected an action shaped [rows, 2]")
    main, side = action.unbind(-1)
    thrust = (main.clamp(-1, 1) + 1) * 0.5
    return torch.stack([side.clamp(-1, 1), thrust], -1)


def kinematic_step(state, action):
    """Integrator. Column 0 is lateral acceleration, column 1 is vertical.

    Vertical is bipolar here. Callers that mean a Lunar engine must pass
    ``throttle_layout`` output, whose vertical column is in ``[0, 1]``.
    """
    x, y, vx, vy = state.unbind(-1)
    ax, ay = action.unbind(-1)
    vx = vx + GAIN * ax
    vy = vy + GAIN * ay - GRAVITY
    x = x + GAIN * vx
    y = y + GAIN * vy
    return torch.stack([x, y, vx, vy], -1)


def closed_loop_step(state, action):
    """One step of the scored pad. ``action`` is ``[main, side]``."""
    return kinematic_step(state, throttle_layout(action))


def expert_action(state, sink=SLOW, side_bias=0.):
    """Up-only throttle that tracks ``vy = -sink``, plus a pad-centering side engine."""
    x, _, vx, vy = state.unbind(-1)
    side = (-1.2 * x - 0.8 * vx + side_bias).clamp(-1, 1)
    thrust = (GRAVITY / GAIN - TRACK * (vy + sink)).clamp(0, 1)
    main = thrust * 2 - 1
    return torch.stack([main, side], -1)


def _controls(beta, gamma):
    sink = SLOW + SINK_SPAN * torch.sigmoid(beta)
    side_bias = 0.9 * torch.tanh(gamma)
    return sink, side_bias


def initial_states(n, seed):
    generator = torch.Generator().manual_seed(int(seed))
    x = torch.rand(n, generator=generator) * 0.5 - 0.25
    y = torch.rand(n, generator=generator) * 0.40 + 0.28
    vx = torch.rand(n, generator=generator) * 0.12 - 0.06
    vy = torch.rand(n, generator=generator) * 0.08 - 0.04
    return torch.stack([x, y, vx, vy], -1)


def rollout_cost(state, transition, horizon, time_weight, crash_weight, success_bonus,
                 speed_limit, pad_half, x_limit=X_LIMIT, y_limit=Y_LIMIT, temp=TEMP):
    """Differentiable safe-fast cost. Lower is better. ``transition`` maps a 4-vector."""
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


def _policy_transition(beta, gamma, kind):
    sink, side_bias = _controls(beta, gamma)

    def transition(state):
        action = expert_action(state, sink, side_bias)
        if kind == "v21":
            return kinematic_step(state, action)
        return closed_loop_step(state, action)

    return transition


def shaping_cost(kind, beta, gamma, states):
    """``v21`` is the shipped wiring. ``loose`` keeps its speed limit. ``fixed`` is the revision."""
    if kind not in ("v21", "loose", "fixed"):
        raise ValueError(f"Unknown shaping {kind}")
    limit = V21_SPEED_LIMIT if kind in ("v21", "loose") else SOFT_SPEED_LIMIT
    return rollout_cost(states, _policy_transition(beta, gamma, kind), HORIZON, TIME_WEIGHT,
                        CRASH_WEIGHT, SUCCESS_BONUS, limit, PAD)


def gym_shaping_cost(states, previous, terrain, predict_physical, horizon, time_weight,
                     crash_weight, success_bonus, speed_limit, pad_half):
    """Closed-loop-consistent cost. The main engine is up-only throttle.

    ``states`` is the 8-wide Lunar record. Only ``(x, y, vx, vy)`` move.
    Angle and contacts stay at the batch values. ``predict_physical`` returns
    the physical ``[main, side]`` command. This is not Box2D, and it is not
    the #21 bipolar unpack.
    """
    if states.ndim != 2 or states.shape[1] < 4:
        raise ValueError("Expected a Lunar-style state with x, y, vx, vy in front")
    if speed_limit <= 0 or speed_limit > MAX_SPEED_LIMIT:
        raise ValueError(f"speed_limit must be in (0, {MAX_SPEED_LIMIT}]; "
                         f"{V21_SPEED_LIMIT} is the #21 crash-as-success limit")
    rest = states[:, 4:]
    previous_action = previous

    def transition(kin):
        nonlocal previous_action
        full = torch.cat([kin, rest], 1) if rest.shape[1] else kin
        physical = predict_physical(full, previous_action, terrain).clamp(-1, 1)
        previous_action = physical
        return kinematic_step(kin, throttle_layout(physical))

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
            state = closed_loop_step(state, policy(state))
            x, y, vx, vy = state.unbind(-1)
            contact = (y <= 0) & ~done
            safe = (contact & (x.abs() <= PAD) & (vx.abs() <= SOFT_SPEED_LIMIT)
                    & (vy.abs() <= SOFT_SPEED_LIMIT))
            oob = (~done) & ~contact & ((x.abs() > X_LIMIT) | (y > Y_LIMIT))
            landed = landed | safe
            crashed = crashed | (contact & ~safe) | oob
            steps = torch.where(safe, torch.full_like(steps, float(t)), steps)
            done = done | contact | oob
    return landed, crashed, steps


def evaluate_policy(policy, states=None):
    """Closed-loop landing rate and mean steps among successes."""
    if states is None:
        states = initial_states(EVAL_ROWS, 1000)
    landed, crashed, steps = _hard_rollout(policy, states)
    landings = float(landed.float().mean())
    crash_rate = float(crashed.float().mean())
    mean_steps = float(steps[landed].mean()) if bool(landed.any()) else float(HORIZON)
    return dict(landings=landings, mean_steps=mean_steps, crash_rate=crash_rate,
                score=landings - TIME_SCORE * (mean_steps / HORIZON))


def _frozen_policy(beta, gamma):
    sink, side_bias = _controls(beta.detach(), gamma.detach())
    sink, side_bias = float(sink), float(side_bias)
    return (lambda state, sink=sink, side_bias=side_bias: expert_action(state, sink, side_bias)), sink, side_bias


def reference_policies():
    """Fixed policies on the scored pad. No training."""
    states = initial_states(EVAL_ROWS, 1000)
    rows = {}
    for name, sink in (("hover", 0.), ("crash_sink", CRASH_SINK), ("quick_soft", QUICK_SINK)):
        row = evaluate_policy(lambda state, sink=sink: expert_action(state, sink), states)
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
    register_paired_error_norm(module, expert_action(pool), torch.zeros(512, 2))
    return module


def _cap():
    return GradientPenalty(arm="b_cap", coeff=1., kappa=1., lazy_k=4, norm="l2",
                           method="autograd", target_anneal="none")


def _finish(arm, beta, gamma, adv_weight, safe_fast_weight, accepted, reason, gan_grad_abs,
            late_gan_grad, safe_fast_grad_abs, applications):
    policy, sink, side_bias = _frozen_policy(beta, gamma)
    result = evaluate_policy(policy)
    result.update(arm=arm, adv_weight=float(adv_weight), safe_fast_weight=float(safe_fast_weight),
                  accepted=accepted, sink=sink, side_bias=side_bias, gan_grad_abs=gan_grad_abs,
                  late_gan_grad=late_gan_grad, safe_fast_grad_abs=safe_fast_grad_abs,
                  b_cap_applications=applications, reason=reason)
    return result


def train_arm(mode, steps=STEPS, adv_weight=ADV_WEIGHT, safe_fast_weight=SAFE_FAST_WEIGHT):
    """``baseline`` is GAN only. ``v21`` and ``loose`` are the failed recipes.

    ``fixed`` is throttle-up shaping at the soft speed limit. ``supervised``
    is that shaping with ``adv_weight=0`` and is not accepted.
    """
    if mode not in ("baseline", "v21", "loose", "fixed", "supervised"):
        raise ValueError(f"Unknown arm {mode}")
    if mode == "supervised":
        if adv_weight != 0:
            raise ValueError("The supervised ablation is adv_weight 0")
    else:
        require_live_adversary(adv_weight)
    torch.manual_seed(SEED)
    beta = nn.Parameter(torch.tensor(-4.))
    gamma = nn.Parameter(torch.tensor(0.))
    kind = {"baseline": None, "v21": "v21", "loose": "loose", "fixed": "fixed",
            "supervised": "fixed"}[mode]
    if mode == "supervised":
        opt = torch.optim.SGD((beta, gamma), lr=BETA_LR)
        safe_fast_grad_abs = 0.
        for step in range(1, steps + 1):
            loss = shaping_cost("fixed", beta, gamma, initial_states(24, 2 + step))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            safe_fast_grad_abs += float(beta.grad.detach().abs() + gamma.grad.detach().abs())
            opt.step()
        return _finish("supervised_only", beta, gamma, 0., safe_fast_weight, False,
                       "adv_weight=0 leaves RpGAN and b_cap configured but not applied",
                       0., 0., safe_fast_grad_abs, 0)

    norm = _edit_scale()
    critic = _Critic()
    cap = _cap()
    opt = torch.optim.SGD((beta, gamma), lr=BETA_LR)
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0., 0.999))
    generator = torch.Generator().manual_seed(SEED + 2)
    hold = 1.3 * float(norm.edit_rms)
    gan_grad_abs = late_gan = late_count = safe_fast_grad_abs = 0.
    applications = 0
    for step in range(1, steps + 1):
        state = initial_states(64, 10000 + step)
        target = expert_action(state)
        sink_d, bias_d = _controls(beta.detach(), gamma.detach())
        pred = expert_action(state, sink_d, bias_d)
        residual = (pred - target) / norm.target_std
        sigma = noise_std(step - 1, start=norm.noise_start, decay_steps=steps, hold=hold)
        noise = torch.randn(residual.shape, generator=generator) * sigma
        penalty = cap(critic, noise.detach(), (noise + residual).detach(), step=step)
        loss_d = rp_d_loss(critic(noise), critic(noise + residual)) + penalty
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
        applications += int(step % cap.lazy_k == 0)
        sink, bias = _controls(beta, gamma)
        pred = expert_action(state, sink, bias)
        residual = (pred - target) / norm.target_std
        noise = torch.randn(residual.shape, generator=generator) * sigma
        loss_g = adv_weight * rp_g_loss(critic(noise).detach(), critic(noise + residual))
        opt.zero_grad(set_to_none=True)
        loss_g.backward()
        gan_now = float(beta.grad.detach().abs() + gamma.grad.detach().abs())
        gan_grad_abs += gan_now
        if step > steps * 0.6:
            late_gan += gan_now
            late_count += 1
        if kind is not None:
            before = torch.stack((beta.grad.detach(), gamma.grad.detach()))
            (safe_fast_weight * shaping_cost(kind, beta, gamma, initial_states(24, 20000 + step))).backward()
            after = torch.stack((beta.grad.detach(), gamma.grad.detach()))
            safe_fast_grad_abs += float((after - before).abs().sum())
        opt.step()
    reasons = dict(
        baseline="RpGAN weight 1 only; slow expert on the closed-loop pad",
        v21="shipped #21 bipolar unpack of [main, side] plus speed limit 0.62",
        loose="throttle map with the #21 speed limit 0.62; soft closed loop still crashes",
        fixed="RpGAN weight 1 plus throttle-up shaping at speed limit 0.18")
    names = dict(baseline="baseline_rpgan", v21="v21_shipped", loose="loose_limit", fixed="fixed")
    weight = 0. if kind is None else safe_fast_weight
    return _finish(names[mode], beta, gamma, adv_weight, weight, mode == "fixed", reasons[mode],
                   gan_grad_abs, late_gan / max(late_count, 1), safe_fast_grad_abs, applications)


def _beats(fixed, baseline):
    return (fixed["landings"] >= FIXED_LAND_MIN and fixed["mean_steps"] <= FIXED_STEPS_MAX
            and fixed["crash_rate"] <= 0.02 and baseline["crash_rate"] <= 0.02
            and baseline["landings"] >= BASELINE_LAND_MIN and baseline["mean_steps"] >= BASELINE_STEPS_MIN
            and fixed["landings"] >= baseline["landings"] + LAND_GAP_MIN
            and fixed["mean_steps"] <= baseline["mean_steps"] - STEP_GAP_MIN
            and fixed["sink"] <= SOFT_SPEED_LIMIT and abs(fixed["side_bias"]) < 0.05)


def _collapsed(row, land_max, crash_min=None):
    ok = row["landings"] <= land_max and row["adv_weight"] == 1. and row["b_cap_applications"] > 0
    if crash_min is not None:
        ok = ok and row["crash_rate"] >= crash_min
    return ok


def run_gate():
    """Pass only when #21's recipe fails the closed-loop pad and the revision lands."""
    torch.set_num_threads(1)
    refs = reference_policies()
    baseline = train_arm("baseline")
    shipped = train_arm("v21")
    loose = train_arm("loose")
    fixed = train_arm("fixed")
    supervised = train_arm("supervised", adv_weight=0)
    metric_ok = (refs["hover"]["landings"] == 0 and refs["crash_sink"]["landings"] == 0
                 and refs["quick_soft"]["landings"] >= FIXED_LAND_MIN
                 and refs["hover"]["score"] < refs["quick_soft"]["score"]
                 and refs["crash_sink"]["score"] < refs["quick_soft"]["score"])
    shipped_fails = _collapsed(shipped, V21_LAND_MAX, V21_CRASH_MIN) and shipped["accepted"] is False
    loose_fails = _collapsed(loose, LOOSE_LAND_MAX) and loose["accepted"] is False
    gan_ok = (baseline["adv_weight"] == 1. and baseline["safe_fast_weight"] == 0.
              and baseline["b_cap_applications"] > 0 and baseline["safe_fast_grad_abs"] == 0.
              and fixed["adv_weight"] == 1. and fixed["safe_fast_weight"] == SAFE_FAST_WEIGHT
              and fixed["gan_grad_abs"] > GAN_GRAD_MIN and fixed["late_gan_grad"] > FIXED_LATE_GAN_MIN
              and fixed["b_cap_applications"] > 0 and fixed["safe_fast_grad_abs"] > GAN_GRAD_MIN
              and fixed["accepted"] is True)
    rejected = (supervised["adv_weight"] == 0 and supervised["accepted"] is False
                and supervised["landings"] >= FIXED_LAND_MIN)
    passed = bool(metric_ok and shipped_fails and loose_fails and gan_ok and rejected
                  and _beats(fixed, baseline))
    return dict(passed=passed, hover=refs["hover"], crash_sink=refs["crash_sink"],
                quick_soft=refs["quick_soft"], baseline=baseline, v21=shipped, loose=loose,
                fixed=fixed, supervised=supervised, mapping=MAPPING,
                thresholds=dict(v21_land_max=V21_LAND_MAX, v21_crash_min=V21_CRASH_MIN,
                                loose_land_max=LOOSE_LAND_MAX, baseline_land_min=BASELINE_LAND_MIN,
                                baseline_steps_min=BASELINE_STEPS_MIN, fixed_land_min=FIXED_LAND_MIN,
                                fixed_steps_max=FIXED_STEPS_MAX, land_gap_min=LAND_GAP_MIN,
                                step_gap_min=STEP_GAP_MIN, fixed_late_gan_min=FIXED_LATE_GAN_MIN,
                                adv_weight=ADV_WEIGHT, safe_fast_weight=SAFE_FAST_WEIGHT,
                                time_weight=TIME_WEIGHT, crash_weight=CRASH_WEIGHT,
                                success_bonus=SUCCESS_BONUS, speed_limit=SOFT_SPEED_LIMIT,
                                v21_speed_limit=V21_SPEED_LIMIT, pad_half=PAD, horizon=HORIZON,
                                action_map=ACTION_MAP))


def _fmt_arm(row):
    extra = ""
    if row.get("sink") is not None:
        extra += f" sink={row['sink']:.3f} side_bias={row['side_bias']:.3f}"
    if row.get("adv_weight") is not None:
        extra += (f" adv_weight={row['adv_weight']} safe_fast_weight={row['safe_fast_weight']} "
                  f"gan_grad_abs={row['gan_grad_abs']:.3f} late_gan_grad={row['late_gan_grad']:.5f} "
                  f"safe_fast_grad_abs={row['safe_fast_grad_abs']:.3f} "
                  f"b_cap_applications={row['b_cap_applications']} accepted={row['accepted']} "
                  f"reason={row['reason']}")
    return (f"[safe-fast-2d] {row['arm']} landings={row['landings']:.3f} "
            f"steps={row['mean_steps']:.2f} crash={row['crash_rate']:.3f} "
            f"score={row['score']:.3f}{extra}")


# Reported Lunar run of the revised throttle-up yaml (PR #23), shared 20-ep
# validation. Rounded figures from that board. Not remeasured in this checkout.
# Checkpoints on disk were 250, 1000, and 2500. Steps 1750 and 2000 are log rows.
PR23_CHECKPOINTS = (250, 1000, 2500)
PR23_TRACE = (
    dict(step=250, success=0.0, loss=None, g_loss=None, d_loss=None,
         action_mse=None, safe_fast=None),
    dict(step=1000, success=17 / 20, loss=0.44, g_loss=0.44, d_loss=0.60,
         action_mse=0.045, safe_fast=None),
    dict(step=1750, success=None, loss=0.65, g_loss=0.65, d_loss=0.63,
         action_mse=0.069, safe_fast=-0.46),
    dict(step=2000, success=None, loss=12.2, g_loss=12.2, d_loss=0.012,
         action_mse=2.28, safe_fast=4.9),
    dict(step=2500, success=0.0, loss=13.7, g_loss=13.7, d_loss=0.01,
         action_mse=2.2, safe_fast=None),
)
MSE_JUMP = 10.0
MSE_CORPSE_FLOOR = 0.2
D_COLLAPSE_MAX = 0.05
D_HEALTHY_MIN = 0.30
LOSS_EXPLODE_MIN = 2.0
SAFE_EXPLODE_MIN = 1.0
SUCCESS_HIGH = 0.70
SUCCESS_LOW = 0.10
COLLAPSE_STEPS = 200
COLLAPSE_BETA_LR = 0.04
COLLAPSE_LOG_EVERY = 25


def _metric(row, key):
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def collapse_reason(row, rows):
    """Why this row is a corpse given earlier rows. None means it can still ship.

    A row fails when an earlier row was healthy and this one matches the #23
    signature: diagnostic MSE leaves the healthy window by an order of
    magnitude, the critic loss falls apart while the generator or safe-fast
    term blows up, or closed-loop success drops from a high bar to a low one.
    """
    earlier = [item for item in rows if item["step"] < row["step"]]
    mse = _metric(row, "action_mse")
    if mse is not None:
        refs = [value for value in (_metric(item, "action_mse") for item in earlier) if value and value > 0]
        if refs and mse >= MSE_CORPSE_FLOOR and mse > MSE_JUMP * min(refs):
            return (f"diag_action_mse {mse:.4f} jumped >{MSE_JUMP:.0f}x "
                    f"above the earlier minimum {min(refs):.4f}")
    d_loss = _metric(row, "d_loss")
    loss = _metric(row, "loss")
    safe = _metric(row, "safe_fast")
    if d_loss is not None and d_loss <= D_COLLAPSE_MAX:
        earlier_d = any((_metric(item, "d_loss") or 0) >= D_HEALTHY_MIN for item in earlier)
        exploded = ((loss is not None and loss >= LOSS_EXPLODE_MIN)
                    or (safe is not None and safe >= SAFE_EXPLODE_MIN))
        if earlier_d and exploded:
            return (f"D-loss {d_loss:.4f} collapsed while loss/safe_fast exploded "
                    f"(loss={loss}, safe_fast={safe})")
    success = _metric(row, "success")
    if success is not None and success <= SUCCESS_LOW:
        if any((_metric(item, "success") or 0) >= SUCCESS_HIGH for item in earlier):
            return (f"closed-loop success {success:.3f} fell to <={SUCCESS_LOW} "
                    f"after an earlier row cleared {SUCCESS_HIGH}")
    return None


def select_shipped(rows, candidates=None):
    """Pick the checkpoint to export. Never the final step after a diag spike.

    Closed-loop success breaks ties toward the better landing rate. Without
    that column, the last row that is still healthy is the one that ships.
    """
    if not rows:
        raise ValueError("late-collapse selection has no rows")
    allowed = None if candidates is None else {int(step) for step in candidates}
    pool = [row for row in rows if allowed is None or int(row["step"]) in allowed]
    if not pool:
        raise ValueError("late-collapse selection has no checkpoint rows")
    alive = [row for row in pool if collapse_reason(row, rows) is None]
    final = max(pool, key=lambda row: row["step"])
    if not alive:
        chosen = min(pool, key=lambda row: row["step"])
        return chosen, f"every checkpoint is a corpse; refused step {final['step']}"
    ranked = [row for row in alive if _metric(row, "success") is not None]
    if ranked:
        chosen = max(ranked, key=lambda row: (_metric(row, "success"), row["step"]))
    else:
        chosen = max(alive, key=lambda row: row["step"])
    if chosen["step"] != final["step"]:
        return chosen, f"refused step {final['step']}: {collapse_reason(final, rows)}"
    return chosen, "final step stayed inside the late-collapse gate"


def judge_export(rows, shipped_step, candidates=None):
    """PASS when the shipped row is healthy, or it is the selected healthy row.

    Shipping the final corpse while an earlier row would pass is a FAIL.
    """
    chosen, selection_reason = select_shipped(rows, candidates)
    shipped = next(row for row in rows if int(row["step"]) == int(shipped_step))
    reason = collapse_reason(shipped, rows)
    matches = int(shipped_step) == int(chosen["step"])
    mid_ok = any(collapse_reason(row, rows) is None and row["step"] < shipped["step"] for row in rows)
    passed = reason is None and (matches or mid_ok)
    return dict(passed=passed, shipped_step=int(shipped_step), selected_step=int(chosen["step"]),
                corpse_reason=reason, selection_reason=selection_reason, shipped=shipped, selected=chosen)


def train_late_collapse(steps=COLLAPSE_STEPS, log_every=COLLAPSE_LOG_EVERY):
    """Constant-aux walk-off on the up-only pad.

    Same RpGAN at ``adv_weight=1`` as the other arms. The aux is the loose
    speed limit: soft success still pays past the hard pad, so a slow step
    size climbs through a healthy window and finishes crashed. Snapshots are
    the export candidates. This is the CPU sample of shipping step=max.
    """
    require_live_adversary(ADV_WEIGHT)
    torch.manual_seed(SEED)
    beta = nn.Parameter(torch.tensor(-4.))
    gamma = nn.Parameter(torch.tensor(0.))
    norm = _edit_scale()
    critic = _Critic()
    cap = _cap()
    opt = torch.optim.SGD((beta, gamma), lr=COLLAPSE_BETA_LR)
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0., 0.999))
    generator = torch.Generator().manual_seed(SEED + 2)
    hold = 1.3 * float(norm.edit_rms)
    panel = initial_states(EVAL_ROWS, 1000)
    rows = []
    snapshots = {}
    for step in range(1, steps + 1):
        state = initial_states(64, 10000 + step)
        target = expert_action(state)
        sink_d, bias_d = _controls(beta.detach(), gamma.detach())
        pred = expert_action(state, sink_d, bias_d)
        residual = (pred - target) / norm.target_std
        sigma = noise_std(step - 1, start=norm.noise_start, decay_steps=steps, hold=hold)
        noise = torch.randn(residual.shape, generator=generator) * sigma
        penalty = cap(critic, noise.detach(), (noise + residual).detach(), step=step)
        loss_d = rp_d_loss(critic(noise), critic(noise + residual)) + penalty
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
        sink, bias = _controls(beta, gamma)
        pred = expert_action(state, sink, bias)
        residual = (pred - target) / norm.target_std
        noise = torch.randn(residual.shape, generator=generator) * sigma
        loss_g = ADV_WEIGHT * rp_g_loss(critic(noise).detach(), critic(noise + residual))
        safe = shaping_cost("loose", beta, gamma, initial_states(24, 20000 + step))
        loss = loss_g + safe
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            policy, sink_v, side_v = _frozen_policy(beta, gamma)
            scored = evaluate_policy(policy, panel)
            with torch.no_grad():
                action_mse = float(((pred.detach() - target) ** 2).mean())
            row = dict(step=step, loss=float(loss.detach()), g_loss=float(loss_g.detach()),
                       d_loss=float(loss_d.detach()), safe_fast=float(safe.detach()),
                       action_mse=action_mse, success=scored["landings"],
                       mean_steps=scored["mean_steps"], crash_rate=scored["crash_rate"],
                       sink=sink_v, side_bias=side_v, adv_weight=ADV_WEIGHT)
            rows.append(row)
            snapshots[step] = (sink_v, side_v)
            print(f"[late-collapse] step={step}/{steps} loss={row['loss']:.4f} "
                  f"D={row['d_loss']:.4f} safe_fast={row['safe_fast']:.4f} "
                  f"diag_action_mse={row['action_mse']:.5f} success={row['success']:.3f} "
                  f"sink={sink_v:.3f}", flush=True)
    return dict(rows=rows, snapshots=snapshots, panel_seed=1000, adv_weight=ADV_WEIGHT)


def run_collapse_gate():
    """Fail a blind final export of #23 and of the live walk-off. Pass selection."""
    torch.set_num_threads(1)
    trace = [dict(row) for row in PR23_TRACE]
    frozen_blind = judge_export(trace, PR23_CHECKPOINTS[-1], PR23_CHECKPOINTS)
    frozen_selected = judge_export(trace, frozen_blind["selected_step"], PR23_CHECKPOINTS)
    live = train_late_collapse()
    rows = live["rows"]
    steps = [row["step"] for row in rows]
    live_blind = judge_export(rows, steps[-1], steps)
    live_selected = judge_export(rows, live_blind["selected_step"], steps)
    panel = initial_states(EVAL_ROWS, live["panel_seed"])

    def rescore(step):
        sink, side = live["snapshots"][step]
        return evaluate_policy(lambda state, sink=sink, side=side: expert_action(state, sink, side), panel)

    shipped_eval = rescore(live_selected["shipped_step"])
    final_eval = rescore(live_blind["shipped_step"])
    logged_shipped = next(row for row in rows if row["step"] == live_selected["shipped_step"])
    artifact_matches = abs(shipped_eval["landings"] - logged_shipped["success"]) < 1e-6
    frozen_ok = ((not frozen_blind["passed"]) and frozen_selected["passed"]
                 and frozen_selected["selected_step"] == 1000
                 and frozen_blind["shipped_step"] == 2500)
    live_ok = ((not live_blind["passed"]) and live_selected["passed"]
               and live_selected["selected_step"] != steps[-1]
               and logged_shipped["success"] >= SUCCESS_HIGH
               and final_eval["landings"] <= SUCCESS_LOW
               and shipped_eval["landings"] >= SUCCESS_HIGH
               and artifact_matches and logged_shipped["adv_weight"] == 1.)
    return dict(passed=bool(frozen_ok and live_ok), frozen_blind=frozen_blind,
                frozen_selected=frozen_selected, live_blind=live_blind,
                live_selected=live_selected, live_rows=rows, shipped_eval=shipped_eval,
                final_eval=final_eval, artifact_matches=artifact_matches)


def format_collapse_report(result):
    lines = [f"[late-collapse] GATE {'PASS' if result['passed'] else 'FAIL'}"]
    frozen = result["frozen_blind"]
    picked = result["frozen_selected"]
    lines.append("[late-collapse] pr23_blind " + ("FAIL" if not frozen["passed"] else "PASS")
                 + f" shipped={frozen['shipped_step']} selected_would_be={frozen['selected_step']} "
                 + f"reason={frozen['corpse_reason']}")
    lines.append("[late-collapse] pr23_selected " + ("PASS" if picked["passed"] else "FAIL")
                 + f" shipped={picked['shipped_step']} {picked['selection_reason']}")
    for row in PR23_TRACE:
        lines.append("[late-collapse] pr23 "
                     f"step={row['step']} success={row['success']} loss={row['loss']} "
                     f"D={row['d_loss']} safe_fast={row['safe_fast']} "
                     f"diag_action_mse={row['action_mse']}")
    lines.append("[late-collapse] live_blind " + ("FAIL" if not result["live_blind"]["passed"] else "PASS")
                 + f" shipped={result['live_blind']['shipped_step']} "
                 + f"reason={result['live_blind']['corpse_reason']}")
    lines.append("[late-collapse] live_selected " + ("PASS" if result["live_selected"]["passed"] else "FAIL")
                 + f" shipped={result['live_selected']['shipped_step']} "
                 + f"{result['live_selected']['selection_reason']} "
                 + f"rescore={result['shipped_eval']['landings']:.3f} "
                 + f"final_rescore={result['final_eval']['landings']:.3f} "
                 + f"artifact_matches={result['artifact_matches']}")
    for row in result["live_rows"]:
        lines.append("[late-collapse] live "
                     f"step={row['step']} success={row['success']:.3f} loss={row['loss']:.4f} "
                     f"D={row['d_loss']:.4f} safe_fast={row['safe_fast']:.4f} "
                     f"diag_action_mse={row['action_mse']:.5f} sink={row['sink']:.3f}")
    lines.append("[late-collapse] rule ship the last healthy checkpoint; "
                 "closed-loop success wins when it was logged; "
                 "diag_action_mse >10x with floor 0.2, or D<=0.05 while loss/safe_fast explode, "
                 "or success drops from >=0.7 to <=0.1, refuses step=max")
    lines.append("[late-collapse] adv_weight=1. Lunar retrain of this export rule has not been run.")
    return "\n".join(lines)


def format_report(result):
    lines = [f"[safe-fast-2d] GATE {'PASS' if result['passed'] else 'FAIL'}"]
    lines.append("[safe-fast-2d] v21_shipped " + ("FAIL" if result["v21"]["landings"] <= V21_LAND_MAX else "PASS")
                 + " closed-loop landings; this is the #21 recipe")
    lines.append("[safe-fast-2d] loose_limit " + ("FAIL" if result["loose"]["landings"] <= LOOSE_LAND_MAX else "PASS")
                 + " closed-loop landings; throttle map with speed limit 0.62")
    for key in ("hover", "crash_sink", "quick_soft", "baseline", "v21", "loose", "fixed", "supervised"):
        lines.append(_fmt_arm(result[key]))
    limits = result["thresholds"]
    lines.append(
        "[safe-fast-2d] thresholds "
        f"v21_land<={limits['v21_land_max']} v21_crash>={limits['v21_crash_min']} "
        f"loose_land<={limits['loose_land_max']} baseline_land>={limits['baseline_land_min']} "
        f"baseline_steps>={limits['baseline_steps_min']} fixed_land>={limits['fixed_land_min']} "
        f"fixed_steps<={limits['fixed_steps_max']} land_gap>={limits['land_gap_min']} "
        f"step_gap>={limits['step_gap_min']} late_gan>={limits['fixed_late_gan_min']} "
        f"adv_weight={limits['adv_weight']} safe_fast_weight={limits['safe_fast_weight']} "
        f"speed_limit={limits['speed_limit']} v21_speed_limit={limits['v21_speed_limit']} "
        f"action_map={limits['action_map']} horizon={limits['horizon']}")
    lines.append("[safe-fast-2d] mapping")
    for row in result["mapping"]:
        lines.append(f"[safe-fast-2d] toy: {row['toy']}")
        lines.append(f"[safe-fast-2d] gym: {row['gym']}")
    return "\n".join(lines)
