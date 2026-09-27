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

Each arm is a ``SafeFastLanding`` problem on ``benchmarks.toy_runner``: the
plant, the expert, the sink-gain controller, the critic, the safe-fast cost
and the metrics are declared here; optimizers (and their LR schedule), the
RpGAN loss, the critic penalty, critic input noise, generator output noise
and EMA come from the shipped recipe. The gate compares the arms' runs.
"""
import torch
from torch import nn

from benchmarks.toy_runner import Networks, Sample, ToyProblem, View, run
from particlegan import get_recipe, init

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
BETA_INIT = -4.0
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
CRASH_MAX = 0.02
# GAN-only has no other force on the sink gain, so holding the slow expert is
# the evidence that its GAN is live.
BASELINE_SINK_TOL = 0.02
# The GAN gradient on the gain under the final critic, for the combined arm.
COMBINED_LATE_GAN_MIN = 0.2

ARMS = {"baseline": "baseline_rpgan", "combined": "combined", "supervised": "supervised_only"}
REASONS = {"baseline": "RpGAN weight 1 only; slow expert match",
           "combined": "RpGAN weight 1 plus the safe-fast cost",
           "supervised": "adv_weight=0 leaves RpGAN and its critic penalty configured but not applied"}

MAPPING = (
    dict(toy="Paired-error RpGAN on the action residual, adv_weight 1, recipe critic penalty. "
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
        raise ValueError("adv_weight=0 leaves RpGAN and its critic penalty configured but not applied")
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


def sample_states(n, generator):
    """Start states (x, y, vx, vy) drawn from ``generator``."""
    x = torch.rand(n, generator=generator) * 0.8 - 0.4
    y = torch.rand(n, generator=generator) * 1.3 + 1.05
    vx = torch.rand(n, generator=generator) * 0.3 - 0.15
    vy = torch.rand(n, generator=generator) * 0.2 - 0.1
    return torch.stack([x, y, vx, vy], -1)


def initial_states(n, seed):
    return sample_states(n, torch.Generator().manual_seed(int(seed)))


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


class SinkGain(nn.Module):
    """The controller: one scalar gain on the PD law's sink rate. Starts far from the expert."""

    def __init__(self, beta=BETA_INIT):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, state):
        return pd_action(state, sink_of(self.beta))


def residual_scale(pool=None):
    """Per-coordinate scale of the expert's paired edit (its action against zero),
    with one scalar gain so the median row RMS of the normalized edit is 1."""
    edits = expert_action(initial_states(512, 3) if pool is None else pool)
    scale = edits.std(0).clamp_min(1e-6)
    return scale * (edits / scale).pow(2).mean(-1).sqrt().median()


def _safe_fast(beta, states):
    sink = sink_of(beta)

    def transition(state):
        return kinematic_step(state, pd_action(state, sink))

    return rollout_cost(states, transition, HORIZON, TIME_WEIGHT, CRASH_WEIGHT, SUCCESS_BONUS,
                        V_LIMIT, PAD)


class SafeFastLanding(ToyProblem):
    """Paired-error GAN on the action residual of a scalar sink gain.

    ``real`` is the expert's residual against itself (zero) at sampled starts;
    ``fake`` is the controller's residual at the same starts. ``baseline`` is
    that GAN only; ``combined`` adds ``safe_fast_weight`` times the safe-fast
    cost as a generator loss; ``supervised`` has no critic (``adv_weight=0``)
    and is never an accepted arm.
    """

    def __init__(self, mode, *, adv_weight=ADV_WEIGHT, safe_fast_weight=SAFE_FAST_WEIGHT, steps=STEPS):
        if mode not in ARMS:
            raise ValueError(f"Unknown arm {mode}")
        if mode == "supervised":
            if adv_weight != 0:
                raise ValueError("The supervised ablation is adv_weight 0")
        else:
            require_live_adversary(adv_weight)
        self.mode, self.adv_weight, self.steps = mode, float(adv_weight), int(steps)
        self.safe_fast_weight = 0. if mode == "baseline" else float(safe_fast_weight)
        self.name = f"safe_fast_2d_{mode}"
        self.scale = residual_scale()
        self.loss = self.recipe().make_loss()

    def recipe(self):
        return get_recipe(total_steps=self.steps, batch_size=64)

    def networks(self, recipe, seed):
        critics = {} if self.mode == "supervised" else init.deterministic_orthogonal_(_Critic(), seed=seed + 1)
        return Networks(generator=SinkGain(), critics=critics, prior=None)

    def real(self, n, stream):
        states = sample_states(n, stream)
        return Sample(torch.zeros(n, 2), condition=(states,))

    def _residual(self, gain, states):
        return (gain(states) - expert_action(states)) / self.scale

    def fake(self, nets, n, stream, real):
        states = sample_states(n, stream) if real is None else real.condition[0]
        return Sample(self._residual(nets.generator, states), condition=(states,))

    def views(self, nets, real, fake):
        # The critic reads the residual only; the starts ride along for fake() and losses().
        return [View("critic", real.x, fake.x, (), self.adv_weight)] if nets.critics else []

    def losses(self, role, nets, real, fake):
        if role != "generator" or self.safe_fast_weight == 0:
            return {}
        return {"safe_fast": self.safe_fast_weight * _safe_fast(nets.generator.beta, real.condition[0])}

    def metrics(self, model):
        beta = model.nets.generator.beta.detach()
        row = evaluate_policy(_sink_policy(sink_of(beta)))
        row.update(arm=ARMS[self.mode], adv_weight=self.adv_weight, safe_fast_weight=self.safe_fast_weight,
                   accepted=self.mode == "combined", beta=float(beta), sink=float(sink_of(beta)),
                   reason=REASONS[self.mode], **self._gradients(model))
        return row

    def _gradients(self, model):
        """|d loss / d beta| on fixed starts: the GAN term under the current critic, and the safe-fast term."""
        states = initial_states(256, 1001)
        gain = SinkGain(model.nets.generator.beta.detach())
        out = {"gan_grad": 0., "safe_fast_grad": 0.}
        with torch.enable_grad():
            if model.nets.critics:
                critic, residual = model.nets.critics, self._residual(gain, states)
                g = self.adv_weight * self.loss.g_loss(critic(residual), critic(torch.zeros_like(residual)))
                out["gan_grad"] = abs(float(torch.autograd.grad(g, gain.beta)[0]))
            if self.safe_fast_weight:
                cost = self.safe_fast_weight * _safe_fast(gain.beta, states[:64])
                out["safe_fast_grad"] = abs(float(torch.autograd.grad(cost, gain.beta)[0]))
        return out

    def verdict(self, m):
        if self.mode == "baseline":
            ok = (m["landings"] <= BASELINE_LAND_MAX and m["mean_steps"] >= BASELINE_STEPS_MIN
                  and abs(m["sink"] - SLOW) <= BASELINE_SINK_TOL and m["gan_grad"] > 0)
        elif self.mode == "combined":
            ok = (m["landings"] >= COMBINED_LAND_MIN and m["mean_steps"] <= COMBINED_STEPS_MAX
                  and m["sink"] < V_LIMIT and m["gan_grad"] >= COMBINED_LATE_GAN_MIN
                  and m["safe_fast_grad"] > 0)
        else:  # the ablation may land; the gate still rejects it
            ok = m["landings"] >= COMBINED_LAND_MIN
        return "PASS" if ok and m["crash_rate"] <= CRASH_MAX else "FAIL"


def train_arm(mode, steps=STEPS, adv_weight=ADV_WEIGHT, safe_fast_weight=SAFE_FAST_WEIGHT, log=None):
    """Train one arm on the shared toy runner; its live metrics row, verdict and hold summary."""
    problem = SafeFastLanding(mode, adv_weight=adv_weight, safe_fast_weight=safe_fast_weight, steps=steps)
    result = run(problem, seed=SEED, log=log)
    return {**result["live"], "hold": result["hold"]}


def run_gate(log=None):
    """Pass when the combined arm beats GAN-only on rate and steps, and the GAN stays on."""
    torch.set_num_threads(1)
    refs = reference_policies()
    baseline = train_arm("baseline", log=log)
    combined = train_arm("combined", log=log)
    supervised = train_arm("supervised", adv_weight=0, log=log)
    metric_ok = (refs["hover"]["landings"] == 0
                 and refs["crash_sink"]["landings"] == 0
                 and refs["quick_soft"]["landings"] >= COMBINED_LAND_MIN
                 and refs["hover"]["score"] < refs["quick_soft"]["score"]
                 and refs["crash_sink"]["score"] < refs["quick_soft"]["score"])
    gap_ok = (combined["landings"] >= baseline["landings"] + LAND_GAP_MIN
              and combined["mean_steps"] <= baseline["mean_steps"] - STEP_GAP_MIN)
    arms_ok = (baseline["verdict"] == "PASS" and combined["verdict"] == "PASS"
               and supervised["verdict"] == "PASS" and supervised["accepted"] is False)
    passed = bool(metric_ok and gap_ok and arms_ok)
    return dict(passed=passed, hover=refs["hover"], crash_sink=refs["crash_sink"],
                quick_soft=refs["quick_soft"], baseline=baseline, combined=combined,
                supervised=supervised, mapping=MAPPING,
                thresholds=dict(baseline_land_max=BASELINE_LAND_MAX,
                                baseline_steps_min=BASELINE_STEPS_MIN,
                                combined_land_min=COMBINED_LAND_MIN,
                                combined_steps_max=COMBINED_STEPS_MAX,
                                land_gap_min=LAND_GAP_MIN, step_gap_min=STEP_GAP_MIN,
                                baseline_sink_tol=BASELINE_SINK_TOL,
                                combined_late_gan_min=COMBINED_LATE_GAN_MIN, adv_weight=ADV_WEIGHT,
                                safe_fast_weight=SAFE_FAST_WEIGHT, time_weight=TIME_WEIGHT,
                                crash_weight=CRASH_WEIGHT, success_bonus=SUCCESS_BONUS,
                                speed_limit=V_LIMIT, pad_half=PAD, horizon=HORIZON))


def _fmt_arm(row):
    extra = ""
    if row.get("sink") is not None:
        extra += f" sink={row['sink']:.3f}"
    if row.get("adv_weight") is not None:
        extra += (f" verdict={row['verdict']} adv_weight={row['adv_weight']} "
                  f"safe_fast_weight={row['safe_fast_weight']} gan_grad={row['gan_grad']:.4f} "
                  f"safe_fast_grad={row['safe_fast_grad']:.4f} accepted={row['accepted']} "
                  f"reason={row['reason']}")
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
        f"baseline_sink_tol={limits['baseline_sink_tol']} combined_late_gan>={limits['combined_late_gan_min']} "
        f"adv_weight={limits['adv_weight']} safe_fast_weight={limits['safe_fast_weight']} "
        f"time={limits['time_weight']} crash={limits['crash_weight']} "
        f"success={limits['success_bonus']} speed_limit={limits['speed_limit']} "
        f"pad_half={limits['pad_half']} horizon={limits['horizon']}")
    lines.append("[safe-fast-2d] mapping")
    for row in result["mapping"]:
        lines.append(f"[safe-fast-2d] toy: {row['toy']}")
        lines.append(f"[safe-fast-2d] gym: {row['gym']}")
    return "\n".join(lines)
