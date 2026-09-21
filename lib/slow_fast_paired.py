"""CPU gate: finetune a slow pad landing into a fast one with YuE2 paired-error RpGAN.

A working slow law and a faster successful law are oracle stand-ins used only
to label data. The student is a separate linear policy, initialized at the
slow law. It never plays back the fast law. Training rows are (slow action,
fast action) on states from starts where both rollouts land and the fast one
uses fewer steps. Crashes stay out of that fast set.

The controller step is the #18 game: relativistic logistic loss on
noise versus noise plus the normalized action residual, sample-point b_cap
every fourth update, adversarial weight 1. Diagnostic MSE is logged under
no_grad and is not in the loss. The edit scale is std(fast − slow).

Controls that must not win the rank key: the slow initialization with no
update, a finetune whose target is the slow member, and a finetune whose
"fast" actions are crashes. Unpaired fast actions from other starts, and
MSE with adv_weight 0, are failure modes. This file does not run Lunar Lander.
"""
import torch
from torch import nn
import torch.nn.functional as F

from lib.vendor.concept_slider_core.reference import (noise_std, register_paired_error_norm,
    rp_d_loss, rp_g_loss)
from particlegan.grad_regularizers import GradientPenalty

# One gate seed. Not a sweep.
SEED = 0
GAIN = 0.25
GRAVITY = 0.04
HORIZON = 64
PAD = 0.35
V_LIMIT = 0.62
X_LIMIT = 2.8
Y_LIMIT = 3.5
TIME_SCORE = 0.5
STEPS = 400
BATCH = 96
POLICY_LR = 0.02
CRITIC_LR = 1e-3
GRAD_CLIP = 1.0
ADV_WEIGHT = 1.0
COLLECT_STARTS = 80
EVAL_ROWS = 400
LOG_EVERY = 100

# Slow law: gentle sink, moderate lateral tracking. Lands, and takes a while.
SLOW_W = torch.tensor([[-1.6, 0., -1.2, 0.], [0., 0., 0., -0.55]])
SLOW_B = torch.tensor([0., GRAVITY / GAIN - 0.55 * 0.22])
# Fast law: stronger lateral tracking plus an altitude flare. Still under the
# speed limit, in fewer steps. Used as a data label, not as the student.
FAST_W = torch.tensor([[-2.1, 0., -1.4, 0.], [0., -0.48, 0., -0.80]])
FAST_B = torch.tensor([0., 0.02])
# Crash law: the fast lateral weights with a bias that hits the ground too hard.
CRASH_B = torch.tensor([0., -0.55])

# Thresholds sit inside the gap measured for seed 0 on this plant.
PAIRED_LAND_MIN = 0.90
PAIRED_STEPS_MAX = 18.0
PAIRED_CRASH_MAX = 0.10
PAIRED_DIAG_MAX = 0.02
STEP_GAP_MIN = 12.0
RANK_MARGIN = 0.05
BASELINE_LAND_MIN = 0.98
BASELINE_STEPS_MIN = 30.0
SLOW_ONLY_LAND_MIN = 0.95
SLOW_ONLY_STEPS_MIN = 28.0
SLOW_ONLY_DIAG_MAX = 0.01
CONTROL_LAND_MAX = 0.20
SUPERVISED_LAND_MIN = 0.90

MAPPING = (
    dict(toy="Collect successful slow and fast rollouts from the same starts. "
             "Drop crashes and anything that is not strictly faster. "
             "Rows are (state, slow action, fast action) at states the slow law visits.",
         gym="experiments/collect_slow_fast_lunar.py rolls the #18 controller, "
             "splits successes by steps-to-land, and pairs nearby starts. "
             "Crashes stay out of the fast set. This toy must GATE PASS first."),
    dict(toy="Student starts at the slow law. Controller loss is paired-error "
             "RpGAN, adv_weight 1, target = fast action, scale = std(fast − slow), "
             "sample-point b_cap every 4th update. Diagnostic MSE is not in the loss.",
         gym="experiments/train_gym_slow_fast.py calls controller_objective. "
             "Neutral = slow action, target = fast action. adv_weight stays 1. "
             "safe_fast_weight stays 0. Diagnostic MSE stays outside the loss."),
    dict(toy="Slow-only target, a zero-update baseline, and crash actions in the "
             "fast slot lose the rank key. Unpaired fast actions crash. "
             "MSE-only can land quickly and is rejected because adv_weight is 0.",
         gym="Do not ship adv_weight 0, action MSE in place of the GAN, or a "
             "fast set built from crashes or from unmatched episodes."),
)


def require_live_adversary(adv_weight):
    """Reject a configured GAN that the controller step does not apply."""
    if adv_weight == 0:
        raise ValueError("adv_weight=0 leaves RpGAN and b_cap configured but not applied")
    if adv_weight != 1.:
        raise ValueError("adv_weight stays 1 so the controller step is the adversarial loss")


def pair_is_kept(slow_success, fast_success, slow_steps, fast_steps):
    """Same world, both land, and the fast member is strictly sooner."""
    return bool(slow_success and fast_success and fast_steps < slow_steps)


def _action(state, weight, bias):
    return (state @ weight.T + bias).clamp(-1, 1)


def slow_action(state):
    return _action(state, SLOW_W, SLOW_B)


def fast_action(state):
    """Oracle label for the fast set. Not a student forward."""
    return _action(state, FAST_W, FAST_B)


def crash_action(state):
    """Hard contact. Must not enter the honest fast set."""
    return _action(state, FAST_W, CRASH_B)


def step_plant(state, action):
    """Semi-implicit point-mass step. Column 1 is altitude."""
    x, y, vx, vy = state.unbind(-1)
    ax, ay = action.unbind(-1)
    vx = vx + GAIN * ax
    vy = vy + GAIN * ay - GRAVITY
    x = x + GAIN * vx
    y = y + GAIN * vy
    return torch.stack([x, y, vx, vy], -1)


def initial_states(n, seed, xspan=1.15):
    generator = torch.Generator().manual_seed(int(seed))
    x = (torch.rand(n, generator=generator) * 2 - 1) * xspan
    y = torch.rand(n, generator=generator) * 0.9 + 1.25
    vx = (torch.rand(n, generator=generator) * 2 - 1) * 0.18
    vy = (torch.rand(n, generator=generator) * 2 - 1) * 0.08
    return torch.stack([x, y, vx, vy], -1)


def _episode(policy, state):
    """One rollout. Returns success, steps, and the states visited before contact."""
    visited = []
    current = state.reshape(1, -1)
    for t in range(HORIZON):
        visited.append(current.squeeze(0).detach().clone())
        current = step_plant(current, policy(current))
        x, y, vx, vy = current[0].tolist()
        if y <= 0 or abs(x) > X_LIMIT or y > Y_LIMIT:
            success = y <= 0 and abs(x) <= PAD and abs(vx) <= V_LIMIT and abs(vy) <= V_LIMIT
            return success, t + 1, visited
    return False, HORIZON, visited


def collect_pairs(n_starts=COLLECT_STARTS, seed=4):
    """Successful slow/fast pairs from shared starts. Crashes never join the fast set.

    Training states are the slow trajectory of each kept start. The fast label
    is the fast law at that same state, so the pair is one world flown faster,
    not a different world's action pasted on.
    """
    starts = initial_states(n_starts, seed)
    rows = []
    start_ids = []
    slow_steps = []
    fast_steps = []
    fast_failures = 0
    not_faster = 0
    for index in range(n_starts):
        slow_ok, slow_n, slow_states = _episode(slow_action, starts[index])
        fast_ok, fast_n, _ = _episode(fast_action, starts[index])
        if not fast_ok:
            fast_failures += 1
            continue
        if not pair_is_kept(slow_ok, fast_ok, slow_n, fast_n):
            not_faster += 1
            continue
        slow_steps.append(slow_n)
        fast_steps.append(fast_n)
        for state in slow_states:
            rows.append(state)
            start_ids.append(index)
    if not rows:
        raise RuntimeError("collector kept no slow/fast pairs")
    state = torch.stack(rows)
    start_id = torch.tensor(start_ids)
    neutral = slow_action(state)
    target = fast_action(state)
    crashed = crash_action(state)
    generator = torch.Generator().manual_seed(1)
    perm = torch.randperm(len(state), generator=generator)
    # A one-step nudge stays inside the same trajectory. Walk until the start differs.
    for row in range(len(state)):
        if start_id[perm[row]] != start_id[row]:
            continue
        for shift in range(1, len(state)):
            candidate = int((perm[row] + shift) % len(state))
            if start_id[candidate] != start_id[row]:
                perm[row] = candidate
                break
    unpaired_start = start_id[perm]
    if torch.any(unpaired_start == start_id):
        raise RuntimeError("unpaired rows must come from a different start")
    print(f"[slow-fast] collect kept_starts={len(slow_steps)}/{n_starts} "
          f"rows={len(state)} fast_failures_excluded={fast_failures} "
          f"not_faster={not_faster} "
          f"slow_steps={sum(slow_steps)/len(slow_steps):.2f} "
          f"fast_steps={sum(fast_steps)/len(fast_steps):.2f}", flush=True)
    return dict(state=state, neutral=neutral, target=target, crash_target=crashed,
                unpaired_target=target[perm], start_id=start_id,
                unpaired_start_id=unpaired_start,
                slow_steps=torch.tensor(slow_steps, dtype=torch.float32),
                fast_steps=torch.tensor(fast_steps, dtype=torch.float32),
                fast_failures_excluded=fast_failures, not_faster=not_faster,
                kept_starts=len(slow_steps), rows=len(state))


class SlowPolicy(nn.Module):
    """Linear pad law. Weights start at the slow oracle and are not the fast law."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(SLOW_W.clone())
        self.b = nn.Parameter(SLOW_B.clone())

    def forward(self, state):
        return (state @ self.w.T + self.b).clamp(-1, 1)


class _Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

    def forward(self, coordinates):
        return self.net(coordinates).squeeze(-1)


def _cap():
    return GradientPenalty(arm="b_cap", coeff=1., kappa=1., lazy_k=4, norm="l2",
                           method="autograd", target_anneal="none")


def evaluate_policy(policy, states=None):
    """Landing rate and steps among successes. A crash is not a fast success."""
    if states is None:
        states = initial_states(EVAL_ROWS, 1000)
    state = states.clone()
    landed = torch.zeros(len(state), dtype=torch.bool)
    crashed = torch.zeros(len(state), dtype=torch.bool)
    success_steps = torch.full((len(state),), float(HORIZON))
    contact_steps = torch.full((len(state),), float(HORIZON))
    done = torch.zeros(len(state), dtype=torch.bool)
    with torch.no_grad():
        for t in range(1, HORIZON + 1):
            state = step_plant(state, policy(state))
            x, y, vx, vy = state.unbind(-1)
            contact = (y <= 0) & ~done
            safe = contact & (x.abs() <= PAD) & (vx.abs() <= V_LIMIT) & (vy.abs() <= V_LIMIT)
            oob = (~done) & ~contact & ((x.abs() > X_LIMIT) | (y > Y_LIMIT))
            landed = landed | safe
            crashed = crashed | (contact & ~safe) | oob
            success_steps = torch.where(safe, torch.full_like(success_steps, float(t)), success_steps)
            contact_steps = torch.where(contact, torch.full_like(contact_steps, float(t)),
                                        contact_steps)
            done = done | contact | oob
    landings = float(landed.float().mean())
    crash_rate = float(crashed.float().mean())
    timeout_rate = float((~done).float().mean())
    mean_steps = float(success_steps[landed].mean()) if bool(landed.any()) else float(HORIZON)
    contacted = contact_steps < HORIZON
    mean_contact = float(contact_steps[contacted].mean()) if bool(contacted.any()) else float(HORIZON)
    score = landings - TIME_SCORE * (mean_steps / HORIZON)
    return dict(landings=landings, crash_rate=crash_rate, timeout_rate=timeout_rate,
                mean_steps=mean_steps, mean_contact_steps=mean_contact, score=score)


def success_at_speed(row):
    """Higher is better. Crashes contribute no speed credit: steps are successes only."""
    return float(row["score"])


def rank_key(row):
    """Eligible only when the #18 step actually ran at adv_weight 1.

    MSE and a frozen initialization can look fast on the raw score. They are
    not eligible for the rank key. Crash and unpaired arms stay eligible and
    have to lose on the score itself.
    """
    if row.get("adv_weight") != 1. or row.get("b_cap_applications", 0) <= 0:
        return -1.
    return success_at_speed(row)


def _targets_for(mode, table):
    """Neutral is one member of the pair. Target is what the student is asked to match."""
    if mode == "paired":
        return table["target"], table["neutral"]
    if mode == "slow_only":
        # Same edit scale, target is the slow member. This must not get faster.
        return table["neutral"], table["target"]
    if mode == "unpaired":
        return table["unpaired_target"], table["neutral"]
    if mode == "crash_fast":
        return table["crash_target"], table["neutral"]
    raise ValueError(f"Unknown GAN arm {mode}")


def _gan_step(policy, critic, cap, opt, opt_d, norm, state, target, step, steps, generator):
    """One paired-error update. Returns the GAN grad mass and the detached diagnostic MSE."""
    hold = 1.3 * float(norm.edit_rms)
    index = torch.randint(0, len(state), (BATCH,), generator=generator)
    batch_state, batch_target = state[index], target[index]
    predicted = policy(batch_state)
    residual = (predicted.detach() - batch_target) / norm.target_std
    sigma = noise_std(step - 1, start=norm.noise_start, decay_steps=steps, hold=hold)
    noise = torch.randn(residual.shape, generator=generator) * sigma
    fake = noise + residual
    penalty = cap(critic, noise.detach(), fake.detach(), step=step)
    loss_d = rp_d_loss(critic(noise), critic(fake)) + penalty
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()
    predicted = policy(batch_state)
    residual = (predicted - batch_target) / norm.target_std
    noise = torch.randn(residual.shape, generator=generator) * sigma
    # Controller loss. Diagnostic MSE is computed beside it and is not added.
    loss_g = ADV_WEIGHT * rp_g_loss(critic(noise).detach(), critic(noise + residual))
    with torch.no_grad():
        diagnostic = float(F.mse_loss(predicted.detach(), batch_target))
    opt.zero_grad(set_to_none=True)
    loss_g.backward()
    grad_mass = float(policy.w.grad.detach().abs().sum() + policy.b.grad.detach().abs().sum())
    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    opt.step()
    return grad_mass, diagnostic, int(step % cap.lazy_k == 0)


def train_arm(mode, steps=STEPS, adv_weight=ADV_WEIGHT, table=None, eval_states=None):
    """Train one arm from the slow initialization.

    `paired` matches the fast member. `slow_only` matches the slow member.
    `crash_fast` matches crash actions. `unpaired` matches fast actions from
    other starts. `supervised` is MSE only. `zero` is the slow initialization.
    """
    if mode not in ("paired", "slow_only", "crash_fast", "unpaired", "supervised", "zero"):
        raise ValueError(f"Unknown arm {mode}")
    if mode == "supervised":
        if adv_weight != 0:
            raise ValueError("The supervised ablation is adv_weight 0")
    elif mode != "zero":
        require_live_adversary(adv_weight)
    torch.manual_seed(SEED)
    policy = SlowPolicy()
    if table is None and mode != "zero":
        table = collect_pairs()
    if eval_states is None:
        eval_states = initial_states(EVAL_ROWS, 1000)
    applications = 0
    gan_grad_abs = 0.
    diagnostic = None
    if mode == "zero":
        reason = "no update; slow initialization"
        accepted = False
        reported_adv = None
    elif mode == "supervised":
        opt = torch.optim.Adam(policy.parameters(), lr=POLICY_LR)
        generator = torch.Generator().manual_seed(SEED + 2)
        target = table["target"]
        state = table["state"]
        for step in range(1, steps + 1):
            index = torch.randint(0, len(state), (BATCH,), generator=generator)
            loss = F.mse_loss(policy(state[index]), target[index])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % LOG_EVERY == 0 or step == 1:
                with torch.no_grad():
                    diagnostic = float(F.mse_loss(policy(state), target))
                print(f"[slow-fast] arm=supervised step={step}/{steps} "
                      f"diag_mse={diagnostic:.4f} adv_weight=0", flush=True)
        reason = "adv_weight=0 leaves RpGAN and b_cap configured but not applied"
        accepted = False
        reported_adv = 0.
    else:
        target, neutral = _targets_for(mode, table)
        norm = nn.Module()
        register_paired_error_norm(norm, target.detach(), neutral.detach())
        if norm.normalization != "paired_edit_per_coordinate_std_median_rms_gain":
            raise RuntimeError("edit scale must be std(target − neutral), median RMS gain")
        if float(norm.edit_rms) <= 0:
            raise RuntimeError("paired edit RMS must be positive")
        critic = _Critic()
        cap = _cap()
        opt = torch.optim.Adam(policy.parameters(), lr=POLICY_LR)
        opt_d = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR, betas=(0., 0.999))
        generator = torch.Generator().manual_seed(SEED + 2)
        print(f"[slow-fast] arm={mode} edit_rms={float(norm.edit_rms):.3f} "
              f"noise_start={norm.noise_start:.3f} adv_weight={ADV_WEIGHT}", flush=True)
        for step in range(1, steps + 1):
            grad_mass, diagnostic, applied = _gan_step(
                policy, critic, cap, opt, opt_d, norm, table["state"], target, step, steps, generator)
            gan_grad_abs += grad_mass
            applications += applied
            if step % LOG_EVERY == 0 or step == 1:
                print(f"[slow-fast] arm={mode} step={step}/{steps} diag_mse={diagnostic:.4f} "
                      f"b_cap_applied={applied} gan_grad={grad_mass:.3f}", flush=True)
        with torch.no_grad():
            diagnostic = float(F.mse_loss(policy(table["state"]), target))
        reason = ("paired-error RpGAN weight 1, target is the matched fast member"
                  if mode == "paired" else f"control arm {mode}")
        accepted = mode == "paired"
        reported_adv = 1.
    metrics = evaluate_policy(policy, eval_states)
    metrics.update(arm=mode, adv_weight=reported_adv, accepted=accepted,
                   b_cap_applications=applications, gan_grad_abs=gan_grad_abs,
                   diag_mse=diagnostic, reason=reason,
                   weight_norm=float(policy.w.detach().norm()))
    metrics["rank_key"] = rank_key(metrics)
    print(f"[slow-fast] arm={mode} landings={metrics['landings']:.3f} "
          f"steps={metrics['mean_steps']:.2f} crash={metrics['crash_rate']:.3f} "
          f"contact={metrics['mean_contact_steps']:.2f} score={metrics['score']:.3f} "
          f"rank_key={metrics['rank_key']:.3f} adv_weight={reported_adv} "
          f"b_cap_applications={applications} accepted={accepted}", flush=True)
    return metrics


def _check(failures, name, ok):
    if not ok:
        failures.append(name)
    return ok


def run_gate():
    """Pass only the matched slow→fast RpGAN arm. One seed. CPU only."""
    torch.set_num_threads(1)
    print("[slow-fast] gate start seed=0 adv_weight=1 b_cap_every=4", flush=True)
    table = collect_pairs()
    eval_states = initial_states(EVAL_ROWS, 1000)
    zero = train_arm("zero", table=table, eval_states=eval_states)
    slow_only = train_arm("slow_only", table=table, eval_states=eval_states)
    crash_fast = train_arm("crash_fast", table=table, eval_states=eval_states)
    unpaired = train_arm("unpaired", table=table, eval_states=eval_states)
    supervised = train_arm("supervised", adv_weight=0, table=table, eval_states=eval_states)
    paired = train_arm("paired", table=table, eval_states=eval_states)
    arms = (zero, slow_only, crash_fast, unpaired, supervised, paired)
    eligible = [row for row in arms if row["rank_key"] >= 0]
    winner = max(eligible, key=lambda row: row["rank_key"])["arm"] if eligible else None
    failures = []
    _check(failures, "winner_is_paired", winner == "paired")
    _check(failures, "paired_adv_weight", paired["adv_weight"] == 1. and paired["accepted"] is True)
    _check(failures, "paired_b_cap", paired["b_cap_applications"] == STEPS // 4)
    _check(failures, "paired_gan_grad", paired["gan_grad_abs"] > 0.)
    _check(failures, "paired_landings", paired["landings"] >= PAIRED_LAND_MIN)
    _check(failures, "paired_crash", paired["crash_rate"] <= PAIRED_CRASH_MAX)
    _check(failures, "paired_steps", paired["mean_steps"] <= PAIRED_STEPS_MAX)
    _check(failures, "paired_diag_mse", paired["diag_mse"] is not None and paired["diag_mse"] <= PAIRED_DIAG_MAX)
    _check(failures, "faster_than_zero",
           paired["mean_steps"] <= zero["mean_steps"] - STEP_GAP_MIN)
    _check(failures, "faster_than_slow_only",
           paired["mean_steps"] <= slow_only["mean_steps"] - STEP_GAP_MIN)
    _check(failures, "zero_is_slow_success",
           zero["landings"] >= BASELINE_LAND_MIN and zero["mean_steps"] >= BASELINE_STEPS_MIN
           and zero["b_cap_applications"] == 0 and zero["rank_key"] < 0)
    _check(failures, "slow_only_stays_slow",
           slow_only["landings"] >= SLOW_ONLY_LAND_MIN
           and slow_only["mean_steps"] >= SLOW_ONLY_STEPS_MIN
           and slow_only["diag_mse"] <= SLOW_ONLY_DIAG_MAX
           and slow_only["adv_weight"] == 1. and slow_only["rank_key"] + RANK_MARGIN <= paired["rank_key"])
    _check(failures, "crash_does_not_win",
           crash_fast["landings"] <= CONTROL_LAND_MAX
           and crash_fast["rank_key"] + RANK_MARGIN <= paired["rank_key"]
           and crash_fast["mean_contact_steps"] < zero["mean_steps"])
    _check(failures, "unpaired_does_not_win",
           unpaired["landings"] <= CONTROL_LAND_MAX
           and unpaired["rank_key"] + RANK_MARGIN <= paired["rank_key"])
    _check(failures, "mse_rejected",
           supervised["adv_weight"] == 0 and supervised["accepted"] is False
           and supervised["b_cap_applications"] == 0 and supervised["rank_key"] < 0
           and supervised["landings"] >= SUPERVISED_LAND_MIN)
    _check(failures, "fast_set_excludes_crashes", table["fast_failures_excluded"] == 0
           and bool(torch.all(table["fast_steps"] < table["slow_steps"])))
    passed = not failures
    print(f"[slow-fast] GATE {'PASS' if passed else 'FAIL'} winner={winner} "
          f"failures={failures or 'none'}", flush=True)
    return dict(passed=passed, winner=winner, failures=failures, zero=zero, slow_only=slow_only,
                crash_fast=crash_fast, unpaired=unpaired, supervised=supervised, paired=paired,
                table=dict(kept_starts=table["kept_starts"], rows=table["rows"],
                           fast_failures_excluded=table["fast_failures_excluded"],
                           slow_steps=float(table["slow_steps"].mean()),
                           fast_steps=float(table["fast_steps"].mean())),
                mapping=MAPPING,
                thresholds=dict(paired_land_min=PAIRED_LAND_MIN, paired_steps_max=PAIRED_STEPS_MAX,
                                paired_crash_max=PAIRED_CRASH_MAX, step_gap_min=STEP_GAP_MIN,
                                rank_margin=RANK_MARGIN, baseline_land_min=BASELINE_LAND_MIN,
                                baseline_steps_min=BASELINE_STEPS_MIN,
                                slow_only_steps_min=SLOW_ONLY_STEPS_MIN,
                                control_land_max=CONTROL_LAND_MAX, adv_weight=ADV_WEIGHT,
                                horizon=HORIZON, steps=STEPS))


def _fmt(row):
    adv = "none" if row["adv_weight"] is None else f"{row['adv_weight']}"
    diag = "none" if row["diag_mse"] is None else f"{row['diag_mse']:.4f}"
    return (f"[slow-fast] {row['arm']} landings={row['landings']:.3f} "
            f"steps={row['mean_steps']:.2f} crash={row['crash_rate']:.3f} "
            f"contact={row['mean_contact_steps']:.2f} timeout={row['timeout_rate']:.3f} "
            f"score={row['score']:.3f} rank_key={row['rank_key']:.3f} "
            f"adv_weight={adv} b_cap_applications={row['b_cap_applications']} "
            f"gan_grad_abs={row['gan_grad_abs']:.3f} diag_mse={diag} "
            f"accepted={row['accepted']} reason={row['reason']}")


def format_report(result):
    lines = [f"[slow-fast] GATE {'PASS' if result['passed'] else 'FAIL'} winner={result['winner']}"]
    if result["failures"]:
        lines.append("[slow-fast] failed_checks=" + ",".join(result["failures"]))
    for key in ("zero", "slow_only", "crash_fast", "unpaired", "supervised", "paired"):
        lines.append(_fmt(result[key]))
    limits = result["thresholds"]
    collected = result["table"]
    lines.append(
        "[slow-fast] collect "
        f"kept_starts={collected['kept_starts']} rows={collected['rows']} "
        f"fast_failures_excluded={collected['fast_failures_excluded']} "
        f"slow_steps={collected['slow_steps']:.2f} fast_steps={collected['fast_steps']:.2f}")
    lines.append(
        "[slow-fast] thresholds "
        f"paired_land>={limits['paired_land_min']} paired_steps<={limits['paired_steps_max']} "
        f"paired_crash<={limits['paired_crash_max']} step_gap>={limits['step_gap_min']} "
        f"rank_margin>={limits['rank_margin']} baseline_land>={limits['baseline_land_min']} "
        f"baseline_steps>={limits['baseline_steps_min']} "
        f"slow_only_steps>={limits['slow_only_steps_min']} "
        f"control_land<={limits['control_land_max']} adv_weight={limits['adv_weight']} "
        f"horizon={limits['horizon']} updates={limits['steps']}")
    lines.append("[slow-fast] GATE PASS is required before a Lunar speed claim. "
                 "Commands: docs/gym-slow-fast.md. No Lunar landing is claimed here.")
    lines.append("[slow-fast] mapping")
    for row in result["mapping"]:
        lines.append(f"[slow-fast] toy: {row['toy']}")
        lines.append(f"[slow-fast] gym: {row['gym']}")
    return "\n".join(lines)


def board_markdown(result):
    """Leaderboard for reports/slow_fast_paired. Numbers come from this run."""
    def cell(row, key, spec):
        value = row[key]
        if value is None:
            return "—"
        if spec == "s":
            return str(value)
        if spec == "d":
            return str(int(value))
        return format(value, spec)

    header = ("| Arm | Landings | Steps | Crash | Contact | Score | Rank key | adv | "
              "b_cap | diag MSE | Accepted |")
    split = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    body = []
    for key in ("zero", "slow_only", "crash_fast", "unpaired", "supervised", "paired"):
        row = result[key]
        body.append(
            "| {arm} | {landings} | {steps} | {crash} | {contact} | {score} | {rank} | "
            "{adv} | {bcap} | {diag} | {accepted} |".format(
                arm=row["arm"],
                landings=cell(row, "landings", ".3f"),
                steps=cell(row, "mean_steps", ".2f"),
                crash=cell(row, "crash_rate", ".3f"),
                contact=cell(row, "mean_contact_steps", ".2f"),
                score=cell(row, "score", ".3f"),
                rank=cell(row, "rank_key", ".3f"),
                adv=cell(row, "adv_weight", "s"),
                bcap=cell(row, "b_cap_applications", "d"),
                diag=cell(row, "diag_mse", ".4f"),
                accepted=row["accepted"]))
    collected = result["table"]
    status = "PASS" if result["passed"] else "FAIL"
    return "\n".join((
        "# Slow→fast paired finetune (CPU gate)",
        "",
        f"Gate **{status}**. Winner of the rank key: `{result['winner']}`.",
        "This gate must **PASS** before a Lunar speed claim. "
        "Lunar commands are in `docs/gym-slow-fast.md`.",
        "These numbers are a 2D pad. They are not Lunar landings.",
        "",
        "One seed (`0`), fixed eval starts, no seed sweep. Rank key is",
        "`landings − 0.5 × (mean steps among successes / horizon)` and is",
        "eligible only when `adv_weight=1` and sample-point `b_cap` ran.",
        "Steps count successful pad contacts only. `contact` is any ground hit,",
        "so a crash can look fast there and still lose the rank key.",
        "",
        header,
        split,
        *body,
        "",
        f"Collector: {collected['kept_starts']} matched starts, {collected['rows']} rows, "
        f"fast failures excluded {collected['fast_failures_excluded']}. "
        f"Mean slow steps {collected['slow_steps']:.2f}, mean fast steps {collected['fast_steps']:.2f}.",
        "",
        "## Why the controls lose",
        "",
        "- `zero` is the slow initialization. No GAN step, so the rank key is ineligible. "
        "It lands, and it is slower than the paired arm.",
        "- `slow_only` is the same RpGAN step with the slow member as the target. "
        "It stays a successful slow landing and does not take the rank key.",
        "- `crash_fast` puts crash actions in the fast slot. Contact can be sooner. "
        "Success collapses, so it does not win.",
        "- `unpaired` uses fast actions from other starts. The student leaves the pad "
        "or hits too hard. That is not the same world flown faster.",
        "- `supervised` matches the fast member with MSE and `adv_weight=0`. "
        "Landings may be excellent. The rank key rejects it because the #18 step did not run.",
        "",
        "## Lunar path",
        "",
        "Keep this gate green. The gym collector, trainer, and shared-seed eval are "
        "documented in `docs/gym-slow-fast.md`. They use the same paired-error step "
        "(`adv_weight=1`, `b_cap` every fourth update, diagnostic MSE outside the loss) "
        "and do not use the safe-fast kinematic cost. This board is not a Lunar result.",
        "",
        "```bash",
        "python -u examples/slow_fast_paired_2d.py",
        "```",
        "",
    ))
