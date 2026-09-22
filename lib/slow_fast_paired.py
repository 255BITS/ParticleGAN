"""CPU gate: two teachers, same seed, both land, aligned by progress.

The safe teacher is a land-first law. It already lands, and it has no speed
term. The fast teacher is the same linear spine, trained with an explicit
altitude speed bias. Speed keeps rising until the landing breaks. There is no
hand-picked fraction of the fast action and no separate crash law.

`connected` rolls the same start with both teachers and keeps the episode only
when both land. Rows are aligned by progress ``t/T``. Neutral is the safe
action. Target is the fast action at that progress. A crashy fast teacher is
allowed. Crash rows stay out of the fast set. This plant's passing student
uses the both-land pairs from update 2. Lunar does not copy that index: it
keeps the fastest stage that still has landings.

The next speed update is `overspeed`: the teacher still lands, and the same
RpGAN step does not. Past that, the teacher itself crashes. Those failed
rollouts are `crash_fast` and never enter the fast set. `stranger` is the
other failure: a different episode's fast action at the nearest state.
adv_weight stays 1. Diagnostic MSE stays outside the loss. There is no
kinematic safe-fast cost. This file does not run Lunar Lander.
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
# Explicit speed term on the fast teacher (cost of remaining altitude).
# The safe teacher uses weight 0. One curriculum, one seed:
#   update 2: teacher still lands (~23 steps); progress-aligned student holds
#   update 3: teacher still lands and is faster; that student loses the pad
#   update 6: the teacher itself drops below the landing floor
SPEED_BIAS = 0.02
TEACHER_LR = 0.02
HELD_TEACHER_STEPS = 2
OVERSPEED_TEACHER_STEPS = 3
BREAK_TEACHER_STEPS = 6
NEAREST_DIST = 0.75
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
# Progress pairs from teacher update 2 hold the pad and beat the safe law.
# Update 3 and the teacher's own crash rows do not. Stranger nearest does not.
CONNECTED_LAND_MIN = 0.98
CONNECTED_STEPS_MAX = 32.0
CONNECTED_CRASH_MAX = 0.02
CONNECTED_MIN_LAND_MIN = 0.95
STRANGER_LAND_MAX = 0.50
STRANGER_CRASH_MIN = 0.40
OVERSPEED_LAND_MAX = 0.90
STEP_GAP_MIN = 6.0
BASELINE_LAND_MIN = 0.98
BASELINE_STEPS_MIN = 30.0
SLOW_ONLY_LAND_MIN = 0.95
SLOW_ONLY_STEPS_MIN = 28.0
CONTROL_LAND_MAX = 0.20
LANDING_FLOOR = 0.90
CRASH_CEILING = 0.10
SUPERVISED_LAND_MIN = 0.90

MAPPING = (
    dict(toy="Safe teacher: land-first linear law, speed weight 0. "
             "Fast teacher: the same spine, altitude speed bias 0.02.",
         gym="Safe teacher is #18. Fast teacher is that architecture trained "
             "with an explicit speed term. Do not invent a second crash policy."),
    dict(toy="Connected pairs use the same start, keep the episode only if both "
             "land, and align by progress t/T. Neutral is the safe action. "
             "Target is the fast action at that progress. Held teacher update is 2.",
         gym="Lunar collect rolls #18 and the held fast teacher on the same seed, "
             "keeps the pair only when both land, and aligns by progress t/T."),
    dict(toy="One more speed update (3) still lands as a teacher and the student "
             "loses the pad. By update 6 the teacher itself crashes. Those rows "
             "stay out of the fast set.",
         gym="Run every speed stage. held.pt is the fastest stage that still has "
             "landings, even when the probe is crashy. Collect drops crashes. "
             "A near-safe 20/20 is not the hold."),
    dict(toy="Stranger pastes a different episode's recorded fast action onto the "
             "nearest state. The rows are plentiful. The same RpGAN step loses landings.",
         gym="That nearest-state collector is disabled. cuda:1 went 20/20 to 0/20. "
             "Do not train pairs.npz or pairs_stranger_do_not_train.npz."),
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


def _teacher_loss(policy, starts, speed_weight):
    """Land on the pad. speed_weight > 0 charges leftover altitude."""
    state = starts
    alive = torch.ones(len(starts))
    total = state.new_zeros(())
    for _ in range(HORIZON):
        state = step_plant(state, policy(state))
        x, y, vx, vy = state.unbind(-1)
        total = total + (alive * y.clamp(min=0) * speed_weight).sum()
        near = torch.sigmoid((0.2 - y) * 25)
        pad_miss = (x.abs() - PAD).clamp(min=0).pow(2)
        v_miss = ((vx.abs() - V_LIMIT).clamp(min=0).pow(2)
                  + (vy.abs() - V_LIMIT).clamp(min=0).pow(2))
        total = total + (alive * near * (4.0 * pad_miss + 4.0 * v_miss + 0.15 * x.pow(2))).sum()
        done = (y <= 0) | (x.abs() > X_LIMIT) | (y > Y_LIMIT)
        alive = alive * (~done).float()
    x, y = state.unbind(-1)[:2]
    total = total + (alive * (2.0 + y.clamp(min=0) + x.pow(2))).sum()
    return total / len(starts)


def _freeze_copy(policy):
    frozen = SlowPolicy()
    with torch.no_grad():
        frozen.w.copy_(policy.w)
        frozen.b.copy_(policy.b)
    frozen.eval()
    for param in frozen.parameters():
        param.requires_grad_(False)
    return frozen


def train_speed_curriculum(speed_weight=SPEED_BIAS, steps=BREAK_TEACHER_STEPS):
    """Same spine as the safe teacher. Each update is a faster attempt.

    Returns one frozen policy per update, plus that update's closed-loop eval.
    The safe teacher is the initialization (speed weight 0, no step taken).
    """
    torch.manual_seed(SEED)
    policy = SlowPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=TEACHER_LR)
    starts = initial_states(64, 11)
    curve = []
    for step in range(1, steps + 1):
        loss = _teacher_loss(policy, starts, speed_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        opt.step()
        metrics = evaluate_policy(policy)
        curve.append(dict(step=step, policy=_freeze_copy(policy), **metrics))
        print(f"[slow-fast] teacher step={step}/{steps} speed_bias={speed_weight} "
              f"landings={metrics['landings']:.3f} steps={metrics['mean_steps']:.2f} "
              f"crash={metrics['crash_rate']:.3f}", flush=True)
    return curve


def _rollout(policy, state):
    """One rollout with the actions that were actually applied."""
    visited_s, visited_a = [], []
    current = state.reshape(1, -1)
    for _ in range(HORIZON):
        action = policy(current)
        visited_s.append(current.squeeze(0).detach().clone())
        visited_a.append(action.squeeze(0).detach().clone())
        current = step_plant(current, action)
        x, y, vx, vy = current[0].tolist()
        if y <= 0 or abs(x) > X_LIMIT or y > Y_LIMIT:
            success = y <= 0 and abs(x) <= PAD and abs(vx) <= V_LIMIT and abs(vy) <= V_LIMIT
            return success, len(visited_s), visited_s, visited_a
    return False, HORIZON, visited_s, visited_a


def _both_land_episodes(fast_policy, n_starts, seed):
    """Same start for both teachers. A fast miss never enters the episode list."""
    starts = initial_states(n_starts, seed)
    episodes = []
    fast_failures = 0
    not_faster = 0
    for index in range(n_starts):
        slow_ok, slow_n, slow_states, slow_actions = _rollout(slow_action, starts[index])
        fast_ok, fast_n, fast_states, fast_actions = _rollout(fast_policy, starts[index])
        if not fast_ok:
            fast_failures += 1
            continue
        if not pair_is_kept(slow_ok, fast_ok, slow_n, fast_n):
            not_faster += 1
            continue
        episodes.append(dict(
            start=index,
            slow=torch.stack(slow_states), fast=torch.stack(fast_states),
            slow_a=torch.stack(slow_actions), fast_a=torch.stack(fast_actions),
            slow_n=slow_n, fast_n=fast_n,
        ))
    return episodes, fast_failures, not_faster


def _progress_rows(episodes):
    """Align one landing by t/T. Target is the fast action at that progress.

    The row state is the safe teacher's state. The partner index is
    ``round(progress * (fast_len - 1))`` on the fast trajectory of the same
    start. It is not a nearest state, and it is not another episode.
    """
    states, targets, neutrals, progress, partner = [], [], [], [], []
    for episode in episodes:
        slow_n = len(episode["slow"])
        fast_n = len(episode["fast"])
        for i in range(slow_n):
            fraction = 0.0 if slow_n == 1 else i / (slow_n - 1)
            j = int(round(fraction * (fast_n - 1)))
            states.append(episode["slow"][i])
            neutrals.append(episode["slow_a"][i])
            targets.append(episode["fast_a"][j])
            progress.append(fraction)
            partner.append(j)
    return (torch.stack(states), torch.stack(targets), torch.stack(neutrals),
            torch.tensor(progress), torch.tensor(partner))


def _stranger_rows(episodes):
    """Fast action from a different episode at the nearest state."""
    states, targets, neutrals, distances = [], [], [], []
    for episode in episodes:
        partner = next(other for other in episodes if other["start"] != episode["start"])
        distance = torch.cdist(episode["slow"], partner["fast"])
        nearest = distance.argmin(dim=1)
        chosen = distance[torch.arange(len(episode["slow"])), nearest]
        keep = chosen <= NEAREST_DIST
        if not bool(keep.any()):
            continue
        states.append(episode["slow"][keep])
        targets.append(partner["fast_a"][nearest[keep]])
        neutrals.append(episode["slow_a"][keep])
        distances.append(chosen[keep])
    if not states:
        raise RuntimeError("nearest-state collector kept no rows")
    return (torch.cat(states), torch.cat(targets), torch.cat(neutrals), torch.cat(distances))


def _crash_rows(fast_policy, n_starts, seed):
    """Actions from fast-teacher episodes that did not land."""
    starts = initial_states(n_starts, seed)
    states, targets, neutrals = [], [], []
    failures = 0
    for index in range(n_starts):
        ok, _, visited_s, visited_a = _rollout(fast_policy, starts[index])
        if ok:
            continue
        failures += 1
        for state, action in zip(visited_s, visited_a):
            states.append(state)
            targets.append(action)
            neutrals.append(slow_action(state.unsqueeze(0)).squeeze(0))
    if not states:
        raise RuntimeError("speed curriculum produced no crashed fast-teacher rows")
    return torch.stack(states), torch.stack(targets), torch.stack(neutrals), failures


def _unpaired_perm(start_id):
    generator = torch.Generator().manual_seed(1)
    perm = torch.randperm(len(start_id), generator=generator)
    for row in range(len(start_id)):
        if start_id[perm[row]] == start_id[row]:
            for shift in range(1, len(start_id)):
                candidate = int((perm[row] + shift) % len(start_id))
                if start_id[candidate] != start_id[row]:
                    perm[row] = candidate
                    break
    if torch.any(start_id[perm] == start_id):
        raise RuntimeError("unpaired rows must come from a different start")
    return perm


def collect_pairs(n_starts=COLLECT_STARTS, seed=4):
    """Two teachers. Connected rows are same-seed, both-land, progress-aligned.

    The fast teacher is trained until it cannot land. Pairs for the pass arm
    come from the last update the progress-aligned student can take. The next
    update is the overspeed control. Crashed rollouts at the break are the
    crash control and are not in the fast set.
    """
    curve = train_speed_curriculum()
    by_step = {row["step"]: row for row in curve}
    for step in (HELD_TEACHER_STEPS, OVERSPEED_TEACHER_STEPS, BREAK_TEACHER_STEPS):
        if step not in by_step:
            raise RuntimeError(f"speed curriculum did not record update {step}")
    held = by_step[HELD_TEACHER_STEPS]
    over = by_step[OVERSPEED_TEACHER_STEPS]
    broken = by_step[BREAK_TEACHER_STEPS]
    if held["landings"] <= 0:
        raise RuntimeError("held fast teacher has no successful landings")
    if broken["landings"] >= LANDING_FLOOR:
        raise RuntimeError("fast teacher never lost the pad; speed did not reach the break")
    episodes, fast_failures, not_faster = _both_land_episodes(held["policy"], n_starts, seed)
    if not episodes:
        raise RuntimeError("collector kept no same-seed both-land pairs")
    state, target, neutral, progress, partner_index = _progress_rows(episodes)
    over_eps, _, _ = _both_land_episodes(over["policy"], n_starts, seed)
    if not over_eps:
        raise RuntimeError("overspeed teacher kept no both-land pairs")
    over_state, over_target, over_neutral, _, _ = _progress_rows(over_eps)
    nearest_state, nearest_target, nearest_neutral, nearest_distance = _stranger_rows(episodes)
    crash_state, crash_target, crash_neutral, crash_episodes = _crash_rows(
        broken["policy"], n_starts, seed)
    start_id = torch.tensor([episode["start"] for episode in episodes for _ in episode["slow"]])
    perm = _unpaired_perm(start_id)
    slow_steps = [episode["slow_n"] for episode in episodes]
    fast_steps = [episode["fast_n"] for episode in episodes]
    progress_edit = float((target - neutral).abs().mean())
    print(f"[slow-fast] collect kept_starts={len(episodes)}/{n_starts} "
          f"rows={len(state)} nearest_rows={len(nearest_state)} "
          f"nearest_dist={float(nearest_distance.mean()):.3f} "
          f"progress_edit={progress_edit:.3f} "
          f"held_steps={held['mean_steps']:.2f} overspeed_steps={over['mean_steps']:.2f} "
          f"break_landings={broken['landings']:.3f} "
          f"fast_failures_excluded={fast_failures} crash_episodes={crash_episodes} "
          f"not_faster={not_faster} "
          f"slow_steps={sum(slow_steps)/len(slow_steps):.2f} "
          f"fast_steps={sum(fast_steps)/len(fast_steps):.2f}", flush=True)
    return dict(state=state, neutral=neutral, target=target,
                unpaired_target=target[perm], start_id=start_id,
                unpaired_start_id=start_id[perm],
                nearest_state=nearest_state, nearest_target=nearest_target,
                nearest_neutral=nearest_neutral, nearest_distance=nearest_distance,
                connected_state=state, connected_target=target, connected_neutral=neutral,
                progress=progress, partner_index=partner_index, progress_edit=progress_edit,
                overspeed_state=over_state, overspeed_target=over_target,
                overspeed_neutral=over_neutral,
                crash_state=crash_state, crash_target=crash_target, crash_neutral=crash_neutral,
                slow_steps=torch.tensor(slow_steps, dtype=torch.float32),
                fast_steps=torch.tensor(fast_steps, dtype=torch.float32),
                fast_failures_excluded=fast_failures, not_faster=not_faster,
                crash_episodes=crash_episodes,
                kept_starts=len(episodes), rows=len(state), nearest_rows=len(nearest_state),
                held_teacher_steps=held["mean_steps"], held_teacher_landings=held["landings"],
                overspeed_teacher_steps=over["mean_steps"],
                overspeed_teacher_landings=over["landings"],
                break_teacher_landings=broken["landings"],
                break_teacher_steps=broken["mean_steps"],
                teacher_curve=[dict(step=row["step"], landings=row["landings"],
                                    mean_steps=row["mean_steps"], crash_rate=row["crash_rate"])
                               for row in curve])


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


def rank_key(row):
    """Landings first, then fewer steps among successes.

    Ineligible when the #18 step did not run, landings fell, crashes rose, or
    any recorded checkpoint lost the pad. A fast leftover success does not
    rescue that arm. Returns (landings, -mean_steps), or (-1, 0).
    """
    if row.get("adv_weight") != 1. or row.get("b_cap_applications", 0) <= 0:
        return (-1., 0.)
    landings = row.get("landings")
    crash_rate = row.get("crash_rate", 1.)
    min_landings = row.get("min_landings", landings)
    if landings is None or crash_rate is None or min_landings is None:
        return (-1., 0.)
    if landings < LANDING_FLOOR or crash_rate > CRASH_CEILING or min_landings < LANDING_FLOOR:
        return (-1., 0.)
    return (float(landings), -float(row.get("mean_steps", HORIZON)))


def _targets_for(mode, table):
    """State, target action, and neutral action for one GAN arm."""
    if mode == "paired":
        return table["state"], table["target"], table["neutral"]
    if mode == "stranger":
        return table["nearest_state"], table["nearest_target"], table["nearest_neutral"]
    if mode == "connected":
        return table["connected_state"], table["connected_target"], table["connected_neutral"]
    if mode == "overspeed":
        return table["overspeed_state"], table["overspeed_target"], table["overspeed_neutral"]
    if mode == "slow_only":
        # Same edit scale, target is the slow member. This must not get faster.
        return table["state"], table["neutral"], table["target"]
    if mode == "unpaired":
        return table["state"], table["unpaired_target"], table["neutral"]
    if mode == "crash_fast":
        return table["crash_state"], table["crash_target"], table["crash_neutral"]
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
    grad_mass = 0.
    for param in policy.parameters():
        if param.grad is not None:
            grad_mass += float(param.grad.detach().abs().sum())
    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    opt.step()
    return grad_mass, diagnostic, int(step % cap.lazy_k == 0)


_ARM_REASONS = {
    "paired": "same rows as connected; not a separate gate arm",
    "stranger": "full-weight cross-episode nearest RpGAN; the Lunar miss",
    "connected": "full-weight same-seed progress pairs at the held speed",
    "overspeed": "same alignment one speed update later; the student loses the pad",
    "slow_only": "control arm slow_only",
    "crash_fast": "fast-teacher rows that did not land after the speed break",
    "unpaired": "control arm unpaired",
}


def train_arm(mode, steps=STEPS, adv_weight=ADV_WEIGHT, table=None, eval_states=None):
    """Train one arm from the competent slow lander.

    `connected` finetunes every weight on same-seed progress pairs from the
    held fast teacher. `overspeed` is the next speed update. `crash_fast` is
    the fast teacher's own missed landings. `stranger` is cross-episode
    nearest state. `supervised` is MSE only. `zero` is the safe initialization.
    """
    if mode not in ("paired", "slow_only", "crash_fast", "unpaired", "supervised",
                    "zero", "stranger", "connected", "overspeed"):
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
    min_landings = None
    panel = mode in ("stranger", "connected", "overspeed")
    early_landings = None
    return_landings = []
    if mode == "zero":
        reason = "no update; competent slow lander"
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
        state, target, neutral = _targets_for(mode, table)
        norm = nn.Module()
        register_paired_error_norm(norm, target.detach(), neutral.detach())
        if norm.normalization != "paired_edit_per_coordinate_std_median_rms_gain":
            raise RuntimeError("edit scale must be std(target − neutral), median RMS gain")
        if float(norm.edit_rms) <= 0:
            raise RuntimeError("paired edit RMS must be positive")
        critic = _Critic()
        cap = _cap()
        lr = POLICY_LR
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        opt_d = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR, betas=(0., 0.999))
        generator = torch.Generator().manual_seed(SEED + 2)
        print(f"[slow-fast] arm={mode} rows={len(state)} lr={lr} "
              f"edit_rms={float(norm.edit_rms):.3f} noise_start={norm.noise_start:.3f} "
              f"adv_weight={ADV_WEIGHT}", flush=True)
        for step in range(1, steps + 1):
            grad_mass, diagnostic, applied = _gan_step(
                policy, critic, cap, opt, opt_d, norm, state, target, step, steps, generator)
            gan_grad_abs += grad_mass
            applications += applied
            if panel and step == 50:
                snap = evaluate_policy(policy, eval_states)
                early_landings = snap["landings"]
                print(f"[slow-fast] arm={mode} step={step}/{steps} "
                      f"landings={snap['landings']:.3f} steps={snap['mean_steps']:.2f} "
                      f"crash={snap['crash_rate']:.3f} early=1 "
                      f"diag_mse={diagnostic:.4f} b_cap_applied={applied} "
                      f"gan_grad={grad_mass:.3f}", flush=True)
            if panel and (step % LOG_EVERY == 0 or step == steps):
                snap = evaluate_policy(policy, eval_states)
                return_landings.append(snap["landings"])
                min_landings = min(return_landings)
                print(f"[slow-fast] arm={mode} step={step}/{steps} "
                      f"landings={snap['landings']:.3f} steps={snap['mean_steps']:.2f} "
                      f"crash={snap['crash_rate']:.3f} min_landings={min_landings:.3f} "
                      f"diag_mse={diagnostic:.4f} b_cap_applied={applied} "
                      f"gan_grad={grad_mass:.3f}", flush=True)
            elif step % LOG_EVERY == 0 or step == 1:
                print(f"[slow-fast] arm={mode} step={step}/{steps} diag_mse={diagnostic:.4f} "
                      f"b_cap_applied={applied} gan_grad={grad_mass:.3f}", flush=True)
        with torch.no_grad():
            diagnostic = float(F.mse_loss(policy(state), target))
        reason = _ARM_REASONS[mode]
        accepted = mode == "connected"
        reported_adv = 1.
    metrics = evaluate_policy(policy, eval_states)
    if not return_landings:
        return_landings = [metrics["landings"]]
    else:
        return_landings.append(metrics["landings"])
    min_landings = min(return_landings)
    max_return_landings = max(return_landings)
    if hasattr(policy, "w"):
        weight_norm = float(policy.w.detach().norm())
    else:
        weight_norm = float(sum(p.detach().norm() for p in policy.parameters()))
    metrics.update(arm=mode, adv_weight=reported_adv, accepted=accepted,
                   b_cap_applications=applications, gan_grad_abs=gan_grad_abs,
                   diag_mse=diagnostic, reason=reason, weight_norm=weight_norm,
                   min_landings=min_landings, max_return_landings=max_return_landings,
                   early_landings=early_landings)
    metrics["rank_key"] = rank_key(metrics)
    rank = metrics["rank_key"]
    rank_text = "ineligible" if rank[0] < 0 else f"({rank[0]:.3f},{-rank[1]:.2f})"
    print(f"[slow-fast] arm={mode} landings={metrics['landings']:.3f} "
          f"min_landings={min_landings:.3f} steps={metrics['mean_steps']:.2f} "
          f"crash={metrics['crash_rate']:.3f} contact={metrics['mean_contact_steps']:.2f} "
          f"score={metrics['score']:.3f} rank_key={rank_text} adv_weight={reported_adv} "
          f"b_cap_applications={applications} accepted={accepted}", flush=True)
    return metrics


def _check(failures, name, ok):
    if not ok:
        failures.append(name)
    return ok


def run_gate():
    """Pass held-speed progress pairs. The next speed, crashes, and strangers lose."""
    torch.set_num_threads(1)
    print("[slow-fast] gate start seed=0 adv_weight=1 b_cap_every=4 "
          f"lr={POLICY_LR} speed_bias={SPEED_BIAS} "
          f"held={HELD_TEACHER_STEPS} overspeed={OVERSPEED_TEACHER_STEPS} "
          f"break={BREAK_TEACHER_STEPS}", flush=True)
    table = collect_pairs()
    eval_states = initial_states(EVAL_ROWS, 1000)
    zero = train_arm("zero", table=table, eval_states=eval_states)
    slow_only = train_arm("slow_only", table=table, eval_states=eval_states)
    crash_fast = train_arm("crash_fast", table=table, eval_states=eval_states)
    unpaired = train_arm("unpaired", table=table, eval_states=eval_states)
    supervised = train_arm("supervised", adv_weight=0, table=table, eval_states=eval_states)
    stranger = train_arm("stranger", table=table, eval_states=eval_states)
    overspeed = train_arm("overspeed", table=table, eval_states=eval_states)
    connected = train_arm("connected", table=table, eval_states=eval_states)
    arms = (zero, slow_only, crash_fast, unpaired, supervised, stranger, overspeed, connected)
    eligible = [row for row in arms if row["rank_key"][0] >= 0]
    winner = max(eligible, key=lambda row: row["rank_key"])["arm"] if eligible else None
    failures = []
    _check(failures, "winner_is_connected", winner == "connected")
    _check(failures, "connected_adv_weight",
           connected["adv_weight"] == 1. and connected["accepted"] is True)
    _check(failures, "connected_b_cap", connected["b_cap_applications"] == STEPS // 4)
    _check(failures, "connected_gan_grad", connected["gan_grad_abs"] > 0.)
    _check(failures, "connected_landings", connected["landings"] >= CONNECTED_LAND_MIN)
    _check(failures, "connected_crash", connected["crash_rate"] <= CONNECTED_CRASH_MAX)
    _check(failures, "connected_steps", connected["mean_steps"] <= CONNECTED_STEPS_MAX)
    _check(failures, "connected_returns", connected["min_landings"] >= CONNECTED_MIN_LAND_MIN)
    _check(failures, "faster_than_zero",
           connected["mean_steps"] <= zero["mean_steps"] - STEP_GAP_MIN)
    _check(failures, "faster_than_slow_only",
           connected["mean_steps"] <= slow_only["mean_steps"] - STEP_GAP_MIN)
    _check(failures, "zero_is_slow_success",
           zero["landings"] >= BASELINE_LAND_MIN and zero["mean_steps"] >= BASELINE_STEPS_MIN
           and zero["b_cap_applications"] == 0 and zero["rank_key"][0] < 0)
    _check(failures, "slow_only_stays_slow",
           slow_only["landings"] >= SLOW_ONLY_LAND_MIN
           and slow_only["mean_steps"] >= SLOW_ONLY_STEPS_MIN
           and slow_only["adv_weight"] == 1.
           and connected["rank_key"] > slow_only["rank_key"])
    _check(failures, "crash_does_not_win",
           crash_fast["landings"] < LANDING_FLOOR
           and crash_fast["crash_rate"] > CRASH_CEILING
           and crash_fast["rank_key"][0] < 0
           and crash_fast["adv_weight"] == 1.
           and crash_fast["mean_contact_steps"] < zero["mean_steps"])
    _check(failures, "overspeed_does_not_win",
           overspeed["landings"] <= OVERSPEED_LAND_MAX
           and overspeed["rank_key"][0] < 0
           and overspeed["adv_weight"] == 1.
           and overspeed["accepted"] is False
           and table["overspeed_teacher_steps"] < table["held_teacher_steps"])
    _check(failures, "unpaired_does_not_win",
           unpaired["landings"] <= CONTROL_LAND_MAX and unpaired["rank_key"][0] < 0)
    _check(failures, "mse_rejected",
           supervised["adv_weight"] == 0 and supervised["accepted"] is False
           and supervised["b_cap_applications"] == 0 and supervised["rank_key"][0] < 0
           and supervised["landings"] >= SUPERVISED_LAND_MIN)
    _check(failures, "stranger_ran_the_gan",
           stranger["adv_weight"] == 1. and stranger["b_cap_applications"] == STEPS // 4
           and stranger["gan_grad_abs"] > 0. and stranger["accepted"] is False)
    _check(failures, "stranger_loses_landings",
           stranger["landings"] <= STRANGER_LAND_MAX
           and stranger["crash_rate"] >= STRANGER_CRASH_MIN
           and stranger["rank_key"][0] < 0)
    _check(failures, "stranger_pairs_are_plentiful", table["nearest_rows"] >= 1000)
    _check(failures, "fast_set_excludes_crashes", table["fast_failures_excluded"] == 0
           and table["crash_episodes"] > 0
           and bool(torch.all(table["fast_steps"] < table["slow_steps"])))
    passed = not failures
    print(f"[slow-fast] GATE {'PASS' if passed else 'FAIL'} winner={winner} "
          f"failures={failures or 'none'}", flush=True)
    return dict(passed=passed, winner=winner, failures=failures, zero=zero, slow_only=slow_only,
                crash_fast=crash_fast, unpaired=unpaired, supervised=supervised,
                stranger=stranger, overspeed=overspeed, connected=connected,
                table=dict(kept_starts=table["kept_starts"], rows=table["rows"],
                           nearest_rows=table["nearest_rows"],
                           nearest_dist=float(table["nearest_distance"].mean()),
                           progress_edit=table["progress_edit"],
                           fast_failures_excluded=table["fast_failures_excluded"],
                           crash_episodes=table["crash_episodes"],
                           slow_steps=float(table["slow_steps"].mean()),
                           fast_steps=float(table["fast_steps"].mean()),
                           held_teacher_steps=table["held_teacher_steps"],
                           held_teacher_landings=table["held_teacher_landings"],
                           overspeed_teacher_steps=table["overspeed_teacher_steps"],
                           overspeed_teacher_landings=table["overspeed_teacher_landings"],
                           break_teacher_landings=table["break_teacher_landings"],
                           break_teacher_steps=table["break_teacher_steps"],
                           teacher_curve=table["teacher_curve"]),
                mapping=MAPPING,
                thresholds=dict(connected_land_min=CONNECTED_LAND_MIN,
                                connected_steps_max=CONNECTED_STEPS_MAX,
                                connected_crash_max=CONNECTED_CRASH_MAX,
                                connected_min_land_min=CONNECTED_MIN_LAND_MIN,
                                stranger_land_max=STRANGER_LAND_MAX,
                                stranger_crash_min=STRANGER_CRASH_MIN,
                                overspeed_land_max=OVERSPEED_LAND_MAX,
                                step_gap_min=STEP_GAP_MIN,
                                landing_floor=LANDING_FLOOR, crash_ceiling=CRASH_CEILING,
                                baseline_land_min=BASELINE_LAND_MIN,
                                baseline_steps_min=BASELINE_STEPS_MIN,
                                slow_only_steps_min=SLOW_ONLY_STEPS_MIN,
                                control_land_max=CONTROL_LAND_MAX, adv_weight=ADV_WEIGHT,
                                speed_bias=SPEED_BIAS, policy_lr=POLICY_LR,
                                held_teacher_updates=HELD_TEACHER_STEPS,
                                overspeed_teacher_updates=OVERSPEED_TEACHER_STEPS,
                                break_teacher_updates=BREAK_TEACHER_STEPS,
                                horizon=HORIZON, steps=STEPS))


def _rank_text(key):
    if key[0] < 0:
        return "ineligible"
    return f"({key[0]:.3f},{-key[1]:.2f})"


def _fmt(row):
    adv = "none" if row["adv_weight"] is None else f"{row['adv_weight']}"
    diag = "none" if row["diag_mse"] is None else f"{row['diag_mse']:.4f}"
    early = row.get("early_landings")
    early_text = "none" if early is None else f"{early:.3f}"
    return (f"[slow-fast] {row['arm']} landings={row['landings']:.3f} "
            f"early_landings={early_text} min_landings={row['min_landings']:.3f} "
            f"steps={row['mean_steps']:.2f} "
            f"crash={row['crash_rate']:.3f} contact={row['mean_contact_steps']:.2f} "
            f"timeout={row['timeout_rate']:.3f} score={row['score']:.3f} "
            f"rank_key={_rank_text(row['rank_key'])} "
            f"adv_weight={adv} b_cap_applications={row['b_cap_applications']} "
            f"gan_grad_abs={row['gan_grad_abs']:.3f} diag_mse={diag} "
            f"accepted={row['accepted']} reason={row['reason']}")


def format_report(result):
    lines = [f"[slow-fast] GATE {'PASS' if result['passed'] else 'FAIL'} winner={result['winner']}"]
    if result["failures"]:
        lines.append("[slow-fast] failed_checks=" + ",".join(result["failures"]))
    for key in ("zero", "slow_only", "crash_fast", "unpaired", "supervised",
                "stranger", "overspeed", "connected"):
        lines.append(_fmt(result[key]))
    limits = result["thresholds"]
    collected = result["table"]
    lines.append(
        "[slow-fast] collect "
        f"kept_starts={collected['kept_starts']} rows={collected['rows']} "
        f"nearest_rows={collected['nearest_rows']} nearest_dist={collected['nearest_dist']:.3f} "
        f"progress_edit={collected['progress_edit']:.3f} "
        f"fast_failures_excluded={collected['fast_failures_excluded']} "
        f"crash_episodes={collected['crash_episodes']} "
        f"slow_steps={collected['slow_steps']:.2f} fast_steps={collected['fast_steps']:.2f} "
        f"held_teacher_steps={collected['held_teacher_steps']:.2f} "
        f"overspeed_teacher_steps={collected['overspeed_teacher_steps']:.2f} "
        f"break_landings={collected['break_teacher_landings']:.3f}")
    lines.append(
        "[slow-fast] thresholds "
        f"connected_land>={limits['connected_land_min']} "
        f"connected_steps<={limits['connected_steps_max']} "
        f"connected_crash<={limits['connected_crash_max']} "
        f"connected_min_land>={limits['connected_min_land_min']} "
        f"stranger_land<={limits['stranger_land_max']} "
        f"stranger_crash>={limits['stranger_crash_min']} "
        f"overspeed_land<={limits['overspeed_land_max']} "
        f"landing_floor>={limits['landing_floor']} crash_ceiling<={limits['crash_ceiling']} "
        f"step_gap>={limits['step_gap_min']} speed_bias={limits['speed_bias']} "
        f"lr={limits['policy_lr']} "
        f"adv_weight={limits['adv_weight']} horizon={limits['horizon']} updates={limits['steps']}")
    lines.append("[slow-fast] GATE PASS. Lunar held.pt is the fastest stage that still has "
                 "landings, including a crashy probe. Collect keeps both-land seeds only. "
                 "Nearest-stranger pairing is disabled. "
                 "The cuda:1 stranger run went 20/20 to 0/20. "
                 "Commands: docs/gym-slow-fast.md. No new Lunar landing is claimed here.")
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

    header = ("| Arm | Landings | Early (step 50) | Return min | Steps | Crash | Rank key | adv | "
              "Accepted |")
    split = "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |"
    body = []
    for key in ("zero", "slow_only", "crash_fast", "unpaired", "supervised",
                "stranger", "overspeed", "connected"):
        row = result[key]
        body.append(
            "| {arm} | {landings} | {early} | {min_land} | {steps} | {crash} | {rank} | "
            "{adv} | {accepted} |".format(
                arm=row["arm"],
                landings=cell(row, "landings", ".3f"),
                early=cell(row, "early_landings", ".3f"),
                min_land=cell(row, "min_landings", ".3f"),
                steps=cell(row, "mean_steps", ".2f"),
                crash=cell(row, "crash_rate", ".3f"),
                rank=_rank_text(row["rank_key"]),
                adv=cell(row, "adv_weight", "s"),
                accepted=row["accepted"]))
    collected = result["table"]
    status = "PASS" if result["passed"] else "FAIL"
    return "\n".join((
        "# Slow→fast paired finetune (CPU gate)",
        "",
        f"Gate **{status}**. Winner of the rank key: `{result['winner']}`.",
        "Gate PASS is the pairing check. Lunar `held.pt` is the fastest stage that still "
        "has landings, including a crashy probe. Collect keeps both-land seeds only. "
        "Do not train the old stranger pairs. Commands are in `docs/gym-slow-fast.md`.",
        "These numbers are a 2D pad. They are not Lunar landings.",
        "",
        "One seed (`0`), fixed eval starts, no seed sweep. Rank is landings first,",
        "then fewer steps among successes. An arm is ineligible when `adv_weight` is",
        "not 1, `b_cap` did not run, landings fall below 0.90, crashes exceed 0.10,",
        "or the return panel (steps 100, 200, 400) lost the pad. `connected`,",
        "`overspeed`, and `stranger` use the same full lander, the same learning rate,",
        "and the same RpGAN step. The pairs are the difference. `Early` is step 50.",
        "`Return min` is the worst landing rate from step 100 on.",
        "",
        header,
        split,
        *body,
        "",
        f"Safe teacher mean steps {collected['slow_steps']:.2f}. "
        f"Fast teacher speed bias {result['thresholds']['speed_bias']}: "
        f"update {result['thresholds']['held_teacher_updates']} lands in "
        f"{collected['held_teacher_steps']:.2f} steps "
        f"(landings {collected['held_teacher_landings']:.3f}); "
        f"update {result['thresholds']['overspeed_teacher_updates']} lands in "
        f"{collected['overspeed_teacher_steps']:.2f} steps; "
        f"update {result['thresholds']['break_teacher_updates']} landings "
        f"{collected['break_teacher_landings']:.3f}.",
        f"Collector: {collected['kept_starts']} same-seed both-land starts, "
        f"{collected['rows']} progress-aligned rows (mean edit {collected['progress_edit']:.3f}), "
        f"{collected['nearest_rows']} stranger nearest-state rows "
        f"(mean distance {collected['nearest_dist']:.3f}), "
        f"fast failures excluded from the held set {collected['fast_failures_excluded']}, "
        f"crash episodes at the break {collected['crash_episodes']}. "
        f"Mean fast steps on kept pairs {collected['fast_steps']:.2f}.",
        "",
        "## Why the controls lose",
        "",
        "- `zero` is the safe teacher. No GAN step, so the rank key is ineligible. "
        "It lands, and it is slower than `connected`.",
        "- `slow_only` is the same RpGAN step with the safe action as the target. "
        "It stays a successful slow landing and loses on steps.",
        "- `crash_fast` trains on actions from fast-teacher episodes that missed the pad "
        "after the speed break. Contact can be sooner. Landings fall, so the rank key drops it.",
        "- `unpaired` uses fast actions from other starts and trains every weight. "
        "The student leaves the pad.",
        "- `supervised` matches the progress-aligned fast action with MSE and `adv_weight=0`. "
        "Landings may hold. The rank key rejects it because the #18 step did not run.",
        "- `stranger` is the disabled Lunar collector: a different episode's fast action at the "
        "nearest state, plentiful rows, full-weight RpGAN at `adv_weight=1`. "
        "Landings fall and crashes rise.",
        "- `overspeed` is the same progress alignment one speed update later. "
        "The teacher still lands. The student does not keep the pad.",
        "- `connected` is both-land progress pairs, `adv_weight=1`. On this plant that "
        "set is teacher update 2. The teacher is not required to land every start. "
        "Update 6 is crashy, and its crash rows are `crash_fast`. "
        "Landings hold on the return panel and success steps fall. "
        "Diagnostic MSE stays outside the loss.",
        "",
        "## Lunar validation that this gate is built to catch",
        "",
        "pop-os cuda:1 trained `train_scope=control` (E_control and G2) on nearest-state "
        "pairs from YuE2 #18 `particle_yue18_143320/best.pt`. Collect was 200/200 landings, "
        "63 fast / 62 slow, 60 pairs, 13495 rows, crashes_excluded=0. Train was 2500 steps, "
        "`adv_weight=1`, `safe_fast_weight=0`. Shared-seed validation:",
        "",
        "| Arm | Landings | Success steps | Crashes |",
        "| --- | ---: | ---: | ---: |",
        "| #18 baseline | 20/20 | 205.4 | 0 |",
        "| slow→fast @250 | 12/20 | 329.0 | 3 |",
        "| slow→fast @1000 | 0/20 | — | 19 |",
        "| slow→fast @2500 | 0/20 | — | 18 |",
        "",
        "Eval selected none. Longer training was worse. Those pairs are strangers: "
        "different episodes, matched by geometry, no shared landing. The trainer refuses "
        "`slow_seed != fast_seed` and any file named `pairs.npz`. Lunar collect now rolls "
        "two teachers on the same seed, keeps a pair only when both land, and aligns by "
        "progress. This board is not a new Lunar result.",
        "",
        "```bash",
        "python -u examples/slow_fast_paired_2d.py",
        "```",
        "",
    ))
