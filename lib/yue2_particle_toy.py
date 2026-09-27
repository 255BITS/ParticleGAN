"""CPU 2D lander gate for the particle controller collapse.

The state law is symmetric and the expert thrust is odd, so negating a pair
keeps the joint law of (state, action). Marginal RpGAN plus the recipe critic
penalty has no restoring force on that sign. Playback still applies the action on the
true state, so the flipped controller misses the pad.

The accepted arm is the YuE2 paired-error game: the recipe RpGAN loss on the
expert's paired error (zero) versus the student's normalized action error,
with the recipe critic penalty and adversarial weight 1. A soft L2 arm at
adv_weight 0 is the rejected supervised control.

This module defines ONLY the three problems (samplers, scalar-gain student,
critic, rollout metrics, verdict). Optimizers and their LR schedule, the
loss, the critic penalty, critic input noise, generator output noise and EMA
come from the shipped recipe through ``benchmarks.toy_runner``. The gate
runs each arm on that runner and combines the arm verdicts.

This file does not run YuE2. Late-layer weighting is not in FORMULATION.md
(every AR projection is rank/alpha 8/8). UNI16 feature matching and end-margin
MSE are not in the v2 teacher. Distillation rel-L2 is the rejected supervised
arm, not the shipped update.
"""
import json

import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.toy_runner import Networks, ToyProblem, View, run
from lib.vendor.concept_slider_core.reference import register_paired_error_norm
from particlegan import get_recipe, init

BATCH = 64
# Sized for the recipe LR (the old 200-update budget went with a caller-set 12x LR).
STEPS = 1000
GAIN = 0.12
HORIZON = 40
POSITION_LIMIT = 0.15
VELOCITY_LIMIT = 0.20
COLLAPSED_LAND_MAX = 0.05
COLLAPSED_REL_MIN = 2.0
FIXED_LAND_MIN = 0.95
FIXED_REL_MAX = 0.05

MAPPING = (
    dict(yue2="Paired-error critic. real = noise, fake = noise + (g - t) / s. "
              "FORMULATION.md, Paired-error game and noise. A marginal critic is "
              "unchanged if two rows exchange targets.",
         toy="Accepted arm: recipe Rp logistic on the normalized action residual. "
             "real is the expert's paired error (0) on the same states the student "
             "acts on (the runner pairs fake with the real batch). The noise is the "
             "recipe's critic input noise and generator output noise, drawn "
             "independently for real and fake. A shared draw could be passed through "
             "the same pairing; the card's step-dependent schedule cannot, because a "
             "problem does not see the step.",
         gym="controller_objective. adv_weight locked at 1. The four-path sample "
             "RpGAN is not the controller step."),
    dict(yue2="Edit scale from std(target - neutral), then a gain so median row "
              "RMS is 1. FORMULATION.md, Normalize the paired edit.",
         toy="Neutral is zero thrust. Scale is fit on expert minus 0.",
         gym="Neutral is the frozen initialization action. Expert is the target. "
             "build_edit_critic refuses absolute-target whitening."),
    dict(yue2="Lazy sample-point b_cap every fourth update, coefficient 1, "
              "compensated by 4. FORMULATION.md, Objectives, gradient cap and moving average.",
         toy="Replaced by the recipe critic penalty, applied by the shared runner on "
             "every critic update of both GAN arms.",
         gym="edit_game(recipe, critic) on the global-mix critic. Logged as penalty."),
    dict(yue2="AR QKVO only. NAR, MLP, embeddings, and VAE stay frozen. "
              "FORMULATION.md, Inference: a nonlinear correction in AR attention.",
         toy="Accepted arm trains the action sign only. The collapsed arm also "
             "trains a state sign, which can flip with the action and still match the joint.",
         gym="train_scope control. E_control and G2 train. G1, G3, E_pair, prior, "
             "and transition D stay frozen."),
    dict(yue2="Distillation rel-L2, MSE(student, teacher) / MSE(teacher, base). "
              "DISTILLATION.md, Hidden-state refinement. The v2 teacher itself "
              "has no output MSE.",
         toy="Supervised arm uses only that ratio on the noise-free student action, "
             "adv_weight 0 (no critic; the recipe generator optimizer). It lands, "
             "and the gate still rejects it.",
         gym="Not a training knob. imitation_weight, real_encoding_weight, and "
             "synthetic_reconstruction_weight stay 0."),
    dict(yue2="Late-layer weighting is not in the card or the v2 configs.",
         toy="Not implemented.",
         gym="Not a knob."),
    dict(yue2="Strength sampling in distillation is outside a linear action head. "
              "The teacher residual scale is linear in the adapter output, so it "
              "does not add a second action target on this lander.",
         toy="Not required for the sign to recover under paired RpGAN.",
         gym="Not a knob. Playback is full-strength G2."),
)


def expert_action(state):
    """Odd PD law. With a symmetric state distribution, a and -a share a law."""
    position, velocity = state[:, :2], state[:, 2:]
    return (-2.2 * position - 1.4 * velocity).clamp(-1, 1)


def _states(generator, rows):
    return torch.rand(rows, 4, generator=generator) * 2 - 1


def _rollout(policy, rows=80):
    generator = torch.Generator().manual_seed(123)
    position = torch.rand(rows, 2, generator=generator) * 1.2 - 0.6
    velocity = torch.rand(rows, 2, generator=generator) * 0.4 - 0.2
    for _ in range(HORIZON):
        action = policy(torch.cat([position, velocity], 1))
        velocity = velocity + GAIN * action
        position = position + GAIN * velocity
    landed = (position.norm(dim=1) < POSITION_LIMIT) & (velocity.norm(dim=1) < VELOCITY_LIMIT)
    states = torch.rand(256, 4, generator=torch.Generator().manual_seed(9)) * 2 - 1
    rel = F.mse_loss(policy(states), expert_action(states)) / F.mse_loss(
        expert_action(states), torch.zeros(256, 2)).clamp_min(1e-4)
    return float(landed.float().mean()), float(rel)


def _edit_std():
    """Per-coordinate paired-edit scale (expert minus zero thrust)."""
    module = nn.Module()
    states = _states(torch.Generator().manual_seed(3), 512)
    targets = expert_action(states)
    register_paired_error_norm(module, targets, torch.zeros_like(targets))
    return module.target_std


class ScalarGain(nn.Module):
    """The student: action = clamp(alpha * expert(state)); optionally a state sign beta."""

    def __init__(self, state_sign=False):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(-1.0))
        self.beta = nn.Parameter(torch.tensor(-1.0)) if state_sign else None

    def forward(self, state):
        return (self.alpha * expert_action(state)).clamp(-1, 1)


class _Critic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

    def forward(self, coordinates):
        return self.net(coordinates).squeeze(-1)


def arm_verdict(metrics):
    """PASS lands with the expert sign; FAIL is the flipped controller; else INCONCLUSIVE."""
    if (metrics["landings"] >= FIXED_LAND_MIN and metrics["rel_l2"] <= FIXED_REL_MAX
            and metrics["alpha"] > 0.5):
        return "PASS"
    if (metrics["landings"] <= COLLAPSED_LAND_MAX and metrics["rel_l2"] >= COLLAPSED_REL_MIN
            and metrics["alpha"] < 0):
        return "FAIL"
    return "INCONCLUSIVE"


class _LanderArm(ToyProblem):
    """Shared problem pieces: the recipe at the gate's shape, rollout metrics, verdict."""

    adv_weight = 1.0
    accepted = False
    reason = ""

    def recipe(self):
        return get_recipe(batch_size=BATCH, total_steps=STEPS)

    def metrics(self, model):
        gain = model.nets.generator
        land, rel = _rollout(gain)
        row = dict(alpha=float(gain.alpha), landings=land, rel_l2=rel)
        if gain.beta is not None:
            row["beta"] = float(gain.beta)
        return row

    def verdict(self, metrics):
        return arm_verdict(metrics)


class CollapsedJoint(_LanderArm):
    """Joint RpGAN on (state, action) + recipe penalty. The flipped sign matches the joint law."""

    name = "yue2_collapsed_joint_rpgan"
    reason = "control_fail"

    def networks(self, recipe, seed):
        critic = init.deterministic_orthogonal_(_Critic(6), seed=seed + 1)
        return Networks(generator=ScalarGain(state_sign=True), critics=critic, prior=None)

    def real(self, n, stream):
        state = _states(stream, n)
        return torch.cat([state, expert_action(state)], 1)

    def fake(self, nets, n, stream, real):
        state = _states(stream, n) if real is None else real.x[:, :4]
        gain = nets.generator
        return torch.cat([gain.beta * state, gain(state)], 1)


class SupervisedOnly(_LanderArm):
    """adv_weight 0: relative action L2 only, no critic. This is not an accepted fix."""

    name = "yue2_supervised_only"
    adv_weight = 0.0
    reason = "adv_weight=0 leaves RpGAN and its critic penalty unapplied"

    def networks(self, recipe, seed):
        return Networks(generator=ScalarGain(), critics={}, prior=None)

    def real(self, n, stream):
        return _states(stream, n)

    def fake(self, nets, n, stream, real):
        return nets.generator(_states(stream, n) if real is None else real.x)

    def losses(self, role, nets, real, fake):
        # On the noise-free action: fake.x carries the recipe's generator output noise.
        target = expert_action(real.x)
        denom = F.mse_loss(target, torch.zeros_like(target)).clamp_min(1e-4)
        return {"supervised_loss": F.mse_loss(nets.generator(real.x), target) / denom}


class PairedError(_LanderArm):
    """Paired-error RpGAN + recipe penalty. No action MSE in the controller step."""

    name = "yue2_paired_rpgan_penalty"
    accepted = True
    reason = "controller step is RpGAN weight 1 plus the recipe critic penalty"

    def __init__(self):
        self.edit_std = _edit_std()

    def networks(self, recipe, seed):
        critic = init.deterministic_orthogonal_(_Critic(2), seed=seed + 1)
        return Networks(generator=ScalarGain(), critics=critic, prior=None)

    def real(self, n, stream):
        # The states of the pair; the expert's error on them is zero (see views).
        return _states(stream, n)

    def fake(self, nets, n, stream, real):
        state = _states(stream, n) if real is None else real.x
        return (nets.generator(state) - expert_action(state)) / self.edit_std

    def views(self, nets, real, fake):
        # real = the expert's paired error (0) on the same states; fake = the student's.
        return [View("critic", torch.zeros_like(fake.x), fake.x)]


ARMS = (CollapsedJoint, SupervisedOnly, PairedError)


def train_arm(problem, log=None):
    """One arm on the shared runner; the final live metrics plus the arm's declared role."""
    result = run(problem, log=log)
    row = dict(result["live"])
    row.update(arm=problem.name, adv_weight=problem.adv_weight, accepted=problem.accepted,
               reason=problem.reason, hold=result["hold"])
    return row


def run_gate(log=None):
    """Pass only when marginal GAN fails, supervised-only is rejected, and paired GAN lands."""
    torch.set_num_threads(1)
    collapsed, supervised, paired = (train_arm(arm(), log) for arm in ARMS)
    supervised_rejected = supervised["adv_weight"] == 0 and supervised["accepted"] is False
    passed = (collapsed["verdict"] == "FAIL" and supervised_rejected and paired["verdict"] == "PASS"
              and paired["adv_weight"] == 1 and paired["accepted"] is True)
    return dict(passed=bool(passed), collapsed=collapsed, supervised=supervised, paired=paired,
                mapping=MAPPING,
                thresholds=dict(collapsed_land_max=COLLAPSED_LAND_MAX, collapsed_rel_min=COLLAPSED_REL_MIN,
                                fixed_land_min=FIXED_LAND_MIN, fixed_rel_max=FIXED_REL_MAX))


def format_report(result):
    lines = [f"[yue2-2d] GATE {'PASS' if result['passed'] else 'FAIL'}"]
    for arm in (result["collapsed"], result["supervised"], result["paired"]):
        lines.append(
            f"[yue2-2d] {arm['arm']} verdict={arm['verdict']} landings={arm['landings']:.3f} "
            f"rel_l2={arm['rel_l2']:.3f} alpha={arm['alpha']:.3f} adv_weight={arm['adv_weight']} "
            f"accepted={arm['accepted']} reason={arm['reason']}")
    lines.append("[yue2-2d] mapping")
    for row in result["mapping"]:
        lines.append(f"[yue2-2d] yue2: {row['yue2']}")
        lines.append(f"[yue2-2d] toy: {row['toy']}")
        lines.append(f"[yue2-2d] gym: {row['gym']}")
    return "\n".join(lines)


def print_row(row):
    """Observation log: one JSON line per observation (tail -f friendly)."""
    print(json.dumps(row, default=float), flush=True)
