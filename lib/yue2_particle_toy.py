"""CPU 2D lander gate for the particle controller collapse.

The state law is symmetric and the expert thrust is odd, so negating a pair
keeps the joint law of (state, action). Marginal RpGAN plus sample-point b_cap
has no restoring force on that sign. Playback still applies the action on the
true state, so the flipped controller misses the pad.

The accepted arm is the YuE2 paired-error game: relativistic logistic loss on
noise versus noise plus the normalized action error, with the same sample-point
b_cap, and with adversarial weight 1. A soft L2 anchor at adv_weight 0 can also
land and is rejected. That is the failure mode of the closed model-glue PR.

This file does not run YuE2. Late-layer weighting is not in FORMULATION.md
(every AR projection is rank/alpha 8/8). UNI16 feature matching and end-margin
MSE are not in the v2 teacher. Distillation rel-L2 is the rejected supervised
arm, not the shipped update.
"""
import torch
from torch import nn
import torch.nn.functional as F

from lib.vendor.concept_slider_core.reference import (noise_std, register_paired_error_norm,
    rp_d_loss, rp_g_loss)
from particlegan.grad_regularizers import GradientPenalty

# One gate seed. Not a sweep.
SEED = 0
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
         toy="Accepted arm: Rp logistic on the normalized action residual.",
         gym="controller_objective. adv_weight locked at 1. The four-path sample "
             "RpGAN is not the controller step."),
    dict(yue2="Edit scale from std(target - neutral), then a gain so median row "
              "RMS is 1. FORMULATION.md, Normalize the paired edit.",
         toy="Neutral is zero thrust. Scale is fit on expert minus 0.",
         gym="Neutral is the frozen initialization action. Expert is the target. "
             "build_edit_critic refuses absolute-target whitening."),
    dict(yue2="Lazy sample-point b_cap every fourth update, coefficient 1, "
              "compensated by 4. FORMULATION.md, Objectives, gradient cap and moving average.",
         toy="GradientPenalty arm b_cap, lazy_k 4, on the critic coordinates. "
             "The accepted arm counts applications and requires at least one.",
         gym="edit_cap() on the global-mix critic. Logged as b_cap_applied."),
    dict(yue2="AR QKVO only. NAR, MLP, embeddings, and VAE stay frozen. "
              "FORMULATION.md, Inference: a nonlinear correction in AR attention.",
         toy="Accepted arm trains the action sign only. The collapsed arm also "
             "trains a state sign, which can flip with the action and still match the joint.",
         gym="train_scope control. E_control and G2 train. G1, G3, E_pair, prior, "
             "and transition D stay frozen."),
    dict(yue2="Distillation rel-L2, MSE(student, teacher) / MSE(teacher, base). "
              "DISTILLATION.md, Hidden-state refinement. The v2 teacher itself "
              "has no output MSE.",
         toy="Supervised arm uses only that ratio, adv_weight 0. Landings may "
             "pass. The gate still rejects the arm.",
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


def _rollout(policy, rows=80):
    generator = torch.Generator().manual_seed(123)
    position = torch.rand(rows, 2, generator=generator) * 1.2 - 0.6
    velocity = torch.rand(rows, 2, generator=generator) * 0.4 - 0.2
    with torch.no_grad():
        for _ in range(HORIZON):
            action = policy(torch.cat([position, velocity], 1))
            velocity = velocity + GAIN * action
            position = position + GAIN * velocity
        landed = (position.norm(dim=1) < POSITION_LIMIT) & (velocity.norm(dim=1) < VELOCITY_LIMIT)
        states = torch.rand(256, 4, generator=torch.Generator().manual_seed(9)) * 2 - 1
        neutral = torch.zeros(256, 2)
        rel = F.mse_loss(policy(states), expert_action(states)) / F.mse_loss(
            expert_action(states), neutral).clamp_min(1e-4)
    return float(landed.float().mean()), float(rel)


class _Critic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

    def forward(self, coordinates):
        return self.net(coordinates).squeeze(-1)


def _edit_scale():
    module = nn.Module()
    states = torch.rand(512, 4, generator=torch.Generator().manual_seed(3)) * 2 - 1
    targets = expert_action(states)
    register_paired_error_norm(module, targets, torch.zeros_like(targets))
    return module


def _cap():
    return GradientPenalty(arm="b_cap", coeff=1., kappa=1., lazy_k=4, norm="l2",
                           method="autograd", target_anneal="none")


def _states(generator, rows):
    return torch.rand(rows, 4, generator=generator) * 2 - 1


def _policy_metrics(alpha):
    policy = lambda state, sign=alpha.detach(): (sign * expert_action(state)).clamp(-1, 1)
    land, rel = _rollout(policy)
    return dict(alpha=float(alpha.detach()), landings=land, rel_l2=rel)


def train_collapsed(steps=200):
    """Joint RpGAN + b_cap. The flipped sign matches the joint law."""
    torch.manual_seed(SEED)
    alpha = nn.Parameter(torch.tensor(-1.0))
    beta = nn.Parameter(torch.tensor(-1.0))
    critic = _Critic(6)
    cap = _cap()
    opt = torch.optim.SGD((alpha, beta), lr=0.05)
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.0, 0.999))
    generator = torch.Generator().manual_seed(SEED + 2)
    applications = 0
    for step in range(1, steps + 1):
        state = _states(generator, 64)
        target = expert_action(state)
        fake = torch.cat([beta * state, (alpha * target).clamp(-1, 1)], 1)
        real = torch.cat([state, target], 1)
        penalty = cap(critic, real.detach(), fake.detach(), step=step)
        loss_d = rp_d_loss(critic(real), critic(fake.detach())) + penalty
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        applications += int(step % cap.lazy_k == 0)
        fake = torch.cat([beta * state, (alpha * target).clamp(-1, 1)], 1)
        loss_g = rp_g_loss(critic(real).detach(), critic(fake))
        opt.zero_grad()
        loss_g.backward()
        opt.step()
    result = _policy_metrics(alpha)
    result.update(arm="collapsed_joint_rpgan", adv_weight=1., accepted=False,
                  b_cap_applications=applications, beta=float(beta.detach()),
                  reason="control_fail")
    return result


def train_supervised(steps=200):
    """adv_weight 0. Relative action L2 only. This is not an accepted fix."""
    torch.manual_seed(SEED)
    alpha = nn.Parameter(torch.tensor(-1.0))
    opt = torch.optim.SGD((alpha,), lr=0.08)
    generator = torch.Generator().manual_seed(SEED + 2)
    for _ in range(steps):
        state = _states(generator, 64)
        target = expert_action(state)
        pred = (alpha * target).clamp(-1, 1)
        denom = F.mse_loss(target, torch.zeros_like(target)).detach().clamp_min(1e-4)
        loss = F.mse_loss(pred, target) / denom
        opt.zero_grad()
        loss.backward()
        opt.step()
    result = _policy_metrics(alpha)
    result.update(arm="supervised_only", adv_weight=0., accepted=False,
                  b_cap_applications=0, gan_grad_abs=0.,
                  reason="adv_weight=0 leaves RpGAN and b_cap unapplied")
    return result


def train_paired(steps=200):
    """Paired-error RpGAN + b_cap. No action MSE in the controller step."""
    torch.manual_seed(SEED)
    alpha = nn.Parameter(torch.tensor(-1.0))
    norm = _edit_scale()
    critic = _Critic(2)
    cap = _cap()
    opt = torch.optim.SGD((alpha,), lr=0.08)
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.0, 0.999))
    generator = torch.Generator().manual_seed(SEED + 2)
    hold = 1.3 * float(norm.edit_rms)
    applications = 0
    gan_grad_abs = 0.
    for step in range(1, steps + 1):
        state = _states(generator, 64)
        target = expert_action(state)
        pred = (alpha * target).clamp(-1, 1)
        residual = (pred.detach() - target) / norm.target_std
        sigma = noise_std(step - 1, start=norm.noise_start, decay_steps=steps, hold=hold)
        noise = torch.randn(residual.shape, generator=generator) * sigma
        fake = noise + residual
        penalty = cap(critic, noise.detach(), fake.detach(), step=step)
        loss_d = rp_d_loss(critic(noise), critic(fake)) + penalty
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        applications += int(step % cap.lazy_k == 0)
        pred = (alpha * target).clamp(-1, 1)
        residual = (pred - target) / norm.target_std
        noise = torch.randn(residual.shape, generator=generator) * sigma
        loss_g = rp_g_loss(critic(noise).detach(), critic(noise + residual))
        opt.zero_grad()
        loss_g.backward()
        gan_grad_abs += abs(float(alpha.grad.detach()))
        opt.step()
    result = _policy_metrics(alpha)
    result.update(arm="paired_rpgan_bcap", adv_weight=1., accepted=True,
                  b_cap_applications=applications, gan_grad_abs=gan_grad_abs,
                  reason="controller step is RpGAN weight 1 plus sample-point b_cap")
    return result


def run_gate():
    """Pass only when marginal GAN fails, supervised-only is rejected, and paired GAN lands."""
    torch.set_num_threads(1)
    collapsed = train_collapsed()
    supervised = train_supervised()
    paired = train_paired()
    collapsed_fail = (collapsed["landings"] <= COLLAPSED_LAND_MAX
                      and collapsed["rel_l2"] >= COLLAPSED_REL_MIN
                      and collapsed["alpha"] < 0
                      and collapsed["b_cap_applications"] > 0)
    supervised_rejected = supervised["adv_weight"] == 0 and supervised["accepted"] is False
    paired_pass = (paired["landings"] >= FIXED_LAND_MIN and paired["rel_l2"] <= FIXED_REL_MAX
                   and paired["alpha"] > 0.5 and paired["adv_weight"] == 1.
                   and paired["gan_grad_abs"] > 0. and paired["b_cap_applications"] > 0
                   and paired["accepted"] is True)
    return dict(passed=bool(collapsed_fail and supervised_rejected and paired_pass),
                collapsed=collapsed, supervised=supervised, paired=paired, mapping=MAPPING,
                thresholds=dict(collapsed_land_max=COLLAPSED_LAND_MAX,
                                collapsed_rel_min=COLLAPSED_REL_MIN,
                                fixed_land_min=FIXED_LAND_MIN, fixed_rel_max=FIXED_REL_MAX))


def format_report(result):
    lines = [f"[yue2-2d] GATE {'PASS' if result['passed'] else 'FAIL'}"]
    for arm in (result["collapsed"], result["supervised"], result["paired"]):
        lines.append(
            f"[yue2-2d] {arm['arm']} landings={arm['landings']:.3f} rel_l2={arm['rel_l2']:.3f} "
            f"alpha={arm['alpha']:.3f} adv_weight={arm['adv_weight']} "
            f"b_cap_applications={arm['b_cap_applications']} accepted={arm['accepted']} "
            f"reason={arm['reason']}")
    lines.append("[yue2-2d] mapping")
    for row in result["mapping"]:
        lines.append(f"[yue2-2d] yue2: {row['yue2']}")
        lines.append(f"[yue2-2d] toy: {row['toy']}")
        lines.append(f"[yue2-2d] gym: {row['gym']}")
    return "\n".join(lines)
