"""Fine-tune the controller with YuE2's paired-error adversarial game.

Playback stays E_control(st, previous at) -> z -> G2. Absolute imitation and
reconstruction L2 stay out of the graph. The controller step is relativistic
logistic loss on the edit-normalized action residual, plus sample-point b_cap
on that critic. A configured GAN with weight 0 is rejected.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.vendor.concept_slider_core.reference import (GlobalMixErrorCritic, noise_std,
    rp_d_loss, rp_g_loss)
from particlegan.grad_regularizers import GradientPenalty

from experiments.train_gym_transition import build_models, load_checkpoint
from lib.safe_fast_landing import require_live_adversary as _require_live_adversary
from lib.gym_previous_gan import adversarial_loss
from lib.gym_transition import (GymTransitionEncoder, GymTransitionScaler,
    composed_transition, contact_record, encoded_transition)

MODULE_KEYS = ("G", "E", "prior", "D", "E_control")
FAKE_PATHS = ("control", "prior", "encoded", "composed")
REMOVED_L2 = (
    "imitation action MSE on E_control -> G2",
    "real reconstruction MSE and contact BCE on E_pair(st, current at) -> G1/G2/G3",
    "synthetic reconstruction MSE and contact BCE on the composed E_pair target",
)


def require_classic_particle_gan(gan, reg):
    """Lock Rp logistic and sample-point b_cap. Interpolated caps are a different arm."""
    if gan.loss_type != "logistic" or gan.mode != "rp":
        raise ValueError("Arm A requires GANLoss logistic mode='rp'")
    if reg.arm != "b_cap" or reg.method != "autograd" or reg.norm != "l2" or reg.lazy_k != 1:
        raise ValueError("Arm A requires sample-point autograd L2 b_cap on every step")
    if reg.target_anneal != "none" or reg.coeff != 1. or reg.kappa != 1.:
        raise ValueError("Arm A keeps b_cap coeff 1, kappa 1, and no center anneal")


# YuE2 FORMULATION.md: the cap is applied every fourth update and multiplied by 4.
EDIT_CAP_EVERY = 4
# Action residual bound. 0.20 dipped closed-loop landings on the 2D gate.
RESIDUAL_SCALE = 0.15
RESIDUAL_IN_DIM = 21  # state 8 + previous action 2 + terrain 11
# FORMULATION.md: the other controls hold scheduled noise at 1.3 times edit RMS.
EDIT_NOISE_HOLD = 1.3


def require_live_adversary(adv_weight):
    """Reject a configured GAN that the controller step does not apply."""
    _require_live_adversary(adv_weight)


def configure_control_scope(bundle):
    """Train E_control and G2. Freeze the world the way YuE2 freezes NAR/MLP/VAE."""
    frozen = (bundle["E"], bundle["prior"], bundle["D"],
              bundle["G"].branches[0], bundle["G"].branches[2])
    for module in frozen:
        module.requires_grad_(False)
    bundle["E_control"].requires_grad_(True)
    bundle["G"].branches[1].requires_grad_(True)
    if not any(p.requires_grad for p in bundle["E_control"].parameters()):
        raise ValueError("E_control must train")
    if not any(p.requires_grad for p in bundle["G"].branches[1].parameters()):
        raise ValueError("G2 must train")
    if any(p.requires_grad for module in frozen for p in module.parameters()):
        raise ValueError("G1, G3, E_pair, prior, and transition D stay frozen")
    return bundle


class ActionResidual(nn.Module):
    """Bounded edit on a frozen physical action. Last layer starts at zero.

    `forward` returns the edit, at most `scale` on each channel. Step 0 matches
    the frozen controller.
    """

    def __init__(self, scale=RESIDUAL_SCALE, width=32, in_dim=RESIDUAL_IN_DIM):
        super().__init__()
        if float(scale) != RESIDUAL_SCALE:
            raise ValueError("residual_scale stays 0.15; 0.20 dipped landings on the toy")
        if in_dim != RESIDUAL_IN_DIM:
            raise ValueError(f"residual features are state, previous action, and terrain ({RESIDUAL_IN_DIM})")
        self.scale = float(scale)
        self.net = nn.Sequential(nn.Linear(in_dim, width), nn.Tanh(), nn.Linear(width, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.scale * self.net(features).tanh()


def residual_physical_action(bundle, states, previous, terrain):
    """Frozen tanh(G2) plus the residual, clipped to the physical action box.

    E_control and G2 are evaluated without grad. Only the residual can train.
    """
    scaler = bundle["scaler"]
    with torch.no_grad():
        encoded = bundle["E_control"](torch.cat([scaler.state(states), scaler.action(previous)], 1),
                                      terrain, bundle["prior"])
        frozen = bundle["G"].branches[1](torch.cat([encoded.codes[:, 0], terrain], 1)).tanh()
    residual = bundle.get("residual")
    if residual is None:
        return frozen
    features = torch.cat([states, previous, terrain], 1)
    if features.shape[1] != RESIDUAL_IN_DIM:
        raise ValueError(f"residual features must be {RESIDUAL_IN_DIM}-D")
    return (frozen + residual(features)).clamp(-1, 1)


@torch.no_grad()
def playback_action(bundle, state, previous_action, terrain):
    """Physical action for a rollout. Applies the residual when the checkpoint has one."""
    from lib.gym_control import control_action_details
    action, metadata = control_action_details(bundle, state, previous_action, terrain)
    residual = bundle.get("residual")
    if residual is None:
        return action, metadata
    device = next(residual.parameters()).device
    features = torch.cat([
        torch.as_tensor(np.asarray(state), device=device, dtype=torch.float32).reshape(1, -1),
        torch.as_tensor(np.asarray(previous_action), device=device, dtype=torch.float32).reshape(1, -1),
        torch.as_tensor(np.asarray(terrain), device=device, dtype=torch.float32).reshape(1, -1),
    ], 1)
    edit = residual(features)[0].detach().cpu().numpy()
    physical = np.clip(np.asarray(action, dtype=np.float32) + edit, -1., 1.).astype(np.float32)
    return physical, metadata


def normalized_g2_action(bundle, states, previous, terrain):
    """Normalized tanh action from E_control -> G2. No current action and no G1/G3."""
    scaler = bundle["scaler"]
    encoded = bundle["E_control"](torch.cat([scaler.state(states), scaler.action(previous)], 1),
                                  terrain, bundle["prior"])
    raw = bundle["G"].branches[1](torch.cat([encoded.codes[:, 0], terrain], 1))
    return (raw.tanh() - bundle["G"].action_mean) / bundle["G"].action_scale


def edit_cap():
    """Sample-point b_cap on the edit critic. Lazy every fourth step, coeff 1, times 4."""
    return GradientPenalty(arm="b_cap", coeff=1., kappa=1., lazy_k=EDIT_CAP_EVERY, norm="l2",
                           method="autograd", target_anneal="none")


def build_edit_critic(targets, neutrals, cfg):
    """Global-mix critic on the paired action edit. Card setting gmix_t8_w48_l1."""
    if targets.shape != neutrals.shape or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("Edit critic expects matching [rows, 2] action tensors")
    critic = GlobalMixErrorCritic(targets.detach().cpu(), neutrals=neutrals.detach().cpu(),
                                  tokens=cfg["error_tokens"], width=cfg["error_width"], layers=1,
                                  heads=cfg["error_heads"], score_bound=8.)
    if critic.normalization != "paired_edit_per_coordinate_std_median_rms_gain":
        raise ValueError("Edit critic must whiten target-minus-neutral, not absolute targets")
    return critic


def paired_noise(critic, predicted, target, step, rng, total_steps):
    """Shared Gaussian on the real/fake pair. Fake adds the normalized action error."""
    residual = (predicted - target.detach()) / critic.target_std
    sigma = noise_std(step - 1, start=float(critic.noise_start), decay_steps=total_steps,
                      hold=EDIT_NOISE_HOLD * float(critic.edit_rms))
    noise = torch.randn(residual.shape, device=residual.device, dtype=residual.dtype,
                        generator=rng) * sigma
    return noise, noise + residual


def discriminator_objective(critic, predicted, target, step, rng, reg, total_steps):
    """Rp logistic plus sample-point b_cap. The action graph is detached."""
    noise, fake = paired_noise(critic, predicted.detach(), target, step, rng, total_steps)
    adversarial = rp_d_loss(critic(noise), critic(fake))
    penalty = reg(critic, noise, fake, step=step)
    return adversarial + penalty, dict(error_d=adversarial.detach(), b_cap=penalty.detach(),
                                       b_cap_applied=float(step % reg.lazy_k == 0))


def controller_objective(critic, predicted, target, step, rng, total_steps, adv_weight,
                         safe_fast_cost=None, safe_fast_weight=0.):
    """RpGAN controller loss, plus an optional safe-fast cost.

    `safe_fast_weight=0` does not read `safe_fast_cost`, so the proven
    paired-error graph stays the adversarial term alone. There is no action MSE.
    """
    require_live_adversary(adv_weight)
    noise, fake = paired_noise(critic, predicted, target, step, rng, total_steps)
    with torch.no_grad():
        real_score = critic(noise)
    adversarial = rp_g_loss(real_score, critic(fake))
    if not adversarial.requires_grad:
        raise RuntimeError("Controller adversarial loss has no gradient")
    loss = adv_weight * adversarial
    terms = dict(error_g=adversarial.detach(), adv_weight=float(adv_weight),
                 safe_fast_weight=float(safe_fast_weight), safe_fast=predicted.new_zeros(()))
    if safe_fast_weight == 0:
        return loss, terms
    if not torch.is_tensor(safe_fast_cost) or not safe_fast_cost.requires_grad:
        raise ValueError("safe_fast_weight>0 requires a differentiable safe-fast cost")
    loss = loss + float(safe_fast_weight) * safe_fast_cost
    terms["safe_fast"] = safe_fast_cost.detach()
    return loss, terms


def initialize_particle_finetune(checkpoint, device="cpu"):
    """Copy the paired encoder into E_control and train the full adversarial graph."""
    bundle = load_checkpoint(checkpoint, device)
    if any(bundle[key] is None for key in ("G", "E", "prior", "D")):
        raise ValueError("Particle finetune requires the adversarial three-generator checkpoint")
    bundle["world_config"] = copy.deepcopy(bundle["config"])
    bundle["E_control"] = copy.deepcopy(bundle["E"])
    for key in MODULE_KEYS:
        bundle[key].train().requires_grad_(True)
    return bundle


def control_decode(bundle, states, previous, terrain):
    """Normalized G1/G2/G3 record from the previous-action encoder. No current action."""
    scaler = bundle["scaler"]
    encoded = bundle["E_control"](torch.cat([scaler.state(states), scaler.action(previous)], 1),
                                  terrain, bundle["prior"])
    return bundle["G"](encoded.codes[:, 0], terrain), encoded


def transition_batch(bundle, states, previous, actions, next_states, terrain,
                     latent_rng, contact_rng, straight_through=False):
    """Real expert triple plus the four fake paths that replace paired L2.

    Contact draws are consumed in path order: prior, composed successor,
    control, encoded. The encoded path still conditions E_pair on the expert
    current action, matching the reconstruction term it replaces. The control
    path does not.
    """
    scaler = bundle["scaler"]
    real = scaler(torch.cat([states, actions, next_states], 1))
    control, _ = control_decode(bundle, states, previous, terrain)
    z, _ = bundle["prior"].sample(len(states), latent_rng)
    prior = bundle["G"](z, terrain)
    encoded, _ = encoded_transition(bundle["E"], bundle["G"], bundle["prior"], real[:, :10], terrain)
    prior_obs = contact_record(prior, rng=contact_rng, straight_through=straight_through)
    composed = composed_transition(bundle["E"], bundle["G"], bundle["prior"], prior_obs, terrain,
                                   rng=contact_rng, straight_through=straight_through)[0]
    fakes = dict(
        control=contact_record(control, rng=contact_rng, straight_through=straight_through),
        prior=prior_obs,
        encoded=contact_record(encoded, rng=contact_rng, straight_through=straight_through),
        composed=composed)
    return real, fakes, control


@torch.no_grad()
def diagnostic_l2(decoded, real):
    """Logged paired errors. Built without grad so they cannot enter the objective."""
    return dict(
        action_mse=F.mse_loss(decoded[:, 8:10], real[:, 8:10]),
        state_mse=F.mse_loss(decoded[:, :6], real[:, :6]),
        next_mse=F.mse_loss(decoded[:, 10:16], real[:, 10:16]),
        state_bce=F.binary_cross_entropy_with_logits(decoded[:, 6:8], real[:, 6:8]),
        next_bce=F.binary_cross_entropy_with_logits(decoded[:, 16:18], real[:, 16:18]))


def particle_game(d, real, fakes, terrain, gan, **kwargs):
    """RpGAN over control, prior, encoded, and composed paths. See adversarial_loss."""
    unknown = set(fakes) - set(FAKE_PATHS)
    if unknown or not fakes:
        raise ValueError(f"Unexpected fake paths: {sorted(unknown) or 'none'}")
    return adversarial_loss(d, real, fakes, terrain, gan, **kwargs)


def load_particle_checkpoint(path, device="cpu", formats=("gym_particle_finetune_v1",)):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("format") not in formats:
        allowed = ", ".join(formats)
        raise ValueError(f"Expected checkpoint format {allowed}")
    cfg = saved["config"]
    for key in ("imitation_weight", "real_encoding_weight", "synthetic_reconstruction_weight"):
        if cfg[key] != 0:
            raise ValueError(f"{key} must stay 0 in a particle finetune checkpoint")
    scaler = GymTransitionScaler(**saved["scaler"]).to(device)
    bundle = build_models(saved["world_config"], scaler, device)
    world = saved["world_config"]
    bundle["E_control"] = GymTransitionEncoder(z_dim=world["z_dim"], width=world["encoder_width"],
                                               context_dim=world["context_dim"]).to(device)
    for key in MODULE_KEYS:
        bundle[key].load_state_dict(saved[key])
        bundle[key].eval().requires_grad_(False)
    bundle.update(config={**cfg, "context_dim": world["context_dim"]}, world_config=world,
                  step=saved["step"], provenance=saved["provenance"], validation=saved["validation"])
    _attach_residual(bundle, saved, device)
    summary = Path(path).parent / "summary.json"
    if summary.exists():
        bundle["training_summary"] = json.loads(summary.read_text())
    return bundle


def _attach_residual(bundle, saved, device):
    """#18 checkpoints have no residual. A slow→fast file must carry both keys."""
    has_weights = "residual" in saved
    has_spec = "residual_spec" in saved
    if not has_weights and not has_spec:
        bundle["residual"] = None
        return bundle
    if not has_weights or not has_spec:
        raise ValueError("slow-fast checkpoint needs both residual and residual_spec")
    spec = saved["residual_spec"]
    residual = ActionResidual(scale=float(spec["scale"]), width=int(spec["width"]),
                              in_dim=int(spec["in_dim"])).to(device)
    residual.load_state_dict(saved["residual"])
    residual.eval().requires_grad_(False)
    bundle["residual"] = residual
    bundle["residual_spec"] = dict(spec)
    return bundle


PAIRED_FORMATS = ("gym_particle_finetune_v1", "gym_slow_fast_finetune_v1")


def load_paired_controller(path, device="cpu"):
    """Load a #18 particle controller or a slow→fast continuation of one.

    Both formats play back `E_control -> G2`. A checkpoint that records
    `adv_weight` other than 1, or a nonzero safe-fast kinematic weight, is
    refused. Missing `safe_fast_weight` means the #18 graph, which is 0.
    """
    bundle = load_particle_checkpoint(path, device, formats=PAIRED_FORMATS)
    cfg = bundle["config"]
    if "adv_weight" not in cfg or float(cfg["adv_weight"]) != 1.:
        raise ValueError("paired controller checkpoint must record adv_weight 1")
    if float(cfg.get("safe_fast_weight", 0.) or 0.) != 0.:
        raise ValueError("slow-fast refuses a checkpoint trained with the safe-fast kinematic cost")
    return bundle
