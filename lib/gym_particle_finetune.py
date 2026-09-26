"""Fine-tune the controller with YuE2's paired-error adversarial game.

Playback stays E_control(st, previous at) -> z -> G2. Absolute imitation and
reconstruction L2 stay out of the graph. The controller step is the recipe's
GAN loss on the edit-normalized action residual; the edit critic trains with
the recipe's critic optimizer and penalty (``recipe.make_critic_optimizer`` /
``recipe.make_critic_penalty``). A configured GAN with weight 0 is rejected.
"""
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from lib.vendor.concept_slider_core.reference import GlobalMixErrorCritic, noise_std

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


def normalized_g2_action(bundle, states, previous, terrain):
    """Normalized tanh action from E_control -> G2. No current action and no G1/G3."""
    scaler = bundle["scaler"]
    encoded = bundle["E_control"](torch.cat([scaler.state(states), scaler.action(previous)], 1),
                                  terrain, bundle["prior"])
    raw = bundle["G"].branches[1](torch.cat([encoded.codes[:, 0], terrain], 1))
    return (raw.tanh() - bundle["G"].action_mean) / bundle["G"].action_scale


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


def edit_game(recipe, critic, **adam_kwargs):
    """``(gan, opt_r, penalty)`` for the edit critic, all from the recipe.

    ``opt_r`` is ``recipe.make_critic_optimizer`` with a fresh EMA copy of the
    critic; ``penalty`` is ``recipe.make_critic_penalty(opt_r)``.
    """
    opt_r = recipe.make_critic_optimizer(critic, ema_critic=copy.deepcopy(critic), **adam_kwargs)
    return recipe.make_loss(), opt_r, recipe.make_critic_penalty(opt_r)


def paired_noise(critic, predicted, target, step, rng, total_steps):
    """Shared Gaussian on the real/fake pair. Fake adds the normalized action error."""
    residual = (predicted - target.detach()) / critic.target_std
    sigma = noise_std(step - 1, start=float(critic.noise_start), decay_steps=total_steps,
                      hold=EDIT_NOISE_HOLD * float(critic.edit_rms))
    noise = torch.randn(residual.shape, device=residual.device, dtype=residual.dtype,
                        generator=rng) * sigma
    return noise, noise + residual


def discriminator_objective(critic, predicted, target, step, rng, penalty, total_steps, gan):
    """Recipe GAN loss plus the recipe critic penalty. The action graph is detached.

    ``penalty`` is ``recipe.make_critic_penalty(opt_r)`` for this edit critic.
    """
    noise, fake = paired_noise(critic, predicted.detach(), target, step, rng, total_steps)
    adversarial = gan.d_loss(critic(noise), critic(fake))
    value = penalty(critic, noise, fake)
    return adversarial + value, dict(error_d=adversarial.detach(), penalty=value.detach())


def controller_objective(critic, predicted, target, step, rng, total_steps, adv_weight,
                         safe_fast_cost=None, safe_fast_weight=0., *, gan):
    """Recipe GAN controller loss, plus an optional safe-fast cost.

    `safe_fast_weight=0` does not read `safe_fast_cost`, so the proven
    paired-error graph stays the adversarial term alone. There is no action MSE.
    """
    require_live_adversary(adv_weight)
    noise, fake = paired_noise(critic, predicted, target, step, rng, total_steps)
    with torch.no_grad():
        real_score = critic(noise)
    fake_score = critic(fake)
    adversarial = gan.g_loss(fake_score, real_score)
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


def load_particle_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("format") != "gym_particle_finetune_v1":
        raise ValueError("Expected a gym_particle_finetune_v1 checkpoint")
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
    summary = Path(path).parent / "summary.json"
    if summary.exists():
        bundle["training_summary"] = json.loads(summary.read_text())
    return bundle
