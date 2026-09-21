"""Fine-tune G2 with the model-glue paired continuation.

The collapsed arm trained the whole graph with RpGAN and sample-point b_cap
after deleting paired L2. This continuation keeps those GAN objects configured
and inactive. It freezes the stem, particle cloud, and non-action heads, and
trains G2 with a 0.1 paired action anchor plus a functional kinematic match.
Playback stays E_control(st, previous at) -> z -> G2. There is no slider critic.
"""
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.train_gym_transition import build_models, load_checkpoint
from lib.gym_previous_gan import adversarial_loss
from lib.gym_transition import (GymTransitionEncoder, GymTransitionScaler,
    composed_transition, contact_record, encoded_transition)
from lib.model_glue_control import glue_objective, kinematic_response

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


def initialize_particle_finetune(checkpoint, device="cpu"):
    """Copy the paired encoder into E_control and train only the G2 action head."""
    bundle = load_checkpoint(checkpoint, device)
    if any(bundle[key] is None for key in ("G", "E", "prior", "D")):
        raise ValueError("Particle finetune requires the adversarial three-generator checkpoint")
    bundle["world_config"] = copy.deepcopy(bundle["config"])
    bundle["E_control"] = copy.deepcopy(bundle["E"])
    for key in MODULE_KEYS:
        bundle[key].eval().requires_grad_(False)
    action_head = bundle["G"].branches[1]
    action_head.train().requires_grad_(True)
    if not any(parameter.requires_grad for parameter in action_head.parameters()):
        raise RuntimeError("G2 must be the trainable action head")
    frozen = [name for name in ("E", "prior", "D", "E_control")
              if any(parameter.requires_grad for parameter in bundle[name].parameters())]
    frozen += [f"G{index + 1}" for index, branch in enumerate(bundle["G"].branches)
               if index != 1 and any(parameter.requires_grad for parameter in branch.parameters())]
    if frozen:
        raise RuntimeError(f"Model-glue scope left parameters trainable: {frozen}")
    return bundle


def glue_control_loss(bundle, states, previous, actions, next_states, terrain):
    """Paired anchor plus kinematic response. Diagnostics stay out of the graph."""
    scaler = bundle["scaler"]
    real = scaler(torch.cat([states, actions, next_states], 1))
    decoded, _ = control_decode(bundle, states, previous, terrain)
    student = decoded[:, 8:10]
    expert = real[:, 8:10]
    empty = torch.zeros_like(actions)
    loss, terms = glue_objective(
        student, expert,
        kinematic_response(states[:, :4], scaler.inverse_action(student)),
        kinematic_response(states[:, :4], actions),
        kinematic_response(states[:, :4], empty))
    return loss, terms, diagnostic_l2(decoded, real)


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
