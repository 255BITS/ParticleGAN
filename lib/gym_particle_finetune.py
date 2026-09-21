"""Fine-tune control with the ParticleGAN game that still works after L2 is removed.

Observation-only critics match the marginal of transitions. They do not make
E_control(st, previous at) emit the latent of that expert transition, so paired
action error drifts. examples/five_modes.py drops reconstruction the same way
and stays invertible because the critic scores the joint pair (x, z) and the
encoder stays on the live real side of the relativistic loss.

Playback stays E_control(st, previous at) -> z -> G2. G1, G3, E_pair, the MoG
table, and the observation critics stay frozen, which is the control-stack
scope whose action head can be trained without dragging the world model.
"""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.train_gym_transition import build_models, load_checkpoint
from lib.gym_transition import (GymTransitionEncoder, GymTransitionScaler,
    RECORD_DIM, contact_record)

MODULE_KEYS = ("G", "E", "prior", "D", "E_control", "D_latent")
EMA_KEYS = ("G", "E", "prior", "E_control")
FAKE_PATHS = ("control", "prior")
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


class LatentJointCritic(nn.Module):
    """D(record, z) with terrain as context. The penalty is on (record, z) only."""

    def __init__(self, z_dim, context_dim, width):
        super().__init__()
        self.z_dim, self.context_dim = z_dim, context_dim
        self.net = nn.Sequential(
            nn.Linear(RECORD_DIM + z_dim + context_dim, width), nn.LeakyReLU(.2),
            nn.Linear(width, width), nn.LeakyReLU(.2),
            nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, 1))

    def forward(self, record_latent, terrain):
        return self.net(torch.cat([record_latent, terrain], 1)).squeeze(1)


def assert_finetune_scope(bundle):
    """E_control, G2, and D(record, z) learn. The world model does not."""
    def trainable(module):
        params = [p for p in module.parameters()]
        return bool(params) and all(p.requires_grad for p in params)

    def frozen(module):
        params = [p for p in module.parameters()]
        return bool(params) and all(not p.requires_grad for p in params)

    if not trainable(bundle["E_control"]) or not trainable(bundle["G"].branches[1]) or not trainable(bundle["D_latent"]):
        raise ValueError("E_control, G2, and the latent-joint critic must be trainable")
    frozen_mods = (bundle["E"], bundle["prior"], bundle["D"], bundle["G"].branches[0], bundle["G"].branches[2])
    if any(not frozen(module) for module in frozen_mods):
        raise ValueError("G1, G3, E_pair, the MoG table, and observation D stay frozen")


def initialize_particle_finetune(checkpoint, device="cpu", critic_seed=0):
    """Copy E_pair into E_control. Train only the control path and a fresh D(record, z)."""
    bundle = load_checkpoint(checkpoint, device)
    if any(bundle[key] is None for key in ("G", "E", "prior", "D")):
        raise ValueError("Particle finetune requires the adversarial three-generator checkpoint")
    bundle["world_config"] = copy.deepcopy(bundle["config"])
    bundle["E_control"] = copy.deepcopy(bundle["E"])
    world = bundle["world_config"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(critic_seed)
        bundle["D_latent"] = LatentJointCritic(world["z_dim"], world["context_dim"], world["d_width"]).to(device)
    for key in ("G", "E", "prior", "D", "E_control", "D_latent"):
        bundle[key].requires_grad_(False)
    bundle["E_control"].requires_grad_(True)
    bundle["G"].branches[1].requires_grad_(True)
    bundle["D_latent"].requires_grad_(True)
    assert_finetune_scope(bundle)
    return bundle


def control_decode(bundle, states, previous, terrain):
    """Normalized G1/G2/G3 record from the previous-action encoder. No current action."""
    scaler = bundle["scaler"]
    encoded = bundle["E_control"](torch.cat([scaler.state(states), scaler.action(previous)], 1),
                                  terrain, bundle["prior"])
    return bundle["G"](encoded.codes[:, 0], terrain), encoded


def transition_batch(bundle, states, previous, actions, next_states, terrain,
                     latent_rng, contact_rng, straight_through=False):
    """Expert record plus the control and prior pairs that the latent-joint game scores.

    Each fake is (record, code). The control code is E_control(st, previous at).
    The prior code is an unconditional MoG draw. Encoded and composed paths were
    observation-space stand-ins for reconstruction; the live (record, z) pair
    replaces them.
    """
    scaler = bundle["scaler"]
    real = scaler(torch.cat([states, actions, next_states], 1))
    control, encoded = control_decode(bundle, states, previous, terrain)
    z_prior, _ = bundle["prior"].sample(len(states), latent_rng)
    prior = bundle["G"](z_prior, terrain)
    fakes = dict(
        control=(contact_record(control, rng=contact_rng, straight_through=straight_through),
                 encoded.codes[:, 0]),
        prior=(contact_record(prior, rng=contact_rng, straight_through=straight_through), z_prior))
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


def particle_game(critic, real, fakes, terrain, gan, *, reg=None, step=1, rng=None):
    """RpGAN on (record, z). The generator step keeps E_control live on the real pair.

    reg is set only on the discriminator step, which detaches both pairs and adds
    sample-point b_cap on (record, z) for the control pair. The prior fake is
    scored against that same real pair. On the generator step the prior term
    reads D(record, z_control) without detaching z_control; detaching it is the
    observation-only game and is rejected.
    """
    if set(fakes) != set(FAKE_PATHS):
        raise ValueError(f"Unexpected fake paths: {sorted(set(fakes))}")
    generator_step = reg is None
    packed = {}
    for name, (record, code) in fakes.items():
        packed[name] = (record, code) if generator_step else (record.detach(), code.detach())
    real_obs = real if generator_step else real.detach()
    code_real = packed["control"][1]
    if generator_step and not code_real.requires_grad:
        raise RuntimeError("Latent-joint generator step needs a live E_control code on the real pair")
    real_in = torch.cat([real_obs, code_real], 1)

    def score(record_latent):
        return critic(record_latent, terrain)

    terms, losses = {}, []
    for name in FAKE_PATHS:
        fake_in = torch.cat(packed[name], 1)
        if generator_step:
            real_score = score(real_in).detach() if name == "control" else score(real_in)
            loss = gan.g_loss(score(fake_in), real_score)
        else:
            loss = gan.d_loss(score(real_in), score(fake_in))
        terms[name + "_gan"] = loss
        losses.append(loss)
    total = torch.stack(losses).sum()
    if not generator_step:
        penalty = reg(score, real_in, torch.cat(packed["control"], 1), step, rng)
        terms["joint_penalty"] = penalty
        total = total + penalty
    return total, terms


def load_particle_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("format") != "gym_particle_finetune_v2":
        raise ValueError("Expected a gym_particle_finetune_v2 checkpoint")
    cfg = saved["config"]
    for key in ("imitation_weight", "real_encoding_weight", "synthetic_reconstruction_weight"):
        if cfg[key] != 0:
            raise ValueError(f"{key} must stay 0 in a particle finetune checkpoint")
    scaler = GymTransitionScaler(**saved["scaler"]).to(device)
    bundle = build_models(saved["world_config"], scaler, device)
    world = saved["world_config"]
    bundle["E_control"] = GymTransitionEncoder(z_dim=world["z_dim"], width=world["encoder_width"],
                                               context_dim=world["context_dim"]).to(device)
    bundle["D_latent"] = LatentJointCritic(world["z_dim"], world["context_dim"], world["d_width"]).to(device)
    for key in MODULE_KEYS:
        bundle[key].load_state_dict(saved[key])
        bundle[key].eval().requires_grad_(False)
    bundle.update(config={**cfg, "context_dim": world["context_dim"]}, world_config=world,
                  step=saved["step"], provenance=saved["provenance"], validation=saved["validation"])
    summary = Path(path).parent / "summary.json"
    if summary.exists():
        bundle["training_summary"] = json.loads(summary.read_text())
    return bundle
