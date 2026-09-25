"""Imitation fine-tune whose action MSE is the slider paired-error game.

E_control and G2 start from the adversarial world-model checkpoint. The only
optimized control term is the Anima-style critic on noise versus noise plus the
normalized action error. Joint and marginal transition critics stay frozen, and
their sample-point gradient penalty is not used.
"""
import json
from pathlib import Path

import torch

from lib.gym_control import initialize_control, predict_control
from lib.gym_slider_gan import PairedErrorCritic, error_loss
from lib.gym_transition import GymTransitionEncoder, GymTransitionScaler
from experiments.train_gym_transition import build_models

MODULE_KEYS = ("G", "E", "prior", "D", "E_control", "R")


def critic_config(cfg):
    return dict(slider_scope="action", steps=cfg["steps"], error_tokens=cfg["error_tokens"],
                error_width=cfg["error_width"], error_heads=cfg["error_heads"])


def build_error_critic(cfg, normalized_actions, device):
    """Fit the error scale on training actions only, then build R."""
    if normalized_actions.ndim != 2 or normalized_actions.shape[1] != 2 or len(normalized_actions) < 2:
        raise ValueError("Error critic expects at least two normalized action rows")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg["seed"] + 204)
        critic = PairedErrorCritic(normalized_actions.detach().cpu(), critic_config(cfg))
    return critic.to(device)


def assert_finetune_scope(bundle):
    """E_control, G2, and R learn. G1, G3, E_pair, the prior, and D do not."""
    def all_train(module):
        params = list(module.parameters())
        if not params:
            raise ValueError("Expected parameters")
        return all(p.requires_grad for p in params)

    def all_frozen(module):
        return all(not p.requires_grad for p in module.parameters())

    if not all_train(bundle["E_control"]) or not all_train(bundle["G"].branches[1]) or not all_train(bundle["R"]):
        raise ValueError("E_control, G2, and the error critic must be trainable")
    frozen = (bundle["E"], bundle["prior"], bundle["D"], bundle["G"].branches[0], bundle["G"].branches[2])
    if any(not all_frozen(module) for module in frozen):
        raise ValueError("G1, G3, E_pair, prior, and transition D stay frozen")


def action_decoded(bundle, states, previous, terrain, real):
    """18-wide view whose columns 8:10 are normalized G2 actions.

    predict_control returns physical tanh commands. The imitation target lives
    in the frozen scaler's action coordinates, so the residual uses that same
    map. Other columns are zeros and are outside the action-scope residual.
    """
    action, _ = predict_control(bundle, states, previous, terrain)
    width = real.shape[1]
    return torch.cat([real.new_zeros(len(real), 8), bundle["scaler"].action(action),
                      real.new_zeros(len(real), width - 10)], 1)


def generator_objective(critic, decoded, real, step, rng, weight=1.):
    """Paired-error generator loss. Action MSE is a detached diagnostic."""
    learned, terms = error_loss(critic, decoded, real, step, rng)
    with torch.no_grad():
        action_mse = torch.nn.functional.mse_loss(decoded[:, 8:10], real[:, 8:10])
    if action_mse.requires_grad:
        raise RuntimeError("Action MSE diagnostic leaked into the autograd graph")
    return weight * learned, {**terms, "action_mse": action_mse}


def load_slider_finetune(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("format") != "gym_slider_finetune_v1" or saved.get("step", 0) < 1:
        raise ValueError("Expected a slider-error fine-tune checkpoint")
    cfg = saved["config"]
    if cfg.get("arm") != "slider_finetune" or cfg.get("paired_error_weight") != 1.:
        raise ValueError("Fine-tune checkpoint must keep the action paired-error objective")
    scaler = GymTransitionScaler(**saved["scaler"]).to(device)
    bundle = build_models(saved["world_config"], scaler, device)
    world = saved["world_config"]
    bundle["E_control"] = GymTransitionEncoder(z_dim=world["z_dim"], width=world["encoder_width"],
                                                context_dim=world["context_dim"]).to(device)
    bundle["R"] = PairedErrorCritic(torch.zeros(2, 2), critic_config(cfg)).to(device)
    for key in MODULE_KEYS:
        bundle[key].load_state_dict(saved[key])
        bundle[key].eval().requires_grad_(False)
    bundle.update(config=cfg, world_config=world, step=saved["step"], provenance=saved["provenance"],
                  validation=saved.get("validation"))
    summary = Path(path).parent / "summary.json"
    if summary.exists():
        bundle["training_summary"] = json.loads(summary.read_text())
    return bundle


def initialize_finetune(cfg, device):
    """Same module scope as imitation, plus a fresh action-error critic."""
    bundle = initialize_control(cfg["checkpoint"], "imitation", device)
    if bundle["D"] is None:
        raise ValueError("Fine-tune expects the adversarial world-model checkpoint")
    return bundle
