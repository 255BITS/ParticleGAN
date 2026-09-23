"""One state-only encoder: E(st) -> z; G1 -> st, G2 -> at, G3 -> st+1."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.gym_transition import (GymTransitionGenerator, GymTransitionScaler, _check,
    mlp, contact_record, state_reconstruction)
from particlegan import get_recipe
from particlegan.autoencoder import particle_ae

MODULE_KEYS = ("G", "E", "prior")


class GymStateEncoder(nn.Module):
    """Only observed state and terrain are inputs; expert actions are targets."""
    def __init__(self, z_dim=32, width=128, context_dim=11):
        super().__init__()
        self.context_dim = context_dim
        self.features = mlp(8 + context_dim, width, width)
        self.query = nn.Linear(width, z_dim)
        self.offset = nn.Linear(width, z_dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, state, terrain, prior):
        _check(state, 8, "E current state input")
        _check(terrain, self.context_dim, "terrain")
        features = self.features(torch.cat([state, terrain], 1))
        query = F.layer_norm(self.query(features), (self.query.out_features,))
        return particle_ae(query, self.offset(features), prior, temperature=.25,
                           distance_reduction="sum", offset_bound=3.)


def training_recipe(cfg):
    return get_recipe(prior_kind='mog', sigma_rel=0.025, z_dim=cfg["z_dim"], num_particles=cfg["num_particles"],
                      total_steps=cfg["steps"], batch_size=cfg["batch_size"])


def build_state_models(cfg, scaler, device="cpu"):
    """Fresh identical initialization across arms with separate initialization RNGs."""
    device = torch.device(device)
    recipe = training_recipe(cfg)
    # Neural modules initialize on CPU, without consuming either experiment GPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg["seed"])
        g = GymTransitionGenerator(scaler, cfg["z_dim"], cfg["width"], cfg["context_dim"]).to(device)
        torch.random.default_generator.manual_seed(cfg["seed"] + 102)
        e = GymStateEncoder(cfg["z_dim"], cfg["encoder_width"], cfg["context_dim"]).to(device)
    prior = recipe.make_prior(device=device,
        generator=torch.Generator(device=device).manual_seed(cfg["seed"] + 101))
    return dict(G=g, E=e, prior=prior, scaler=scaler.to(device), device=device, config=cfg)


def parameter_hashes(bundle):
    result = {}
    for key in MODULE_KEYS:
        digest = hashlib.sha256()
        for name, value in bundle[key].state_dict().items():
            digest.update(name.encode())
            if isinstance(value, torch.Tensor):
                digest.update(str((tuple(value.shape), value.dtype)).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        result[key] = digest.hexdigest()
    return result


def predict_state_control(bundle, states, terrain, *, detach_probes=False):
    """Differentiable normalized record18/contact logits, plus particle encoding."""
    states, terrain = [torch.as_tensor(x, device=bundle["device"], dtype=torch.float32)
                       for x in (states, terrain)]
    encoding = bundle["E"](bundle["scaler"].state(states), terrain, bundle["prior"])
    z = encoding.codes[:, 0]
    auxiliary_z = z.detach() if detach_probes else z
    g = bundle["G"]
    state = g.branches[0](torch.cat([auxiliary_z, terrain], 1))
    action = g.branches[1](torch.cat([z, terrain], 1)).tanh()
    successor = g.branches[2](torch.cat([auxiliary_z, terrain], 1))
    return torch.cat([state, bundle["scaler"].action(action), successor], 1), encoding


def task_losses(bundle, states, actions, next_states, terrain):
    cfg, scaler = bundle["config"], bundle["scaler"]
    decoded, encoding = predict_state_control(bundle, states, terrain,
                                              detach_probes=cfg["arm"] == "probes")
    state, st = state_reconstruction(decoded[:, :8], scaler.state(states),
        cfg["continuous_weight"], cfg["contact_weight"])
    successor, sn = state_reconstruction(decoded[:, 10:], scaler.state(next_states),
        cfg["continuous_weight"], cfg["contact_weight"])
    action = F.mse_loss(decoded[:, 8:10], scaler.action(actions))
    terms = dict(action_loss=action, state_loss=state, next_loss=successor,
        state_continuous=st["continuous"], state_contact=st["contact"],
        next_continuous=sn["continuous"], next_contact=sn["contact"])
    return action + cfg["lambda_state"] * state + cfg["lambda_next"] * successor, terms, encoding


@torch.no_grad()
def control_action_details(bundle, state, terrain):
    state, terrain = [torch.as_tensor(np.asarray(x)[None], device=bundle["device"], dtype=torch.float32)
                      for x in (state, terrain)]
    encoding = bundle["E"](bundle["scaler"].state(state), terrain, bundle["prior"])
    action = bundle["G"].branches[1](torch.cat([encoding.codes[:, 0], terrain], 1)).tanh()[0].cpu().numpy()
    if not np.isfinite(action).all():
        raise FloatingPointError("Nonfinite state-control action")
    center = bundle["prior"].means()[encoding.indices[:, 0]]
    offset = (encoding.codes[:, 0] - center) / bundle["prior"].sigma
    return action, dict(component_id=int(encoding.indices[0, 0]),
        offset_saturation=float((offset.abs() >= 2.97).float().mean()),
        offset_norm=float(offset.norm(dim=1).mean()))


@torch.no_grad()
def state_control_action(bundle, state, terrain):
    action, info = control_action_details(bundle, state, terrain)
    return action, info["component_id"]


def load_state_control_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("format") != "gym_state_control_v1":
        raise ValueError("Expected a gym_state_control_v1 checkpoint")
    bundle = build_state_models(saved["config"], GymTransitionScaler(**saved["scaler"]), device)
    for key in MODULE_KEYS:
        bundle[key].load_state_dict(saved[key])
        bundle[key].eval().requires_grad_(False)
    bundle.update(step=saved["step"], provenance=saved["provenance"], validation=saved["validation"])
    summary = Path(path).parent / "summary.json"
    if summary.exists():
        bundle["training_summary"] = json.loads(summary.read_text())
    return bundle
