"""Expert records and E_control(st, previous at) -> z -> G2 action selection."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.train_gym_transition import build_models, load_checkpoint
from lib.gym_transition import GymTransitionScaler, GymTransitionEncoder

OFF_ACTION = np.array([-1., 0.], dtype=np.float32)


def build_expert_records(episodes_path, split="train"):
    """Read actual heuristic transitions; previous commands precede shuffling."""
    episodes = json.loads(Path(episodes_path).read_text())
    records = {key: [] for key in ("states", "previous_actions", "actions", "next_states",
                                   "terrain", "episode_ids", "steps")}
    selected = [e for e in episodes if e["split"] == split and e["behavior"] == "heuristic"]
    if not selected:
        raise ValueError(f"No heuristic episodes for {split}")
    seen = set()
    for episode in selected:
        eid = episode["episode_id"]
        if eid in seen:
            raise ValueError("Duplicate expert episode ID")
        seen.add(eid)
        actions = np.asarray(episode["actions"], dtype=np.float32)
        n = len(actions)
        if n == 0:
            raise ValueError("Empty expert episode")
        previous = np.concatenate([OFF_ACTION[None], actions[:-1]])
        values = dict(states=episode["states"], previous_actions=previous, actions=actions,
                      next_states=episode["next_states"], terrain=np.tile(episode["terrain"], (n, 1)),
                      episode_ids=np.full(n, eid, dtype=np.int64), steps=np.arange(n, dtype=np.int64))
        for key, value in values.items():
            value = np.asarray(value, dtype=np.int64 if key in ("episode_ids", "steps") else np.float32)
            width = {"states": 8, "previous_actions": 2, "actions": 2, "next_states": 8, "terrain": 11}.get(key)
            expected = (n,) if width is None else (n, width)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} in episode {eid}")
            records[key].append(value)
        if (np.abs(actions) > 1).any():
            raise ValueError("Expert actions outside [-1,1]")
        for key in ("states", "next_states"):
            if not np.isin(values[key], [0, 1])[:, 6:].all():
                raise ValueError("Contacts must be binary")
    return {key: np.concatenate(value) for key, value in records.items()}


def initialize_control(checkpoint, arm, device="cpu"):
    if arm not in ("imitation", "joint"):
        raise ValueError("Expected imitation or joint arm")
    bundle = load_checkpoint(checkpoint, device)
    if bundle["G"] is None or bundle["E"] is None or bundle["D"] is None:
        raise ValueError("Control initialization requires the adversarial three-G checkpoint")
    bundle["world_config"] = copy.deepcopy(bundle["config"])
    bundle["E_control"] = copy.deepcopy(bundle["E"])
    for key in ("G", "E", "prior", "D", "E_control"):
        bundle[key].requires_grad_(False)
    if arm == "joint":
        for key in ("G", "E", "prior", "D"):
            bundle[key].train().requires_grad_(True)
    else:
        bundle["G"].branches[1].train().requires_grad_(True)
    bundle["E_control"].train().requires_grad_(True)
    return bundle


def predict_control(bundle, states, previous_actions, terrain):
    """Differentiable batched physical actions and particle routing information."""
    device, scaler = bundle["device"], bundle["scaler"]
    s, a, c = [torch.as_tensor(x, device=device, dtype=torch.float32)
               for x in (states, previous_actions, terrain)]
    encoded = bundle["E_control"](torch.cat([scaler.state(s), scaler.action(a)], 1), c, bundle["prior"])
    action = bundle["G"].branches[1](torch.cat([encoded.codes[:, 0], c], 1)).tanh()
    return action, encoded


@torch.no_grad()
def control_action_details(bundle, state, previous_action, terrain):
    action, encoded = predict_control(bundle, np.asarray(state)[None], np.asarray(previous_action)[None],
                                       np.asarray(terrain)[None])
    result = action[0].cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Control G2 produced nonfinite command")
    center = bundle["prior"].means()[encoded.indices[:, 0]]
    offset = (encoded.codes[:, 0] - center) / bundle["prior"].sigma
    return result, dict(component_id=int(encoded.indices[0, 0]),
                        offset_saturation=float((offset.abs() >= 2.97).float().mean()),
                        offset_norm=float(offset.norm(dim=1).mean()))


@torch.no_grad()
def control_action(bundle, state, previous_action, terrain):
    action, encoded = predict_control(bundle, np.asarray(state)[None], np.asarray(previous_action)[None],
                                       np.asarray(terrain)[None])
    result = action[0].cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Control G2 produced nonfinite command")
    return result, int(encoded.indices[0, 0])


def load_control_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("format") != "gym_control_v1":
        raise ValueError("Expected a gym_control_v1 checkpoint")
    scaler = GymTransitionScaler(**saved["scaler"]).to(device)
    bundle = build_models(saved["world_config"], scaler, device)
    bundle["E_control"] = GymTransitionEncoder(z_dim=saved["world_config"]["z_dim"],
        width=saved["world_config"]["encoder_width"], context_dim=saved["world_config"]["context_dim"]).to(device)
    for key in ("G", "E", "prior", "D", "E_control"):
        bundle[key].load_state_dict(saved[key])
        bundle[key].eval().requires_grad_(False)
    bundle.update(config=saved["config"], world_config=saved["world_config"], step=saved["step"],
                  validation=saved["validation"], provenance=saved["provenance"])
    # Preserve context_dim for the existing world-model predictor helper.
    bundle["config"] = {**bundle["config"], "context_dim": saved["world_config"]["context_dim"]}
    summary = Path(path).parent / "summary.json"
    if summary.exists():
        bundle["training_summary"] = json.loads(summary.read_text())
    return bundle
