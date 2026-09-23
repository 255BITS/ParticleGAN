"""Whole-episode sparse action labels with abundant expert state/successor pairs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lib.gym_transition import GymTransitionScaler, state_reconstruction

LABEL_SELECTION_SALT = "lunar-sparse-actions-v1"


def build_sparse_records(episodes_path, labeled_episode_count=5):
    """Never inspect hidden action values, including for validation or scaling."""
    episodes = [e for e in json.loads(Path(episodes_path).read_text())
                if e["split"] == "train" and e["behavior"] == "heuristic"]
    ids = [e["episode_id"] for e in episodes]
    if len(set(ids)) != len(ids) or any(type(i) is not int for i in ids):
        raise ValueError("Expert episode IDs must be distinct integers")
    if type(labeled_episode_count) is not int or not 1 <= labeled_episode_count <= len(ids):
        raise ValueError("Invalid labeled episode count")
    hashes = {i: hashlib.sha256(f"{LABEL_SELECTION_SALT}:{i}".encode()).hexdigest() for i in ids}
    selected = sorted(ids, key=lambda i: (hashes[i], i))[:labeled_episode_count]
    columns = {k: [] for k in ("states", "next_states", "terrain", "episode_ids", "steps", "label_mask")}
    labeled_actions = []
    for episode in episodes:
        eid = episode["episode_id"]
        states = np.asarray(episode["states"], dtype=np.float32)
        n = len(states)
        if not n:
            raise ValueError("Empty expert episode")
        values = dict(states=states, next_states=episode["next_states"],
            terrain=np.tile(episode["terrain"], (n, 1)), episode_ids=np.full(n, eid, dtype=np.int64),
            steps=np.arange(n, dtype=np.int64), label_mask=np.full(n, eid in selected, dtype=bool))
        for key, value in values.items():
            dtype = bool if key == "label_mask" else np.int64 if key in ("episode_ids", "steps") else np.float32
            value = np.asarray(value, dtype=dtype)
            width = {"states": 8, "next_states": 8, "terrain": 11}.get(key)
            if value.shape != ((n,) if width is None else (n, width)) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} in episode {eid}")
            if key in ("states", "next_states") and not np.isin(value[:, 6:], [0, 1]).all():
                raise ValueError("Contacts must be binary")
            columns[key].append(value)
        if eid in selected:
            actions = np.asarray(episode["actions"], dtype=np.float32)
            if actions.shape != (n, 2) or not np.isfinite(actions).all() or (np.abs(actions) > 1).any():
                raise ValueError(f"Invalid labeled actions in episode {eid}")
            labeled_actions.append(actions)
    records = {key: np.concatenate(values) for key, values in columns.items()}
    records["labeled_indices"] = np.flatnonzero(records["label_mask"])
    records["labeled_actions"] = np.concatenate(labeled_actions)
    selection = dict(salt=LABEL_SELECTION_SALT, rule="SHA256(salt:episode_id), ascending digest, first five",
        labeled_episode_ids=selected, candidate_episode_ids=ids,
        ordering=[dict(episode_id=i, sha256=hashes[i]) for i in sorted(ids, key=lambda i: (hashes[i], i))])
    return records, selection


def fit_sparse_scaler(records):
    states = torch.as_tensor(np.concatenate([records["states"][:, :6], records["next_states"][:, :6]], 0))
    actions = torch.as_tensor(records["labeled_actions"])
    return GymTransitionScaler(states.mean(0), states.std(0, unbiased=False).clamp_min(1e-3),
        actions.mean(0), actions.std(0, unbiased=False).clamp_min(1e-3))


def sparse_task_losses(bundle, labeled_states, labeled_actions, labeled_terrain, states, next_states, terrain):
    """Separate label-normalized action batch and all-record auxiliary batch."""
    scaler, g, e, prior, cfg = [bundle[k] for k in ("scaler", "G", "E", "prior", "config")]
    action_encoding = e(scaler.state(labeled_states), labeled_terrain, prior)
    action = g.branches[1](torch.cat([action_encoding.codes[:, 0], labeled_terrain], 1)).tanh()
    action_loss = F.mse_loss(scaler.action(action), scaler.action(labeled_actions))
    auxiliary_encoding = e(scaler.state(states), terrain, prior)
    z = auxiliary_encoding.codes[:, 0]
    if cfg["arm"] == "probes":
        z = z.detach()
    inputs = torch.cat([z, terrain], 1)
    state, st = state_reconstruction(g.branches[0](inputs), scaler.state(states),
        cfg["continuous_weight"], cfg["contact_weight"])
    successor, sn = state_reconstruction(g.branches[2](inputs), scaler.state(next_states),
        cfg["continuous_weight"], cfg["contact_weight"])
    terms = dict(action_loss=action_loss, state_loss=state, next_loss=successor,
        state_continuous=st["continuous"], state_contact=st["contact"],
        next_continuous=sn["continuous"], next_contact=sn["contact"])
    return action_loss + cfg["lambda_state"] * state + cfg["lambda_next"] * successor, terms
