"""Pinned Lunar Lander collection and exact seed/prefix replay utilities.

Only individual transitions and terrain reach the trainer. Episode prefixes are
provenance/evaluation data, never model inputs. Gym remains an optional dependency.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np

GYM_VERSION = "1.2.3"
CONTEXT_DIM = 11
PHASE_NAMES = ("flight", "approach", "contact")


def make_env():
    import gymnasium as gym
    if gym.__version__ != GYM_VERSION:
        raise RuntimeError(f"Replay adapter requires gymnasium=={GYM_VERSION}; got {gym.__version__}")
    return gym.make("LunarLander-v3", continuous=True, enable_wind=False)


def terrain_context(env) -> np.ndarray:
    """11 ground heights in observation coordinates; x is linspace(-1,1,11)."""
    from gymnasium.envs.box2d.lunar_lander import LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W
    base = env.unwrapped
    vertices = [poly[0] for poly in base.sky_polys] + [base.sky_polys[-1][1]]
    vertices = np.asarray(vertices, dtype=np.float64)
    expected = np.linspace(0, VIEWPORT_W / SCALE, CONTEXT_DIM)
    if vertices.shape != (CONTEXT_DIM, 2) or not np.allclose(vertices[:, 0], expected):
        raise RuntimeError("Installed terrain geometry differs from pinned adapter")
    return ((vertices[:, 1] - base.helipad_y - LEG_DOWN / SCALE) / (VIEWPORT_H / SCALE / 2)).astype(np.float32)


def phases(states: np.ndarray, next_states: np.ndarray | None = None) -> np.ndarray:
    """0 flight, 1 approach (y<.25), 2 contact in either observed role."""
    states = np.asarray(states)
    contact = np.any(states[..., 6:8] > .5, axis=-1)
    if next_states is not None:
        contact |= np.any(np.asarray(next_states)[..., 6:8] > .5, axis=-1)
    return np.where(contact, 2, np.where(states[..., 1] < .25, 1, 0))


def replay_anchor(env, seed: int, prefix) -> np.ndarray:
    """Reset full Box2D world and RNG, then replay actions to a live anchor."""
    state, _ = env.reset(seed=int(seed))
    for index, action in enumerate(prefix):
        state, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            raise ValueError(f"Replay prefix reached episode end at step {index + 1}")
    return state


def replay_episode(seed: int, actions) -> dict[str, np.ndarray]:
    env = make_env()
    try:
        state, _ = env.reset(seed=int(seed))
        terrain = terrain_context(env)
        rows = []
        for action in actions:
            action = np.asarray(action, dtype=np.float32)
            following, reward, terminated, truncated, _ = env.step(action)
            rows.append((state, action, following, reward, terminated, truncated))
            state = following
            if terminated or truncated:
                break
        result = rows_to_arrays(rows)
        result["terrain"] = np.repeat(terrain[None], len(rows), axis=0)
        return result
    finally:
        env.close()


def rows_to_arrays(rows):
    names = ("states", "actions", "next_states", "rewards", "terminated", "truncated")
    return {name: np.asarray([row[i] for row in rows], dtype=bool if i >= 4 else np.float32)
            for i, name in enumerate(names)}


def load_split(directory, split):
    with np.load(Path(directory) / f"{split}.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simulator_provenance():
    import Box2D
    import gymnasium
    from gymnasium.envs.box2d import lunar_lander
    source = Path(inspect.getfile(lunar_lander))
    return {"gymnasium": gymnasium.__version__, "box2d": Box2D.__version__,
            "environment": "LunarLander-v3", "continuous": True, "enable_wind": False,
            "source_sha256": sha256(source), "terrain_dim": CONTEXT_DIM,
            "terrain_encoding": "11 observation-frame ground heights at x=linspace(-1,1,11)",
            "privileged_terrain_context": True}


def branch_episode(task):
    """Process-pool worker: one behavior and three replayed alternatives/anchor."""
    episode, anchors, selection_seed = task
    rng = np.random.default_rng(selection_seed)
    env = make_env()
    rows, steps, resets = [], 0, 0
    try:
        for anchor in anchors:
            state = np.asarray(episode["states"][anchor], np.float32)
            behavior = np.asarray(episode["actions"][anchor], np.float32)
            # Full main off/on and lateral off/left/right coverage, with independent
            # command magnitudes, stratified by anchor and alternative index.
            alternatives = [behavior]
            for j in range(3):
                main_on = (anchor + j) % 2 == 0
                main = rng.uniform(.001, 1.) if main_on else rng.uniform(-1., 0.)
                lateral_role = (anchor + j) % 3
                side = rng.uniform(-.5, .5) if lateral_role == 0 else (
                    rng.uniform(-1., -.501) if lateral_role == 1 else rng.uniform(.501, 1.))
                alternatives.append(np.asarray([main, side], np.float32))
            for j, action in enumerate(alternatives):
                if j == 0:
                    following = np.asarray(episode["next_states"][anchor], np.float32)
                    reward = episode["rewards"][anchor]
                    terminated, truncated = episode["terminated"][anchor], episode["truncated"][anchor]
                else:
                    restored = replay_anchor(env, episode["seed"], episode["actions"][:anchor])
                    steps += anchor
                    resets += 1
                    if not np.array_equal(restored, state):
                        raise RuntimeError(f"Replay mismatch in episode {episode['episode_id']} anchor {anchor}")
                    following, reward, terminated, truncated, _ = env.step(action)
                    steps += 1
                rows.append((state, action, following, reward, terminated, truncated,
                             episode["terrain"], episode["episode_id"], anchor))
        result = rows_to_arrays(rows)
        result.update(terrain=np.asarray([r[6] for r in rows], np.float32),
                      episode_ids=np.asarray([r[7] for r in rows], np.int64),
                      anchor_steps=np.asarray([r[8] for r in rows], np.int64))
        return result, steps, resets
    finally:
        env.close()
