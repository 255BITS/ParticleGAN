"""Procedural bouncing, spinning sprite: exact dynamics, renderer and splits.

State is [x, y, vx, vy, sin(phase), cos(phase)] in a unit box. Dynamics are
deterministic given the state, so a dreamed trajectory can be scored against
the simulator from the same start. Frames are rendered from states, never
stored, so every split is pinned by its state arrays alone.

The in-distribution splits start low and slow enough that no state ever
reaches the OOD band (y >= OOD_BAND); ood_test starts inside that band with
faster speeds, which adds ceiling bounces training never shows.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


STATE_DIM = 6
IMAGE_SIZE = 32
RADIUS = .1
DT = 1 / 30
GRAVITY = 4.
RESTITUTION = .8
SPIN = 4.  # rad/s
OOD_BAND = .75
EPISODE_LENGTH = 80
SPLITS = dict(  # name: (episodes, y0 range, |v0| range, launch angle range, seed offset)
    train=(2000, (.2, .6), (0., 1.), (0., 2 * math.pi), 0),
    validation=(200, (.2, .6), (0., 1.), (0., 2 * math.pi), 1),
    test=(200, (.2, .6), (0., 1.), (0., 2 * math.pi), 2),
    ood_test=(200, (.75, .9), (1.2, 1.6), (math.pi / 6, 5 * math.pi / 6), 3))  # upward: ceiling bounces


def step(state):
    """One exact transition for [B, 6] float64/float32 arrays or tensors."""
    lib = torch if torch.is_tensor(state) else np
    x, y, vx, vy, s, c = (state[:, i] for i in range(6))
    vy = vy - GRAVITY * DT
    x, y = x + vx * DT, y + vy * DT
    lo, hi = RADIUS, 1 - RADIUS
    for pos_is_x in (True, False):
        p, v = (x, vx) if pos_is_x else (y, vy)
        below, above = p < lo, p > hi
        p = lib.where(below, 2 * lo - p, lib.where(above, 2 * hi - p, p))
        v = lib.where(below | above, -RESTITUTION * v, v)
        if pos_is_x:
            x, vx = p, v
        else:
            y, vy = p, v
    ds, dc = math.sin(SPIN * DT), math.cos(SPIN * DT)
    s, c = s * dc + c * ds, c * dc - s * ds
    return lib.stack([x, y, vx, vy, s, c], 1)


def simulate(start, length):
    """[B, 6] -> [B, length + 1, 6] including the start state."""
    states = [start]
    for _ in range(length):
        states.append(step(states[-1]))
    lib = torch if torch.is_tensor(start) else np
    return lib.stack(states, 1)


def sample_starts(n, y_range, speed_range, angle_range, rng):
    x = rng.uniform(RADIUS + .05, 1 - RADIUS - .05, n)
    y = rng.uniform(*y_range, n)
    angle, speed = rng.uniform(*angle_range, n), rng.uniform(*speed_range, n)
    phase = rng.uniform(0, 2 * math.pi, n)
    return np.stack([x, y, speed * np.cos(angle), speed * np.sin(angle),
                     np.sin(phase), np.cos(phase)], 1)


def render(states, size=IMAGE_SIZE):
    """[B, 6] tensor -> [B, 1, size, size] in [-1, 1]: disk with a dark spin notch."""
    states = torch.as_tensor(states, dtype=torch.float32)
    axis = (torch.arange(size, device=states.device, dtype=torch.float32) + .5) / size
    px, py = axis[None, None, :], (1 - axis)[None, :, None]  # row 0 is the top of the box

    def disk(cx, cy, radius):
        d = ((px - cx[:, None, None]) ** 2 + (py - cy[:, None, None]) ** 2).sqrt()
        return ((radius - d) * size + .5).clamp(0, 1)

    x, y, s, c = states[:, 0], states[:, 1], states[:, 4], states[:, 5]
    body = disk(x, y, RADIUS)
    notch = disk(x + .55 * RADIUS * c, y + .55 * RADIUS * s, .4 * RADIUS)
    return ((body - notch).clamp(0, 1) * 2 - 1)[:, None]


def make_dataset(out_dir, seed=24002, episode_length=EPISODE_LENGTH, scale=1.):
    """Write <split>.npz state episodes and metadata.json; `scale` shrinks episode counts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(seed=seed, episode_length=episode_length, radius=RADIUS, dt=DT, gravity=GRAVITY,
                restitution=RESTITUTION, spin=SPIN, image_size=IMAGE_SIZE, ood_band=OOD_BAND, splits={})
    for name, (episodes, y_range, speed_range, angle_range, offset) in SPLITS.items():
        rng = np.random.default_rng(seed + 1000 * offset)
        episodes = max(2, round(episodes * scale))
        states = simulate(sample_starts(episodes, y_range, speed_range, angle_range, rng), episode_length)
        states = states.astype(np.float32)
        path = out / f"{name}.npz"
        np.savez(path, states=states)
        meta["splits"][name] = dict(episodes=episodes, y0=y_range, speed0=speed_range, angle0=angle_range,
            max_y=float(states[..., 1].max()), in_band=float((states[..., 1] >= OOD_BAND).mean()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for name in ("train", "validation", "test"):
        if meta["splits"][name]["in_band"] > 0:
            raise AssertionError(f"{name} reaches the OOD band y >= {OOD_BAND}")
    train = np.load(out / "train.npz")["states"].reshape(-1, STATE_DIM)
    ood = np.load(out / "ood_test.npz")["states"].reshape(-1, STATE_DIM)
    lo, hi = train.min(0), train.max(0)
    meta["ood_outside_train_box"] = float(((ood < lo) | (ood > hi)).any(1).mean())
    first = np.load(out / "ood_test.npz")["states"][:, :51]
    meta["ood_first50_outside_train_box"] = float(((first < lo) | (first > hi)).any(2).mean())
    meta["ood_ceiling_bounce_episodes"] = float((first[..., 1] > 1 - RADIUS - .02).any(1).mean())
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_episodes(data_dir, split, device="cpu"):
    with np.load(Path(data_dir) / f"{split}.npz", allow_pickle=False) as data:
        states = np.array(data["states"], copy=True)
    if states.ndim != 3 or states.shape[2] != STATE_DIM or not np.isfinite(states).all():
        raise ValueError(f"Invalid {split} episodes")
    return torch.as_tensor(states, device=device)


def transitions(episodes):
    """[E, T+1, 6] -> (s_t, s_t+1) pairs, each [E*T, 6]."""
    return episodes[:, :-1].reshape(-1, STATE_DIM), episodes[:, 1:].reshape(-1, STATE_DIM)
