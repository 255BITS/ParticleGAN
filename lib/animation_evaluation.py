"""Fixed dream-rollout benchmark for the sprite animation world model.

Every leaderboard row is scored by `evaluate_split` with a predictor
f(scaled st) -> flat record [st, st+1, frame]. Errors are standardized-state
MSE against the exact simulator from the same start, capped at ERROR_CAP per
state (diverged_50 reports how often the cap is hit); frames are [-1, 1] MSE.
"""
import math

import torch

from lib.sprite_animation import render, step as sim_step, STATE_DIM
from lib.animation_transition import split, encoded_step, OFFSET_BOUND


HORIZONS = (1, 5, 20, 50)
START_OFFSETS = (0, 15, 30)
FAIL_THRESHOLD = 1.
ERROR_CAP = 10.  # per-state cap so one divergent rollout cannot dominate the mean


@torch.no_grad()
def dream(predict, scaler, start, n):
    """Returns physical dreamed states [B, n+1, 6] and frames [B, n, 1024]."""
    states, frames, current = [start], [], scaler(start)
    for _ in range(n):
        _, successor, frame = split(predict(current))
        frames.append(frame)
        current = successor
        states.append(scaler.inverse(successor))
    return torch.stack(states, 1), torch.stack(frames, 1)


@torch.no_grad()
def evaluate_split(predict, scaler, episodes, horizon=max(HORIZONS)):
    """episodes [E, T+1, 6] physical; rolls out from each START_OFFSET."""
    starts = [o for o in START_OFFSETS if o + horizon < episodes.shape[1]]
    truth = torch.cat([episodes[:, o:o + horizon + 1] for o in starts])
    states, frames = dream(predict, scaler, truth[:, 0], horizon)
    raw = ((states - truth) / scaler.scale).square().mean(2)            # [B, horizon+1]
    raw = torch.nan_to_num(raw, nan=float("inf"))
    err = raw.clamp_max(ERROR_CAP)
    failed = err[:, 1:] > FAIL_THRESHOLD
    first_fail = torch.where(failed.any(1), failed.float().argmax(1) + 1,
                             torch.full_like(failed[:, 0], horizon + 1, dtype=torch.long))
    flat = lambda x: x.reshape(-1, x.shape[-1])
    self_frames = render(flat(torch.nan_to_num(states[:, :-1]).clamp(-1e3, 1e3))).flatten(1)
    true_frames = render(flat(truth[:, :-1])).flatten(1)
    out = {f"dream_mse_{h}": float(err[:, h].mean()) for h in HORIZONS}
    out.update(dream_score=sum(out[f"dream_mse_{h}"] for h in HORIZONS) / len(HORIZONS),
               first_fail_median=float(first_fail.float().median()),
               frame_self_mse=float((flat(frames) - self_frames).square().mean()),
               frame_true_mse=float((flat(frames) - true_frames).square().mean()),
               diverged_50=float((raw[:, horizon] >= ERROR_CAP).float().mean()),
               rollouts=len(truth))
    s, n = episodes[:, :-1].reshape(-1, STATE_DIM), episodes[:, 1:].reshape(-1, STATE_DIM)
    pred = torch.cat([split(predict(scaler(c)))[1] for c in s.split(4096)])
    out["one_step_mse"] = float(((scaler.inverse(pred) - n) / scaler.scale).square().mean(1)
                                .clamp_max(ERROR_CAP).mean())
    return out


@torch.no_grad()
def prior_validity(g, prior, scaler, n=4096, seed=7):
    rng = torch.Generator(device=prior.z.device).manual_seed(seed)
    state, successor, frame = split(g(prior.sample(n, rng)[0]))
    state = scaler.inverse(state)
    exact = scaler(sim_step(state))
    return dict(prior_dynamics_mse=float((successor - exact).square().mean()),
                prior_frame_mse=float((frame - render(state).flatten(1)).square().mean()))


@torch.no_grad()
def encoder_health(e, prior, scaler, states):
    encoding, offset = e(scaler(states), prior)
    ids = encoding.indices[:, 0]
    counts = torch.bincount(ids, minlength=prior.num_particles).float()
    p = counts[counts > 0] / counts.sum()
    return dict(offset_at_bound=float((torch.tanh(offset / OFFSET_BOUND).abs() > .99).float().mean()),
                components_used=int((counts > 0).sum()),
                components_effective=float(math.exp(float(-(p * p.log()).sum()))))


def gan_predictor(e, g, prior):
    return lambda state: encoded_step(e, g, prior, state)[0]


def persistence_predictor(scaler):
    """st+1 = st; the frame is the exact render of st (no learned image)."""
    def predict(state):
        return torch.cat([state, state, render(scaler.inverse(state)).flatten(1)], 1)
    return predict
