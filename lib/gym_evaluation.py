"""Frozen numerical protocol for the first Lunar Lander comparison.

Contact errors are separate from six-coordinate continuous errors. All arrays
are physical Gym observations; only continuous errors use training-set scales.
"""
import numpy as np
import torch

from lib.toy_metrics import sliced_w1


PROTOCOL = {
    "version": 1,
    "primary": "mean squared error of six continuous fields / common train std",
    "scaler": "union of training current/next states, population std, floor .001",
    "contact_threshold": .5,
    "phase": "contact if either endpoint contact; else approach if current y < .25; else flight",
    "rollout_horizons": [1, 5, 20, 50],
    "rollout_feedback": "contacts >= .5; no state refresh or ground snapping",
    "rollout_anchor_rule": "first stored anchor per behavior-reference phase in each of first 8 sorted test episodes",
    "distribution_contexts": "first 16 sorted test episodes with >=64 reference records, equal context weight",
    "distribution_samples": "up to 512 records per context, fixed random subset",
    "distribution_radius": "95th percentile reference leave-one-out nearest neighbor distance",
    "distribution_projections": 128,
    "distribution_contact_representation": "hard Bernoulli samples, raw 0/1",
    "action_pairs": "all six unordered command pairs per four-action anchor, coupled simulator RNG",
}


def fit_scales(train):
    states = np.concatenate([train["states"], train["next_states"]])
    return states[:, :6].mean(0), states[:, :6].std(0).clip(.001)


def phases(states, next_states):
    contact = (states[:, 6:] > .5).any(1) | (next_states[:, 6:] > .5).any(1)
    return np.where(contact, "contact", np.where(states[:, 1] < .25, "approach", "flight"))


def prediction_metrics(pred, target, scale):
    pred, target = np.asarray(pred), np.asarray(target)
    if not len(pred):
        return {"count": 0}
    finite = np.isfinite(pred).all(1)
    if not finite.all():
        # Fail a run instead of silently improving its score by dropping errors.
        raise ValueError(f"{int((~finite).sum())} nonfinite predictions")
    error = (pred[:, :6] - target[:, :6]) / scale
    per_sample = (error ** 2).mean(1)
    p = pred[:, 6:].clip(1e-7, 1-1e-7).astype(np.float64)
    truth = target[:, 6:]
    hard, positives = p >= .5, truth >= .5
    tp = int((hard & positives).sum())
    return {
        "count": len(pred), "continuous_mse": float(per_sample.mean()),
        "continuous_mse_p95": float(np.quantile(per_sample, .95)),
        "position_l2": float(np.linalg.norm(pred[:, :2]-target[:, :2], axis=1).mean()),
        "velocity_l2": float(np.linalg.norm(pred[:, 2:4]-target[:, 2:4], axis=1).mean()),
        "angle_mae": float(np.abs(pred[:, 4]-target[:, 4]).mean()),
        "angular_velocity_mae": float(np.abs(pred[:, 5]-target[:, 5]).mean()),
        "contact_brier": float(((p-truth)**2).mean()),
        "contact_bce": float(-(truth*np.log(p)+(1-truth)*np.log1p(-p)).mean()),
        "contact_precision": tp / max(int(hard.sum()), 1),
        "contact_recall": tp / max(int(positives.sum()), 1),
        "contact_positive_labels": int(positives.sum()),
        "nonfinite": 0,
        "outside_horizontal_bounds": float((np.abs(pred[:, 0]) > 1).mean()),
    }


def score_predictions(pred, data, scale):
    result = prediction_metrics(pred, data["next_states"], scale)
    labels = phases(data["states"], data["next_states"])
    result["phases"] = {p: prediction_metrics(pred[labels == p], data["next_states"][labels == p], scale)
                        for p in ("flight", "approach", "contact")}
    return result


def engine_regime(actions):
    main = actions[:, 0] > 0
    side = np.where(np.abs(actions[:, 1]) > .5, np.sign(actions[:, 1]), 0).astype(int)
    return np.stack([main.astype(int), side], 1)


def action_metrics(pred, data, scale):
    keys = np.stack([data["episode_ids"], data["anchor_steps"]], 1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    groups = {name: [] for name in ("all", "threshold_crossing", "same_active_regime", "both_inactive")}
    magnitudes = {name: [] for name in groups}
    regimes = engine_regime(data["actions"])
    for group in range(int(inverse.max()) + 1):
        ids = np.flatnonzero(inverse == group)
        if len(ids) != 4 or not np.all(data["states"][ids] == data["states"][ids[0]]):
            raise ValueError("Action protocol requires four identical replayed anchor states")
        for i in range(4):
            for j in range(i+1, 4):
                a, b = ids[i], ids[j]
                real_delta = (data["next_states"][a, :6]-data["next_states"][b, :6])/scale
                delta = (pred[a, :6]-pred[b, :6])/scale
                bucket = "threshold_crossing" if np.any(regimes[a] != regimes[b]) else (
                    "both_inactive" if not regimes[a].any() else "same_active_regime")
                for name in ("all", bucket):
                    groups[name].append(float(((delta-real_delta)**2).mean()))
                    magnitudes[name].append(float((real_delta**2).mean()))
    return {name: {"pairs": len(values),
                   "effect_mse": float(np.mean(values)) if values else None,
                   "zero_response_mse": float(np.mean(magnitudes[name])) if values else None}
            for name, values in groups.items()}


def normalize_joint(states, actions, next_states, mean, scale, action_mean, action_scale):
    def state(x):
        return np.concatenate([(x[:, :6]-mean)/scale, x[:, 6:]], 1)
    return np.concatenate([state(states), (actions-action_mean)/action_scale, state(next_states)], 1)


def distribution_metrics(fake, real, device="cpu"):
    a, b = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in (fake, real)]
    distances = torch.cdist(a, b)
    reference = torch.cdist(b, b)
    reference.fill_diagonal_(float("inf"))
    radius = reference.min(1).values.quantile(.95).clamp_min(1e-6)
    def contact_hist(x):
        codes = x[:, 6].long()+2*x[:, 7].long()+4*x[:, 16].long()+8*x[:, 17].long()
        return torch.bincount(codes, minlength=16).float()/len(x)
    return {
        "joint_sw1": sliced_w1(a, b, 128, seed=31415),
        "coverage": float((distances.min(0).values <= radius).float().mean()),
        "precision": float((distances.min(1).values <= radius).float().mean()),
        "contact_pattern_tv": float((contact_hist(a)-contact_hist(b)).abs().sum()/2),
        "reference_radius": float(radius),
    }
