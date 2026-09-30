"""Frozen N=32, batch-two critic-memory experiment. See SPEC.md."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


N, C, B, T = 32, 4, 2, 1600
SEED, LR_D, LR_G = 20260929, .00425, .0085
P0 = np.array([.5, .25, .25, 0.])
P1 = np.array([.25, .5, .25, 0.])
P_RARE = np.array([.5, .25, 7 / 32, 1 / 32])
ARMS = ("ordinary", "current_bd", "rolling_bd", "gated_rolling_bd", "direct_mass")
CASES = ("null", "overmass", "rare", "shift", "frozen_d", "gauge")
ROOT = Path(__file__).parent


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def draw(p, u):
    return np.searchsorted(np.cumsum(p), u, side="right")


def tv(a, b):
    return float(np.abs(a - b).sum() / 2)


class Adam:
    """Adam with E19 betas (0,.999) and PyTorch's default eps."""
    def __init__(self, shape, lr):
        self.x = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0
        self.lr = lr

    def step(self, g):
        self.t += 1
        self.v = .999 * self.v + .001 * g * g
        self.x -= self.lr * g / (np.sqrt(self.v / (1 - .999 ** self.t)) + 1e-8)


def init_rows(case):
    counts = {
        "null": (16, 8, 8, 0), "overmass": (12, 8, 8, 4),
        "rare": (14, 8, 9, 1), "shift": (16, 8, 8, 0),
        "frozen_d": (12, 8, 8, 4), "gauge": (12, 8, 8, 4),
    }[case]
    cat = np.repeat(np.arange(C), counts)
    assert len(cat) == N
    if case == "null":
        return np.eye(C)[cat], cat
    logits = np.zeros((N, C))
    logits[np.arange(N), cat] = math.log(.97 / .01)
    return logits, cat


def row_prob(g, fixed):
    return g if fixed else softmax(g)


def true_law(case, t):
    if case == "rare":
        return P_RARE
    if case == "shift" and t >= 800:
        return P1
    return P0


def critic_regret(p, q, d):
    a = p[:, None] * q[None, :]
    b = q[:, None] * p[None, :]
    current = float((a * np.logaddexp(0., d[None, :] - d[:, None])).sum())
    oracle = float((a[a > 0] * np.log1p(b[a > 0] / a[a > 0])).sum())
    assert current + 1e-10 >= oracle
    return max(0., current - oracle)


def self_check():
    """Algebraic checks for the two gradients and the prequential e-value."""
    p = np.array([.47, .28, .19, .06])
    q = np.array([.22, .31, .36, .11])
    d = np.array([.3, -.2, .1, -.4])
    e = float((p[:, None] * q[None, :] *
               2 * sigmoid(d[:, None] - d[None, :])).sum())
    same_law_e = float((p[:, None] * p[None, :] *
                        2 * sigmoid(d[:, None] - d[None, :])).sum())
    assert abs(same_law_e - 1.) < 1e-14 and e != 1.
    vote = p[:, None] * q[None, :] * sigmoid(d[None, :] - d[:, None])
    grad = vote.sum(0) - vote.sum(1)
    objective = lambda x: float((p[:, None] * q[None, :] *
                                 np.logaddexp(0., x[None, :] - x[:, None])).sum())
    eps = 1e-6
    numeric = np.array([(objective(d + eps * np.eye(C)[i]) -
                         objective(d - eps * np.eye(C)[i])) / (2 * eps)
                        for i in range(C)])
    assert np.max(np.abs(grad - numeric)) < 1e-9
    logits = np.array([.2, -.3, .4, .1])
    row = softmax(logits)
    per_cat = np.array([.7, .3, 1.1, .5])
    grad_g = row * (per_cat - row @ per_cat)
    score = lambda x: float(softmax(x) @ per_cat)
    numeric_g = np.array([(score(logits + eps * np.eye(C)[i]) -
                           score(logits - eps * np.eye(C)[i])) / (2 * eps)
                          for i in range(C)])
    assert np.max(np.abs(grad_g - numeric_g)) < 1e-9
    oracle_d = np.log(p / q)
    assert critic_regret(p, q, oracle_d) < 1e-12


def run(case, arm, real_u, row_u, cat_u):
    initial, assigned = init_rows(case)
    fixed_g = case == "null"
    g = Adam((N, C), LR_G)
    g.x[...] = initial
    d = Adam(C, LR_D)
    mass = Adam(N, LR_G)
    scale = np.ones(C)
    if case == "frozen_d":
        d.x[:] = [-.4, 0., 0., .4]
    memory = np.zeros(N)
    gate_loge = 0.
    lifetime_loge = 0.
    moves = []
    event_no = 1
    histories = {k: [] for k in ("tv", "critic_regret", "prequential_d_loss",
                                  "log_evidence", "min_mass", "ess", "rare_rows")}
    gauge_check = None
    tic = time.perf_counter()
    for t in range(T):
        p = true_law(case, t)
        if case == "gauge" and t == 800:
            old_d = d.x * scale
            old_payoff = row_prob(g.x, fixed_g) @ np.logaddexp(
                0., old_d[draw(p, real_u[t])][:, None] - old_d[None, :]).mean(0)
            change = np.array([16., .25, 1., 1.])
            scale *= change
            d.x /= change
            new_d = d.x * scale
            new_payoff = row_prob(g.x, fixed_g) @ np.logaddexp(
                0., new_d[draw(p, real_u[t])][:, None] - new_d[None, :]).mean(0)
            gauge_check = {"critic_max_abs": float(np.max(np.abs(new_d - old_d))),
                           "payoff_max_abs": float(np.max(np.abs(new_payoff - old_payoff)))}
            assert gauge_check["critic_max_abs"] < 1e-12
            assert gauge_check["payoff_max_abs"] < 1e-12
        rows = row_prob(g.x, fixed_g)
        pi = softmax(mass.x) if arm == "direct_mass" else np.full(N, 1 / N)
        q = pi @ rows
        real = draw(p, real_u[t])
        fake_row = draw(pi, row_u[t])
        fake = np.array([draw(rows[i], u) for i, u in zip(fake_row, cat_u[t])])
        dscore = d.x * scale
        # All candidate payoffs are computed with D_t, before fitting D on this batch.
        per_cat = np.logaddexp(0., dscore[real][:, None] - dscore[None, :]).mean(0)
        payoff = rows @ per_cat
        memory = payoff.copy() if t == 0 else (15 / 16) * memory + payoff / 16
        gap = dscore[real] - dscore[fake]
        pred = sigmoid(gap)
        preq_loss = float(np.logaddexp(0., -gap).mean())
        log_increment = float(np.log(2 * pred).sum())
        gate_loge += log_increment
        lifetime_loge += log_increment

        # D loss = mean softplus(D(fake)-D(real)); parameter is head weight.
        if case != "frozen_d":
            vote = sigmoid(dscore[fake] - dscore[real]) / B
            grad_d = (np.bincount(fake, weights=vote, minlength=C)
                      - np.bincount(real, weights=vote, minlength=C))
            d.step(grad_d * scale)
        post_score = d.x * scale
        post_cat_loss = np.logaddexp(0., post_score[real][:, None] - post_score[None, :]).mean(0)
        if not fixed_g:
            grad_g = np.zeros((N, C))
            for i in fake_row:
                r = rows[i]
                grad_g[i] += r * (post_cat_loss - r @ post_cat_loss) / B
            g.step(grad_g)
        if arm == "direct_mass":
            post_rows = row_prob(g.x, fixed_g)
            costs = post_rows @ post_cat_loss
            mass.step(pi * (costs - pi @ costs))

        if arm in ("current_bd", "rolling_bd", "gated_rolling_bd") and (t + 1) % 16 == 0:
            score = payoff if arm == "current_bd" else memory
            child, parent = int(np.argmax(score)), int(np.argmin(score))
            permitted = score[child] > score[parent] + 1e-12
            if arm == "gated_rolling_bd":
                alpha = .05 / (event_no * (event_no + 1))
                permitted &= gate_loge >= -math.log(alpha)
            if permitted and child != parent:
                old_cat = int(np.argmax(row_prob(g.x, fixed_g)[child]))
                new_cat = int(np.argmax(row_prob(g.x, fixed_g)[parent]))
                g.x[child] = g.x[parent].copy()
                g.v[child] = 0.
                memory[child] = payoff[parent]
                moves.append([t + 1, child, parent, old_cat, new_cat])
                if arm == "gated_rolling_bd":
                    gate_loge = 0.
                    event_no += 1
        rows = row_prob(g.x, fixed_g)
        pi = softmax(mass.x) if arm == "direct_mass" else np.full(N, 1 / N)
        q = pi @ rows
        histories["tv"].append(tv(p, q))
        histories["critic_regret"].append(critic_regret(p, q, d.x * scale))
        histories["prequential_d_loss"].append(preq_loss)
        histories["log_evidence"].append(lifetime_loge)
        histories["min_mass"].append(float(pi.min()))
        histories["ess"].append(float(1 / np.dot(pi, pi)))
        histories["rare_rows"].append(int((np.argmax(rows, axis=1) == 3).sum()))
    elapsed = time.perf_counter() - tic
    values = histories["tv"]
    first_alarm_after_shift = next((i + 1 for i in range(800, T)
                                    if histories["log_evidence"][i] >= math.log(20)), None)
    return {
        "final_tv": values[-1], "last_quarter_mean_tv": float(np.mean(values[1200:])),
        "cumulative_tv": float(np.sum(values)),
        "last_quarter_critic_regret": float(np.mean(histories["critic_regret"][1200:])),
        "last_quarter_prequential_loss": float(np.mean(histories["prequential_d_loss"][1200:])),
        "moves": len(moves), "move_log": moves,
        "final_q": [float(x) for x in q], "final_real": [float(x) for x in p],
        "final_rare_rows": histories["rare_rows"][-1],
        "min_rare_rows": min(histories["rare_rows"]),
        "max_log_evidence": max(histories["log_evidence"]),
        "first_lifetime_e20_after_shift": first_alarm_after_shift,
        "final_min_row_mass": histories["min_mass"][-1],
        "final_effective_rows": histories["ess"][-1],
        "gauge_check": gauge_check, "seconds": elapsed,
    }, histories


def main():
    self_check()
    rng = np.random.default_rng(SEED)
    real_u = rng.random((T, B))
    row_u = rng.random((T, B))
    cat_u = rng.random((T, B))
    results = {"seed": SEED, "N": N, "batch_size": B, "steps": T, "cases": {}}
    curves = {}
    for case in CASES:
        results["cases"][case] = {}
        for arm in ARMS:
            outcome, history = run(case, arm, real_u, row_u, cat_u)
            results["cases"][case][arm] = outcome
            for key, values in history.items():
                curves[f"{case}/{arm}/{key}"] = np.array(values)
            print(case, arm, "TV", round(outcome["final_tv"], 4),
                  "moves", outcome["moves"], flush=True)
            (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(ROOT / "trajectories.npz", **curves)
    print("wrote results.json and trajectories.npz", flush=True)


if __name__ == "__main__":
    main()
