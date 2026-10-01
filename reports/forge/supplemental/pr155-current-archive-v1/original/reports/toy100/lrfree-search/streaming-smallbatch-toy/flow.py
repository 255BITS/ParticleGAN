"""Magnitude-sensitive reaction follow-up; see FLOW_SPEC.md."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import toy


ROOT = Path(__file__).parent
ARMS = ("stream_payoff", "exact_real_payoff", "oracle_critic")


def rates(payoff):
    out = np.maximum(payoff[:, None] - payoff[None, :], 0.) / toy.N
    np.fill_diagonal(out, 0.)
    return out


def self_check():
    payoff = np.linspace(-.4, .5, toy.N)
    r = rates(payoff)
    observed = (r.sum(0) - r.sum(1)) / toy.N
    expected = (payoff.mean() - payoff) / toy.N
    assert np.max(np.abs(observed - expected)) < 1e-14
    assert rates(np.ones(toy.N)).sum() == 0.
    toy.self_check()


def run(case, arm, real_u, row_u, cat_u):
    initial, _ = toy.init_rows(case)
    fixed_g = case == "null"
    g = toy.Adam((toy.N, toy.C), toy.LR_G)
    g.x[...] = initial
    d = toy.Adam(toy.C, toy.LR_D)
    scale = np.ones(toy.C)
    if case == "frozen_d":
        d.x[:] = [-.4, 0., 0., .4]
    private = np.random.default_rng(771234)
    moves = []
    lifetime_loge = 0.
    histories = {k: [] for k in ("tv", "critic_regret", "prequential_d_loss",
                                  "log_evidence", "rare_rows", "instantaneous_rate")}
    gauge_check = None
    tic = time.perf_counter()
    for t in range(toy.T):
        p = toy.true_law(case, t)
        if case == "gauge" and t == 800:
            old_d = d.x * scale
            change = np.array([16., .25, 1., 1.])
            scale *= change
            d.x /= change
            gauge_check = float(np.max(np.abs(d.x * scale - old_d)))
            assert gauge_check < 1e-12
        rows = toy.row_prob(g.x, fixed_g)
        pi = np.full(toy.N, 1. / toy.N)
        q = pi @ rows
        real = toy.draw(p, real_u[t])
        fake_row = toy.draw(pi, row_u[t])
        fake = np.array([toy.draw(rows[i], u) for i, u in zip(fake_row, cat_u[t])])
        dscore = d.x * scale
        gap = dscore[real] - dscore[fake]
        preq_loss = float(np.logaddexp(0., -gap).mean())
        lifetime_loge += float(np.log(2 * toy.sigmoid(gap)).sum())
        if arm == "oracle_critic":
            evaluator_score = np.log((p + 1e-12) / (q + 1e-12))
            per_cat = (p[:, None] * np.logaddexp(
                0., evaluator_score[:, None] - evaluator_score[None, :])).sum(0)
        elif arm == "exact_real_payoff":
            per_cat = (p[:, None] * np.logaddexp(
                0., dscore[:, None] - dscore[None, :])).sum(0)
        else:
            per_cat = np.logaddexp(0., dscore[real][:, None] - dscore[None, :]).mean(0)
        payoff = rows @ per_cat
        histories["instantaneous_rate"].append(float(rates(payoff).sum()))

        if case != "frozen_d":
            vote = toy.sigmoid(dscore[fake] - dscore[real]) / toy.B
            grad_d = (np.bincount(fake, weights=vote, minlength=toy.C)
                      - np.bincount(real, weights=vote, minlength=toy.C))
            d.step(grad_d * scale)
        post_score = d.x * scale
        post_cat_loss = np.logaddexp(0., post_score[real][:, None] - post_score[None, :]).mean(0)
        if not fixed_g:
            grad_g = np.zeros((toy.N, toy.C))
            for i in fake_row:
                r = rows[i]
                grad_g[i] += r * (post_cat_loss - r @ post_cat_loss) / toy.B
            g.step(grad_g)

        remaining = toy.LR_G
        while remaining > 0:
            r = rates(payoff)
            total = float(r.sum())
            if total == 0:
                break
            delay = float(private.exponential(1. / total))
            if delay > remaining:
                break
            flat = int(np.searchsorted(np.cumsum(r.ravel()), private.random() * total, side="right"))
            child, parent = divmod(flat, toy.N)
            assert child != parent and payoff[child] > payoff[parent]
            old_cat = int(np.argmax(toy.row_prob(g.x, fixed_g)[child]))
            new_cat = int(np.argmax(toy.row_prob(g.x, fixed_g)[parent]))
            g.x[child] = g.x[parent].copy()
            g.v[child] = 0.
            payoff[child] = payoff[parent]
            moves.append([t + 1, child, parent, old_cat, new_cat])
            remaining -= delay

        rows = toy.row_prob(g.x, fixed_g)
        q = rows.mean(0)
        histories["tv"].append(toy.tv(p, q))
        histories["critic_regret"].append(toy.critic_regret(p, q, d.x * scale))
        histories["prequential_d_loss"].append(preq_loss)
        histories["log_evidence"].append(lifetime_loge)
        histories["rare_rows"].append(int((np.argmax(rows, axis=1) == 3).sum()))
    elapsed = time.perf_counter() - tic
    values = histories["tv"]
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
        "mean_instantaneous_event_rate": float(np.mean(histories["instantaneous_rate"])),
        "gauge_check": gauge_check, "seconds": elapsed,
    }, histories


def main():
    self_check()
    rng = np.random.default_rng(toy.SEED)
    real_u = rng.random((toy.T, toy.B))
    row_u = rng.random((toy.T, toy.B))
    cat_u = rng.random((toy.T, toy.B))
    prior = json.loads((ROOT / "results.json").read_text())
    result = {"seed": toy.SEED, "private_seed": 771234,
              "N": toy.N, "batch_size": toy.B, "steps": toy.T, "cases": {}}
    curves = {}
    for case in toy.CASES:
        reference, _ = toy.run(case, "ordinary", real_u, row_u, cat_u)
        recorded = prior["cases"][case]["ordinary"]
        for key in ("final_tv", "last_quarter_mean_tv", "cumulative_tv",
                    "last_quarter_critic_regret", "final_q"):
            assert reference[key] == recorded[key], (case, key)
        result["cases"][case] = {}
        for arm in ARMS:
            outcome, history = run(case, arm, real_u, row_u, cat_u)
            result["cases"][case][arm] = outcome
            for key, values in history.items():
                curves[f"{case}/{arm}/{key}"] = np.array(values)
            print(case, arm, "TV", round(outcome["final_tv"], 4),
                  "moves", outcome["moves"], flush=True)
            (ROOT / "flow_results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(ROOT / "flow_trajectories.npz", **curves)
    print("wrote flow_results.json and flow_trajectories.npz", flush=True)


if __name__ == "__main__":
    main()
