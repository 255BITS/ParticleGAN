"""Continuous probability flow follow-up; see MASS_SPEC.md."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import toy


ROOT = Path(__file__).parent
ARMS = ("learned_payoff", "oracle_payoff")


def self_check():
    toy.self_check()
    p = toy.softmax(np.linspace(-.7, .8, toy.N))
    payoff = np.linspace(.2, 1.1, toy.N) ** 2
    dt = 1e-7
    moved = toy.softmax(np.log(p) - dt * payoff)
    expected = p * (p @ payoff - payoff)
    assert np.max(np.abs((moved - p) / dt - expected)) < 1e-9


def run(case, arm, real_u, row_u, cat_u):
    initial, _ = toy.init_rows(case)
    fixed_g = case == "null"
    g = toy.Adam((toy.N, toy.C), toy.LR_G)
    g.x[...] = initial
    d = toy.Adam(toy.C, toy.LR_D)
    scale = np.ones(toy.C)
    if case == "frozen_d":
        d.x[:] = [-.4, 0., 0., .4]
    logmass = np.zeros(toy.N)
    lifetime_loge = 0.
    gauge_check = None
    histories = {k: [] for k in ("tv", "critic_regret", "prequential_d_loss",
                                  "log_evidence", "min_mass", "ess", "rare_mass",
                                  "rare_rows")}
    tic = time.perf_counter()
    for t in range(toy.T):
        p = toy.true_law(case, t)
        if case == "gauge" and t == 800:
            old_d = d.x * scale
            old_rows = toy.row_prob(g.x, fixed_g)
            old_payoff = old_rows @ np.logaddexp(
                0., old_d[toy.draw(p, real_u[t])][:, None] - old_d[None, :]).mean(0)
            change = np.array([16., .25, 1., 1.])
            scale *= change
            d.x /= change
            new_d = d.x * scale
            new_payoff = old_rows @ np.logaddexp(
                0., new_d[toy.draw(p, real_u[t])][:, None] - new_d[None, :]).mean(0)
            gauge_check = {"critic_max_abs": float(np.max(np.abs(new_d - old_d))),
                           "payoff_max_abs": float(np.max(np.abs(new_payoff - old_payoff)))}
            assert max(gauge_check.values()) < 1e-12
        rows = toy.row_prob(g.x, fixed_g)
        pi = toy.softmax(logmass)
        q = pi @ rows
        real = toy.draw(p, real_u[t])
        fake_row = toy.draw(pi, row_u[t])
        fake = np.array([toy.draw(rows[i], u) for i, u in zip(fake_row, cat_u[t])])
        dscore = d.x * scale
        gap = dscore[real] - dscore[fake]
        preq_loss = float(np.logaddexp(0., -gap).mean())
        lifetime_loge += float(np.log(2 * toy.sigmoid(gap)).sum())
        if arm == "oracle_payoff":
            evaluator_score = np.log((p + 1e-12) / (q + 1e-12))
            per_cat = (p[:, None] * np.logaddexp(
                0., evaluator_score[:, None] - evaluator_score[None, :])).sum(0)
        else:
            per_cat = np.logaddexp(0., dscore[real][:, None] - dscore[None, :]).mean(0)
        payoff = rows @ per_cat

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

        logmass -= toy.LR_G * payoff
        logmass -= np.max(logmass)  # common offset only; avoid needless exponent range growth
        pi = toy.softmax(logmass)
        rows = toy.row_prob(g.x, fixed_g)
        q = pi @ rows
        histories["tv"].append(toy.tv(p, q))
        histories["critic_regret"].append(toy.critic_regret(p, q, d.x * scale))
        histories["prequential_d_loss"].append(preq_loss)
        histories["log_evidence"].append(lifetime_loge)
        histories["min_mass"].append(float(pi.min()))
        histories["ess"].append(float(1 / np.dot(pi, pi)))
        histories["rare_mass"].append(float(q[3]))
        histories["rare_rows"].append(int((np.argmax(rows, axis=1) == 3).sum()))
    elapsed = time.perf_counter() - tic
    values = histories["tv"]
    return {
        "final_tv": values[-1], "last_quarter_mean_tv": float(np.mean(values[1200:])),
        "cumulative_tv": float(np.sum(values)),
        "last_quarter_critic_regret": float(np.mean(histories["critic_regret"][1200:])),
        "last_quarter_prequential_loss": float(np.mean(histories["prequential_d_loss"][1200:])),
        "final_q": [float(x) for x in q], "final_real": [float(x) for x in p],
        "final_min_row_mass": histories["min_mass"][-1],
        "final_effective_rows": histories["ess"][-1],
        "final_rare_mass": histories["rare_mass"][-1],
        "final_rare_rows": histories["rare_rows"][-1],
        "min_rare_rows": min(histories["rare_rows"]),
        "max_log_evidence": max(histories["log_evidence"]),
        "gauge_check": gauge_check, "seconds": elapsed,
    }, histories


def main():
    self_check()
    rng = np.random.default_rng(toy.SEED)
    real_u = rng.random((toy.T, toy.B))
    row_u = rng.random((toy.T, toy.B))
    cat_u = rng.random((toy.T, toy.B))
    prior = json.loads((ROOT / "results.json").read_text())
    result = {"seed": toy.SEED, "N": toy.N, "batch_size": toy.B,
              "steps": toy.T, "cases": {}}
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
                  "ESS", round(outcome["final_effective_rows"], 2), flush=True)
            (ROOT / "mass_results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(ROOT / "mass_trajectories.npz", **curves)
    print("wrote mass_results.json and mass_trajectories.npz", flush=True)


if __name__ == "__main__":
    main()
