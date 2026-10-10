"""Replay learned mass flow and measure one-step payoff prediction; see OPTIMISM_SPEC.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import toy
import mass_flow


ROOT = Path(__file__).parent


def center(v, pi):
    return v - pi @ v


def error(v, oracle, pi):
    delta = center(v, pi) - center(oracle, pi)
    return float(pi @ (delta * delta))


def run(case, real_u, row_u, cat_u):
    initial, _ = toy.init_rows(case)
    fixed_g = case == "null"
    g = toy.Adam((toy.N, toy.C), toy.LR_G)
    g.x[...] = initial
    d = toy.Adam(toy.C, toy.LR_D)
    scale = np.ones(toy.C)
    if case == "frozen_d":
        d.x[:] = [-.4, 0., 0., .4]
    logmass = np.zeros(toy.N)
    previous_online = previous_population = None
    errors = {key: 0. for key in (
        "online_current", "online_extrapolated", "population_current",
        "population_extrapolated", "online_sampling_error")}
    tv_values = []
    for t in range(toy.T):
        p = toy.true_law(case, t)
        if case == "gauge" and t == 800:
            change = np.array([16., .25, 1., 1.])
            before = d.x * scale
            scale *= change
            d.x /= change
            assert np.max(np.abs(before - d.x * scale)) < 1e-12
        rows = toy.row_prob(g.x, fixed_g)
        pi = toy.softmax(logmass)
        q = pi @ rows
        real = toy.draw(p, real_u[t])
        fake_row = toy.draw(pi, row_u[t])
        fake = np.array([toy.draw(rows[i], u) for i, u in zip(fake_row, cat_u[t])])
        dscore = d.x * scale
        paired_loss = np.logaddexp(0., dscore[:, None] - dscore[None, :])
        online = rows @ paired_loss[real].mean(0)
        population = rows @ (p @ paired_loss)
        oracle_score = np.log((p + 1e-12) / (q + 1e-12))
        oracle_loss = np.logaddexp(0., oracle_score[:, None] - oracle_score[None, :])
        oracle = rows @ (p @ oracle_loss)
        if t:
            errors["online_current"] += error(online, oracle, pi)
            errors["online_extrapolated"] += error(
                2 * online - previous_online, oracle, pi)
            errors["population_current"] += error(population, oracle, pi)
            errors["population_extrapolated"] += error(
                2 * population - previous_population, oracle, pi)
            errors["online_sampling_error"] += error(online, population, pi)
        previous_online = online.copy()
        previous_population = population.copy()

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
        logmass -= toy.LR_G * online
        logmass -= np.max(logmass)
        pi = toy.softmax(logmass)
        q = pi @ toy.row_prob(g.x, fixed_g)
        tv_values.append(toy.tv(p, q))
    return {**errors,
            "online_ratio": errors["online_extrapolated"] / errors["online_current"],
            "population_ratio": errors["population_extrapolated"] / errors["population_current"],
            "final_tv": tv_values[-1], "cumulative_tv": sum(tv_values)}


def main():
    mass_flow.self_check()
    rng = np.random.default_rng(toy.SEED)
    real_u = rng.random((toy.T, toy.B))
    row_u = rng.random((toy.T, toy.B))
    cat_u = rng.random((toy.T, toy.B))
    saved = json.loads((ROOT / "mass_results.json").read_text())["cases"]
    result = {"seed": toy.SEED, "cases": {}}
    for case in toy.CASES:
        output = run(case, real_u, row_u, cat_u)
        prior = saved[case]["learned_payoff"]
        assert output["final_tv"] == prior["final_tv"], case
        assert abs(output["cumulative_tv"] - prior["cumulative_tv"]) < 1e-10, case
        result["cases"][case] = output
        print(case, "online ratio", round(output["online_ratio"], 3),
              "population ratio", round(output["population_ratio"], 3), flush=True)
    moving = ("overmass", "rare", "shift", "gauge")
    wins = sum(result["cases"][case]["online_ratio"] < 1 for case in moving)
    result["optimistic_go"] = (
        wins >= 3 and result["cases"]["rare"]["online_ratio"] < 1
        and result["cases"]["null"]["online_ratio"] <= 1)
    (ROOT / "lag_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("optimistic_go", result["optimistic_go"], flush=True)


if __name__ == "__main__":
    main()
