#!/usr/bin/env python3
"""Evaluate the fixed signed-location-mixture design in SPEC.md."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np


N_ROWS = 20_000
SEEDS = (20261002, 20261003, 20261004)
OUT = Path(__file__).with_name("results.json")
T_SCALE = math.sqrt(3.0 / 5.0)
T_LOG_NORM = (math.lgamma(3.0) - math.lgamma(2.5)
              - 0.5 * math.log(5.0 * math.pi) - math.log(T_SCALE))


def bounded_minimize(fun, lo: float, hi: float, atol: float,
                     max_evaluations: int = 100) -> tuple[float, float, bool]:
    """Golden-section scalar minimization; endpoints are compared by caller."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = hi - ratio * (hi - lo)
    right = lo + ratio * (hi - lo)
    f_left, f_right = fun(left), fun(right)
    evaluations = 2
    while hi - lo > atol and evaluations < max_evaluations:
        if f_left <= f_right:
            hi, right, f_right = right, left, f_left
            left = hi - ratio * (hi - lo)
            f_left = fun(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + ratio * (hi - lo)
            f_right = fun(right)
        evaluations += 1
    best = min((f_left, left), (f_right, right))
    return best[0], best[1], hi - lo <= atol


def log_density(x: np.ndarray, noise: str) -> np.ndarray:
    if noise == "normal":
        return -0.5 * (x * x + math.log(2.0 * math.pi))
    return T_LOG_NORM - 3.0 * np.log1p((x / T_SCALE) ** 2 / 5.0)


def fit_location(x: np.ndarray, noise: str) -> tuple[np.ndarray, float, float, int]:
    log_f0 = log_density(x, noise)
    eval_count = 0

    def profile(mu: float) -> tuple[float, float]:
        nonlocal eval_count
        if mu == 0.0:
            return 0.0, 0.0
        log_f1 = np.logaddexp(log_density(x - mu, noise),
                             log_density(x + mu, noise)) - math.log(2.0)
        log_ratio = log_f1 - log_f0

        def objective(pi: float) -> float:
            nonlocal eval_count
            eval_count += 1
            if pi == 0.0:
                return 0.0
            if pi == 1.0:
                return -float(np.sum(log_ratio))
            value = -float(np.sum(np.logaddexp(math.log1p(-pi),
                                                 math.log(pi) + log_ratio)))
            if not math.isfinite(value):
                raise ArithmeticError("nonfinite profile likelihood")
            return value

        opt_nll, opt_pi, success = bounded_minimize(
            objective, 0.0, 1.0, 1e-6)
        if not success:
            raise ArithmeticError("pi optimizer failed")
        return min((objective(0.0), 0.0), (objective(1.0), 1.0),
                   (opt_nll, opt_pi))

    max_abs_x = float(np.max(np.abs(x)))
    mu_grid = np.linspace(0.0, max_abs_x, 33)
    fits = [(nll, float(mu), pi) for mu in mu_grid
            for nll, pi in [profile(float(mu))]]
    for i in range(1, len(mu_grid) - 1):
        if fits[i][0] < fits[i - 1][0] and fits[i][0] < fits[i + 1][0]:
            opt_nll, opt_mu, success = bounded_minimize(
                lambda mu: profile(mu)[0],
                float(mu_grid[i - 1]), float(mu_grid[i + 1]),
                1e-5 * max(1.0, max_abs_x))
            if not success:
                raise ArithmeticError("mu optimizer failed")
            nll, pi = profile(opt_mu)
            fits.append((nll, opt_mu, pi))
    nll, mu_hat, pi_hat = min(fits)
    if not math.isfinite(nll):
        raise ArithmeticError("nonfinite maximum likelihood")
    if mu_hat == 0.0 or pi_hat == 0.0:
        return np.zeros_like(x), pi_hat, mu_hat, eval_count
    log_f1 = np.logaddexp(log_density(x - mu_hat, noise),
                         log_density(x + mu_hat, noise)) - math.log(2.0)
    log_signal = math.log(pi_hat) + log_f1
    log_null = (math.log1p(-pi_hat) + log_f0
                if pi_hat < 1.0 else np.full_like(x, -np.inf))
    posterior = np.exp(log_signal - np.logaddexp(log_null, log_signal))
    gain = posterior * mu_hat**2 / (1.0 + mu_hat**2)
    return gain, pi_hat, mu_hat, eval_count


def run() -> None:
    cases = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        grid = list(itertools.product(
            ("normal", "t5"), (0.01, 0.05), (0.5, 1.0), (128, 700),
            (0.0, 0.5, 0.9)
        ))
        grid.extend((noise, 0.0, 0.0, 0, 0.0) for noise in ("normal", "t5"))
        for noise, pi, delta, m, rho in grid:
            n_signal = round(N_ROWS * pi)
            null = np.zeros(N_ROWS, dtype=bool)
            null[n_signal:] = True
            info = ((m * (1.0 - rho) + 2.0 * rho) / (1.0 + rho)
                    if n_signal else 0.0)
            signs = rng.choice((-1.0, 1.0), size=n_signal)
            if noise == "normal":
                x = rng.normal(size=N_ROWS)
            else:
                x = rng.standard_t(df=5, size=N_ROWS) * T_SCALE
            x[:n_signal] += signs * delta * math.sqrt(info)
            try:
                gain, pi_hat, mu_hat, evaluations = fit_location(x, noise)
                converged = True
            except ArithmeticError:
                gain = np.full(N_ROWS, np.nan)
                pi_hat = mu_hat = float("nan")
                evaluations = 0
                converged = False
            null_gain = float(np.mean(gain[null]))
            drift_gain = float(np.mean(gain[~null])) if n_signal else None
            checks = [converged, null_gain <= 0.05]
            if noise == "normal" and rho <= 0.5 and n_signal:
                if m == 128 and delta == 1.0:
                    checks.append(drift_gain >= 0.5)
                if m == 700 and delta == 0.5:
                    checks.append(drift_gain >= 0.5)
            case = {
                "seed": seed, "noise": noise, "pi": pi, "delta": delta,
                "m": m, "rho": rho, "information": info,
                "null_mean_gain": null_gain, "drift_mean_gain": drift_gain,
                "null_median_gain": float(np.quantile(gain[null], 0.5)),
                "null_p95_gain": float(np.quantile(gain[null], 0.95)),
                "null_p99_gain": float(np.quantile(gain[null], 0.99)),
                "null_fraction_above_half": float(np.mean(gain[null] > 0.5)),
                "pi_hat": pi_hat, "mu_hat": mu_hat,
                "likelihood_evaluations": evaluations,
                "fit_converged": converged, "pass": all(checks),
            }
            cases.append(case)
            print(
                f"seed={seed} {noise:6} pi={pi:.2f} d={delta:.1f} "
                f"m={m:3d} rho={rho:.1f} null={null_gain:.4f} "
                f"drift={drift_gain if drift_gain is not None else 'n/a'} "
                f"p99={case['null_p99_gain']:.4f} "
                f"pi_hat={pi_hat:.4f} mu_hat={mu_hat:.3g} "
                f"eval={evaluations:4d} {'PASS' if all(checks) else 'FAIL'}",
                flush=True,
            )
    result = {"design": "SPEC.md", "n_rows": N_ROWS, "fixed_seeds": SEEDS,
              "passed": all(c["pass"] for c in cases), "cases": cases}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"OVERALL: {'PASS' if result['passed'] else 'FAIL'} "
          f"({sum(c['pass'] for c in cases)}/{len(cases)} cases); {OUT}")


if __name__ == "__main__":
    run()
