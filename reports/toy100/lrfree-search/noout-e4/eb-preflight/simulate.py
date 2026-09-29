#!/usr/bin/env python3
"""Run the fixed synthetic design in SPEC.md; no benchmark inputs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


N_ROWS = 20_000
SEEDS = (20260929, 20260930, 20261001)
OUT = Path(__file__).with_name("results.json")


def fit_gain(x: np.ndarray) -> tuple[np.ndarray, float, float, int, bool]:
    """Moment-started EM for the specified unit-null Gaussian scale mixture."""
    m2 = float(np.mean(x * x))
    m4 = float(np.mean(x**4))
    u = max(m2 - 1.0, 0.0)
    v = max(m4 / 3.0 - 1.0 - 2.0 * u, 0.0)
    if u <= 1e-12:
        return np.zeros_like(x), 0.0, 0.0, 0, True
    a = max(v / u, 1e-9)
    pi = float(np.clip(u / a, 1e-9, 1.0 - 1e-9))

    converged = False
    for iteration in range(1, 501):
        log_bf = -0.5 * np.log1p(a) + 0.5 * (a / (1.0 + a)) * x * x
        log_odds = np.log(pi) - np.log1p(-pi) + log_bf
        # The log odds can be large for t5 innovations.
        response = np.exp(-np.logaddexp(0.0, -log_odds))
        pi_new = float(np.clip(np.mean(response), 1e-9, 1.0 - 1e-9))
        a_new = max(float(np.sum(response * x * x) / np.sum(response) - 1.0), 1e-9)
        if max(abs(pi_new - pi) / max(pi, 1e-9),
               abs(a_new - a) / max(a, 1e-9)) < 1e-8:
            pi, a = pi_new, a_new
            converged = True
            break
        pi, a = pi_new, a_new

    log_bf = -0.5 * np.log1p(a) + 0.5 * (a / (1.0 + a)) * x * x
    log_odds = np.log(pi) - np.log1p(-pi) + log_bf
    posterior = np.exp(-np.logaddexp(0.0, -log_odds))
    return posterior * a / (1.0 + a), pi, a, iteration, converged


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
                x = rng.standard_t(df=5, size=N_ROWS) * np.sqrt(3.0 / 5.0)
            x[:n_signal] += signs * delta * np.sqrt(info)
            gain, pi_hat, a_hat, iterations, converged = fit_gain(x)
            null_gain = float(np.mean(gain[null]))
            drift_gain = float(np.mean(gain[~null])) if n_signal else None
            checks = [null_gain <= 0.05, converged]
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
                "pi_hat": pi_hat, "a_hat": a_hat,
                "em_iterations": iterations, "em_converged": converged,
                "pass": all(checks),
            }
            cases.append(case)
            print(
                f"seed={seed} {noise:6} pi={pi:.2f} d={delta:.1f} "
                f"m={m:3d} rho={rho:.1f} null={null_gain:.4f} "
                f"drift={drift_gain if drift_gain is not None else 'n/a'} "
                f"p99={case['null_p99_gain']:.4f} "
                f"pi_hat={pi_hat:.4f} a_hat={a_hat:.3g} "
                f"em={iterations:3d} {'PASS' if all(checks) else 'FAIL'}",
                flush=True,
            )
    result = {"design": "SPEC.md", "n_rows": N_ROWS, "fixed_seeds": SEEDS,
              "passed": all(c["pass"] for c in cases), "cases": cases}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"OVERALL: {'PASS' if result['passed'] else 'FAIL'} "
          f"({sum(c['pass'] for c in cases)}/{len(cases)} cases); {OUT}")


if __name__ == "__main__":
    run()
