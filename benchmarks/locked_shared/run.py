"""Run measured toy outcomes: one row per locked_shared host toy."""

from __future__ import annotations

import math

import torch

from . import mode_hold, trajectory, two_pole


def experiments():
    """One row per host toy, each trained on its own configuration.

    Formulation and pairing variants (thinned_cap, stranger, nearest_stranger,
    cap_off, vanilla, fm_on) are not problem arms and were dropped; their
    measured results are recorded in reports/locked_shared.
    """
    return (
        ("two_pole", "locked_shared", lambda: two_pole.train()),
        ("trajectory", "locked_shared", lambda: trajectory.train()),
        ("ring", "locked_shared", lambda: mode_hold.train_mode_hold()),
    )


def run_all(*, log=True):
    torch.set_num_threads(1)
    rows = []
    for toy, arm, train in experiments():
        if log:
            print(f"start toy={toy} arm={arm}", flush=True)
        row = {"toy": toy, "arm": arm, **train()}
        # A numerical blow-up cannot accidentally win a comparison.
        if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
            raise ValueError(f"non-finite metrics for {toy}/{arm}: {row}")
        rows.append(row)
        if log:
            print(f"done toy={toy} arm={arm} verdict={row['verdict']} {metrics(row)}", flush=True)
    return rows


def metrics(row):
    if row["toy"] == "two_pole":
        return f"travel={row['mean_abs']:.6f}; median slope={row['grad_med']:.6f}"
    if row["toy"] == "trajectory":
        return f"identity MSE={row['identity_mse']:.6f}"
    return f"modes={row['modes']}/8; HQ={row['hq']:.2%}; effective modes={row['effective_modes']:.3f}"


def markdown(report):
    rows = report["rows"]
    passed = sum(r["verdict"] == "PASS" for r in rows)
    lines = ["# Locked shared: measured behavior", "",
             f"**locked_shared meets {passed}/{len(rows)} behavioral targets in this run.**", "",
             "Every row trains. Verdicts use measured outputs only; no configuration gates.", "",
             "| Toy | Measurement | Result |",
             "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['toy']} | {metrics(row)} | **{row['verdict']}** |")
    lines += ["", "Thresholds copied from the reference:", "",
              "- Two-pole cloud: mean absolute travel ≥ 0.30 **and** median critic slope ≤ 1.0 (80 steps).",
              "- Trajectory: same-seed identity MSE ≤ 0.02 (400 steps).",
              "- Ring: ≥ 7/8 modes and ≥ 90% of samples within 3σ of a center (1,200 steps). "
              "≤ 2 modes is FAIL; other results are INCONCLUSIVE. The ring row is its EMA generator.", "",
              "Each host trains on its own configuration; hosts migrated to `benchmarks.toy_runner` "
              "(currently the ring) take optimizers, loss, penalty, noise and EMA from their recipe. "
              "Seed 0, CPU, one thread. Budgets, seed and thresholds were not retuned to make this table pass.", "",
              "Rel-1e-6 parity with the pinned conceptmod source is a frozen record in `reports/locked_shared`; "
              "it is not rerun because recipe-built hosts no longer replay the original training loops.", "",
              "Source: [conceptmod 5571213](https://github.com/HyperGAN/conceptmod/tree/5571213f5e8e129cfda45c785c3f30aad9c1d8c9/conceptmod/toys); "
              "[ParticleGAN PR #36](https://github.com/255BITS/ParticleGAN/pull/36).", "",
              f"Runtime: Python {report['python']}; PyTorch {report['torch']}. Full precision metrics are in `results.json`.", "",
              "CPU toy results only; no GPU or downstream application transfer is claimed.", ""]
    return "\n".join(lines)
