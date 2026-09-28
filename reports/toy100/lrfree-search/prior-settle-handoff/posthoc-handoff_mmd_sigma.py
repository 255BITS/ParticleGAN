"""Read-only local-bandwidth MMD sigma gradient on H2 handoff checkpoints."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

import evaluate as previous
from native100.problems import sample_real

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs/h2-prior-handoff"
TASKS = ("grid100", "staggered100")
B = 8192
N_BATCHES = 12
BASE_SIGMA = 0.02
BRACKET_SIGMA = 0.022
CALIBRATION_SEEDS = {"grid100": 74721, "staggered100": 74821}


def summary(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    mean = float(values.mean())
    half = float(2.200985 * values.std(ddof=1) / math.sqrt(len(values)))
    return dict(mean=mean, ci95_t=[mean-half, mean+half],
                negative_batches=int((values < 0).sum()),
                positive_batches=int((values > 0).sum()),
                sd_between_batches=float(values.std(ddof=1)))


def gradient_rows(clean, eps, real, h, sigma):
    previous.CURRENT_SIGMA = sigma
    rows = []
    for batch in range(N_BATCHES):
        sl = slice(batch * B, (batch+1) * B)
        x = clean[sl] + sigma * eps[sl]
        row = previous._pair_terms(x, eps[sl], real[sl], h)
        row["batch"] = batch
        rows.append(row)
    return rows


def main():
    started = time.monotonic()
    report = []
    for task in TASKS:
        t0 = time.monotonic()
        run = RUNS / task
        with np.load(run / "native-clean/holdout_samples.npz") as a:
            clean = a["live"].astype(np.float64)
            real = a["target"].astype(np.float64)
        with np.load(run / "native-noisy/holdout_samples.npz") as a:
            noisy = a["live"].astype(np.float64)
            assert np.array_equal(real, a["target"].astype(np.float64))
        assert len(clean) == len(noisy) == len(real) == 100000
        eps = (noisy-clean)/BASE_SIGMA
        calibration = sample_real(task, B,
                                  generator=torch.Generator().manual_seed(CALIBRATION_SEEDS[task])).numpy()
        kth, _ = cKDTree(calibration).query(calibration, k=11, workers=4)
        h = float(np.median(kth[:, 10]))
        base_rows = gradient_rows(clean, eps, real, h, BASE_SIGMA)
        base_summary = summary(base_rows, "grad_logsigma")
        widths = [dict(sigma=BASE_SIGMA, mmd2=summary(base_rows, "mmd2"),
                       grad_logsigma=base_summary, rows=base_rows)]
        if base_summary["ci95_t"][1] < 0:
            bracket_rows = gradient_rows(clean, eps, real, h, BRACKET_SIGMA)
            widths.append(dict(sigma=BRACKET_SIGMA, mmd2=summary(bracket_rows, "mmd2"),
                               grad_logsigma=summary(bracket_rows, "grad_logsigma"),
                               rows=bracket_rows))
        report.append(dict(task=task, bandwidth=h, calibration_seed=CALIBRATION_SEEDS[task],
                           widths=widths, eps_mean=np.mean(eps,axis=0).tolist(),
                           eps_std=np.std(eps,axis=0).tolist(), seconds=time.monotonic()-t0))
        print(json.dumps({"task": task, "bandwidth": h,
                          "widths": [{k:v for k,v in w.items() if k!="rows"} for w in widths],
                          "seconds": time.monotonic()-t0}, indent=2), flush=True)
    result = dict(method="unbiased Gaussian MMD²; h=median 10th other-neighbor distance in independent real calibration draw",
                  kernel="exp(-||x-y||²/(2h²))", batch_size=B, nonoverlapping_batches=N_BATCHES,
                  generated_and_real_pool="independent saved 100k holdout, live generated clean/noisy paired noise",
                  base_sigma=BASE_SIGMA, bracket_sigma=BRACKET_SIGMA,
                  tasks=report, seconds=time.monotonic()-started,
                  diagnostic_only=True)
    (HERE / "handoff_mmd_sigma.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
