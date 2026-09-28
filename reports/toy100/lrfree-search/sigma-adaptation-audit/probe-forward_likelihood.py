"""Read-only empirical forward likelihood width gradient, frozen generator.

q_sigma(y) = M^-1 sum_i N(y | clean_i, sigma^2 I_2), where the independent
clean_i are exact draws of the saved particle prior + DV12 latent perturbation
+ affine generator. A nearest-component-relative cutoff at log weight -50
omits at most M*exp(-50) of the nearest component's mass, about 1.93e-17.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "runs/h2-prior-couple-sigma/grid100"
SIGMAS = (0.017, 0.019)
N_BATCHES = 12
B = 8192
LOG_WEIGHT_CUTOFF = 50.0


def _summary(rows, key):
    a = np.asarray([r[key] for r in rows], dtype=np.float64)
    mean = float(a.mean())
    half = float(2.200985 * a.std(ddof=1) / math.sqrt(len(a)))
    return dict(mean=mean, ci95_t=[mean-half, mean+half],
                negative_batches=int((a < 0).sum()), positive_batches=int((a > 0).sum()),
                sd_between_batches=float(a.std(ddof=1)))


def main():
    t0 = time.monotonic()
    with np.load(RUN / "native-clean/holdout_samples.npz") as archive:
        clean = archive["live"].astype(np.float64)
        real = archive["target"].astype(np.float64)
    assert len(clean) == len(real) == 100000
    tree = cKDTree(clean)
    outputs = {sigma: [] for sigma in SIGMAS}
    for batch in range(N_BATCHES):
        begin = time.monotonic()
        y = real[batch * B:(batch+1) * B]
        r_near, _ = tree.query(y, k=1, workers=4)
        # One component set for both sigmas. Every omitted kernel weight is
        # below exp(-50) relative to each target's nearest included component.
        radius = np.sqrt(r_near**2 + 2 * LOG_WEIGHT_CUTOFF * max(SIGMAS)**2)
        neighbors = tree.query_ball_point(y, r=radius, workers=4)
        sizes = np.fromiter((len(row) for row in neighbors), dtype=np.int64, count=B)
        assert np.all(sizes > 0)
        i = np.repeat(np.arange(B, dtype=np.int32), sizes)
        j = np.concatenate(neighbors).astype(np.int32, copy=False)
        delta = clean[j] - y[i]
        d2 = np.einsum("ij,ij->i", delta, delta)
        nearest2 = r_near**2
        for sigma in SIGMAS:
            sigma2 = sigma * sigma
            w = np.exp(-(d2 - nearest2[i]) / (2 * sigma2))
            sum_w = np.bincount(i, weights=w, minlength=B)
            sum_wr2 = np.bincount(i, weights=w*d2, minlength=B)
            mean_r2 = sum_wr2 / sum_w
            # L = -E_real log q_sigma(real): 2D Gaussian normalization included.
            nll = (math.log(len(clean)) + math.log(2*math.pi*sigma2)
                   + nearest2/(2*sigma2) - np.log(sum_w))
            grad = 2 - mean_r2/sigma2
            outputs[sigma].append(dict(batch=batch, nll=float(nll.mean()),
                                       grad_logsigma=float(grad.mean()),
                                       grad_se_within_batch=float(grad.std(ddof=1)/math.sqrt(B)),
                                       pairs=int(len(i))))
        print(f"batch={batch} pairs={len(i)} seconds={time.monotonic()-begin:.3f}", flush=True)
    summary = []
    for sigma in SIGMAS:
        rows = outputs[sigma]
        summary.append(dict(sigma=sigma, nll=_summary(rows, "nll"),
                            grad_logsigma=_summary(rows, "grad_logsigma"), rows=rows))
    report = dict(method="empirical clean-mixture forward NLL with bounded tail",
                  formula="L=-mean_y log[(1/M)sum_i N(y|c_i,sigma^2 I2)]; dL/dlog sigma=mean_y[2-E_{i|y}(||y-c_i||^2)/sigma^2]",
                  clean_pool_size=len(clean), real_pool_size=N_BATCHES*B,
                  independent_clean_and_real=True, clean_law="saved prior + DV12 latent jitter + affine generator",
                  cutoff_log_relative_weight=LOG_WEIGHT_CUTOFF,
                  max_omitted_mass_relative_to_nearest=float(len(clean)*math.exp(-LOG_WEIGHT_CUTOFF)),
                  width_results=summary, seconds=time.monotonic()-t0,
                  diagnostic_only=True)
    (HERE / "forward_likelihood.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"width_results": [{k:v for k,v in r.items() if k!="rows"} for r in summary],
                      "seconds": report["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
