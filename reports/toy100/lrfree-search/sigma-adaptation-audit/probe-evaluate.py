"""Read-only held-out log-output-sigma derivative at the frozen grid100 state.

The characteristic Gaussian MMD bandwidth is fixed by the median 10th other
neighbor distance in an independent real calibration draw.  No accuracy-gate
quantity or evaluator geometry enters either objective.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/h2-prior-couple-sigma/grid100"
PACKAGE = ROOT / "candidate-prior-couple-sigma/package"
HOST = Path("/ml2/hypergan/lrfree-20260926/harness/hosts")
sys.path[:0] = [str(PACKAGE), str(HOST)]
from native100.problems import sample_real  # noqa: E402
from native100.toy_models import SimpleMLPDiscriminator  # noqa: E402
from particlegan.gan_loss import GANLoss  # noqa: E402

B = 8192
N_BATCHES = 12
CALIBRATION_SEED = 74721
CONTROL_SEED = 74722
K_OTHER = 10
RADIUS_IN_BANDWIDTHS = 7.0


def _pair_terms(x: np.ndarray, eps: np.ndarray, y: np.ndarray, h: float):
    """Sparse exact-within-7h Gaussian U-statistic and analytic derivative.

    Returned derivative is with respect to log(sigma), where x=clean+sigma*eps.
    Outside 7h, each omitted kernel is <= exp(-49/2)=2.29e-11.
    """
    n = len(x)
    radius = RADIUS_IN_BANDWIDTHS * h
    sigma = CURRENT_SIGMA
    xx = cKDTree(x).query_pairs(radius, output_type="ndarray")
    i, j = xx[:, 0], xx[:, 1]
    dx = x[i] - x[j]
    de = eps[i] - eps[j]
    k = np.exp(-np.einsum("ij,ij->i", dx, dx) / (2 * h * h))
    u_xx = 2 * k.sum() / (n * (n - 1))
    g_xx = -2 * sigma * np.sum(k * np.einsum("ij,ij->i", dx, de)) / (h * h * n * (n - 1))

    xy = cKDTree(x).sparse_distance_matrix(cKDTree(y), radius, output_type="ndarray")
    i, j = xy["i"], xy["j"]
    dx = x[i] - y[j]
    k = np.exp(-np.einsum("ij,ij->i", dx, dx) / (2 * h * h))
    u_xy = k.sum() / (n * n)
    g_xy = -sigma * np.sum(k * np.einsum("ij,ij->i", dx, eps[i])) / (h * h * n * n)

    yy = cKDTree(y).query_pairs(radius, output_type="ndarray")
    i, j = yy[:, 0], yy[:, 1]
    dy = y[i] - y[j]
    ky = np.exp(-np.einsum("ij,ij->i", dy, dy) / (2 * h * h))
    u_yy = 2 * ky.sum() / (n * (n - 1))
    return dict(mmd2=u_xx + u_yy - 2 * u_xy, grad_logsigma=g_xx - 2 * g_xy,
                u_xx=u_xx, u_yy=u_yy, u_xy=u_xy,
                xx_pairs=len(xx), xy_pairs=len(xy), yy_pairs=len(yy))


def _gan_gradient(critic, clean, eps, real, sigma, device):
    model = GANLoss()
    total_n = len(clean)
    total_loss = total_grad = 0.0
    for start in range(0, total_n, 1024):
        sl = slice(start, min(start + 1024, total_n))
        c = torch.as_tensor(clean[sl], dtype=torch.float32, device=device)
        e = torch.as_tensor(eps[sl], dtype=torch.float32, device=device)
        y = torch.as_tensor(real[sl], dtype=torch.float32, device=device)
        log_s = torch.tensor(math.log(sigma), device=device, requires_grad=True)
        loss = model.g_loss(critic(c + log_s.exp() * e), critic(y))
        grad = torch.autograd.grad(loss, log_s)[0]
        weight = len(c) / total_n
        total_loss += weight * float(loss.detach())
        total_grad += weight * float(grad.detach())
    return dict(loss=total_loss, grad_logsigma=total_grad)


def _summary(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    # t_(11,0.975)=2.200985; independent nonoverlapping held-out batches.
    mean = float(values.mean())
    half = float(2.200985 * values.std(ddof=1) / math.sqrt(len(values)))
    return dict(mean=mean, ci95_t=[mean - half, mean + half],
                negative_batches=int(np.sum(values < 0)), positive_batches=int(np.sum(values > 0)),
                sd_between_batches=float(values.std(ddof=1)))


def main():
    global CURRENT_SIGMA
    started = time.monotonic()
    state = torch.load(RUN / "final-state.pt", map_location="cpu", weights_only=False)["trainer"]
    assert state["completed_steps"] == 7000
    CURRENT_SIGMA = float(state["output_noise"]["log_sigma"].exp())
    with np.load(RUN / "native-clean/holdout_samples.npz") as z:
        clean = z["live"].astype(np.float64)
        real = z["target"].astype(np.float64)
    with np.load(RUN / "native-noisy/holdout_samples.npz") as z:
        noisy = z["live"].astype(np.float64)
        assert np.array_equal(real, z["target"].astype(np.float64))
    assert len(clean) == len(noisy) == len(real) == 100000
    eps = (noisy - clean) / CURRENT_SIGMA
    calibration = sample_real("grid100", B, generator=torch.Generator().manual_seed(CALIBRATION_SEED)).numpy()
    kth, _ = cKDTree(calibration).query(calibration, k=K_OTHER + 1, workers=4)
    h = float(np.median(kth[:, K_OTHER]))
    assert h > 0
    control_clean = sample_real("grid100", N_BATCHES * B,
                                generator=torch.Generator().manual_seed(CONTROL_SEED)).numpy().astype(np.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    critic = SimpleMLPDiscriminator(in_dim=2, hidden_dim=128, n_hidden=3, fourier=3).to(device)
    critic.load_state_dict(state["models"]["D"])
    critic.eval().requires_grad_(False)
    rows = []
    controls = []
    for batch in range(N_BATCHES):
        sl = slice(batch * B, (batch + 1) * B)
        q = _pair_terms(noisy[sl], eps[sl], real[sl], h)
        q["gan"] = _gan_gradient(critic, clean[sl], eps[sl], real[sl], CURRENT_SIGMA, device)
        q["batch"] = batch
        rows.append(q)
        xc = control_clean[sl] + CURRENT_SIGMA * eps[sl]
        control = _pair_terms(xc, eps[sl], real[sl], h)
        control["batch"] = batch
        controls.append(control)
    # Coupled-noise central difference against the analytic derivative on one
    # batch. It is only a numerical derivative check, not a bandwidth sweep.
    sl = slice(0, B)
    delta = 1e-3
    def at_log_shift(shift):
        scaled = clean[sl] + CURRENT_SIGMA * math.exp(shift) * eps[sl]
        # _pair_terms differentiates around CURRENT_SIGMA, but its mmd2 is
        # valid for any supplied cloud; only use its value here.
        return _pair_terms(scaled, eps[sl], real[sl], h)["mmd2"]
    fd = (at_log_shift(delta) - at_log_shift(-delta)) / (2 * delta)
    report = {
        "run": str(RUN), "checkpoint_step": 7000,
        "sigma": CURRENT_SIGMA, "batch_size": B, "nonoverlapping_batches": N_BATCHES,
        "generated_and_real_pool": "frozen independent 100k holdout; live clean/noisy paired noise; target seed independent",
        "bandwidth": {"definition": "median 10th other-neighbor distance in 8192 independent real draws",
                      "seed": CALIBRATION_SEED, "value": h},
        "kernel": "exp(-||x-y||^2/(2*h^2))",
        "mmd2": "U_XX + U_YY - 2 V_XY, U excludes diagonal; gradient wrt log applied sigma",
        "truncation_radius": RADIUS_IN_BANDWIDTHS * h,
        "max_omitted_kernel": math.exp(-RADIUS_IN_BANDWIDTHS ** 2 / 2),
        "mmd_gradient": _summary(rows, "grad_logsigma"),
        "gan_gradient": _summary([r["gan"] for r in rows], "grad_logsigma"),
        "control_real_plus_noise_mmd_gradient": _summary(controls, "grad_logsigma"),
        "finite_difference": {"batch": 0, "delta_logsigma": delta,
                              "numeric": fd, "analytic": rows[0]["grad_logsigma"]},
        "eps_mean": np.mean(eps, axis=0).tolist(),
        "eps_std": np.std(eps, axis=0).tolist(),
        "seconds": time.monotonic() - started,
        "rows": rows, "control_rows": controls,
    }
    path = Path(__file__).with_name("result.json")
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("sigma", "bandwidth", "mmd_gradient", "gan_gradient",
                       "control_real_plus_noise_mmd_gradient", "finite_difference", "seconds")},
                     indent=2))


if __name__ == "__main__":
    main()
