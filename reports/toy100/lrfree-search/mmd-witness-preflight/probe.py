"""Read-only, task-agnostic Gaussian MMD witness at frozen native100 states.

The objective uses only unlabeled real draws and generated samples. The
Gaussian output noise is integrated exactly; no evaluator centers, mode labels,
scores, or task-specific schedule enter the witness or proposed direction.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "runs/h2-handoff-critic-floor"
PACKAGE = ROOT / "candidate-prior-handoff-critic-floor/package"
HOST = Path("/ml2/hypergan/lrfree-20260926/harness/hosts")
sys.path[:0] = [str(PACKAGE), str(HOST)]

from native100.problems import sample_real  # noqa: E402
from particlegan.particle_prior import ParticlePrior  # noqa: E402

TASKS = ("grid100", "rotated100", "staggered100")
B = 2048
BLOCKS = 8
CALIBRATION_N = 8192
NEIGHBOR_K = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_stream(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def kernel(a, b, width_sq):
    # Explicit distances avoid small relative errors from subtracting large
    # dot products at a kernel width of about 0.02 on coordinates near +/-6.
    diff = a[:, None, :] - b[None, :, :]
    return torch.exp(-diff.square().sum(-1) / (2 * width_sq))


def gaussian_mmd_integrated(means, real, sigma, h):
    """Unbiased MMD2, integrating independent N(0,sigma^2 I) output draws.

    E k(mu_i + eps_i, mu_j + eps_j) has width h^2+2 sigma^2.
    E k(real_i, mu_j + eps_j) has width h^2+sigma^2.
    The q-q diagonal is removed because i and j denote independent sampled
    components in the population expectation, not a sample's self-pair.
    """
    n, d = means.shape
    h2 = h * h
    q_width = h2 + 2 * sigma.square()
    rq_width = h2 + sigma.square()
    qq_factor = (h2 / q_width).pow(d / 2)
    rq_factor = (h2 / rq_width).pow(d / 2)
    qq = kernel(means, means, q_width)
    rr = kernel(real, real, h2)
    rq = kernel(real, means, rq_width)
    qq_u = qq_factor * (qq.sum() - qq.diagonal().sum()) / (n * (n - 1))
    rr_u = (rr.sum() - rr.diagonal().sum()) / (n * (n - 1))
    rq_v = rq_factor * rq.mean()
    return qq_u + rr_u - 2 * rq_v, (qq_u, rr_u, rq_v)


@torch.no_grad()
def real_null(real_a, real_b, h):
    n = len(real_a)
    aa = kernel(real_a, real_a, h * h)
    bb = kernel(real_b, real_b, h * h)
    ab = kernel(real_a, real_b, h * h)
    return float((aa.sum()-aa.diagonal().sum())/(n*(n-1))
                 + (bb.sum()-bb.diagonal().sum())/(n*(n-1)) - 2*ab.mean())


@torch.no_grad()
def draw_latent(prior, idx, nearest_radius, latent_width, stream):
    # Matches the saved DV12 controller's clipped local Gaussian support.
    base = prior.z.detach()[idx]
    displacement = latent_width * torch.randn(base.shape, device=DEVICE, generator=stream)
    fraction = (nearest_radius[idx] / displacement.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)
    return displacement * fraction.unsqueeze(1)


def stats(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    tcrit = 2.364624251 if len(values) == 8 else 2.200985160
    mean = float(values.mean())
    half = float(tcrit * values.std(ddof=1) / math.sqrt(len(values)))
    return {"mean": mean, "ci95_t": [mean-half, mean+half],
            "negative_blocks": int((values < 0).sum()),
            "positive_blocks": int((values > 0).sum()),
            "sd_between_blocks": float(values.std(ddof=1))}


def one_task(task, task_id):
    started = time.monotonic()
    path = RUNS / task / "final-state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)["trainer"]
    assert state["completed_steps"] == 7000
    g = nn.Linear(2, 2).to(DEVICE)
    g.load_state_dict(state["models"]["G"])
    prior = ParticlePrior(20000, 2, device=DEVICE)
    prior.load_state_dict(state["models"]["prior"])
    log_sigma = state["output_noise"]["log_sigma"].detach().to(DEVICE).requires_grad_()
    sigma = float(log_sigma.exp().detach())
    width = state["controller"]["latent_bandwidth"].to(DEVICE)
    assert width.shape == (2,)
    prior_np = prior.z.detach().cpu().numpy().astype(np.float64)
    distances, indices = cKDTree(prior_np).query(prior_np, k=2, workers=4)
    assert np.array_equal(indices[:, 0], np.arange(len(prior_np)))
    nearest_radius = torch.as_tensor(distances[:, 1] * .5, device=DEVICE, dtype=prior.z.dtype)
    calibration_seed = 990100 + task_id * 10000
    calibration = sample_real(task, CALIBRATION_N,
                              generator=torch.Generator().manual_seed(calibration_seed)).numpy()
    distances_cal, _ = cKDTree(calibration).query(calibration, k=NEIGHBOR_K + 1, workers=4)
    h = float(np.median(distances_cal[:, NEIGHBOR_K]))
    assert h > 0
    lr_g = float(state["optimizers"][0]["param_groups"][0]["lr"])
    lr_prior = float(state["optimizers"][0]["param_groups"][1]["lr"])
    rows = []
    params = (g.weight, g.bias, prior.z, log_sigma)
    for block in range(BLOCKS):
        base_seed = 1900000 + task_id * 100000 + block * 100
        idx = torch.randint(len(prior.z), (B,), device=DEVICE,
                            generator=seed_stream(base_seed))
        contexts = []
        for split in ("fit", "heldout"):
            offset = 0 if split == "fit" else 20
            displacement = draw_latent(prior, idx, nearest_radius, width,
                                       seed_stream(base_seed + offset + 1))
            real = sample_real(task, B, device=DEVICE,
                               generator=seed_stream(base_seed + offset + 2))
            means = g(prior.z[idx] + displacement)
            loss, parts = gaussian_mmd_integrated(means, real, log_sigma.exp(), h)
            grads = torch.autograd.grad(loss, params)
            contexts.append((loss, parts, grads, real))
        (fit_loss, fit_parts, fit_grads, fit_real), (held_loss, held_parts, held_grads, held_real) = contexts
        null_real = sample_real(task, B, device=DEVICE,
                                generator=seed_stream(base_seed + 50))
        control = real_null(fit_real, null_real, h)
        dg = tuple(-lr_g * f.sign() for f in fit_grads[:2])
        dp = -lr_prior * fit_grads[2].sign()
        g_pred = float(sum((d * grad).sum() for d, grad in zip(dg, held_grads[:2])))
        p_pred = float((dp * held_grads[2]).sum())
        fit_s, held_s = float(fit_grads[3]), float(held_grads[3])
        # Sigma is frozen in this checkpoint; this is the unit-log-sigma
        # hypothetical descent direction, not an applied optimizer step.
        s_pred_unit = -math.copysign(1.0, fit_s) * held_s if fit_s else 0.0
        with torch.no_grad():
            stepped_means = torch.nn.functional.linear(
                prior.z[idx] + dp[idx] + displacement,
                g.weight + dg[0], g.bias + dg[1])
            stepped_loss, stepped_parts = gaussian_mmd_integrated(
                stepped_means, held_real, log_sigma.exp(), h)
            observed_delta = float(stepped_loss - held_loss)
            observed_qq_delta = float(stepped_parts[0] - held_parts[0])
            observed_real_q_delta = float(-2 * (stepped_parts[2] - held_parts[2]))
            if block == 0:
                eps = 1e-3
                positive, _ = gaussian_mmd_integrated(
                    means, held_real, (log_sigma + eps).exp(), h)
                negative, _ = gaussian_mmd_integrated(
                    means, held_real, (log_sigma - eps).exp(), h)
                sigma_central_diff = float((positive - negative) / (2 * eps))
            else:
                sigma_central_diff = None
        gf = torch.cat([v.detach().flatten() for v in fit_grads[:2]])
        gh = torch.cat([v.detach().flatten() for v in held_grads[:2]])
        g_cos = float(torch.nn.functional.cosine_similarity(gf, gh, dim=0))
        pf = fit_grads[2].detach()[idx.unique()]
        ph = held_grads[2].detach()[idx.unique()]
        p_cos = float(torch.nn.functional.cosine_similarity(pf.flatten(), ph.flatten(), dim=0))
        row = {"block": block, "fit_mmd2": float(fit_loss), "heldout_mmd2": float(held_loss),
               "real_real_null_mmd2": control,
               "heldout_mmd2_minus_real_null": float(held_loss) - control,
               "fit_terms": [float(x) for x in fit_parts],
               "heldout_terms": [float(x) for x in held_parts],
               "heldout_pred_delta_g_at_native_sign_lr": g_pred,
               "heldout_pred_delta_prior_at_native_sign_lr": p_pred,
               "heldout_pred_delta_g_plus_prior": g_pred + p_pred,
               "heldout_observed_delta_g_plus_prior": observed_delta,
               "heldout_observed_delta_q_self": observed_qq_delta,
               "heldout_observed_delta_real_q_attraction": observed_real_q_delta,
               "fit_grad_logsigma": fit_s, "heldout_grad_logsigma": held_s,
               "heldout_sigma_central_diff": sigma_central_diff,
               "heldout_pred_delta_sigma_per_unit_log_step": s_pred_unit,
               "g_fit_heldout_grad_cosine": g_cos,
               "prior_shared_rows_fit_heldout_grad_cosine": p_cos,
               "shared_prior_rows": int(idx.unique().numel())}
        rows.append(row)
        print(json.dumps({"task": task, "block": block, "held_mmd": row["heldout_mmd2"],
                          "g_delta": g_pred, "prior_delta": p_pred,
                          "sigma_grad": held_s, "seconds": time.monotonic()-started}), flush=True)
    keys = ("fit_mmd2", "heldout_mmd2", "real_real_null_mmd2",
            "heldout_mmd2_minus_real_null",
            "heldout_pred_delta_g_at_native_sign_lr",
            "heldout_pred_delta_prior_at_native_sign_lr",
            "heldout_pred_delta_g_plus_prior", "heldout_observed_delta_g_plus_prior",
            "heldout_observed_delta_q_self", "heldout_observed_delta_real_q_attraction",
            "fit_grad_logsigma",
            "heldout_grad_logsigma", "heldout_pred_delta_sigma_per_unit_log_step",
            "g_fit_heldout_grad_cosine", "prior_shared_rows_fit_heldout_grad_cosine")
    return {"task": task, "checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "step": state["completed_steps"], "sigma": sigma, "latent_width": width.cpu().tolist(),
            "calibration_seed": calibration_seed, "bandwidth": h, "native_lr_g": lr_g,
            "native_lr_prior": lr_prior, "rows": rows,
            "summary": {key: stats(rows, key) for key in keys}, "seconds": time.monotonic()-started}


def main():
    torch.set_num_threads(2)
    start = time.monotonic()
    reports = []
    for task_id, task in enumerate(TASKS):
        report = one_task(task, task_id)
        reports.append(report)
        (HERE / f"{task}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    all_report = {"method": "Gaussian-kernel unbiased MMD2; analytically integrated Gaussian output noise; independent real fit/heldout, independent latent jitter, shared sampled prior rows per block; exact native G/prior sign-LRs for directional predictions; sigma frozen in checkpoint",
                  "bandwidth": "median 10th other-neighbor distance from 8192 independent unlabeled real draws per task",
                  "batch_size": B, "blocks_per_task": BLOCKS, "tasks": reports,
                  "seconds": time.monotonic()-start, "diagnostic_only": True}
    (HERE / "result.json").write_text(json.dumps(all_report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"seconds": all_report["seconds"],
                      "summary": {r["task"]: r["summary"] for r in reports}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
