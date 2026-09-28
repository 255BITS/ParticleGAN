"""Paired read-only control: full MMD versus q-q-only sample repulsion.

Both proposals use the same frozen G/prior checkpoint, sampled component
rows, real batches, latent jitter, kernel width, and applied G/prior LR scales.
The q-q-only proposal is rescaled separately in G and prior parameter groups
to match the corresponding full-MMD proposal's Euclidean norm exactly.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

import probe


def norm(tensors):
    return torch.sqrt(sum(t.detach().square().sum() for t in tensors))


def equal_norm_control(full, null):
    full_norm = norm(full)
    null_norm = norm(null)
    if null_norm == 0:
        raise ValueError("q-q-only proposal has zero group norm")
    scale = full_norm / null_norm
    matched = tuple(t * scale for t in null)
    assert torch.isclose(norm(matched), full_norm, rtol=1e-6, atol=1e-10)
    return matched, float(scale), float(full_norm)


def step_objective(g, prior, idx, displacement, real, log_sigma, h, dg, dp):
    means = torch.nn.functional.linear(prior.z[idx] + dp[idx] + displacement,
                                       g.weight + dg[0], g.bias + dg[1])
    return probe.gaussian_mmd_integrated(means, real, log_sigma.exp(), h)


def one_task(task, task_id):
    start = time.monotonic()
    state = torch.load(probe.RUNS / task / "final-state.pt", map_location="cpu",
                       weights_only=False)["trainer"]
    assert state["completed_steps"] == 7000
    g = nn.Linear(2, 2).to(probe.DEVICE)
    g.load_state_dict(state["models"]["G"])
    prior = probe.ParticlePrior(20000, 2, device=probe.DEVICE)
    prior.load_state_dict(state["models"]["prior"])
    log_sigma = state["output_noise"]["log_sigma"].detach().to(probe.DEVICE)
    latent_width = state["controller"]["latent_bandwidth"].to(probe.DEVICE)
    points = prior.z.detach().cpu().numpy().astype(np.float64)
    nearest, indices = cKDTree(points).query(points, k=2, workers=4)
    assert np.array_equal(indices[:, 0], np.arange(len(points)))
    nearest_radius = torch.as_tensor(nearest[:, 1] * .5, device=probe.DEVICE,
                                     dtype=prior.z.dtype)
    calibration_seed = 990100 + task_id * 10000
    calibration = probe.sample_real(task, probe.CALIBRATION_N,
                                    generator=torch.Generator().manual_seed(calibration_seed)).numpy()
    calibration_distances, _ = cKDTree(calibration).query(
        calibration, k=probe.NEIGHBOR_K + 1, workers=4)
    h = float(np.median(calibration_distances[:, probe.NEIGHBOR_K]))
    lr_g = float(state["optimizers"][0]["param_groups"][0]["lr"])
    lr_prior = float(state["optimizers"][0]["param_groups"][1]["lr"])
    assert lr_g > 0 and lr_prior > 0
    rows = []
    for block in range(probe.BLOCKS):
        base_seed = 1900000 + task_id * 100000 + block * 100
        idx = torch.randint(len(prior.z), (probe.B,), device=probe.DEVICE,
                            generator=probe.seed_stream(base_seed))
        fit_disp = probe.draw_latent(prior, idx, nearest_radius, latent_width,
                                     probe.seed_stream(base_seed + 1))
        fit_real = probe.sample_real(task, probe.B, device=probe.DEVICE,
                                     generator=probe.seed_stream(base_seed + 2))
        fit_means = g(prior.z[idx] + fit_disp)
        fit_loss, fit_terms = probe.gaussian_mmd_integrated(
            fit_means, fit_real, log_sigma.exp(), h)
        params = (g.weight, g.bias, prior.z)
        qq_grads = torch.autograd.grad(fit_terms[0], params, retain_graph=True)
        full_grads = torch.autograd.grad(fit_loss, params)
        # Translation leaves q-q distances invariant. Set the exact q-q bias
        # derivative to zero, avoiding arbitrary signs from float32 roundoff.
        qq_grads = (qq_grads[0], torch.zeros_like(qq_grads[1]), qq_grads[2])
        full_g = tuple(-lr_g * x.sign() for x in full_grads[:2])
        full_p = (-lr_prior * full_grads[2].sign(),)
        qq_g_raw = tuple(-lr_g * x.sign() for x in qq_grads[:2])
        qq_p_raw = (-lr_prior * qq_grads[2].sign(),)
        qq_g, g_scale, g_norm = equal_norm_control(full_g, qq_g_raw)
        qq_p, p_scale, p_norm = equal_norm_control(full_p, qq_p_raw)
        held_disp = probe.draw_latent(prior, idx, nearest_radius, latent_width,
                                      probe.seed_stream(base_seed + 21))
        held_real = probe.sample_real(task, probe.B, device=probe.DEVICE,
                                      generator=probe.seed_stream(base_seed + 22))
        zero_g = (torch.zeros_like(g.weight), torch.zeros_like(g.bias))
        zero_p = torch.zeros_like(prior.z)
        with torch.no_grad():
            baseline, base_terms = step_objective(g, prior, idx, held_disp,
                                                   held_real, log_sigma, h,
                                                   zero_g, zero_p)
            full, full_terms = step_objective(g, prior, idx, held_disp,
                                              held_real, log_sigma, h,
                                              full_g, full_p[0])
            qq, qq_terms = step_objective(g, prior, idx, held_disp,
                                          held_real, log_sigma, h,
                                          qq_g, qq_p[0])
        full_delta = float(full - baseline)
        qq_delta = float(qq - baseline)
        row = {
            "block": block, "baseline_heldout_mmd2": float(baseline),
            "full_delta_total_mmd2": full_delta,
            "qq_only_delta_total_mmd2": qq_delta,
            "full_minus_qq_only_delta": full_delta - qq_delta,
            "full_delta_qq_term": float(full_terms[0] - base_terms[0]),
            "full_delta_real_q_term": float(-2 * (full_terms[2] - base_terms[2])),
            "qq_only_delta_qq_term": float(qq_terms[0] - base_terms[0]),
            "qq_only_delta_real_q_term": float(-2 * (qq_terms[2] - base_terms[2])),
            "full_g_norm": g_norm, "full_prior_norm": p_norm,
            "qq_g_norm": float(norm(qq_g)), "qq_prior_norm": float(norm(qq_p)),
            "qq_g_norm_scale": g_scale, "qq_prior_norm_scale": p_scale,
            "g_proposal_cosine": float(torch.nn.functional.cosine_similarity(
                torch.cat([x.flatten() for x in full_g]),
                torch.cat([x.flatten() for x in qq_g]), dim=0)),
            "prior_proposal_cosine": float(torch.nn.functional.cosine_similarity(
                full_p[0].flatten(), qq_p[0].flatten(), dim=0)),
            "shared_prior_rows": int(idx.unique().numel()),
        }
        rows.append(row)
        print(json.dumps({"task": task, "block": block,
                          "full_delta": full_delta, "qq_delta": qq_delta,
                          "advantage": full_delta-qq_delta,
                          "seconds": time.monotonic()-start}), flush=True)
    keys = ("baseline_heldout_mmd2", "full_delta_total_mmd2",
            "qq_only_delta_total_mmd2", "full_minus_qq_only_delta",
            "full_delta_qq_term", "full_delta_real_q_term",
            "qq_only_delta_qq_term", "qq_only_delta_real_q_term",
            "g_proposal_cosine", "prior_proposal_cosine")
    return {"task": task, "checkpoint_step": 7000, "bandwidth": h,
            "sigma": float(log_sigma.exp()), "native_lr_g": lr_g,
            "native_lr_prior": lr_prior, "rows": rows,
            "summary": {key: probe.stats(rows, key) for key in keys},
            "seconds": time.monotonic()-start}


def main():
    torch.set_num_threads(2)
    start = time.monotonic()
    tasks = [one_task(task, task_id) for task_id, task in enumerate(probe.TASKS)]
    result = {
        "method": "Paired fixed-checkpoint full-MMD versus q-q-only sign proposals; G and prior proposal L2 norms separately matched; identical sampled rows, independent heldout real and DV12 jitter; total integrated-noise MMD2 evaluated after each proposal",
        "batch_size": probe.B, "blocks_per_task": probe.BLOCKS,
        "full_informed_advantage_sign": "negative full_minus_qq_only_delta favors the real-informed full MMD proposal",
        "tasks": tasks, "seconds": time.monotonic()-start,
        "diagnostic_only": True,
    }
    path = probe.HERE / "attribution_result.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"seconds": result["seconds"],
                      "advantage": {t["task"]: t["summary"]["full_minus_qq_only_delta"] for t in tasks}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
