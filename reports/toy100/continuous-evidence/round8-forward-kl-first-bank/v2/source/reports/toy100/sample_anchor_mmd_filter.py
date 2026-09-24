"""One-bank, free-output noise-aware MMD falsifier for sampled anchors.

The emitted law is the equal-weight Gaussian mixture centered at clean
particles with the frozen late output sigma .029. Every MMD term involving
that law integrates output noise analytically. No model, GAN, or optimizer is
updated. The local L-BFGS warm diagnosis is bounded to 20 iterations, with
one data-derived fixed kernel width and no parameter or seed sweep.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.optimize import minimize
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.locked_shared import mode_hold
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.sample_anchor_free1200 import initial_support, load_states, sha
from reports.toy100.sample_group_anchor import mst_groups, output_mm_step

SIGMA_OUT = .029
SOURCE_NAMES = (
    "reports/toy100/sample_anchor_mmd_filter.py",
    "reports/toy100/sample_anchor_free1200.py",
    "reports/toy100/sample_group_anchor.py",
    "reports/toy100/coverage_fixed_eval.py",
    "reports/toy100/pr84_early_geometry.py",
    "benchmarks/locked_shared/mode_hold.py",
    "benchmarks/locked_shared/mlp.py",
)


def squared_distances(x, y):
    return (x[:, None, :] - y[None, :, :]).square().sum(-1)


def gaussian_mmd_emitted(real, centers, width, sigma=SIGMA_OUT):
    """V-statistic MMD²(real empirical, equal Gaussian-mixture model)."""
    x = real.double()
    y = centers.double()
    h2 = float(width) ** 2
    s2 = float(sigma) ** 2
    d = x.shape[1]
    pp = torch.exp(-squared_distances(x, x) / (2 * h2)).mean()
    pq = (h2 / (h2 + s2)) ** (d / 2) * torch.exp(
        -squared_distances(x, y) / (2 * (h2 + s2))).mean()
    qq = (h2 / (h2 + 2 * s2)) ** (d / 2) * torch.exp(
        -squared_distances(y, y) / (2 * (h2 + 2 * s2))).mean()
    return pp + qq - 2 * pq


def declared_width(group_centers):
    distances = squared_distances(group_centers.double(), group_centers.double()).sqrt()
    positive = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
    positive = positive[positive > 0]
    if len(positive) == 0:
        raise RuntimeError("at least two distinct sampled groups required")
    return float(torch.quantile(positive, .5))


def quality(points, means):
    index, noise = fixed_draw(240, points.float())
    grade = score_support(points.float(), index, noise, means)
    assignments = torch.cdist(points.float(), means).argmin(1)
    count = torch.bincount(assignments, minlength=len(means))
    mass_tv = float((count.double() / len(points) - 1 / len(means)).abs().sum() / 2)
    variance_ratios = []
    for group in range(len(means)):
        own = points[assignments == group].double()
        if len(own) == 0:
            variance_ratios.append(None)
            continue
        mean = own.mean(0)
        variance = (own - mean).square().mean(0) + SIGMA_OUT ** 2
        variance_ratios.append((variance / mode_hold.SIGMA ** 2).tolist())
    return dict(modes=grade["modes"], hq=grade["hq"],
                nearest_mode_counts=count.tolist(), mode_mass_tv=mass_tv,
                per_mode_coordinate_variance_ratios=variance_ratios,
                points=points.tolist())


def same_law_rest(points, width):
    """Check only equality => zero MMD and gradient, not its converse."""
    source = points.double().detach().clone()
    centers = source.detach().clone().requires_grad_(True)
    # Both sides are the *same Gaussian mixture*. The real-side convolution
    # adds sigma²; q-q and p-p each add 2sigma².
    h2 = width ** 2
    s2 = SIGMA_OUT ** 2
    prefactor = (h2 / (h2 + 2 * s2)) ** (source.shape[1] / 2)
    pp = prefactor * torch.exp(-squared_distances(source, source) / (2 * (h2 + 2*s2))).mean()
    pq = prefactor * torch.exp(-squared_distances(source, centers) / (2 * (h2 + 2*s2))).mean()
    qq = prefactor * torch.exp(-squared_distances(centers, centers) / (2 * (h2 + 2*s2))).mean()
    value = pp + qq - 2 * pq
    gradient = torch.autograd.grad(value, centers)[0]
    checked_value = float(value.detach())
    gradient_max = float(gradient.detach().abs().max())
    return dict(scope="identical Gaussian-mixture laws, same support and sigma on both sides",
                mmd2=checked_value, gradient_max_abs=gradient_max,
                equality_implies_rest=bool(abs(checked_value) < 1e-12 and gradient_max < 1e-12),
                converse_claim=False)


def warm_local_descent(real, initial, width, means):
    """Unconstrained fixed-bank L-BFGS diagnosis; not a policy or global optimum."""
    rows = []

    def value_gradient(flat):
        points = torch.from_numpy(np.asarray(flat, dtype=np.float64).copy()).reshape_as(initial)
        points.requires_grad_(True)
        with torch.enable_grad():
            value = gaussian_mmd_emitted(real, points, width)
            grad = torch.autograd.grad(value, points)[0]
        if not torch.isfinite(value) or not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite MMD value or gradient")
        return float(value.detach()), grad.detach().numpy().ravel().copy()

    def callback(flat):
        points = torch.from_numpy(np.asarray(flat, dtype=np.float64).copy()).reshape_as(initial)
        value = float(gaussian_mmd_emitted(real, points, width))
        grade = quality(points, means)
        rows.append(dict(iteration=len(rows)+1, mmd2=value, quality=grade,
                         max_displacement=float((points-initial).norm(dim=1).max())))

    initial_value, initial_gradient = value_gradient(initial.detach().numpy().ravel())
    result = minimize(value_gradient, initial.detach().numpy().ravel(), method="L-BFGS-B",
                      jac=True, callback=callback,
                      options=dict(maxiter=20, maxfun=100, ftol=1e-12, gtol=1e-8))
    final = torch.from_numpy(result.x.copy()).reshape_as(initial)
    final_value = float(gaussian_mmd_emitted(real, final, width))
    if not np.isfinite(final_value) or final_value > initial_value + 1e-12:
        raise RuntimeError("bounded MMD descent did not preserve objective decrease")
    return dict(solver="scipy L-BFGS-B on 12x2 freely movable output coordinates",
                maxiter=20, maxfun=100, ftol=1e-12, gtol=1e-8,
                success=bool(result.success), message=str(result.message),
                iterations=int(result.nit), function_evaluations=int(result.nfev),
                initial_mmd2=initial_value, initial_gradient_l2=float(np.linalg.norm(initial_gradient)),
                initial_negative_gradient=(-initial_gradient.reshape(initial.shape)).tolist(),
                final_mmd2=final_value, final_quality=quality(final, means),
                final_max_displacement=float((final-initial).norm(dim=1).max()),
                trajectory=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("cold1", "warm1324"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    cold, warm, input_hashes = load_states()
    state = cold if args.case == "cold1" else warm
    support = initial_support(state).double().detach()
    means = mode_hold.ring_means()
    stream = torch.Generator().set_state(state["rng"]["data"])
    real_native = mode_hold.sample_ring(means, 128, mode_hold.SIGMA, stream)
    real = real_native.double()
    groups, grouping = mst_groups(real)
    if not 0 < len(groups) < len(support):
        raise RuntimeError("unsupported sample-group count for anchor MM")
    width = declared_width(groups)
    anchor = output_mm_step(support, groups)
    target = torch.tensor(anchor["target"], dtype=torch.float64)
    before = float(gaussian_mmd_emitted(real, support, width))
    after = float(gaussian_mmd_emitted(real, target, width))
    rest = same_law_rest(support, width)
    source = {}
    for name in SOURCE_NAMES:
        raw = (ROOT/name).read_bytes()
        path = args.output/"source"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        source[name] = sha(raw)
    declaration = dict(scope="one-bank free-output diagnostic, no GAN or neural update",
                       case=args.case, input_hashes=input_hashes, source_sha256=source,
                       bank_sha256=sha(real_native.contiguous().numpy().tobytes()),
                       native_real_batch=128, kernel="Gaussian exp(-||x-y||²/(2h²))",
                       width_rule="median positive pairwise inferred-group centroid distance on this first real128 bank; frozen",
                       width=width, output_sigma=SIGMA_OUT,
                       output_noise_integration="exact Gaussian convolution in p-q and q-q",
                       cold_proposal="existing one-anchor-per-group active-quadratic MM target",
                       warm_solver="unconstrained output-space L-BFGS-B, 20 iterations/100 calls maximum, no tuning",
                       evaluator="same fixed 4096 late-noise draws at diagnostic clock 240 for all clouds",
                       no_oracle_objective=True,
                       theorem_scope="identical full probability laws imply zero MMD and functional gradient; not converse from finite-support stationarity")
    result = dict(status="COMPLETE", declaration=declaration,
                  grouping=grouping, group_centers=groups.tolist(),
                  initial=dict(mmd2=before, quality=quality(support, means)),
                  anchor_target=dict(mmd2=after, change=after-before,
                                     improved=after < before, quality=quality(target, means),
                                     anchor_objective_before=anchor["before"],
                                     anchor_objective_after=anchor["after"],
                                     max_displacement=anchor["max_output_displacement"]),
                  equal_law_rest_check=rest,
                  warm_local_descent=(warm_local_descent(real, support, width, means)
                                      if args.case == "warm1324" else None),
                  data_rng_after_bank_sha256=sha(stream.get_state().numpy().tobytes()),
                  runtime=dict(torch=torch.__version__, scipy=scipy.__version__,
                               threads=torch.get_num_threads(),
                               cpu_capability=torch.backends.cpu.get_cpu_capability()))
    (args.output/"declaration.json").write_text(json.dumps(declaration, indent=2, allow_nan=False)+"\n")
    (args.output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    summary=dict(case=args.case, width=width, groups=len(groups),
                 initial_mmd2=before, anchor_mmd2=after, anchor_change=after-before,
                 anchor_improved=after < before,
                 initial_grade={k:result["initial"]["quality"][k] for k in ("modes","hq","mode_mass_tv")},
                 anchor_grade={k:result["anchor_target"]["quality"][k] for k in ("modes","hq","mode_mass_tv")},
                 warm_solver=(None if args.case == "cold1" else {
                     k:result["warm_local_descent"][k] for k in (
                         "success","iterations","final_mmd2","final_max_displacement")}))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
