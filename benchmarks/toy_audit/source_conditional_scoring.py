"""Frozen distribution checks for shipped sparse and analytic-posterior laws.

These diagnostic gates are separate from the original source convergence bars.
They do not certify a partial run or confer Forge/default promotion.
"""
from __future__ import annotations

import math
import numpy as np
import torch

VERSION = "conditional-source-distribution-v1"
COUNT = 4096
EVAL_SEED = 99123
GATES = dict(hq_min=.95, conditional_quality_mass_tv_max=.10,
             exact_inactive_zero_fraction=1., symbol_tv_max=.10,
             mean_error_sigma_max=.10, covariance_eigenvalues=[.85, 1.15],
             radial_ks_max=.075, max_projection_ks_max=.06,
             full_original_update_budget=True, terminal_observations=5)


def ks(values, cdf):
    values = np.sort(np.asarray(values, dtype=np.float64))
    expected = cdf(values)
    rank = np.arange(1, len(values) + 1) / len(values)
    return float(max(np.max(rank - expected), np.max(expected - (rank - 1 / len(values)))))


def normal_cdf(values):
    return np.array([.5 * (1 + math.erf(float(v) / math.sqrt(2))) for v in values])


def shape(residual):
    residual = np.asarray(residual, dtype=np.float64)
    n, dimension = residual.shape
    eigenvalues = np.linalg.eigvalsh(np.cov(residual, rowvar=False)).tolist()
    norms = np.linalg.norm(residual, axis=1)
    if dimension == 2:
        cdf = lambda radius: 1 - np.exp(-radius**2 / 2)
    else:
        cdf = lambda radius: np.array([math.erf(float(r) / math.sqrt(2)) for r in radius]) - math.sqrt(2 / math.pi) * radius * np.exp(-radius**2 / 2)
    directions = np.random.default_rng(5501).normal(size=(16, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    out = dict(sample_count=n, mean_error_sigma=float(np.linalg.norm(residual.mean(0))),
               covariance_eigenvalues=eigenvalues, radial_ks=ks(norms, cdf),
               max_projection_ks=max(ks(residual @ direction, normal_cdf) for direction in directions))
    out["passed"] = bool(n >= 1024 and out["mean_error_sigma"] <= .10
                         and min(eigenvalues) >= .85 and max(eigenvalues) <= 1.15
                         and out["radial_ks"] <= .075 and out["max_projection_ks"] <= .06)
    return out


@torch.no_grad()
def sparse_metrics(toy, x, symbols, requested):
    mode, distance = toy.assign(x)
    active = toy.active[mode]
    hq = distance <= 3 * toy.sigma * math.sqrt(toy.k)
    correct_class = toy.class_of_mode[mode] == requested
    coherent_symbol = toy.symbol_of_mode[mode] == symbols
    joint = hq & correct_class & coherent_symbol
    tv, symbol_tv, confusion = [], [], []
    for cls in range(toy.n_classes):
        selected = requested == cls
        count = int(selected.sum())
        masses = torch.bincount(mode[selected & joint], minlength=toy.n_modes).double() / count
        target = (toy.class_of_mode == cls).double() / len(toy.class_modes[cls])
        tv.append(float((masses - target).abs().sum() / 2 + (1 - masses.sum()) / 2))
        emitted = torch.bincount(symbols[selected], minlength=toy.n_symbols).double() / count
        confusion.append(emitted.tolist())
        symbol_tv.append(float((emitted - toy.p_symbol_given_class[cls]).abs().sum() / 2))
    inactive = ~active
    zero_fraction = float(((x == 0) & inactive).sum() / inactive.sum())
    residual = ((x - toy.centers[mode])[active].reshape(len(x), toy.k) / toy.sigma).cpu().numpy()
    local_shape = shape(residual)
    masses = torch.bincount(mode[hq], minlength=toy.n_modes)
    out = dict(sample_count=len(x), modes=int((masses >= 5).sum()), hq=float(hq.float().mean()),
               cond_acc=float(correct_class.float().mean()), sym_acc_mode=float(coherent_symbol.float().mean()),
               joint_hq=float(joint.float().mean()), exact_zero_frac=zero_fraction,
               **{"sparse_prec@0p01": float(((x.abs() < .01) & inactive).sum() / inactive.sum())},
               conditional_quality_mass_tv=tv, max_conditional_quality_mass_tv=max(tv),
               symbol_tv_by_class=symbol_tv, max_symbol_tv=max(symbol_tv), confusion=confusion,
               mode_mass=(torch.bincount(mode, minlength=toy.n_modes).double() / len(x)).tolist(),
               local_shape=local_shape)
    out["passed"] = bool(out["modes"] == toy.n_modes and out["hq"] >= .95
                         and zero_fraction == 1 and max(tv) <= .10 and max(symbol_tv) <= .10
                         and local_shape["passed"])
    return out


@torch.no_grad()
def posterior_metrics(toy, x, observation, cls, abar):
    """Score repeated clean draws at one fixed observation/class/time."""
    condition = torch.full((1,), cls, dtype=torch.long)
    weights, means, variance = toy.posterior(observation.reshape(1, 2), condition, abar)
    means, weights = means[0], weights[0].double()
    sigma = float(variance.flatten()[0].sqrt())
    distance, assigned = torch.cdist(x, means).min(1)
    hq = distance <= 3 * sigma
    quality_mass = torch.bincount(assigned[hq], minlength=len(means)).double() / len(x)
    tv = float((quality_mass - weights).abs().sum() / 2 + (1 - quality_mass.sum()) / 2)
    local = shape(((x - means[assigned]) / sigma).cpu().numpy())
    return dict(sample_count=len(x), conditional_quality_mass_tv=tv, hq=float(hq.float().mean()),
                local_shape=local, passed=bool(tv <= .10 and float(hq.float().mean()) >= .95 and local["passed"]))


def calibration(sparse_type, grid_type):
    """Exact-law positives and targeted counterexamples; no learned training."""
    records = []
    for symbol_map in ("identity", "split"):
        toy = sparse_type(symbol_map=symbol_map, n_symbols=8 if symbol_map == "identity" else 16)
        requested = torch.arange(COUNT) % 8
        x, symbols, mode = toy.sample_given_class(requested, torch.Generator().manual_seed(EVAL_SEED))
        inactive = ~toy.active[mode]
        tests = {"oracle": (x, symbols, requested),
                 "tiny_inactive_smear": (x + inactive * 1e-6, symbols, requested),
                 "zero_width_centres": (toy.centers[mode], symbols, requested),
                 "wrong_requested_class": (x, symbols, (requested + 1) % 8)}
        if symbol_map == "split":
            tests["one_valid_symbol_per_class"] = (x, 2 * requested, requested)
            tests["balanced_but_incoherent_symbols"] = (x, symbols ^ 1, requested)
        for name, (points, emitted, classes) in tests.items():
            result = sparse_metrics(toy, points, emitted, classes)
            expected = name == "oracle"
            if result["passed"] != expected:
                raise AssertionError((symbol_map, name, result))
            records.append(dict(family="sparse", variant=symbol_map, control=name, expected_pass=expected,
                                passed=result["passed"], metrics=result))
    for classes in (1, 4):
        toy = grid_type(classes=classes)
        observation, abar = torch.zeros(2), .5
        requested = torch.zeros(COUNT, dtype=torch.long)
        observed = observation.expand(COUNT, -1)
        x = toy.oracle_clean(observed, requested, abar, torch.Generator().manual_seed(EVAL_SEED))
        weights, means, _ = toy.posterior(observation[None], requested[:1], abar)
        mean = (weights[..., None] * means).sum(1).expand(COUNT, -1)
        unconditional = toy.sample(requested, torch.Generator().manual_seed(EVAL_SEED))
        tests = {"oracle": x, "posterior_mean_collapse": mean, "ignores_observation": unconditional}
        if classes == 4:
            wrong = torch.ones(COUNT, dtype=torch.long)
            tests["wrong_class_posterior"] = toy.oracle_clean(observed, wrong, abar, torch.Generator().manual_seed(EVAL_SEED))
        for name, points in tests.items():
            result = posterior_metrics(toy, points, observation, 0, abar)
            expected = name == "oracle"
            if result["passed"] != expected:
                raise AssertionError((classes, name, result))
            records.append(dict(family="denoising", variant=classes, control=name, expected_pass=expected,
                                passed=result["passed"], metrics=result))
    return dict(version=VERSION, scope="Analytic diagnostic-gate controls, no trained evidence", controls=records)
