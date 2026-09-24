"""Scratch, label-free output-bandwidth diagnostic for the shared toy suite.

This is a screen-only estimator. It reads the first real discriminator batch
from each frozen host without taking a training update, then maximizes the
leave-one-out likelihood of an isotropic Gaussian KDE in flattened data space.
It does not inspect geometry, class labels, target standard deviations, or
frozen verdicts. A batch whose every row has an exact duplicate has the
mathematical limiting maximizer h=0; mixed duplicate/singleton batches do not.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
import torch

from particlegan import GANTrainer
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from benchmarks.transfer_suite.public_default_verification import load_declaration, declared_spec
from benchmarks.transfer_suite.toy100_compatibility import (
    declared_recipe, run_vector, run_image, run_noisy_legacy,
)


class _FirstRealCaptured(Exception):
    pass


def estimate_bandwidth(batch: np.ndarray) -> dict:
    """Return the global LOO Gaussian KDE likelihood maximizer, in data units.

    Search bounds derive from observed pair distances and batch size. The
    coarse log grid is only a numerical bracket; the reported width is refined
    by scalar optimization. A boundary optimum raises instead of silently
    replacing the result with a fixed numerical width.
    """
    original = np.asarray(batch)
    if original.ndim < 2 or len(original) < 2 or not np.isfinite(original).all():
        raise ValueError("batch needs at least two finite data rows")
    points = original.reshape(len(original), -1).astype(np.float64)
    n, d = points.shape
    _, inverse, counts = np.unique(points, axis=0, return_inverse=True,
                                    return_counts=True)
    duplicated = int(np.count_nonzero(counts[inverse] > 1))
    if duplicated == n:
        return dict(width=0.0, n=n, d=d, duplicate_rows=duplicated,
                    boundary="all_observations_duplicated")
    squared = cdist(points, points, metric="sqeuclidean")
    np.fill_diagonal(squared, np.inf)
    finite_positive = squared[np.isfinite(squared) & (squared > 0)]
    if not len(finite_positive):
        raise AssertionError("mixed singleton batch must have a positive pair distance")
    lower = math.sqrt(float(finite_positive.min())) / n
    upper = math.sqrt(float(finite_positive.max())) * math.sqrt(n)
    logs = np.linspace(math.log(lower), math.log(upper), 96)

    def nll(log_width: float) -> float:
        width = math.exp(float(log_width))
        average = logsumexp(-squared / (2 * width * width), axis=1).mean()
        return float(-(average - math.log(n - 1) - d * math.log(width)
                       - d * math.log(2 * math.pi) / 2))

    coarse = np.array([nll(value) for value in logs])
    best = int(np.argmin(coarse))
    if best in (0, len(logs) - 1):
        raise ArithmeticError("LOO bandwidth hit a data-derived search boundary")
    refined = minimize_scalar(nll, method="bounded",
                              bounds=(logs[best - 1], logs[best + 1]),
                              options={"xatol": 1e-8})
    if not refined.success or not math.isfinite(refined.fun):
        raise ArithmeticError("LOO bandwidth optimizer did not converge")
    return dict(width=math.exp(float(refined.x)), n=n, d=d,
                duplicate_rows=duplicated, boundary=None,
                negative_log_likelihood=float(refined.fun),
                search_bounds=[lower, upper])


def capture_first_real(spec: dict, card: dict | None, base, noise: dict) -> np.ndarray:
    """Intercept the first real D batch from an unmodified frozen host run."""
    captured: list[torch.Tensor] = []
    original_input = NoisePolicy.input

    def native_step(self, real, **kwargs):
        captured.append(real.detach().cpu().clone())
        raise _FirstRealCaptured

    def legacy_input(self, data):
        if self._step_calls and not self._evaluating:
            captured.append(data.detach().cpu().clone())
            raise _FirstRealCaptured
        return original_input(self, data)

    try:
        with patch.object(GANTrainer, "step", native_step), patch.object(
            NoisePolicy, "input", legacy_input,
        ):
            if spec["runner"] == "vector":
                run_vector(spec, card, base, noise)
            elif spec["runner"] == "image":
                run_image(spec, base, noise)
            else:
                run_noisy_legacy(spec, base, noise)
    except _FirstRealCaptured:
        pass
    if len(captured) != 1:
        raise RuntimeError(f"first real capture failed for {spec['name']}")
    return captured[0].numpy()


def capture_all(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config_bytes = config_path.read_bytes()
    base, noise, _ = declared_recipe(json.loads(config_bytes))
    jobs, profile = load_declaration()
    batches = {}
    rows = []
    for job in jobs:
        spec, card, _ = declared_spec(job, profile, base)
        batch = capture_first_real(spec, card, base, noise)
        batches[spec["name"]] = batch
        row = estimate_bandwidth(batch)
        row.update(host=spec["name"], runner=spec["runner"],
                   first_real_sha256=hashlib.sha256(batch.tobytes()).hexdigest())
        rows.append(row)
    np.savez_compressed(output / "first_real_batches.npz", **batches)
    receipt = dict(
        algorithm="first-real-batch Gaussian KDE leave-one-out log likelihood",
        status="screen_only_common_gate_ineligible",
        first_batch_uses_no_training_update=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        batches_sha256=hashlib.sha256((output / "first_real_batches.npz").read_bytes()).hexdigest(),
        rows=rows,
    )
    (output / "bandwidth_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture_all(args.config, args.output)
