"""Scratch two-batch extension of the first-real LOO bandwidth diagnostic.

Two batches are the minimum temporal observation that can reveal a repeated
finite support when one batch contains each support point only once. The
preflight run is discarded; a candidate run starts afresh from the fixed seed.
This is screen-only and does not authorize a common-22 gate claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from particlegan import GANTrainer
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from benchmarks.transfer_suite.public_default_verification import load_declaration, declared_spec
from benchmarks.transfer_suite.toy100_compatibility import (
    declared_recipe, run_vector, run_image, run_noisy_legacy,
)
from reports.toy100.bandwidth_kde_probe import estimate_bandwidth


class _TwoRealBatchesCaptured(Exception):
    pass


def capture_two_real_batches(spec: dict, card: dict | None, base, noise: dict) -> np.ndarray:
    captured: list[torch.Tensor] = []
    original_step = GANTrainer.step
    original_input = NoisePolicy.input
    seen_legacy_steps: set[int] = set()

    def native_step(self, real, **kwargs):
        captured.append(real.detach().cpu().clone())
        if len(captured) == 2:
            raise _TwoRealBatchesCaptured
        return original_step(self, real, **kwargs)

    def legacy_input(self, data):
        if (self._step_calls > 0 and not self._evaluating
                and self._step_calls not in seen_legacy_steps):
            seen_legacy_steps.add(self._step_calls)
            captured.append(data.detach().cpu().clone())
            if len(captured) == 2:
                raise _TwoRealBatchesCaptured
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
    except _TwoRealBatchesCaptured:
        pass
    if len(captured) != 2 or captured[0].shape != captured[1].shape:
        raise RuntimeError(f"two-batch real capture failed for {spec['name']}")
    return np.stack([value.numpy() for value in captured])


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
        pair = capture_two_real_batches(spec, card, base, noise)
        batches[spec["name"]] = pair
        row = estimate_bandwidth(pair.reshape((-1, *pair.shape[2:])))
        row.update(host=spec["name"], runner=spec["runner"],
                   first_two_real_sha256=hashlib.sha256(pair.tobytes()).hexdigest())
        rows.append(row)
    np.savez_compressed(output / "first_two_real_batches.npz", **batches)
    receipt = dict(
        algorithm="first-two-real-batches Gaussian KDE leave-one-out log likelihood",
        status="screen_only_common_gate_ineligible",
        preflight_discarded_and_training_restarted=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        estimator_source_sha256=hashlib.sha256(
            Path(__file__).with_name("bandwidth_kde_probe.py").read_bytes(),
        ).hexdigest(),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        batches_sha256=hashlib.sha256((output / "first_two_real_batches.npz").read_bytes()).hexdigest(),
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
