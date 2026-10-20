"""Rebuild an affine initialization without leaking process-wide init hooks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


PREFIX = "AFFINE_REPLAY "


def replay_initialization(config):
    """Audit on the recorded device; missing CUDA remains a failed audit."""
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.toy100.affine_replay"],
        input=json.dumps(config, allow_nan=False), text=True, capture_output=True,
        cwd=Path(__file__).resolve().parents[2], timeout=120,
    )
    if result.returncode:
        raise ValueError("100-mode affine initialization replay requires the recorded device "
                         "and initializer: " + (result.stderr or result.stdout)[-1000:])
    lines = [line[len(PREFIX):] for line in result.stdout.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("100-mode affine initialization replay receipt is absent")
    return json.loads(lines[0])


def main():
    import torch
    from benchmarks.init_research.init_registry import install
    from benchmarks.toy100.device import apply_device_policy
    from benchmarks.toy100.train import make_trainer, resolve_config

    config = json.load(sys.stdin)
    apply_device_policy(config["device"])
    if config.get("deterministic_init") is not None:
        install(config["deterministic_init"])
    _, recipe = resolve_config(config)
    trainer = make_trainer(config, recipe)
    prior = trainer.prior.z.detach().cpu().contiguous()
    generator = trainer.G.model if config["output_noise_std"] else trainer.G
    print(PREFIX + json.dumps(dict(
        prior_initial_sha256=hashlib.sha256(prior.numpy().tobytes()).hexdigest(),
        prior_initial_min=float(prior.min()), prior_initial_max=float(prior.max()),
        weight=generator.weight.detach().cpu().tolist(),
        bias=generator.bias.detach().cpu().tolist(),
    ), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
