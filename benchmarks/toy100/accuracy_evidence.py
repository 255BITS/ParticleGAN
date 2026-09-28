"""Capture fidelity evidence without changing training or its random streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .accuracy import PROTOCOL, evaluate_accuracy
from .accuracy_gate import HOLDOUT_N, HOLDOUT_SEED_OFFSETS
from .gate import MIN_STABLE_CHECKS
from .problems import sample_real


class AccuracyEvidence:
    def __init__(self, config: dict, output: Path, eval_steps: list[int], target):
        self.config = config
        self.output = Path(output)
        self.check_steps = eval_steps[-MIN_STABLE_CHECKS:]
        self.target = target[:config["eval_samples"]].detach().cpu().numpy()
        self.pending: dict[int, dict[str, np.ndarray]] = {}
        (self.output / "quality_checks").mkdir(exist_ok=True)

    def observe(self, step: int, model: str, samples, coverage_metrics: dict) -> dict:
        values = samples[:self.config["eval_samples"]]
        accuracy = evaluate_accuracy(values, self.config["problem"], gate_metrics=coverage_metrics)
        if step in self.check_steps:
            models = self.pending.setdefault(step, {})
            models[model] = values.detach().cpu().numpy()
            if set(models) == {"live", "ema"}:
                np.savez_compressed(
                    self.output / "quality_checks" / f"step_{step:06d}.npz",
                    **models, target=self.target,
                )
                del self.pending[step]
        return accuracy

    def finish(self, trainer) -> tuple[dict, dict]:
        if self.pending:
            raise RuntimeError("incomplete terminal accuracy samples")
        config = self.config
        device = torch.device(config["device"])
        generator = lambda offset: torch.Generator(device=device).manual_seed(config["seed"] + offset)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        arrays = {}
        metrics = {}
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(config["seed"] + HOLDOUT_SEED_OFFSETS["noise"])
            for model in ("live", "ema"):
                draws = trainer.sample(HOLDOUT_N, ema=model == "ema",
                                       generator=generator(HOLDOUT_SEED_OFFSETS["latent"]))
                arrays[model] = draws.detach().cpu().numpy()
            arrays["target"] = sample_real(
                config["problem"], HOLDOUT_N, device=device,
                generator=generator(HOLDOUT_SEED_OFFSETS["target"]),
            ).detach().cpu().numpy()
            for model, points in arrays.items():
                metrics[model] = evaluate_accuracy(points, config["problem"])
        np.savez_compressed(self.output / "holdout_samples.npz", **arrays)
        metadata = {"protocol": PROTOCOL, "check_steps": self.check_steps,
                    "sample_count": config["eval_samples"], "holdout_samples": HOLDOUT_N,
                    "holdout_seed_offsets": HOLDOUT_SEED_OFFSETS}
        return metadata, metrics
