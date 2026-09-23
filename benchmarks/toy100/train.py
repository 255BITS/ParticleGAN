"""Reproducible, tail-friendly training runner for the toy100 coverage gate.

The target sampler supplies only unlabelled real batches to ``GANTrainer``.
Mode centers and assignments are used by the evaluation module after updates;
they never enter the generator, discriminator, loss, or optimizer.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import GANTrainer, Recipe, get_recipe

from .metrics import EVAL_N, evaluate_samples
from .problems import PROBLEM_NAMES, sample_real


ROOT = Path(__file__).resolve().parents[2]
EARLY_EVAL_STEPS = (0, 1, 10, 25, 50, 100)
RUN_DEFAULTS = {
    "problem": "grid100",
    "seed": 1234,
    "device": "cpu",
    "steps": 7000,
    "fourier": 2,
    "g_hidden": 128,
    "d_hidden": 128,
    "n_hidden": 3,
    "eval_samples": EVAL_N,
    "snapshot_samples": 4096,
    "eval_interval": 250,
    "snapshot_interval": 250,
    "early_eval_steps": list(EARLY_EVAL_STEPS),
    "stable_evals": 5,
    "log_interval": 250,
    "fused_adam": False,
    "threads": 1,
}
RECIPE_FIELDS = {field.name for field in fields(Recipe)}
RUN_FIELDS = set(RUN_DEFAULTS)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a flat JSON or TOML recipe; ``train`` validates all fields."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        config = json.loads(path.read_text())
    elif path.suffix.lower() == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10 with the experiment extra
            import tomli as tomllib
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    else:
        raise ValueError("config must be JSON or TOML")
    if not isinstance(config, dict) or not all(isinstance(k, str) for k in config):
        raise ValueError("config must be an object with string keys")
    return config


def evaluation_steps(
    budget_steps: int,
    eval_interval: int = 250,
    early_eval_steps: tuple[int, ...] | list[int] = EARLY_EVAL_STEPS,
) -> list[int]:
    """The complete observable schedule, including untouched step 0 and final."""
    if type(budget_steps) is not int or budget_steps < 1:
        raise ValueError("budget_steps must be a positive integer")
    if type(eval_interval) is not int or eval_interval < 1:
        raise ValueError("eval_interval must be a positive integer")
    if not isinstance(early_eval_steps, (tuple, list)) or any(
        type(step) is not int or step < 0 for step in early_eval_steps
    ):
        raise ValueError("early_eval_steps must be nonnegative integers")
    return sorted(
        {0, budget_steps}
        | {step for step in early_eval_steps if step <= budget_steps}
        | set(range(eval_interval, budget_steps + 1, eval_interval))
    )


def snapshot_steps(
    budget_steps: int,
    eval_steps: list[int],
    snapshot_interval: int,
) -> list[int]:
    if type(snapshot_interval) is not int or snapshot_interval < 1:
        raise ValueError("snapshot_interval must be a positive integer")
    # Every scored point is replayable, even if a recipe requests extra frames.
    return sorted(set(eval_steps) | set(range(snapshot_interval, budget_steps + 1, snapshot_interval)))


def resolve_config(user: Mapping[str, Any]) -> tuple[dict[str, Any], Recipe]:
    if not isinstance(user, Mapping) or not all(isinstance(key, str) for key in user):
        raise ValueError("config must be a mapping with string keys")
    unknown = set(user) - RUN_FIELDS - RECIPE_FIELDS
    if unknown:
        raise ValueError(f"unknown config fields: {', '.join(sorted(unknown))}")
    run = {**RUN_DEFAULTS, **{key: user[key] for key in user if key in RUN_FIELDS}}
    if "steps" in user and "total_steps" in user and user["steps"] != user["total_steps"]:
        raise ValueError("steps and total_steps disagree")
    if "steps" not in user and "total_steps" in user:
        run["steps"] = user["total_steps"]
    if run["problem"] not in PROBLEM_NAMES:
        raise ValueError(f"unknown problem {run['problem']!r}; choose {', '.join(PROBLEM_NAMES)}")
    for key in ("steps", "seed", "fourier", "g_hidden", "d_hidden", "n_hidden",
                "eval_samples", "snapshot_samples", "eval_interval", "snapshot_interval",
                "stable_evals", "log_interval", "threads"):
        value = run[key]
        minimum = 0 if key in ("seed", "fourier") else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if run["device"] != "cpu" and not str(run["device"]).startswith("cuda"):
        raise ValueError("device must be cpu or cuda[:index]")
    device = torch.device(run["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if type(run["fused_adam"]) is not bool:
        raise ValueError("fused_adam must be a boolean")
    recipe_kwargs = {key: user[key] for key in user if key in RECIPE_FIELDS and key != "name"}
    recipe_kwargs["total_steps"] = run["steps"]
    recipe = get_recipe(**recipe_kwargs)
    if "name" in user:
        recipe = recipe.replace(name=user["name"])
    if (recipe.model != "gan" or recipe.conditioning != "scalar"
            or recipe.encoder_mode != "none" or recipe.prior_kind != "particles"):
        raise ValueError("toy100 requires an unconditional scalar GAN with a learned particle prior")
    run["device"] = str(device)
    # Store all resolved inputs, including the public recipe's inherited values.
    return {**recipe.to_dict(), **run}, recipe


def _init_linear(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def make_trainer(config: Mapping[str, Any], recipe: Recipe) -> GANTrainer:
    """Match the public 100-Gaussian example's model and initialization."""
    device = torch.device(config["device"])
    seed = config["seed"]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        prior = recipe.make_prior(learnable=True).to(device)
        generator = SimpleMLPGenerator(
            z_dim=recipe.z_dim, hidden_dim=config["g_hidden"], n_hidden=config["n_hidden"]
        ).to(device)
        discriminator = SimpleMLPDiscriminator(
            in_dim=2, hidden_dim=config["d_hidden"], n_hidden=config["n_hidden"],
            fourier=config["fourier"],
        ).to(device)
        _init_linear(generator)
        _init_linear(discriminator)
        return GANTrainer(
            recipe, generator, discriminator, prior=prior, seed=seed,
            optimizer_options={"fused": config["fused_adam"]},
        )


def _source_provenance() -> dict[str, Any]:
    paths = (
        "benchmarks/toy100/train.py", "benchmarks/toy100/problems.py",
        "benchmarks/toy100/metrics.py", "lib/toy_models.py",
        "particlegan/training.py", "particlegan/recipes.py",
    )
    hashes = {}
    for name in paths:
        path = ROOT / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    return {"git_sha": git_sha, "source_sha256": hashes,
            "generated_at_utc": datetime.now(timezone.utc).isoformat()}


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class _Log:
    def __init__(self, directory: Path):
        self.events = (directory / "events.jsonl").open("x", buffering=1)
        self.progress = (directory / "progress.log").open("x", buffering=1)

    def event(self, row: dict[str, Any]) -> None:
        self.events.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")

    def say(self, message: str) -> None:
        print(message, flush=True)
        self.progress.write(message + "\n")

    def close(self) -> None:
        self.events.close()
        self.progress.close()


def train(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Train exactly one named problem and write complete replayable evidence.

    An existing empty output directory is accepted. Any earlier run evidence
    causes an error, so a failed or partial run cannot silently be overwritten.
    """
    resolved, recipe = resolve_config(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    (out_dir / "snapshots").mkdir(exist_ok=True)
    torch.set_num_threads(resolved["threads"])
    if torch.device(resolved["device"]).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    _write_json(out_dir / "config.json", resolved)
    provenance = _source_provenance()
    _write_json(out_dir / "provenance.json", provenance)
    device = torch.device(resolved["device"])
    environment = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__, "platform": platform.platform(),
        "device": str(device), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "threads": torch.get_num_threads(), "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    budget = resolved["steps"]
    eval_steps = evaluation_steps(budget, resolved["eval_interval"], resolved["early_eval_steps"])
    snap_steps = snapshot_steps(budget, eval_steps, resolved["snapshot_interval"])
    eval_set, snap_set = set(eval_steps), set(snap_steps)
    summary: dict[str, Any] = {
        "status": "running", "problem": resolved["problem"], "budget_steps": budget,
        "config": resolved, "eval_steps": eval_steps, "snapshot_steps": snap_steps,
        "provenance": provenance, "environment": environment,
    }
    _write_json(out_dir / "summary.json", summary)
    logger = _Log(out_dir)
    start = time.perf_counter()
    train_seconds = 0.0
    eval_seconds = 0.0
    first_full = {"live": None, "ema": None}
    first_pass = {"live": None, "ema": None}
    stable_pass = {"live": None, "ema": None}
    pass_streak = {"live": 0, "ema": 0}
    final_metrics: dict[str, dict[str, Any]] = {}
    device_seed = lambda value: torch.Generator(device=device).manual_seed(value)

    try:
        trainer = make_trainer(resolved, recipe)
        train_data_rng = device_seed(resolved["seed"])
        target_rng = device_seed(resolved["seed"] + 401)
        target = sample_real(
            resolved["problem"], max(resolved["eval_samples"], resolved["snapshot_samples"]),
            device=device, generator=target_rng,
        )
        logger.say(
            f"START problem={resolved['problem']} steps={budget} device={device} "
            f"events={out_dir / 'events.jsonl'}"
        )

        def observe(step: int) -> None:
            nonlocal eval_seconds
            observed_start = time.perf_counter()
            is_eval, is_snapshot = step in eval_set, step in snap_set
            count = max(
                resolved["eval_samples"] if is_eval else 0,
                resolved["snapshot_samples"] if is_snapshot else 0,
            )
            arrays = {}
            final_draws = {}
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            # Any random work inside metrics is replayable and leaves the
            # caller's global training RNG state exactly as it was.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(resolved["seed"] + 402)
                for model in ("live", "ema"):
                    draw = trainer.sample(
                        count, ema=(model == "ema"),
                        generator=device_seed(resolved["seed"] + 403),
                    )
                    if is_eval:
                        metrics = evaluate_samples(draw[:resolved["eval_samples"]], resolved["problem"])
                        metrics = dict(metrics)
                        elapsed = time.perf_counter() - start
                        logger.event({"event": "eval", "step": step, "model": model,
                                      "metrics": metrics, "elapsed": elapsed})
                        final_metrics[model] = metrics
                        if step > 0 and metrics["modes"] == 100 and first_full[model] is None:
                            first_full[model] = step
                        if step > 0 and metrics["passed"]:
                            if first_pass[model] is None:
                                first_pass[model] = step
                            pass_streak[model] += 1
                            if (pass_streak[model] >= resolved["stable_evals"]
                                    and stable_pass[model] is None):
                                stable_pass[model] = step
                        else:
                            pass_streak[model] = 0
                        logger.say(
                            f"EVAL step={step}/{budget} model={model} "
                            f"modes={metrics['modes']}/100 hq={metrics['hq']:.4f} "
                            f"pass={metrics['passed']} elapsed={elapsed:.1f}s"
                        )
                    if is_snapshot:
                        arrays[model] = draw[:resolved["snapshot_samples"]].detach().cpu().numpy()
                    if step == budget:
                        # Retain the exact scored draws for an independent
                        # final metric audit, separate from the smaller GIF frame.
                        final_draws[model] = draw[:resolved["eval_samples"]].detach().cpu().numpy()
            if is_snapshot:
                filename = out_dir / "snapshots" / f"step_{step:06d}.npz"
                np.savez_compressed(
                    filename, live=arrays["live"], ema=arrays["ema"],
                    target=target[:resolved["snapshot_samples"]].detach().cpu().numpy(),
                )
            if step == budget:
                np.savez_compressed(
                    out_dir / "final_samples.npz", live=final_draws["live"],
                    ema=final_draws["ema"],
                    target=target[:resolved["eval_samples"]].detach().cpu().numpy(),
                )
            eval_seconds += time.perf_counter() - observed_start

        # Capture actual unmodified G and prior, before the first optimizer step.
        observe(0)
        for step in range(1, budget + 1):
            step_start = time.perf_counter()
            real = sample_real(
                resolved["problem"], recipe.batch_size, device=device, generator=train_data_rng,
            )
            stats = trainer.step(
                real,
                generator_real=lambda: sample_real(
                    resolved["problem"], recipe.batch_size, device=device,
                    generator=train_data_rng,
                ),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds += time.perf_counter() - step_start
            losses = {key: float(stats[key]) for key in
                      ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty")}
            if not all(math.isfinite(value) for value in losses.values()):
                raise FloatingPointError(f"nonfinite training loss at step {step}: {losses}")
            logger.event({"event": "train", "step": step, "elapsed": time.perf_counter() - start,
                          **losses})
            if step == 1 or step % resolved["log_interval"] == 0 or step == budget:
                logger.say(
                    f"TRAIN step={step}/{budget} d={losses['loss_d']:.4f} "
                    f"g={losses['loss_g']:.4f} train_seconds={train_seconds:.1f}"
                )
            if step in eval_set or step in snap_set:
                observe(step)

        summary.update({
            "status": "complete", "completed_steps": trainer.completed_steps,
            "first_full_coverage_step": first_full, "first_pass_step": first_pass,
            "stable_pass_step": stable_pass, "final": final_metrics,
            "final_samples_file": "final_samples.npz",
            "train_seconds": train_seconds, "eval_seconds": eval_seconds,
            "total_seconds": time.perf_counter() - start,
            "steps_per_second": budget / train_seconds,
        })
        _write_json(out_dir / "summary.json", summary)
        logger.say(
            f"COMPLETE problem={resolved['problem']} steps={budget} "
            f"live_modes={final_metrics['live']['modes']}/100 "
            f"live_hq={final_metrics['live']['hq']:.4f} "
            f"stable_pass={stable_pass['live']} total_seconds={summary['total_seconds']:.1f}"
        )
        return summary
    except Exception as error:
        summary.update({
            "status": "error", "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "completed_steps": trainer.completed_steps if "trainer" in locals() else 0,
            "train_seconds": train_seconds, "eval_seconds": eval_seconds,
            "total_seconds": time.perf_counter() - start,
        })
        _write_json(out_dir / "summary.json", summary)
        logger.say(f"ERROR problem={resolved['problem']} {summary['error']}")
        raise
    finally:
        logger.close()
