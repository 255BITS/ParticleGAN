"""Reproducible, tail-friendly training runner for the toy100 coverage gate.

The target sampler supplies only unlabelled real batches to ``GANTrainer``.
Mode centers and assignments are used by the evaluation module after updates;
they never enter the generator, discriminator, loss, or optimizer.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import GANTrainer, Recipe, get_recipe

from .metrics import EVAL_N, evaluate_samples
from .models import InputNoise, OutputNoise, linear_input_noise, linear_output_noise
from .problems import PROBLEM_NAMES, sample_real
from .schedule import policy_rate_action, step_with_policy


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
    "output_noise_std": 0.0,
    "input_noise_std": 0.0,
    "input_noise_anneal_end": 0.5,
}
OPTIONAL_RUN_FIELDS = {
    "output_noise_warmup", "output_noise_learnable",
    "toy100_model", "network_lr_horizon_cap", "network_lr_floor",
}
RECIPE_FIELDS = {field.name for field in fields(Recipe)}
RUN_FIELDS = set(RUN_DEFAULTS) | OPTIONAL_RUN_FIELDS

# The v1 policy archive held only the first 13 files below. v2 includes every
# public package module, including the losses and prior imported by GANTrainer.
POLICY_SOURCE_FILES_V1 = (
    "benchmarks/toy100/train.py", "benchmarks/toy100/models.py",
    "benchmarks/toy100/problems.py", "benchmarks/toy100/metrics.py",
    "benchmarks/toy100/accuracy.py", "benchmarks/toy100/accuracy_evidence.py",
    "benchmarks/toy100/accuracy_gate.py", "lib/toy_models.py",
    "particlegan/training.py", "particlegan/recipes.py",
    "benchmarks/toy100/schedule.py", "benchmarks/toy100/config.py",
    "benchmarks/toy100/__main__.py",
)
POLICY_PUBLIC_SOURCE_FILES = (
    "particlegan/__init__.py", "particlegan/autoencoder.py",
    "particlegan/conditioning.py", "particlegan/diffusion.py",
    "particlegan/discriminators.py", "particlegan/gan_loss.py",
    "particlegan/grad_regularizers.py", "particlegan/locked_shared.py",
    "particlegan/particle_prior.py", "particlegan/recipes.py",
    "particlegan/training.py", "particlegan/vicreg_loss.py",
)
POLICY_SOURCE_SCOPE_V1 = "native-policy-limited-v1"
POLICY_SOURCE_SCOPE_V2 = "native-policy-public-package-v2"


def policy_source_scope(provenance: Mapping[str, Any]) -> str:
    """Classify a policy receipt without consulting the current source tree."""
    sources = provenance.get("source_sha256")
    if not isinstance(sources, dict):
        raise ValueError("policy source hash map is absent")
    scope = provenance.get("source_archive_scope")
    version = provenance.get("source_archive_version")
    if scope is None and version is None:
        if set(sources) != set(POLICY_SOURCE_FILES_V1):
            raise ValueError("unversioned policy archive is not the historical 13-file scope")
        return POLICY_SOURCE_SCOPE_V1
    if scope != POLICY_SOURCE_SCOPE_V2 or type(version) is not int or version != 2:
        raise ValueError("policy source archive scope or version differs")
    if not set(POLICY_SOURCE_FILES_V1 + POLICY_PUBLIC_SOURCE_FILES) <= set(sources):
        raise ValueError("full policy archive omits a required public source file")
    return scope


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
    # Optional fields stay absent from resolved old manifests. Explicit zero
    # is equivalent at runtime but should not silently rewrite old receipts.
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
    for key in ("output_noise_std", "input_noise_std", "input_noise_anneal_end"):
        value = run[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if run["output_noise_std"] < 0 or run["input_noise_std"] < 0:
        raise ValueError("noise standard deviations must be nonnegative")
    if not 0 < run["input_noise_anneal_end"] <= 1:
        raise ValueError("input_noise_anneal_end must be in (0, 1]")
    if "output_noise_warmup" in run:
        warmup = run["output_noise_warmup"]
        if (isinstance(warmup, bool) or not isinstance(warmup, (int, float))
                or not math.isfinite(warmup) or not 0 <= warmup <= 1):
            raise ValueError("output_noise_warmup must be a finite fraction in [0, 1]")
    if "output_noise_learnable" in run:
        if type(run["output_noise_learnable"]) is not bool:
            raise ValueError("output_noise_learnable must be a boolean")
        if run["output_noise_learnable"] and run["output_noise_std"] <= 0:
            raise ValueError("output_noise_learnable requires output_noise_std > 0")
    if "toy100_model" in run and run["toy100_model"] != "affine_square_v1":
        raise ValueError("toy100_model must be 'affine_square_v1'")
    if "network_lr_horizon_cap" in run and (
        type(run["network_lr_horizon_cap"]) is not int
        or run["network_lr_horizon_cap"] <= 0
    ):
        raise ValueError("network_lr_horizon_cap must be a positive integer")
    if "network_lr_floor" in run:
        value = run["network_lr_floor"]
        if "network_lr_horizon_cap" not in run:
            raise ValueError("network_lr_floor requires network_lr_horizon_cap")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("network_lr_floor must be a finite fraction in [0, 1]")
    recipe_kwargs = {key: user[key] for key in user if key in RECIPE_FIELDS and key != "name"}
    recipe_kwargs["total_steps"] = run["steps"]
    recipe = get_recipe(**recipe_kwargs)
    if "name" in user:
        recipe = recipe.replace(name=user["name"])
    if (recipe.model != "gan" or recipe.conditioning != "scalar"
            or recipe.encoder_mode != "none" or recipe.prior_kind != "particles"):
        raise ValueError("toy100 requires an unconditional scalar GAN with a learned particle prior")
    if run.get("toy100_model") == "affine_square_v1" and recipe.z_dim != 2:
        raise ValueError("affine_square_v1 requires z_dim=2")
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
        if config.get("toy100_model") == "affine_square_v1":
            # Preserve the scratch probe's RNG order: construct the ordinary
            # prior, redraw its coordinates, construct nn.Linear (which draws
            # its defaults), then replace those defaults with identity.
            with torch.no_grad():
                prior.z.uniform_(-5.0, 5.0)
            generator = nn.Linear(2, 2).to(device)
            with torch.no_grad():
                generator.weight.copy_(torch.eye(2, device=device, dtype=generator.weight.dtype))
                generator.bias.zero_()
        else:
            generator = SimpleMLPGenerator(
                z_dim=recipe.z_dim, hidden_dim=config["g_hidden"], n_hidden=config["n_hidden"]
            ).to(device)
        discriminator = SimpleMLPDiscriminator(
            in_dim=2, hidden_dim=config["d_hidden"], n_hidden=config["n_hidden"],
            fourier=config["fourier"],
        ).to(device)
        if config.get("toy100_model") != "affine_square_v1":
            _init_linear(generator)
        _init_linear(discriminator)
        if config["output_noise_std"]:
            generator = OutputNoise(
                generator, config["output_noise_std"],
                learnable=config.get("output_noise_learnable", False),
            ).to(device)
        if config["input_noise_std"]:
            discriminator = InputNoise(discriminator, seed=seed + 901, device=device)
        return GANTrainer(
            recipe, generator, discriminator, prior=prior, seed=seed,
            optimizer_options={"fused": config["fused_adam"]},
        )


def _set_output_sigma(trainer: GANTrainer, config: Mapping[str, Any], completed_steps: int) -> float:
    sigma = linear_output_noise(
        config["output_noise_std"], completed_steps, config["steps"],
        config.get("output_noise_warmup", 0.0),
    )
    if config["output_noise_std"]:
        # GANTrainer deep-copies G for EMA. Its scalar noise standard deviation
        # is not an EMA parameter; both copies must follow the same schedule.
        trainer.G.std = sigma
        trainer.ema_G.std = sigma
    return sigma


def _effective_output_sigma(generator: nn.Module) -> float:
    value = generator.effective_std()
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def _learnable_output_receipt(trainer: GANTrainer, initial_std: float) -> dict[str, Any]:
    parameter = trainer.G.output_scale.raw_scale
    g_groups = [index for index, group in enumerate(trainer.opt_g.param_groups)
                if any(value is parameter for value in group["params"])]
    d_groups = [index for index, group in enumerate(trainer.opt_d.param_groups)
                if any(value is parameter for value in group["params"])]
    if g_groups != [0] or d_groups:
        raise RuntimeError("learnable output scale must belong only to the generator optimizer group")
    return {
        "initial_std": float(initial_std),
        "added_trainable_parameters": parameter.numel(),
        "parameter": "G.output_scale.raw_scale",
        "optimizer": "G",
        "optimizer_group": g_groups[0],
        "initial_output_sigma_live": _effective_output_sigma(trainer.G),
        "initial_output_sigma_ema": _effective_output_sigma(trainer.ema_G),
    }


def _source_provenance(*, include_policy: bool = False) -> dict[str, Any]:
    paths = POLICY_SOURCE_FILES_V1[:10]
    if include_policy:
        public = {str(path.relative_to(ROOT)) for path in ROOT.glob("particlegan/**/*.py")}
        if not set(POLICY_PUBLIC_SOURCE_FILES) <= public:
            raise RuntimeError("required public package source is missing")
        paths = tuple(sorted(set(POLICY_SOURCE_FILES_V1) | public))
    hashes = {}
    for name in paths:
        path = ROOT / name
        if not path.exists():
            if include_policy:
                raise RuntimeError(f"required policy source is missing: {name}")
            continue
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    provenance = {"git_sha": git_sha, "source_sha256": hashes,
                  "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    if include_policy:
        provenance.update(source_archive_scope=POLICY_SOURCE_SCOPE_V2,
                          source_archive_version=2)
    return provenance


def _write_source_archive(directory: Path, provenance: dict[str, Any]) -> None:
    """Store the exact policy-run source bytes for portable offline regrading."""
    archive = directory / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, digest in sorted(provenance["source_sha256"].items()):
            contents = (ROOT / name).read_bytes()
            if hashlib.sha256(contents).hexdigest() != digest:
                raise RuntimeError(f"source changed before archive: {name}")
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            info.mode = 0o644
            info.mtime = 0
            stream.addfile(info, io.BytesIO(contents))
    provenance["source_archive_file"] = archive.name
    provenance["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()


def verify_source_archive(directory: Path, provenance: Mapping[str, Any]) -> None:
    """Verify the policy archive against its manifest without the live tree."""
    policy_source_scope(provenance)
    archive = directory / provenance["source_archive_file"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != provenance["source_archive_sha256"]:
        raise ValueError("policy source archive SHA-256 mismatch")
    expected = provenance["source_sha256"]
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        names = [member.name for member in members]
        if (len(names) != len(set(names)) or set(names) != set(expected)
                or any(not member.isfile() for member in members)):
            raise ValueError("policy source archive members differ from provenance")
        for member in members:
            contents = stream.extractfile(member).read()
            if hashlib.sha256(contents).hexdigest() != expected[member.name]:
                raise ValueError(f"policy source archive member SHA-256 mismatch: {member.name}")


def _verify_live_source(provenance: Mapping[str, Any]) -> None:
    for name, digest in provenance["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source changed during training: {name}")


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _model_policy_receipt(trainer: GANTrainer, config: Mapping[str, Any]) -> dict[str, Any]:
    generator = trainer.G.model if isinstance(trainer.G, OutputNoise) else trainer.G
    discriminator = trainer.D.model if isinstance(trainer.D, InputNoise) else trainer.D
    receipt = {
        "toy100_model": config.get("toy100_model", "mlp_v1"),
        "generator_class": type(generator).__name__,
        "generator_wrapper_class": type(trainer.G).__name__ if generator is not trainer.G else None,
        "discriminator_class": type(discriminator).__name__,
        "discriminator_wrapper_class": type(trainer.D).__name__ if discriminator is not trainer.D else None,
        "prior_class": type(trainer.prior).__name__,
        "generator_parameters": sum(p.numel() for p in trainer.G.parameters()),
        "generator_base_parameters": sum(p.numel() for p in generator.parameters()),
        "discriminator_parameters": sum(p.numel() for p in trainer.D.parameters()),
        "prior_parameters": sum(p.numel() for p in trainer.prior.parameters()),
    }
    if config.get("toy100_model") == "affine_square_v1":
        receipt.update({
            "prior_initialization": "uniform_square",
            "prior_scale": 5.0,
            "prior_shape": list(trainer.prior.z.shape),
            "prior_initial_min": float(trainer.prior.z.detach().min()),
            "prior_initial_max": float(trainer.prior.z.detach().max()),
            "prior_initial_sha256": _tensor_sha256(trainer.prior.z),
            "generator_initial_weight": generator.weight.detach().cpu().tolist(),
            "generator_initial_bias": generator.bias.detach().cpu().tolist(),
            "generator_initial_weight_sha256": _tensor_sha256(generator.weight),
            "generator_initial_bias_sha256": _tensor_sha256(generator.bias),
        })
    return receipt


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
    policy_enabled = ("toy100_model" in resolved or "network_lr_horizon_cap" in resolved
                      or "network_lr_floor" in resolved)
    provenance = (_source_provenance(include_policy=True) if policy_enabled
                  else _source_provenance())
    if policy_enabled:
        _write_source_archive(out_dir, provenance)
        verify_source_archive(out_dir, provenance)
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
    # Large production runs retain all scored draws at the final five checks,
    # plus a separate 100k-draw holdout. Small smoke runs keep their old scope.
    accuracy_enabled = budget >= 1000 and resolved["eval_samples"] >= EVAL_N
    accuracy_steps: list[int] = []
    if accuracy_enabled:
        from .accuracy import PROTOCOL as ACCURACY_PROTOCOL
        from .accuracy_gate import HOLDOUT_N, HOLDOUT_SEED_OFFSETS
        from .accuracy_evidence import AccuracyEvidence
        from .gate import MIN_STABLE_CHECKS

        accuracy_steps = eval_steps[-MIN_STABLE_CHECKS:]
    summary: dict[str, Any] = {
        "status": "running", "problem": resolved["problem"], "budget_steps": budget,
        "config": resolved, "eval_steps": eval_steps, "snapshot_steps": snap_steps,
        "provenance": provenance, "environment": environment,
    }
    if "network_lr_horizon_cap" in resolved:
        summary["network_lr_horizon_cap"] = resolved["network_lr_horizon_cap"]
    if "network_lr_floor" in resolved:
        summary["network_lr_floor"] = resolved["network_lr_floor"]
    if accuracy_enabled:
        summary["accuracy"] = {
            "protocol": ACCURACY_PROTOCOL,
            "check_steps": accuracy_steps,
            "sample_count": resolved["eval_samples"],
            "holdout_samples": HOLDOUT_N,
            "holdout_seed_offsets": HOLDOUT_SEED_OFFSETS,
        }
        summary["accuracy_check_steps"] = accuracy_steps
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
    final_accuracy: dict[str, dict[str, Any]] = {}
    device_seed = lambda value: torch.Generator(device=device).manual_seed(value)

    try:
        trainer = make_trainer(resolved, recipe)
        _set_output_sigma(trainer, resolved, trainer.completed_steps)
        if policy_enabled:
            summary["model_policy"] = _model_policy_receipt(trainer, resolved)
        if resolved.get("output_noise_learnable", False):
            summary["learnable_output_noise"] = _learnable_output_receipt(
                trainer, resolved["output_noise_std"],
            )
        if resolved["output_noise_std"]:
            # OutputNoise uses the global stream during updates. Evaluation
            # forks it, so denser observations cannot alter the training path.
            torch.manual_seed(resolved["seed"])
        train_data_rng = device_seed(resolved["seed"])
        target_rng = device_seed(resolved["seed"] + 401)
        target = sample_real(
            resolved["problem"], max(resolved["eval_samples"], resolved["snapshot_samples"]),
            device=device, generator=target_rng,
        )
        target_eval = target[:resolved["eval_samples"]].detach().cpu().numpy()
        accuracy_evidence = (
            AccuracyEvidence(resolved, out_dir, eval_steps, target)
            if accuracy_enabled else None
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
                        accuracy = (accuracy_evidence.observe(step, model, draw, metrics)
                                    if accuracy_evidence is not None else None)
                        elapsed = time.perf_counter() - start
                        event = {"event": "eval", "step": step, "model": model,
                                 "metrics": metrics, "elapsed": elapsed}
                        if resolved.get("output_noise_learnable", False):
                            event["output_sigma"] = _effective_output_sigma(
                                trainer.ema_G if model == "ema" else trainer.G,
                            )
                        if accuracy is not None:
                            event["accuracy"] = accuracy
                            final_accuracy[model] = accuracy
                        logger.event(event)
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
                        sigma_text = (f" sigma={event['output_sigma']:.6f}"
                                      if "output_sigma" in event else "")
                        logger.say(
                            f"EVAL step={step}/{budget} model={model} "
                            f"modes={metrics['modes']}/100 hq={metrics['hq']:.4f} "
                            f"pass={metrics['passed']}{sigma_text} elapsed={elapsed:.1f}s"
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
                    target=target_eval,
                )
            eval_seconds += time.perf_counter() - observed_start

        # Capture actual unmodified G and prior, before the first optimizer step.
        observe(0)
        for step in range(1, budget + 1):
            step_start = time.perf_counter()
            _set_output_sigma(trainer, resolved, trainer.completed_steps)
            if resolved["input_noise_std"]:
                trainer.D.sigma = linear_input_noise(
                    resolved["input_noise_std"], trainer.completed_steps,
                    budget, resolved["input_noise_anneal_end"],
                )
            real = sample_real(
                resolved["problem"], recipe.batch_size, device=device, generator=train_data_rng,
            )
            stats = step_with_policy(
                trainer,
                real,
                network_lr_horizon_cap=resolved.get("network_lr_horizon_cap"),
                network_lr_floor=resolved.get("network_lr_floor"),
                generator_real=lambda: sample_real(
                    resolved["problem"], recipe.batch_size, device=device,
                    generator=train_data_rng,
                ),
            )
            _set_output_sigma(trainer, resolved, trainer.completed_steps)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds += time.perf_counter() - step_start
            losses = {key: float(stats[key]) for key in
                      ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty")}
            if not all(math.isfinite(value) for value in losses.values()):
                raise FloatingPointError(f"nonfinite training loss at step {step}: {losses}")
            train_event = {"event": "train", "step": step,
                           "elapsed": time.perf_counter() - start, **losses}
            if "network_lr_horizon_cap" in resolved:
                train_event.update(policy_rate_action(
                    trainer, step,
                    network_lr_horizon_cap=resolved["network_lr_horizon_cap"],
                    network_lr_floor=resolved.get("network_lr_floor"),
                ))
            if resolved.get("output_noise_learnable", False):
                train_event["output_sigma_live"] = _effective_output_sigma(trainer.G)
                train_event["output_sigma_ema"] = _effective_output_sigma(trainer.ema_G)
            logger.event(train_event)
            if step == 1 or step % resolved["log_interval"] == 0 or step == budget:
                sigma_text = (f" sigma={train_event['output_sigma_live']:.6f}"
                              if "output_sigma_live" in train_event else "")
                logger.say(
                    f"TRAIN step={step}/{budget} d={losses['loss_d']:.4f} "
                    f"g={losses['loss_g']:.4f}{sigma_text} train_seconds={train_seconds:.1f}"
                )
            if step in eval_set or step in snap_set:
                observe(step)

        if accuracy_enabled:
            holdout_start = time.perf_counter()
            summary["accuracy"], summary["holdout"] = accuracy_evidence.finish(trainer)
            summary["holdout_samples_file"] = "holdout_samples.npz"
            summary["holdout_sha256"] = hashlib.sha256(
                (out_dir / "holdout_samples.npz").read_bytes()
            ).hexdigest()
            summary["quality_check_sha256"] = {
                str(step): hashlib.sha256(
                    (out_dir / "quality_checks" / f"step_{step:06d}.npz").read_bytes()
                ).hexdigest()
                for step in accuracy_steps
            }
            summary["final_accuracy"] = final_accuracy
            eval_seconds += time.perf_counter() - holdout_start

        summary.update({
            "status": "complete", "completed_steps": trainer.completed_steps,
            "first_full_coverage_step": first_full, "first_pass_step": first_pass,
            "stable_pass_step": stable_pass, "final": final_metrics,
            "final_samples_file": "final_samples.npz",
            "final_samples_sha256": hashlib.sha256(
                (out_dir / "final_samples.npz").read_bytes()
            ).hexdigest(),
            "train_seconds": train_seconds, "eval_seconds": eval_seconds,
            "total_seconds": time.perf_counter() - start,
            "steps_per_second": budget / train_seconds,
        })
        if resolved.get("output_noise_learnable", False):
            summary["learnable_output_noise"].update({
                "final_output_sigma_live": _effective_output_sigma(trainer.G),
                "final_output_sigma_ema": _effective_output_sigma(trainer.ema_G),
                "final_base_std_live": float(trainer.G.output_scale().detach()),
                "final_base_std_ema": float(trainer.ema_G.output_scale().detach()),
            })
        if policy_enabled:
            _verify_live_source(provenance)
            verify_source_archive(out_dir, provenance)
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
