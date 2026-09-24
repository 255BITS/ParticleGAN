"""Run the actual 100-Gaussian example, with separately scored live/EMA curves.

    python -m benchmarks.locked_shared.grid_study --output reports/grid_study

The 100-mode/HQ criterion is independent of the nine extracted toy bounds.
"""

from __future__ import annotations
from benchmarks.locked_shared.recorded_recipes import GAN_V1

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import torch

from lib.denoising_toy import GaussianGrid, grid_metrics
from lib.toy_metrics import per_mode_moments
from lib.toy_models import mode_coverage
from .baseline import digest, protocol as behavioral_protocol, score_metrics, write_json
from .observation import sustained

TOTAL_STEPS = 7000
INTERVAL = 250
N_EVAL = 20_000
EVAL_SEED = 999
STD = 0.03
REQUIREMENTS = [("modes", ">=", 100), ("hq", ">=", 0.90)]
EXPECTED_STEPS = list(range(INTERVAL, TOTAL_STEPS + 1, INTERVAL))
ARMS = (
    ("stock", {}),
    ("toy_transfer", {"reg_kappa": 1.25, "reg_coeff": 3.0,
                      "lr": 0.0006 * 0.85, "lambda_ep": 0.05}),
    ("penalty_only", {"reg_kappa": 1.25, "reg_coeff": 3.0}),
)


def json_safe(value):
    """Preserve unavailable numbers as null; never turn them into a passing zero."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_example():
    path = Path(__file__).resolve().parents[2] / "examples" / "100gaussians.py"
    spec = importlib.util.spec_from_file_location("particlegan_grid_study_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_protocol(device):
    base = behavioral_protocol()
    root = Path(__file__).resolve().parents[2]
    hashes = dict(base["source_sha256"])
    dependencies = [root / "examples" / "100gaussians.py", *sorted((root / "lib").rglob("*.py"))]
    for path in dependencies:
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {"name": properties.name, "index": device.index if device.index is not None else torch.cuda.current_device(),
               "total_memory": properties.total_memory, "capability": [properties.major, properties.minor]}
    return {"version": "actual-100gaussians-v1", "training_entrypoint": "examples/100gaussians.py:train",
            "seed": 0, "device": str(device), "gpu": gpu,
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "platform": platform.platform(), "threads": torch.get_num_threads(),
            "total_steps": TOTAL_STEPS, "batch_size": 256, "num_particles": 20_000,
            "fourier": 2, "evaluation_samples": N_EVAL, "evaluation_seed": EVAL_SEED,
            "std": STD, "coverage_min_count": 10, "requirements": REQUIREMENTS,
            "expected_steps": EXPECTED_STEPS, "minimum_passing_suffix": 5,
            "evaluation": "fresh independent device Generator(seed=999) for each live/EMA measurement",
            "timing": "wall seconds include setup and evaluation; training seconds exclude callback/maintenance",
            "distribution_metrics": "grid_metrics plus per_mode_moments, min_count=20; diagnostic only",
            "source_sha256": hashes}


def resolved_kwargs(example, output, name, device, overrides):
    legacy = GAN_V1
    parameters = inspect.signature(example.train).parameters
    required = {"reg_kappa", "metric_callback", "metric_interval", "save_plots"}
    if not required <= parameters.keys():
        raise RuntimeError(f"example train is missing measurement arguments: {sorted(required - parameters.keys())}")
    values = {key: p.default for key, p in parameters.items() if p.default is not inspect.Parameter.empty}
    values.update(epochs=7, steps_per_epoch=1000, batch_size=256, num_particles=20_000,
                  fourier=2, seed=0, device_str=str(device), out_dir=str(output / name),
                  return_details=True, metric_interval=INTERVAL, save_plots=False,
                  lr=legacy.lr, d_lr_mult=legacy.d_lr_mult, prior_lr_mult=legacy.prior_lr_mult, beta1=legacy.betas[0], beta2=legacy.betas[1],
                  reg_coeff=legacy.reg_coeff, reg_kappa=legacy.reg_kappa, lambda_ep=legacy.prior_reg)
    values.update(overrides)
    values["metric_callback"] = "benchmarks.locked_shared.grid_study:checkpoint_callback"
    return values


def convergence(row, kind):
    curve = [{"step": p["step"], "seconds": p["seconds"], **p.get(kind, {})} for p in row.get("curve", [])]
    result = sustained(curve, REQUIREMENTS, expected_steps=EXPECTED_STEPS)
    by_step = {p["step"]: p for p in row.get("curve", [])}
    for key in ("first_pass", "stable_from", "confirmed"):
        point = by_step.get(result.get(key + "_step"), {})
        result[key + "_seconds"] = point.get("seconds")
        result[key + "_train_seconds"] = point.get("train_seconds")
    return result


def coverage_status(row, kind):
    curve = row.get("curve", [])
    if row.get("error"):
        return "ERROR"
    if [p["step"] for p in curve] != EXPECTED_STEPS:
        return "INCOMPLETE"
    cells = score_metrics(curve[-1].get(kind, {}), REQUIREMENTS)
    if any(cell["status"] == "MISSING" for cell in cells):
        return "MISSING"
    return "PASS" if all(cell["status"] == "PASS" for cell in cells) else "FAIL"


def fmt(value, spec=".3f"):
    return format(value, spec) if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def render(report, destination):
    lines = ["# Actual 100-Gaussian comparison", "",
             "Runs the existing `examples/100gaussians.py` trainer: seed 0, 7,000 updates, 20,000 particles, "
             "batch 256, Fourier 2. Arms run sequentially on the same device. Each checkpoint evaluates "
             "20,000 fresh fixed-seed draws separately for live and EMA weights.", "",
             "**100-Gaussian coverage/HQ PASS means all 100 modes have at least 10 HQ samples and HQ ≥90% at the final step. "
             "It does not certify the nine-toy suite or distributional calibration.** A stable suffix requires at least "
             "five consecutive passing observations through step 7,000, with the entire 250-step observation schedule present.", "",
             "| Arm | Live modes | Live HQ | Last-five worst HQ | Live coverage/HQ | EMA modes | EMA HQ |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for row in report["rows"]:
        points = row.get("curve", [])
        final = points[-1] if points else {}
        live, ema = final.get("live", {}), final.get("ema", {})
        tail = [p.get("live", {}).get("hq") for p in points if p["step"] >= TOTAL_STEPS - 4 * INTERVAL]
        worst = min(tail) if len(tail) == 5 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in tail) else None
        lines.append(f"| `{row['name']}` | {fmt(live.get('modes'), '.0f')}/100 | {fmt(live.get('hq'), '.2%')} | "
                     f"{fmt(worst, '.2%')} | {coverage_status(row, 'live')} | {fmt(ema.get('modes'), '.0f')}/100 | {fmt(ema.get('hq'), '.2%')} |")
    lines += ["", "First-pass, stable-start and confirmation are observed checkpoint steps, not interpolated convergence times. "
              "Stable-start is retrospective: every later observation must pass. Wall time includes setup and measurement; "
              "training time excludes callbacks and maintenance. Throughput uses training time.", "",
              "| Arm / weights | First pass | Stable from | Confirmed | Stable wall / train sec | Final train sec | Updates/sec |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["rows"]:
        for kind in ("live", "ema"):
            summary = convergence(row, kind)
            train_seconds = row.get("train_seconds")
            throughput = TOTAL_STEPS / train_seconds if row.get("finished") and isinstance(train_seconds, (int, float)) and train_seconds > 0 else None
            lines.append(f"| `{row['name']}` / {kind} | {fmt(summary['first_pass_step'], '.0f')} | "
                         f"{fmt(summary['stable_from_step'], '.0f')} | {fmt(summary['confirmed_step'], '.0f')} | "
                         f"{fmt(summary['stable_from_seconds'], '.1f')} / {fmt(summary['stable_from_train_seconds'], '.1f')} | "
                         f"{fmt(train_seconds, '.1f')} | {fmt(throughput, '.1f')} |")
    lines += ["", "Final distribution diagnostics use separate fixed-seed 20,000-sample fake and real draws, isolated from training. "
              "Width and core ratios near 1 indicate matching scale; covariance ratios expose collapsed axes. "
              "TV and SW1 are lower-is-better. These measurements have no tuned PASS threshold. "
              "Width/core summaries omit modes with fewer than 20 samples; audited counts are shown.", "",
              "| Arm / weights | Width / core ratio | Covariance min / max | Audited modes | Mode TV | SW1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["rows"]:
        for kind in ("live", "ema"):
            metrics = row.get("distribution", {}).get(kind, {})
            lines.append(f"| `{row['name']}` / {kind} | {fmt(metrics.get('per_mode_std_ratio'))} / {fmt(metrics.get('per_mode_core_ratio'))} | "
                         f"{fmt(metrics.get('per_mode_cov_eig_min_ratio'))} / {fmt(metrics.get('per_mode_cov_eig_max_ratio'))} | "
                         f"{fmt(metrics.get('per_mode_cov_audited_modes'), '.0f')} | {fmt(metrics.get('mode_tv'))} | {fmt(metrics.get('sw1'))} |")
    lines += ["", "`stock` keeps the trainer's recipe. `toy_transfer` uses cap κ=1.25, coefficient=3, "
              "LR=0.00051 and prior regularization=0.05. `penalty_only` changes only cap κ/coefficient. "
              "The existing 60%-delay cosine schedule and 5% floor remain shared.", "",
              "Exact resolved arguments, source hashes, curves, convergence times and raw diagnostics are in [results.json](results.json). "
              "Nonfinite values become null and cannot pass. Missing observations cannot certify stability.", ""]
    for row in report["rows"]:
        if row.get("error"):
            lines += [f"`{row['name']}` error: {row['error'].splitlines()[-1]}", ""]
    destination.write_text("\n".join(lines))


@torch.no_grad()
def distribution_metrics(generator, prior, device):
    previous_mode = generator.training
    generator.eval()
    try:
        rng = torch.Generator(device=device).manual_seed(EVAL_SEED)
        fake = generator(prior.sample(N_EVAL, generator=rng)[0])
        toy = GaussianGrid(device=device, std=STD, classes=1)
        labels = torch.zeros(N_EVAL, dtype=torch.long, device=device)
        real = toy.sample(labels, torch.Generator(device=device).manual_seed(EVAL_SEED))
        values = grid_metrics(fake, labels, toy, real, seed=EVAL_SEED)
        values.update(per_mode_moments(fake, min_count=20, std=STD))
        return json_safe(values)
    finally:
        generator.train(previous_mode)


def run(output, device, *, training_api=False):
    output = Path(output)
    if (output / "results.json").exists():
        raise FileExistsError(f"refusing to overwrite {output / 'results.json'}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    example = load_example()
    fingerprint = make_protocol(device)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": fingerprint,
              "protocol_sha256": digest(fingerprint), "rows": []}
    for name, overrides in ARMS:
        kwargs = resolved_kwargs(example, output, name, device, {**overrides, "use_training_api": training_api})
        report["rows"].append({"name": name, "overrides": overrides, "kwargs": kwargs,
                               "config_sha256": digest(kwargs), "curve": []})
    output.mkdir(parents=True, exist_ok=True)

    def save():
        for entry in report["rows"]:
            entry["convergence"] = {kind: convergence(entry, kind) for kind in ("live", "ema")}
        write_json(output / "results.json", json_safe(report))
        render(report, output / "README.md")

    save()
    for row in report["rows"]:
        print(f"START 100gaussians arm={row['name']} steps={TOTAL_STEPS} device={device}", flush=True)
        synchronize(device)
        started = time.perf_counter()

        @torch.no_grad()
        def checkpoint_callback(step, generator, prior, ema_generator, ema_prior, train_seconds):
            if step not in EXPECTED_STEPS or (row["curve"] and step <= row["curve"][-1]["step"]):
                raise ValueError(f"unexpected or duplicate evaluation step {step}")
            point = {"step": step, "train_seconds": float(train_seconds)}
            for kind, model, latent in (("live", generator, prior), ("ema", ema_generator, ema_prior)):
                modes, hq = mode_coverage(model, latent, device, n_eval=N_EVAL, std=STD, min_count=10,
                                         sample_generator=torch.Generator(device=device).manual_seed(EVAL_SEED))
                point[kind] = json_safe({"modes": modes, "hq": hq})
            synchronize(device)
            point["seconds"] = time.perf_counter() - started
            row["curve"].append(point)
            save()
            print(json.dumps({"event": "100G_CHECKPOINT", "arm": row["name"], **point}, allow_nan=False), flush=True)

        kwargs = dict(row["kwargs"], metric_callback=checkpoint_callback)
        details = None
        try:
            details = example.train(**kwargs)
            row["train_seconds"] = float(details["train_seconds"])
            row["trainer_total_seconds"] = float(details["total_seconds"])
            row["distribution"] = {
                "live": distribution_metrics(details["G"], details["prior"], device),
                "ema": distribution_metrics(details["ema_G"], details["ema_prior"], device),
            }
            synchronize(device)
            row["finished"] = True
        except Exception:
            row["error"] = traceback.format_exc()
        finally:
            synchronize(device)
            row["seconds"] = time.perf_counter() - started
            save()
            print(f"DONE 100gaussians arm={row['name']} live={coverage_status(row, 'live')} "
                  f"ema={coverage_status(row, 'ema')} wall_seconds={row['seconds']:.1f}", flush=True)
            del details
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--training-api", action="store_true", help="Benchmark the public GANTrainer on the same reference task.")
    args = parser.parse_args()
    report = run(args.output, torch.device(args.device), training_api=args.training_api)
    return 0 if all(row.get("finished") for row in report["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
