"""Run the 100-Gaussian example problem on the shared toy runner, with separately
scored live/EMA curves. Each arm is a recipe (the shipped defaults at this
study's shape, GAN_V1's recorded rates/penalty, plus the arm's fields).

    python -m benchmarks.locked_shared.grid_study --output reports/grid_study

The 100-mode/HQ criterion is independent of the nine extracted toy bounds.
"""

from __future__ import annotations
from benchmarks.locked_shared.recorded_recipes import GAN_V1

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import torch

from particlegan import get_recipe
from benchmarks.toy_runner import run as toy_run
from .baseline import digest, protocol as behavioral_protocol, score_metrics, write_json
from .observation import sustained

TOTAL_STEPS = 7000
INTERVAL = 250
N_EVAL = 20_000
STD = 0.03
REQUIREMENTS = [("modes", ">=", 100), ("hq", ">=", 0.90)]
EXPECTED_STEPS = list(range(INTERVAL, TOTAL_STEPS + 1, INTERVAL))
ARMS = (
    ("stock", {}),
    ("toy_transfer", {"reg_kappa": 1.25, "reg_coeff": 3.0,
                      "lr": 0.0006 * 0.85, "prior_reg": 0.05}),
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


def arm_recipe(overrides):
    """The shipped recipe at this study's shape, with GAN_V1's recorded rates and
    penalty (the historical stock arm), plus the arm's recipe-field overrides."""
    legacy = GAN_V1
    fields = dict(batch_size=256, num_particles=20_000, total_steps=TOTAL_STEPS,
                  lr=legacy.lr, d_lr_mult=legacy.d_lr_mult, prior_lr_mult=legacy.prior_lr_mult,
                  betas=legacy.betas, reg_coeff=legacy.reg_coeff, reg_kappa=legacy.reg_kappa,
                  prior_reg=legacy.prior_reg)
    return get_recipe(**{**fields, **overrides})


def make_protocol(device):
    base = behavioral_protocol()
    root = Path(__file__).resolve().parents[2]
    hashes = dict(base["source_sha256"])
    dependencies = [root / "examples" / "100gaussians.py", root / "benchmarks" / "toy_runner.py",
                    *sorted((root / "lib").rglob("*.py"))]
    for path in dependencies:
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {"name": properties.name, "index": device.index if device.index is not None else torch.cuda.current_device(),
               "total_memory": properties.total_memory, "capability": [properties.major, properties.minor]}
    return {"version": "actual-100gaussians-v2-toy-runner",
            "training_entrypoint": "benchmarks/toy_runner.py:run(examples/100gaussians.py:Gaussians100)",
            "seed": 0, "device": str(device), "gpu": gpu,
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "platform": platform.platform(), "threads": torch.get_num_threads(),
            "total_steps": TOTAL_STEPS, "batch_size": 256, "num_particles": 20_000,
            "fourier": 2, "evaluation_samples": N_EVAL, "evaluation_seed": "runner eval stream (seed + 9)",
            "std": STD, "coverage_min_count": 10, "requirements": REQUIREMENTS,
            "expected_steps": EXPECTED_STEPS, "minimum_passing_suffix": 5,
            "evaluation": "the runner's evaluation stream, restarted for each live/EMA measurement",
            "timing": "wall seconds include setup and evaluation",
            "distribution_metrics": "grid_metrics plus per_mode_moments, min_count=20; diagnostic only",
            "source_sha256": hashes}


def convergence(row, kind):
    curve = [{"step": p["step"], "seconds": p["seconds"], **p.get(kind, {})} for p in row.get("curve", [])]
    result = sustained(curve, REQUIREMENTS, expected_steps=EXPECTED_STEPS)
    by_step = {p["step"]: p for p in row.get("curve", [])}
    for key in ("first_pass", "stable_from", "confirmed"):
        point = by_step.get(result.get(key + "_step"), {})
        result[key + "_seconds"] = point.get("seconds")
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
             "Runs the `examples/100gaussians.py` problem on `benchmarks/toy_runner.py`: seed 0, 7,000 updates, 20,000 particles, "
             "batch 256, Fourier 2. Arms run sequentially on the same device. Each checkpoint evaluates "
             "20,000 draws from the runner's restarted evaluation stream, separately for live and EMA weights.", "",
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
              "Stable-start is retrospective: every later observation must pass. Wall time includes setup and measurement, "
              "so updates/sec is a lower bound on training throughput.", "",
              "| Arm / weights | First pass | Stable from | Confirmed | Stable wall sec | Final wall sec | Updates/sec |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["rows"]:
        for kind in ("live", "ema"):
            summary = convergence(row, kind)
            seconds = row.get("seconds")
            throughput = TOTAL_STEPS / seconds if row.get("finished") and isinstance(seconds, (int, float)) and seconds > 0 else None
            lines.append(f"| `{row['name']}` / {kind} | {fmt(summary['first_pass_step'], '.0f')} | "
                         f"{fmt(summary['stable_from_step'], '.0f')} | {fmt(summary['confirmed_step'], '.0f')} | "
                         f"{fmt(summary['stable_from_seconds'], '.1f')} | "
                         f"{fmt(seconds, '.1f')} | {fmt(throughput, '.1f')} |")
    lines += ["", "Final distribution diagnostics use 20,000 fake and real draws from the runner's evaluation stream, isolated from training. "
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
              "The recipe's schedule, applied inside its optimizers, remains shared.", "",
              "Exact resolved recipes, source hashes, curves, convergence times and raw diagnostics are in [results.json](results.json). "
              "Nonfinite values become null and cannot pass. Missing observations cannot certify stability.", ""]
    for row in report["rows"]:
        if row.get("error"):
            lines += [f"`{row['name']}` error: {row['error'].splitlines()[-1]}", ""]
    destination.write_text("\n".join(lines))


def run(output, device):
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
        recipe = arm_recipe(overrides).to_dict()
        report["rows"].append({"name": name, "overrides": overrides, "recipe": recipe,
                               "config_sha256": digest(recipe), "curve": []})
    output.mkdir(parents=True, exist_ok=True)

    def save():
        for entry in report["rows"]:
            entry["convergence"] = {kind: convergence(entry, kind) for kind in ("live", "ema")}
        write_json(output / "results.json", json_safe(report))
        render(report, output / "README.md")

    save()
    for row in report["rows"]:
        print(f"START 100gaussians arm={row['name']} steps={TOTAL_STEPS} device={device}", flush=True)
        started = time.perf_counter()

        def checkpoint(step, measure):
            if step not in EXPECTED_STEPS:
                return
            point = {"step": step}
            for kind in ("live", "ema"):
                values = measure(ema=kind == "ema")
                point[kind] = json_safe({"modes": values["modes"], "hq": values["hq"]})
            point["seconds"] = time.perf_counter() - started
            row["curve"].append(point)
            save()
            print(json.dumps({"event": "100G_CHECKPOINT", "arm": row["name"], **point}, allow_nan=False), flush=True)

        try:
            result = toy_run(example.Gaussians100(distribution=True), recipe=arm_recipe(row["overrides"]),
                             seed=0, device=device, observe_every=TOTAL_STEPS, observer=checkpoint,
                             log_path=output / f"{row['name']}.log")
            row["distribution"] = {kind: json_safe(result[kind]["distribution"]) for kind in ("live", "ema")}
            row["finished"] = True
        except Exception:
            row["error"] = traceback.format_exc()
        finally:
            row["seconds"] = time.perf_counter() - started
            save()
            print(f"DONE 100gaussians arm={row['name']} live={coverage_status(row, 'live')} "
                  f"ema={coverage_status(row, 'ema')} wall_seconds={row['seconds']:.1f}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from benchmarks.toy100.device import apply_device_policy
    apply_device_policy(args.device, log=True)
    report = run(args.output, torch.device(args.device))
    return 0 if all(row.get("finished") for row in report["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
