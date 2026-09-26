"""Measure KA2 arrival and retention through the unmodified public GANTrainer.

The original public K3P baseline's model, seed, streams and real-batch reuse are
preserved. Only the KA2 formulation and requested LR schedule change. The run
extends to 4600 updates while retaining the baseline's absolute noise milestones
(input noise ends at 360; output noise reaches its maximum at 720).

No seed variants, private update loops or optimizer patches. Start with constant
LR; the optional decay arm is intended only for investigating a failed run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from particlegan import GANTrainer, get_recipe
from particlegan.training import input_noise_std, output_noise_std
from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator

SEED = 0
SHIFT_STEP = 2400
NOISE_HORIZON = 3600
OBSERVE_EVERY = 10


def digest(value):
    """Portable content hash, including tensor dtype/shape but not device."""
    out = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            cpu = item.detach().cpu().contiguous()
            out.update(json.dumps(["tensor", str(cpu.dtype), list(cpu.shape)]).encode())
            out.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            out.update(b"dict[")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
            out.update(b"]")
        elif isinstance(item, (list, tuple)):
            out.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            out.update(b"]")
        else:
            out.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())

    visit(value)
    return out.hexdigest()


def make_recipe(schedule, steps):
    baseline = get_recipe(total_steps=NOISE_HORIZON)
    # GANTrainer enforces its total_steps budget. Express the existing absolute
    # noise milestones as fractions of the longer budget using public settings.
    overrides = dict(total_steps=steps,
                     input_noise_anneal_end=baseline.input_noise_anneal_end * NOISE_HORIZON / steps,
                     output_noise_warmup=baseline.output_noise_warmup * NOISE_HORIZON / steps)
    if schedule == "constant":
        overrides.update(lr_floor=1.0, network_lr_floor=1.0)
    recipe = get_recipe(**overrides)
    assert recipe.input_noise_anneal_end * recipe.total_steps == 360.0
    assert recipe.output_noise_warmup * recipe.total_steps == 720.0
    return recipe


def make_trainer(recipe, device):
    # Identical construction order to public_default_baseline.py, including
    # initializing networks on CPU before moving them to the training device.
    torch.manual_seed(SEED)
    generator = SimpleMLPGenerator(recipe.z_dim, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2).to(device)
    critic = SimpleMLPDiscriminator(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER).to(device)
    return GANTrainer(recipe, generator, critic, seed=SEED,
                      optimizer_options={"foreach": False, "fused": False})


def measure(trainer, means, *, ema=False):
    stream = torch.Generator(device=trainer.device).manual_seed(SEED + 9)
    return mode_hold.diversity(trainer.sample(mode_hold.EVAL_N, ema=ema, generator=stream), means)


def good(point):
    return point["modes"] == 8 and .90 <= point["hq"] <= 1.0


def window(points, begin, end, spacing=OBSERVE_EVERY):
    selected = [p for p in points if begin <= p["step"] <= end and (p["step"] - begin) % spacing == 0]
    expected = list(range(begin, end + 1, spacing))
    return dict(checks=len(selected), expected_checks=len(expected),
                complete=[p["step"] for p in selected] == expected,
                passing_checks=sum(good(p) for p in selected),
                failing_steps=[p["step"] for p in selected if not good(p)],
                min_hq=min((p["hq"] for p in selected), default=None),
                min_modes=min((p["modes"] for p in selected), default=None))


def recovery(points, end):
    selected = [p for p in points if SHIFT_STEP < p["step"] <= end]
    first = next((p["step"] for p in selected if good(p)), None)
    after = [p for p in selected if first is not None and p["step"] >= first]
    suffix = []
    for point in reversed(selected):
        if not good(point):
            break
        suffix.append(point)
    suffix.reverse()
    return dict(observed_through=end, first_passing_step=first,
                updates_to_first_pass=None if first is None else first - SHIFT_STEP,
                checks_from_first_pass=len(after), passing_checks_from_first_pass=sum(map(good, after)),
                failing_steps_after_first_pass=[p["step"] for p in after if not good(p)],
                stable_suffix_start=None if not suffix else suffix[0]["step"],
                stable_suffix_checks=len(suffix),
                stable_suffix_span_updates=0 if not suffix else suffix[-1]["step"] - suffix[0]["step"],
                stable_suffix_min_hq=min((p["hq"] for p in suffix), default=None),
                final=None if not selected else selected[-1])


def rates(trainer):
    return {f"{role}_{i}": float(group["lr"])
            for optimizer, roles in zip((trainer.opt_g, trainer.opt_d), trainer.roles)
            for i, (group, role) in enumerate(zip(optimizer.param_groups, roles))}


def state_receipt(trainer, stream, means):
    state = trainer.state_dict()
    return {"step": trainer.completed_steps,
            "trainer_sha256": digest(state),
            "models_sha256": {name: digest(values) for name, values in state["models"].items()},
            "optimizers_sha256": [digest(values) for values in state["optimizers"]],
            "streams_sha256": {name: digest(values) for name, values in state["streams"].items()},
            "cpu_rng_sha256": digest(state["cpu_rng"]),
            "cuda_rng_sha256": digest(state["cuda_rng"]),
            "real_stream_sha256": digest(stream.get_state()), "means_sha256": digest(means)}


def save_checkpoint(path, trainer, stream, means):
    state = trainer.state_dict()
    torch.save({"trainer": state, "real_stream": stream.get_state(), "means": means.clone()}, path)
    return state


def run(args):
    recipe = make_recipe(args.schedule, args.steps)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT / "particlegan").glob("*.py")) + [
        Path(__file__), ROOT / "benchmarks/locked_shared/mode_hold.py",
        ROOT / "benchmarks/locked_shared/mlp.py"]
    manifest = {
        "schema": 1, "experiment": f"ka2_{args.schedule}_lr_public_trainer", "status": "NOT_RUN",
        "schedule": args.schedule, "seed": SEED, "recipe": recipe.to_dict(),
        "training_api": "particlegan.get_recipe + particlegan.GANTrainer.step",
        "generator_real": "reuse discriminator real batch, matching archived public K3P baseline",
        "model": {"hidden": mode_hold.HIDDEN, "layers": mode_hold.N_HIDDEN,
                  "fourier": mode_hold.FOURIER, "z_dim": recipe.z_dim},
        "optimizer_options": {"foreach": False, "fused": False},
        "initialization": "seed 0; original public baseline construction order; no external fixture",
        "noise_horizon": NOISE_HORIZON, "training_budget": args.steps,
        "noise_milestones": {"input_zero_at": 360, "output_full_at": 720},
        "decay_prior_horizon_if_requested": args.steps,
        "evaluation": {"samples": mode_hold.EVAL_N, "modes": 8, "min_hq": .90,
                       "every": OBSERVE_EVERY, "live_is_primary": True,
                       "shift_after_step": SHIFT_STEP, "shift": [1., 0.],
                       "criterion": "time to reach shifted target, then measured stability; no 81/81 gate"},
        "historical_scores_inherited": False, "requested_device": args.device,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tracked_tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    (args.output / "declaration.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(args.output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in manifest["source_sha256"]:
            archive.write(ROOT / name, name)
    if args.prepare_only:
        print(json.dumps({"event": "prepared", "output": str(args.output)}), flush=True)
        return manifest
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    trainer = make_trainer(recipe, args.device)
    stream = torch.Generator(device=args.device).manual_seed(SEED)
    means = mode_hold.ring_means().to(args.device)
    initial_rates = rates(trainer)
    lr_ranges = {name: {"min": value, "max": value, "observed_steps": 0} for name, value in initial_rates.items()}
    live, frozen_points = [], []
    frozen = None
    started = time.monotonic()
    save_checkpoint(args.output / "initial-state.pt", trainer, stream, means)
    torch.save({name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                for name, model in (("G", trainer.G), ("D", trainer.D), ("prior", trainer.prior))},
               args.output / "initial-models-cpu.pt")
    (args.output / "initial.json").write_text(json.dumps(state_receipt(trainer, stream, means), indent=2) + "\n")
    with (args.output / "metrics.jsonl").open("w", buffering=1) as metrics, \
         (args.output / "learning-rates.jsonl").open("w", buffering=1) as lr_log, \
         (args.output / "state-hashes.jsonl").open("w", buffering=1) as states:
        for step in range(1, args.steps + 1):
            idx = torch.randint(0, 8, (recipe.batch_size,), device=args.device, generator=stream)
            real = means[idx] + mode_hold.SIGMA * torch.randn(
                recipe.batch_size, 2, device=args.device, generator=stream)
            observe = step % OBSERVE_EVERY == 0 or step == args.steps
            stats = trainer.step(real, collect_stats=observe)
            actual = rates(trainer)
            for name, value in actual.items():
                lr_ranges[name]["min"] = min(lr_ranges[name]["min"], value)
                lr_ranges[name]["max"] = max(lr_ranges[name]["max"], value)
                lr_ranges[name]["observed_steps"] += 1
            if args.schedule == "constant" and actual != initial_rates:
                raise RuntimeError(f"LR changed at update {step}: {actual} != {initial_rates}")
            lr_log.write(json.dumps({"step": step, **actual}) + "\n")
            if observe:
                point = {"step": step, **measure(trainer, means)}
                live.append(point)
                row = {"event": "observation", **point, "ema": measure(trainer, means, ema=True),
                       "losses": {k: float(v) for k, v in stats.items() if isinstance(v, torch.Tensor)},
                       "penalty": stats["penalty_stats"], "controller": trainer.penalty.diagnostics(),
                       "learning_rates": actual,
                       "input_noise": input_noise_std(recipe, step - 1),
                       "output_noise": output_noise_std(recipe, step - 1)}
                if frozen is not None:
                    frozen_point = {"step": step, **measure(frozen, means)}
                    frozen_points.append(frozen_point)
                    row["frozen"] = frozen_point
                if not all(math.isfinite(v) for v in row["losses"].values()):
                    raise RuntimeError(f"Nonfinite loss at update {step}")
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                if step % 100 == 0 or step == args.steps:
                    receipt = state_receipt(trainer, stream, means)
                    states.write(json.dumps(receipt) + "\n")
                    print(json.dumps({**row, "seconds": time.monotonic() - started}), flush=True)
            if step == SHIFT_STEP:
                checkpoint = save_checkpoint(args.output / "shift-state.pt", trainer, stream, means)
                # Same complete checkpoint control as the archived baseline.
                # Loading restores the global RNG changed during construction.
                frozen = make_trainer(recipe, args.device)
                frozen.load_state_dict(checkpoint)
                means.add_(means.new_tensor([1., 0.]))
                print(json.dumps({"event": "shift", "after_step": step, "shift": [1., 0.]}), flush=True)
        result = {"schema": 1, "status": "COMPLETE", "experiment": manifest["experiment"],
                  "schedule": args.schedule, "completed_steps": trainer.completed_steps,
                  "seconds": time.monotonic() - started, "torch": str(torch.__version__),
                  "cuda_version": torch.version.cuda, "device": args.device,
                  "device_name": torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else "cpu",
                  "stationary": window(live, 1000, 1200, 50),
                  "prehold": window(live, 1210, 2400),
                  "recovery_at_3600": recovery(live, min(3600, args.steps)),
                  "recovery_extended": recovery(live, args.steps),
                  "frozen_control": window(frozen_points, 2410, args.steps),
                  "lr_ranges": lr_ranges, "constant_lr_verified_every_step": args.schedule == "constant",
                  "final": live[-1], "final_ema": measure(trainer, means, ema=True),
                  "final_state": state_receipt(trainer, stream, means)}
        save_checkpoint(args.output / "final-state.pt", trainer, stream, means)
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"event": "complete", **result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", choices=("constant", "decay"), default="constant")
    parser.add_argument("--steps", type=int, default=4600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.steps <= SHIFT_STEP or args.steps % OBSERVE_EVERY:
        parser.error("--steps must be a multiple of 10 greater than the shift at 2400")
    run(args)


if __name__ == "__main__":
    main()
