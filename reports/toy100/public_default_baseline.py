"""Fresh ring baseline using the package default; historical scores stay separate.

Use --prepare-only to record parameters without constructing or training models.
Training writes one flushed JSON object per observation to metrics.jsonl/stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

import torch

from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.training import input_noise_std, output_noise_std
from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from reports.toy100.h_stability.convergence_gate import ConvergenceGate

ROOT = Path(__file__).resolve().parents[2]
MASTER = "0ff9a7af"
BUDGETS = {"hold": 7500, "shift": 3600}
SEED = 0  # Existing ring host seed; no seed-search CLI.
SHIFT_STEP = 2400


def declaration(protocol):
    recipe = get_recipe(total_steps=BUDGETS[protocol])
    files = sorted((ROOT / "particlegan").glob("*.py")) + [
        Path(__file__), ROOT / "benchmarks/locked_shared/mode_hold.py",
        ROOT / "benchmarks/locked_shared/mlp.py",
        ROOT / "reports/toy100/h_stability/convergence_gate.py",
    ]
    return {
        "schema": 1, "experiment": "public_default_k3p_v1", "status": "NOT_RUN",
        "master_reference": MASTER, "protocol": protocol, "seed": SEED,
        "recipe": recipe.to_dict(), "training_api": "particlegan.GANTrainer",
        "model": {"hidden": mode_hold.HIDDEN, "layers": mode_hold.N_HIDDEN,
                  "fourier": mode_hold.FOURIER, "z_dim": recipe.z_dim},
        "optimizer_options": {"foreach": False, "fused": False},
        "evaluation": {"samples": mode_hold.EVAL_N, "modes": 8, "min_hq": .90,
                       "live_is_primary": True, "shift_step": SHIFT_STEP,
                       "shift": [1., 0.], "deadline_step": 2800,
                       "hold_gate": ConvergenceGate().declaration(), "extension": 300},
        "historical_scores_inherited": False,
        "horizon_independent": False,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in files},
    }


def make_trainer(recipe, device):
    torch.manual_seed(SEED)
    generator = SimpleMLPGenerator(recipe.z_dim, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2).to(device)
    critic = SimpleMLPDiscriminator(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER).to(device)
    return GANTrainer(recipe, generator, critic, seed=SEED,
                      optimizer_options={"foreach": False, "fused": False})


def measure(trainer, means, *, ema=False):
    stream = torch.Generator(device=trainer.device).manual_seed(SEED + 9)
    return mode_hold.diversity(trainer.sample(mode_hold.EVAL_N, ema=ema, generator=stream), means)


def window(points, expected_steps):
    expected = list(expected_steps)
    expected_set = set(expected)
    selected = [p for p in points if p["step"] in expected_set]
    complete = [p["step"] for p in selected] == expected
    passing = [p["modes"] == 8 and .90 <= p["hq"] <= 1 for p in selected]
    return {"checks": len(selected), "expected_checks": len(expected), "complete": complete,
            "passing_checks": sum(passing), "pass_all": complete and bool(expected) and all(passing),
            "min_hq": min((p["hq"] for p in selected), default=None),
            "min_modes": min((p["modes"] for p in selected), default=None),
            "failing_steps": [p["step"] for p, good in zip(selected, passing) if not good]}


def shift_summary(live, frozen):
    stationary = window(live, range(1000, 1201, 50))
    prehold = window(live, range(1210, 2401, 10))
    deadline = window(live, range(2800, 3601, 10))
    control = window(frozen, range(2800, 3601, 10))
    passed = (stationary["pass_all"] and prehold["pass_all"] and deadline["pass_all"]
              and control["complete"] and control["passing_checks"] == 0)
    return {"status": "PASS" if passed else "FAIL", "stationary": stationary,
            "prehold": prehold, "deadline": deadline, "frozen_deadline": control}


def run(protocol, output, device, *, prepare_only=False):
    manifest = declaration(protocol)
    output.mkdir(parents=True, exist_ok=False)
    manifest["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest["tracked_tree_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
    manifest["requested_device"] = device
    (output / "declaration.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in manifest["source_sha256"]:
            archive.write(ROOT / name, name)
    if prepare_only:
        print(json.dumps({"event": "prepared", "status": "NOT_RUN", "output": str(output)}), flush=True)
        return manifest
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --prepare-only for a zero-training declaration")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    recipe = Recipe(**manifest["recipe"])
    trainer = make_trainer(recipe, device)
    stream = torch.Generator(device=device).manual_seed(SEED)
    means = mode_hold.ring_means().to(device)
    gate = ConvergenceGate()
    live, control = [], []
    frozen = None
    start = time.monotonic()
    with (output / "metrics.jsonl").open("w", buffering=1) as log:
        for step in range(1, recipe.total_steps + 1):
            idx = torch.randint(0, 8, (recipe.batch_size,), device=device, generator=stream)
            real = means[idx] + mode_hold.SIGMA * torch.randn(
                recipe.batch_size, 2, device=device, generator=stream)
            observe = (step % 50 == 0 if step <= 1200 else protocol == "hold" or step % 10 == 0)
            stats = trainer.step(real, collect_stats=observe)
            if observe:
                point = {"step": step, **measure(trainer, means)}
                live.append(point)
                row = {"event": "observation", **point, "ema": measure(trainer, means, ema=True),
                       "penalty": stats["penalty_stats"],
                       "lr_g_prior": [g["lr"] for g in trainer.opt_g.param_groups],
                       "lr_d": [g["lr"] for g in trainer.opt_d.param_groups],
                       "input_noise": input_noise_std(recipe, step - 1),
                       "output_noise": output_noise_std(recipe, step - 1)}
                if frozen is not None:
                    frozen_point = {"step": step, **measure(frozen, means)}
                    control.append(frozen_point)
                    row["frozen"] = frozen_point
                if protocol == "hold" and step > 1200 and not gate.done:
                    gate.observe(point)
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if protocol == "shift" and step == SHIFT_STEP:
                checkpoint = trainer.state_dict()
                torch.save({"trainer": checkpoint, "real_stream": stream.get_state(),
                            "means": means.clone()}, output / "shift-state.pt")
                # Restore the very same complete learner state into the control.
                # load_state_dict also restores global RNG after construction.
                frozen = make_trainer(recipe, device)
                frozen.load_state_dict(checkpoint)
                means.add_(means.new_tensor([1., 0.]))
        if protocol == "shift":
            result = shift_summary(live, control)
        else:
            end = None if gate.converged_step is None else gate.converged_step + gate.hold_budget
            extension = window(live, [] if end is None else range(end + 1, end + 301))
            result = {"status": "PASS" if gate.status == "PASS" and extension["pass_all"] else "FAIL",
                      "hold": gate.summary(), "extension": extension}
        result.update(experiment=manifest["experiment"], protocol=protocol, device=device,
                      historical_scores_inherited=False, seconds=time.monotonic() - start,
                      torch=str(torch.__version__), completed_steps=trainer.completed_steps,
                      final=live[-1], final_ema=measure(trainer, means, ema=True))
        torch.save({"trainer": trainer.state_dict(), "real_stream": stream.get_state(),
                    "means": means}, output / "final-state.pt")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"event": "complete", **result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=BUDGETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.protocol, args.output, args.device, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
