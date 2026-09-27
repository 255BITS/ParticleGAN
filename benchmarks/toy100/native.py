"""toy100 on the shared problem-only runner (``benchmarks.toy_runner``).

This module declares only the 100-Gaussian problem: the unlabelled sampler
(``problems.sample_real``), the networks, and the frozen metrics/verdict
(``metrics.evaluate_samples`` / ``metrics.passes``). Optimizers and their LR
schedule, loss, critic penalty, prior group, critic input noise, generator
output noise and EMA all come from the shipped recipe through ``ToyRun``.

Two ways to run it::

    # the gate protocol: evidence that gate.py and accuracy_gate.py regrade unchanged
    python -u -m benchmarks.toy100 run --output runs/toy100/native
    # the shared runner CLI, one problem, one JSON line per observation
    python -m benchmarks.toy100.native grid100 --log runs/toy-refactor/toy100_grid100.log

``record`` is the gate-protocol observer: it drives ``ToyRun.step()`` and
writes the same files as the legacy runner (config, summary, events,
snapshots, final draw, terminal quality checks and the 100k holdout). It
only reads learning rates back from the optimizers as receipts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from lib.toy_models import SimpleMLPDiscriminator
from particlegan import get_recipe
from benchmarks.toy_runner import Model, Networks, ToyProblem, ToyRun, main as runner_main

from .metrics import EVAL_N, evaluate_samples, passes
from .problems import PROBLEM_NAMES, sample_real

STEPS = 7000
SEED = 1234
D_HIDDEN = 128
N_HIDDEN = 3
FOURIER = 3
PRIOR_BOX = 5.0
EARLY_EVAL_STEPS = (0, 1, 10, 25, 50, 100)
EVAL_INTERVAL = 250
SNAPSHOT_SAMPLES = 4096
TARGET_SEED_OFFSET, EVAL_SEED_OFFSET = 401, 403


def xavier_(module: nn.Module) -> nn.Module:
    """The benchmark's critic init: ``xavier_uniform_`` weights, zero biases.

    Kept on purpose (#207): the toy100 critic was tuned and gated on it, and
    ``particlegan.init`` has no xavier-scale initializer yet.
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return module


class Toy100(ToyProblem):
    """One of grid100 / rotated100 / staggered100: 100 equal-weight modes, sigma 0.03.

    Networks are the default config's ``affine_square_v1`` model: a 2-D particle
    table drawn uniform on [-5, 5]^2, an identity ``nn.Linear(2, 2)`` generator,
    and a 3x128 Fourier-3 MLP critic with the xavier init above.
    """

    def __init__(self, problem: str = "grid100", *, steps: int = STEPS):
        if problem not in PROBLEM_NAMES:
            raise ValueError(f"unknown toy100 problem {problem!r}")
        self.problem, self.steps, self.name = problem, int(steps), f"toy100_{problem}"

    def recipe(self):
        return get_recipe(z_dim=2, num_particles=20_000, batch_size=2048, total_steps=self.steps)

    def networks(self, recipe, seed):
        prior = recipe.make_prior()
        generator = nn.Linear(2, 2)
        critic = SimpleMLPDiscriminator(in_dim=2, hidden_dim=D_HIDDEN, n_hidden=N_HIDDEN, fourier=FOURIER)
        with torch.no_grad():
            prior.z.uniform_(-PRIOR_BOX, PRIOR_BOX)
            generator.weight.copy_(torch.eye(2))
            generator.bias.zero_()
        return Networks(generator=generator, critics=xavier_(critic), prior=prior)

    def real(self, n, stream):
        return sample_real(self.problem, n, device=stream.device, generator=stream)

    def score(self, samples: torch.Tensor) -> dict:
        return dict(evaluate_samples(samples, self.problem))

    def metrics(self, model):
        return self.score(model.sample(EVAL_N).x)

    def verdict(self, metrics):
        return "PASS" if passes(self.problem, metrics) else "FAIL"


# -- gate-protocol recorder ----------------------------------------------------

def evaluation_steps(budget: int) -> list[int]:
    from .train import evaluation_steps as steps
    return steps(budget, EVAL_INTERVAL, EARLY_EVAL_STEPS)


def native_config(problem: str, *, steps: int = STEPS, device: str = "cpu", seed: int = SEED) -> dict:
    """The executed declaration that ``gate.py`` reads (no legacy recipe switches)."""
    toy = Toy100(problem, steps=steps)
    config = {"problem": problem, "runner": "benchmarks.toy_runner", "declaration": "benchmarks.toy100.native.Toy100",
              "seed": seed, "device": str(device), "steps": steps, "eval_interval": EVAL_INTERVAL,
              "early_eval_steps": list(EARLY_EVAL_STEPS), "eval_samples": EVAL_N,
              "snapshot_samples": SNAPSHOT_SAMPLES, "snapshot_interval": EVAL_INTERVAL,
              "stable_evals": 5, "threads": 1, "batch_size": toy.recipe().batch_size,
              "networks": {"prior": f"uniform[-{PRIOR_BOX}, {PRIOR_BOX}]^2", "generator": "identity Linear(2, 2)",
                           "critic": f"SimpleMLPDiscriminator({D_HIDDEN}x{N_HIDDEN}, fourier={FOURIER}), xavier"},
              "recipe": toy.recipe().to_dict()}
    return json.loads(json.dumps(config))


class _Sampler:
    """``accuracy_evidence.finish`` draws the holdout through ``sample``."""

    def __init__(self, toy: ToyRun):
        self.toy = toy

    @torch.no_grad()
    def sample(self, n, *, ema=False, generator=None):
        nets = self.toy.ema_nets if ema else self.toy.nets
        return Model(self.toy.problem, nets, generator, ema=ema).sample(n).x


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _say(row: dict, progress) -> None:
    line = json.dumps(row, allow_nan=False)
    print(line, flush=True)
    progress.write(line + "\n")


def record(problem: str, out_dir: str | Path, *, steps: int = STEPS, device: str = "cpu",
           seed: int = SEED) -> dict:
    """Train one problem on the shared runner and write regradable gate evidence."""
    from .accuracy import PROTOCOL as ACCURACY_PROTOCOL
    from .accuracy_evidence import AccuracyEvidence
    from .accuracy_gate import HOLDOUT_N, HOLDOUT_SEED_OFFSETS

    config = native_config(problem, steps=steps, device=device, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    (out / "snapshots").mkdir()
    torch.set_num_threads(config["threads"])
    _write_json(out / "config.json", config)
    eval_steps = evaluation_steps(steps)
    accuracy_steps = eval_steps[-5:]
    summary = {"status": "running", "problem": problem, "budget_steps": steps, "config": config,
               "eval_steps": eval_steps, "snapshot_steps": eval_steps,
               "accuracy": {"protocol": ACCURACY_PROTOCOL, "check_steps": accuracy_steps, "sample_count": EVAL_N,
                            "holdout_samples": HOLDOUT_N, "holdout_seed_offsets": HOLDOUT_SEED_OFFSETS},
               "accuracy_check_steps": accuracy_steps}
    _write_json(out / "summary.json", summary)
    events = (out / "events.jsonl").open("x", buffering=1)
    progress = (out / "progress.log").open("x", buffering=1)
    start = time.perf_counter()
    toy = None
    try:
        toy = ToyRun(Toy100(problem, steps=steps), seed=seed, device=device)
        dev = toy.device
        target = sample_real(problem, EVAL_N, device=dev,
                             generator=torch.Generator(device=dev).manual_seed(seed + TARGET_SEED_OFFSET))
        target_np = target.cpu().numpy()
        evidence = AccuracyEvidence(config, out, eval_steps, target)
        sampler, final = _Sampler(toy), {}

        def observe(step):
            clouds = {}
            for model in ("live", "ema"):
                stream = torch.Generator(device=dev).manual_seed(seed + EVAL_SEED_OFFSET)
                draw = sampler.sample(EVAL_N, ema=model == "ema", generator=stream)
                metrics = toy.problem.score(draw)
                accuracy = evidence.observe(step, model, draw, metrics)
                elapsed = time.perf_counter() - start
                events.write(json.dumps({"event": "eval", "step": step, "model": model, "metrics": metrics,
                                         "accuracy": accuracy, "elapsed": elapsed}, allow_nan=False) + "\n")
                final[model] = metrics
                clouds[model] = draw.cpu().numpy()
            np.savez_compressed(out / "snapshots" / f"step_{step:06d}.npz", live=clouds["live"][:SNAPSHOT_SAMPLES],
                                ema=clouds["ema"][:SNAPSHOT_SAMPLES], target=target_np[:SNAPSHOT_SAMPLES])
            if step == steps:
                np.savez_compressed(out / "final_samples.npz", live=clouds["live"], ema=clouds["ema"],
                                    target=target_np)
            _say({"toy": toy.problem.name, "event": "eval", "step": step, "elapsed": round(elapsed, 1),
                  **{f"{m}_{k}": final[m][k] for m in ("live", "ema") for k in ("modes", "hq", "mass_tv", "passed")}},
                 progress)

        observe(0)
        wanted = set(eval_steps)
        for _ in range(steps):
            losses = toy.step()
            step = toy.completed_steps
            row = {key: float(value) for key, value in losses.items() if key != "step"}
            if not all(math.isfinite(value) for value in row.values()):
                raise FloatingPointError(f"nonfinite training loss at step {step}: {row}")
            # Read-only receipts of the rates the recipe-built optimizers applied.
            groups = toy.opt_g.param_groups
            row.update(lr_g=groups[0]["lr"], lr_prior=groups[-1]["lr"], lr_d=toy.opt_d["critic"].param_groups[0]["lr"])
            events.write(json.dumps({"event": "train", "step": step, "elapsed": time.perf_counter() - start, **row},
                                    allow_nan=False) + "\n")
            if step in wanted:
                observe(step)
        summary["accuracy"], summary["holdout"] = evidence.finish(sampler)
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        summary.update({
            "status": "complete", "completed_steps": toy.completed_steps, "final": final,
            "final_samples_file": "final_samples.npz", "final_samples_sha256": digest(out / "final_samples.npz"),
            "holdout_samples_file": "holdout_samples.npz", "holdout_sha256": digest(out / "holdout_samples.npz"),
            "quality_check_sha256": {str(s): digest(out / "quality_checks" / f"step_{s:06d}.npz")
                                     for s in accuracy_steps},
            "total_seconds": time.perf_counter() - start,
        })
        _write_json(out / "summary.json", summary)
        _say({"toy": toy.problem.name, "event": "complete", "live": final["live"]["passed"],
              "seconds": round(summary["total_seconds"], 1)}, progress)
        return summary
    except Exception as error:
        summary.update(status="error", error=f"{type(error).__name__}: {error}",
                       completed_steps=0 if toy is None else toy.completed_steps,
                       total_seconds=time.perf_counter() - start)
        _write_json(out / "summary.json", summary)
        raise
    finally:
        events.close()
        progress.close()


if __name__ == "__main__":
    argv = sys.argv[1:]
    name = argv.pop(0) if argv and argv[0] in PROBLEM_NAMES else "grid100"
    raise SystemExit(runner_main(Toy100(name), argv))
