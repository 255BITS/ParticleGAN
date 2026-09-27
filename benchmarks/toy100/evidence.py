"""Gate evidence for a toy100 problem trained by ``benchmarks.toy_runner.run``.

Harness code, not problem code: ``record`` trains ``native.Toy100`` with the
shared ``run`` (its loop, non-finite check and JSON observation log) and plugs
in an observer that writes what ``gate.py`` and ``accuracy_gate.py`` regrade:
config, summary, per-checkpoint live/EMA eval events and snapshots, the final
draw, the terminal quality checks and the 100k holdout. Metrics come from the
declared ``Toy100.metrics``; the observer only keeps the draw it scored.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from benchmarks.toy_runner import Model, ToyRun, run

from .accuracy import PROTOCOL as ACCURACY_PROTOCOL
from .accuracy_evidence import AccuracyEvidence
from .accuracy_gate import HOLDOUT_N, HOLDOUT_SEED_OFFSETS
from .gate import MANDATORY_EARLY_STEPS, MIN_STABLE_CHECKS, evaluation_steps
from .metrics import EVAL_N
from .native import D_HIDDEN, FOURIER, N_HIDDEN, PRIOR_STD, STEPS, Toy100
from .problems import sample_real

EVAL_INTERVAL = 250
SNAPSHOT_SAMPLES = 4096
TARGET_SEED_OFFSET = 401


def native_config(problem: str, *, steps: int = STEPS, device: str = "cpu", seed: int = 0) -> dict:
    """The executed declaration that ``gate.py`` reads."""
    recipe = Toy100(problem, steps=steps).recipe()
    return json.loads(json.dumps({
        "problem": problem, "runner": "benchmarks.toy_runner", "declaration": "benchmarks.toy100.native.Toy100",
        "seed": seed, "device": str(device), "steps": steps, "eval_interval": EVAL_INTERVAL,
        "early_eval_steps": list(MANDATORY_EARLY_STEPS), "eval_samples": EVAL_N,
        "snapshot_samples": SNAPSHOT_SAMPLES, "snapshot_interval": EVAL_INTERVAL,
        "stable_evals": MIN_STABLE_CHECKS, "threads": 1, "batch_size": recipe.batch_size,
        "networks": {"prior": f"R2Normal(0, {PRIOR_STD:.6g}) particle table", "generator": "identity Linear(2, 2)",
                     "critic": f"SimpleMLPDiscriminator({D_HIDDEN}x{N_HIDDEN}, fourier={FOURIER}), "
                               "deterministic_orthogonal_"},
        "recipe": recipe.to_dict()}))


class _Tap:
    """The declared problem, unchanged, remembering the draw and nets its metrics saw."""

    # ``models.sample_clean`` (via ``AccuracyEvidence.finish``) disables output-noise
    # wrappers on ``G``/``ema_G``; the runner's ``Model.sample`` never adds training
    # output noise, so there are none to disable.
    G = ema_G = None

    def __init__(self, problem):
        self.inner, self.draw, self.nets, self.holdout = problem, None, None, {}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def metrics(self, model):
        sample = model.sample

        def keep(n):
            drawn = sample(n)
            self.draw = drawn.x
            return drawn
        model.sample, self.nets = keep, model.nets
        return self.inner.metrics(model)

    @torch.no_grad()
    def sample(self, n, *, ema=False, generator=None):
        """``AccuracyEvidence.finish`` draws the holdout through this."""
        return Model(self, self.holdout[ema], generator, ema=ema).sample(n).x


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def record(problem: str, out_dir: str | Path, *, steps: int = STEPS, device: str = "cpu", seed: int = 0) -> dict:
    """Train one problem on the shared runner and write regradable gate evidence."""
    config = native_config(problem, steps=steps, device=device, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    (out / "snapshots").mkdir()
    torch.set_num_threads(config["threads"])
    _write_json(out / "config.json", config)
    eval_steps = evaluation_steps(steps, EVAL_INTERVAL, MANDATORY_EARLY_STEPS)
    accuracy_steps = eval_steps[-MIN_STABLE_CHECKS:]
    summary = {"status": "running", "problem": problem, "budget_steps": steps, "config": config,
               "eval_steps": eval_steps, "snapshot_steps": eval_steps,
               "accuracy": {"protocol": ACCURACY_PROTOCOL, "check_steps": accuracy_steps, "sample_count": EVAL_N,
                            "holdout_samples": HOLDOUT_N, "holdout_seed_offsets": HOLDOUT_SEED_OFFSETS},
               "accuracy_check_steps": accuracy_steps}
    _write_json(out / "summary.json", summary)
    tap, final, start = _Tap(Toy100(problem, steps=steps)), {}, time.perf_counter()
    dev = torch.device(device)
    target = sample_real(problem, EVAL_N, device=dev,
                         generator=torch.Generator(device=dev).manual_seed(seed + TARGET_SEED_OFFSET)).cpu().numpy()
    evidence = AccuracyEvidence(config, out, eval_steps, torch.from_numpy(target))
    wanted = set(eval_steps)
    events = (out / "events.jsonl").open("x", buffering=1)

    def observe(step, measure):
        # run() passes ToyRun.measure, which also scores the EMA networks.
        if step not in wanted:
            return
        clouds, nets = {}, {}
        for model in ("live", "ema"):
            metrics = {k: v for k, v in measure(ema=model == "ema").items() if k != "verdict"}
            accuracy = evidence.observe(step, model, tap.draw, metrics)
            events.write(json.dumps({"event": "eval", "step": step, "model": model, "metrics": metrics,
                                     "accuracy": accuracy, "elapsed": time.perf_counter() - start},
                                    allow_nan=False) + "\n")
            final[model], clouds[model], nets[model == "ema"] = metrics, tap.draw.cpu().numpy(), tap.nets
        np.savez_compressed(out / "snapshots" / f"step_{step:06d}.npz", live=clouds["live"][:SNAPSHOT_SAMPLES],
                            ema=clouds["ema"][:SNAPSHOT_SAMPLES], target=target[:SNAPSHOT_SAMPLES])
        if step == steps:
            np.savez_compressed(out / "final_samples.npz", live=clouds["live"], ema=clouds["ema"], target=target)
            tap.holdout = nets
            summary["accuracy"], summary["holdout"] = evidence.finish(tap)

    def say(row):
        print(json.dumps(row, allow_nan=False, default=float), flush=True)
    try:
        # run() observes after each update; step 0 is the same networks before any update.
        initial = ToyRun(tap, seed=seed, device=device)
        observe(0, initial.measure)
        del initial
        result = run(tap, steps=steps, seed=seed, device=device, observer=observe, log=say,
                     log_path=out / "progress.log")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        summary.update({
            "status": "complete", "completed_steps": result["steps"], "final": final,
            "final_samples_file": "final_samples.npz", "final_samples_sha256": digest(out / "final_samples.npz"),
            "holdout_samples_file": "holdout_samples.npz", "holdout_sha256": digest(out / "holdout_samples.npz"),
            "quality_check_sha256": {str(s): digest(out / "quality_checks" / f"step_{s:06d}.npz")
                                     for s in accuracy_steps},
            "total_seconds": time.perf_counter() - start,
        })
        _write_json(out / "summary.json", summary)
        return summary
    except Exception as error:
        summary.update(status="error", error=f"{type(error).__name__}: {error}", completed_steps=0,
                       total_seconds=time.perf_counter() - start)
        _write_json(out / "summary.json", summary)
        raise
    finally:
        events.close()
