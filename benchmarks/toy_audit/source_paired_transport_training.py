"""Observe one unchanged baseline/movable source job per paired target law.

The source host keeps its 6000 updates, seed0, batch64 and checkpoint cadence.
Full traces/checkpoints/arrays stay outside Git. The original twelve-job test
selection prerequisite is never weakened by this two-job source audit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

from benchmarks.paired_error_2d import run as source_run, task as source_task
from .source_routed_ring_training import ROOT, WallLimit, load_module, runtime, sha, state_digest, suffix, wall_cap, write

VERSION = "paired-source-transport-v1"
JOBS = [dict(catalog_id="source-family-08", name="Paired affine2 transport", task="affine2", arm="baseline", cloud="movable", wall_cap=120),
        dict(catalog_id="source-family-09", name="Paired swirl2 transport", task="swirl2", arm="baseline", cloud="movable", wall_cap=120)]


def provenance(quality_path):
    names = ["benchmarks/paired_error_2d/task.py", "benchmarks/paired_error_2d/run.py",
             "benchmarks/paired_error_2d/__main__.py", "benchmarks/paired_error_2d/SOURCE.md",
             "benchmarks/legacy/gan_loss.py", "benchmarks/legacy/grad_regularizers.py",
             "particlegan/vicreg_loss.py", "particlegan/recipes.py",
             "benchmarks/toy_audit/source_paired_transport_training.py",
             "benchmarks/toy_audit/source_routed_ring_training.py"]
    return dict(source_sha256={name: sha(ROOT / name) for name in names},
                source_provenance=source_run.provenance(), runtime=runtime(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                declared_protocol=deepcopy(source_task.PROTOCOL), declared_arms=deepcopy(source_task.ARMS),
                executed_arm=deepcopy(source_task.ARMS["baseline"]),
                quality_evaluator=dict(path=str(quality_path), sha256=sha(quality_path)))


class Capture:
    def __init__(self, task, quality):
        self.task, self.quality = task, quality
        self.x, self.y = source_task.data(task, "validation")
        self.frames, self.metrics, self.purity = [], [], []

    @torch.no_grad()
    def __call__(self, state, step):
        fields = ("model", "critic", "g", "d", "ema", "sampler", "torch_rng")
        def digest():
            return state_digest(dict(state={key: state[key] for key in fields}, global_rng=torch.get_rng_state()))
        before = digest()
        with torch.random.fork_rng(devices=[]):
            # Independent copies keep the source model mode and checkpoint law;
            # current source architecture contains only Linear/LeakyReLU layers.
            model = source_task.Student("movable", source_task.PROTOCOL).requires_grad_(False)
            model.load_state_dict(state["model"])
            live = model(self.x).detach().cpu().numpy().copy()
            model.load_state_dict(state["ema"])
            ema = model(self.x).detach().cpu().numpy().copy()
            row = dict(step=step,
                original_live=source_task.metrics(torch.from_numpy(live), self.y),
                original_ema=source_task.metrics(torch.from_numpy(ema), self.y),
                added_live=self.quality.paired_edit_metrics(live, self.y.numpy(), self.x.numpy()),
                added_ema=self.quality.paired_edit_metrics(ema, self.y.numpy(), self.x.numpy()))
        after = digest()
        if before != after:
            raise RuntimeError("paired checkpoint observer altered training state/RNG")
        self.purity.append(True)
        self.frames.append(dict(step=step, live=live, ema=ema))
        self.metrics.append(row)
        return row


def prefix_parity(task, quality):
    """Four exact source updates include its first every-fourth-step cap."""
    x, y = source_task.data(task, "train")
    rows = []
    initial_global = torch.get_rng_state().clone()
    for observed in (False, True):
        game = source_task.Game(y)  # source defaults baseline + movable
        capture = Capture(task, quality)
        if observed:
            capture(game.state_dict(), 0)
        trace = []
        for step in range(1, 5):
            trace.append(game.update(x, y, step))
            if observed:
                capture(game.state_dict(), step)
        rows.append(dict(state=state_digest(game.state_dict()), losses=state_digest(trace)))
    torch.set_rng_state(initial_global)
    if rows[0] != rows[1]:
        raise RuntimeError(f"paired source observer prefix mismatch: {rows}")
    return dict(updates=4, exact=True, unobserved=rows[0], observed=rows[1],
                scope="Software observer parity, including native lazy cap; not a convergence or seed-study result")


def run_one(task, quality_path, artifacts):
    job = next(row for row in JOBS if row["task"] == task)
    artifacts.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    quality = load_module(quality_path, "paired_source_definition_quality")
    binding = provenance(quality_path)
    (artifacts / "executed-quality.py").write_bytes(quality_path.read_bytes())
    (artifacts / "executed-observer.py").write_bytes(Path(__file__).read_bytes())
    capture = Capture(task, quality)
    original_save = source_run.save
    source_output = artifacts / "source-output"
    prefix, result = None, None
    report = dict(version=VERSION, **job, label=source_run.identifier(task, "baseline", "movable"),
                  fresh_execution_status="UNRUN", original_scientific_status="UNRUN", added_gate_status="UNRUN",
                  media=None, binding=binding, expected_updates=source_task.PROTOCOL["steps"],
                  sampling_policy="Clean ordinary live and EMA residual-student forwards on all1024 original validation rows; no prediction noise, no test rows",
                  heldout_not_a_training_loss=True,
                  subset_scope="One source-default baseline/movable job per data law. Other declared recipe/cloud controls and all12-job test selection are not executed or qualified.",
                  test_evaluation_status="NOT_RUN: source freeze_and_evaluate requires all12 declared jobs complete before checkpoint freeze/test evaluation")

    def save(path, state):
        original_save(path, state)
        if Path(path).name == "latest.pt":
            row = capture(state, state["step"])
            print(json.dumps({"event": "audit_checkpoint", "task": task, **row}), flush=True)

    started = time.perf_counter()
    try:
        with wall_cap(job["wall_cap"]):
            prefix = prefix_parity(task, quality)
            with patch.object(source_run, "save", save):
                result = source_run.train(source_output, task, "baseline", "movable")
            report["fresh_execution_status"] = "COMPLETE"
    except Exception as error:
        report["fresh_execution_status"] = "CAPPED" if isinstance(error, WallLimit) else "BLOCKED"
        report["error"] = repr(error)
        (artifacts / "error.txt").write_text(traceback.format_exc())
        report["error_archive"] = dict(path=str(artifacts / "error.txt"), sha256=sha(artifacts / "error.txt"))
        print(json.dumps({"event": "error", "task": task, "error": repr(error)}), flush=True)
    report["seconds"] = time.perf_counter() - started
    report["observer_prefix_parity"] = prefix
    run_dir = source_output / "runs" / source_run.identifier(task, "baseline", "movable")
    original_metrics = run_dir / "metrics.json"
    latest = run_dir / "latest.pt"
    if original_metrics.exists():
        result = json.loads(original_metrics.read_text())
        report["source_metrics_archive"] = dict(path=str(original_metrics), sha256=sha(original_metrics))
        report["original_stable_live"] = source_run.stable(result, "live_validation")
        report["original_stable_ema"] = source_run.stable(result, "validation")
        report["original_scientific_status"] = "INCOMPLETE" if not result["complete"] else \
            "PASS" if report["original_stable_live"] and report["original_stable_ema"] else "FAIL"
        report["initial_hashes"] = result["initial"]
    else:
        report["original_scientific_status"] = report["fresh_execution_status"]
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        report["latest_checkpoint"] = dict(path=str(latest), sha256=sha(latest))
        report["last_saved_step"] = state["step"]
        report["actual_model_shapes"] = {name: {key: list(value.shape) for key, value in state[name].items()}
                                         for name in ("model", "critic")}
    report["full_budget_complete"] = bool(result is not None and result["complete"])
    report["observer_state_purity"] = dict(checks=len(capture.purity), exact=bool(capture.purity) and all(capture.purity))
    if capture.frames:
        observations = artifacts / "observations.npz"
        np.savez_compressed(observations, source=capture.x.numpy(), target=capture.y.numpy(),
                            steps=np.array([r["step"] for r in capture.frames]),
                            live=np.stack([r["live"] for r in capture.frames]), ema=np.stack([r["ema"] for r in capture.frames]))
        curve = artifacts / "captured-metrics.json"
        write(curve, capture.metrics)
        report["observation_archive"] = dict(path=str(observations), sha256=sha(observations))
        report["curve_archive"] = dict(path=str(curve), sha256=sha(curve))
        report["observation_count"] = len(capture.frames)
        report["actual_step_indices"] = [r["step"] for r in capture.frames]
        report["final"] = capture.metrics[-1]
        report["best_validation_ema_observation"] = min(capture.metrics, key=lambda row: row["original_ema"]["nmse"])
        flags = [r["added_live"]["passed"] and r["added_ema"]["passed"] for r in capture.metrics if r["step"] > 0]
        report["added_terminal_window"] = suffix(flags)
        report["added_gate_status"] = "INCOMPLETE" if not report["full_budget_complete"] else "PASS" if report["added_terminal_window"]["passed"] else "FAIL"
    else:
        report["added_gate_status"] = "BLOCKED"
    if binding != provenance(quality_path):
        raise RuntimeError("paired source/runtime/protocol changed during unchanged audit")
    write(artifacts / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=source_task.TASKS)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-module", type=Path, default=ROOT / "benchmarks/toy_audit/definition_quality.py")
    args = parser.parse_args()
    report = run_one(args.task, args.quality_module, args.artifacts)
    write(args.output, report)
    print(json.dumps({"event": "complete", "task": args.task, "execution": report["fresh_execution_status"],
                      "original": report["original_scientific_status"], "added": report["added_gate_status"]}), flush=True)


if __name__ == "__main__":
    main()
