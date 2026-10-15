"""Bounded original-only restore and disposable next-update verification.

No candidate recipe, training loop, score, or original artifact is modified.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

START = time.monotonic()
signal.alarm(450)
HERE = Path(__file__).resolve().parent
INVENTORY = HERE.parent / "EXECUTION-SOURCE-RECOVERY.json"
inventory = json.loads(INVENTORY.read_text())
SOURCE = Path(inventory["source_root"])
sys.path.insert(0, str(SOURCE))

import numpy as np
import scipy
import torch

from experiments.forge.sources import verify_snapshot
from experiments.forge.state import state_digest, require_consistent_rng
from experiments.forge.gaussian_tasks import build as build_gaussian
from experiments.forge.adapters import _context
from experiments.forge.vectorprofiles import resolve_vector_spec, build_vector_models
from benchmarks.toy_audit.gaussian1d_quality import sample_target as gaussian_target
from benchmarks.transfer_suite.vector_tasks import sample_target as vector_target


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_report(report):
    report["process_wall_seconds"] = time.monotonic() - START
    (HERE / "original-restore-parity.json").write_text(json.dumps(report, indent=2) + "\n")


def load_original(attempt, relative):
    path = Path(attempt["root"]) / relative
    record = next(a for a in attempt["tensor_artifacts"] if a["relative_path"] == relative)
    assert sha(path) == record["sha256"], f"original artifact changed: {path}"
    packet = torch.load(path, map_location="cpu", weights_only=True)
    assert state_digest(packet) == record["complete_state_sha256"]
    require_consistent_rng(packet)
    wrapper = json.loads((Path(attempt["root"]) / "request.json").read_text())
    assert sha(Path(attempt["root"]) / "request.json") == attempt["receipts"]["request.json"]
    return path, packet, wrapper["request"], wrapper["job"]["task_id"]


def construct_and_restore(request, task_id, packet):
    task = request["tasks"][task_id]
    cap = packet["trainer"].get("max_steps", packet["recipe"]["total_steps"])
    if task_id.startswith("gaussian"):
        context, trainer, spec = build_gaussian(request, task, "cuda:0", max_steps=cap)
    else:
        spec = resolve_vector_spec(task)
        resources = {"num_particles": spec["particles"], "z_dim": spec["z_dim"], "batch_size": spec["batch"]}
        context = _context(request, task, "cuda:0", resources)
        generator, discriminator = build_vector_models(context, spec)
        trainer = context.build_trainer(generator, discriminator, max_steps=cap)
    # The real host registers data, observer and confirmation streams after
    # model construction. Recreate exactly those public stream bindings before
    # invoking the unchanged strict loader; never rewrite a saved packet.
    for binding in packet["streams"]["manifest"]["bindings"].values():
        context.streams.generator(binding["family"], component=binding["component"],
                                  purpose=binding["purpose"], device=binding["device"])
    context.load_state_dict(packet)
    restored = context.state_dict()
    require_consistent_rng(restored)
    assert state_digest(restored) == state_digest(packet), "strict round trip changed original state"
    return context, trainer, spec


def one_disposable_update(request, task_id, packet, *, extend_smoke=False):
    context, trainer, spec = construct_and_restore(request, task_id, packet)
    if extend_smoke:
        # This is the original smoke's own recorded stability execution cap,
        # applied only after its unmodified 1000-step checkpoint round trip.
        assert trainer.completed_steps == 1000 and trainer.max_steps == 1000
        assert request["tasks"]["gaussian1d_stability"]["execution"]["continuation_of"] == "gaussian1d_smoke"
        assert request["tasks"]["gaussian1d_stability"]["execution"]["steps"] == 6000
        trainer.extend_execution(6000)
    if trainer.completed_steps >= trainer.max_steps:
        return {"status": "BLOCKED", "reason": "original terminal execution budget exhausted",
                "completed_steps": trainer.completed_steps, "max_steps": trainer.max_steps}
    before = state_digest(context.state_dict())
    step = trainer.completed_steps
    if task_id.startswith("gaussian") and step >= 4000:
        spec["means"] = [[3.0]]  # Original switch occurs before update 4001.
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    sample_target = gaussian_target if task_id.startswith("gaussian") else vector_target
    real = sample_target(spec, context.recipe.batch_size, data, step)
    batch_sha = state_digest(real)
    result = trainer.step(real.to("cuda:0"), collect_stats=True)
    torch.cuda.synchronize()
    after = context.state_dict()
    require_consistent_rng(after)
    assert trainer.completed_steps == step + 1
    return {"status": "PASS", "before_sha256": before, "after_sha256": state_digest(after),
            "real_batch_sha256": batch_sha, "completed_steps": trainer.completed_steps,
            "max_steps": trainer.max_steps, "result_sha256": state_digest(result)}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_multithreading_enabled(False)
    verify_snapshot(SOURCE)
    runtime = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
               "scipy": scipy.__version__, "cuda": torch.version.cuda,
               "device": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               "threads": torch.get_num_threads(), "autograd_multithreading": torch.autograd.is_multithreading_enabled()}
    assert (torch.__version__, np.__version__, scipy.__version__) == ("2.11.0+cu128", "2.4.4", "1.17.1")
    report = {"schema_version": 1, "kind": "original_only_software_diagnostic", "runtime": runtime,
              "source_root": str(SOURCE), "source_digest": inventory["source_digest"],
              "script_sha256": sha(__file__), "inventory_sha256": sha(INVENTORY),
              "preflight_sha256": sha("/ml2/hypergan/particlegan-selective-history-20261015/reports/forge/selective-history/PREFLIGHT.md"),
              "campaign_training": False, "candidate_outputs": False, "results": [], "disposable_updates": 0}
    cases = [(0, "evaluator/state.pt", "gaussian1000", True),
             (1, "evaluator/pre-shift-state.pt", "gaussian4000", False),
             (1, "evaluator/state.pt", "gaussian6000", False),
             (2, "provenance/provenance-state.pt", "unequal1200", False)]
    for ai, relative, name, extend in cases:
        row = {"name": name}
        print(json.dumps({"event": "start", "name": name}), flush=True)
        try:
            path, packet, request, task_id = load_original(inventory["attempts"][ai], relative)
            row.update(path=str(path), checkpoint_sha256=sha(path), original_state_sha256=state_digest(packet), task_id=task_id)
            context, trainer, _ = construct_and_restore(request, task_id, packet)
            row["strict_constructor_load_roundtrip"] = "PASS"
            row["original_completed_steps"] = trainer.completed_steps
            row["original_max_steps"] = trainer.max_steps
            del context, trainer
            operations = []
            for _ in range(2):
                result = one_disposable_update(request, task_id, packet, extend_smoke=extend)
                operations.append(result)
                if result["status"] == "PASS": report["disposable_updates"] += 1
                else: break
            row["operations"] = operations
            row["next_update_clone_parity"] = "PASS" if len(operations) == 2 and operations[0] == operations[1] else operations[0]["status"] if len(operations) == 1 else "FAIL"
            assert row["next_update_clone_parity"] != "FAIL", "cloned original next updates differ"
        except Exception as exc:
            row.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        report["results"].append(row)
        save_report(report)
        print(json.dumps(row), flush=True)
    report["all_roundtrips_pass"] = all(r.get("strict_constructor_load_roundtrip") == "PASS" for r in report["results"])
    report["preflight_status"] = "BLOCKED"
    save_report(report)
    print(json.dumps({"event": "finished", "disposable_updates": report["disposable_updates"],
                      "all_roundtrips_pass": report["all_roundtrips_pass"], "process_wall_seconds": report["process_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
