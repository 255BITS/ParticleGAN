"""Bounded task-scoped parameter diagnostic through the existing public fixture."""
from __future__ import annotations

import argparse
import fcntl
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.toy_audit.api_images import WordFixture, word_oracle_controls
from benchmarks.toy_audit.api_run import render_gif
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.sources import inspect_source
from experiments.forge.views import grade_result, load_tasks, task_fingerprint
from experiments.forge.word_adapter import run_word, word_context

PROTOCOL_PATH = Path(__file__).with_name("round1-protocol.json")
ALLOWED = {"lr", "d_lr_mult", "prior_lr_mult", "input_noise_std", "output_noise_std",
           "network_lr_horizon_cap"}


def prepare(protocol, arm, device):
    task = load_tasks(ROOT)[protocol["task"]]
    parent_path = ROOT / arm["base_candidate"]
    if file_hash(parent_path) != arm["base_candidate_sha256"]:
        raise ValueError("frozen parent candidate changed")
    parent = read_json(parent_path)
    candidate = deepcopy(parent)
    if set(arm["recipe_delta"]) - ALLOWED:
        raise ValueError("diagnostic includes an undeclared parameter")
    candidate["id"] = arm["id"]
    candidate["recipe_overrides"].update(arm["recipe_delta"])
    for name in ("resolved_configuration_recipe", "configuration_id", "search_study_id",
                 "search_report", "configuration_settings"):
        candidate.pop(name, None)
    context = word_context({"candidate": candidate, "protocol": {"seed": protocol["protocol_seed"]}},
                           task, device, root=ROOT)
    reference = word_context({"candidate": parent, "protocol": {"seed": protocol["protocol_seed"]}},
                             task, device, root=ROOT).recipe.to_dict()
    changed = {key for key, value in context.recipe.to_dict().items() if reference[key] != value}
    if changed - ALLOWED or context.recipe.prior_reg != arm["prior_reg"] or arm["prior_reg"] != 0:
        raise ValueError("diagnostic changed a mechanism or enabled prior spread")
    if (task["execution"]["steps"], task["evaluation"]["observations"],
        task["evaluation"]["minimum_stable_checks"], task["resources"]["timeout_seconds"]) != (20001, 24, 5, 900):
        raise ValueError("diagnostic must retain the complete word task")
    if (protocol["task_updates"], protocol["observations"], protocol["terminal_passes"]) != (20001, 24, 5):
        raise ValueError("declared diagnostic budget differs from the task")
    return candidate, task, context


def validate(protocol):
    if (len(protocol["arms"]) != protocol["maximum_runs"] or not 1 <= protocol["maximum_runs"] <= 9
            or protocol["round_reserved_seconds"] != 900 * protocol["maximum_runs"]
            or protocol["run_timeout_seconds"] != 900
            or protocol["device"] != "cuda:0" or protocol["workers"] != 1
            or protocol["cpu_threads"] != 1 or protocol["protocol_seed"] != 0):
        raise ValueError("invalid bounded execution protocol")
    if not all(row["passed"] == row["expected_pass"] for row in word_oracle_controls().values()):
        raise ValueError("word oracle/destructive controls disagree")
    identities = set()
    rows = []
    for arm in protocol["arms"]:
        candidate, task, context = prepare(protocol, arm, "cpu")
        identity = stable_hash({"recipe": context.recipe.to_dict(), "task": task_fingerprint(task)})
        if identity in identities:
            raise ValueError("duplicate unchanged scientific arm")
        identities.add(identity)
        # Construction checks real optimizer groups without taking a training update.
        fixture = WordFixture(device="cpu", seed=0, recipe_name=None, max_steps=20001, components=context)
        if tuple(fixture.opt_g.param_groups[-1]["betas"]) != tuple(context.recipe.prior_betas or context.recipe.betas):
            raise ValueError("prior optimizer failed to consume the declared beta setting")
        rows.append({"id": candidate["id"], "scientific_identity": identity,
                     "recipe_delta": arm["recipe_delta"], "reservation_seconds": 900})
    return rows


def execute(protocol, arm, output):
    output.mkdir(parents=True, exist_ok=False)
    candidate, task, context = prepare(protocol, arm, protocol["device"])
    source = inspect_source(ROOT, extra_paths=[str(PROTOCOL_PATH.relative_to(ROOT)),
        str(Path(__file__).relative_to(ROOT)), arm["base_candidate"], f"configs/forge/tasks/{task['id']}.json"])
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("freeze the execution source commit before training")
    atomic_json(output / "source-manifest.json", source)
    atomic_json(output / "request.json", {"candidate": candidate, "protocol": protocol,
                                         "arm": arm, "task": task})
    def timeout(signum, frame):
        raise TimeoutError("900-second reserved word diagnostic exhausted")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, protocol["run_timeout_seconds"])
    started = time.monotonic()
    try:
        request = {"candidate": candidate, "protocol": {"seed": protocol["protocol_seed"]},
                   "candidate_revision": stable_hash(candidate), "tasks": {task["id"]: task}}
        raw, records = run_word(request, task, output, protocol["device"], capture_media=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    wall = time.monotonic() - started
    grade = grade_result(task, raw)
    torch.save(records, output / "observed-records.pt")
    np.savez_compressed(output / "observations.npz", **{
        f"step{record['step']}_view{index}_{role}": view[role].numpy()
        for record in records for index, view in enumerate(record["views"])
        for role in ("target", "samples")})
    artifacts = {name: {"sha256": file_hash(output / name), "bytes": (output / name).stat().st_size}
                 for name in ("source-manifest.json", "request.json", "adapter-receipt.json",
                              "state.pt", "observed-records.pt", "observations.npz")}
    compact = {"schema_version": 1, "id": arm["id"], "family": arm["family"],
        "protocol_id": protocol["id"], "protocol_sha256": file_hash(PROTOCOL_PATH),
        "scope": protocol["purpose"], "qualification_input": False, "eligible_for_default": False,
        "candidate_revision": request["candidate_revision"], "parent": arm["base_candidate"],
        "parent_sha256": arm["base_candidate_sha256"], "recipe_delta": arm["recipe_delta"],
        "source_commit": source["origin_commit"], "source_digest": source["digest"],
        "task_fingerprint": task_fingerprint(task), "task": task,
        "grade": grade, "final_metrics": raw["evidence"]["live"],
        "recipe": raw["recipe"], "prior": raw["prior"], "initializer": raw["initializer"],
        "initialization": raw["initialization"], "host": raw["evidence"]["host"],
        "field_ownership": raw["field_ownership"], "guards": raw["evidence"]["guards"],
        "rng_manifest_sha256": stable_hash(raw["rng"]), "sampling_law": raw["evidence"]["sampling_law"],
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": protocol["device"], "gpu": torch.cuda.get_device_name(0), "cpu_threads": 1,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
        "cost": {**raw["cost"], "wall_seconds": wall, "reservation_seconds": 900},
        "raw_directory": str(output.resolve()), "raw_artifacts": artifacts}
    atomic_json(output / "compact-receipt.json", compact)
    print(json.dumps({"event": "concluded", "id": arm["id"], "grade": grade["gate_status"],
                      "wall_seconds": wall, "final": raw["evidence"]["live"]}), flush=True)


def render(protocol, arm, output):
    compact = read_json(output / "compact-receipt.json")
    records = torch.load(output / "observed-records.pt", map_location="cpu", weights_only=False)
    indices = np.rint(np.linspace(0, len(records) - 1, 9)).astype(int)
    case = {"id": arm["id"], "goal": "Learn five equally likely words and their correctly paired inverse",
            "scope": protocol["purpose"], "default_recipe": arm["family"], "default_steps": 20001,
            "eval_samples": 1024, "sampling": {"evaluation": compact["sampling_law"]}}
    render_gif(case, [records[i] for i in indices], output / "goal.gif", full_budget=True,
               requested_steps=20001, final_verdict=compact["grade"]["gate_status"])
    print(json.dumps({"id": arm["id"], "gif_sha256": file_hash(output / "goal.gif")}), flush=True)


def main():
    global PROTOCOL_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--run")
    parser.add_argument("--render")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/forge/word-root-cause-round1")
    args = parser.parse_args()
    PROTOCOL_PATH = args.protocol.resolve()
    protocol = read_json(PROTOCOL_PATH)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = validate(protocol)
    if not args.run and not args.render:
        print(json.dumps({"protocol": protocol["id"], "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "arms": rows}), flush=True)
        return
    arm = next(row for row in protocol["arms"] if row["id"] == (args.run or args.render))
    if args.run:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(protocol, arm, args.output / arm["id"])
    else:
        render(protocol, arm, args.output / arm["id"])


if __name__ == "__main__":
    main()
