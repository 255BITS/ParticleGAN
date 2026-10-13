"""Read-only archived/fresh baseline audit; no training or sampling.

Requires the original local artifact archives. Default execution verifies and
prints a compact summary; --write publishes the compact validation receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.state import state_digest


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/forge/r1r2-tier1-search"
ORIGINAL_ARCHIVE = Path("/mnt/ml7tb/ParticleGAN-forge/recipe-owned-priors-20261010/queue/recipe-prior-refactor-selected-leaders-v1")
FRESH_ARCHIVE = Path("/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012/queue/r1r2-tier1-matched-baseline-v1")
SOURCE_FILES = ("experiments/forge/comparison_protocol.py", "experiments/forge/adapters.py",
    "experiments/forge/behavior_adapters.py", "experiments/forge/word_adapter.py",
    "experiments/forge/word_tasks.py", "experiments/forge/clockfree.py", "particlegan/recipe_schedules.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text())


def proof(result):
    checkpoint = result["evidence"]["provenance_checkpoint"]
    path = Path(checkpoint["artifact_root"]) / checkpoint["path"]
    require(file_hash(path) == checkpoint["sha256"], f"Checkpoint bytes changed: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(state_digest(state) == checkpoint["state_sha256"], f"Checkpoint state changed: {path}")
    return state, {"path": str(path), "sha256": checkpoint["sha256"],
                   "state_sha256": checkpoint["state_sha256"]}


def learning_components(state):
    """Select actual learned tensors, optimizer histories and all named RNGs.

    Caller-global cpu_rng/cuda_rng are independently restored by the public
    runner and are excluded. Source/ownership/receipt metadata is also excluded.
    """
    components = {"named_streams": state["streams"]}
    if "trainer" in state:
        components.update({f"trainer.{key}": state["trainer"][key]
                           for key in ("models", "optimizers", "streams")})
    elif "fixture" in state:
        fixture = state["fixture"]
        api = fixture["api_state"]
        components.update({f"fixture.api_state.{key}": api[key]
                           for key in ("models", "optimizers", "streams", "table", "averaged_table", "averages")})
        components["fixture.data_generator"] = fixture["data_generator"]
    else:
        components.update({key: state[key] for key in ("models", "optimizers", "role_parameters")})
    return components


def identity(source, attempt, path, result):
    binding = result.get("applied", result)
    return {"source_commit": source, "attempt_id": attempt, "result_path": str(path),
            "result_sha256": file_hash(path), "initializer": binding["initializer"],
            "initialization_sha256": stable_hash(binding["initialization"]),
            "initialization": binding["initialization"], "prior": binding["prior"]}


def run():
    old_path = ROOT / "reports/forge/r1r2-baseline/results.json"
    fresh_path = REPORT / "publication/results.json"
    original, publication = load_json(old_path), load_json(fresh_path)
    old_rows = {row["task_id"]: row for row in original["tasks"]}
    comparisons = []
    for row in publication["certified_results"]:
        if row["role"] != "baseline" or row["task_id"] == "clockfree_audit_measurement_v1":
            continue
        task = row["task_id"]
        previous = old_rows[task]
        paths = (ORIGINAL_ARCHIVE / previous["attempt_id"] / "result.json",
                 FRESH_ARCHIVE / row["attempt_id"] / "result.json")
        require(file_hash(paths[0]) == previous["original_files"]["result"]["sha256"],
                f"Original result identity changed: {task}")
        raw = [load_json(path)["raw"]["result"] for path in paths]
        states, descriptors = zip(*(proof(value) for value in raw))
        evidence = [value["evidence"] for value in raw]
        curves = {}
        for field in ("observations", "live"):
            curves[field] = stable_hash(evidence[0][field])
            require(evidence[0][field] == evidence[1][field], f"Numeric {field} changed: {task}")
        if "confirmations" in evidence[0]:
            confirmations = [[{"step": item["step"], "metrics": item["metrics"]}
                              for item in value["confirmations"]] for value in evidence]
            require(confirmations[0] == confirmations[1], f"Confirmation metrics changed: {task}")
            curves["confirmation_metrics"] = stable_hash(confirmations[0])
        parts = [learning_components(state) for state in states]
        require(parts[0].keys() == parts[1].keys(), f"Learning component scope changed: {task}")
        digests = {name: state_digest(value) for name, value in parts[0].items()}
        require(all(digest == state_digest(parts[1][name]) for name, digest in digests.items()),
                f"Learning tensors or consumed RNG changed: {task}")
        comparisons.append({"task_id": task, "status": "PASS",
            "observations": len(evidence[0]["observations"]), "curve_sha256": curves,
            "learning_components_sha256": digests, "checkpoints": list(descriptors),
            "original": identity(original["frozen_source"]["commit"], previous["attempt_id"], paths[0], raw[0]),
            "fresh": identity(publication["source_commit"], row["attempt_id"], paths[1], raw[1])})
    require(len(comparisons) == 6, "All six ordinary baseline tasks must be independently checked")
    registrations = [load_json(REPORT / name) for name in ("registration.json", "round2-registration.json")]
    baseline = registrations[0]["candidates"]["baseline"]
    unchanged = ("candidate", "candidate_revision", "protocol_sha256", "runtime_sha256",
                 "source_digest", "task_contracts", "task_keys", "execution_policy", "view_sha256")
    require(all(baseline[key] == registrations[1]["candidates"]["baseline"][key] for key in unchanged),
            "Round 2 changed the cached baseline's comparison identity")
    for registration in registrations:
        for role, candidate in registration["candidates"].items():
            require(all(candidate[key] == baseline[key] for key in
                        ("protocol_sha256", "runtime_sha256", "source_digest", "task_contracts", "view_sha256")),
                    f"A registered comparison condition changed: {role}")
    inputs = (old_path, fresh_path, REPORT / "registration.json", REPORT / "round2-registration.json")
    return {"schema_version": 1, "status": "PASS", "scope": "read_only_archived_vs_fresh_protocol_validation",
        "qualification_input": False, "training_updates_added": 0, "sampling_draws_added": 0,
        "excluded": ["caller-global cpu_rng/cuda_rng", "source/ownership/applied receipt metadata"],
        "clock_scope": "Existing certified own-state clock probes are separate; this audit compares six ordinary tasks.",
        "round2_scope": "Declared scientific source/protocol/runtime/tasks and cached baseline identity match; final round2 observations require its collector.",
        "source_digest": baseline["source_digest"], "rounds_declared_contracts_match": True,
        "input_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in inputs},
        "source_file_sha256": {name: file_hash(ROOT / name) for name in SOURCE_FILES},
        "audit_source_sha256": file_hash(Path(__file__)), "comparisons": comparisons,
        "previous_software_validation": [
            {"command": "/home/martyn/dev/ParticleGAN/.venv/bin/python -m pytest -q tests/test_forge_comparison_protocol.py tests/test_forge_adapters.py tests/test_forge_behavior_adapters.py",
             "passed": 63, "seconds": 9.55, "implementation_commit": "e4bdfcf4c", "log_sha256": None, "output_storage": "tool_output_only"},
            {"command": "/home/martyn/dev/ParticleGAN/.venv/bin/python -m pytest -q tests/test_forge_word_joint.py tests/test_forge_word_smoke_hold.py tests/test_forge_word_provenance_cuda.py tests/test_forge_gaussian_smoke.py",
             "passed": 60, "seconds": 5.36, "implementation_commit": "e4bdfcf4c", "log_sha256": None, "output_storage": "tool_output_only"},
            {"command": "/home/martyn/dev/ParticleGAN/.venv/bin/python -m pytest -q --tb=short tests/test_forge_clockfree.py tests/test_forge_policy_integration.py tests/test_forge_tier1_policy_coverage.py",
             "existing_tests_passed": 43, "new_tests_fixture_errors": 13,
             "seconds": 14.96, "log_sha256": None, "output_storage": "tool_output_only"},
            {"command": "/home/martyn/dev/ParticleGAN/.venv/bin/python -m pytest -q --tb=short tests/test_forge_clockfree.py -k source_audit",
             "passed": 14, "deselected": 7, "seconds": .12,
             "implementation_commit": "c3194c303", "log_sha256": None,
             "output_storage": "tool_output_only", "note": "After correcting the new pure-test recipe fixture; includes 13 newly added tests and one existing test."}],
        "software_validation_counts": {"instrumentation_passes": 123,
            "clock_policy_union_passes": 56, "clock_union_method": "43 existing passes plus 13 corrected new tests; not a fabricated single-suite rerun"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = run()
    if args.write:
        atomic_json(REPORT / "protocol-validation.json", result)
    print(json.dumps({"status": result["status"], "ordinary_tasks": len(result["comparisons"]),
        "rounds_declared_contracts_match": True, "training_updates_added": 0, "sampling_draws_added": 0}))
