"""Independently audit the fixed 19-cell Atlas replay, without training.

Read only completed cells. Use the original byte-bound native scorers and
construction validator; inspect full checkpoints for the Atlas backend rule.
A partial execution always retains the complete 19-cell denominator.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
from replay_atlas import ADAPTER, HARNESS, INITIALIZER, ROTATE, STUDY, plan, sha, verify, write

VALIDATORS = Path("/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/validation-cb64-ra11/screens")
VALIDATOR_HASHES = {
    "collect.py": "4ae7dd6f5f2b540a70ed689ba03b0a8e0addcf01f83572e97f0a2369c340250d",
    "lane.py": "59dae645f4f89aae5d38c265d9ee94ac0349a7198719aeffebb0a839e5661cc7",
}
EXPECTED_OPTIONS = dict(eval_output_noise=True, strict_streams=True,
    save_final_state=True, diagnostics=True, evaluation_generate="indexed",
    serial_backward_argument=True, initialization="batch_feature_zero",
    image_prior_perturb=False, ring_frozen_control=False)


def original_protocol_sources():
    # This exact receipt is also present in the original qualified PR #223
    # source bdf05d1b. Remap only the two archived construction adapter paths.
    original = STUDY / "validation-ra15/SOURCE-FREEZE.json"
    assert sha(original) == "78a8aaadaf97b58c7489e64632485ff66ebc3037a87a4e0ab8cd84dec19b2b4d"
    pins = read(original)["hashes"]
    checked = {}
    for filename, expected in pins.items():
        path = Path(filename)
        if filename.startswith(str(HARNESS) + "/") or filename.startswith(str(INITIALIZER) + "/") or path == ROTATE:
            assert sha(path) == expected, filename
            checked[filename] = expected
    for name in ("screen_current.py", "current_api_fixtures.py"):
        matches = {digest for filename, digest in pins.items()
                   if filename.endswith("/validation-ra15/" + name)}
        assert matches == {sha(ADAPTER / name)}, name
        checked[str(ADAPTER / name)] = sha(ADAPTER / name)
    assert len(checked) == 60
    return dict(status="IDENTICAL", files=len(checked), original_source="bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f",
                original_freeze_sha256=sha(original), verified_source_sha256=checked)


def read(path):
    return json.loads(path.read_text())


def load_validator():
    for name, expected in VALIDATOR_HASHES.items():
        assert sha(VALIDATORS / name) == expected, name
    sys.path.insert(0, str(VALIDATORS))
    spec = importlib.util.spec_from_file_location("_original_atlas_validator", VALIDATORS / "collect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


def package_digest(root):
    digest = hashlib.sha256()
    package = root / "particlegan"
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def checkpoint_contract(path, particles, steps, reasons):
    import torch
    from particlegan.feature_cells import population_policy
    from particlegan.output_moments import MAX_RANK
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("trainer", raw)
    if state.get("recipe", {}).get("reopen_guard") != "settled":
        reasons.append("checkpoint recipe lacks settled re-open guard")
    guard = state.get("reopen_guard")
    if not isinstance(guard, dict) or guard.get("schema") != 1:
        reasons.append("checkpoint lacks typed settled guard state")
    selected = state.get("backend_selection")
    if not isinstance(selected, dict):
        reasons.append("checkpoint lacks backend selection")
        return None
    shape = selected.get("output_shape")
    if not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
        reasons.append("invalid checkpoint output shape")
        return None
    width = math.prod(shape)
    population = population_policy(particles)
    backend = "feature_cells" if population["finite_resolution_feasible"] and width <= MAX_RANK else "knn"
    factor = .25 if backend == "feature_cells" else 1.
    if (selected.get("actual_backend") != backend
            or selected.get("generator_noise_factor") != factor
            or selected.get("requested_backend") != "auto"
            or selected.get("population_policy") != population
            or selected.get("raw_output_width") != width
            or selected.get("moment_rank_bound") != MAX_RANK):
        reasons.append("backend selection differs from original finite-population/output-width rule")
    mapping = selected.get("rate_mapping", [])
    if state.get("initial_lrs") != [[item["rate"] for item in row] for row in mapping]:
        reasons.append("optimizer base rates differ from selected rate mapping")
    for row in mapping:
        for item in row:
            expected = factor if item["role"] in ("generator", "noise") else 1.
            if item["factor"] != expected or item["rate"] != item["base_rate"] * expected:
                reasons.append("backend calibration changes an incorrect optimizer role")
    if state.get("completed_steps") != steps:
        reasons.append("checkpoint did not complete original update budget")
    if state.get("recipe", {}).get("num_particles") != particles:
        reasons.append("checkpoint particle count differs from original host")
    return dict(backend=selected["actual_backend"], output_shape=shape,
                generator_noise_factor=selected["generator_noise_factor"],
                completed_steps=state["completed_steps"], guard_schema=guard.get("schema"),
                checkpoint_sha256=sha(path))


def native_regrade(target, task, validator, result, reasons):
    original = validator.validate_native(target, task, result, reasons)
    statuses = {}
    for law in ("noisy", "clean"):
        directory = target / f"native-{law}"
        completed = subprocess.run([sys.executable, "-B", str(HARNESS / "native100_score.py"),
            str(directory), task], env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            capture_output=True, text=True, timeout=180, check=True)
        scored = json.loads(completed.stdout)
        stored = read(directory / "verdict.json")
        if scored != stored:
            reasons.append(f"{law} native independent regrade differs from original stored verdict")
        statuses[law] = {kind: scored[kind]["status"] for kind in ("coverage", "accuracy")}
    return dict(official_status=statuses, initial_parameters_match=all(
        value for role in original["initial_parameter_match"].values() for value in role.values()),
        prior_range_match=original["prior_range_match"],
        terminal_steps=original["event_steps"]["live"][-5:],
        terminal_passes=result["native"]["terminal_accuracy"],
        holdout=result["native"]["holdout"], clean_holdout=result["clean_native"]["holdout"])


def portability_regrade(task, rows, result, reasons):
    if task == "mode_hold":
        requirements = read(HARNESS / "tasks/mode_hold_protocol.json")["host"]["thresholds"]
    elif task.startswith("img_"):
        limits = read(HARNESS / "tasks/image_task_specs.json")[task]["thresholds"]
        requirements = [["modes", ">=", limits["modes"]], ["hq", ">=", limits["hq_min"]]]
    elif task.startswith("vector_"):
        limits = read(HARNESS / "tasks/vector_task_specs.json")[task]["spec"]["thresholds"]
        requirements = ([["modes", ">=", limits["modes"]], ["hq", ">=", limits["hq_min"]]]
                        if isinstance(limits, dict) else limits)
        if result.get("thresholds") != requirements:
            reasons.append("vector result thresholds differ from frozen task declaration")
    else:
        requirements = [["modes", "==", 8], ["hq", ">=", .9]]
    def passes(row):
        for key, operation, bound in requirements:
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                return False
            if not {"<=": value <= bound, ">=": value >= bound, "==": value == bound}[operation]:
                return False
        return True
    flags = [passes(row) for row in rows]
    if any(type(row.get("pass")) is not bool or row["pass"] != flag for row, flag in zip(rows, flags)):
        reasons.append("recorded portability observations differ from frozen metric thresholds")
    if result.get("passing_checks") != sum(flags):
        reasons.append("portability passing-check count differs from recomputed observations")
    spans = [(0, 2400), (2400, 4600)] if task == "ring_shift" else [(0, rows[-1]["step"])]
    segments = []
    for start, end in spans:
        selected = [(row, flag) for row, flag in zip(rows, flags) if start < row["step"] <= end]
        streak = 0
        for _, flag in reversed(selected):
            if not flag:
                break
            streak += 1
        arrival = next((row["step"] for row, flag in selected if flag), None)
        segments.append(dict(start=start, end=end, first_arrival=arrival, final_streak=streak))
    wanted = "PASS" if all(row["first_arrival"] is not None and row["final_streak"] >= 5 for row in segments) else "FAIL"
    if result["status"] != wanted or result.get("final_streak") != segments[-1]["final_streak"]:
        reasons.append("portability verdict differs from original sustained/segment rule")
    return dict(status=wanted, requirements=requirements, passing_checks=sum(flags), segments=segments)


def audit_cell(root, group, task, validator, digest):
    target = root / group / task
    execution = read(target / "execution.json")
    reasons = []
    if execution.get("returncode") != 0:
        reasons.append("execution did not exit successfully")
    freeze_sha = sha(root / "source-freeze.json")
    for key in ("source_integrity_before", "source_integrity_after"):
        if (execution.get(key, {}).get("status") != "VALID"
                or execution[key].get("source_freeze_sha256") != freeze_sha):
            reasons.append(f"{key} not bound to the unchanged source freeze")
    if sha(target / "runner.py") != execution["runner_sha256"]:
        reasons.append("execution wrapper source changed")
    filename = "frames.npz.verdict.json" if group == "moving" else "result.json"
    result = read(target / filename)
    if sha(target / filename) != execution["result_sha256"]:
        reasons.append("result bytes changed after execution")
    if result.get("task") != task or result.get("status") not in ("PASS", "FAIL"):
        reasons.append("result lacks completed original task verdict")
    native = None
    portability = None
    if group == "moving":
        periods = result.get("periods", [])
        if (result.get("turns") != 2 or len(periods) != 3
                or [row.get("period_end") for row in periods] != [500, 1000, 1500]
                or [row.get("target_deg") for row in periods] != [0, 30, 60]):
            reasons.append("moving evaluation omitted an original period/turn")
        baseline = result["pre_turn_hq"]
        passes = [row["modes"] >= 95 and row["hq"] >= .9 * baseline for row in periods[1:]]
        wanted = "PASS" if all(passes) and len(passes) == 2 else "FAIL"
        if result["status"] != wanted or result["passed_periods"] != sum(passes):
            reasons.append("moving verdict differs from original 95-mode/relative-quality rule")
        for step in (500, 1000, 1500):
            checkpoint_contract(target / f"frames.npz.checkpoint-{step:06d}.pt", 20000, step, reasons)
        checkpoint = checkpoint_contract(target / "frames.npz.checkpoint-001500.pt", 20000, 1500, reasons)
        steps, observations, final, clean_status = 1500, 3, periods[-1], None
    else:
        original_plan = validator.task_plan(task)
        header = result.get("header", {})
        if (header.get("package_sha256") != digest or header.get("options") != EXPECTED_OPTIONS
                or header.get("device") != "cuda:0" or header.get("cuda_visible_devices") != "0"
                or header.get("overrides") != {**read(root / "source/atlas.json"),
                                               "initialization": "batch_feature_zero"}):
            reasons.append("runner package/options/device/recipe differs from frozen original protocol")
        if result.get("completed_steps") != original_plan["steps"] or result.get("stream_deviations") != 0:
            reasons.append("incomplete original budget or deviating random streams")
        rows = [json.loads(line) for line in (target / "metrics.jsonl").read_text().splitlines() if line.strip()]
        if [row.get("step") for row in rows] != original_plan["observation_steps"]:
            reasons.append("observation schedule differs from original protocol")
        if any("construction RNG" in warning or "initial receipt unavailable" in warning
               for warning in result.get("warnings", [])):
            reasons.append("original construction fixture warning")
        if group == "native":
            native = native_regrade(target, task, validator, result, reasons)
        else:
            portability = portability_regrade(task, rows, result, reasons)
        checkpoint = checkpoint_contract(target / "final-state.pt", original_plan["num_particles"],
                                         original_plan["steps"], reasons)
        steps, observations = original_plan["steps"], len(rows)
        final, clean_status = result.get("final"), result.get("clean_status")
    return dict(group=group, task=task, quality_status=result["status"],
        validity="INVALID" if reasons else "VALID",
        acceptance_status="INVALID" if reasons else result["status"], reasons=reasons,
        completed_steps=steps, observations=observations, final=final,
        clean_diagnostic_status=clean_status, checkpoint=checkpoint, native=native, portability=portability,
        result_sha256=sha(target / filename), execution_sha256=sha(target / "execution.json"),
        wall_seconds=execution["wall_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing replay artifact root")
    args = parser.parse_args()
    root = args.output.resolve()
    verify(root)
    original_sources = original_protocol_sources()
    validator = load_validator()
    sys.path.insert(0, str(root / "source"))
    import torch
    torch.set_num_threads(1)
    digest = package_digest(root / "source")
    board = read(root / "scoreboard.json")
    declared = {(row["group"], row["task"]) for row in board["results"]}
    expected = plan()
    assert len(expected) == 19 and declared.issubset(set(expected))
    rows = [audit_cell(root, group, task, validator, digest)
            for group, task in expected if (group, task) in declared]
    pending = [dict(group=group, task=task) for group, task in expected if (group, task) not in declared]
    passed = sum(row["acceptance_status"] == "PASS" for row in rows)
    receipt = dict(schema_version=1, status="INCOMPLETE" if pending else "PASS" if passed == 19 else "FAIL",
        required=19, completed=len(rows), passes=passed, pending=pending, results=rows,
        new_training_updates=sum(row["completed_steps"] for row in rows),
        execution_wall_seconds=sum(row["wall_seconds"] for row in rows),
        package_sha256=digest, config_sha256=sha(root / "source/atlas.json"),
        source_freeze_sha256=sha(root / "source-freeze.json"), validator_hashes=VALIDATOR_HASHES,
        original_protocol_sources=original_sources,
        source_integrity_after=verify(root), audit_source_sha256=sha(Path(__file__)),
        artifact_root=str(root), sampling_law="original noisy primary; clean diagnostics preserved",
        current_forge_mog_clean_qualification=False, thresholds_changed=False,
        budgets_changed=False, seeds_changed=False)
    write(root / "independent-audit.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "required", "completed", "passes", "pending")}))


if __name__ == "__main__":
    main()
