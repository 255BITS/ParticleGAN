"""Predeclare, execute and independently regrade a bounded ExtraAdam probe.

The two complete frozen host budgets are mode-hold (hard failure) followed by
trajectory (cheap auxiliary behavior). Passing both is READY_EXPAND, never a
shared-22 pass. All records remain ineligible for the production common gate.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100.extra_adam_scratch import HOSTS, METHODS, PAPER, extra_adam, transformed_function

IMPLEMENTATION = ROOT / "reports/toy100/extra_adam_scratch.py"
DRIVER = Path(__file__)
ORDER = ("mode_hold", "trajectory")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def candidates() -> list[dict]:
    rows = [dict(method="extra_adam", core=core, lr=lr, beta2=beta2)
            for core, lr, beta2 in itertools.product(
                ("winner", "simple"), (.0005, .001, .0025, .00425), (.9, .99, .999))]
    rows.extend(dict(method="sim_adam", core=core, lr=lr, beta2=beta2)
                for core, lr, beta2 in itertools.product(
                    ("winner", "simple"), (.001, .0025), (.99, .999)))
    return rows


def prepare(root: Path) -> dict:
    from benchmarks.locked_shared import mode_hold, trajectory
    from benchmarks.transfer_suite import suite
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe

    if root.exists() or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise RuntimeError("prepare requires committed source and a fresh output root")
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    base_path = ROOT / "configs/toy100/shared_candidate.json"
    base = json.loads(base_path.read_text())
    (root / "base_config.json").write_bytes(base_path.read_bytes())
    rows = []
    for index, values in enumerate(candidates()):
        row = dict(id=f"eg{index:03d}", **values)
        config = dict(base)
        config.pop("network_lr_horizon_cap")
        config.pop("network_lr_floor")
        config.update(name=row["id"], lr=row["lr"], betas=[0.0, row["beta2"]],
                      lr_anneal_start=0.0, lr_floor=1.0)
        if row["core"] == "simple":
            config.update(reg_kappa=1.0, reg_coeff=1.0, prior_reg=0.0)
        recipe, _, _ = declared_recipe(config)
        assert recipe.lr_floor == 1 and recipe.lr_anneal_start == 0
        row["config_file"] = f"configs/{row['id']}.json"
        write(root / row["config_file"], config)
        row["config_sha256"] = digest(root / row["config_file"])
        rows.append(row)
    with tempfile.TemporaryDirectory(prefix="extra-adam-source-") as temporary:
        sources = suite.snapshot(Path(temporary))["source_sha256"]
    sources["benchmarks/toy100/models.py"] = digest(ROOT / "benchmarks/toy100/models.py")
    hosts = {}
    for task, module in (("mode_hold", mode_hold), ("trajectory", trajectory)):
        _, generated, original_sha = transformed_function(module, task)
        (root / f"generated_{task}.py").write_text(generated)
        hosts[task] = dict(original_function_sha256=original_sha,
                           generated_function_sha256=digest(root / f"generated_{task}.py"))
    manifest = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        implementation_sha256=digest(IMPLEMENTATION), driver_sha256=digest(DRIVER),
        source_sha256=sources, host_sources=hosts, rows=rows, task_order=list(ORDER),
        base_config_sha256=digest(base_path), fixed_seed=0,
        fixed_outer_budgets=dict(mode_hold=1200, trajectory=400),
        update_rule="Gidel Algorithm4 option1: joint game gradients, moments at both evaluations, final update from original weights; no averaging",
        control="One-evaluation simultaneous Adam, same joint-gradient capture",
        cost="ExtraAdam: two D and two G gradients per outer update, sampler block replayed twice; SimAdam: one each",
        new_tuning_coefficients=[],
        selection="Complete mode_hold before trajectory; both strict passes mean READY_EXPAND only. Need remaining frozen hosts, fresh19, native3/common22 before a winner claim.",
        production_common_gate_eligible=False,
    )
    write(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(digest(root / "manifest.json") + "\n")
    return manifest


def validate(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    assert digest(root / "manifest.json") == (root / "manifest.sha256").read_text().strip()
    assert manifest["source_commit"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert manifest["implementation_sha256"] == digest(IMPLEMENTATION)
    assert manifest["driver_sha256"] == digest(DRIVER)
    assert manifest["task_order"] == list(ORDER)
    assert [{key: row[key] for key in ("method", "core", "lr", "beta2")}
            for row in manifest["rows"]] == candidates()
    for file, expected in manifest["source_sha256"].items():
        assert digest(ROOT / file) == expected, file
    for row in manifest["rows"]:
        assert digest(root / row["config_file"]) == row["config_sha256"]
    return manifest


def _same(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def _verify_transform(directory: Path, task: str, expected: dict) -> None:
    generated = (directory / "generated_host.py").read_text()
    assert digest(directory / "generated_host.py") == expected["generated_function_sha256"]
    with tarfile.open(directory / "source.tar.gz", "r:gz") as archive:
        full = archive.extractfile(f"benchmarks/locked_shared/{task}.py").read().decode()
    original = next(node for node in ast.parse(full).body
                    if isinstance(node, ast.FunctionDef) and node.name == HOSTS[task])
    tree = ast.parse(generated)
    class Inverse(ast.NodeTransformer):
        def visit_For(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == "_extra_phase":
                assert ast.unparse(node.iter) == "_extra_state.phases(step, opt_d, opt_g)"
                return [self.visit(statement) for statement in node.body]
            return self.generic_visit(node)
        def visit_If(self, node):
            if ast.unparse(node.test) == "_extra_phase == 0":
                assert len(node.body) == 1 and not node.orelse
                assert ast.unparse(node.body[0].value.func) == "schedule_optimizer"
                return node.body[0]
            return self.generic_visit(node)
    inverted = Inverse().visit(tree)
    assert ast.dump(inverted.body[0], include_attributes=False) == ast.dump(original, include_attributes=False)


def regrade(directory: Path, row: dict, manifest: dict, manifest_sha: str) -> dict:
    from benchmarks.toy_suite import _episode_rows

    task = directory.name
    assert task in ORDER
    protocol = json.loads((directory / "protocol.json").read_text())
    index = json.loads((directory / "index.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert len(index["records"]) == 1 and index["records"][0]["name"] == task
    indexed = index["records"][0]
    record = json.loads(gzip.decompress((directory / indexed["artifact"]).read_bytes()))
    receipt = json.loads((directory / "optimizer_receipt.json").read_text())
    expected_binding = dict(
        shared_gate_eligible=False, scratch_optimizer="joint_extra_adam_v1",
        method=row["method"], config_sha256=row["config_sha256"],
        manifest_sha256=manifest_sha,
        implementation_sha256=manifest["implementation_sha256"],
        driver_sha256=manifest["driver_sha256"],
        optimizer_receipt_sha256=digest(directory / "optimizer_receipt.json"),
        generated_function_sha256=manifest["host_sources"][task]["generated_function_sha256"],
    )
    for saved in (protocol, summary, indexed, record, summary["cases"][0]):
        assert saved["scratch_optimizer_policy"] == expected_binding
        assert saved["shared_gate_eligible"] is False
    assert digest(directory / "extra_adam_implementation.py") == manifest["implementation_sha256"]
    assert digest(directory / "extra_adam_driver.py") == manifest["driver_sha256"]
    assert protocol["source_sha256"] == manifest["source_sha256"]
    assert protocol["config_sha256"] == row["config_sha256"]
    _verify_transform(directory, task, manifest["host_sources"][task])
    steps = record["spec"]["steps"]
    evaluations = METHODS[row["method"]]
    assert receipt["method"] == row["method"] and receipt["paper"] == PAPER
    assert receipt["outer_steps"] == steps == manifest["fixed_outer_budgets"][task]
    assert receipt["gradient_evaluations_per_player_per_outer_step"] == evaluations
    assert receipt["extra_gradient_evaluations_per_player_per_outer_step"] == evaluations - 1
    assert receipt["joint_points_verified"] == steps * evaluations
    assert receipt["base_restores_verified"] == (steps if evaluations == 2 else 0)
    assert receipt["common_gate_eligible"] is False
    assert receipt["moments_updated_at_every_gradient_evaluation"] is True
    assert receipt["host_source"] == dict(task=task, **manifest["host_sources"][task],
                                          source_transform_inverse_verified=True)
    applied = {item["role"]: item for item in record["applied"]}
    assert len(receipt["optimizers"]) == 2
    groups = []
    for optimizer in receipt["optimizers"]:
        assert optimizer["calls"] == steps * evaluations
        assert len(optimizer["rates"]) == len(optimizer["diagnostics"]) == steps * evaluations
        assert all(len(item) == len(optimizer["groups"]) for item in optimizer["rates"])
        for position, group in enumerate(optimizer["groups"]):
            groups.append(group["role"])
            expected = applied[group["role"]]
            assert group["parameters"] == expected["parameters"]
            assert group["betas"] == expected["betas"]
            assert _same(group["lr"], expected["lr"])
            assert group["moment_steps"] and all(value == steps * evaluations for value in group["moment_steps"])
            assert all(_same(rates[position], expected["lr"]) for rates in optimizer["rates"])
            for diagnostics in optimizer["diagnostics"]:
                observed = diagnostics[position]
                assert observed["parameters"] == expected["parameters"]
                assert all(math.isfinite(observed[key]) and observed[key] >= 0
                           for key in ("gradient_rms", "move_from_outer_base_rms"))
    assert sorted(groups) == sorted(applied)
    frozen = _episode_rows(directory, (task,), candidate=True, allow_scratch=True)
    assert frozen["status"] in ("PASS", "FAIL"), frozen.get("reason")
    assert frozen["cases"][task]["noise_applied"] is True
    assert _episode_rows(directory, (task,), candidate=True)["status"] == "INVALID"
    case = frozen["cases"][task]
    return dict(task=task, status=case["status"], final=case["final"],
                passing_suffix=case["passing_suffix"],
                gradient_evaluations_per_player=steps * evaluations,
                measured_case_seconds=indexed["seconds"], common_gate_eligible=False,
                source_archive_and_transform_verified=True)


def episode(root: Path, row: dict, task: str) -> dict:
    from benchmarks.transfer_suite.toy100_compatibility import run

    manifest = validate(root)
    output = root / "runs" / row["id"] / task
    manifest_sha = digest(root / "manifest.json")
    with extra_adam(task, row["method"]) as (recorder, generated):
        records = run(root / row["config_file"], output, tasks=(task,))
    assert len(records) == 1 and records[0]["name"] == task
    record = records[0]
    raw_episode = json.loads(gzip.decompress((output / record["artifact"]).read_bytes()))
    if raw_episode["result"].get("error"):
        raise RuntimeError(raw_episode["result"]["error"])
    receipt = recorder.receipt()
    write(output / "optimizer_receipt.json", receipt)
    (output / "generated_host.py").write_text(generated)
    (output / "extra_adam_implementation.py").write_bytes(IMPLEMENTATION.read_bytes())
    (output / "extra_adam_driver.py").write_bytes(DRIVER.read_bytes())
    binding = dict(
        shared_gate_eligible=False, scratch_optimizer="joint_extra_adam_v1",
        method=row["method"], config_sha256=row["config_sha256"],
        manifest_sha256=manifest_sha,
        implementation_sha256=manifest["implementation_sha256"],
        driver_sha256=manifest["driver_sha256"],
        optimizer_receipt_sha256=digest(output / "optimizer_receipt.json"),
        generated_function_sha256=digest(output / "generated_host.py"),
    )
    for name in ("protocol", "summary"):
        payload = json.loads((output / f"{name}.json").read_text())
        payload.update(scratch_optimizer_policy=binding, shared_gate_eligible=False)
        if name == "summary":
            payload["cases"][0].update(scratch_optimizer_policy=binding, shared_gate_eligible=False)
        write(output / f"{name}.json", payload)
    index = json.loads((output / "index.json").read_text())
    artifact = output / index["records"][0]["artifact"]
    saved = json.loads(gzip.decompress(artifact.read_bytes()))
    saved.update(scratch_optimizer_policy=binding, shared_gate_eligible=False)
    raw = (json.dumps(saved, sort_keys=True, allow_nan=False) + "\n").encode()
    artifact.write_bytes(gzip.compress(raw, mtime=0))
    index["records"][0].update(scratch_optimizer_policy=binding, shared_gate_eligible=False,
                                uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    write(output / "index.json", index)
    validate(root)
    verdict = regrade(output, row, manifest, manifest_sha)
    write(output / "scratch_result.json", verdict)
    return verdict


def candidate(root_text: str, row: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    root = Path(root_text)
    directory = root / "runs" / row["id"]
    if directory.exists():
        # A complete declared row can serve as the full-budget harness smoke.
        # Reuse requires a fresh raw/source/transform/optimizer regrade.
        manifest = validate(root)
        saved = json.loads((directory / "row_result.json").read_text())
        assert all(saved[key] == value for key, value in row.items())
        assert saved["status"] in ("READY_EXPAND", "STOPPED")
        for observation in saved["attempted"]:
            checked = regrade(directory / observation["task"], row, manifest, digest(root / "manifest.json"))
            assert checked == observation
        return saved
    directory.mkdir(parents=True)
    result = dict(**row, status="running", attempted=[], skipped=[])
    with (directory / "worker.log").open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            for task in ORDER:
                print(json.dumps(dict(event="START", id=row["id"], task=task)), flush=True)
                observed = episode(root, row, task)
                result["attempted"].append(observed)
                print(json.dumps(dict(event="DONE", id=row["id"], **observed)), flush=True)
                if observed["status"] != "PASS":
                    break
            result["skipped"] = list(ORDER[len(result["attempted"]):])
            result["status"] = "READY_EXPAND" if not result["skipped"] and all(
                row["status"] == "PASS" for row in result["attempted"]) else "STOPPED"
        except Exception as error:
            import traceback
            traceback.print_exc()
            result.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        write(directory / "row_result.json", result)
    return result


def screen(root: Path) -> dict:
    manifest = validate(root)
    result = dict(status="running", rows=[], source_commit=manifest["source_commit"])
    write(root / "screen_result.json", result)
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(candidate, str(root), row) for row in manifest["rows"]]
        for future in as_completed(futures):
            row = future.result()
            result["rows"].append(row)
            result["rows"].sort(key=lambda row: row["id"])
            write(root / "screen_result.json", result)
            print(json.dumps(dict(event="ROW_DONE", id=row["id"], status=row["status"],
                                  attempted=row["attempted"], error=row.get("error"))), flush=True)
    result["status"] = "complete"
    write(root / "screen_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "regrade"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.root)
    elif args.action == "run":
        result = screen(args.root)
    else:
        manifest = json.loads((args.root / "manifest.json").read_text())
        assert digest(args.root / "manifest.json") == (args.root / "manifest.sha256").read_text().strip()
        checked = []
        for row in manifest["rows"]:
            for task in ORDER:
                directory = args.root / "runs" / row["id"] / task
                if directory.exists():
                    checked.append(dict(id=row["id"], **regrade(directory, row, manifest, digest(args.root / "manifest.json"))))
        result = dict(status="regraded", rows=checked)
        write(args.root / "independent_regrade.json", result)
    print(json.dumps(dict(action=args.action, rows=len(result["rows"]))), flush=True)


if __name__ == "__main__":
    main()
