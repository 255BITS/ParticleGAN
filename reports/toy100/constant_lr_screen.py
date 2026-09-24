"""Predeclared, fail-fast search for one constant-LR recipe across 22 toys.

Every attempted older host runs its complete frozen budget. A row advances
only after an independent strict regrade; ten screening passes permit a fresh
full-19 replay. The three 100-mode problems are reserved for full-19 winners.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO / "configs/toy100/shared_candidate.json"
STAGE_ORDER = (
    "trajectory", "residual_student", "img_stripes2", "mode_hold",
    "img_bars4", "vector_overlap", "img_blobs4", "img_intensity2",
    "vector_unequal_mass", "vector_unequal_width",
)
SOURCE_PATTERNS = ("particlegan/**/*.py", "benchmarks/locked_shared/**/*.py",
                   "benchmarks/smart_descent/*.py", "benchmarks/transfer_suite/*.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def halton(index: int, base: int) -> float:
    value, denominator = 0.0, 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def log_range(low: float, high: float, position: float) -> float:
    return math.exp(math.log(low) + position * (math.log(high) - math.log(low)))


def candidates() -> list[dict]:
    """Sixteen interpretable anchors plus 80 fixed Halton points; no RNG."""
    baseline = dict(lr=.00425, d_lr_mult=1.0, prior_lr_mult=2.0,
                    beta2=.999, reg_kappa=1.176, reg_coeff=6.0)
    rows = [("winner_rate", baseline)]
    for value in (.0002, .00035, .0006, .001, .0016, .0025, .0035):
        rows.append((f"lr_{value:g}", {**baseline, "lr": value}))
    for value in (.5, .75, 1.5, 2.5):
        rows.append((f"d_{value:g}", {**baseline, "lr": .0012,
                                      "d_lr_mult": value}))
    for value in (.5, 1.0, 3.0, 5.0):
        rows.append((f"prior_{value:g}", {**baseline, "lr": .0012,
                                          "prior_lr_mult": value}))
    assert len(rows) == 16
    for index in range(1, 81):
        coordinates = [halton(index, base) for base in (2, 3, 5, 7, 11, 13)]
        beta2 = 1 - 10 ** -(1.6 + 2.1 * coordinates[3])
        proposal = dict(
            lr=log_range(.0002, .005, coordinates[0]),
            d_lr_mult=log_range(.5, 2.5, coordinates[1]),
            prior_lr_mult=log_range(.5, 5.0, coordinates[2]),
            beta2=beta2,
            reg_kappa=.7 + .8 * coordinates[4],
            reg_coeff=log_range(2.0, 10.0, coordinates[5]),
        )
        rows.append((f"halton_{index:03d}", proposal))
    return [dict(index=index, tag=f"c{index:03d}_{tag.replace('.', '_')}", values=values)
            for index, (tag, values) in enumerate(rows)]


def source_hashes() -> dict[str, str]:
    from benchmarks.transfer_suite import suite

    with tempfile.TemporaryDirectory(prefix="constant-lr-source-") as temporary:
        snapshot = suite.snapshot(Path(temporary))
    hashes = snapshot["source_sha256"]
    models = REPO / "benchmarks/toy100/models.py"
    hashes[str(models.relative_to(REPO))] = digest(models)
    return hashes


def prepare(root: Path) -> None:
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    from particlegan.recipes import learning_rate_scale

    if root.exists():
        raise FileExistsError("predeclaration root already exists")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise RuntimeError("commit the constant-LR harness before predeclaration")
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    base_bytes = BASE_CONFIG.read_bytes()
    (root / "base_config.json").write_bytes(base_bytes)
    base = json.loads(base_bytes)
    declared = []
    for row in candidates():
        if re.fullmatch(r"[a-z0-9_-]+", row["tag"]) is None:
            raise RuntimeError("candidate name violates frozen legacy host syntax")
        config = dict(base)
        config.pop("network_lr_horizon_cap", None)
        config.pop("network_lr_floor", None)
        config.update(name=row["tag"], lr_floor=1.0, lr_anneal_start=0.0,
                      lr=row["values"]["lr"],
                      d_lr_mult=row["values"]["d_lr_mult"],
                      prior_lr_mult=row["values"]["prior_lr_mult"],
                      betas=[0.0, row["values"]["beta2"]],
                      reg_kappa=row["values"]["reg_kappa"],
                      reg_coeff=row["values"]["reg_coeff"])
        recipe, _, _ = declared_recipe(config)
        if recipe.lr_floor != 1 or any(learning_rate_scale(step, 7000, 0, 1) != 1
                                       for step in (0, 1, 1600, 4200, 6999)):
            raise RuntimeError("proposed schedule is not constant")
        file = f"configs/{row['tag']}.json"
        write_json(root / file, config)
        declared.append({**row, "config_file": file,
                         "config_sha256": digest(root / file)})
    manifest = dict(
        status="predeclared", source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        driver_sha256=digest(Path(__file__)),
        base_config_sha256=hashlib.sha256(base_bytes).hexdigest(),
        source_sha256=source_hashes(),
        fixed_legacy_seed=0, fixed_toy100_seed=1234,
        stage_order=list(STAGE_ORDER),
        all_attempted_cases_use_complete_frozen_budgets=True,
        constant_rate_definition="lr_floor=1, lr_anneal_start=0, no network cap/floor; G/D/prior multipliers are exactly 1",
        full19_rule="fresh 19-host replay only after 10/10 strict staged passes",
        native_rule="run all three 100-mode problems only after fresh strict full19 PASS",
        rows=declared,
    )
    write_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(digest(root / "manifest.json") + "\n")
    print(json.dumps(dict(event="PREDECLARED", rows=len(declared),
                          manifest_sha256=digest(root / "manifest.json"),
                          source_commit=manifest["source_commit"])), flush=True)


def validate(root: Path) -> dict:
    if digest(root / "manifest.json") != (root / "manifest.sha256").read_text().strip():
        raise RuntimeError("predeclared manifest hash differs")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["driver_sha256"] != digest(Path(__file__)):
        raise RuntimeError("screen driver changed after predeclaration")
    if manifest["source_commit"] != subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
    ).strip():
        raise RuntimeError("source commit differs from predeclaration")
    if digest(root / "base_config.json") != manifest["base_config_sha256"]:
        raise RuntimeError("frozen base configuration changed")
    if manifest["stage_order"] != list(STAGE_ORDER) or len(manifest["rows"]) != 96:
        raise RuntimeError("candidate count or stage order changed")
    for name, expected in manifest["source_sha256"].items():
        if digest(REPO / name) != expected:
            raise RuntimeError(f"source changed: {name}")
    for row in manifest["rows"]:
        if digest(root / row["config_file"]) != row["config_sha256"]:
            raise RuntimeError(f"candidate configuration changed: {row['tag']}")
    return manifest


def grade_stage(directory: Path, task_names: tuple[str, ...],
                manifest: dict, row: dict) -> dict:
    from benchmarks.toy_suite import _episode_rows

    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["source_sha256"] != manifest["source_sha256"]:
        raise RuntimeError("stage source archive differs from predeclaration")
    if protocol["config_sha256"] != row["config_sha256"]:
        raise RuntimeError("stage configuration differs from predeclaration")
    grade = _episode_rows(directory, task_names, candidate=True)
    cases = grade.get("cases", {})
    strict_pass = (grade["status"] == "PASS" and grade["passed"] == len(task_names)
                   and set(cases) == set(task_names)
                   and all(case.get("noise_applied") is True for case in cases.values()))
    index = json.loads((directory / "index.json").read_text())
    seconds = sum(float(item.get("seconds", 0)) for item in index["records"])
    return dict(status="PASS" if strict_pass else grade["status"]
                if grade["status"] != "PASS" else "FAIL",
                strict_pass=strict_pass, passed=grade.get("passed", 0),
                required=len(task_names), tasks=list(task_names),
                cases={name: {key: case.get(key) for key in
                              ("status", "passing_suffix", "noise_applied", "final")}
                       for name, case in cases.items()},
                reason=grade.get("reason"), measured_case_seconds=seconds,
                output=str(directory))


def run_candidate(root_text: str, row: dict, manifest: dict,
                  all_names: tuple[str, ...]) -> dict:
    import torch
    from benchmarks.transfer_suite.toy100_compatibility import run

    root = Path(root_text)
    torch.set_num_threads(1)
    candidate = root / "candidates" / row["tag"]
    candidate.mkdir(parents=True, exist_ok=True)
    state_file = candidate / "stage_status.json"
    state = dict(index=row["index"], tag=row["tag"], values=row["values"],
                 config_sha256=row["config_sha256"],
                 manifest_sha256=digest(root / "manifest.json"),
                 source_commit=manifest["source_commit"],
                 stages={}, shared_gate_eligible=False)
    config = root / row["config_file"]
    log_path = candidate / "worker.log"
    start = time.perf_counter()
    current_stage = None
    with log_path.open("x", buffering=1) as logfile, redirect_stdout(logfile), redirect_stderr(logfile):
        try:
            for name in STAGE_ORDER:
                current_stage = name
                validate(root)
                stage_output = candidate / name
                print(json.dumps(dict(event="START", row=row["tag"], task=name)), flush=True)
                tick = time.perf_counter()
                run(config, stage_output, tasks=(name,))
                grade = grade_stage(stage_output, (name,), manifest, row)
                grade["wall_seconds"] = time.perf_counter() - tick
                state["stages"][name] = grade
                write_json(state_file, state)
                print(json.dumps(dict(event="STRICT_DONE", row=row["tag"], task=name,
                                      status=grade["status"], final=grade["cases"].get(name, {}).get("final"))),
                      flush=True)
                if not grade["strict_pass"]:
                    break
            state["skipped"] = [name for name in STAGE_ORDER if name not in state["stages"]]
            strict_ten = (not state["skipped"] and
                          all(state["stages"][name]["strict_pass"] for name in STAGE_ORDER))
            if strict_ten:
                validate(root)
                current_stage = "full19"
                stage_output = candidate / "full19"
                print(json.dumps(dict(event="FULL19_START", row=row["tag"])), flush=True)
                tick = time.perf_counter()
                run(config, stage_output, tasks=all_names)
                grade = grade_stage(stage_output, all_names, manifest, row)
                grade["wall_seconds"] = time.perf_counter() - tick
                state["stages"]["full19"] = grade
                print(json.dumps(dict(event="FULL19_DONE", row=row["tag"],
                                      status=grade["status"], passed=grade["passed"])), flush=True)
            else:
                state["stages"]["full19"] = dict(status="SKIPPED", strict_pass=False,
                                                 reason="at least one screening host failed")
            state["status"] = ("READY_NATIVE3" if state["stages"]["full19"]["strict_pass"]
                               else "SCREENED")
        except Exception as error:
            state["status"] = "ERROR"
            state["error"] = repr(error)
            if current_stage is not None and current_stage not in state["stages"]:
                state["stages"][current_stage] = dict(
                    status="ERROR", strict_pass=False, reason=repr(error),
                    output=str(candidate / current_stage),
                )
            state["skipped"] = [name for name in STAGE_ORDER if name not in state["stages"]]
            state["stages"].setdefault("full19", dict(
                status="SKIPPED", strict_pass=False,
                reason="screen did not attain ten strict passes",
            ))
            print(json.dumps(dict(event="ERROR", row=row["tag"], error=repr(error))), flush=True)
        state["wall_seconds"] = time.perf_counter() - start
        write_json(state_file, state)
    return state


def run_screen(root: Path, workers: int) -> None:
    from benchmarks.transfer_suite.public_default_verification import load_declaration

    if not 1 <= workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    required_env = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                        ATEN_CPU_CAPABILITY="avx2", MKL_ENABLE_INSTRUCTIONS="AVX2",
                        ONEDNN_MAX_CPU_ISA="AVX2", DNNL_MAX_CPU_ISA="AVX2")
    for name, expected in required_env.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"set {name}={expected} before the screen")
    manifest = validate(root)
    all_names = tuple(job["spec"]["name"] for job in load_declaration()[0])
    if len(all_names) != 19 or not set(STAGE_ORDER) <= set(all_names):
        raise RuntimeError("frozen 19-host declaration differs")
    (root / "candidates").mkdir(exist_ok=True)
    result_file = root / "ledger.jsonl"
    if result_file.exists():
        raise FileExistsError("screen ledger exists; use a fresh predeclaration root")
    with result_file.open("x", buffering=1) as ledger:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = [pool.submit(run_candidate, str(root), row, manifest, all_names)
                    for row in manifest["rows"]]
            for future in as_completed(jobs):
                state = future.result()
                ledger.write(json.dumps(dict(index=state["index"], tag=state["tag"],
                                             status=state["status"],
                                             attempted=list(state["stages"]),
                                             stage_statuses={name: grade["status"]
                                                             for name, grade in state["stages"].items()},
                                             skipped=state.get("skipped", []),
                                             wall_seconds=state["wall_seconds"])) + "\n")
                print(json.dumps(dict(event="CANDIDATE_DONE", index=state["index"],
                                      tag=state["tag"], status=state["status"],
                                      stages={name: grade["status"]
                                              for name, grade in state["stages"].items()},
                                      seconds=round(state["wall_seconds"], 2))), flush=True)
    validate(root)
    statuses = [json.loads((root / "candidates" / row["tag"] / "stage_status.json").read_text())
                for row in manifest["rows"]]
    summary = dict(source_commit=manifest["source_commit"],
                   manifest_sha256=digest(root / "manifest.json"),
                   rows=len(statuses),
                   pass_by_stage={name: sum(state["stages"].get(name, {}).get("strict_pass") is True
                                            for state in statuses)
                                  for name in (*STAGE_ORDER, "full19")},
                   errors=[state["tag"] for state in statuses if state["status"] == "ERROR"],
                   winners=[state["tag"] for state in statuses
                            if state["status"] == "READY_NATIVE3"],
                   total_measured_case_seconds=sum(
                       grade.get("measured_case_seconds", 0)
                       for state in statuses for grade in state["stages"].values()),
                   total_candidate_wall_seconds=sum(state["wall_seconds"] for state in statuses))
    write_json(root / "summary.json", summary)
    print(json.dumps(dict(event="SCREEN_COMPLETE", **summary)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root.resolve())
    else:
        run_screen(args.root.resolve(), args.workers)


if __name__ == "__main__":
    main()
