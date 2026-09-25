"""Predeclare and run parallel, fail-fast config screens on frozen older toys.

Each attempted host keeps its entire original budget and strict live gate.
This is a search screen, not a replacement for the fresh common-22 replay.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("trajectory", "residual_student", "img_stripes2", "mode_hold",
         "img_bars4", "vector_overlap", "img_blobs4", "img_intensity2",
         "vector_unequal_mass", "vector_unequal_width")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def source_check(manifest):
    for name, expected in manifest["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"source changed after predeclaration: {name}")
    if digest(__file__) != manifest["driver_sha256"]:
        raise RuntimeError("screen driver changed after predeclaration")


def prepare(base_path, variants_path, output):
    from benchmarks.toy100.config import validate_manifest
    from benchmarks.transfer_suite import suite
    from benchmarks.transfer_suite.public_default_verification import load_declaration
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    (output / "logs").mkdir()
    (output / "candidates").mkdir()
    (output / "source").mkdir()
    base = json.loads(base_path.read_text())
    variants = json.loads(variants_path.read_text())
    names = tuple(job["spec"]["name"] for job in load_declaration()[0])
    if len(names) != 19 or not set(TASKS) <= set(names):
        raise RuntimeError("unexpected frozen toy suite")
    source = suite.snapshot(output / "source")
    source["source_sha256"]["benchmarks/toy100/models.py"] = digest(ROOT / "benchmarks/toy100/models.py")
    (output / "source" / "noise_source.py").write_bytes((ROOT / "benchmarks/toy100/models.py").read_bytes())
    (output / "base.json").write_bytes(base_path.read_bytes())
    rows, seen = [], set()
    for variant in variants:
        tag = variant["tag"]
        if not tag or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in tag) or tag in seen:
            raise ValueError(f"invalid or duplicate tag: {tag}")
        seen.add(tag)
        config = dict(base)
        for key in variant.get("remove", []):
            config.pop(key, None)
        config.update(variant.get("set", {}))
        config["name"] = tag
        validate_manifest(config)
        declared_recipe(config)
        declared_model_policy(config)
        path = output / "configs" / f"{tag}.json"
        write(path, config)
        rows.append({**variant, "config": str(path.relative_to(output)), "config_sha256": digest(path)})
    manifest = dict(schema="toy-constraint-screen-v1", rows=rows, tasks=TASKS,
                    full19=names, base_sha256=digest(output / "base.json"),
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256=source["source_sha256"], driver_sha256=digest(__file__),
                    fixed_training_seeds=True, unchanged_host_budgets=True,
                    unchanged_thresholds=True)
    write(output / "manifest.json", manifest)
    (output / "manifest.sha256").write_text(digest(output / "manifest.json") + "\n")
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(event="PREDECLARED", rows=len(rows), output=str(output),
                          sha256=digest(output / "manifest.json"))), flush=True)


def grade(directory, tasks, manifest, row):
    from benchmarks.toy_suite import _episode_rows
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["source_sha256"] != manifest["source_sha256"]:
        raise RuntimeError("episode source differs from predeclaration")
    if protocol["config_sha256"] != row["config_sha256"]:
        raise RuntimeError("episode config differs from predeclaration")
    result = _episode_rows(directory, tuple(tasks), candidate=True)
    if result["status"] not in ("PASS", "FAIL"):
        raise RuntimeError(f"invalid evidence: {result}")
    result["output"] = str(directory)
    return result


def worker(root_string, row):
    import torch
    from benchmarks.transfer_suite.toy100_compatibility import run
    root = Path(root_string)
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(root / "manifest.json") != (root / "manifest.sha256").read_text().strip():
        raise RuntimeError("manifest hash mismatch")
    source_check(manifest)
    config = root / row["config"]
    if digest(config) != row["config_sha256"]:
        raise RuntimeError("config hash mismatch")
    torch.set_num_threads(1)
    candidate = root / "candidates" / row["tag"]
    candidate.mkdir(exist_ok=True)
    status = dict(tag=row["tag"], config_sha256=row["config_sha256"], stages={},
                  shared_gate_eligible=False, constraints=row.get("constraints", []))
    started = time.perf_counter()
    with (root / "logs" / f"{row['tag']}.log").open("a", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            eligible = True
            stages = [(task, (task,)) for task in manifest["tasks"]] + [("full19", tuple(manifest["full19"]))]
            for name, tasks in stages:
                if not eligible:
                    status["stages"][name] = dict(status="SKIPPED", reason="earlier strict stage failed")
                    continue
                source_check(manifest)
                directory = candidate / name
                print(json.dumps(dict(event="START", tag=row["tag"], stage=name)), flush=True)
                if not directory.exists():
                    run(config, directory, tasks=tasks)
                result = grade(directory, tasks, manifest, row)
                status["stages"][name] = result
                eligible = result["status"] == "PASS" and result["passed"] == len(tasks)
                write(candidate / "status.json", status)
                print(json.dumps(dict(event="DONE", tag=row["tag"], stage=name,
                                      status=result["status"], passed=result["passed"])), flush=True)
            source_check(manifest)
            status["status"] = "PASS19" if eligible else "FAIL"
        except Exception:
            status["status"] = "INVALID"
            status["error"] = traceback.format_exc()
            print(status["error"], flush=True)
    status["seconds"] = time.perf_counter() - started
    write(candidate / "status.json", status)
    return status


def run_screen(output, workers):
    manifest = json.loads((output / "manifest.json").read_text())
    if digest(output / "manifest.json") != (output / "manifest.sha256").read_text().strip():
        raise RuntimeError("manifest hash mismatch")
    source_check(manifest)
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(worker, str(output.resolve()), row): row for row in manifest["rows"]}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write(output / "results.json", dict(rows=results, elapsed_seconds=time.perf_counter()-started))
            print(json.dumps(dict(event="CANDIDATE_DONE", tag=result["tag"], status=result["status"],
                                  stages={key:value["status"] for key,value in result["stages"].items()},
                                  seconds=result["seconds"])), flush=True)
    source_check(manifest)
    summary = dict(rows=len(results), passed19=[r["tag"] for r in results if r["status"] == "PASS19"],
                   invalid=[r["tag"] for r in results if r["status"] == "INVALID"],
                   attempted_stages=sum(v["status"] != "SKIPPED" for r in results for v in r["stages"].values()),
                   elapsed_seconds=time.perf_counter()-started)
    write(output / "summary.json", summary)
    print(json.dumps(dict(event="COMPLETE", **summary)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("prepare")
    pre.add_argument("--base", type=Path, required=True)
    pre.add_argument("--variants", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.base, args.variants, args.output)
    else:
        run_screen(args.output, args.workers)


if __name__ == "__main__":
    main()
