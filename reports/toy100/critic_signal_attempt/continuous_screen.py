"""Declared, parallel, fail-fast screens for constant-rate research candidates.

These scratch-policy receipts cannot qualify for the production common gate.
Each attempted host retains its complete frozen budget and live thresholds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ORDER = ["mode_hold", "trajectory", "residual_student", "img_stripes2",
         "img_bars4", "vector_overlap", "img_blobs4", "img_intensity2",
         "vector_unequal_mass", "vector_unequal_width"]


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    from benchmarks.transfer_suite import suite
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        result = suite.snapshot(Path(directory))["source_sha256"]
    for path in ("benchmarks/toy100/models.py", "reports/toy100/continuous_screen.py",
                 "reports/toy100/continuous_candidates.py"):
        result[path] = sha(ROOT / path)
    return result


def verify_receipt(receipt, config, *, task):
    """Verify applied fixed-rate policy before using any numeric result."""
    options = receipt["options"]
    expected_eps = {"g": options.get("g_eps", options["network_eps"]),
                    "d": options.get("d_eps", options["network_eps"]),
                    "prior": options["prior_eps"]}
    if receipt.get("effective_eps", expected_eps) != expected_eps:
        raise RuntimeError("effective epsilon differs from declared policy")
    expected_lr = {"g": config["lr"], "d": config["lr"] * config["d_lr_mult"],
                   "prior": config["lr"] * config["prior_lr_mult"]}
    counts = {"g": 0, "d": 0}
    seen = set()
    for update in receipt["updates"]:
        role = update["optimizer_role"]
        counts[role] += 1
        if update["optimizer_step"] != counts[role]:
            raise RuntimeError("missing or repeated optimizer receipt")
        for group in update["groups"]:
            kind = group["role"]
            seen.add(kind)
            if group["eps"] != expected_eps[kind] or group["lr"] != expected_lr[kind]:
                raise RuntimeError("applied epsilon or learning rate differs from declaration")
            expected_betas = config.get("prior_betas") or config["betas"] if kind == "prior" else config["betas"]
            if group["betas"] != expected_betas:
                raise RuntimeError("applied Adam moments differ from declaration")
    if task == "mode_hold" and (counts != {"g": 1200, "d": 1200} or seen != {"g", "d", "prior"}):
        raise RuntimeError("incomplete mode-hold optimizer evidence")
    if not all(counts.values()):
        raise RuntimeError("missing player updates")


def mark_scratch(directory, options):
    """Bind the experimental update to every transferable evidence envelope."""
    marker = {"shared_gate_eligible": False,
              "scratch_optimizer_policy": {"name": "fixed_adam_denominator_floor_v1", "options": options}}
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text())
    index.update(marker)
    for row in index["records"]:
        artifact = directory / row["artifact"]
        record = json.loads(gzip.decompress(artifact.read_bytes()))
        record.update(marker)
        raw = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        artifact.write_bytes(gzip.compress(raw, mtime=0))
        row.update(marker, uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    write(index_path, index)
    for name in ("protocol.json", "summary.json"):
        path = directory / name
        value = json.loads(path.read_text())
        value.update(marker)
        write(path, value)


def worker(output_string, row, sources, tasks):
    import torch
    from benchmarks.transfer_suite.toy100_compatibility import run
    from continuous_candidates import candidate_update
    output = Path(output_string)
    tag = row["tag"]
    directory = output / tag
    directory.mkdir()
    torch.set_num_threads(1)
    config = directory / "config.json"
    write(config, row["config"])
    result = {"tag": tag, "config": row["config"], "options": row.get("options", {}),
              "stages": [], "shared_gate_eligible": False}
    started = time.perf_counter()
    with (directory / "run.log").open("w", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            for task in tasks:
                changed = [name for name, digest in sources.items() if sha(ROOT / name) != digest]
                if changed:
                    raise RuntimeError(f"Source changed since declaration: {changed}")
                print(json.dumps({"event": "START", "tag": tag, "task": task}), flush=True)
                stage_start = time.perf_counter()
                with candidate_update(row.get("options", {"network_eps": 1e-8, "prior_eps": 1e-8})) as receipt:
                    records = run(config, directory / task, tasks=[task])
                # Keep the non-production policy explicit beside every episode.
                policy = {"schema": "constant-rate-scratch-v1", "shared_gate_eligible": False,
                          "options": row.get("options", {}), "receipt": receipt}
                (directory / task / "scratch-policy.json.gz").write_bytes(
                    gzip.compress(json.dumps(policy, sort_keys=True, allow_nan=False).encode(), mtime=0))
                record = records[0]
                if record["verdict"]["status"] not in ("PASS", "FAIL"):
                    raise RuntimeError(f"Invalid host result: {record['verdict']['status']}")
                if not row.get("scheduled_control", False):
                    verify_receipt(receipt, row["config"], task=task)
                if not record["noise_applied"]:
                    raise RuntimeError("noise application was not verified")
                mark_scratch(directory / task, row.get("options", {}))
                stage = {"task": task, "verdict": record["verdict"],
                         "live": record.get("live"),
                         "seconds": time.perf_counter() - stage_start,
                         "episode": record["artifact"] if "artifact" in record else None}
                # Reconstruct the gate from the saved raw episode, not the returned verdict.
                from benchmarks.transfer_suite.protocol import test_verdict
                episode_path = next((directory / task / "episodes").glob("*.json.gz"))
                saved = json.loads(gzip.decompress(episode_path.read_bytes()))
                independent = test_verdict(saved["spec"], saved["result"])
                if independent != record["verdict"]:
                    raise RuntimeError("Saved episode regrade differs")
                from benchmarks.toy_suite import _episode_rows
                audit = _episode_rows(directory / task, (task,), candidate=True, allow_scratch=True)
                if audit["status"] not in ("PASS", "FAIL") or audit["passed"] != int(independent["passed"]):
                    raise RuntimeError(f"Frozen source/config/noise gate disagrees: {audit}")
                if _episode_rows(directory / task, (task,), candidate=True)["status"] != "INVALID":
                    raise RuntimeError("production gate accepted scratch policy")
                stage["frozen_evidence_audit"] = audit["status"]
                stage["episode"] = str(episode_path.relative_to(directory))
                stage["episode_sha256"] = sha(episode_path)
                stage["terminal"] = saved["result"].get("observations", [])[-5:]
                result["stages"].append(stage)
                write(directory / "status.json", result)
                print(json.dumps({"event": "DONE", "tag": tag, "task": task,
                                  "passed": independent["passed"], "seconds": stage["seconds"]}), flush=True)
                if not independent["passed"]:
                    break
            result["status"] = "SCREEN_NUMERIC_PASS" if len(result["stages"]) == len(tasks) and all(
                stage["verdict"]["passed"] for stage in result["stages"]) else "FAIL"
        except Exception:
            result["status"] = "INVALID"
            result["error"] = traceback.format_exc()
            print(result["error"], flush=True)
    result["seconds"] = time.perf_counter() - started
    result["skipped"] = tasks[len(result["stages"]):]
    write(directory / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tasks", nargs="+", default=ORDER)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.declaration.read_text())
    if not rows or len({row["tag"] for row in rows}) != len(rows):
        raise ValueError("nonempty unique candidate tags required")
    for row in rows:
        if not row["tag"] or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in row["tag"]):
            raise ValueError("invalid tag")
        cfg = row["config"]
        if not row.get("scheduled_control", False) and (
                cfg.get("lr_floor") != 1 or "network_lr_horizon_cap" in cfg or "network_lr_floor" in cfg):
            raise ValueError("candidate must remove all LR decay policies")
    sources = source_hashes()
    write(args.output / "manifest.json", {"rows": rows, "tasks": args.tasks,
          "source_sha256": sources, "shared_gate_eligible": False,
          "fixed_seeds": True, "unchanged_host_budgets_and_thresholds": True})
    # Archive exact source bytes; mutations after declaration are rejected.
    import tarfile
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for name in sources:
            archive.add(ROOT / name, arcname=name)
    print(json.dumps({"event": "PREDECLARED", "rows": len(rows), "tasks": args.tasks,
                      "manifest_sha256": sha(args.output / "manifest.json")}), flush=True)
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(worker, str(args.output.resolve()), row, sources, args.tasks) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write(args.output / "results.json", results)
            print(json.dumps({"event": "CANDIDATE_DONE", "tag": result["tag"],
                              "status": result["status"], "seconds": result["seconds"],
                              "stages": [{"task": s["task"], "status": s["verdict"]["status"],
                                          "live": s["live"]} for s in result["stages"]]}), flush=True)
    write(args.output / "summary.json", {"survivors": [r["tag"] for r in results if r["status"] == "SCREEN_NUMERIC_PASS"],
          "invalid": [r["tag"] for r in results if r["status"] == "INVALID"],
          "elapsed_seconds": time.perf_counter() - started,
          "training_case_seconds": sum(s["seconds"] for r in results for s in r["stages"])})


if __name__ == "__main__":
    main()
