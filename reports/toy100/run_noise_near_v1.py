"""Run the predeclared four-variant, eight-host shared-noise follow-up.

The manifest is frozen before training. One row reuses an exact completed
episode; the other four are run once with unchanged host seeds, budgets,
thresholds, and source bytes. Run outputs are deliberately outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy_suite import _episode_rows
from benchmarks.transfer_suite import suite
from benchmarks.transfer_suite.compare_defaults import plan

MANIFEST = ROOT / "reports/toy100/noise-near-v1-manifest.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _verify_frozen(manifest: dict) -> None:
    suite.verify_source({"source_sha256": manifest["source_sha256"]})
    for name, key in (("benchmarks/toy100/models.py", "toy100_models_sha256"),
                      ("benchmarks/toy100/train.py", "toy100_train_sha256")):
        if _hash(ROOT / name) != manifest[key]:
            raise RuntimeError(f"frozen source changed: {name}")
    jobs = {job["spec"]["name"]: job for job in plan()}
    if set(manifest["tasks"]) != set(manifest["canonical_jobs_sha256"]):
        raise ValueError("manifest tasks and frozen job hashes differ")
    for name, expected in manifest["canonical_jobs_sha256"].items():
        actual = hashlib.sha256(json.dumps(jobs[name], sort_keys=True).encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"canonical frozen job changed: {name}")
    if _hash(ROOT / manifest["source_archive"]) != manifest["source_archive_sha256"]:
        raise RuntimeError("manifest source archive changed")
    if len(manifest["rows"]) != 5 or sum(row["kind"] == "new" for row in manifest["rows"]) != 4:
        raise ValueError("declared table is not four variants plus reuse")
    if len({row["id"] for row in manifest["rows"]}) != 5:
        raise ValueError("declared row identifiers are duplicated")
    for row in manifest["rows"]:
        if _hash(ROOT / row["config"]) != row["config_sha256"]:
            raise RuntimeError(f"frozen candidate config changed: {row['id']}")


def _grade(row: dict, tasks: tuple[str, ...], source: dict[str, str]) -> dict:
    directory = ROOT / row["output"]
    protocol = _read(directory / "protocol.json")
    summary = _read(directory / "summary.json")
    if (protocol["config_sha256"] != row["config_sha256"]
            or tuple(summary["tasks"]) != tasks or summary["attempted"] != len(tasks)
            or not summary["full_mechanism"]):
        raise RuntimeError(f"candidate evidence differs from frozen table: {row['id']}")
    if protocol["source_sha256"] != {
        **source, "benchmarks/toy100/models.py": _read(MANIFEST)["toy100_models_sha256"]
    }:
        raise RuntimeError(f"candidate source differs from frozen table: {row['id']}")
    grade = _episode_rows(directory, tasks, candidate=True)
    if grade["status"] not in ("PASS", "FAIL") or grade["passed"] != summary["passed"]:
        raise RuntimeError(f"strict episode regrade failed: {row['id']}: {grade.get('reason')}")
    index = _read(directory / "index.json")
    verdicts = [record["verdict"] for record in index["records"]]
    gap = sum(max(0, 5 - verdict["convergence"]["passing_suffix"])
              for verdict in verdicts)
    shortfall = sum(verdict["shortfall"] for verdict in verdicts)
    return dict(id=row["id"], kind=row["kind"], output=row["output"],
                passed=grade["passed"], required=len(tasks), strict_status=grade["status"],
                terminal_suffix_gap=gap, normalized_final_metric_shortfall=shortfall,
                failures=[case["name"] for case in summary["cases"] if case["live"] != "PASS"])


def run(*, verify_only: bool = False) -> dict:
    manifest = _read(MANIFEST)
    _verify_frozen(manifest)
    tasks = tuple(manifest["tasks"])
    output = ROOT / "artifacts/toy100-accuracy/compatibility/noise-near-v1"
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    results_path = output / "results.json"
    results = {"protocol": "toy100-shared-noise-near-results-v1",
               "manifest_sha256": _hash(MANIFEST), "rows": []}
    if verify_only:
        reuse = next(row for row in manifest["rows"] if row["kind"] == "reuse")
        print(json.dumps({"event": "VERIFY", "manifest_sha256": _hash(MANIFEST),
                          "reused": _grade(reuse, tasks, manifest["source_sha256"])}), flush=True)
        return results
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    for index, row in enumerate(manifest["rows"], start=1):
        _verify_frozen(manifest)
        directory = ROOT / row["output"]
        if row["kind"] == "new" and not directory.exists():
            command = [sys.executable, "-u", "-m",
                       "benchmarks.transfer_suite.toy100_compatibility",
                       "--config", str(ROOT / row["config"]), "--tasks", *tasks,
                       "--output", str(directory)]
            print(json.dumps({"event": "START", "index": index,
                              "id": row["id"], "log": str(logs / f"{row['id']}.log")}), flush=True)
            with (logs / f"{row['id']}.log").open("w") as stream:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, check=False)
            if process.returncode:
                raise RuntimeError(f"runner exited {process.returncode}: {row['id']}")
        elif row["kind"] == "new" and not (directory / "summary.json").exists():
            raise RuntimeError(f"partial prior output needs manual audit: {row['id']}")
        result = _grade(row, tasks, manifest["source_sha256"])
        results["rows"].append(result)
        _write(results_path, results)
        print(json.dumps({"event": "DONE", "index": index, **result}), flush=True)
    results["ranking"] = sorted(
        (row["id"] for row in results["rows"]),
        key=lambda identifier: next(
            (-row["passed"], row["terminal_suffix_gap"],
             row["normalized_final_metric_shortfall"])
            for row in results["rows"] if row["id"] == identifier
        ),
    )
    _write(results_path, results)
    print(json.dumps({"event": "COMPLETE", "ranking": results["ranking"]}), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run(verify_only=args.verify_only)
