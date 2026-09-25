"""Run four frozen one-field prior/cap variants on nine canonical transfer hosts."""

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

MANIFEST = ROOT / "reports/toy100/regularization-screen-v1-manifest.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _verify_frozen(manifest: dict) -> None:
    if len(manifest["rows"]) != 4 or len({r["id"] for r in manifest["rows"]}) != 4:
        raise ValueError("expected four unique declared variants")
    suite.verify_source({"source_sha256": manifest["source_sha256"]})
    if _hash(ROOT / "benchmarks/toy100/models.py") != manifest["toy100_models_sha256"]:
        raise RuntimeError("frozen toy100 noise source changed")
    if _hash(ROOT / manifest["source_archive"]) != manifest["source_archive_sha256"]:
        raise RuntimeError("frozen source archive changed")
    if _hash(ROOT / manifest["base_config"]) != manifest["base_config_sha256"]:
        raise RuntimeError("frozen shared base config changed")
    jobs = {job["spec"]["name"]: job for job in plan()}
    if set(manifest["tasks"]) != set(manifest["canonical_jobs_sha256"]):
        raise ValueError("declared tasks and frozen host hashes differ")
    for name, expected in manifest["canonical_jobs_sha256"].items():
        actual = hashlib.sha256(json.dumps(jobs[name], sort_keys=True).encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"canonical frozen host changed: {name}")
    base = _read(ROOT / manifest["base_config"])
    allowed = {("prior_reg", 0.02), ("prior_reg", 0.1),
               ("reg_kappa", 1.0), ("reg_kappa", 1.5)}
    if {(r["changed_field"], r["changed_value"]) for r in manifest["rows"]} != allowed:
        raise ValueError("four-row table differs from predeclared changes")
    for row in manifest["rows"]:
        config_path = ROOT / row["config"]
        if _hash(config_path) != row["config_sha256"]:
            raise RuntimeError(f"frozen candidate config changed: {row['id']}")
        config = _read(config_path)
        changed = {key for key in set(base) | set(config)
                   if key != "name" and base.get(key) != config.get(key)}
        if changed != {row["changed_field"]} or config[row["changed_field"]] != row["changed_value"]:
            raise RuntimeError(f"not a single-field variant: {row['id']}")
        if (row["model_policy"] != {key: config[key] for key in
                ("toy100_model", "network_lr_horizon_cap")}):
            raise RuntimeError(f"model policy changed: {row['id']}")
        if row["noise"] != {key: config[key] for key in
                ("output_noise_std", "output_noise_warmup", "input_noise_std",
                 "input_noise_anneal_end")}:
            raise RuntimeError(f"noise policy changed: {row['id']}")


def _grade(row: dict, tasks: tuple[str, ...], source: dict[str, str], models_hash: str) -> dict:
    directory = ROOT / row["output"]
    protocol = _read(directory / "protocol.json")
    summary = _read(directory / "summary.json")
    if (protocol["config_sha256"] != row["config_sha256"]
            or tuple(summary["tasks"]) != tasks or summary["attempted"] != len(tasks)
            or not summary["full_mechanism"]
            or protocol.get("model_policy") != row["model_policy"]
            or summary.get("model_policy") != row["model_policy"]):
        raise RuntimeError(f"evidence differs from frozen table: {row['id']}")
    if protocol["source_sha256"] != {**source, "benchmarks/toy100/models.py": models_hash}:
        raise RuntimeError(f"source differs from frozen table: {row['id']}")
    grade = _episode_rows(directory, tasks, candidate=True)
    if grade["status"] not in ("PASS", "FAIL") or grade["passed"] != summary["passed"]:
        raise RuntimeError(f"strict episode regrade failed: {row['id']}: {grade.get('reason')}")
    index = _read(directory / "index.json")
    verdicts = [record["verdict"] for record in index["records"]]
    gap = sum(max(0, 5 - verdict["convergence"]["passing_suffix"])
              for verdict in verdicts)
    shortfall = sum(verdict["shortfall"] for verdict in verdicts)
    return dict(id=row["id"], output=row["output"], passed=grade["passed"],
                required=len(tasks), strict_status=grade["status"],
                terminal_suffix_gap=gap, normalized_final_metric_shortfall=shortfall,
                failures=[case["name"] for case in summary["cases"] if case["live"] != "PASS"])


def run(*, verify_only: bool = False) -> dict:
    manifest = _read(MANIFEST)
    _verify_frozen(manifest)
    tasks = tuple(manifest["tasks"])
    output = ROOT / "artifacts/toy100-accuracy/compatibility/regularization-screen-v1"
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    results = {"protocol": "toy100-shared-regularization-results-v1",
               "manifest_sha256": _hash(MANIFEST), "rows": []}
    if verify_only:
        print(json.dumps({"event": "VERIFY", "manifest_sha256": _hash(MANIFEST),
                          "rows": len(manifest["rows"])}), flush=True)
        return results
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    for index, row in enumerate(manifest["rows"], start=1):
        _verify_frozen(manifest)
        directory = ROOT / row["output"]
        if not directory.exists():
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
        elif not (directory / "summary.json").exists():
            raise RuntimeError(f"partial output needs manual audit: {row['id']}")
        result = _grade(row, tasks, manifest["source_sha256"],
                        manifest["toy100_models_sha256"])
        results["rows"].append(result)
        _write(results_path, results)
        print(json.dumps({"event": "DONE", "index": index, **result}), flush=True)
    results["ranking"] = [row["id"] for row in sorted(
        results["rows"], key=lambda row: (-row["passed"], row["terminal_suffix_gap"],
                                          row["normalized_final_metric_shortfall"]))]
    _write(results_path, results)
    print(json.dumps({"event": "COMPLETE", "ranking": results["ranking"]}), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    run(verify_only=parser.parse_args().verify_only)
