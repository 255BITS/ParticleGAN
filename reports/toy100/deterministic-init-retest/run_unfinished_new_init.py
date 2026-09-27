#!/usr/bin/env python3
"""Run the 22 user-stopped, reviewed research screens with the merged initializer.

Outputs and per-case logs live outside the checkout. Existing results are never
overwritten; re-running this command resumes only cases without a result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SCOPE = HERE / "retest-closure/coverage-scope.json"
QUEUE = HERE / "retest-queue.json"
PYTHON = Path("/tmp/pr38-default-env/bin/python")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", default="GPU-72c1b506-891d-b8bc-b353-e020585e1c47")
    p.add_argument("--shard", type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    scope = json.loads(SCOPE.read_text())
    rows = {r["id"]: r for r in json.loads(QUEUE.read_text())["rows"]}
    selected_all = [r for r in scope["reviewed_research_cases"]
                if rows[r["queue_row"]]["execution_status"] == "NOT_RUN_USER_STOP"]
    assert len(selected_all) == 22, len(selected_all)
    selected = selected_all[a.shard::2]
    assert PYTHON.is_file()
    manifest = {"schema": 1, "scope_sha256": sha(SCOPE), "queue_sha256": sha(QUEUE),
                "initializer_commit": "c720645ecae6b648e9fc6034e9d6b48ccff06ed3",
                "cases": [r["case_id"] for r in selected_all]}
    manifest_path = a.output / "run-manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    for index, row in enumerate(selected, 1):
        case = row["case_id"]
        preparation = Path(row["preparation_manifest"]["path"]).parent
        proof = Path(row["cpu_proof"]["path"])
        assert sha(preparation / "manifest.json") == row["preparation_manifest"]["sha256"]
        assert sha(proof) == row["cpu_proof"]["sha256"]
        assert json.loads(proof.read_text())["status"] == "PASS"
        output = a.output / case
        if (output / "result.json").exists():
            print(f"{index}/{len(selected)} {case}: already complete", flush=True)
            continue
        assert not output.exists(), f"Partial output requires review: {output}"
        command = [str(PYTHON), "-u", str(preparation / "run_research_mode_hold.py"),
                   "--reviewed-cpu-proof", str(proof), "--output", str(output)]
        log = a.output / f"{case}.log"
        print(f"{index}/{len(selected)} START {case} {datetime.now(timezone.utc).isoformat()} log={log}", flush=True)
        with log.open("wb") as stream:
            result = subprocess.run(command, env=env, cwd=HERE.parents[2], stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        receipt = {"case": case, "exit_code": result.returncode, "command": command,
                   "log": str(log), "output": str(output),
                   "ended_utc": datetime.now(timezone.utc).isoformat()}
        with (a.output / "executions.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt) + "\n")
        print(f"{index}/{len(selected)} END {case} exit={result.returncode}", flush=True)
        if result.returncode:
            raise SystemExit(f"Stopped after error; see {log}")
    print(f"Shard {a.shard} complete", flush=True)


if __name__ == "__main__":
    main()
