"""Compact publication join for the two registered paired source jobs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .source_paired_transport_training import JOBS, VERSION
from .source_routed_ring_training import ROOT, sha, write


def identity(path):
    return dict(path=str(path), sha256=sha(path))


def assemble(artifacts, output_dir):
    artifacts, output_dir = Path(artifacts), Path(output_dir)
    plan = json.loads((artifacts / "plan.json").read_text())
    if plan["jobs"] != JOBS:
        raise ValueError("published paired jobs differ from the plan registered before execution")
    records = []
    for job in plan["jobs"]:
        name = job["task"]
        path = artifacts / (name + "-receipt.json")
        record = json.loads(path.read_text())
        record["training_receipt_archive"] = identity(path)
        record["original_gate_definition"] = "Source stable_live and stable_ema each require completed6000budget and NMSE<=.01,p95<=.2 at all final3original validation observations; publication PASS requires both flags"
        record["added_gate_definition"] = "Clean paired MSE/identity-neutral MSE<=.10 for bothlive+EMA; at least5passing terminal original-cadence observations and full6000budget"
        record["best_observation_scope"] = "Source validation-only EMA NMSE selection; distinct from terminal sustained gates; no test qualification"
        record["model_control_scope"] = "baseline/movable is one model/recipe control on this target law; fixed and candidate controls are not additional laws and were not run"
        media_path = artifacts / (name + "-media.json")
        if media_path.exists():
            media = json.loads(media_path.read_text())
            record["media_receipt_archive"] = identity(media_path)
            record["media_receipt"] = media
            if media["media"] is not None:
                gif = Path(media["media"]["path"])
                record["media"] = str(gif.relative_to(output_dir))
                if sha(ROOT / gif) != media["media"]["sha256"]:
                    raise ValueError("paired GIF differs from actual captured-state media receipt")
        record["executed_source_archive"] = {
            "observer": identity(artifacts / name / "executed-observer.py"),
            "quality": identity(artifacts / name / "executed-quality.py"),
        }
        if record["executed_source_archive"]["quality"]["sha256"] != record["binding"]["quality_evaluator"]["sha256"]:
            raise ValueError("paired executed evaluator archive does not match source binding")
        records.append(record)
    return dict(version=VERSION, records=records, execution_plan_archive=identity(artifacts / "plan.json"),
                attempts_per_family=1, target_laws=2, scientific_jobs=2,
                measured_job_seconds=sum(r["seconds"] for r in records),
                report_generator_sha256=sha(__file__),
                scope="Two unchanged baseline/movable source jobs; original12-arm frozen-selection/test prerequisite retained. No Forge/default qualification, seed study, configuration repair, retry or budget extension.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.output, assemble(args.artifacts, args.output.parent))


if __name__ == "__main__":
    main()
