"""Regrade saved word arrays, retain GIFs and certify local bulk evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import sys
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.toy_audit.api_images import score_words
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    report = Path(__file__).parent
    request, original = read_json(args.run / "request.json"), read_json(args.run / "readout.json")
    compact = deepcopy(original)
    metrics_recomputed, graded = 0, []
    api_report = ROOT / "reports/toy_audit/api_contract/five_word_smoke_hold"
    media = api_report / "media"
    media.mkdir(parents=True, exist_ok=True)
    cases, readouts, runs = [], [], []
    for row in compact["rows"]:
        if row["gate_status"] == "BLOCKED":
            continue
        name = row["task"]
        raw = read_json(args.run / name / "adapter-receipt.json")
        grade = grade_result(request["tasks"][name], raw)
        assert grade["gate_status"] == row["gate_status"]
        graded.append(dict(task=name, gate_status=grade["gate_status"]))
        records = torch.load(args.run / name / "observed-records.pt", weights_only=True, map_location="cpu")
        for record in records:
            for views, stored in ((record["views"], record["metrics"]),
                                  (record["confirmation_views"], record["confirmation"]["metrics"])):
                actual = score_words(views[2]["samples"].numpy(), views[0]["samples"].numpy())["metrics"]
                for key, value in actual.items():
                    assert math.isclose(value, stored[key], rel_tol=0, abs_tol=1e-12), (name, record["step"], key)
                metrics_recomputed += 1
        certificate = raw["evidence"]["provenance_checkpoint"]
        verify_artifacts(certificate["artifact_root"], certificate["artifact_manifest"])
        final = torch.load(Path(certificate["artifact_root"]) / certificate["path"], weights_only=True, map_location="cpu")
        assert state_digest(final) == certificate["state_sha256"]
        checkpoint = raw["evidence"].get("checkpoint")
        if checkpoint:
            evidence = raw["evidence"]
            verify_artifacts(evidence["artifact_root"], evidence["artifact_manifest"])
            saved = torch.load(Path(evidence["artifact_root"]) / checkpoint["path"], weights_only=True, map_location="cpu")
            assert state_digest(saved) == checkpoint["state_sha256"]
            assert state_digest(saved["fixture"]) == checkpoint["training_state_sha256"]
        target = media / (name + ".gif")
        shutil.copyfile(args.run / name / "goal.gif", target)
        assert file_hash(target) == row["gif_sha256"]
        row["gif_path"] = "../../toy_audit/api_contract/five_word_smoke_hold/media/" + target.name
        task = request["tasks"][name]
        case = dict(id="image-" + name.replace("_", "-") + "-confirmed-v1", legacy_ids=["source-family-15"],
            goal=task["description"], default_recipe="bcap", default_steps=task["execution"]["steps"],
            eval_samples=1024, thresholds=task["evaluation"]["thresholds"],
            scope="task-only selected BCAP DualNorm verification; no ordinary family qualification",
            sampling=dict(evaluation=raw["evidence"]["sampling_law"]))
        summary = dict(id=case["id"], goal=case["goal"], execution_status="COMPLETE", verdict=row["gate_status"],
            completed_updates=row["new_optimizer_updates"], completed_steps=row["completed_steps"],
            default_updates=task["execution"]["steps"], source_commit=compact["source_commit"],
            gif="media/" + target.name, qualification_input=False)
        cases.append(case); readouts.append(summary)
        runs.append(dict(**summary, recipe=raw["recipe"], source_identity=compact["source_digest"],
                         runtime=compact["runtime"], final_metrics=row["final_metrics"], guards=row["guards"],
                         scope=case["scope"], gif_sha256=row["gif_sha256"]))
    warning_count = args.log.read_text().count("During SVD computation with the selected cusolver driver")
    compact["runtime"].update(cusolver_automatic_fallback_warnings=warning_count,
        driver_changed=False, scientific_retry=False, warning_claim="Observed automatic fallback; its causal role was not tested.")
    compact["saved_array_verification"] = dict(metrics_recomputed=metrics_recomputed,
        grades_recomputed=graded, final_checkpoints_verified=True, acquisition_checkpoint_verified=True,
        new_optimizer_updates=0, new_sampling_draws=0)
    atomic_json(report / "readout.json", compact)
    atomic_json(api_report / "publication.json", dict(schema_version=1, cases=cases, readouts=readouts, runs=runs))
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        archive.add(args.run, arcname="full")
        archive.add(args.log, arcname="full.log")
        for path in sorted(args.run.parent.glob("*.log")):
            if path != args.log:
                archive.add(path, arcname="software/" + path.name)
        for path in sorted(args.run.parent.glob("*.json")):
            archive.add(path, arcname="software/" + path.name)
    receipt = dict(schema_version=1, artifact_archive=str(args.archive.resolve()),
        sha256=file_hash(args.archive), bytes=args.archive.stat().st_size,
        members=len(tarfile.open(args.archive).getmembers()), source_commit=original["source_commit"],
        source_digest=original["source_digest"], stdout_sha256=file_hash(args.log), bulk_tracked=False,
        restoration="Extract archive to an empty directory; frozen run request, source manifest, complete checkpoints, scored arrays and logs are retained.")
    atomic_json(report / "archive.json", receipt)
    print(receipt["sha256"], metrics_recomputed, "saved metric sets verified")


if __name__ == "__main__":
    main()
