"""Publish the six frozen wide-image diagnostics without training or sampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import export_attempt
from experiments.forge.views import task_fingerprint
from reports.forge.regenerate_technique_inventory import project_receipt

CAMPAIGN = "bispf-fastpath-alternatives-v1"
SOURCE = "d9d72d53761c898bce20102e1995fc3d899c1447"
SOURCE_DIGEST = "c82ed2dc712e4c9ebda80678e453635bd8a16dc74fae1450234ed368aa956213"
CANDIDATES = ("bispf-v2", "bispf-optimized-frobenius-v1", "bispf-optimized-quintic-v1")
TASKS = ("img_stripes2_w64_optimizer_diagnostic", "img_bars4_w64_optimizer_diagnostic")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pointer(path):
    return {"path": str(path), "sha256": file_hash(path)}


def publish(root, queue_root, durable_root):
    root, queue_root, durable_root = map(Path, (root, queue_root, durable_root))
    state = read_json(queue_root / "queue/state.json")
    campaign = state["campaigns"][CAMPAIGN]
    jobs = [job for job in state["jobs"].values()
            if (job.get("cost_owner") or {}).get("campaign") == CAMPAIGN]
    require(len(jobs) == 6 and all(job["status"] == "terminal" for job in jobs),
            "All six declared arms must finish before publication")
    require(campaign["reserved_seconds"] == 0 and campaign["spent_seconds"] <= 1800,
            "Wide campaign reservation/budget incomplete")
    require(all(len(job["attempts"]) == 1 for job in jobs), "Unexpected paid retry")
    rows, raw_rows, receipts = [], {}, []
    destination = root / "reports/forge/bispf-optimized"
    for job in jobs:
        attempt = job["attempts"][0]["attempt_id"]
        source = durable_root / "reports/forge/attempts" / attempt
        durable = root / "reports/forge/attempts" / attempt
        durable.mkdir(parents=True, exist_ok=True)
        for name in ("request", "evidence", "result"):
            original, copied = source / f"{name}.json", durable / f"{name}.json"
            if copied.exists():
                require(file_hash(original) == file_hash(copied), "Hydrated receipt differs")
            else:
                shutil.copyfile(original, copied)
        compact = project_receipt(root, attempt)
        envelope, result = (read_json(durable / f"{name}.json") for name in ("request", "result"))
        request = envelope.get("request", envelope)
        require(request["source"]["origin_commit"] == SOURCE
                and request["source"]["digest"] == SOURCE_DIGEST, "Wide source differs")
        require(request["protocol"]["seed"] == 0
                and request["view"]["evidence_scope"] == "research_diagnostic", "Wide scope differs")
        candidate = request["candidate"]["id"]
        require(candidate in CANDIDATES and len(result["task_results"]) == 1, "Unexpected wide cell")
        row = result["task_results"][0]
        task = row["task_id"]
        require(task in TASKS and (candidate, task) not in raw_rows, "Duplicate/unknown wide cell")
        require(row["gate_status"] in {"PASS", "FAIL"}, "Incomplete or invalid wide result")
        evidence, cost = row["evidence"], row["cost"]
        require(cost["completed_steps"] == 600 and len(evidence["observations"]) == 24,
                "Incomplete update or evaluation budget")
        require(evidence["guards"]["all_finite"]
                and evidence["guards"]["unintended_rng_deviations"] == 0, "Invalid numerical/RNG guard")
        require(all(cost["optimizer_updates"][k] == 600 for k in ("generator", "discriminator", "prior")),
                "Incomplete optimizer budget")
        require(evidence["sampling_law"] == "enumerated_prior_without_output_noise", "Sampling differs")
        media_directory = destination / "wide-media" / candidate
        media = export_attempt(durable, media_directory)[0]
        gif = media_directory / (task + ".gif")
        local = Path(read_json(durable / "evidence.json")["local_artifact_root"])
        local_gif = local / "training.gif"
        if local_gif.exists():
            require(file_hash(local_gif) == file_hash(gif), "Original/local training GIF differs")
        else:
            shutil.copyfile(gif, local_gif)
        checkpoint = evidence["provenance_checkpoint"]
        data_key = '["data","target","training","cuda:0"]'
        row_summary = dict(candidate=candidate, task=task, attempt=attempt,
                           request=request["request_id"], status=row["gate_status"],
                           metrics=compact["task_results"][0]["metrics"],
                           convergence=row["evaluator_result"]["convergence"],
                           completed_steps=600, observations=24,
                           optimizer_updates=cost["optimizer_updates"],
                           wall_seconds=cost["wall_seconds"],
                           sampling_law=evidence["sampling_law"],
                           task_contract=task_fingerprint(request["tasks"][task]),
                           initialization_sha256=stable_hash(row["initialization"]),
                           initial_public_prior=row["initialization"]["prior"],
                           initial_rng_sha256=stable_hash(row["rng"]),
                           final_data_stream_sha256=checkpoint["named_stream_state_sha256"][data_key],
                           guards=dict(all_finite=True, unintended_rng_deviations=0),
                           convolution_rule=row["recipe"].get("optimizer_convolution_rule", "polar"),
                           actual_fastpath_scope="CUDA width64 generator and width64/128 critic convolutions; dense/prior native",
                           media=dict(path=gif.relative_to(root).as_posix(), sha256=file_hash(gif),
                                      receipt=pointer(gif.with_suffix(".json")),
                                      optimizer_updates_added=0, sampling_draws_added=0),
                           raw_artifacts=dict(root=str(local), saved_observer_outputs=evidence["saved_observer_outputs"],
                                              provenance_checkpoint=checkpoint),
                           canonical_result_sha256=compact["provenance"]["canonical_result_hash"])
        require(media["optimizer_updates_added"] == media["sampling_draws_added"] == 0,
                "Media added training or sampling")
        rows.append(row_summary)
        raw_rows[candidate, task] = row_summary
        receipts.append(compact)
    comparisons = {}
    for task in TASKS:
        native = raw_rows[CANDIDATES[0], task]
        checks = {key: all(raw_rows[c, task][key] == native[key] for c in CANDIDATES)
                  for key in ("task_contract", "initialization_sha256", "initial_public_prior",
                              "initial_rng_sha256", "final_data_stream_sha256")}
        require(all(checks.values()), "Matched wide-task conditions differ")
        comparisons[task] = checks
    interpretations = {}
    for candidate in CANDIDATES[1:]:
        regressions = [t for t in TASKS if raw_rows[CANDIDATES[0], t]["status"] == "PASS"
                       and raw_rows[candidate, t]["status"] != "PASS"]
        interpretations[candidate] = dict(native_pass_regressions=regressions,
                                          stripes_positive=raw_rows[candidate, TASKS[0]]["status"] == "PASS",
                                          eligible_for_further_comparison=not regressions
                                          and raw_rows[candidate, TASKS[0]]["status"] == "PASS")
    output = dict(schema="bispf_fastpath_quality_readout_v1", source_commit=SOURCE,
                  source_digest=SOURCE_DIGEST, campaign=CAMPAIGN,
                  budget=dict(reservation_seconds=1800, candidate_reservation_seconds=600,
                              per_arm_seconds=300, spent_seconds=campaign["spent_seconds"],
                              reserved_seconds=0, arms=6, paid_retries=0),
                  scope="Architecture portability diagnostic, not ordinary qualification or natural-image quality",
                  qualification_reuse=False, optimizer_updates_added_by_publication=0,
                  sampling_draws_added_by_publication=0, comparisons=comparisons,
                  rows=sorted(rows,key=lambda r:(r["candidate"],r["task"])),
                  candidate_interpretation=interpretations, attempt_summaries=receipts,
                  publisher_sha256=file_hash(Path(__file__)))
    atomic_json(destination / "wide-quality.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    args = parser.parse_args()
    output = publish(args.root, args.queue_root, args.durable_root)
    print(json.dumps({"rows": len(output["rows"]), "candidate_interpretation": output["candidate_interpretation"],
                      "spent_seconds": output["budget"]["spent_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
