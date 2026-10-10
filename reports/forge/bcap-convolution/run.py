"""Run the four unchanged Tier 2 image gates as source-bound Forge diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json
from experiments.forge.planning import plan_summary, resolve_idea
from experiments.forge.queue import Queue, drain

CANDIDATE = "bcap-dualnorm-convolution-v1"
STUDY = "bcap-convolution-images-v1"
TASKS = ("img_stripes2", "img_bars4", "img_blobs4", "img_intensity2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "enqueue", "drain"), default="plan")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    queue = Queue(ROOT / "runs/forge", report_root=ROOT / "reports/forge", on_completion=None)
    if args.stage == "drain":
        drain(queue, args.gpus.split(","), campaign=STUDY)
        print(json.dumps(queue.inspect()["campaigns"][STUDY]), flush=True)
        return
    request = resolve_idea(ROOT, CANDIDATE, study=STUDY, queue_root=queue.root,
                           freeze_source=args.stage == "enqueue")
    assert not request["preflight_blockers"], request["preflight_blockers"]
    assert request["view"]["evidence_scope"] == "research_diagnostic"
    assert set(request["tasks"]) == set(TASKS)
    assert request["protocol"]["seed"] == 0
    assert request["execution_policy"]["mode"] == "complete_current_tier"
    assert all(job["science"]["evidence_use"] == "research_diagnostic" for job in request["jobs"])
    recipe = request["candidate"]["resolved_recipe"]
    assert recipe["optimizer_convolution"] == "per_offset"
    assert recipe["optimizer_family"] == "dualnorm" and recipe["optimizer_smoothing"] == 1e-5
    assert recipe["lr"] == .012 and recipe["d_lr_mult"] == 1.5 and recipe["prior_lr_mult"] == 2.5
    assert recipe["lr_floor"] == recipe["network_lr_floor"] == 1 and recipe["optimizer_momentum"] == 0
    deltas = request["study_review"]["expected"]["substantive_delta"]
    assert len(deltas) == len(TASKS)
    assert all(delta["path"][0] == "recipe" and delta["path"][-1] == "optimizer_convolution"
               and delta["before"] == "none" and delta["after"] == "per_offset" for delta in deltas)
    summary = plan_summary(request, queue.inspect())
    assert summary["worst_case_seconds"] == 7200
    if args.stage == "enqueue":
        entry = queue.submit(request, request["study"]["campaign"])
        summary["request_id"] = entry["request"]["request_id"]
        compact = {key: value for key, value in summary.items() if key != "study_binding"}
        compact.update(source_digest=request["source"]["digest"],
                       source_origin_commit=request["source"]["origin_commit"],
                       candidate_revision=request["candidate_revision"],
                       qualification_input=False, original_task_tier=2,
                       study_admission=request["study_admission"])
        atomic_json(ROOT / "reports/forge/bcap-convolution/plan.json", compact)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
