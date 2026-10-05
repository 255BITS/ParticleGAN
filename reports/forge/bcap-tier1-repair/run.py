"""Execute frozen repair declarations through Forge, never a copied trainer."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

from prepare import BASE, QUEUE, REPORT, ROOT, STUDY, bind_contract, emit

from experiments.forge.configuration_search import enqueue_search, report_search
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.planning import load_idea, resolve_idea
from experiments.forge.queue import Queue, drain


def verify_source(source):
    """Verify every scientific input against its committed Git blob."""
    import hashlib
    import subprocess
    references = "".join(source["origin_commit"] + ":" + name + "\n"
                         for name in sorted(source["files"]))
    data = subprocess.run(["git", "cat-file", "--batch"], input=references.encode(),
        cwd=ROOT, stdout=subprocess.PIPE, check=True).stdout
    cursor = 0
    for name in sorted(source["files"]):
        end = data.index(b"\n", cursor)
        header = data[cursor:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("scientific source is not committed: " + name)
        size = int(header[2]); cursor = end + 1
        blob = data[cursor:cursor + size]; cursor += size + 1
        if hashlib.sha256(blob).hexdigest() != source["files"][name]:
            raise ValueError("scientific source differs from committed bytes: " + name)


def search_enqueue():
    plans = read_json(REPORT / "plans.json")
    if file_hash(ROOT / plans["spec"]) != plans["spec_sha256"]:
        raise ValueError("search specification changed since reviewed preparation")
    for trial in plans["trials"]:
        if file_hash(ROOT / trial["declaration"]) != trial["declaration_sha256"]:
            raise ValueError("candidate declaration changed since reviewed preparation")
        request = resolve_idea(ROOT, trial["candidate"], through_tier=1,
            execution_backend="cuda", cuda_model="NVIDIA RTX A6000", queue_root=QUEUE)
        verify_source(request["source"])
        if request["source"]["digest"] != plans["source_digest"]:
            raise ValueError("source changed since reviewed preparation")
    result = enqueue_search(ROOT, QUEUE, STUDY,
                            queue=Queue(QUEUE, report_root=ROOT / "reports/forge", on_completion=None))
    if result["blocked_count"]:
        raise ValueError("search admission blocked; inspect saved report")
    emit("search_enqueued", study=STUDY, configurations=result["submitted_count"],
         logs=str(QUEUE / "events.jsonl"))


def prepare_diagnostics():
    candidate_id = "bcap-original-horizon-diagnostics-v1"
    campaign_id = "bcap-original-horizon-diagnostics-v1"
    if campaign_id in Queue(QUEUE, on_completion=None).inspect().get("campaigns", {}):
        raise ValueError("diagnostics already admitted; declarations are immutable")
    incumbent = "bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212"
    idea = deepcopy(load_idea(ROOT, incumbent))
    for field in ("configuration_id", "search_study_id", "search_report", "resolved_configuration_recipe", "decision_contract"):
        idea.pop(field, None)
    idea.update(id=candidate_id, parent=incumbent, goal="bcap_budget_diagnostics_v1",
        hypothesis="Extra low-rate training at the original schedule horizon may correct incumbent Gaussian drift and ring tails without rescaling the original trajectory.",
        changed_factors=["execution duration3000/1600 with original schedule horizon1000/400"],
        mechanism_class="floor_constant", lifecycle="proposed")
    path = ROOT / f"configs/forge/ideas/{candidate_id}.json"
    atomic_json(path, idea)
    gaussian = "gaussian1d_acquisition_3000_schedule1000_diagnostic_v1"
    ring = "ring16_acquisition_1600_schedule400_diagnostic_v1"
    evidence = read_json(REPORT / "plans.json")["prior_evidence"]
    request = bind_contract(path, view="bcap_budget_diagnostics_v1", candidate_cap=1560,
        campaign_cap=1560, evidence=evidence, control=incumbent,
        task_map={gaussian: "gaussian1d_acquisition", ring: "ring16_acquisition"},
        prediction={"task_id": ring, "metric": "component_covariance_error", "op": "<=", "threshold": .85, "phase": "final"},
        falsifier={"task_id": ring, "metric": "component_covariance_error", "op": ">", "threshold": .85, "phase": "final"})
    campaign = {"id": campaign_id, "budget_seconds": 1560, "candidate_budget_seconds": 1560,
                "description": "Two schedule-preserving incumbent diagnostics, no numerical gate changes or scientific retries."}
    atomic_json(ROOT / f"configs/forge/campaigns/{campaign_id}.json", campaign)
    atomic_json(REPORT / "diagnostic-plans.json", {"schema_version": 1, "scope": "schedule_preserving_duration_diagnostic",
        "candidate": candidate_id, "source_digest": request["source"]["digest"],
        "declaration": str(path.relative_to(ROOT)), "declaration_sha256": file_hash(path),
        "decision_status": request["decision_review"]["status"], "campaign": campaign,
        "task_map": request["decision_review"]["expected"]["task_map"], "qualification_input": False})
    emit("diagnostics_prepared", source_digest=request["source"]["digest"], candidate=candidate_id)


def enqueue_diagnostics():
    plan = read_json(REPORT / "diagnostic-plans.json")
    if file_hash(ROOT / plan["declaration"]) != plan["declaration_sha256"]:
        raise ValueError("diagnostic declaration changed since reviewed preparation")
    if read_json(ROOT / f"configs/forge/campaigns/{plan['campaign']['id']}.json") != plan["campaign"]:
        raise ValueError("diagnostic campaign changed since reviewed preparation")
    request = resolve_idea(ROOT, plan["candidate"], view_id="bcap_budget_diagnostics_v1",
        through_tier=1, execution_backend="cuda", cuda_model="NVIDIA RTX A6000",
        queue_root=QUEUE, freeze_source=True)
    verify_source(request["source"])
    if request["source"]["digest"] != plan["source_digest"]:
        raise ValueError("diagnostic source changed since review")
    queue = Queue(QUEUE, report_root=ROOT / "reports/forge", on_completion=None)
    result = queue.submit(request, plan["campaign"])
    atomic_json(REPORT / "diagnostic-submission.json", {"schema_version": 1, "candidate": plan["candidate"],
        "request_id": result["request"]["request_id"], "campaign": plan["campaign"]["id"], "qualification_input": False})
    emit("diagnostics_enqueued", request_id=result["request"]["request_id"], logs=str(QUEUE / "events.jsonl"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("enqueue", "diagnostic-prepare", "diagnostic-enqueue", "run", "report"))
    args = parser.parse_args()
    if args.stage == "enqueue":
        search_enqueue()
    elif args.stage == "diagnostic-prepare":
        prepare_diagnostics()
    elif args.stage == "diagnostic-enqueue":
        enqueue_diagnostics()
    elif args.stage == "run":
        drain(Queue(QUEUE, report_root=ROOT / "reports/forge", on_completion=None),
              ["0", "1"], workers_per_gpu=1, allow_sharing=False)
    else:
        report = report_search(ROOT, QUEUE, STUDY,
                              queue=Queue(QUEUE, report_root=ROOT / "reports/forge", on_completion=None))
        emit("search_complete", selection=report["selection"], cost=report["cost"])


if __name__ == "__main__":
    main()
