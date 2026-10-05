"""Freeze the bounded BCAP repair search; this command never starts training."""
from copy import deepcopy
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import materialize_search, plan_search
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue

STUDY = "bcap-tier1-repair-rates-v1"
QUEUE = ROOT / "runs/forge/bcap-tier1-repair/queue"
REPORT = ROOT / "reports/forge/bcap-tier1-repair"
BASE = "k3p-bcap-matched-v1"
ROUND_CAP = 36000


def emit(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def bind_contract(path, *, view, candidate_cap, campaign_cap,
                  prediction, falsifier, evidence, control=BASE):
    """Bind the existing admission contract to exact current task/source values."""
    idea = read_json(path)
    from experiments.forge.decision_contracts import scaffold
    contract = scaffold(control)
    idea["schema_version"] = 2
    idea["decision_contract"] = contract
    atomic_json(path, idea)
    request = resolve_idea(ROOT, idea["id"], view_id=view, through_tier=1,
                           execution_backend="cuda", cuda_model="NVIDIA RTX A6000",
                           queue_root=QUEUE)
    expected = request["decision_review"]["expected"]
    if not expected["substantive_delta"]:
        raise ValueError("repair candidate must have a substantive effective change")
    contract.update(status="ready", prior_evidence=evidence,
        candidate_binding_sha256=expected["candidate_binding_sha256"],
        substantive_delta=expected["substantive_delta"],
        prediction=prediction, falsifier=falsifier,
        competing_explanation=("Slower prior transport and stronger cap penalties may "
            "reduce drift/tails but impede acquisition, movement or word reconstruction. "
            "A final ring covariance improvement alone cannot replace all sustained gates."))
    contract["control"].update(binding_sha256=expected["control_binding_sha256"],
                                task_map=expected["task_map"])
    contract["scope"].update(view=view, through_tier=1,
        task_ids=expected["task_ids"], max_rounds=1,
        candidate_budget_seconds=candidate_cap, campaign_budget_seconds=campaign_cap,
        **{k: expected[k] for k in ("protocol_sha256", "source_digest", "execution_backend",
                                  "runtime_cohort_sha256", "jobs_sha256")})
    atomic_json(path, idea)
    request = resolve_idea(ROOT, idea["id"], view_id=view, through_tier=1,
        execution_backend="cuda", cuda_model="NVIDIA RTX A6000", queue_root=QUEUE)
    if request["decision_review"]["status"] != "READY" or request["preflight_blockers"]:
        raise ValueError(f"repair admission blocked: {request['decision_review']}")
    atomic_json(ROOT / "runs/forge/bcap-tier1-repair/requests" / (idea["id"] + ".json"), request)
    return request


def prepare():
    if (ROOT / f"reports/forge/configuration-search/{STUDY}.json").exists():
        raise ValueError("registered study is immutable; inspect its saved report")
    state = Queue(QUEUE, on_completion=None).inspect()
    if STUDY in state.get("campaigns", {}):
        raise ValueError("admitted study is immutable; create a new round")
    view = read_json(ROOT / "configs/forge/views/discriminator_stability.json")
    assignments = [a for a in view["assignments"] if a["qualification_tier"] == 1]
    candidate_cap = sum(read_json(ROOT / f"configs/forge/tasks/{a['task']}.json")
                        ["resources"]["timeout_seconds"] for a in assignments)
    campaign_cap = candidate_cap * 8
    if campaign_cap > ROUND_CAP:
        raise ValueError("search exceeds the initial repair-round reservation")
    spec = {"schema_version": 1, "id": STUDY, "trainer_family": "bcap",
        "base_candidate": BASE,
        "grid": {"lr": [.002125, .0010625], "d_lr_mult": [2.0],
                 "prior_lr_mult": [.5, 1.0], "reg_coeff": [1.0, 2.0]},
        "tuning_through_tier": 1, "view": view["id"], "execution_backend": "cuda",
        "cuda_model": "NVIDIA RTX A6000", "protocol": "screening",
        "protocol_hash": stable_hash(read_json(ROOT / "configs/forge/protocols/screening.json")),
        "campaign": {"id": STUDY, "budget_seconds": campaign_cap,
            "candidate_budget_seconds": candidate_cap, "accept_shared_cost_transfer": False},
        "hypothesis": "Lower global/prior rates and stronger fixed BCap penalties reduce Gaussian location drift and ring tails while one global recipe retains every Tier 1 behavior.",
        "rationale": "Saved Gaussian outputs have acceptable late standardized shape but drifting location; matched ring contrasts favor prior1/coefficient1 over prior4/coefficient.5. Use lower global rates rather than repeat the measured LR.00425/prior1/coefficient1 arm. The numerical grid keeps D2/cap1 and all mechanisms fixed. Complete independent Tier 1 jobs; no seed experiments, scientific retries, higher tiers or public-default adoption.",
        "guide": "EXPERIMENTATION.md"}
    spec_path = ROOT / f"configs/forge/searches/{STUDY}.json"
    atomic_json(spec_path, spec)
    paths = materialize_search(ROOT, spec)
    evidence = []
    for attempt in ("d0638ad5ce5a47e5b2fcb00b369768b2", "d8185ca486b54af79ef422395eac8065"):
        p = f"reports/forge/technique-receipts/{attempt}.json"
        evidence.append({"path": p, "sha256": file_hash(ROOT / p), "selector": [],
                         "identity": {"attempt_id": attempt}, "use": "motivation_only"})
    reviews = []
    for path in paths:
        idea = read_json(path)
        if idea["search_study_id"] != STUDY:
            raise ValueError("would repeat an existing configuration; revise the grid")
        req = bind_contract(path, view=view["id"], candidate_cap=candidate_cap,
            campaign_cap=campaign_cap, evidence=evidence,
            prediction={"task_id": "ring16_acquisition", "metric": "component_covariance_error",
                        "op": "<=", "threshold": .85, "phase": "final"},
            falsifier={"task_id": "ring16_acquisition", "metric": "component_covariance_error",
                       "op": ">", "threshold": .85, "phase": "final"})
        reviews.append({"candidate": idea["id"], "declaration": str(path.relative_to(ROOT)),
            "declaration_sha256": file_hash(path), "source_digest": req["source"]["digest"],
            "decision_status": req["decision_review"]["status"],
            "tasks": req["decision_review"]["expected"]["task_ids"]})
    plan = plan_search(ROOT, QUEUE, spec)
    if any(t["submission_status"] != "READY" for t in plan["trials"]):
        raise ValueError("all eight contracts must be ready before enqueue")
    atomic_json(ROOT / "runs/forge/bcap-tier1-repair/plan.json", plan)
    atomic_json(REPORT / "plans.json", {"schema_version": 1, "study": STUDY,
        "scope": "bounded_bcap_tier1_repair", "spec": str(spec_path.relative_to(ROOT)),
        "spec_sha256": file_hash(spec_path), "source_digest": plan["source_digest"],
        "policy_fingerprint": plan["policy_fingerprint"], "runtime_cohort": plan["runtime_cohort"],
        "candidate_cap_seconds": candidate_cap, "campaign_cap_seconds": campaign_cap,
        "round_cap_seconds": ROUND_CAP, "gpu_ids": [0, 1], "workers_per_gpu": 1,
        "prior_evidence": evidence, "trials": reviews, "qualification_input": False})
    emit("repair_prepared", configurations=len(paths), campaign_cap_seconds=campaign_cap,
         source_digest=plan["source_digest"], logs=str(QUEUE / "events.jsonl"))


if __name__ == "__main__":
    prepare()
