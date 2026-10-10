"""Human research plans and generated execution bindings, outside recipes.

The legacy decision engine remains the compatibility reader and enforces the
same admission rules. Its internal contract is generated here, never authored
in a v3 candidate and never used as a scientific gate.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from .contracts import canonical, identifier, positive_number, read_json, stable_hash
from . import decision_contracts as decisions


def study_path(root, study_id):
    return Path(root) / "configs/forge/studies" / f"{identifier(study_id, 'study')}.json"


def scaffold(study_id, candidate, control, view, *, hypothesis=None, execution_backend="cuda"):
    return {"schema_version": 1, "id": identifier(study_id, "study"), "status": "draft",
            "candidate": candidate, "control": {"candidate_id": control, "task_map": {}},
            "hypothesis": hypothesis or "TODO: state the bounded research question",
            "scope": {"view": view, "through_tier": 1, "execution_backend": execution_backend},
            "campaign": {"id": study_id, "candidate_budget_seconds": 900, "budget_seconds": 900},
            "max_rounds": 1,
            "prior_evidence": [{"path": "TODO: compact original JSON", "selector": [],
                                "identity": {"record_id": "TODO"}, "use": "motivation_only"}],
            "prediction": {"task_id": "TODO", "metric": "TODO", "op": ">=", "threshold": 0., "phase": "final"},
            "falsifier": {"task_id": "TODO", "metric": "TODO", "op": "<", "threshold": 0., "phase": "final"},
            "competing_explanation": "TODO: distinguish a competing explanation",
            "terminal_rules": deepcopy(decisions.OUTCOMES)}


def validate_study(study):
    required = {"schema_version", "id", "status", "candidate", "control", "hypothesis", "scope",
                "campaign", "max_rounds", "prior_evidence", "prediction", "falsifier",
                "competing_explanation", "terminal_rules"}
    if (not isinstance(study, dict) or set(study) != required
            or type(study.get("schema_version")) is not int or study["schema_version"] != 1
            or study.get("status") not in {"draft", "ready"}):
        raise ValueError("study requires schema_version 1 and its explicit human plan fields")
    canonical(study)
    for key in ("id", "candidate"):
        identifier(study[key], f"study {key}")
    control = study["control"]
    if not isinstance(control, dict) or set(control) != {"candidate_id", "task_map"}:
        raise ValueError("study control requires candidate_id and task_map; bindings are generated")
    identifier(control["candidate_id"], "study control")
    if not isinstance(control["task_map"], dict):
        raise ValueError("study control.task_map must be a mapping")
    for key, value in control["task_map"].items():
        identifier(key, "study task")
        identifier(value, "study control task")
    scope = study["scope"]
    if (not isinstance(scope, dict) or set(scope) - {"view", "through_tier", "execution_backend", "cuda_model"}
            or {"view", "through_tier", "execution_backend"} - scope.keys()
            or type(scope["through_tier"]) is not int or scope["through_tier"] not in (1, 2, 3)
            or scope["execution_backend"] not in {"cpu", "cuda"}):
        raise ValueError("study scope must select a view, tier and CPU/CUDA cohort; task bindings are generated")
    identifier(scope["view"], "study view")
    if "cuda_model" in scope and (scope["execution_backend"] != "cuda" or not isinstance(scope["cuda_model"], str)
                                  or not scope["cuda_model"].strip()):
        raise ValueError("study cuda_model requires a named CUDA cohort")
    campaign = study["campaign"]
    if not isinstance(campaign, dict) or set(campaign) != {"id", "budget_seconds", "candidate_budget_seconds"}:
        raise ValueError("study campaign requires one authoritative finite campaign and candidate budget")
    identifier(campaign["id"], "study campaign")
    for key in ("budget_seconds", "candidate_budget_seconds"):
        positive_number(campaign[key], f"study {key}")
    if campaign["candidate_budget_seconds"] > campaign["budget_seconds"]:
        raise ValueError("study candidate budget exceeds campaign budget")
    if type(study["max_rounds"]) is not int or study["max_rounds"] != 1 or study["terminal_rules"] != decisions.OUTCOMES:
        raise ValueError("study freezes one bounded round with deterministic stop/review rules")
    for key in ("hypothesis", "competing_explanation"):
        if not isinstance(study[key], str) or not study[key].strip():
            raise ValueError(f"study requires {key}")
    evidence = study["prior_evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("study requires source-bound prior evidence")
    for item in evidence:
        if (not isinstance(item, dict) or set(item) != {"path", "selector", "identity", "use"}
                or not isinstance(item["path"], str) or not isinstance(item["selector"], list)
                or not isinstance(item["identity"], dict) or not item["identity"] or item["use"] != "motivation_only"):
            raise ValueError("study prior evidence requires path, selector, exact identity and motivation_only use")
    for key in ("prediction", "falsifier"):
        signature = study[key]
        if (not isinstance(signature, dict) or set(signature) != {"task_id", "metric", "op", "threshold", "phase"}
                or signature["op"] not in decisions.OPS or signature["phase"] != "final"
                or not isinstance(signature["metric"], str) or not signature["metric"].strip()
                or type(signature["threshold"]) not in (int, float) or not math.isfinite(signature["threshold"])):
            raise ValueError("study prediction/falsifier require finite numerical final-metric signatures")
        identifier(signature["task_id"], "study signature task")
    if study["status"] == "ready" and "TODO" in canonical(study):
        raise ValueError("ready study contains unfinished draft fields")


def load_study(root, value):
    if isinstance(value, dict):
        study = deepcopy(value)
    else:
        path = Path(value)
        path = study_path(root, value) if path.suffix != ".json" and len(path.parts) == 1 else Path(root) / path
        study = read_json(path)
        if path.stem != study.get("id"):
            raise ValueError("study filename must match id")
    validate_study(study)
    return study


def _legacy_request(request, contract):
    return {**request, "candidate": {**request["candidate"], "decision_contract": contract}}


def _contract(study, expected, *, status):
    """Generated compatibility projection; declarations retain sole ownership."""
    return {"schema_version": 1, "status": status,
            "prior_evidence": expected.get("prior_evidence", []),
            "control": {"candidate_id": study["control"]["candidate_id"], "task_map": expected["task_map"],
                        "binding_sha256": expected["control_binding_sha256"]},
            "candidate_binding_sha256": expected["candidate_binding_sha256"],
            "substantive_delta": expected["substantive_delta"],
            "scope": {**{key: expected[key] for key in ("task_ids", "protocol_sha256", "source_digest",
                        "execution_backend", "runtime_cohort_sha256", "jobs_sha256")},
                      "view": study["scope"]["view"], "through_tier": study["scope"]["through_tier"],
                      "max_rounds": study["max_rounds"],
                      "candidate_budget_seconds": study["campaign"]["candidate_budget_seconds"],
                      "campaign_budget_seconds": study["campaign"]["budget_seconds"]},
            **{key: deepcopy(study[key]) for key in ("prediction", "falsifier", "competing_explanation", "terminal_rules")}}


def contract_for(request):
    """Read saved bindings without a checkout or reconstructing old science."""
    if "study" in request:
        return _contract(request["study"], request["study_review"]["expected"], status=request["study"]["status"])
    return request.get("candidate", {}).get("decision_contract")


def inspect_study(root, request):
    from .contracts import file_hash
    study = request["study"]
    validate_study(study)
    # First compute real public-API bindings using the unchanged legacy engine.
    draft = decisions.scaffold(study["control"]["candidate_id"], study["scope"]["view"])
    draft["control"]["task_map"] = study["control"]["task_map"]
    def bind(candidate, tasks, protocol, checkout):
        # Unsupported candidate or primary-control cells stay in the denominator and cannot run.
        # A task-local binding refusal must not erase independently runnable
        # peers. Legacy contracts continue using their original strict resolver.
        combined = decisions._bindings(candidate, {}, protocol, checkout)
        for name, task in tasks.items():
            try:
                resolved = decisions._bindings(candidate, {name: task}, protocol, checkout)
            except ValueError as exc:
                from .api import task_policy_blockers
                if not (task.get("preflight_blockers") or task_policy_blockers(task, candidate)):
                    raise  # A missing preflight refusal must never authorize a host.
                from .views import task_execution_fingerprint, task_evaluation_fingerprint
                resolved = {key: {name: {"status": "BLOCKED", "reason": str(exc)}} for key in combined}
                resolved["task_identity"][name] = {"execution": task_execution_fingerprint(task),
                                                   "evaluation": task_evaluation_fingerprint(task)}
                resolved["evaluation"][name] = deepcopy(task["evaluation"])
            for key in combined:
                combined[key].update(resolved[key])
        return combined
    try:
        initial = decisions.inspect_contract(Path(root), _legacy_request(request, draft), binding_resolver=bind)
    except (ValueError, FileNotFoundError) as exc:
        return {"schema_version": 1, "status": "BLOCKED", "blockers": [f"study binding is unsupported: {exc}"],
                "expected": {}, "actual_bindings": {}, "receipt": {"study_sha256": stable_hash(study)},
                "qualification_input": False, "causal_judgment": "requires_review"}
    expected = initial["expected"]
    evidence, blockers = [], []
    for declared in study["prior_evidence"]:
        item = deepcopy(declared)
        path = Path(root) / item["path"]
        if (Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts
                or not path.resolve().is_relative_to(Path(root).resolve()) or path.suffix != ".json" or not path.is_file()):
            blockers.append("study prior evidence is missing or unsafe; retain the exact original identity")
            continue
        item["sha256"] = file_hash(path)
        try:
            decisions._evidence(Path(root), item)
        except ValueError as exc:
            blockers.append(str(exc))
        evidence.append(item)
    expected["prior_evidence"] = evidence
    if not expected["substantive_delta"]:
        blockers.append("study changes no exercised mechanism or host condition; unchanged/seed-only work is insufficient")
    if study["status"] == "draft":
        blockers.append("study is draft: review actual bindings, prior evidence, finite scope and numerical signatures, then set status=ready")
    contract = _contract(study, expected, status="draft")
    review = initial
    if not blockers:
        contract["status"] = "ready"
        try:
            review = decisions.inspect_contract(Path(root), _legacy_request(request, contract), binding_resolver=bind)
            blockers.extend(review["blockers"])
        except ValueError as exc:
            blockers.append(str(exc))
    receipt = {**review["receipt"], "study_sha256": stable_hash(study),
               "policy_fingerprint": request["policy_fingerprint"]}
    return {**review, "status": "BLOCKED" if blockers else "READY", "blockers": blockers,
            "expected": expected, "receipt": receipt}


def validate_admission(root, request, campaign):
    if root is None:
        raise ValueError("study admission requires repository-bound report_root")
    study = load_study(root, request["study"])
    if load_study(root, study["id"]) != study:
        raise ValueError("study declaration changed since planning; resolve its authoritative declaration again")
    if request["candidate"].get("decision_contract") is not None:
        raise ValueError("study cannot replace a legacy embedded decision_contract")
    if campaign != study["campaign"]:
        raise ValueError("campaign differs from the study's authoritative finite budgets")
    from .planning import resolve_idea
    # Re-resolve trusted declarations, source, gates, jobs and cohort before
    # queue mutation. Caller-edited bindings cannot authorize altered science.
    current = resolve_idea(root, study["candidate"], study=study)
    review = current["study_review"]
    from .technique_inventory import _signature
    if (current["preflight_blockers"] or review["status"] != "READY"
            or request.get("study_admission") != review["receipt"]
            or request.get("study_review") != review
            or request.get("view") != current["view"]
            or request.get("source", {}).get("digest") != current["source"]["digest"]
            or request.get("source", {}).get("files") != current["source"]["files"]
            or _signature(request) != _signature(current)):
        raise ValueError("study admission blocked: " + "; ".join(current["preflight_blockers"] or
                         ["missing/mismatched generated execution binding"]))


def register(state, request):
    if "study" not in request:
        return
    study = request["study"]
    definition = {"declaration": study, "binding": request["study_admission"]}
    registrations = state.setdefault("studies", {})
    previous = registrations.get(study["id"])
    if previous is not None and previous != definition:
        raise ValueError("study declaration and execution binding are immutable after enqueue; use a new study id")
    registrations[study["id"]] = definition


def selected_study_id(request):
    return request.get("study", {}).get("id") or request.get("search_plan", {}).get("study_id")


def frozen_declaration(request):
    return request.get("study") or request.get("search_plan", {}).get("declaration")


def select_observations(states, attempts, candidate_id, study_id, revision_prefix=None):
    """Attach exact compatible receipts to a frozen study without rewriting them.

    The producer request can belong to another study. Scientific compatibility
    keys, rather than study labels, authorize reuse. Original receipt identities
    and costs remain visible and cannot be counted as independent replications.
    """
    requests = [entry["request"] for state in states.values() for entry in state.get("submissions", {}).values()
                if entry["request"].get("candidate", {}).get("id") == candidate_id
                and selected_study_id(entry["request"]) == study_id
                and (revision_prefix is None or entry["request"]["candidate_revision"].startswith(revision_prefix))]
    if not requests:
        raise ValueError("No frozen queue request exists for this candidate/study")
    from .technique_inventory import _signature
    if len({stable_hash({"declaration": frozen_declaration(r), "science": _signature(r),
                         "study_binding": r.get("study_admission")}) for r in requests}) != 1:
        raise ValueError("readout cannot combine different study execution bindings")
    request = requests[0]
    active = set(decisions._authorized_tasks(request))
    keys = {job["compatibility_key"] for job in request["jobs"]
            if set(job.get("task_ids", [job["task_id"]])) <= active}
    selected = []
    for attempt in attempts:
        if attempt["request"].get("candidate_revision") != request["candidate_revision"]:
            continue
        rows = [row for row in attempt["task_results"] if row["task_id"] in active and row.get("compatibility_key") in keys]
        if rows:
            selected.append({**attempt, "task_results": rows})
    return request, selected


def conclude_observations(request, attempts):
    active = [a for a in attempts if not a.get("superseded_by")]
    outcome = decisions.evaluate(request, [row for a in active for row in a["task_results"]])
    outcome.update(study_id=request["study"]["id"], study_sha256=stable_hash(request["study"]),
                   study_binding=request["study_admission"], attempt_ids=sorted(a["attempt_id"] for a in attempts),
                   provenance=[{"attempt_id": a["attempt_id"], "result_hash": a["result_hash"],
                                "valid_receipt": a["valid_receipt"],
                                "producer_request_id": a["request"].get("request_id"),
                                "producer_study_id": a["request"].get("study", {}).get("id")}
                               for a in attempts],
                   reuse_note="Compatible original receipts; overlapping readouts are not independent runs or additive costs")
    return outcome
