"""Explicit bounded diagnostics for calibrating a screen, never qualification.

Registration freezes a selected part of a current calibration profile. Planning
is read-only unless source freezing is requested; no function launches training.
"""
from copy import deepcopy
from pathlib import Path
import json

from . import planning, views
from .api import CapabilityError
from .calibration import _current_profile, calibration_cohort
from .contracts import atomic_json, canonical, file_hash, file_lock, identifier, positive_number, read_json, stable_hash
from .promotion import validate_screening_submission
from .sources import snapshot_source, verify_snapshot


VERSION = "forge-calibration-lane-v1"
EVIDENCE_USE = "calibration_diagnostic"
FIELDS = {"schema_version", "id", "profile", "profile_sha256", "view", "selections", "budgets",
          "execution_backend", "cuda_model", "purpose", "failure_policy", "qualification_reuse"}


def _block(message):
    raise CapabilityError([message])


def _inside(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        _block("calibration artifacts must use repository-relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        _block("calibration artifact is missing or outside the repository")
    return path


def _profile(root, contract):
    identifier(contract["profile"], "calibration profile")
    path = _inside(root, f"configs/forge/calibration/{contract['profile']}.json")
    config = read_json(path)
    if file_hash(path) != contract["profile_sha256"]:
        _block("calibration profile changed from the selected frozen declaration")
    relative = str((path.parent / config["criteria"]).relative_to(root))
    criteria_path = _inside(root, relative)
    criteria = read_json(criteria_path)
    if file_hash(criteria_path) != config.get("criteria_sha256"):
        _block("calibration criteria changed from the frozen profile")
    _current_profile(config, criteria)
    return {"profile": config, "profile_path": str(path.relative_to(root)),
            "profile_sha256": file_hash(path), "criteria": criteria,
            "criteria_path": str(criteria_path.relative_to(root)), "criteria_sha256": file_hash(criteria_path)}


def _validate(contract):
    if not isinstance(contract, dict) or set(contract) != FIELDS or contract.get("schema_version") != 1:
        _block("calibration lane contract has missing or unsupported fields")
    for key in ("id", "profile", "view"):
        identifier(contract[key], key)
    if (contract["qualification_reuse"] is not False
            or contract["failure_policy"] != "continue_registered_diagnostics"):
        _block("calibration lane requires diagnostic-only evidence and explicit bounded continuation")
    if not isinstance(contract["purpose"], str) or not contract["purpose"].strip():
        _block("declare the calibration question before selecting diagnostics")
    if contract["execution_backend"] not in {"cpu", "cuda"} or (
            contract["cuda_model"] is not None and not isinstance(contract["cuda_model"], str)):
        _block("invalid calibration execution backend")
    selections = contract["selections"]
    if not isinstance(selections, list) or not selections:
        _block("explicitly select a nonempty bounded set of calibration lineages and tasks")
    seen, selected = set(), set()
    for row in selections:
        if not isinstance(row, dict) or set(row) != {"lineage_id", "tasks", "reason"}:
            _block("each calibration selection needs lineage_id, tasks and reason")
        identifier(row["lineage_id"], "calibration lineage")
        if row["lineage_id"] in seen:
            _block("select each frozen calibration lineage only once")
        seen.add(row["lineage_id"])
        tasks = row["tasks"]
        if (not isinstance(tasks, list) or not tasks or any(not isinstance(t, str) for t in tasks)
                or len(tasks) != len(set(tasks)) or not isinstance(row["reason"], str) or not row["reason"].strip()):
            _block("each lineage needs explicit unique tasks and a diagnostic rationale")
        for name in tasks:
            identifier(name, "calibration task")
        selected.update(tasks)
    budgets = contract["budgets"]
    if not isinstance(budgets, dict) or set(budgets) != {"task_seconds", "candidate_seconds", "campaign_seconds"}:
        _block("freeze task, candidate and campaign calibration budgets")
    if not isinstance(budgets["task_seconds"], dict) or set(budgets["task_seconds"]) != selected:
        _block("calibration task budgets must match exactly the selected tasks")
    for key in ("candidate_seconds", "campaign_seconds"):
        positive_number(budgets[key], key)
    for name, seconds in budgets["task_seconds"].items():
        positive_number(seconds, name + " task budget")


def _resolve(root, lineage, contract, profile):
    request = planning.resolve_idea(root, lineage["candidate_id"], view_id=contract["view"],
        through_tier=3, freeze_source=False, execution_backend=contract["execution_backend"],
        cuda_model=contract["cuda_model"])
    if request.get("preflight_blockers"):
        _block("calibration candidate has unresolved public API capabilities: " + "; ".join(request["preflight_blockers"]))
    validate_screening_submission(request)
    if request["candidate_revision"] != lineage["candidate_revision"]:
        _block("calibration candidate differs from its pinned profile revision")
    names = profile["smoke_tasks"] + profile["reference_tasks"]
    if calibration_cohort(request, names) != profile["cohort"]:
        _block("calibration source, task, prior, RNG, runtime or compute differs from the frozen cohort")
    return request


def _selected_jobs(request, selection, contract, profile):
    selected = set(selection["tasks"])
    if not selected.issubset(set(profile["smoke_tasks"] + profile["reference_tasks"])):
        _block("calibration selections must name tasks from the frozen smoke/reference profile")
    for name in selected:
        task = request["tasks"][name]
        if task.get("preflight_blockers"):
            _block(f"calibration task {name} has unresolved capabilities")
        if any(dep["task"] not in selected for dep in task.get("dependencies", [])):
            _block("calibration task selection must include every gate/checkpoint dependency")
        if task["resources"]["timeout_seconds"] != contract["budgets"]["task_seconds"][name]:
            _block("calibration budgets must match frozen task resource limits")
    jobs = []
    for job in request["jobs"]:
        members = set(job.get("task_ids", [job["task_id"]]))
        if members & selected:
            if not members.issubset(selected):
                _block("calibration selection cannot split an uninterrupted execution group")
            jobs.append(job)
    return jobs


def register(root: Path, contract_path: Path) -> dict:
    """Freeze selected diagnostics and maximum cost; do not enqueue or train."""
    root = Path(root).resolve()
    contract = read_json(contract_path)
    _validate(contract)
    frozen = _profile(root, contract)
    profile = frozen["profile"]
    lineages = {row["id"]: row for row in profile["lineages"]}
    subjects, total = {}, 0
    for selection in contract["selections"]:
        name = selection["lineage_id"]
        if name not in lineages:
            _block("calibration selection names an undeclared lineage")
        request = _resolve(root, lineages[name], contract, profile)
        jobs = _selected_jobs(request, selection, contract, profile)
        maximum = sum(job["budget_seconds"] for job in jobs)
        if maximum > contract["budgets"]["candidate_seconds"]:
            _block("calibration candidate budget cannot reserve every selected task")
        total += maximum
        subjects[name] = {"lineage": lineages[name], "selection": selection, "base_request": request}
    if total > contract["budgets"]["campaign_seconds"]:
        _block("calibration campaign budget cannot reserve the full selected diagnostic set")
    payload = json.loads(canonical({"schema_version": 1, "version": VERSION, "registration_id": contract["id"],
        "contract": contract, "frozen": frozen, "subjects": subjects, "maximum_reserved_seconds": total}))
    artifact = {**payload, "registration_sha256": stable_hash(payload)}
    path = root / "reports/forge/calibration-lanes" / contract["id"] / "registration.json"
    with file_lock(root / "runs/forge/calibration-registration.lock"):
        if path.exists():
            if _load(root, contract["id"]) != artifact:
                _block("calibration registration is immutable; do not tune a registered lane")
        else:
            atomic_json(path, artifact)
    return artifact


def _load(root, registration_id):
    identifier(registration_id, "calibration registration")
    artifact = read_json(root / "reports/forge/calibration-lanes" / registration_id / "registration.json")
    payload = {k: v for k, v in artifact.items() if k != "registration_sha256"}
    if (artifact.get("version") != VERSION or artifact.get("registration_id") != registration_id
            or artifact.get("registration_sha256") != stable_hash(payload)):
        _block("calibration registration changed or has an invalid content hash")
    _validate(artifact["contract"])
    return artifact


def _request(artifact, lineage_id):
    if not isinstance(lineage_id, str) or lineage_id not in artifact["subjects"]:
        _block("calibration request names an unregistered lineage")
    subject, contract = artifact["subjects"][lineage_id], artifact["contract"]
    request = deepcopy(subject["base_request"])
    request["calibration_lane"] = {"registration_id": artifact["registration_id"],
        "registration_sha256": artifact["registration_sha256"], "lineage_id": lineage_id,
        "profile_sha256": artifact["frozen"]["profile_sha256"],
        "criteria_sha256": artifact["frozen"]["criteria_sha256"],
        "cohort_sha256": artifact["frozen"]["profile"]["cohort"]["sha256"],
        "qualification_reuse": False, "failure_policy": contract["failure_policy"]}
    # Keep full task definitions for cohort auditing; assignments authorize only
    # the explicitly selected subset. Separate use keys prevent queue reuse from
    # granting ordinary qualification credit to these diagnostic results.
    for job in request["jobs"]:
        job["qualification_compatibility_key"] = job["compatibility_key"]
        job["science"]["evidence_use"] = EVIDENCE_USE
    planning.rekey_jobs(request["jobs"])
    request["view"] = {**request["view"], "id": "calibration-" + artifact["registration_id"],
        "evidence_scope": EVIDENCE_USE, "assignments": [
            {"task": name, "qualification_tier": 1, "importance": "diagnostic", "order": index}
            for index, name in enumerate(subject["selection"]["tasks"])]}
    views.validate_view(request["view"], request["tasks"])
    request.update(through_tier=1, policy_fingerprint=views.view_fingerprint(request["view"]),
        calibration_campaign={"id": "calibration-" + artifact["registration_id"],
            "budget_seconds": contract["budgets"]["campaign_seconds"],
            "candidate_budget_seconds": contract["budgets"]["candidate_seconds"], "accept_shared_cost_transfer": False})
    return request


def _current(root, artifact):
    if _profile(root, artifact["contract"]) != artifact["frozen"]:
        _block("calibration profile or criteria changed after registration")
    for subject in artifact["subjects"].values():
        fresh = _resolve(root, subject["lineage"], artifact["contract"], artifact["frozen"]["profile"])
        if canonical(fresh) != canonical(subject["base_request"]):
            _block("calibration source, formulation, policy or resources changed after registration")


def plan_calibration(root: Path, registration_id: str, queue_root: Path, *, freeze_source=False) -> list[dict]:
    """Preview all explicitly registered diagnostics; freezing never enqueues."""
    root = Path(root).resolve()
    artifact = _load(root, registration_id)
    _current(root, artifact)
    requests = [_request(artifact, name) for name in artifact["subjects"]]
    if freeze_source:
        for request in requests:
            request["source"]["snapshot_path"] = str(snapshot_source(root, queue_root, request["source"]))
    return requests


def verify_request(root: Path, request: dict) -> dict:
    """Verify archived diagnostics against immutable registration, without I/O mutation."""
    root = Path(root).resolve()
    marker = request.get("calibration_lane")
    if not isinstance(marker, dict) or "promotion" in request:
        _block("calibration evidence requires a registered diagnostic declaration")
    try:
        artifact = _load(root, marker.get("registration_id"))
    except (ValueError, OSError, TypeError) as error:
        _block(f"calibration registration cannot be verified: {error}")
    expected = _request(artifact, marker.get("lineage_id"))
    actual = deepcopy(request)
    for key in ("request_id", "queue_root"):
        actual.pop(key, None)
    campaign_id = actual.pop("campaign_id", None)
    if campaign_id is not None and campaign_id != expected["calibration_campaign"]["id"]:
        _block("calibration campaign differs from the frozen registration")
    actual.get("source", {}).pop("snapshot_path", None)
    if canonical(actual) != canonical(expected):
        _block("calibration evidence differs from its exact frozen diagnostic request")
    return {"registration_id": artifact["registration_id"], "registration_sha256": artifact["registration_sha256"],
            "profile_sha256": artifact["frozen"]["profile_sha256"], "criteria_sha256": artifact["frozen"]["criteria_sha256"],
            "cohort_sha256": marker["cohort_sha256"], "qualification_reuse": False,
            "campaign": deepcopy(expected["calibration_campaign"])}


def validate_submission(root: Path, request: dict) -> dict:
    """Queue mutation-boundary authorization including current source and snapshot."""
    receipt = verify_request(root, request)
    root = Path(root).resolve()
    _current(root, _load(root, receipt["registration_id"]))
    snapshot = request.get("source", {}).get("snapshot_path")
    if not isinstance(snapshot, str) or not snapshot:
        _block("calibration submission requires a frozen source snapshot")
    try:
        verify_snapshot(Path(snapshot), request["source"])
    except (OSError, ValueError) as error:
        _block(f"calibration source snapshot changed: {error}")
    return receipt["campaign"]
