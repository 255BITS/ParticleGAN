"""One preregistered robustness stage for a finished, qualified candidate.

Registration and planning never enqueue or train. Screening keeps its fixed
seed; this resolver is the explicit, narrow seed-set exception. Missing cells
and control failures remain in the denominator and cannot confer a default claim.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from . import knowledge, planning, views
from .api import CapabilityError, FormulationContext
from .contracts import atomic_json, canonical, file_lock, identifier, positive_number, read_json, stable_hash
from .sources import snapshot_source, verify_snapshot


VERSION = "forge-promotion-v1"
SCREENING_SEED = 0
FIELDS = {"schema_version", "id", "qualification_view", "candidate_revision", "seeds", "tasks",
          "controls", "budgets", "scoring_weights", "aggregation", "acceptance", "no_tuning",
          "early_stop", "execution_backend", "cuda_model"}
ACCEPTANCE = {"candidate": "all_pass", "controls": "match_declared", "missing": "reject", "conflicts": "reject"}


def _block(message):
    raise CapabilityError([message])


def _validate(contract):
    if not isinstance(contract, dict) or set(contract) != FIELDS or contract.get("schema_version") != 1:
        _block("promotion contract has missing or unsupported fields")
    identifier(contract["id"], "registration id")
    identifier(contract["qualification_view"], "qualification view")
    seeds = contract["seeds"]
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
            or len(seeds) != len(set(seeds))):
        _block("register one explicit unique seed set with at least two integer seeds")
    tasks = contract["tasks"]
    if not isinstance(tasks, list) or not tasks or any(not isinstance(t, str) for t in tasks) or len(tasks) != len(set(tasks)):
        _block("promotion tasks must be a nonempty unique list")
    for name in tasks:
        identifier(name, "promotion task")
    if (contract["no_tuning"] is not True or contract["early_stop"] != "on_required_failure"
            or contract["aggregation"] != "all_registered_cells" or contract["acceptance"] != ACCEPTANCE):
        _block("promotion requires frozen no-tuning, explicit fail-fast, and all-cell acceptance policies")
    if contract["scoring_weights"] not in ("live", "ema") or contract["execution_backend"] not in ("cpu", "cuda"):
        _block("unsupported scoring weights or execution backend")
    if contract["cuda_model"] is not None and not isinstance(contract["cuda_model"], str):
        _block("cuda_model must be null or an explicit model name")
    controls = contract["controls"]
    if not isinstance(controls, list) or not controls:
        _block("promotion must freeze at least one reference control")
    names = []
    for control in controls:
        if not isinstance(control, dict) or set(control) != {"candidate_id", "candidate_revision", "expected_statuses"}:
            _block("controls require pinned identity and an expectation for every task")
        identifier(control["candidate_id"], "control candidate")
        names.append(control["candidate_id"])
        if set(control["expected_statuses"]) != set(tasks) or any(
                status not in ("PASS", "FAIL") for status in control["expected_statuses"].values()):
            _block("each control must declare PASS or FAIL for every registered task")
    if len(names) != len(set(names)):
        _block("reference controls must have distinct candidate identities")
    budgets = contract["budgets"]
    if not isinstance(budgets, dict) or set(budgets) != {"task_seconds", "campaign_seconds", "candidate_seconds"}:
        _block("freeze task, candidate and campaign budgets")
    if set(budgets["task_seconds"]) != set(tasks):
        _block("every task needs its own frozen time budget")
    for key in ("campaign_seconds", "candidate_seconds"):
        positive_number(budgets[key], key)
    for name, seconds in budgets["task_seconds"].items():
        positive_number(seconds, f"task {name} budget")


def _resolve(root, candidate_id, contract):
    try:
        request = planning.resolve_idea(root, candidate_id, view_id=contract["qualification_view"],
            through_tier=3, freeze_source=False, execution_backend=contract["execution_backend"],
            cuda_model=contract["cuda_model"])
    except (ValueError, FileNotFoundError) as error:
        _block(f"promotion source cannot resolve its frozen declaration: {error}")
    if request.get("preflight_blockers"):
        _block("promotion source request is blocked: " + "; ".join(request["preflight_blockers"]))
    return request


def _signature(request):
    return stable_hash({key: request[key] for key in (
        "candidate_revision", "policy_fingerprint", "protocol", "rng", "jobs", "runtime")})


def _execution_identity(request):
    """Compare current inputs without treating a Git origin as scientific bytes.

    Only the provenance commit is omitted. Source files/digest/schema, runtime,
    policies, resources and all other request fields remain bound. Archived
    request verification still compares the original complete declaration.
    """
    value = deepcopy(request)
    value["source"].pop("origin_commit", None)
    return value


def _registration_identity(artifact):
    """Idempotency projection; callers must validate the saved artifact first."""
    value = deepcopy(artifact)
    # This digest includes original provenance and must remain on the artifact;
    # recomputing it for a fresh origin is not an immutable-contract change.
    value.pop("registration_sha256", None)
    for subject in value["subjects"].values():
        subject["base_request"] = _execution_identity(subject["base_request"])
    return canonical(value)


def _finished(root, request):
    """Require real regraded evidence and its concluded readout, not a stamp."""
    if request["view"].get("calibration", {}).get("status") not in ("accepted", "PASS"):
        _block("promotion requires accepted view calibration; provisional screens cannot establish a default")
    from .calibration import verify_calibration
    try:
        calibration = verify_calibration(root, request)
    except (ValueError, OSError, KeyError, TypeError) as error:
        _block(f"promotion calibration cannot be verified: {error}")
    attempts, _ = knowledge._attempts(root)
    expected = {name: job["compatibility_key"] for job in request["jobs"]
                for name in job.get("task_ids", [job["task_id"]])}
    selected, bound = [], {}
    for attempt in attempts:
        if attempt["request"].get("candidate_revision") != request["candidate_revision"]:
            continue
        rows = [row for row in attempt["task_results"] if expected.get(row["task_id"]) == row.get("compatibility_key")]
        if not rows:
            continue
        if not attempt["valid_receipt"]:
            _block("qualification contains an invalid durable receipt")
        if not attempt.get("superseded_by"):
            selected.extend(rows)
        bound[attempt["attempt_id"]] = attempt["result_hash"]
    qualification = views.qualify(request["view"], request["tasks"], selected, candidate=request["candidate"])
    if qualification["status"] != "PASS" or not qualification["eligible"]:
        _block("candidate has not passed every applicable qualification gate")
    records, conflicts = knowledge._records(root)
    if conflicts:
        _block("conflicting readouts block promotion registration")
    for record in records:
        if (record.get("candidate_id") != request["candidate"]["id"]
                or record.get("candidate_revision") != request["candidate_revision"]
                or record.get("evidence_scope") != "current" or record.get("lifecycle") != "concluded"
                or not all(record.get(k) for k in ("conclusion", "comparison", "next_action"))):
            continue
        provenance = record.get("provenance", {}).get("attempts", [])
        if provenance and set(record.get("attempt_ids", [])) == set(bound) and all(
                entry.get("valid_receipt") is True and bound.get(entry.get("attempt_id")) == entry.get("result_hash")
                for entry in provenance) and {entry.get("attempt_id") for entry in provenance} == set(bound):
            return {"readout_id": record["record_id"], "readout_sha256": stable_hash(record), "calibration": calibration,
                    "qualification": qualification, "attempt_hashes": bound}
    _block("finished candidate needs a concluded readout bound to its exact qualification receipts")


def _selected_jobs(request, contract):
    selected = set(contract["tasks"])
    if not selected.issubset(request["tasks"]):
        _block("registered tasks must belong to the qualifying view")
    for name in selected:
        task = request["tasks"][name]
        if task.get("preflight_blockers"):
            _block(f"registered task {name} has unresolved capabilities")
        deps = {d["task"] if isinstance(d, dict) else d for d in task.get("dependencies", [])}
        if not deps.issubset(selected):
            _block("promotion task set must include every dependency")
        if task["evaluation"].get("scoring_weights", "live") != contract["scoring_weights"]:
            _block("promotion cannot switch the task's frozen scoring weights")
        if task["resources"]["timeout_seconds"] != contract["budgets"]["task_seconds"][name]:
            _block("promotion budgets must match the frozen task resource budgets")
    if request["candidate"].get("claim_contract", {}).get("scoring_weights", "live") != contract["scoring_weights"]:
        _block("promotion cannot switch the finished candidate's scoring weights")
    result = []
    for job in request["jobs"]:
        members = set(job.get("task_ids", [job["task_id"]]))
        if members & selected:
            if not members.issubset(selected):
                _block("promotion task selection cannot split an execution group")
            result.append(job)
    return result


def register(root: Path, candidate_id: str, contract_path: Path) -> dict:
    """Persist one immutable registration; no queue work or seed runs occur."""
    root = Path(root).resolve()
    identifier(candidate_id, "candidate id")
    contract = read_json(contract_path)
    _validate(contract)
    if candidate_id in {c["candidate_id"] for c in contract["controls"]}:
        _block("the promotion candidate cannot be its own reference control")
    candidate = _resolve(root, candidate_id, contract)
    if candidate["candidate_revision"] != contract["candidate_revision"]:
        _block("candidate revision differs from the preregistered finished revision")
    qualification = _finished(root, candidate)
    subjects = {candidate_id: {"role": "candidate", "base_request": candidate,
                 "expected_statuses": {task: "PASS" for task in contract["tasks"]}}}
    for control in contract["controls"]:
        request = _resolve(root, control["candidate_id"], contract)
        if request["candidate_revision"] != control["candidate_revision"]:
            _block("reference control revision differs from its frozen declaration")
        if request["source"]["digest"] != candidate["source"]["digest"]:
            _block("candidate and controls must use the same frozen source environment")
        if request["candidate_revision"] in {subject["base_request"]["candidate_revision"] for subject in subjects.values()}:
            _block("controls must be distinct frozen formulations, not renamed duplicate candidates")
        subjects[control["candidate_id"]] = {"role": "control", "base_request": request,
                                            "expected_statuses": control["expected_statuses"]}
    minimum_total = 0
    for subject in subjects.values():
        jobs = _selected_jobs(subject["base_request"], contract)
        minimum = len(contract["seeds"]) * sum(job["budget_seconds"] for job in jobs)
        if minimum > contract["budgets"]["candidate_seconds"]:
            _block("candidate budget cannot reserve the complete registered seed/task set")
        minimum_total += minimum
        subject["signature"] = _signature(subject["base_request"])
    if minimum_total > contract["budgets"]["campaign_seconds"]:
        _block("campaign budget cannot reserve every registered candidate and control")
    payload = {"schema_version": 1, "version": VERSION, "registration_id": contract["id"],
               "candidate_id": candidate_id, "contract": contract, "subjects": subjects,
               "qualification": qualification, "maximum_reserved_seconds": minimum_total}
    payload = json.loads(canonical(payload))
    artifact = {**payload, "registration_sha256": stable_hash(payload)}
    directory = root / "reports/forge/promotions"
    with file_lock(root / "runs/forge/promotion-registration.lock"):
        for path in directory.glob("*/registration.json"):
            old = _load(root, path.parent.name)
            same_candidate = (old["candidate_id"] == candidate_id
                              and old["contract"]["candidate_revision"] == contract["candidate_revision"])
            if old["registration_id"] == contract["id"] or same_candidate:
                if _registration_identity(old) != _registration_identity(artifact):
                    _block("one immutable promotion registration is allowed per finished candidate revision")
                return old
        atomic_json(directory / contract["id"] / "registration.json", artifact)
    return artifact


def _load(root, registration_id):
    identifier(registration_id, "registration id")
    path = Path(root) / "reports/forge/promotions" / registration_id / "registration.json"
    artifact = read_json(path)
    digest = artifact.get("registration_sha256")
    payload = {k: v for k, v in artifact.items() if k != "registration_sha256"}
    if artifact.get("registration_id") != registration_id or digest != stable_hash(payload) or artifact.get("version") != VERSION:
        _block("promotion registration was changed or has an invalid content hash")
    _validate(artifact["contract"])
    return artifact


def _request(artifact, subject_id, seed):
    contract = artifact["contract"]
    if type(seed) is not int or seed not in contract["seeds"] or not isinstance(subject_id, str) or subject_id not in artifact["subjects"]:
        _block("undeclared seed or subject is outside the registered promotion stage")
    subject = artifact["subjects"][subject_id]
    request = deepcopy(subject["base_request"])
    request["tasks"] = {name: request["tasks"][name] for name in contract["tasks"]}
    request["jobs"] = deepcopy(_selected_jobs(subject["base_request"], contract))
    promotion = {"registration_id": artifact["registration_id"], "registration_sha256": artifact["registration_sha256"],
                 "role": subject["role"], "subject": subject_id, "seed": seed,
                 "early_stop": contract["early_stop"], "no_tuning": True}
    request["protocol"] = {**request["protocol"], "seed": seed,
                            "promotion": {"registration_id": artifact["registration_id"],
                                          "registration_sha256": artifact["registration_sha256"]}}
    candidate = request["candidate"]
    context = FormulationContext(recipe_preset=candidate.get("recipe_preset"),
        recipe_overrides=candidate.get("recipe_overrides", {}), prior=candidate["prior"],
        seed=seed, requires_capabilities=candidate.get("requires_capabilities", ()),
        extensions=candidate.get("extensions", {}), initializer=candidate.get("initializer", "deterministic_orthogonal"),
        execution_path=candidate.get("execution_path", "public_trainer"))
    request["rng"] = context.streams.manifest()
    for job in request["jobs"]:
        job["science"].update(protocol=deepcopy(request["protocol"]), seed=seed, rng=deepcopy(request["rng"]))
    planning.rekey_jobs(request["jobs"])
    # A stage is explicit work on the registered set, not automatic promotion of
    # screening survivors. Fail-fast is frozen; skipped cells remain NOT_RUN.
    request["view"] = {**request["view"], "id": "promotion-" + artifact["registration_id"],
        "assignments": [{"task": name, "qualification_tier": 1, "importance": "required", "order": i}
                        for i, name in enumerate(contract["tasks"])]}
    views.validate_view(request["view"], request["tasks"])
    request.update(promotion=promotion, through_tier=1, policy_fingerprint=views.view_fingerprint(request["view"]),
        promotion_campaign={"id": "promotion-" + artifact["registration_id"],
            "budget_seconds": contract["budgets"]["campaign_seconds"],
            "candidate_budget_seconds": contract["budgets"]["candidate_seconds"], "accept_shared_cost_transfer": False})
    return request


def plan_promotion(root: Path, registration_id: str, queue_root: Path, *, freeze_source=False) -> list[dict]:
    """Resolve every declared seed/control upfront; preview is read-only.

    The explicit enqueue path passes ``freeze_source=True`` to materialize the
    validated source snapshots. This function never enqueues or trains.
    """
    root = Path(root).resolve()
    artifact = _load(root, registration_id)
    for name, subject in artifact["subjects"].items():
        fresh = _resolve(root, name, artifact["contract"])
        if _signature(fresh) != subject["signature"]:
            _block("source, formulation, scoring, tasks, runtime or protocol changed since registration; in-stage tuning is forbidden")
        if subject["role"] == "candidate" and _finished(root, fresh) != artifact["qualification"]:
            _block("qualification or calibrated gate acceptance changed since registration")
    requests = [_request(artifact, name, seed) for name in artifact["subjects"] for seed in artifact["contract"]["seeds"]]
    if freeze_source:
        for request in requests:
            request["source"]["snapshot_path"] = str(snapshot_source(root, queue_root, request["source"]))
    return requests


def validate_screening_submission(request: dict) -> None:
    """Enforce the fixed v1 screening draw at the queue's mutation boundary.

    The source-controlled v1 protocol declares seed zero. A new seed is never
    an ordinary screening variant, even if somebody hand-builds a request or
    edits a local protocol card. Only validate_submission's registered stage
    may authorize another seed.
    """
    if (not isinstance(request, dict) or any(key in request for key in (
            "promotion", "promotion_campaign", "calibration_lane", "calibration_campaign"))
            or request.get("view", {}).get("evidence_scope") == "calibration_diagnostic"):
        _block("screening cannot carry unvalidated promotion or calibration metadata")
    if any(key in request for key in ("seed", "random_seed", "rng_seed")):
        _block("screening seed must belong only to the fixed protocol")
    protocol = request.get("protocol")
    if (not isinstance(protocol, dict) or protocol.get("id") != "screening"
            or type(protocol.get("seed")) is not int or protocol["seed"] != SCREENING_SEED
            or "promotion" in protocol):
        _block("screening uses fixed protocol seed 0; seed variation requires a registered promotion")
    candidate = request.get("candidate", {})
    if any(key in candidate for key in ("seed", "random_seed", "rng_seed")):
        _block("candidate-specific screening seeds are forbidden")
    factors = candidate.get("changed_factors", [])
    if factors and all(isinstance(f, str) and f.strip().lower() in {"seed", "random_seed", "rng_seed"} for f in factors):
        _block("seed-only screening ideas are forbidden")
    # Preserve Track B's existing raw-MoG eligibility at this metadata boundary.
    # Other candidates/views retain the original context identity and guards.
    from .atlas_existing_mog import CANDIDATE_ID, VIEW_ID, supports_candidate
    existing_mog_id = (CANDIDATE_ID if supports_candidate(candidate)
                       and request.get("view", {}).get("id") == VIEW_ID else None)
    if candidate.get('id') == 'atlas-existing-mog-radius-observer844-v1' and request.get('view', {}).get('id') == 'atlas_existing_mog_radius_observer844_v1':
        from .atlas844_radius_owner import supports_candidate as supports844
        if supports844(candidate):
            existing_mog_id = candidate['id']
    if candidate.get('id') == 'atlas-existing-mog-longer871-v1' and request.get('view', {}).get('id') == 'atlas_existing_mog_longer871_v1':
        from .atlas871_longer_owner import supports_candidate as supports871
        if supports871(candidate):
            existing_mog_id = candidate['id']
    context = FormulationContext(recipe_preset=candidate.get("recipe_preset"),
        recipe_overrides=candidate.get("recipe_overrides", {}),
        prior=candidate.get("prior"), seed=SCREENING_SEED,
        requires_capabilities=candidate.get("requires_capabilities", ()),
        extensions=candidate.get("extensions", {}), initializer=candidate.get("initializer", "deterministic_orthogonal"),
        execution_path=candidate.get("execution_path", "public_trainer"),
        candidate_id=existing_mog_id)
    expected_rng = context.streams.manifest()
    if canonical(request.get("rng")) != canonical(expected_rng):
        _block("screening RNG manifest must match the fixed named-stream seed and bindings")
    jobs = request.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        _block("screening request lacks resolved scientific jobs")
    research_diagnostic = request.get("view", {}).get("evidence_scope") == "research_diagnostic"
    if research_diagnostic:
        views.validate_view(request["view"], request.get("tasks", {}))
        from .studies import contract_for
        contract = contract_for(request)
        from .decision_contracts import validate_shape
        if candidate.get("schema_version") not in (2, 3) or not isinstance(contract, dict):
            _block("research diagnostics require a ready v2 bounded decision_contract")
        validate_shape(contract)
        review = request.get("study_review" if "study" in request else "decision_review", {})
        admission = request.get("study_admission" if "study" in request else "decision_admission")
        if (contract.get("status") != "ready" or review.get("status") != "READY"
                or admission != review.get("receipt") or not review.get("receipt")
                or request.get("through_tier") != 1 or contract["scope"]["view"] != request["view"]["id"]
                or contract["scope"]["through_tier"] != 1
                or set(contract["scope"]["task_ids"]) != {item["task"] for item in request["view"]["assignments"]}):
            _block("research diagnostics require exact ready bounded scope and decision admission")
    for job in jobs:
        science = job.get("science", {})
        marker_ok = (science.get("evidence_use") == "research_diagnostic" if research_diagnostic
                     else "evidence_use" not in science)
        if (not marker_ok or (research_diagnostic and job.get("compatibility_key") != stable_hash(science))
                or "qualification_compatibility_key" in job
                or type(science.get("seed")) is not int or science["seed"] != SCREENING_SEED
                or canonical(science.get("protocol")) != canonical(protocol)
                or canonical(science.get("rng")) != canonical(expected_rng)):
            _block("job scientific identity overrides the fixed screening protocol or RNG")
    for task in request.get("tasks", {}).values():
        if any(key in task.get("execution", {}) for key in ("seed", "random_seed", "rng_seed")):
            _block("per-task screening seed overrides are forbidden")


def validate_submission(root: Path, request: dict) -> dict:
    """Authorize only an exact registered promotion request at Queue.submit.

    Return the frozen campaign definition for the caller to compare to its
    proposed campaign. Operational snapshot paths may differ; their source
    bytes must match the registration. This does not enqueue or edit files.
    """
    root = Path(root).resolve()
    declaration = request.get("promotion") if isinstance(request, dict) else None
    if not isinstance(declaration, dict):
        _block("promotion submission requires a registered declaration")
    try:
        artifact = _load(root, declaration.get("registration_id"))
    except (FileNotFoundError, ValueError, TypeError) as error:
        _block(f"promotion registration cannot be verified: {error}")
    if declaration.get("registration_sha256") != artifact["registration_sha256"]:
        _block("promotion registration identity was forged or changed")
    # Revalidate every frozen subject, so one changed control cannot be omitted
    # while only the surviving candidate requests are submitted.
    for name, subject in artifact["subjects"].items():
        fresh = _resolve(root, name, artifact["contract"])
        if _signature(fresh) != subject["signature"]:
            _block("promotion source, formulation, scoring, tasks or runtime changed before submission")
        if subject["role"] == "candidate" and _finished(root, fresh) != artifact["qualification"]:
            _block("promotion qualification evidence changed before submission")
    expected = _request(artifact, declaration.get("subject"), declaration.get("seed"))
    received = deepcopy(request)
    for key in ("request_id", "queue_root"):
        received.pop(key, None)
    campaign_id = received.pop("campaign_id", None)
    if campaign_id is not None and campaign_id != expected["promotion_campaign"]["id"]:
        _block("promotion campaign identity differs from the registered stage")
    source = received.get("source", {})
    snapshot = source.pop("snapshot_path", None)
    expected["source"].pop("snapshot_path", None)
    if not isinstance(snapshot, str) or not snapshot:
        _block("promotion submission requires a verified frozen source snapshot")
    if canonical(received) != canonical(expected):
        _block("promotion submission differs from its exact frozen seed, scoring, budget or formulation request")
    try:
        verify_snapshot(Path(snapshot), expected["source"])
    except (OSError, ValueError) as error:
        _block(f"promotion snapshot does not match registered source: {error}")
    return deepcopy(expected["promotion_campaign"])


def _same_registered_request(received, expected):
    """Compare science and policy while ignoring queue-local addressing."""
    received, expected = deepcopy(received), deepcopy(expected)
    for value in (received, expected):
        for key in ("request_id", "queue_root"):
            value.pop(key, None)
        campaign_id = value.pop("campaign_id", None)
        if campaign_id is not None and campaign_id != expected["promotion_campaign"]["id"]:
            return False
        value.get("source", {}).pop("snapshot_path", None)
    return canonical(received) == canonical(expected)


def summarize_promotion(root: Path, registration_id: str) -> dict:
    """Read all registered cells, retaining failures, missing outcomes and costs."""
    root = Path(root).resolve()
    artifact = _load(root, registration_id)
    attempts, issues = knowledge._attempts(root)
    cells, blockers = [], []
    unexpected = []
    for attempt in attempts:
        declaration = attempt["request"].get("promotion", {})
        if declaration.get("registration_id") != registration_id:
            continue
        name, seed = declaration.get("subject"), declaration.get("seed")
        valid = (name in artifact["subjects"] and type(seed) is int and seed in artifact["contract"]["seeds"])
        if valid:
            expected = _request(artifact, name, seed)
            valid = _same_registered_request(attempt["request"], expected)
        if not valid:
            unexpected.append(attempt["attempt_id"])
    if unexpected:
        blockers.append("undeclared seed, subject, protocol or tuning appeared inside the registered stage")
    try:
        for name, subject in artifact["subjects"].items():
            fresh = _resolve(root, name, artifact["contract"])
            if _signature(fresh) != subject["signature"]:
                blockers.append("current source/formulation differs from the registered finished candidate")
            elif subject["role"] == "candidate":
                current_qualification = _finished(root, fresh)
                if current_qualification != artifact["qualification"]:
                    blockers.append("qualification evidence or concluded readout changed after registration")
    except (CapabilityError, ValueError, FileNotFoundError) as error:
        blockers.append(str(error))
    for name, subject in artifact["subjects"].items():
        for seed in artifact["contract"]["seeds"]:
            request = _request(artifact, name, seed)
            keys = {member: job["compatibility_key"] for job in request["jobs"]
                    for member in job.get("task_ids", [job["task_id"]])}
            for task_id, task in request["tasks"].items():
                matches, history = [], []
                for attempt in attempts:
                    old = attempt["request"]
                    if (attempt["attempt_id"] in unexpected
                            or old.get("candidate_revision") != request["candidate_revision"]
                            or old.get("promotion") != request["promotion"]
                            or old.get("protocol") != request["protocol"] or old.get("rng") != request["rng"]):
                        continue
                    for row in attempt["task_results"]:
                        if row["task_id"] == task_id and row.get("compatibility_key") == keys[task_id]:
                            grade = views.grade_result(task, row) if attempt["valid_receipt"] else {"gate_status": "INVALID"}
                            history.append((attempt, row, grade))
                            if not attempt.get("superseded_by"):
                                matches.append((attempt, row, grade))
                statuses = {grade["gate_status"] for _, _, grade in matches}
                signatures = {stable_hash(grade) for _, _, grade in matches}
                status = "NOT_RUN" if not matches else "INVALID" if len(signatures) != 1 else next(iter(statuses))
                expected = subject["expected_statuses"][task_id]
                cells.append({"subject": name, "role": subject["role"], "seed": seed, "task": task_id,
                    "status": status, "expected": expected, "accepted": status == expected,
                    "attempt_ids": [a["attempt_id"] for a, _, _ in history],
                    "outcomes": [{"attempt_id": a["attempt_id"], "status": g["gate_status"],
                                  "superseded_by": a.get("superseded_by"),
                                  "metrics": g.get("metrics", {}), "cost": row.get("cost", {})} for a, row, g in history]})
    # Even an unauthorized stage attempt consumed resources. Keep that cost
    # while rejecting its scientific result and retaining the missing cell.
    observed = [attempt for attempt in attempts
                if attempt["request"].get("promotion", {}).get("registration_id") == registration_id]
    cost = knowledge._cost([row for attempt in observed for row in attempt["task_results"]])
    complete = all(cell["status"] in ("PASS", "FAIL") for cell in cells)
    if observed and cost["unmeasured_tasks"]:
        blockers.append("registered attempts lack complete measured cost")
    if cost["wall_seconds"] is not None and cost["wall_seconds"] > artifact["contract"]["budgets"]["campaign_seconds"]:
        blockers.append("registered stage exceeded its campaign budget")
    for name in artifact["subjects"]:
        subject_rows = [row for attempt in observed if attempt["request"].get("promotion", {}).get("subject") == name
                        for row in attempt["task_results"]]
        subject_cost = knowledge._cost(subject_rows)
        if subject_cost["wall_seconds"] is not None and subject_cost["wall_seconds"] > artifact["contract"]["budgets"]["candidate_seconds"]:
            blockers.append(f"registered subject {name} exceeded its candidate budget")
    passed = complete and all(cell["accepted"] for cell in cells) and not blockers
    return {"schema_version": 1, "registration_id": registration_id,
            "registration_sha256": artifact["registration_sha256"], "candidate_id": artifact["candidate_id"],
            "status": "PASS" if passed else "BLOCKED" if blockers else "FAIL" if complete else "INCOMPLETE",
            "public_default_claim": passed, "complete": complete, "cells": cells,
            "denominator": len(cells), "accepted_cells": sum(c["accepted"] for c in cells),
            "counts": dict(sorted(Counter(c["status"] for c in cells).items())),
            "cost": cost, "blockers": blockers, "receipt_issues": issues, "unexpected_attempt_ids": unexpected,
            "policy": "All registered candidate/control/seed/task outcomes retained; no seed selection or score switching."}
