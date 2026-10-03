"""Frozen-stage contracts with synthetic receipts; no seed experiments run."""
from copy import deepcopy

import pytest

from experiments.forge import promotion
from experiments.forge.api import CapabilityError
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.knowledge import readout
from experiments.forge.planning import resolve_idea
from experiments.forge.sampling import ADAPTER_POLICIES, executed_receipt, PUBLIC_PRIOR_CLEAN


def save_attempt(root, request, name, *, score=1., stamp="PASS", cost=1., tasks=None):
    rows = []
    keys = {member: job["compatibility_key"] for job in request["jobs"]
            for member in job.get("task_ids", [job["task_id"]])}
    for task in tasks or request["tasks"]:
        rows.append({"task_id": task, "compatibility_key": keys[task], "gate_status": stamp,
                     "evidence": {"observations": [{"step": i, "score": score} for i in range(1, 25)],
                                  "live": {"score": score}, "scoring_weights": "live",
                                  **executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean")},
                     "cost": {"wall_seconds": cost[task] if isinstance(cost, dict) else cost}, "raw_status": "completed"})
    result = {"schema_version": 1, "attempt_id": name, "candidate_revision": request["candidate_revision"],
              "task_results": rows}
    directory = root / "reports/forge/attempts" / name
    atomic_json(directory / "request.json", {"request": request})
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "evidence.json", {"result_hash": stable_hash(result), "source": request["source"],
                                             "runtime": request["runtime"]})
    return directory


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path
    # This fixture grades stored synthetic curves and never launches a host.
    # Real adapter applicability is exercised by adapter/worker integration tests.
    monkeypatch.setattr("experiments.forge.adapters.adapter_preflight", lambda task, candidate, **kwargs: [])
    monkeypatch.setitem(ADAPTER_POLICIES, "fixture", PUBLIC_PRIOR_CLEAN)
    prior = {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}
    atomic_json(root / "configs/forge/defaults.json", {"protocol": "screening", "prior": prior})
    atomic_json(root / "configs/forge/protocols/screening.json", {
        "schema_version": 1, "id": "screening", "seed": 0, "rng": {"version": "forge-rng-v1"},
        "scoring": {"weights": "live"}})
    for name, delta in (("finished", .5), ("control", 1.), ("negative1", 0.), ("negative2", .25)):
        atomic_json(root / f"configs/forge/ideas/{name}.json", {
            "schema_version": 1, "id": name, "goal": "stability", "hypothesis": "a mechanism fixture",
            "changed_factors": ["anchor coefficient"], "mechanism_class": "structural",
            "recipe_overrides": {"reg_anchor_weight": delta},
            "claim_contract": {"schedule": "scheduled", "scoring_weights": "live", "sampling_law": "task_declared"}})
    for name in ("cheap", "quality"):
        atomic_json(root / f"configs/forge/tasks/{name}.json", {
            "schema_version": 1, "id": name, "adapter": "fixture", "execution": {"initializer": "deterministic_orthogonal", "steps": 24, "prior": prior, "fixture_id": name},
            "evaluation": {"kind": "transfer_sustained", "thresholds": [["score", ">=", 1.]], "scoring_weights": "live",
                           **executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean")},
            "resources": {"gpus": 0, "gpu_memory_mb": 16, "cpu_threads": 1, "timeout_seconds": 10},
            "requires_capabilities": ["named_rng"], "dependencies": []})
    atomic_json(root / "configs/forge/views/stability.json", {
        "schema_version": 1, "id": "stability", "revision": 1, "goal": "stability", "eligibility": {},
        "calibration": {"status": "accepted", "scope": "synthetic software fixture"},
        "assignments": [{"task": "cheap", "qualification_tier": 1, "importance": "required", "order": 0},
                        {"task": "quality", "qualification_tier": 2, "importance": "required", "order": 0}]})
    (root / "particlegan").mkdir()
    (root / "particlegan/mechanism.py").write_text("fixture = 1\n")
    candidate = resolve_idea(root, "finished", through_tier=3, execution_backend="cpu")
    control = resolve_idea(root, "control", through_tier=3, execution_backend="cpu")
    costs = {"cheap": .05, "quality": 1.}
    save_attempt(root, candidate, "qualification", cost=costs)
    # A real reducer certifies this synthetic software fixture. An accepted
    # config stamp alone is deliberately insufficient to authorize promotion.
    from pathlib import Path
    from experiments.forge.calibration import calibrate, calibration_cohort
    criteria = read_json(Path(__file__).resolve().parents[1] / "configs/forge/calibration/criteria-v1.json")
    criteria_path = root / "configs/forge/calibration/criteria.json"
    atomic_json(criteria_path, criteria)
    lineages = [{"id": "finished", "candidate_id": "finished", "candidate_revision": candidate["candidate_revision"]}]
    for name in ("negative1", "negative2"):
        negative = resolve_idea(root, name, through_tier=3, execution_backend="cpu")
        save_attempt(root, negative, "calibration-" + name, score=0., cost=costs)
        lineages.append({"id": name, "candidate_id": name, "candidate_revision": negative["candidate_revision"]})
    profile_path = root / "configs/forge/calibration/current-fixture.json"
    atomic_json(profile_path, {"schema_version": 1, "id": "current-fixture", "revision": 1,
        "evidence_scope": "current", "criteria": "criteria.json", "criteria_sha256": file_hash(criteria_path),
        "smoke_tasks": ["cheap"], "reference_tasks": ["quality"], "reference_scope": "independent fixture quality",
        "scoring_weights": "live", "training_allowance_seconds": 0, "lineages": lineages,
        "cohort": calibration_cohort(candidate, ["cheap", "quality"])})
    report = calibrate(root, "current-fixture")
    assert report["adoption"] == "PASS"
    report_path = root / "reports/forge/calibration/current-fixture.json"
    view_path = root / "configs/forge/views/stability.json"
    view = read_json(view_path)
    view["calibration"] = {"status": "accepted", "report": str(report_path.relative_to(root)),
        "report_sha256": file_hash(report_path), "profile_sha256": file_hash(profile_path),
        "criteria_sha256": file_hash(criteria_path), "cohort_sha256": calibration_cohort(candidate, ["cheap", "quality"])["sha256"]}
    atomic_json(view_path, view)
    candidate = resolve_idea(root, "finished", through_tier=3, execution_backend="cpu")
    control = resolve_idea(root, "control", through_tier=3, execution_backend="cpu")
    readout(root, "finished", "Fixture passes full view", "Matches declared fixture control", "Register fixed promotion stage")
    contract = {"schema_version": 1, "id": "finished-robustness", "qualification_view": "stability",
                "candidate_revision": candidate["candidate_revision"], "seeds": [0, 1729],
                "tasks": ["cheap", "quality"], "controls": [{"candidate_id": "control",
                "candidate_revision": control["candidate_revision"], "expected_statuses": {"cheap": "PASS", "quality": "PASS"}}],
                "budgets": {"task_seconds": {"cheap": 10, "quality": 10}, "candidate_seconds": 40, "campaign_seconds": 80},
                "scoring_weights": "live", "aggregation": "all_registered_cells", "acceptance": deepcopy(promotion.ACCEPTANCE),
                "no_tuning": True, "early_stop": "on_required_failure", "execution_backend": "cpu", "cuda_model": None}
    path = root / "contract.json"
    atomic_json(path, contract)
    return root, path, contract, candidate


def register(setup):
    root, path, _, _ = setup
    artifact = promotion.register(root, "finished", path)
    requests = promotion.plan_promotion(root, artifact["registration_id"], root / "runs/promotion")
    return artifact, requests


def test_registered_stage_freezes_every_seed_control_and_has_no_queue_side_effect(setup):
    root, path, _, _ = setup
    artifact, requests = register(setup)
    assert len(requests) == 4
    assert {(r["candidate"]["id"], r["protocol"]["seed"]) for r in requests} == {
        ("finished", 0), ("finished", 1729), ("control", 0), ("control", 1729)}
    assert artifact == promotion.register(root, "finished", path)
    assert not list(root.rglob("queue/state.json"))
    report = promotion.summarize_promotion(root, artifact["registration_id"])
    assert report["status"] == "INCOMPLETE" and report["denominator"] == 8
    assert report["counts"] == {"NOT_RUN": 8} and not report["public_default_claim"]


def test_origin_only_advance_preserves_original_promotion_registration(setup, monkeypatch):
    root, path, _, _ = setup
    artifact, requests = register(setup)
    registration = root / "reports/forge/promotions" / artifact["registration_id"] / "registration.json"
    original_bytes = registration.read_bytes()
    inspect = promotion.planning.inspect_source
    monkeypatch.setattr(promotion.planning, "inspect_source", lambda *a, **kw:
                        {**inspect(*a, **kw), "origin_commit": "d" * 40})
    (root / "DOCS_ONLY.md").write_text("Documentation changed; execution inputs did not.\n")
    fresh = resolve_idea(root, "finished", through_tier=3, execution_backend="cpu")
    original = artifact["subjects"]["finished"]["base_request"]
    assert fresh["source"]["origin_commit"] != original["source"]["origin_commit"]
    assert fresh["source"]["digest"] == original["source"]["digest"]
    assert fresh["source"]["files"] == original["source"]["files"]
    assert promotion.register(root, "finished", path) == artifact
    assert promotion.plan_promotion(root, artifact["registration_id"], root / "queue") == requests
    frozen = promotion.plan_promotion(root, artifact["registration_id"], root / "queue", freeze_source=True)[0]
    assert promotion.validate_submission(root, frozen) == frozen["promotion_campaign"]
    assert registration.read_bytes() == original_bytes
    forged = deepcopy(frozen)
    forged["source"]["origin_commit"] = fresh["source"]["origin_commit"]
    with pytest.raises(CapabilityError, match="exact frozen"):
        promotion.validate_submission(root, forged)

    (root / "particlegan/mechanism.py").write_text("fixture = 2\n")
    with pytest.raises(CapabilityError, match="changed since registration"):
        promotion.plan_promotion(root, artifact["registration_id"], root / "queue")
    with pytest.raises(CapabilityError):
        promotion.register(root, "finished", path)
    assert registration.read_bytes() == original_bytes


def test_exact_seed_protocol_reuses_identity_but_never_cross_seed_or_screen(setup):
    root, _, _, screening = setup
    artifact, first = register(setup)
    second = promotion.plan_promotion(root, artifact["registration_id"], root / "runs/promotion")
    keys = lambda requests: [[job["compatibility_key"] for job in r["jobs"]] for r in requests]
    assert keys(first) == keys(second)
    assert keys(first)[0] != keys(first)[1]
    assert keys(first)[0] != [j["compatibility_key"] for j in screening["jobs"]]
    with pytest.raises(CapabilityError, match="undeclared seed"):
        promotion._request(artifact, "finished", 999)


def test_promotion_dependency_receipts_use_the_declared_seed_identity(setup):
    artifact, _ = register(setup)
    # Exercise the resolver's dependency rekeying independently of the task
    # catalog tests that validate real gate/checkpoint declarations.
    base = artifact["subjects"]["finished"]["base_request"]
    by_task = {job["task_id"]: job for job in base["jobs"]}
    by_task["quality"]["science"]["prerequisites"] = {"cheap": by_task["cheap"]["compatibility_key"]}
    first = promotion._request(artifact, "finished", 0)
    second = promotion._request(artifact, "finished", 1729)
    for request in (first, second):
        jobs = {job["task_id"]: job for job in request["jobs"]}
        assert jobs["quality"]["science"]["prerequisites"]["cheap"] == jobs["cheap"]["compatibility_key"]
        assert jobs["quality"]["science"]["prerequisites"]["cheap"] != by_task["cheap"]["compatibility_key"]
    assert first["jobs"] != second["jobs"]


def test_preview_is_read_only_and_explicit_enqueue_freezes_only_registered_sources(setup):
    root, _, _, _ = setup
    artifact, _ = register(setup)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    preview = promotion.plan_promotion(root, artifact["registration_id"], root / "runs/promotion")
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert all("snapshot_path" not in r["source"] for r in preview)
    frozen = promotion.plan_promotion(root, artifact["registration_id"], root / "runs/promotion", freeze_source=True)
    assert all((root / r["source"]["snapshot_path"] / "forge-source.json").exists() for r in frozen)
    assert not list(root.rglob("queue/state.json"))


def test_all_actual_frozen_cells_are_required_for_default_claim(setup):
    root, _, _, _ = setup
    artifact, requests = register(setup)
    for index, request in enumerate(requests):
        save_attempt(root, request, f"stage-{index}")
    report = promotion.summarize_promotion(root, artifact["registration_id"])
    assert report["public_default_claim"] and report["status"] == "PASS"
    assert report["accepted_cells"] == report["denominator"] == 8


@pytest.mark.parametrize("mutation", ["recipe", "scoring", "budget"])
def test_stage_readout_rejects_tampered_request_with_unchanged_revision(setup, mutation):
    root, _, _, _ = setup
    artifact, requests = register(setup)
    for index, request in enumerate(requests):
        save_attempt(root, request, f"stage-{index}")
    path = root / "reports/forge/attempts/stage-0/request.json"
    saved = read_json(path)
    request = saved["request"]
    if mutation == "recipe":
        request["candidate"]["recipe_overrides"]["reg_anchor_weight"] = 99.
    elif mutation == "scoring":
        request["tasks"]["cheap"]["evaluation"]["scoring_weights"] = "ema"
    else:
        request["jobs"][0]["budget_seconds"] *= 2
    atomic_json(path, saved)
    report = promotion.summarize_promotion(root, artifact["registration_id"])
    assert report["status"] == "BLOCKED" and not report["public_default_claim"]
    assert report["unexpected_attempt_ids"] == ["stage-0"]
    assert report["counts"] == {"NOT_RUN": 2, "PASS": 6}
    assert report["cost"]["wall_seconds"] == 4


def test_one_failed_or_skipped_seed_remains_in_denominator(setup):
    root, _, _, _ = setup
    artifact, requests = register(setup)
    save_attempt(root, requests[0], "stage-zero")
    save_attempt(root, requests[1], "stage-one", score=0., stamp="PASS", tasks=["cheap"])
    report = promotion.summarize_promotion(root, artifact["registration_id"])
    assert report["denominator"] == 8 and report["counts"] == {"FAIL": 1, "NOT_RUN": 5, "PASS": 2}
    assert not report["public_default_claim"]


def test_counterfeit_qualification_pass_stamp_cannot_register(setup):
    root, path, _, candidate = setup
    save_attempt(root, candidate, "qualification", score=0., stamp="PASS")
    with pytest.raises(CapabilityError, match="not passed every|calibration"):
        promotion.register(root, "finished", path)


def test_readout_is_required_and_must_bind_actual_receipts(setup):
    root, path, _, _ = setup
    record = next((root / "reports/forge/records").glob("readout-*.json"))
    value = read_json(record)
    value["provenance"]["attempts"][0]["result_hash"] = "counterfeit"
    atomic_json(record, value)
    with pytest.raises(CapabilityError, match="concluded readout"):
        promotion.register(root, "finished", path)


def test_provisional_screen_survivor_cannot_start_promotion(setup):
    root, path, _, _ = setup
    view_path = root / "configs/forge/views/stability.json"
    view = read_json(view_path)
    view["calibration"]["status"] = "provisional"
    atomic_json(view_path, view)
    with pytest.raises(CapabilityError, match="accepted view calibration"):
        promotion.register(root, "finished", path)


@pytest.mark.parametrize("mutation", ["unbound_stamp", "wrong_cohort", "forged_pass_report"])
def test_accepted_calibration_requires_a_recomputed_hash_bound_report(setup, mutation):
    root, path, _, _ = setup
    view_path = root / "configs/forge/views/stability.json"
    view = read_json(view_path)
    if mutation == "unbound_stamp":
        view["calibration"] = {"status": "accepted"}
    elif mutation == "wrong_cohort":
        view["calibration"]["cohort_sha256"] = "0" * 64
    else:
        report_path = root / view["calibration"]["report"]
        report = read_json(report_path)
        report["adoption"] = "PASS"
        report["matrix"][0]["classification"] = "false_accept"
        atomic_json(report_path, report)
        view["calibration"]["report_sha256"] = file_hash(report_path)
    atomic_json(view_path, view)
    with pytest.raises(CapabilityError, match="calibration"):
        promotion.register(root, "finished", path)


@pytest.mark.parametrize("change", ["scoring", "tuning", "seed_selection", "budget", "extra_field"])
def test_contract_rejects_scoring_switch_tuning_selection_and_unbounded_rules(setup, change):
    root, path, contract, _ = setup
    if change == "scoring":
        contract["scoring_weights"] = "ema"
    elif change == "tuning":
        contract["no_tuning"] = False
    elif change == "seed_selection":
        contract["aggregation"] = "best_seed"
    elif change == "budget":
        contract["budgets"]["campaign_seconds"] = 1
    else:
        contract["recipe_overrides"] = {"lr": .01}
    atomic_json(path, contract)
    with pytest.raises(CapabilityError):
        promotion.register(root, "finished", path)


def test_one_registration_per_finished_revision_cannot_select_a_new_seedset(setup):
    root, path, contract, _ = setup
    register(setup)
    contract.update(id="second-attempt", seeds=[0, 2718])
    atomic_json(path, contract)
    with pytest.raises(CapabilityError, match="one immutable"):
        promotion.register(root, "finished", path)


@pytest.mark.parametrize("change", ["source", "recipe", "scoring"])
def test_submit_rejects_source_formulation_and_scoring_tuning(setup, change):
    root, _, _, _ = setup
    artifact, _ = register(setup)
    if change == "source":
        (root / "particlegan/mechanism.py").write_text("fixture = 2\n")
    elif change == "recipe":
        path = root / "configs/forge/ideas/finished.json"
        candidate = read_json(path)
        candidate["recipe_overrides"]["lr"] = .0001
        atomic_json(path, candidate)
    else:
        path = root / "configs/forge/tasks/cheap.json"
        task = read_json(path)
        task["evaluation"]["scoring_weights"] = "ema"
        atomic_json(path, task)
    with pytest.raises(CapabilityError, match="changed since registration|scoring_weights"):
        promotion.plan_promotion(root, artifact["registration_id"], root / "runs/promotion")


def test_changed_registration_and_undeclared_receipts_block_public_claim(setup):
    root, _, _, _ = setup
    artifact, requests = register(setup)
    for index, request in enumerate(requests):
        save_attempt(root, request, f"stage-{index}")
    rogue = deepcopy(requests[0])
    rogue["promotion"]["seed"] = 999
    rogue["protocol"]["seed"] = 999
    save_attempt(root, rogue, "undeclared")
    report = promotion.summarize_promotion(root, artifact["registration_id"])
    assert report["status"] == "BLOCKED" and report["unexpected_attempt_ids"] == ["undeclared"]
    assert not report["public_default_claim"] and report["denominator"] == 8
    path = root / "reports/forge/promotions" / artifact["registration_id"] / "registration.json"
    modified = read_json(path)
    modified["contract"]["seeds"] = [0, 999]
    atomic_json(path, modified)
    with pytest.raises(CapabilityError, match="invalid content hash"):
        promotion.plan_promotion(root, artifact["registration_id"], root / "runs/promotion")


def test_reference_control_failure_prevents_default_even_when_candidate_passes(setup):
    root, path, contract, _ = setup
    artifact, requests = register(setup)
    for index, request in enumerate(requests):
        save_attempt(root, request, f"stage-{index}", score=0. if request["promotion"]["role"] == "control" else 1.)
    report = promotion.summarize_promotion(root, artifact["registration_id"])
    assert report["complete"] and report["counts"] == {"FAIL": 4, "PASS": 4}
    assert report["status"] == "FAIL" and not report["public_default_claim"]


def test_conflicting_compatible_outcomes_are_not_averaged_or_selected(setup):
    root, _, _, _ = setup
    artifact, requests = register(setup)
    for index, request in enumerate(requests):
        save_attempt(root, request, f"stage-{index}")
    save_attempt(root, requests[0], "contradiction", score=0.)
    report = promotion.summarize_promotion(root, artifact["registration_id"])
    assert report["counts"] == {"INVALID": 2, "PASS": 6}
    assert not report["public_default_claim"]
    assert all(len(cell["outcomes"]) == 2 for cell in report["cells"] if cell["status"] == "INVALID")


def test_duplicate_reference_formulation_cannot_manufacture_a_control_denominator(setup):
    root, path, contract, candidate = setup
    idea = read_json(root / "configs/forge/ideas/finished.json")
    idea["id"] = "control"
    atomic_json(root / "configs/forge/ideas/control.json", idea)
    contract["controls"][0]["candidate_revision"] = candidate["candidate_revision"]
    atomic_json(path, contract)
    with pytest.raises(CapabilityError, match="distinct frozen formulations"):
        promotion.register(root, "finished", path)


def test_negative_control_expectations_are_frozen_before_execution(setup):
    root, path, contract, _ = setup
    contract["controls"][0]["expected_statuses"] = {"cheap": "FAIL", "quality": "FAIL"}
    atomic_json(path, contract)
    artifact, requests = register(setup)
    for index, request in enumerate(requests):
        save_attempt(root, request, f"stage-{index}", score=0. if request["promotion"]["role"] == "control" else 1.)
    report = promotion.summarize_promotion(root, artifact["registration_id"])
    assert report["public_default_claim"] and report["accepted_cells"] == report["denominator"] == 8


def test_validated_queue_submission_allows_verified_snapshot_location_only(setup):
    import shutil
    root, _, _, _ = setup
    artifact, _ = register(setup)
    requests = promotion.plan_promotion(root, artifact["registration_id"], root / "runs/promotion", freeze_source=True)
    request = requests[0]
    assert promotion.validate_submission(root, request) == request["promotion_campaign"]
    alternate = root / "verified-snapshot-copy"
    shutil.copytree(request["source"]["snapshot_path"], alternate)
    moved = deepcopy(request)
    moved["source"]["snapshot_path"] = str(alternate)
    assert promotion.validate_submission(root, moved) == request["promotion_campaign"]
    (alternate / "particlegan/mechanism.py").write_text("tampered = True\n")
    with pytest.raises(CapabilityError, match="snapshot"):
        promotion.validate_submission(root, moved)


@pytest.mark.parametrize("field", ["seed", "scoring", "budget", "recipe", "source", "registration", "missing_snapshot"])
def test_promotion_submission_rejects_forged_mutations_before_queue_write(setup, field):
    root, _, _, _ = setup
    artifact, _ = register(setup)
    request = promotion.plan_promotion(root, artifact["registration_id"], root / "runs/promotion", freeze_source=True)[0]
    if field == "seed":
        request["promotion"]["seed"] = 999
        request["protocol"]["seed"] = 999
    elif field == "scoring":
        request["tasks"]["cheap"]["evaluation"]["scoring_weights"] = "ema"
    elif field == "budget":
        request["jobs"][0]["budget_seconds"] *= 2
    elif field == "recipe":
        request["candidate"]["recipe_overrides"]["lr"] = .01
    elif field == "source":
        request["source"]["digest"] = "forged"
    elif field == "registration":
        request["promotion"]["registration_sha256"] = "forged"
    else:
        request["source"].pop("snapshot_path")
    from experiments.forge.queue import Queue
    queue_root = root / "queue-test"
    q = Queue(queue_root, report_root=root / "reports/forge")
    with pytest.raises((CapabilityError, ValueError)):
        q.submit(request, request["promotion_campaign"])
    assert not (queue_root / "queue/state.json").exists()


def test_promotion_queue_requires_repository_binding_and_exact_frozen_campaign(setup):
    from experiments.forge.queue import Queue
    root, _, _, _ = setup
    artifact, _ = register(setup)
    request = promotion.plan_promotion(root, artifact["registration_id"], root / "runs/promotion", freeze_source=True)[0]
    with pytest.raises(ValueError, match="report_root"):
        Queue(root / "queue-unbound").submit(request, request["promotion_campaign"])
    campaign = {**request["promotion_campaign"], "budget_seconds": 800}
    with pytest.raises(ValueError, match="campaign"):
        Queue(root / "queue-bound", report_root=root / "reports/forge").submit(request, campaign)
    q = Queue(root / "queue-valid", report_root=root / "reports/forge")
    entry = q.submit(request, request["promotion_campaign"])
    assert entry["status"] == "queued"


def test_certified_infrastructure_retry_keeps_cost_and_history_without_poisoning_stage(setup):
    root, _, _, _ = setup
    artifact, requests = register(setup)
    for index, request in enumerate(requests):
        save_attempt(root, request, f"stage-{index}")
    old_dir = save_attempt(root, requests[0], "infra-old", stamp="INCOMPLETE")
    old = read_json(old_dir / "result.json")
    old["raw"] = {"attempt_status": "timeout"}
    for row in old["task_results"]:
        row["raw_status"] = "timeout"
    atomic_json(old_dir / "result.json", old)
    cert = read_json(old_dir / "evidence.json")
    cert["result_hash"] = stable_hash(old)
    atomic_json(old_dir / "evidence.json", cert)
    repaired_dir = root / "reports/forge/attempts/stage-0"
    link = {"attempt_id": "infra-old", "result_hash": stable_hash(old), "reason": "repair fixture environment",
            "authorized_at": "2026-09-28T00:00:00+00:00"}
    resolved = read_json(repaired_dir / "request.json")
    resolved["retry_of"] = link
    atomic_json(repaired_dir / "request.json", resolved)
    result = read_json(repaired_dir / "result.json")
    result["retry_of"] = link
    atomic_json(repaired_dir / "result.json", result)
    cert = read_json(repaired_dir / "evidence.json")
    cert["result_hash"] = stable_hash(result)
    atomic_json(repaired_dir / "evidence.json", cert)
    report = promotion.summarize_promotion(root, artifact["registration_id"])
    assert report["public_default_claim"] and report["cost"]["wall_seconds"] == 5
    cells = [c for c in report["cells"] if c["subject"] == requests[0]["promotion"]["subject"]
             and c["seed"] == requests[0]["protocol"]["seed"]]
    assert all(len(c["outcomes"]) == 2 for c in cells)
    assert all(next(o for o in c["outcomes"] if o["attempt_id"] == "infra-old")["superseded_by"] == "stage-0" for c in cells)
