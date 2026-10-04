"""Metadata-only prospective planning and shared-board cohort boundaries.

No public fixture/model, training, queue, Git, GPU or historical replay is used.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.forge import boundaries, contracts, knowledge, planning, technique_board, views
from experiments.forge.policy_contracts import COHORT, PARENT_TASK_IDS, SELECTION_SOURCE, SUFFIX

ROOT = Path(__file__).resolve().parents[1]
IDEA = "atlas-c6-observed-policy-current-v1"


def read(relative):
    return contracts.read_json(ROOT / relative)


@pytest.fixture(scope="module")
def declaration():
    return read(f"configs/forge/ideas/{IDEA}.json")


@pytest.fixture(scope="module")
def common_view():
    return read("configs/forge/views/discriminator_stability.json")


@pytest.fixture(scope="module")
def planned_request():
    # Compute the real byte manifest without inspect_source's Git metadata call.
    # Planning creates only Recipe/stream metadata; it never creates a model.
    from experiments.forge.sources import source_files
    def inspect(root, extra):
        files = {str(p.relative_to(root)): contracts.file_hash(p) for p in source_files(root, extra)}
        return {"schema_version": 1, "files": files, "digest": contracts.stable_hash(files),
                "origin_commit": None}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(planning, "inspect_source", inspect)
        patch.setattr(planning, "compute_profile", lambda backend, model=None: {
            "backend": backend, "threads": 1, "model": "software-control", "deterministic": True})
        return planning.resolve_idea(ROOT, IDEA, execution_backend="cpu", freeze_source=False)


@pytest.mark.parametrize("cohort", [None, False, True, "", "atlas", [], {"id": COHORT}])
def test_invalid_opt_in_is_not_a_default_or_family_name(cohort, declaration):
    bad = deepcopy(declaration)
    bad["task_cohort"] = cohort
    with pytest.raises(ValueError, match="task_cohort"):
        contracts.validate_idea(bad)


def test_cohort_identity_is_additive_and_old_formula_is_unchanged(declaration):
    old = read("configs/forge/ideas/atlas.json")
    old.update(resolved_recipe={"lr": .0053125}, prior={"kind": "particle_cloud"})
    original_fields = ("recipe_preset", "recipe_overrides", "prior", "extensions", "requires_capabilities",
                       "api_changes", "implementation", "initializer", "claim_contract")
    fields = {key: old.get(key) for key in original_fields}
    fields.update(resolved_recipe=old["resolved_recipe"], prior=old["prior"], api_version=old["api_version"])
    expected = contracts.stable_hash({"source": "frozen", "formulation": fields})
    assert planning.candidate_revision_for("frozen", old) == expected
    current = {**old, "task_cohort": COHORT}
    assert planning.candidate_revision_for("frozen", current) != expected
    renamed = {**current, "id": "another-label", "hypothesis": "Different explanation"}
    assert planning.candidate_revision_for("frozen", renamed) == planning.candidate_revision_for("frozen", current)
    inherited = deepcopy(declaration)
    inherited.pop("task_cohort")
    contracts.validate_idea(inherited)  # The opt-in remains optional for old declarations.


def test_new_declaration_keeps_one_shared_pair_and_honest_prior_scope(declaration):
    contracts.validate_idea(declaration)
    assert declaration["schema_version"] == 2 and declaration["parent"] == "atlas"
    assert declaration["recipe_overrides"] == {"lr": .0053125, "prior_lr_mult": 1.5}
    assert declaration["task_cohort"] == COHORT
    decision = declaration["decision_contract"]
    assert decision["scope"]["through_tier"] == 1 and decision["scope"]["max_rounds"] == 1
    assert decision["control"]["candidate_id"] == "ka2"
    assert decision["control"]["task_map"] == {name + SUFFIX: name for name in PARENT_TASK_IDS[:5]}
    assert decision["scope"]["task_ids"] == [name + SUFFIX for name in PARENT_TASK_IDS[:5]]
    assert decision["scope"]["candidate_budget_seconds"] == decision["scope"]["campaign_budget_seconds"] == 44100
    # The saved record is motivation only and is selected by its actual fields.
    evidence = decision["prior_evidence"][0]
    from experiments.forge.decision_contracts import _evidence
    _evidence(ROOT, evidence)
    assert evidence["use"] == "motivation_only"
    assert evidence["identity"]["original_passes"] == 2 and evidence["identity"]["status"] == "INCOMPLETE"
    scope = declaration["qualification_scope"]
    assert scope["current_protocol_seed"] == 0 and scope["reference_protocol_seed"] == 24002
    assert scope["evidence_reuse"] is scope["shipping_default"] is scope["speed_comparison"] is False


def test_resolved_request_retains_all26_actual_tasks_and_five_initial_slots(planned_request, common_view):
    assert common_view["revision"] == 3 and planned_request["view"]["revision"] == 4
    assert planned_request["view"]["parent_view_fingerprint"] == views.view_fingerprint(common_view)
    assert set(planned_request["tasks"]) == {name + SUFFIX for name in PARENT_TASK_IDS}
    assert planned_request["protocol"]["seed"] == 0
    assert len(planned_request["jobs"]) == 25
    assert planning.plan_summary(planned_request)["worst_case_seconds"] == 2100
    assert [sum(item["qualification_tier"] == tier for item in planned_request["view"]["assignments"])
            for tier in (1, 2, 3)] == [5, 19, 2]
    assert all(job["science"]["seed"] == 0 for job in planned_request["jobs"])
    assert planned_request["preflight_blockers"]  # Unfrozen draft never becomes a runnable admission.
    assert not any("TODO" in planned_request["candidate"][key] for key in ("hypothesis", "mechanism_rationale"))


def test_full_horizon_and_grouped_allowance_do_not_expand_paid_cap(planned_request):
    full = deepcopy(planned_request)
    full["through_tier"] = 3
    assert planning.plan_summary(full)["worst_case_seconds"] == 45300
    campaign = read(f"configs/forge/campaigns/{IDEA}.json")
    assert campaign["budget_seconds"] == campaign["candidate_budget_seconds"] == 44100
    assert campaign["accept_shared_cost_transfer"] is False
    assert planned_request["tasks"]["grid100" + SUFFIX]["execution"]["steps"] == 7000
    hold, extension = (planned_request["tasks"][name + SUFFIX] for name in ("ring_hold", "ring_extension"))
    assert hold["execution"]["steps"] == extension["execution"]["steps"] == 7500
    assert extension["execution"]["extension_steps"] == 300
    assert hold["execution"]["execution_group"] == extension["execution"]["execution_group"]


def test_full_next_slot_required_without_creating_queue(planned_request):
    from experiments.forge.queue import Queue
    planned = {**planned_request, "campaign_id": IDEA}
    campaign = read(f"configs/forge/campaigns/{IDEA}.json")
    state = {"campaigns": {IDEA: {"definition": campaign, "spent_seconds": 42600., "reserved_seconds": 0.}},
             "charges": [], "jobs": {}}
    # This method reads supplied state only; no Queue instance/files/locks exist.
    admitted, reason = Queue._available_budget(None, state, planned, 1800)
    assert admitted is False and "full next task" in reason
    assert Queue._available_budget(None, state, planned, 1500) == (True, None)
    assert state["campaigns"][IDEA]["spent_seconds"] == 42600.


def test_complete_declaration_and_runtime_source_closure_is_frozen(planned_request):
    files = planned_request["source"]["files"]
    assert SELECTION_SOURCE in files
    assert "configs/forge/views/discriminator_stability.json" in files
    for name, task in planned_request["tasks"].items():
        parent = task["policy_parent"]["id"]
        assert files[f"configs/forge/tasks/{parent}.json"] == task["policy_parent"]["task_sha256"]
        relative = f"configs/forge/task-variants/{COHORT}/{name}.json"
        assert files[relative] == contracts.file_hash(ROOT / relative)
        for path, sha in task["execution"]["policy_contract"]["sources"].items():
            assert files[path] == sha


def test_changed_variant_or_delegated_source_fails_snapshot(tmp_path, planned_request):
    from experiments.forge.sources import verify_snapshot
    for path in ["configs/forge/task-variants/" + COHORT + "/two_pole" + SUFFIX + ".json",
                 "experiments/forge/policy_contracts.py"]:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / path).read_bytes())
    files = {str(p.relative_to(tmp_path)): contracts.file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    manifest = {"digest": contracts.stable_hash(files), "files": files}
    contracts.atomic_json(tmp_path / "forge-source.json", manifest)
    verify_snapshot(tmp_path, manifest)
    for path in files:
        original = (tmp_path / path).read_bytes()
        (tmp_path / path).write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="changed|corrupt|mismatch"):
            verify_snapshot(tmp_path, manifest)
        (tmp_path / path).write_bytes(original)


def test_dependencies_and_group_prerequisites_cannot_borrow_original_ids(planned_request):
    jobs = {name: job for job in planned_request["jobs"] for name in job["task_ids"]}
    hold, extension = (planned_request["tasks"][name + SUFFIX] for name in ("ring_hold", "ring_extension"))
    assert hold["dependencies"] == [{"task": "mode_hold" + SUFFIX, "kind": "gate"}]
    assert extension["dependencies"] == [{"task": "ring_hold" + SUFFIX, "kind": "checkpoint"}]
    assert extension["execution"]["continuation_of"] == "ring_hold" + SUFFIX
    group = jobs["ring_hold" + SUFFIX]
    assert group is jobs["ring_extension" + SUFFIX]
    assert set(group["science"]["prerequisites"]) == {"mode_hold" + SUFFIX}
    assert group["science"]["prerequisites"]["mode_hold" + SUFFIX] == jobs["mode_hold" + SUFFIX]["compatibility_key"]
    for name, task in planned_request["tasks"].items():
        original = read(f"configs/forge/tasks/{task['policy_parent']['id']}.json")
        assert views.task_execution_fingerprint(task) != views.task_execution_fingerprint(original)
        assert views.task_evaluation_fingerprint(task) != views.task_evaluation_fingerprint(original)


@pytest.mark.parametrize("name", ["vector_two_broad", "img_intensity2", "grid100"])
def test_c6_actual_task_overrides_are_owned_not_inactive_host_constants(name, planned_request):
    task = planned_request["tasks"][name + SUFFIX]
    receipt = task["field_ownership"]
    overrides = task["execution"]["policy_recipe_overrides"]
    for field, value in overrides.items():
        entry = receipt["recipe_fields"][field]
        assert entry["owner"] == "task" and entry["value"] == value and entry["status"] == "effective"
        assert entry["source"] == f"task.execution.policy_recipe_overrides.{field}"
        assert entry["provenance"]["reference_source_commit"] == "8021a1c50c4aff90ddea5010d368cffdc857b2f6"
        assert entry["provenance"]["evidence_reuse"] is False
    assert receipt["recipe_fields"]["lr"]["owner"] == "hyperparameter"
    assert receipt["recipe_fields"]["lr"]["value"] == .0053125
    assert receipt["recipe_fields"]["prior_lr_mult"]["value"] == 1.5
    assert receipt["task_contract"]["task_cohort"]["value"] == COHORT
    assert receipt["recipe_fields"]["total_steps"]["value"] is None
    assert not {"d_lr_mult", "betas", "prior_reg"} & boundaries.TASK_RECIPE_FIELDS


def test_mismatched_actual_task_override_is_rejected(planned_request):
    from experiments.forge.api import task_formulation_context
    task = planned_request["tasks"]["vector_two_broad" + SUFFIX]
    context = task_formulation_context(planned_request["candidate"], task, planned_request["protocol"], root=ROOT)
    recipe = asdict(context.recipe)
    recipe["d_lr_mult"] = 1.
    with pytest.raises(ValueError, match="task-owned policy adaptation"):
        boundaries.ownership_receipt(planned_request["candidate"], task, recipe, planned_request["protocol"])
    parent = read("configs/forge/tasks/vector_two_broad.json")
    assert "d_lr_mult" not in boundaries.task_owned_recipe_fields(parent)
    assert "d_lr_mult" in boundaries.task_owned_recipe_fields(task)


def policy_row(planned_request, common_view):
    statuses = ["PASS", "PASS", "BLOCKED", "FAIL", "NOT_RUN"] + ["NOT_RUN"] * 21
    tasks = [{"task_id": item["task"], "status": status} for item, status in
             zip(planned_request["view"]["assignments"], statuses)]
    row = {"candidate_id": IDEA, "candidate_revision": planned_request["candidate_revision"],
           "runtime_cohort": {"execution_backend": "cpu"}, "cohort": "software-only",
           "evidence_scope": "current", "status": "FAIL", "qualified_tier": 0,
           "qualification": {"tasks": tasks, "view_revision": 4,
                             "policy_fingerprint": views.view_fingerprint(planned_request["view"])},
           "scientific_bindings": technique_board.request_bindings(planned_request), "cost": {}}
    row.update(technique_board.policy_row_metadata(planned_request))
    return row


def board(row, common_view):
    archive = {"candidate_id": "atlas", "evidence_scope": "historical", "counts": {"PASS": 19},
               "qualification_reuse": False, "cost": {}}
    return {"view": common_view["id"], "view_revision": 3,
            "policy_fingerprint": views.view_fingerprint(common_view), "current_rows": [row],
            "rows": [row, archive], "conflicts": []}


def test_shared_board_projects_complete_slots_but_keeps_real_task_ids(planned_request, common_view):
    source = board(policy_row(planned_request, common_view), common_view)
    before = deepcopy(source)
    result = technique_board.reduce_board(source, common_view)
    assert source == before
    assert result["view_revision"] == 3 and result["tier_requirements"]["1"] == list(PARENT_TASK_IDS[:5])
    row = result["rows"][0]
    assert row["qualification_view"]["revision"] == 4
    assert row["tiers"] == {"1": {"passed": 2, "total": 5, "counts": {"PASS": 2, "FAIL": 1, "BLOCKED": 1, "UNKNOWN": 1}},
                            "2": {"passed": 0, "total": 19, "counts": {"UNKNOWN": 19}},
                            "3": {"passed": 0, "total": 2, "counts": {"UNKNOWN": 2}}}
    assert {item["task_id"] for item in row["tasks"]} == {name + SUFFIX for name in PARENT_TASK_IDS}
    assert set(row["bindings"]["task_contracts"]) == set(planned_request["tasks"])
    assert result["archived_rows"]["historical"][0]["qualified_tier"] is None
    assert result["archived_rows"]["historical"][0]["qualification_reuse"] is False


@pytest.mark.parametrize("change", ["missing_view", "duplicate_slot", "renamed_science", "missing_grade",
                                  "original_grade", "missing_binding", "parent_hash", "wrong_revision", "retier"])
def test_policy_row_metadata_tampering_cannot_fill_parent_cells(change, planned_request, common_view):
    row = policy_row(planned_request, common_view)
    first = PARENT_TASK_IDS[0] + SUFFIX
    if change == "missing_view":
        row.pop("qualification_view")
    elif change == "duplicate_slot":
        row["task_slot_map"][PARENT_TASK_IDS[1] + SUFFIX] = PARENT_TASK_IDS[0]
    elif change == "renamed_science":
        row["qualification_view"]["assignments"][0]["task"] = PARENT_TASK_IDS[0]
    elif change == "missing_grade":
        row["qualification"]["tasks"].pop()
    elif change == "original_grade":
        row["qualification"]["tasks"][0]["task_id"] = PARENT_TASK_IDS[0]
    elif change == "missing_binding":
        row["scientific_bindings"]["tasks"].pop(first)
    elif change == "parent_hash":
        row["scientific_bindings"]["tasks"][first]["policy_parent"]["task_sha256"] = "unknown"
    elif change == "wrong_revision":
        row["qualification_view"]["revision"] = 3
    elif change == "retier":
        row["qualification_view"]["assignments"][0]["qualification_tier"] = 2
    with pytest.raises(ValueError, match="policy row"):
        technique_board.reduce_board(board(row, common_view), common_view)


def test_bindings_reject_false_actual_ids_and_parent_collisions(planned_request):
    modified = deepcopy(planned_request)
    first, second = (name + SUFFIX for name in PARENT_TASK_IDS[:2])
    modified["tasks"][first]["policy_parent"]["id"] = PARENT_TASK_IDS[1]
    with pytest.raises(ValueError, match="parent"):
        technique_board.request_bindings(modified)
    assert first != second


def test_board_qualifies_actual_policy_view_and_never_historical_parent_pass(tmp_path, monkeypatch, planned_request, common_view):
    planned = deepcopy(planned_request)
    planned["preflight_blockers"] = []  # Synthetic board metadata, not an admission.
    contracts.atomic_json(tmp_path / f"configs/forge/ideas/{IDEA}.json", planned["candidate"])
    contracts.atomic_json(tmp_path / "configs/forge/views/discriminator_stability.json", common_view)
    for name in PARENT_TASK_IDS:
        contracts.atomic_json(tmp_path / f"configs/forge/tasks/{name}.json", read(f"configs/forge/tasks/{name}.json"))
    contracts.atomic_json(tmp_path / "reports/forge/records/historical.json", {
        "schema_version": 1, "record_id": "old-atlas", "candidate_id": IDEA,
        "candidate_revision": planned["candidate_revision"], "evidence_scope": "historical",
        "task_results": [{"task_id": name, "gate_status": "PASS"} for name in PARENT_TASK_IDS]})
    monkeypatch.setattr(knowledge, "_current_request", lambda *args: deepcopy(planned))
    observed_views = []
    original_qualify = views.qualify
    def qualify(actual, tasks, results, **kwargs):
        observed_views.append(deepcopy(actual))
        return original_qualify(actual, tasks, results, **kwargs)
    monkeypatch.setattr(views, "qualify", qualify)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    # compile_memory stores this default board result without projecting away
    # fields; policy hashes must survive even without the verbose-detail flag.
    result = knowledge.board(tmp_path, common_view["id"])
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after and not (tmp_path / "runs").exists()
    assert len(result["current_rows"]) == 1
    row = result["current_rows"][0]
    assert all(v["revision"] == 4 for v in observed_views)
    assert row["qualified_tier"] == 0 and row["qualification"]["required_passed"] == 0
    assert set(row["qualification"]["task_statuses"]) == set(planned_request["tasks"])
    assert row["qualification_view"]["revision"] == 4
    assert row["task_slot_map"] == {name + SUFFIX: name for name in PARENT_TASK_IDS}
    assert row["scientific_bindings"]["qualification_view_sha256"] == contracts.stable_hash(planned["view"])
    assert row["scientific_bindings"]["task_slot_map"] == row["task_slot_map"]
    assert set(row["scientific_bindings"]["tasks"]) == set(planned["tasks"])
    assert result["historical_rows"][0]["qualified_tier"] is None
    compact = technique_board.reduce_board(result, common_view)
    assert [compact["rows"][0]["tiers"][str(t)]["total"] for t in (1, 2, 3)] == [5, 19, 2]


def test_current_request_preserves_declared_policy_stage_and_legacy_through3(monkeypatch, declaration):
    calls = []
    monkeypatch.setattr(planning, "load_idea", lambda root, name: deepcopy(
        declaration if name == IDEA else read("configs/forge/ideas/atlas.json")))
    monkeypatch.setattr(planning, "resolve_idea", lambda *args, **kwargs: calls.append(kwargs) or kwargs)
    knowledge._current_request(ROOT, IDEA, "discriminator_stability", "cpu", None)
    knowledge._current_request(ROOT, "atlas", "discriminator_stability", "cpu", None)
    assert [call["through_tier"] for call in calls] == [1, 3]
    assert all(call["freeze_source"] is False for call in calls)


@pytest.mark.parametrize("progress", ["parent_pass", "actual_fail", "actual_blocked", "first_four", "all_five"])
def test_prerequisite_order_uses_actual_variant_ids_and_all_five_gates(progress, planned_request):
    from experiments.forge.queue import Queue
    current = deepcopy(planned_request)
    current["through_tier"] = 3
    jobs = {job["compatibility_key"]: {"status": "pending"} for job in current["jobs"]}
    by_task = {name: job["compatibility_key"] for job in current["jobs"] for name in job["task_ids"]}
    if progress == "parent_pass":
        results = [{"task_id": name, "gate_status": "PASS"} for name in PARENT_TASK_IDS[:5]]
        expected = [by_task[PARENT_TASK_IDS[0] + SUFFIX]]
    elif progress in {"actual_fail", "actual_blocked"}:
        status = "FAIL" if progress == "actual_fail" else "BLOCKED"
        results = [{"task_id": PARENT_TASK_IDS[0] + SUFFIX, "gate_status": status}]
        expected = []
    else:
        count = 4 if progress == "first_four" else 5
        results = [{"task_id": name + SUFFIX, "gate_status": "PASS"} for name in PARENT_TASK_IDS[:count]]
        expected = [by_task[PARENT_TASK_IDS[count] + SUFFIX]]
    class ReadOnlyProjection:
        def _results(self, state, submission):
            return results
    state, submission = {"jobs": jobs}, {"request": current}
    before = deepcopy((state, submission, results))
    eligible, reason, running = Queue._eligible(ReadOnlyProjection(), state, submission)
    assert eligible == expected and running is False
    if progress in {"actual_fail", "actual_blocked"}:
        assert results[0]["gate_status"] in reason
    else:
        assert reason is None
    assert (state, submission, results) == before


def test_no_opt_in_never_loads_variant_sources_or_changes_live_view(monkeypatch):
    from experiments.forge import policy_contracts
    def unexpected(*args, **kwargs):
        raise AssertionError("legacy planning must not load policy variants")
    monkeypatch.setattr(policy_contracts, "load_policy_variants", unexpected)
    captured = []
    def source(root, extra):
        captured.extend(extra)
        return {"files": {}, "digest": contracts.stable_hash({}), "origin_commit": None}
    monkeypatch.setattr(planning, "inspect_source", source)
    monkeypatch.setattr(planning, "compute_profile", lambda backend, model=None: {"backend": backend, "threads": 1})
    legacy = planning.resolve_idea(ROOT, "atlas", execution_backend="cpu")
    assert legacy["view"]["revision"] == 3
    assert set(legacy["tasks"]) == set(PARENT_TASK_IDS)
    assert not any("task-variants" in path for path in captured)
    assert "configs/forge/views/discriminator_stability.json" not in captured
    assert "task_cohort" not in legacy["candidate"]
    assert all(task["preflight_blockers"] for task in legacy["tasks"].values())


def test_same_board_retains_independent_live_and_policy_rows(planned_request, common_view):
    row = policy_row(planned_request, common_view)
    legacy = {"candidate_id": "atlas", "candidate_revision": "legacy-source-cohort",
              "evidence_scope": "current", "qualified_tier": 0, "status": "BLOCKED",
              "runtime_cohort": {"execution_backend": "cpu"}, "cost": {}}
    mixed = board(row, common_view)
    mixed["current_rows"].append(legacy)
    mixed["rows"].append(legacy)
    before = deepcopy(mixed)
    reduced = technique_board.reduce_board(mixed, common_view)
    assert mixed == before
    assert [row["candidate_id"] for row in reduced["rows"]] == [IDEA, "atlas"]
    actual, original = reduced["rows"]
    assert actual["tiers"]["1"]["passed"] == 2
    assert all(cell["passed"] == 0 for cell in original["tiers"].values())
    assert {item["task_id"] for item in original["tasks"]} == set(PARENT_TASK_IDS)
    assert {item["task_id"] for item in actual["tasks"]} == {name + SUFFIX for name in PARENT_TASK_IDS}
