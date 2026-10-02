"""Tier inventories follow live declarations without launching experiment work."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.tier_report import build_report, render_markdown
from experiments.forge.views import load_tasks, task_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def task(name):
    return {
        "schema_version": 1, "id": name, "adapter": "inventory_fixture",
        "execution": {"initializer": "deterministic_orthogonal", "steps": 24, "prior": {"kind": "mog", "sigma": .025,
                                               "standardize": False, "learnable": True}},
        "evaluation": {"kind": "inventory_fixture"},
        "resources": {"timeout_seconds": 60}, "dependencies": [],
        "requires_capabilities": [],
    }


def assignment(name, tier, importance="required", order=0):
    return {"task": name, "qualification_tier": tier, "importance": importance,
            "order": order}


def write_view(root, name, assignments):
    view = {
        "schema_version": 1, "id": name, "revision": 1, "goal": name,
        "eligibility": {}, "calibration": {"status": "provisional"},
        "assignments": assignments,
    }
    atomic_json(root / f"configs/forge/views/{name}.json", view)


@pytest.fixture
def inventory(tmp_path):
    for name in ("smoke", "quality", "other_quality", "endurance", "diagnostic", "unused"):
        atomic_json(tmp_path / f"configs/forge/tasks/{name}.json", task(name))
    # Declaration order is deliberately different from scheduling order.
    write_view(tmp_path, "stability", [
        assignment("diagnostic", 2, "diagnostic", 2),
        assignment("endurance", 3),
        assignment("other_quality", 2, order=1),
        assignment("quality", 2),
        assignment("smoke", 1),
    ])
    write_view(tmp_path, "short", [assignment("smoke", 1), assignment("quality", 2)])
    return tmp_path


def view(report, name):
    return next(item for item in report["views"] if item["id"] == name)


def tier(report, name, number):
    return next(item for item in view(report, name)["tiers"]
                if item["qualification_tier"] == number)


def test_inventory_keeps_diagnostics_empty_tiers_and_unassigned_tasks(inventory):
    result = build_report(inventory)
    assert result["schema_version"] == 1
    assert result["task_count"] == 6
    assert result["assigned_task_count"] == 5
    assert result["view_count"] == 2
    assert [item["id"] for item in result["views"]] == ["short", "stability"]
    assert [item["id"] for item in result["unassigned_tasks"]] == ["unused"]
    assert view(result, "stability")["source"] == "configs/forge/views/stability.json"
    assert view(result, "stability")["calibration"] == {"status": "provisional"}
    second = tier(result, "stability", 2)
    assert second["counts"] == {"required": 2, "ranking": 0, "diagnostic": 1}
    assert [item["id"] for item in second["tasks"]] == ["quality", "other_quality", "diagnostic"]
    diagnostic = second["tasks"][-1]
    assert diagnostic["importance"] == "diagnostic" and diagnostic["order"] == 2
    assert diagnostic["source"] == "configs/forge/tasks/diagnostic.json"
    assert diagnostic["adapter"] == diagnostic["evaluation_kind"] == "inventory_fixture"
    assert diagnostic["steps"] == 24 and diagnostic["timeout_seconds"] == 60
    assert diagnostic["dependencies"] == []
    empty = tier(result, "short", 3)
    assert empty["tasks"] == []
    assert empty["counts"] == {"required": 0, "ranking": 0, "diagnostic": 0}


def test_new_tasks_and_views_are_discovered_without_an_inventory_list(inventory):
    first = build_report(inventory)
    atomic_json(inventory / "configs/forge/tasks/new_host.json", task("new_host"))
    added = build_report(inventory)
    assert added["task_count"] == first["task_count"] + 1
    assert [item["id"] for item in added["unassigned_tasks"]] == ["new_host", "unused"]
    write_view(inventory, "new_goal", [assignment("new_host", 1)])
    assigned = build_report(inventory)
    assert assigned["view_count"] == 3
    assert assigned["assigned_task_count"] == first["assigned_task_count"] + 1
    assert tier(assigned, "new_goal", 1)["tasks"][0]["id"] == "new_host"
    assert [item["id"] for item in assigned["unassigned_tasks"]] == ["unused"]


def test_retiering_updates_report_without_changing_task_identity(inventory):
    before = build_report(inventory)
    tasks = load_tasks(inventory)
    fingerprints = {name: task_fingerprint(value) for name, value in tasks.items()}
    path = inventory / "configs/forge/views/stability.json"
    definition = read_json(path)
    definition["revision"] += 1
    next(item for item in definition["assignments"] if item["task"] == "quality").update(
        qualification_tier=1, order=1)
    atomic_json(path, definition)
    after = build_report(inventory)
    assert view(after, "stability")["revision"] == 2
    assert [item["id"] for item in tier(before, "stability", 1)["tasks"]] == ["smoke"]
    assert [item["id"] for item in tier(after, "stability", 1)["tasks"]] == ["smoke", "quality"]
    assert [item["id"] for item in tier(after, "stability", 2)["tasks"]] == ["other_quality", "diagnostic"]
    assert fingerprints == {name: task_fingerprint(value)
                            for name, value in load_tasks(inventory).items()}


def test_view_filter_preserves_global_coverage(inventory):
    result = build_report(inventory, "short")
    assert [item["id"] for item in result["views"]] == ["short"]
    assert result["view_count"] == 2
    assert result["task_count"] == 6 and result["assigned_task_count"] == 5
    assert [item["id"] for item in result["unassigned_tasks"]] == ["unused"]
    # A task absent from the selected view remains assigned in the catalog.
    assert all(item["id"] != "endurance" for item in result["unassigned_tasks"])


def test_ranking_assignments_keep_their_declared_role(inventory):
    path = inventory / "configs/forge/views/stability.json"
    definition = read_json(path)
    next(item for item in definition["assignments"] if item["task"] == "diagnostic").update(
        importance="ranking")
    atomic_json(path, definition)
    second = tier(build_report(inventory), "stability", 2)
    assert second["counts"] == {"required": 2, "ranking": 1, "diagnostic": 0}
    assert second["tasks"][-1]["importance"] == "ranking"


@pytest.mark.parametrize("mutation,match", [
    (lambda v: v["assignments"][0].update(task="missing"), "missing task"),
    (lambda v: v["assignments"].append(deepcopy(v["assignments"][0])), "duplicate assignment"),
    (lambda v: v["assignments"][0].update(qualification_tier=4), "qualification_tier"),
    (lambda v: v["assignments"][0].update(importance="optional"), "importance"),
])
def test_invalid_view_declarations_are_not_silently_omitted(inventory, mutation, match):
    path = inventory / "configs/forge/views/stability.json"
    definition = read_json(path)
    mutation(definition)
    atomic_json(path, definition)
    with pytest.raises(ValueError, match=match):
        build_report(inventory)


def test_inventory_rejects_duplicate_task_ids(inventory):
    atomic_json(inventory / "configs/forge/tasks/duplicate.json", task("smoke"))
    with pytest.raises(ValueError, match="duplicate task"):
        build_report(inventory)


def test_unassigned_task_missing_dependency_fails_with_named_reference(inventory):
    path = inventory / "configs/forge/tasks/unused.json"
    definition = read_json(path)
    definition["dependencies"] = [{"task": "missing_parent", "kind": "gate"}]
    atomic_json(path, definition)
    with pytest.raises(ValueError, match="unused.*missing_parent"):
        build_report(inventory)


def test_calibration_diagnostic_view_preserves_scope_and_qualification_limits(inventory):
    write_view(inventory, "calibration_probe", [assignment("diagnostic", 1, "diagnostic")])
    path = inventory / "configs/forge/views/calibration_probe.json"
    definition = read_json(path)
    definition["evidence_scope"] = "calibration_diagnostic"
    atomic_json(path, definition)
    result = build_report(inventory, "calibration_probe")
    assert view(result, "calibration_probe")["evidence_scope"] == "calibration_diagnostic"
    assert tier(result, "calibration_probe", 1)["counts"] == {
        "required": 0, "ranking": 0, "diagnostic": 1,
    }
    document = render_markdown(result, inventory)
    assert "calibration_diagnostic" in document
    assert "no ordinary qualification" in document.lower()


def test_task_source_links_follow_renamed_declarations(inventory):
    original = inventory / "configs/forge/tasks/smoke.json"
    original.rename(original.with_name("renamed_smoke.json"))
    result = build_report(inventory)
    smoke = tier(result, "stability", 1)["tasks"][0]
    assert smoke["id"] == "smoke"
    assert smoke["source"] == "configs/forge/tasks/renamed_smoke.json"
    document = render_markdown(result, inventory, inventory / "reports/forge/tiers.md")
    assert "../../configs/forge/tasks/renamed_smoke.json" in document
    assert "../../configs/forge/tasks/smoke.json" not in document


def test_selected_view_must_exist_and_have_safe_id(inventory):
    with pytest.raises((ValueError, FileNotFoundError)):
        build_report(inventory, "missing")
    with pytest.raises(ValueError):
        build_report(inventory, "../stability")


def test_regeneration_is_deterministic_and_links_follow_output_location(inventory):
    first = build_report(inventory)
    stdout = render_markdown(first, inventory)
    assert "configs/forge/tasks/smoke.json" in stdout
    output = inventory / "reports/forge/EXPERIMENTS_BY_TIER.md"
    document = render_markdown(first, inventory, output)
    assert "../../configs/forge/tasks/smoke.json" in document
    assert "../../configs/forge/views/stability.json" in document
    assert "diagnostic" in document and "unused" in document
    assert "provisional" in document
    assert first == build_report(inventory)
    assert document == render_markdown(build_report(inventory), inventory, output)
    assert stdout == render_markdown(build_report(inventory), inventory)
    path = inventory / "configs/forge/views/stability.json"
    definition = read_json(path)
    definition["assignments"].reverse()
    atomic_json(path, definition)
    second = build_report(inventory)
    assert first["views"] == second["views"]
    assert first["unassigned_tasks"] == second["unassigned_tasks"]
    assert first["input_digest"] != second["input_digest"]
    # Rendering leaves the caller's structured report untouched.
    copied = deepcopy(first)
    render_markdown(first, inventory, output)
    assert first == copied


def test_cli_prints_and_writes_without_touching_queue(inventory, monkeypatch, capsys):
    from experiments.forge import __main__

    def unexpected_queue(*args, **kwargs):
        pytest.fail("tier inventory must dispatch before queue initialization")

    monkeypatch.setattr(__main__, "queue_location", unexpected_queue)
    snapshot = {str(path.relative_to(inventory)): path.read_bytes()
                for path in inventory.rglob("*.json")}
    arguments = ["--root", str(inventory), "experiments-by-tier"]
    __main__.main(arguments)
    assert capsys.readouterr().out == render_markdown(build_report(inventory), inventory)
    __main__.main(arguments + ["--json"])
    assert json.loads(capsys.readouterr().out) == build_report(inventory)
    output = inventory / "reports/forge/tiers.md"
    __main__.main(arguments + ["--output", "reports/forge/tiers.md"])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["training_launched"] is False
    assert Path(receipt["output"]).name == output.name
    assert output.read_text() == render_markdown(build_report(inventory), inventory, output)
    json_output = inventory / "reports/forge/tiers.json"
    __main__.main(arguments + ["--json", "--view", "short", "--output", str(json_output)])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["training_launched"] is False
    assert read_json(json_output) == build_report(inventory, "short")
    assert not (inventory / "runs").exists()
    assert all((inventory / path).read_bytes() == content for path, content in snapshot.items())


def test_current_catalog_coverage_and_uninterrupted_extension_are_explicit():
    result = build_report(ROOT)
    assigned = {task["id"] for item in result["views"] for tier in item["tiers"]
                for task in tier["tasks"]}
    unassigned = {item["id"] for item in result["unassigned_tasks"]}
    assert assigned.isdisjoint(unassigned)
    assert assigned | unassigned == set(load_tasks(ROOT))
    assert result["task_count"] == len(assigned | unassigned)
    assert result["assigned_task_count"] == len(assigned)
    declared_views = {path.stem for path in (ROOT / "configs/forge/views").glob("*.json")}
    assert {item["id"] for item in result["views"]} == declared_views
    assert result["view_count"] == len(declared_views)
    extension = next(item for item in tier(result, "discriminator_stability", 3)["tasks"]
                     if item["id"] == "ring_extension")
    assert extension["steps"] == extension["max_total_steps"] == 7500
    assert extension["incremental_steps"] is None and extension["extension_steps"] == 300
    assert extension["uninterrupted"] is True
    assert extension["execution_group"] == "ring_endurance"
    assert extension["dependencies"] == [{"task": "ring_hold", "kind": "checkpoint"}]


def test_prior_inventory_includes_unassigned_and_nonsampled_tasks(inventory):
    value = task("unused")
    value["execution"].update(prior={"kind": "particle_cloud", "sigma": 0,
                                     "standardize": False, "learnable": True,
                                     "exception_reason": "Nonsampled parameter control"},
                              prior_applicability="not_sampled")
    atomic_json(inventory / "configs/forge/tasks/unused.json", value)
    result = build_report(inventory)
    assert result["prior_counts"] == {"mog": 5, "particle_cloud": 1}
    assert result["nonsampled_prior_count"] == 1
    smoke = tier(result, "stability", 1)["tasks"][0]
    assert smoke["prior"] == task("smoke")["execution"]["prior"]
    assert smoke["prior_code_path"] == "MoGParticlePrior"
    unused = result["unassigned_tasks"][0]
    assert unused["prior_code_path"] == "ParticlePrior"
    assert unused["prior_applicability"] == "not_sampled"
    document = render_markdown(result, inventory)
    assert "Prior code path" in document
    assert "MoGParticlePrior (sigma=0.025)" in document
    assert "ParticlePrior (sigma=0; not sampled)" in document


def test_filtered_report_links_unassigned_dependencies_outside_selected_view():
    result = build_report(ROOT, "quality_coverage")
    document = render_markdown(result, ROOT, ROOT / "reports/forge/tiers.md")
    assert "../../configs/forge/tasks/grid100_affine_square_named_v1.json" in document
    assert "grid100_affine_square_named_v1_14k" in document


def publish_demo(root, *, verdict="FAIL", execution="COMPLETE", gif="media/demo.gif"):
    base = root / "reports/toy_audit/api_contract"
    case = {"id": "api-smoke", "legacy_ids": ["develop-smoke"],
            "goal": "Reproduce both poles and their widths", "scope": "New stricter API law",
            "default_recipe": "atlas", "default_steps": 24}
    atomic_json(base / "cases.json", {"cases": [case]})
    atomic_json(base / "readout.json", {"cases": [{
        "id": case["id"], "verdict": verdict, "execution_status": execution,
        "completed_updates": 24 if execution == "COMPLETE" else 0,
        "failed_bounds": ["width"], "gif": gif, "source_commit": "a" * 40,
    }]})
    if gif.startswith("media/"):
        (base / gif).parent.mkdir(parents=True, exist_ok=True)
        (base / gif).write_bytes(b"test-media-placeholder")
    return base


def test_artifacts_keep_api_scope_and_results_separate_from_forge(inventory):
    base = publish_demo(inventory)
    atomic_json(base / "runs.json", {"cases": [{
        "id": "api-smoke", "recipe": {"name": "atlas"},
        "runtime": {"device": "cpu"}, "source_identity": "b" * 64,
    }]})
    result = build_report(inventory)
    guide = next(item for item in result["experiment_guides"] if item["id"] == "smoke")
    demo = guide["api_variants"][0]
    assert demo["verdict"] == "FAIL" and demo["failed_bounds"] == ["width"]
    assert demo["scope"] == "New stricter API law" and demo["recipe"] == "atlas"
    assert demo["source_identity"] == "b" * 64 and demo["runtime"] == {"device": "cpu"}
    assert demo["prior"] is None  # A recipe name cannot establish a historical code path.
    assert demo["media_available"] and demo["qualification_input"] is False
    assert guide["forge_results"] == []  # Same question ID supplies no Forge credit.
    document = render_markdown(result, inventory, inventory / "reports/forge/tiers.md")
    assert "../toy_audit/api_contract/media/demo.gif" in document
    assert "#experiment-smoke" in document
    assert "New stricter API law" in document and "COMPLETE / FAIL" in document
    assert "do not qualify a different Forge task" in document


@pytest.mark.parametrize("kind,width,expected", [
    ("mog", 0., "MoGParticlePrior (sigma_rel=0)"),
    ("mog", .025, "MoGParticlePrior (sigma_rel=0.025)"),
    ("particles", .025, "ParticlePrior (sigma=0)"),
])
def test_api_prior_uses_saved_kind_and_preserves_relative_width(inventory, kind, width, expected):
    base = publish_demo(inventory)
    atomic_json(base / "runs.json", {"cases": [{
        "id": "api-smoke", "recipe": {"name": "saved", "prior_kind": kind, "sigma_rel": width},
    }]})
    document = render_markdown(build_report(inventory), inventory)
    row = next(line for line in document.splitlines() if line.startswith("| [api-smoke]"))
    assert expected in row


def test_explicit_host_mapping_discovers_variants_without_name_guessing(inventory):
    base = publish_demo(inventory)
    definition = read_json(base / "cases.json")
    definition["cases"].append({**definition["cases"][0], "id": "second-smoke"})
    atomic_json(base / "cases.json", definition)
    new_task = task("other_architecture")
    new_task["execution"]["host"] = "smoke"
    atomic_json(inventory / "configs/forge/tasks/other_architecture.json", new_task)
    similar_name = task("smoke_unrelated")
    atomic_json(inventory / "configs/forge/tasks/smoke_unrelated.json", similar_name)
    result = build_report(inventory)
    guide = next(item for item in result["experiment_guides"] if item["id"] == "smoke")
    assert [item["id"] for item in guide["tasks"]] == ["other_architecture", "smoke"]
    assert [item["id"] for item in guide["api_variants"]] == ["api-smoke", "second-smoke"]
    assert guide["api_variants"][1]["verdict"] == "UNKNOWN"
    unrelated = next(item for item in result["experiment_guides"] if item["id"] == "smoke_unrelated")
    assert unrelated["api_variants"] == []


def test_new_task_question_updates_without_editing_the_generator(inventory):
    definition = task("smoke")
    definition["description"] = "Measure a revised target with a new falsifiable question."
    atomic_json(inventory / "configs/forge/tasks/smoke.json", definition)
    guide = next(item for item in build_report(inventory)["experiment_guides"] if item["id"] == "smoke")
    assert guide["goal"] == definition["description"]


def test_task_named_supplement_joins_without_rewriting_frozen_campaign(inventory):
    base = publish_demo(inventory)
    frozen = (base / "cases.json").read_bytes()
    path = base / "new-question/publication.json"
    definition = task("new_question")
    definition["research_artifacts"] = {"api_publication": path.relative_to(inventory).as_posix()}
    atomic_json(inventory / "configs/forge/tasks/new_question.json", definition)
    atomic_json(path, {"cases": [{"id": "api-new", "legacy_ids": ["develop-new_question"],
                                 "goal": "Acquire new target", "scope": "Separate standalone cohort"}],
                       "readouts": [{"id": "api-new", "execution_status": "COMPLETE", "verdict": "FAIL",
                                     "completed_updates": 24, "gif": "goal.gif"}],
                       "runs": [{"id": "api-new", "recipe": {"name": "k3p"}}]})
    (path.parent / "goal.gif").write_bytes(b"actual-media-placeholder")
    report = build_report(inventory)
    guide = next(item for item in report["experiment_guides"] if item["id"] == "new_question")
    variant = guide["api_variants"][0]
    assert variant["media_available"] and variant["qualification_input"] is False
    assert variant["receipt_source"] == path.relative_to(inventory).as_posix()
    assert variant["gif"].endswith("new-question/goal.gif")
    assert guide["forge_results"] == []
    assert (base / "cases.json").read_bytes() == frozen
    document = json.loads(path.read_text())
    document["runs"].append({"id": "api-smoke"})
    atomic_json(path, document)
    with pytest.raises(ValueError, match="own variant definitions"):
        build_report(inventory)


def test_actual_error_media_and_missing_gifs_are_visible(inventory):
    base = publish_demo(inventory, execution="ERROR")
    document = render_markdown(build_report(inventory), inventory)
    assert "ERROR / FAIL; 0/24 updates" in document
    (base / "media/demo.gif").unlink()
    document = render_markdown(build_report(inventory), inventory)
    assert "api-smoke (GIF unavailable)" in document
    assert "[api-smoke](" not in document


def test_artifact_inputs_change_without_changing_declaration_digest(inventory):
    first = build_report(inventory)
    publish_demo(inventory)
    second = build_report(inventory)
    assert first["input_digest"] == second["input_digest"]
    assert first["artifact_input_digest"] != second["artifact_input_digest"]
    assert "reports/toy_audit/api_contract/cases.json" in second["artifact_input_hashes"]
    assert second == build_report(inventory)


def test_changed_task_is_labelled_without_regrading_or_borrowing_outcomes(inventory):
    from experiments.forge.views import task_evaluation_fingerprint, task_execution_fingerprint
    tasks = load_tasks(inventory)
    contract = {"execution_sha256": task_execution_fingerprint(tasks["smoke"]),
                "evaluation_sha256": task_evaluation_fingerprint(tasks["smoke"]),
                "timeout_seconds": 60, "prior": deepcopy(tasks["smoke"]["execution"]["prior"])}
    atomic_json(inventory / "reports/forge/technique-inventory.json", {
        "publication_scope": "current_technique_inventory", "task_contracts": {"contract": contract},
        "rows": [{"candidate_id": "saved-winner", "trainer_family": "test-family",
                  "candidate_revision": "a" * 64, "cohort": "b" * 64,
                  "bindings": {"task_contracts": {"smoke": "contract"}, "source_origin_commit": "c" * 40},
                  "runtime_cohort": {"execution_backend": "cpu"},
                  "tasks": [{"task_id": "smoke", "status": "PASS"}]}],
    })
    def recorded(report):
        return next(guide for guide in report["experiment_guides"] if guide["id"] == "smoke")["forge_results"][0]
    assert recorded(build_report(inventory))["declaration_match"] is True
    tasks["smoke"]["execution"]["steps"] = 48
    tasks["smoke"]["execution"]["prior"]["sigma"] = .1
    atomic_json(inventory / "configs/forge/tasks/smoke.json", tasks["smoke"])
    result = build_report(inventory)
    assert recorded(result)["declaration_match"] is False
    assert recorded(result)["status"] == "PASS"  # Historical result is retained unchanged.
    assert recorded(result)["prior"]["sigma"] == .025  # Use the saved contract, not the current task.
    assert "CHANGED; earlier contract" in render_markdown(result, inventory)


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "duplicate artifact ID"),
    ("unbound", "no matching variant definition"),
    ("escape", "invalid published GIF path"),
])
def test_invalid_artifact_bindings_are_not_silently_accepted(inventory, mutation, match):
    base = publish_demo(inventory)
    if mutation == "duplicate":
        value = read_json(base / "cases.json")
        value["cases"].append(deepcopy(value["cases"][0]))
        atomic_json(base / "cases.json", value)
    else:
        value = read_json(base / "readout.json")
        value["cases"][0].update({"id": "unbound"} if mutation == "unbound" else {"gif": "../../escape.gif"})
        atomic_json(base / "readout.json", value)
    with pytest.raises(ValueError, match=match):
        build_report(inventory)


def test_current_report_has_one_guide_per_question_and_preserves_receipt_identity():
    result = build_report(ROOT)
    guides = result["experiment_guides"]
    assert {task["id"] for guide in guides for task in guide["tasks"]} == set(load_tasks(ROOT))
    assert len({guide["id"] for guide in guides}) == len(guides)
    publication = read_json(ROOT / "reports/forge/technique-inventory.json")
    for guide in guides:
        for outcome in guide["forge_results"]:
            saved = next(row for row in publication["rows"] if row["candidate_id"] == outcome["candidate_id"]
                         and row["cohort"] == outcome["cohort"])
            assert outcome["status"] == next(task["status"] for task in saved["tasks"] if task["task_id"] == outcome["task_id"])
            assert outcome["candidate_revision"] == saved["candidate_revision"]
            assert outcome["source_commit"] == saved["bindings"]["source_origin_commit"]
            assert outcome["backend"] == saved["runtime_cohort"]["execution_backend"]
    clock = next(guide for guide in guides if guide["id"] == "clockfree_audit")
    assert clock["api_variants"] == []
    assert "restart" in clock["goal"]


def test_committed_tier_report_matches_current_declarations_and_artifacts():
    from experiments.forge.tier_report import REPORT_PATH
    assert (ROOT / REPORT_PATH).read_text() == render_markdown(build_report(ROOT), ROOT, ROOT / REPORT_PATH), (
        "Regenerate with: python -m experiments.forge experiments-by-tier "
        "--output reports/forge/EXPERIMENTS_BY_TIER.md")
