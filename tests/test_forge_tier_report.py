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
        "execution": {"steps": 24}, "evaluation": {"kind": "inventory_fixture"},
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


def test_filtered_report_links_unassigned_dependencies_outside_selected_view():
    result = build_report(ROOT, "quality_coverage")
    document = render_markdown(result, ROOT, ROOT / "reports/forge/tiers.md")
    assert "../../configs/forge/tasks/grid100_affine_square_named_v1.json" in document
    assert "grid100_affine_square_named_v1_14k" in document
