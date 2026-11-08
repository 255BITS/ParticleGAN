"""Held and removed-from-count dispositions become archived view revisions; no training."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from experiments.forge import dispositions
from experiments.forge.__main__ import main
from experiments.forge.contracts import atomic_json, canonical
from experiments.forge.views import load_tasks, load_view, qualify, view_fingerprint


ROOT = Path(__file__).resolve().parents[1]
EXPLANATIONS = "reports/forge/test-dispositions-20261015/README.md"
# Revisions the decided dispositions start from (develop 224958a3).
BASE = {"adaptation": 2, "clockfree_continuous": 3, "discriminator_stability": 9,
        "formulation_comparison": 1, "host_profile_transfer": 4, "quality_coverage": 2}
SIX = tuple(BASE)
RING = ("clockfree_continuous", "discriminator_stability", "formulation_comparison", "host_profile_transfer")
DECIDED = [  # task, disposition, reason, counted_by, {view: tier}
    *[(task, "HOLD", "no_global_recipe", None, {view: 2 for view in SIX}) for task in (
        "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic", "vector_overlap",
        "img_bars4", "img_blobs4", "img_intensity2")],
    *[(task, "HOLD", "unsolved", None, {view: 2 for view in SIX}) for task in ("grid100", "rotated100", "staggered100")],
    ("gaussian1d_stability", "HOLD", "no_global_recipe", None, {"discriminator_stability": 2}),
    ("target_shift_recovery", "HOLD", "unsolved", None, {"adaptation": 3, "clockfree_continuous": 3}),
    *[(task, "HOLD", "held_prerequisite", None, {"clockfree_continuous": 3})
      for task in ("grid100_14k", "rotated100_14k", "staggered100_14k")],
    ("clockfree_audit", "REMOVE", "duplicate", "clockfree_audit_measurement_v1", {"clockfree_continuous": 3}),
    ("ring_hold", "REMOVE", "phase", "ring_extension", {view: 3 for view in RING}),
]


@pytest.fixture
def repo(tmp_path):
    for name in ("tasks", "task-variants", "views", "view-history"):
        shutil.copytree(ROOT / "configs/forge" / name, tmp_path / "configs/forge" / name)
    for view_id, revision in BASE.items():
        archive = tmp_path / dispositions.archive_path(view_id, revision)
        if archive.exists():  # A later revision is live; start this copy from the decided base.
            archive.replace(tmp_path / "configs/forge/views" / f"{view_id}.json")
    return tmp_path


def base_fingerprint(view_id, revision):
    """The decided base revision's policy fingerprint, read from the committed archive or live view."""
    archive = ROOT / dispositions.archive_path(view_id, revision)
    return view_fingerprint(json.loads((archive if archive.is_file() else
                                        ROOT / "configs/forge/views" / f"{view_id}.json").read_text()))


def registry(root, rows=DECIDED, **changes):
    views = {view: {"from_revision": revision, "from_fingerprint": base_fingerprint(view, revision),
                    "to_revision": revision + (2 if view == "discriminator_stability" else 1),
                    "archive": dispositions.archive_path(view, revision),
                    "policy_change_reason": f"Fixture dispositions: {view} holds and removes tests from its count."}
             for view, revision in BASE.items()}
    views["discriminator_stability"]["skipped_revision_note"] = "Revision 10 names an unadopted draft; it is skipped."
    views["tier1_policy_coverage"] = {"from_revision": 1, "from_fingerprint": base_fingerprint("tier1_policy_coverage", 1),
                                      "to_revision": 2,
                                      "archive": dispositions.archive_path("tier1_policy_coverage", 1),
                                      "policy_change_reason": "Fixture: the policy view is revised in a follow-up change.",
                                      "applied": False, "apply_in": "follow-up change"}
    records = []
    for task, kind, reason, counted_by, where in rows:
        record = {"task": task, "scope": "required", "disposition": kind, "views": where, "group": None,
                  "explanation": f"{EXPLANATIONS}#{task}"}
        if kind == "HOLD":
            record["hold"] = {"reason": reason, "summary": f"Fixture: {task} is measured and shown, not counted.",
                              "reentry": f"Count {task} again when one global recipe passes it and every counted test."}
        else:
            record["removal"] = {"reason": reason, "counted_by": counted_by,
                                 "summary": f"Fixture: {task} leaves the count; {counted_by} carries its question."}
        records.append(record)
    records += [
        {"task": "ring_extension", "scope": "required", "disposition": "KEEP", "views": {view: 3 for view in RING},
         "group": None, "explanation": f"{EXPLANATIONS}#ring_extension",
         "keep": {"summary": "Fixture: stays counted as the single ring endurance verdict.",
                  "zero_cause": "A tier-prerequisite zero for one recipe and a real failure for another."}},
        {"task": "two_pole_tier1_policy_selected_cloud_v1", "scope": "policy", "disposition": "HOLD",
         "views": {"tier1_policy_coverage": 1}, "group": None,
         "explanation": f"{EXPLANATIONS}#two_pole_tier1_policy_selected_cloud_v1",
         "hold": {"reason": "no_global_recipe", "summary": "Fixture: a policy question decided but not applied here.",
                  "reentry": "Count again under a new declaration when one policy recipe passes it."},
         "applied": False, "apply_in": "follow-up change"},
        {"task": "17 indexed variants", "scope": "variant", "disposition": "NO_CHANGE", "views": {}, "group": None,
         "explanation": f"{EXPLANATIONS}#indexed-variants", "tasks": ["vector_overlap_published", "img_bars4_residual16"]},
        {"task": "tests/test_forge_views.py::test_example", "scope": "pytest", "disposition": "FIX", "views": {},
         "group": None, "explanation": f"{EXPLANATIONS}#pytest", "in_handoff": True},
    ]
    value = {"schema": dispositions.SCHEMA, "decided_on": "2026-10-15",
             "evidence_snapshot": "224958a315a81317b8943de5c4e4f3e1c1127385", "explanations": EXPLANATIONS,
             "qualification_input": False, "historical_verdicts_regraded": False, "training_launched": False,
             "views": views, "dispositions": records, **changes}
    anchors = sorted({record["explanation"].split("#")[1] for record in records})
    readme = root / EXPLANATIONS
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("# Fixture dispositions\n\n" + "".join(f'<a name="{name}"></a>\n\n## {name}\n\n' for name in anchors))
    path = root / "dispositions.json"
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path, value


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted((root / "configs/forge").rglob("*.json"))
            if p.parent.name in {"views", "view-history"}}


def counted(report, key):
    return sum(sum(item[key]["required"]) for item in report["views"].values())


def test_apply_archives_exact_bytes_first_then_writes_held_and_removed_roles(repo):
    path, _ = registry(repo)
    before = snapshot(repo)
    planned = dispositions.check(repo, path)
    assert snapshot(repo) == before
    assert planned["drift"] and {item["state"] for item in planned["views"].values()} == {"pending"}
    assert {item["archive_action"] for item in planned["views"].values()} == {"create"}
    result = dispositions.apply(repo, path)
    assert not result["drift"] and result["applied"] == sorted(BASE)
    assert len(result["archived"]) == len(result["written"]) == 6
    after = snapshot(repo)
    for view_id, revision in BASE.items():
        archive = dispositions.archive_path(view_id, revision)
        assert after[archive] == before[f"configs/forge/views/{view_id}.json"]
    assert after["configs/forge/views/tier1_policy_coverage.json"] == before["configs/forge/views/tier1_policy_coverage.json"]
    # 152 counted view requirements become 81; 66 held and 5 removed assignments stay visible.
    assert counted(result, "counts_before") == 152 and counted(result, "counts_after") == 81
    assert sum(sum(item["counts_after"]["held"]) for item in result["views"].values()) == 66
    assert sum(sum(item["counts_after"]["removed"]) for item in result["views"].values()) == 5
    clock = load_view(repo, "clockfree_continuous")
    roles = {a["task"]: a for a in clock["assignments"]}
    assert clock["revision"] == 4 and clock["previous_revision"]["archive"] == "configs/forge/view-history/clockfree_continuous-v3.json"
    assert roles["grid100_14k"]["hold"]["blocked_by"] == ["grid100"]
    assert roles["grid100_14k"]["hold"]["reason"] == "held_prerequisite"
    assert roles["target_shift_recovery"]["hold"]["blocked_by"] == []
    assert roles["clockfree_audit"]["importance"] == "diagnostic"
    assert roles["clockfree_audit"]["removal"]["counted_by"] == "clockfree_audit_measurement_v1"
    assert roles["ring_hold"]["removal"]["reason"] == "phase" and roles["ring_extension"]["importance"] == "required"
    stability = load_view(repo, "discriminator_stability")
    assert stability["revision"] == 11 and stability["previous_revision"]["policy_fingerprint"] == (
        "fcb7ac52465e555fc6d88a1d53d92c485ae340216eafaa6fe49b03981c8f175a")
    assert [sum(a["importance"] == "required" and a["qualification_tier"] == tier for a in stability["assignments"])
            for tier in (1, 2, 3)] == [6, 10, 1]
    # With its only Tier 3 test held, adaptation can report HELD but never PASS.
    adaptation = load_view(repo, "adaptation")
    assert qualify(adaptation, load_tasks(repo), [])["held_tiers"] == [3]


def test_second_apply_is_a_noop_and_check_then_reports_no_drift(repo):
    path, _ = registry(repo)
    dispositions.apply(repo, path)
    applied = snapshot(repo)
    again = dispositions.apply(repo, path)
    assert again["archived"] == again["written"] == again["applied"] == []
    assert snapshot(repo) == applied
    report = dispositions.check(repo, path)
    assert not report["drift"] and {item["state"] for item in report["views"].values()} == {"applied"}
    assert report["skipped"] == {"not_applied": ["two_pole_tier1_policy_selected_cloud_v1"],
                                 "views_not_applied": ["tier1_policy_coverage"], "keep": ["ring_extension"],
                                 "no_change": ["17 indexed variants"], "fix": ["tests/test_forge_views.py::test_example"]}


def test_conflicting_archive_is_refused_before_any_write(repo):
    path, _ = registry(repo)
    archive = repo / dispositions.archive_path("quality_coverage", 2)
    archive.write_text('{"different": true}\n')
    before = snapshot(repo)
    report = dispositions.check(repo, path)
    assert report["views"]["quality_coverage"]["state"] == "conflict" and report["drift"]
    with pytest.raises(ValueError, match="nothing was written"):
        dispositions.apply(repo, path)
    assert snapshot(repo) == before
    # An existing archive with the exact live bytes is reused, not rewritten.
    archive.write_bytes((repo / "configs/forge/views/quality_coverage.json").read_bytes())
    assert dispositions.check(repo, path)["views"]["quality_coverage"]["archive_action"] == "reuse"
    assert "configs/forge/view-history/quality_coverage-v2.json" not in dispositions.apply(repo, path)["archived"]


def test_registry_pins_the_base_revision_so_an_unbumped_edit_is_never_archived(repo):
    path, _ = registry(repo)
    live = repo / "configs/forge/views/quality_coverage.json"
    original = live.read_bytes()
    view = json.loads(original)
    # Hand edit without a revision bump: a Tier 2 test moves to Tier 1 in revision 2.
    next(a for a in view["assignments"] if a["task"] == "img_stripes2")["qualification_tier"] = 1
    live.write_text(json.dumps(view, indent=2) + "\n")
    before = snapshot(repo)
    report = dispositions.check(repo, path)
    assert report["drift"] and report["views"]["quality_coverage"]["state"] == "conflict"
    assert "not the registry's base revision" in report["views"]["quality_coverage"]["differences"][0]
    with pytest.raises(ValueError, match="nothing was written"):
        dispositions.apply(repo, path)
    assert snapshot(repo) == before
    # After a correct apply, a replaced archive no longer counts as the base revision.
    live.write_bytes(original)
    dispositions.apply(repo, path)
    archive = repo / dispositions.archive_path("quality_coverage", 2)
    archive.write_text(json.dumps(view, indent=2) + "\n")
    report = dispositions.check(repo, path)
    assert report["views"]["quality_coverage"]["state"] == "drift"
    assert "not the registry's base revision" in report["views"]["quality_coverage"]["differences"][0]


def test_each_file_keeps_its_key_order_and_unchanged_text(repo):
    path, value = registry(repo)
    originals = {view: (repo / f"configs/forge/views/{view}.json").read_text() for view in BASE}
    dispositions.apply(repo, path)
    for view_id, original in originals.items():
        raw = (repo / f"configs/forge/views/{view_id}.json").read_text()
        revised = json.loads(raw)
        style = dispositions.key_style(original)
        assert dispositions.key_style(raw) == style
        new_keys = [key for key in revised if key not in json.loads(original)]
        assert new_keys == ["previous_revision"]
        keys = list(revised)
        if style == "sorted":
            assert keys == sorted(keys)
            notes = [a.get("hold") or a.get("removal") for a in revised["assignments"] if "hold" in a or "removal" in a]
            assert all(list(note) == sorted(note) for note in notes)
            assert all(list(a) == sorted(a) for a in revised["assignments"] if "hold" in a or "removal" in a)
        else:
            assert keys[-1] == "previous_revision"
            held = next(a for a in revised["assignments"] if "hold" in a)
            assert list(held)[-1] == "hold" and list(held["hold"])[:3] == ["schema", "previous_importance", "reason"]
        # Undo the policy edits: every other byte is the original file.
        old = json.loads(original)
        for assignment in revised["assignments"]:
            if assignment.pop("hold", None) or assignment.pop("removal", None):
                assignment["importance"] = "required"
        revised.pop("previous_revision")
        revised.update(revision=old["revision"], policy_change_reason=old["policy_change_reason"])
        assert dispositions.dump_view(revised, original) == original
    assert dispositions.key_style(originals["host_profile_transfer"]) == "sorted"
    assert dispositions.key_style(originals["formulation_comparison"]) == "sorted"
    assert dispositions.key_style(originals["adaptation"]) == "insertion"


def test_check_reports_drift_after_a_hand_edit_and_apply_refuses_it(repo):
    path, _ = registry(repo)
    dispositions.apply(repo, path)
    target = repo / "configs/forge/views/discriminator_stability.json"
    view = json.loads(target.read_text())
    next(a for a in view["assignments"] if a["task"] == "gaussian1d_stability")["hold"]["summary"] = (
        "A summary edited by hand without a new archived revision.")
    target.write_text(json.dumps(view, indent=2) + "\n")
    report = dispositions.check(repo, path)
    assert report["drift"] and report["views"]["discriminator_stability"]["state"] == "drift"
    assert report["views"]["discriminator_stability"]["differences"] == ["assignment gaussian1d_stability"]
    assert report["views"]["quality_coverage"]["state"] == "applied"
    with pytest.raises(ValueError, match="views differ from the registry"):
        dispositions.apply(repo, path)


def test_cli_check_exits_one_on_drift_and_apply_is_idempotent(repo, capsys):
    path, _ = registry(repo)
    command = ["--root", str(repo), "test-dispositions"]
    assert main(command + ["check", "--registry", path.name]) == 1
    assert json.loads(capsys.readouterr().out)["drift"] is True
    assert main(command + ["apply", "--registry", path.name]) == 0
    assert json.loads(capsys.readouterr().out)["training_launched"] is False
    assert main(command + ["check", "--registry", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["drift"] is False
    assert not (repo / "runs").exists()


def edit(value, change):
    value = deepcopy(value)
    change(value)
    return value


def record(value, task):
    return next(r for r in value["dispositions"] if r["task"] == task)


@pytest.mark.parametrize("change,match", [
    (lambda v: v.update(schema="forge_test_dispositions_v0"), "schema"),
    (lambda v: v.update(training_launched=True), "must be false"),
    (lambda v: v.update(extra="field"), "exactly"),
    (lambda v: record(v, "grid100")["hold"].update(reason="too_hard"), "hold reason"),
    (lambda v: record(v, "grid100")["hold"].update(summary="short"), "20-280"),
    (lambda v: record(v, "ring_hold")["removal"].update(counted_by=None), "name counted_by"),
    (lambda v: record(v, "clockfree_audit")["removal"].update(reason="superseded"), "name counted_by"),
    (lambda v: record(v, "ring_extension").pop("keep"), "needs"),
    (lambda v: record(v, "ring_extension").update(applied=False), "needs"),
    (lambda v: record(v, "two_pole_tier1_policy_selected_cloud_v1").pop("apply_in"), "apply_in"),
    (lambda v: record(v, "two_pole_tier1_policy_selected_cloud_v1")["hold"].update(reason="later"), "hold reason"),
    (lambda v: record(v, "grid100").update(explanation="reports/other.md#grid100"), "link into"),
    (lambda v: record(v, "grid100").update(views={"adaptation": 0}), "positive integer tiers"),
    (lambda v: record(v, "grid100").update(task="not_a_task"), "unknown task"),
    (lambda v: v["dispositions"].append(deepcopy(record(v, "grid100"))), "one disposition per test"),
    (lambda v: record(v, "17 indexed variants")["tasks"].append("missing_variant"), "declared tasks"),
    (lambda v: v["views"]["quality_coverage"].update(archive="configs/forge/view-history/quality_coverage.json"), "archive must be"),
    (lambda v: v["views"]["discriminator_stability"].pop("skipped_revision_note"), "skipped_revision_note"),
    (lambda v: v["views"]["quality_coverage"].pop("from_fingerprint"), "needs"),
    (lambda v: v["views"]["quality_coverage"].update(from_fingerprint="abc"), "from_fingerprint"),
    (lambda v: v.update(explanations="https://example.com/README.md"), "repo-relative Markdown"),
    (lambda v: v["views"]["adaptation"].update(applied=False, apply_in="follow-up change"), "without an applied"),
    (lambda v: v["views"].update(k3p_two_pole_horizon=dict(v["views"]["quality_coverage"],
                                                           archive="configs/forge/view-history/k3p_two_pole_horizon-v2.json")),
     "no applied disposition"),
])
def test_registry_shape_is_strict(repo, change, match):
    _, value = registry(repo)
    tasks = load_tasks(repo)
    dispositions.validate_registry(deepcopy(value), tasks)
    with pytest.raises(ValueError, match=match):
        dispositions.validate_registry(edit(value, change), tasks)


@pytest.mark.parametrize("change,match", [
    (lambda v: record(v, "vector_overlap")["views"].pop("quality_coverage"), "every view that counts"),
    (lambda v: record(v, "gaussian1d_stability")["views"].update(discriminator_stability=3), "required Tier 3"),
    (lambda v: record(v, "grid100_14k")["hold"].update(reason="unsolved"), "held_prerequisite"),
    (lambda v: record(v, "clockfree_audit")["removal"].update(counted_by="two_pole"), "duplicate removal"),
    (lambda v: record(v, "ring_extension")["views"].update(adaptation=3), "KEEP names view adaptation"),
    (lambda v: v["views"]["quality_coverage"].update(from_revision=3, to_revision=4,
                                                     archive="configs/forge/view-history/quality_coverage-v3.json"),
     "neither 3 nor 4"),
])
def test_registry_must_agree_with_the_live_views(repo, change, match):
    path, value = registry(repo)
    path.write_text(json.dumps(edit(value, change), indent=2) + "\n")
    before = snapshot(repo)
    with pytest.raises(ValueError, match=match):
        dispositions.apply(repo, path)
    assert snapshot(repo) == before


def test_written_explanation_links_need_explicit_anchors(repo):
    path, _ = registry(repo)
    readme = repo / EXPLANATIONS
    readme.write_text(readme.read_text().replace('<a name="grid100_14k"></a>', ""))
    with pytest.raises(ValueError, match='<a name="grid100_14k"></a>'):
        dispositions.check(repo, path)
    readme.unlink()
    with pytest.raises(ValueError, match="explicit"):
        dispositions.check(repo, path)


def test_validate_finds_every_hold_and_removal_link_whose_anchor_is_missing(repo):
    path, _ = registry(repo)
    dispositions.apply(repo, path)
    views = [load_view(repo, name) for name in SIX]
    assert dispositions.unresolved_explanations(repo, views) == []
    readme = repo / EXPLANATIONS
    readme.write_text(readme.read_text().replace('<a name="ring_hold"></a>', ""))
    problems = dispositions.unresolved_explanations(repo, views)
    assert len(problems) == 4 and all("ring_hold explanation" in problem for problem in problems)
    assert all(f'<a name="ring_hold"></a> anchor in {EXPLANATIONS}' in problem for problem in problems)
    readme.unlink()
    assert len(dispositions.unresolved_explanations(repo, views)) == 71


def test_retier_archives_the_previous_revision_first_and_refuses_conflicts(repo, monkeypatch, capsys):
    monkeypatch.setattr("experiments.forge.knowledge.board", lambda root, view_id: {"view": view_id})
    live = repo / "configs/forge/views/quality_coverage.json"
    original = live.read_bytes()
    command = ["--root", str(repo), "--queue-root", str(repo / "unused-queue"), "retier", "quality_coverage",
               "--task", "vector_two_broad", "--tier", "1", "--reason", "Fixture: move one broad vector test to Tier 1."]
    assert main(command) == 0
    capsys.readouterr()
    assert (repo / "configs/forge/view-history/quality_coverage-v2.json").read_bytes() == original
    revised = json.loads(live.read_text())
    assert revised["revision"] == 3 and list(revised)[-1] == "previous_revision"
    assert revised["previous_revision"]["archive"] == "configs/forge/view-history/quality_coverage-v2.json"
    assert dispositions.key_style(live.read_text()) == "insertion"
    third = live.read_bytes()
    (repo / "configs/forge/view-history/quality_coverage-v3.json").write_text('{"different": true}\n')
    with pytest.raises(ValueError, match="never overwritten"):
        main(command[:-1] + ["Fixture: a second move."])
    assert live.read_bytes() == third
    assert not (repo / "unused-queue").exists()


def test_role_changes_keep_planning_science_identical_apart_from_candidate_revision(repo, monkeypatch):
    from experiments.forge import planning
    path, _ = registry(repo)
    names = ("discriminator_stability", "clockfree_continuous")
    base = {name: load_view(repo, name) for name in names}
    dispositions.apply(repo, path)

    def science(request):
        return {tuple(job["task_ids"]): {key: value for key, value in job["science"].items()
                                         if key != "candidate_revision"} for job in request["jobs"]}

    for name in names:
        revised = load_view(repo, name)
        assert any(a["importance"] == "held" for a in revised["assignments"])
        requests = []
        for view in (base[name], revised):
            monkeypatch.setattr(planning, "load_view", lambda root, view_id, view=view: deepcopy(view))
            requests.append(planning.resolve_idea(ROOT, "k3p", view_id=name, through_tier=3, execution_backend="cpu"))
        before, after = requests
        assert science(before) == science(after)
        assert all('"importance"' not in canonical(job["science"]) for job in after["jobs"])
        assert before["policy_fingerprint"] != after["policy_fingerprint"]
        ring = next(job for job in after["jobs"] if "ring_extension" in job["task_ids"])
        assert ring["task_ids"] == ["ring_hold", "ring_extension"]


COMMITTED = ROOT / "reports/forge/test-dispositions-20261015/dispositions.json"


def test_committed_dispositions_match_their_archived_target_views(repo):
    # The 0.10 policy supersedes these dispositions without changing their history.
    explanation = repo / EXPLANATIONS
    explanation.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / EXPLANATIONS, explanation)
    for name, revision in BASE.items():
        archive = dispositions.archive_path(name, revision)
        shutil.copyfile(ROOT / archive, repo / archive)
    for name, entry in json.loads(COMMITTED.read_text())["views"].items():
        if entry.get("applied", True):
            archive = ROOT / dispositions.archive_path(name, entry["to_revision"])
            shutil.copyfile(archive, repo / "configs/forge/views" / f"{name}.json")
    report = dispositions.check(repo, COMMITTED)
    assert not report["drift"] and set(report["views"]) == set(SIX)
    assert {item["state"] for item in report["views"].values()} == {"applied"}
    assert counted(report, "counts_before") == 152 and counted(report, "counts_after") == 81
    registry = json.loads(COMMITTED.read_text())
    records = {r["task"]: r for r in registry["dispositions"]
               if r["disposition"] in {"HOLD", "REMOVE"} and r.get("applied", True)}
    noted = set()
    for path in sorted((repo / "configs/forge/views").glob("*.json")):
        view = json.loads(path.read_text())
        for assignment in view["assignments"]:
            note = assignment.get("hold") or assignment.get("removal")
            if note is None:
                continue
            # Every note in every declared view comes from one applied record, verbatim.
            record = records[assignment["task"]]
            noted.add((assignment["task"], view["id"]))
            assert record["views"][view["id"]] == assignment["qualification_tier"]
            assert record["disposition"] == ("HOLD" if "hold" in assignment else "REMOVE")
            expected = record["hold"] if "hold" in assignment else record["removal"]
            assert {key: note[key] for key in expected} == expected
            assert note["explanation"] == record["explanation"]
    assert noted == {(task, view_id) for task, record in records.items() for view_id in record["views"]}
    # Decided but not applied here: the policy view still counts its seven questions.
    pending = {r["task"] for r in registry["dispositions"] if r.get("applied") is False}
    policy = load_view(repo, "tier1_policy_coverage")
    assert policy["revision"] == 1 and len(pending) == 7
    assert {a["task"] for a in policy["assignments"] if a["importance"] == "required"} == pending


def test_every_hold_and_removal_explanation_resolves_to_a_readme_anchor():
    registry = json.loads(COMMITTED.read_text())
    links = {record["explanation"] for record in registry["dispositions"]}
    for path in sorted((ROOT / "configs/forge/views").glob("*.json")):
        for assignment in json.loads(path.read_text())["assignments"]:
            note = assignment.get("hold") or assignment.get("removal")
            if note:
                links.add(note["explanation"])
    readme = (ROOT / registry["explanations"]).read_text()
    for link in sorted(links):
        relative, anchor = link.split("#")
        assert relative == registry["explanations"]
        assert f'<a name="{anchor}"></a>' in readme, link
