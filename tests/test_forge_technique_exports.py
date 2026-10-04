"""Publication projections preserve identity without committing execution traces."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

from experiments.forge.contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash
from experiments.forge.sources import inspect_source

SCRIPT = Path(__file__).resolve().parents[1] / "reports/forge/regenerate_technique_inventory.py"
spec = importlib.util.spec_from_file_location("forge_technique_publication", SCRIPT)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


@pytest.fixture
def receipt(tmp_path):
    directory = tmp_path / "reports/forge/attempts/attempt-one"
    source = {"files": {"particlegan/fixture.py": "a" * 64}, "origin_commit": "commit-one"}
    source["digest"] = stable_hash(source["files"])
    request = {"candidate": {"id": "new-technique"}, "candidate_revision": "revision-one", "source": source,
               "runtime": {"python": "3.12"}, "request_id": "request-one", "campaign_id": "inventory",
               "jobs": [{"task_id": "toy", "compatibility_key": "key-one"}]}
    row = {"task_id": "toy", "compatibility_key": "key-one", "gate_status": "FAIL", "raw_status": "completed",
           "reason": "terminal failure", "metrics": {"score": .25, "per_update": [1] * 800},
           "cost": {"wall_seconds": 12.5}, "reasons": ["terminal failure"],
           "evidence": {"sampling_law": "public_prior_without_output_noise", "eval_output_noise": "clean",
                        "sampling_contract_version": 1, "observations": [{"score": .25}] * 800},
           "raw": {"trace": [1] * 800}, "applied": {"noise": {"trace": [1] * 800}},
           "evaluator_result": {"status": "FAIL", "passed": False,
                                "convergence": {"observations": 800, "passing_suffix": 0, "states": [1] * 800},
                                "metrics": [{"metric": "score", "value": .25, "threshold": .5, "status": "FAIL"}],
                                "terminal_checks": [{"passed": False}, {"passed": True}]}}
    result = {"attempt_id": "attempt-one", "candidate_revision": "revision-one", "task_results": [row],
              "raw": {"attempt_status": "completed", "trace": [1] * 800}}
    atomic_json(directory / "request.json", {"request": request})
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "evidence.json", {"result_hash": stable_hash(result), "source": source, "runtime": request["runtime"]})
    return tmp_path, directory


def test_projection_keeps_final_values_and_original_hashes_and_drops_traces(receipt):
    root, directory = receipt
    before = {path: path.read_bytes() for path in directory.iterdir()}
    compact = publication.project_receipt(root, "attempt-one")
    row = compact["task_results"][0]
    assert row["metrics"] == {"score": .25}
    assert row["gate_status"] == "FAIL" and row["raw_status"] == "completed"
    assert row["cost"]["wall_seconds"] == 12.5
    assert "raw" not in row and "applied" not in row
    assert row["sampling"]["eval_output_noise"] == "clean"
    assert row["evaluator_summary"]["convergence"] == {"observations": 800, "passing_suffix": 0}
    assert row["evaluator_summary"]["terminal_summary"]["passing_checks"] == 1
    assert row["evaluator_summary"]["metric_checks"]["score"]["threshold"] == .5
    assert compact["qualification_reuse"] is False and compact["qualification_input"] is False
    provenance = compact["provenance"]
    assert provenance["canonical_result_hash"] == stable_hash(read_json(directory / "result.json"))
    assert provenance["source_origin_commit"] == "commit-one"
    assert all(item["sha256"] == file_hash(root / item["path"]) for item in provenance["original_files"].values())
    assert before == {path: path.read_bytes() for path in directory.iterdir()}


@pytest.mark.parametrize("field", ["result_hash", "source", "runtime"])
def test_invalid_certificate_is_rejected(receipt, field):
    root, directory = receipt
    certificate = read_json(directory / "evidence.json")
    certificate[field] = "invalid"
    atomic_json(directory / "evidence.json", certificate)
    with pytest.raises(ValueError, match="invalid"):
        publication.project_receipt(root, "attempt-one")
    assert not (root / "reports/forge/technique-receipts").exists()


def _stage(monkeypatch):
    def fake_write(root, goal, *, execution_backend, output_prefix):
        path = Path(str(output_prefix) + ".json")
        atomic_json(path, {"rows": [{"attempt_ids": ["attempt-one"]}]})
        return {"json": str(path), "report": str(output_prefix) + ".md", "rows": 1, "input_digest": "digest-one"}

    def fake_render(result, *, json_link, repo_link_prefix):
        return (f"[receipt]({repo_link_prefix}/reports/forge/attempts/attempt-one/result.json)\n"
                "```sh\npython -m experiments.forge techniques --goal discriminator_stability\n```\n")

    monkeypatch.setattr(publication, "write_report", fake_write)
    monkeypatch.setattr(publication, "render_markdown", fake_render)


def test_regeneration_repairs_receipt_links_and_is_deterministic_without_original_mutation(receipt, monkeypatch):
    root, directory = receipt
    originals = {path: path.read_bytes() for path in directory.iterdir()}
    _stage(monkeypatch)
    result = publication.regenerate(root)
    report = Path(result["report"]).read_text()
    assert "technique-receipts/attempt-one.json" in report
    assert "attempts/attempt-one/result.json" not in report
    assert "python reports/forge/regenerate_technique_inventory.py --root ." in report
    assert "Hydrate byte-exact original" in report
    assert result["summary_receipts"] == 1
    summary = root / "reports/forge/technique-receipts/attempt-one.json"
    first = {path: path.read_bytes() for path in (Path(result["report"]), Path(result["json"]), summary)}
    publication.regenerate(root)
    assert first == {path: path.read_bytes() for path in first}
    assert originals == {path: path.read_bytes() for path in directory.iterdir()}


def test_invalid_original_cannot_overwrite_existing_published_report(receipt, monkeypatch):
    root, directory = receipt
    _stage(monkeypatch)
    output = root / "reports/forge/technique-inventory.md"
    atomic_text(output, "already measured report\n")
    (directory / "result.json").write_text("{}\n")
    with pytest.raises(ValueError, match="invalid result hash"):
        publication.regenerate(root)
    assert output.read_text() == "already measured report\n"
    assert not (root / "reports/forge/technique-receipts").exists()


def test_missing_original_requires_hydration(receipt):
    root, directory = receipt
    (directory / "result.json").unlink()
    with pytest.raises(ValueError, match="hydrate original"):
        publication.project_receipt(root, "attempt-one")


def test_fresh_checkout_cannot_replace_published_measurements_without_hydration(receipt, monkeypatch):
    root, directory = receipt
    _stage(monkeypatch)
    publication.regenerate(root)
    json_path = root / "reports/forge/technique-inventory.json"
    report_path = root / "reports/forge/technique-inventory.md"
    before = (json_path.read_bytes(), report_path.read_bytes())
    (directory / "result.json").unlink()
    with pytest.raises(ValueError, match="hydrate original receipts referenced"):
        publication.regenerate(root)
    assert before == (json_path.read_bytes(), report_path.read_bytes())


def test_live_git_origin_changes_do_not_change_published_bytes_or_receipt_identity(receipt, monkeypatch):
    root, directory = receipt
    head = {"value": "before-report-commit"}
    source_digest = read_json(directory / "request.json")["request"]["source"]["digest"]

    def live_board(root, goal, *, execution_backend, output_prefix):
        path = Path(str(output_prefix) + ".json")
        atomic_json(path, {
            "rows": [{"attempt_ids": ["attempt-one"], "tiers": {"1": {"passed": 0, "total": 3}},
                      "bindings": {"source_digest": source_digest, "source_origin_commit": head["value"]}},
                     {"attempt_ids": [], "bindings": {"source_digest": "unmeasured-source", "source_origin_commit": head["value"]}}],
            "provenance": {"source_board_sha256": stable_hash(head), "input_digest": stable_hash(head),
                           "view_sha256": "stable-view", "reducer_sha256": "stable-reducer"}})
        return {"json": str(path), "input_digest": stable_hash(head)}

    def digest_render(result, **kwargs):
        return "Published digest: " + result["provenance"]["input_digest"] + "\n"

    monkeypatch.setattr(publication, "write_report", live_board)
    monkeypatch.setattr(publication, "render_markdown", digest_render)
    first = publication.regenerate(root)
    paths = [Path(first["json"]), Path(first["report"]), root / "reports/forge/technique-receipts/attempt-one.json"]
    before = {path: path.read_bytes() for path in paths}
    head["value"] = "after-report-commit"
    second = publication.regenerate(root)
    assert first["input_digest"] == second["input_digest"]
    assert before == {path: path.read_bytes() for path in paths}
    result = read_json(first["json"])
    assert result["rows"][0]["bindings"]["source_origin_commit"] == "commit-one"
    assert result["rows"][1]["bindings"]["source_origin_commit"] is None
    assert result["rows"][0]["tiers"] == {"1": {"passed": 0, "total": 3}}
    provenance = result["provenance"]
    assert "source_board_sha256" not in provenance
    assert provenance["view_sha256"] == "stable-view"
    assert provenance["publication_reducer_sha256"] == file_hash(SCRIPT)
    assert provenance["qualified_receipts"]["attempt-one"]["canonical_result_hash"] == stable_hash(read_json(directory / "result.json"))
    assert first["input_digest"] in Path(first["report"]).read_text()


def _git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, text=True, stderr=subprocess.PIPE).strip()


def test_publication_roster_deduplicates_display_labels_but_rejects_changed_science():
    from copy import deepcopy
    row = {"candidate_id": "trial", "candidate_revision": "revision", "cohort": "cohort",
           "technique": "trial", "attempt_ids": ["attempt-one"],
           "tasks": [{"task_id": "toy", "status": "FAIL"}]}
    selected = deepcopy(row)
    selected.update(technique="family", selection={"selection_kind": "best_observed"})
    report = {"rows": [selected], "configuration_rows": [row]}
    assert publication._publication_rows(report) == [row]
    selected["tasks"][0]["status"] = "PASS"
    with pytest.raises(ValueError, match="conflicting scientific rows"):
        publication._publication_rows(report)


@pytest.fixture
def frozen_checkout(receipt):
    root, directory = receipt
    code = root / "experiments/forge"
    code.mkdir(parents=True)
    (root / "experiments/__init__.py").write_text("")
    (code / "__init__.py").write_text("")
    real_code = SCRIPT.parents[2] / "experiments/forge"
    for name in ("contracts.py", "sources.py"):
        (code / name).write_bytes((real_code / name).read_bytes())
    (root / "particlegan").mkdir()
    (root / "particlegan/__init__.py").write_text("")
    (root / "particlegan/fixture.py").write_text("VALUE = 1\n")
    helper = root / "reports/helpers/evaluator.py"
    atomic_text(helper, "threshold = 1\n")
    atomic_json(root / "configs/forge/ideas/new-technique.json", {"id": "new-technique"})
    (code / "technique_board.py").write_text('''
import json
from pathlib import Path
from particlegan.fixture import VALUE
from .sources import inspect_source
def write_report(root, goal, *, execution_backend, output_prefix):
    assert VALUE == 1, "loaded live model instead of frozen model"
    source = inspect_source(root, ["reports/helpers/evaluator.py"])
    result = {"rows": [{"attempt_ids": ["attempt-one"], "imported_value": VALUE,
                       "tiers": {"1": {"passed": 0, "total": 3}},
                       "bindings": {"source_digest": source["digest"], "source_origin_commit": source["origin_commit"]}}],
              "provenance": {"view_sha256": "stable-view", "reducer_sha256": "stable-reducer"}}
    path = Path(str(output_prefix) + ".json")
    path.write_text(json.dumps(result))
    return {"json": str(path), "rows": 1, "input_digest": "upstream"}
''')
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Forge Test")
    _git(root, "config", "user.email", "forge-test@example.com")
    _git(root, "add", "experiments", "particlegan", "configs", "reports/helpers")
    _git(root, "commit", "-qm", "frozen source")
    commit = _git(root, "rev-parse", "HEAD")
    request = read_json(directory / "request.json")
    request["request"]["source"] = inspect_source(root, ["reports/helpers/evaluator.py"])
    atomic_json(directory / "request.json", request)
    certificate = read_json(directory / "evidence.json")
    certificate["source"] = request["request"]["source"]
    atomic_json(directory / "evidence.json", certificate)
    return root, directory, commit


@pytest.mark.parametrize("hidden_source_mismatch", [False, True])
def test_frozen_publication_keeps_and_validates_unselected_trial_receipts(
        frozen_checkout, monkeypatch, hidden_source_mismatch):
    root, directory, _ = frozen_checkout
    board = root / "experiments/forge/technique_board.py"
    board.write_text('''
import json
from pathlib import Path
from .sources import inspect_source
def write_report(root, goal, *, execution_backend, output_prefix):
    source = inspect_source(root, ["reports/helpers/evaluator.py"])
    canonical = {"candidate_id": "canonical", "candidate_revision": "canonical-revision",
                 "cohort": "canonical-cohort", "attempt_ids": [],
                 "bindings": {"source_digest": source["digest"]}}
    trial = {"candidate_id": "new-technique", "candidate_revision": "revision-one",
             "cohort": "trial-cohort", "attempt_ids": ["attempt-one"],
             "tiers": {"1": {"passed": 0, "total": 3}},
             "bindings": {"source_digest": WRONG_SOURCE or source["digest"]}}
    result = {"rows": [canonical], "configuration_rows": [canonical, trial],
              "provenance": {"view_sha256": "stable-view", "reducer_sha256": "stable-reducer"}}
    path = Path(str(output_prefix) + ".json")
    path.write_text(json.dumps(result))
    return {"json": str(path), "rows": 1, "input_digest": "upstream"}
'''.replace("WRONG_SOURCE", repr("wrong-source" if hidden_source_mismatch else None)))
    _git(root, "add", "experiments/forge/technique_board.py")
    _git(root, "commit", "-qm", "declare configuration alternatives")
    commit = _git(root, "rev-parse", "HEAD")
    request = read_json(directory / "request.json")
    request["request"]["source"] = inspect_source(root, ["reports/helpers/evaluator.py"])
    atomic_json(directory / "request.json", request)
    certificate = read_json(directory / "evidence.json")
    certificate["source"] = request["request"]["source"]
    atomic_json(directory / "evidence.json", certificate)
    monkeypatch.setattr(publication, "render_markdown", lambda *args, **kwargs: "Snapshot\nrows\n")
    if hidden_source_mismatch:
        with pytest.raises(ValueError, match="source cohort absent"):
            publication.regenerate(root, source_commit=commit)
        assert not (root / "reports/forge/technique-receipts").exists()
    else:
        metadata = publication.regenerate(root, source_commit=commit)
        result = read_json(metadata["json"])
        assert len(result["rows"]) == 2
        assert result["snapshot_row_scope"] == "all_configuration_variants"
        assert metadata["summary_receipts"] == 1
        assert "attempt-one" in result["provenance"]["qualified_receipts"]
        assert (root / "reports/forge/technique-receipts/attempt-one.json").is_file()


def test_frozen_publication_reconstructs_tier1_question_without_losing_full_denominator(
        frozen_checkout, monkeypatch):
    root, directory, _ = frozen_checkout
    code = root / "experiments/forge"
    atomic_json(root / "configs/forge/ideas/new-technique.json", {
        "id": "new-technique", "decision_contract": {"scope": {"through_tier": 1}}})
    (code / "planning.py").write_text('''
from .contracts import read_json
def load_idea(root, idea_id):
    return read_json(root / 'configs/forge/ideas' / (idea_id + '.json'))
def resolve_idea(root, idea_id, *, view_id, through_tier, freeze_source, execution_backend, cuda_model):
    assert through_tier == load_idea(root, idea_id)['decision_contract']['scope']['through_tier'], 'changed question scope'
    assert freeze_source is False
    return {'through_tier': through_tier}
''')
    (code / "knowledge.py").write_text('''
from .planning import resolve_idea
def _current_request(root, idea_id, view_id, execution_backend='cuda', cuda_model=None):
    return resolve_idea(root, idea_id, view_id=view_id, through_tier=3, freeze_source=False,
                        execution_backend=execution_backend, cuda_model=cuda_model)
''')
    (code / "technique_board.py").write_text('''
import json
from pathlib import Path
from . import knowledge
from .sources import inspect_source
def write_report(root, goal, *, execution_backend, output_prefix):
    knowledge._current_request(root, 'new-technique', goal, execution_backend)
    source = inspect_source(root, ['reports/helpers/evaluator.py'])
    requirements = {'1': ['toy', 'token', 'ae', 'ring', 'words'],
                    '2': ['q-' + str(i) for i in range(19)], '3': ['hold', 'extension']}
    row = {'candidate_id': 'new-technique', 'candidate_revision': 'revision-one',
           'attempt_ids': ['attempt-one'], 'bindings': {'source_digest': source['digest']},
           'tasks': [{'task_id': name, 'status': 'FAIL' if name == 'toy' else 'UNKNOWN'}
                     for names in requirements.values() for name in names]}
    path = Path(str(output_prefix) + '.json')
    path.write_text(json.dumps({'rows': [row], 'tier_requirements': requirements,
                               'provenance': {'view_sha256': 'stable-view', 'reducer_sha256': 'stable-reducer'}}))
    return {'json': str(path), 'rows': 1}
''')
    _git(root, "add", "experiments/forge", "configs/forge/ideas")
    _git(root, "commit", "-qm", "freeze Tier 1 question")
    commit = _git(root, "rev-parse", "HEAD")
    request = read_json(directory / "request.json")
    request["request"].update(through_tier=1, source=inspect_source(root, ["reports/helpers/evaluator.py"]))
    atomic_json(directory / "request.json", request)
    certificate = read_json(directory / "evidence.json")
    certificate["source"] = request["request"]["source"]
    atomic_json(directory / "evidence.json", certificate)
    originals = {path: path.read_bytes() for path in directory.iterdir()}
    monkeypatch.setattr(publication, "render_markdown", lambda *args, **kwargs: "Snapshot\nrows\n")
    result = read_json(publication.regenerate(root, source_commit=commit)["json"])
    assert [len(tasks) for tasks in result["tier_requirements"].values()] == [5, 19, 2]
    row = result["rows"][0]
    assert row["candidate_revision"] == "revision-one" and row["attempt_ids"] == ["attempt-one"]
    assert len(row["tasks"]) == 26
    assert sum(task["status"] == "FAIL" for task in row["tasks"]) == 1
    assert sum(task["status"] == "UNKNOWN" for task in row["tasks"]) == 25
    assert originals == {path: path.read_bytes() for path in directory.iterdir()}


def test_frozen_subprocess_regrades_exact_source_after_live_source_advance(frozen_checkout, monkeypatch):
    root, directory, commit = frozen_checkout

    def render(result, **kwargs):
        return ("# Report\nEach cell grades the exact current cohort. Current rows bind source.\n" +
                result["provenance"]["input_digest"] + "\npython -m experiments.forge techniques --goal stability\n")

    monkeypatch.setattr(publication, "render_markdown", render)
    first = publication.regenerate(root, source_commit=commit)
    paths = [Path(first["json"]), Path(first["report"]), root / "reports/forge/technique-receipts/attempt-one.json"]
    before = {path: path.read_bytes() for path in paths}
    originals = {path: path.read_bytes() for path in directory.iterdir()}
    (root / "particlegan/fixture.py").write_text("VALUE = 2\n")
    (root / "reports/helpers/evaluator.py").write_text("threshold = 2\n")
    (root / "experiments/forge/new_live_module.py").write_text("new_live_code = True\n")
    _git(root, "add", "particlegan", "experiments", "reports/helpers")
    _git(root, "commit", "-qm", "live source advanced")
    second = publication.regenerate(root, source_commit=commit[:12])
    assert second["source_commit"] == commit
    assert second["publication_scope"] == "frozen_source"
    assert before == {path: path.read_bytes() for path in paths}
    assert originals == {path: path.read_bytes() for path in directory.iterdir()}
    result = read_json(first["json"])
    assert result["rows"][0]["imported_value"] == 1
    assert result["frozen_source"]["qualifies_latest_checkout"] is False
    markdown = Path(first["report"]).read_text()
    assert "frozen source cohort" in markdown and "--source-commit " + commit in markdown
    assert "exact recorded cohort" in markdown and "Recorded rows bind" in markdown
    assert "exact current cohort" not in markdown and "Current rows bind" not in markdown


def test_missing_or_unbound_source_commit_is_rejected_before_publication(frozen_checkout):
    root, directory, commit = frozen_checkout
    with pytest.raises(ValueError, match="unknown source commit"):
        publication.regenerate(root, source_commit="missing-commit")
    (root / "particlegan/fixture.py").write_text("VALUE = 2\n")
    _git(root, "add", "particlegan")
    _git(root, "commit", "-qm", "different unrecorded source")
    with pytest.raises(ValueError, match="no hydrated original receipts bind"):
        publication.regenerate(root, source_commit="HEAD")
    assert not (root / "reports/forge/technique-inventory.json").exists()


def test_frozen_source_manifest_mismatch_cannot_publish(frozen_checkout):
    root, directory, commit = frozen_checkout
    resolved = read_json(directory / "request.json")
    source = resolved["request"]["source"]
    source["files"]["particlegan/fixture.py"] = "0" * 64
    source["digest"] = stable_hash(source["files"])
    atomic_json(directory / "request.json", resolved)
    certificate = read_json(directory / "evidence.json")
    certificate["source"] = source
    atomic_json(directory / "evidence.json", certificate)
    with pytest.raises(ValueError, match="reconstructed scientific source differs"):
        publication.regenerate(root, source_commit=commit)
    assert not (root / "reports/forge/technique-inventory.json").exists()


def test_live_source_advance_requires_frozen_commit_or_new_output_prefix(receipt, monkeypatch):
    root, directory = receipt
    _stage(monkeypatch)
    first = publication.regenerate(root)
    before = Path(first["json"]).read_bytes(), Path(first["report"]).read_bytes()

    def empty_board(root, goal, *, execution_backend, output_prefix):
        path = Path(str(output_prefix) + ".json")
        atomic_json(path, {"rows": []})
        return {"json": str(path), "rows": 0}

    monkeypatch.setattr(publication, "write_report", empty_board)
    with pytest.raises(ValueError, match="--source-commit or a different output prefix"):
        publication.regenerate(root)
    assert before == (Path(first["json"]).read_bytes(), Path(first["report"]).read_bytes())
    current = publication.regenerate(root, output_prefix="reports/forge/new-current")
    assert current["publication_scope"] == "live_current"
    assert read_json(current["json"])["rows"] == []


def test_frozen_runtime_mismatch_cannot_overwrite_existing_measurements(receipt, monkeypatch):
    root, directory = receipt
    _stage(monkeypatch)
    first = publication.regenerate(root)
    paths = [Path(first["json"]), Path(first["report"])]
    before = {path: path.read_bytes() for path in paths}

    def incompatible_frozen_report(*args, **kwargs):
        # The original files are hydrated, but the frozen board cannot select
        # their scientific runtime/hardware cohort in this environment.
        return {"rows": 0}, {"rows": []}, "recorded-commit", ["recorded-source"]

    monkeypatch.setattr(publication, "_frozen_report", incompatible_frozen_report)
    with pytest.raises(ValueError, match="restore the recorded runtime/hardware or use a different output prefix"):
        publication.regenerate(root, source_commit="recorded-commit")
    assert before == {path: path.read_bytes() for path in paths}
