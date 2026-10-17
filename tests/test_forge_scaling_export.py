"""Frozen queue evidence exports without training or bulk artifact copying."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.scaling import SCHEMA
from experiments.forge.scaling_export import publish
from experiments.forge.scaling_publication import MANIFEST, load_performance
from test_forge_scaling_publication import receipt


@pytest.fixture
def campaign(tmp_path):
    root, queue = tmp_path / "checkout", tmp_path / "queue"
    candidate = {"id": "dualnorm", "recipe_overrides": {"prior_regularizer": "sbnp"}}
    candidate_path = root / "configs/forge/ideas/dualnorm.json"
    atomic_json(candidate_path, candidate)
    summary = receipt(candidate="dualnorm")
    task = {"id": "r32", "execution": {"performance": summary["spec"]}}
    summary["task_id"] = task["id"]
    request = {"request_id": "request-one", "campaign_id": "performance-v1",
               "candidate": candidate, "candidate_revision": "revision-one",
               "study": {"id": "performance-v1-bcap-pure"}, "view": {"id": "performance"},
               "protocol": {"seed": 0}, "requires_independent_grading": True,
               "source": {"files": summary["identity"]["source"]["files_sha256"], "origin_commit": "a" * 40},
               "tasks": {"r32": task}}
    request["source"]["digest"] = stable_hash(request["source"]["files"])
    definition = {"task_id": "r32", "compatibility_key": "job-one", "budget_seconds": 180,
                  "science": {"candidate_revision": "revision-one"}}
    request["jobs"] = [definition]
    summary["forge_binding"] = {"candidate_revision": "revision-one", "candidate_sha256": stable_hash(candidate),
                                "request_id": request["request_id"], "source_digest": request["source"]["digest"],
                                "task_sha256": stable_hash(task)}
    summary["applied"] = {"recipe": summary["recipe"]}
    summary["artifacts"] = {"trace": "updates.jsonl", "checkpoint": "state.pt", "gif": "training.gif"}
    directory = queue / "performance-v1/attempt-one"
    directory.mkdir(parents=True)
    for name in summary["artifacts"].values():
        (directory / name).write_bytes(b"bulk evidence remains local")
    summary["artifact_sha256"] = {key: file_hash(directory / name) for key, name in summary["artifacts"].items()}
    raw = {"attempt_status": "completed", "result": summary,
           "grading": {"raw_hash": stable_hash(summary), "source_digest": request["source"]["digest"]}}
    result = {"schema_version": 1, "attempt_id": "attempt-one", "candidate_revision": "revision-one",
              "task_results": [{"task_id": "r32", "gate_status": "PASS", "raw_status": "completed"}], "raw": raw}
    atomic_json(directory / "request.json", {"request": request, "job": definition, "worker": {"attempt": "attempt-one"}})
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "raw-result.json", summary)
    atomic_json(directory / "adapter-receipt.json", summary)
    atomic_json(directory / "terminal.json", raw)
    state = {"schema_version": 1, "campaigns": {}, "events": [], "charges": [],
             "submissions": {"request-one": {"status": "completed", "request": request}},
             "jobs": {"job-one": {"definition": definition, "status": "terminal", "result": result,
                                   "attempts": [{"attempt_id": "attempt-one", "path": str(directory)}]}}}
    atomic_json(queue / "queue/state.json", state)
    registration = {"schema": SCHEMA, "id": "performance-v1", "protocol_seed": 0,
                    "qualification_input": False, "qualification_reuse": False,
                    "campaign": {"id": "performance-v1"}, "view": "performance", "task_ids": ["r32"],
                    "task_contract_sha256": {"r32": stable_hash(task)},
                    "family_selections": [{"family": "bcap-pure", "candidate_id": "dualnorm",
                                            "study": "performance-v1-bcap-pure",
                                            "candidate_declaration_sha256": file_hash(candidate_path)}]}
    atomic_json(root / "configs/forge/performance/performance-v1.json", registration)
    atomic_json(root / "configs/forge/tasks/r32.json", task)
    board = {"rows": [{"trainer_family": "bcap-dualnorm", "candidate_id": "dualnorm", "qualified_tier": 3}],
             "provenance": {}}
    board["provenance"]["input_digest"] = stable_hash(board)
    atomic_json(root / "reports/forge/technique-inventory.json", board)
    (root / "reports/forge/technique-inventory.md").write_text("# Scientific results\n\nRecorded scores.\n\n## Refresh\n")
    return {"root": root, "queue": queue, "state": state, "directory": directory, "board": board}


def save_state(campaign):
    atomic_json(campaign["queue"] / "queue/state.json", campaign["state"])


def test_export_binds_exact_job_source_candidate_and_display_owner(campaign):
    outcome = publish(campaign["root"], campaign["queue"], "performance-v1")
    assert outcome["performance_records"] == 1 and outcome["bulk_artifacts_copied"] is False
    evidence = load_performance(campaign["root"])
    row = evidence["records"][0]
    assert row["trainer_family"] == "bcap-dualnorm"
    assert row["candidate_id"] == "dualnorm"
    proof = row["result"]["export_provenance"]
    assert proof["registration_family"] == "bcap-pure"
    assert proof["attempt_id"] == "attempt-one"
    assert proof["frozen_job_sha256"] == stable_hash(campaign["state"]["jobs"]["job-one"]["definition"])
    assert proof["original_files"]["canonical_result"]["sha256"] == file_hash(campaign["directory"] / "result.json")
    assert proof["local_artifact_root"] == str(campaign["directory"])
    assert proof["bulk_artifacts"]["trace"]["sha256"] == file_hash(campaign["directory"] / "updates.jsonl")
    board = read_json(campaign["root"] / "reports/forge/technique-inventory.json")
    assert board["rows"] == campaign["board"]["rows"]
    assert "bcap-dualnorm" in (campaign["root"] / "reports/forge/technique-inventory.md").read_text()
    assert not list((campaign["root"] / "reports/forge/scaling-receipts").rglob("*.pt"))
    assert not list((campaign["root"] / "reports/forge/scaling-receipts").rglob("*.jsonl"))


def test_export_is_deterministic_and_retains_missing_bulk_digests(campaign):
    (campaign["directory"] / "state.pt").unlink()
    publish(campaign["root"], campaign["queue"], "performance-v1")
    paths = [path for path in campaign["root"].rglob("*") if path.is_file()]
    before = {path: path.read_bytes() for path in paths}
    publish(campaign["root"], campaign["queue"], "performance-v1")
    assert all(path.read_bytes() == data for path, data in before.items())
    proof = load_performance(campaign["root"])["records"][0]["result"]["export_provenance"]
    assert proof["bulk_artifacts"]["checkpoint"]["available_locally"] is False
    assert proof["bulk_artifacts"]["checkpoint"]["sha256"]


def test_registered_roster_excludes_historical_alternatives_and_preserves_new_selection_unknown(campaign):
    path = campaign["root"] / "reports/forge/technique-inventory.json"
    board = read_json(path)
    board["rows"].append({"trainer_family": "bcap-pure", "candidate_id": "historical-adam"})
    board["provenance"].pop("input_digest")
    board["provenance"]["input_digest"] = stable_hash(board)
    atomic_json(path, board)
    publish(campaign["root"], campaign["queue"], "performance-v1")
    markdown = path.with_suffix(".md").read_text()
    assert "historical-adam" not in markdown
    assert "This registration covers 1 complete recipes" in markdown
    board = read_json(path)
    board["rows"][0]["candidate_id"] = "new-selection"
    from experiments.forge.scaling_publication import render_performance
    markdown = "\n".join(render_performance(campaign["root"], board, path.with_suffix(".md")))
    assert "No current selected-recipe update timings are measured" in markdown
    assert "**unknown**" in markdown


def test_changed_bulk_or_queue_result_blocks_current_publication(campaign):
    (campaign["directory"] / "state.pt").write_bytes(b"tampered state")
    with pytest.raises(ValueError, match="bulk artifact bytes changed"):
        publish(campaign["root"], campaign["queue"], "performance-v1")
    assert not (campaign["root"] / MANIFEST).exists()


def test_partial_publication_is_explicit_and_running_cells_stay_unknown(campaign):
    job = campaign["state"]["jobs"]["job-one"]
    job.update(status="running", result=None)
    save_state(campaign)
    with pytest.raises(ValueError, match="unfinished cells"):
        publish(campaign["root"], campaign["queue"], "performance-v1")
    result = publish(campaign["root"], campaign["queue"], "performance-v1", allow_partial=True)
    assert result["partial_snapshot"] is True and result["unavailable_records"] == 1
    evidence = load_performance(campaign["root"])
    assert evidence["records"] == []
    assert evidence["unavailable"][0]["status"] == "UNKNOWN"
    assert "running" in evidence["unavailable"][0]["reason"]
    unavailable = read_json(campaign["root"] / evidence["unavailable"][0]["receipt"])
    assert "metrics" not in unavailable and "identity" not in unavailable
    assert "running" in (campaign["root"] / "reports/forge/technique-inventory.md").read_text()


@pytest.mark.parametrize("attempt_status,status,detail", [("error", "ERROR", {}),
    ("timeout", "INCOMPLETE", {}), ("error", "OOM", {"error": {"type": "OutOfMemoryError", "message": "CUDA out of memory"}}),
    ("error", "UNSUPPORTED", {"applicability": {"status": "unsupported", "reason": "encoder unavailable"}})])
def test_no_core_failures_keep_exact_queue_outcome_without_measurements(campaign, attempt_status, status, detail):
    job = campaign["state"]["jobs"]["job-one"]
    job["result"]["raw"] = {"attempt_status": attempt_status, "result": detail, "reason": "original failure"}
    job["result"]["task_results"][0]["gate_status"] = "INCOMPLETE"
    atomic_json(campaign["directory"] / "result.json", job["result"])
    save_state(campaign)
    result = publish(campaign["root"], campaign["queue"], "performance-v1")
    assert result["counts"] == {status: 1}
    assert load_performance(campaign["root"])["unavailable"][0]["status"] == status


def test_frozen_task_and_recipe_mismatch_rejected_before_writing(campaign):
    request = campaign["state"]["submissions"]["request-one"]["request"]
    request["tasks"]["r32"]["execution"]["performance"]["batch_size"] = 32
    save_state(campaign)
    with pytest.raises(ValueError, match="frozen task differs"):
        publish(campaign["root"], campaign["queue"], "performance-v1")
    assert not (campaign["root"] / MANIFEST).exists()


def test_independent_evaluation_failure_cannot_display_core_success(campaign):
    result = campaign["state"]["jobs"]["job-one"]["result"]
    result["task_results"][0].update(gate_status="INVALID", reason="independent trace verification failed")
    atomic_json(campaign["directory"] / "result.json", result)
    save_state(campaign)
    outcome = publish(campaign["root"], campaign["queue"], "performance-v1")
    assert outcome["counts"] == {"INVALID": 1}
    assert "INVALID" in (campaign["root"] / "reports/forge/technique-inventory.md").read_text()


def test_registration_roster_cannot_drop_a_cell_even_with_a_resealed_manifest(campaign):
    publish(campaign["root"], campaign["queue"], "performance-v1")
    manifest = read_json(campaign["root"] / MANIFEST)
    manifest["entries"] = []
    manifest.pop("input_digest")
    manifest["input_digest"] = stable_hash(manifest)
    atomic_json(campaign["root"] / MANIFEST, manifest)
    with pytest.raises(ValueError, match="every registered recipe/task cell"):
        load_performance(campaign["root"])


def test_cli_has_publish_without_training_dispatch():
    from experiments.forge.__main__ import parser
    args = parser().parse_args(["scaling", "publish", "registration", "--allow-partial"])
    assert args.stage == "publish" and args.allow_partial is True
