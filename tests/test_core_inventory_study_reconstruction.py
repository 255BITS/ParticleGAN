"""Reporting-only reconstruction over admitted synthetic receipts; no training."""
from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

from experiments.forge import knowledge
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue
from reports.forge import regenerate_technique_inventory as publication
from test_forge_configuration_search import checkout, complete
from test_forge_studies import ready_study


@pytest.fixture
def admitted(checkout):
    study = ready_study(checkout)
    # Exact Git-pinned declarations and motivation, without a training process.
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "-c", "user.name=Report Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "Frozen report fixture"], cwd=checkout, check=True)
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    request = resolve_idea(checkout, "reusable", study=study["id"], freeze_source=True, queue_root=queue.root)
    queue.submit(request, study["campaign"])
    complete(queue, 1.)
    manifests = {request["source"]["digest"]: request["source"]}
    return checkout, request, manifests


def test_exact_study_reconstructs_authority_without_changing_numeric_outcomes(admitted, monkeypatch):
    root, original, manifests = admitted
    originals = {p: p.read_bytes() for p in (root / "reports/forge/attempts").rglob("*.json")}
    before = knowledge.board(root, "stability")
    records, _ = publication._recorded_studies(root, manifests)
    proofs = []
    monkeypatch.setattr(knowledge, "_current_request", lambda root, idea, view, backend, model:
                        publication._recorded_study_question(root, idea, view, backend, model, records, proofs))
    after = knowledge.board(root, "stability")
    find = lambda board: next(row for row in board["rows"]
                              if row["candidate_id"] == "reusable" and row["runtime_cohort"]["execution_backend"] == "cpu")
    old, new = find(before), find(after)
    assert old["qualified_tier"] == 0 and new["qualified_tier"] == 1
    assert old["qualification"]["task_statuses"] == new["qualification"]["task_statuses"]
    assert new["qualification"]["task_statuses"]["t1"] == "PASS"
    assert proofs[0]["study_id"] == original["study"]["id"]
    assert proofs[0]["study_sha256"] == stable_hash(original["study"])
    assert all(p.read_bytes() == data for p, data in originals.items())


@pytest.mark.parametrize("field", ["study", "study_admission", "study_review", "study_sha256"])
def test_missing_or_tampered_recorded_binding_is_rejected(admitted, field):
    root, _, manifests = admitted
    path = next((root / "reports/forge/attempts").glob("*/request.json"))
    envelope = read_json(path)
    if field == "study_sha256":
        envelope["request"]["study_admission"][field] = "0" * 64
    else:
        envelope["request"].pop(field)
    atomic_json(path, envelope)
    with pytest.raises(ValueError, match="study.*binding|study.*fingerprint"):
        publication._recorded_studies(root, manifests)


def test_tampered_second_member_review_cannot_hide_in_first_binding(admitted):
    root, _, manifests = admitted
    path = next((root / "reports/forge/attempts").glob("*/request.json"))
    member = read_json(path)
    member["request"]["study_review"]["expected"]["task_map"] = {"t1": "t2"}
    atomic_json(root / "reports/forge/attempts/second-member/request.json", member)
    records, _ = publication._recorded_studies(root, manifests)
    with pytest.raises(ValueError, match="ambiguous original study execution binding"):
        publication._recorded_study_question(root, "reusable", "stability", "cpu", None, records, [])


@pytest.mark.parametrize("change", ["study", "source", "runtime"])
def test_reconstruction_rejects_changed_declared_or_actual_cohort(admitted, change):
    root, original, manifests = admitted
    records, _ = publication._recorded_studies(root, manifests)
    if change == "study":
        path = root / "configs/forge/studies/question-a.json"
        study = read_json(path)
        study["prediction"]["threshold"] = .99
        atomic_json(path, study)
    elif change == "source":
        (root / "particlegan/fixture.py").write_text("mechanism = 2\n")
    else:
        records[0]["request"] = deepcopy(records[0]["request"])
        records[0]["request"]["runtime"]["python"] = "0.0.0"
    with pytest.raises(ValueError, match="reconstructed study/source/view/cohort"):
        publication._recorded_study_question(root, "reusable", "stability", "cpu", None, records, [])


def test_git_pinned_motivation_is_hydrated_before_live_overlay(admitted, tmp_path):
    root, original, manifests = admitted
    _, support = publication._recorded_studies(root, manifests)
    # Live reporting may evolve; the frozen reader must use admitted Git bytes.
    atomic_json(root / "reports/prior.json", {"unrelated": "live report changed"})
    destination = tmp_path / "reconstructed"
    publication._extract_source(root, original["source"]["origin_commit"], destination, manifests, support)
    prior = destination / "reports/prior.json"
    assert not prior.is_symlink() and prior.resolve().is_relative_to(destination.resolve())
    assert file_hash(prior) == support["reports/prior.json"]


@pytest.mark.parametrize("support,message", [
    ({"reports/prior.json": "0" * 64}, "original admission hash"),
    ({"../outside.json": "0" * 64}, "unsafe recorded study"),
])
def test_git_motivation_rejects_wrong_hash_and_outside_path(admitted, tmp_path, support, message):
    root, original, manifests = admitted
    with pytest.raises(ValueError, match=message):
        publication._extract_source(root, original["source"]["origin_commit"], tmp_path / "reconstructed", manifests, support)
