"""A goal-only registration correction retains strict legacy admission."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.forge.contracts import file_hash, read_json
from experiments.forge.decision_contracts import validate_legacy_admission
from experiments.forge.planning import candidate_revision_for, resolve_idea


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = "configs/forge/ideas/five-word-joint-ka2-v1.json"


def _git_blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_word_goal_migration_preserves_exact_original_and_only_changes_metadata():
    manifest = read_json(ROOT / "configs/forge/legacy-ideas-v1.json")
    migration = next(row for row in manifest["metadata_migrations"]
                     if row["declaration"] == DECLARATION)
    original = (ROOT / migration["original"]["archive"]).read_bytes()
    corrected = (ROOT / DECLARATION).read_bytes()
    for name, data in (("original", original), ("corrected", corrected)):
        assert hashlib.sha256(data).hexdigest() == migration[name]["sha256"]
        assert _git_blob(data) == migration[name]["git_blob"]
        assert len(migration[name]["commit"]) == 40
    before, after = json.loads(original), json.loads(corrected)
    assert migration["changed_fields"] == {
        "goal": {"before": "five_word_joint", "after": "discriminator_stability"}}
    assert before.pop("goal") == "five_word_joint"
    assert after.pop("goal") == "discriminator_stability"
    assert before == after
    assert manifest["declarations"][DECLARATION] == file_hash(ROOT / DECLARATION)
    assert migration["scientific_formulation_unchanged"] is True
    assert migration["qualification_regraded"] is migration["training_launched"] is False


def test_word_goal_correction_admits_main_tier1_without_changing_candidate_revision():
    request = resolve_idea(ROOT, "five-word-joint-ka2-v1", through_tier=1,
                           execution_backend="cpu")
    assert request["candidate"]["goal"] == request["view"]["id"] == "discriminator_stability"
    assert request["preflight_blockers"] == []
    assert all(not task["preflight_blockers"] for name, task in request["tasks"].items()
               if name in {"two_pole", "unused_token_hold", "ae_gan_hold",
                           "ring16_acquisition", "five_word_joint_smoke"})
    validate_legacy_admission(request, root=ROOT)
    original = deepcopy(request["candidate"])
    original["goal"] = "five_word_joint"
    assert candidate_revision_for(request["source"]["digest"], original) == request["candidate_revision"]


def test_current_word_checkpoint_source_refresh_preserves_questions_and_archived_variants():
    receipt = read_json(ROOT / "reports/forge/release-090/source-bindings.json")
    member = receipt["source_member"]
    for source in ("previous_bound_source", "current_bound_source"):
        identity = receipt[source]
        original = subprocess.check_output(["git", "show", identity["commit"] + ":" + member], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == identity["sha256"]
    for binding in receipt["tasks"]:
        original_bytes = subprocess.check_output([
            "git", "show", receipt["pre_refresh_task_declaration_commit"] + ":" + binding["path"]], cwd=ROOT)
        assert hashlib.sha256(original_bytes).hexdigest() == binding["previous_sha256"]
        original = json.loads(original_bytes)
        assert file_hash(ROOT / binding["path"]) == binding["current_sha256"]
        current = read_json(ROOT / binding["path"])
        revision = current["evaluation"]["evaluator_revision"]
        assert revision["previous_revision"] == original["evaluation"]["evaluator_revision"]
        assert revision["previous_source_sha256"] == {member: original["evaluation"]["sources"][member]}
        assert current["evaluation"]["sources"][member] == file_hash(ROOT / member)
        current["evaluation"]["sources"][member] = original["evaluation"]["sources"][member]
        current["evaluation"]["evaluator_revision"] = original["evaluation"]["evaluator_revision"]
        assert current == original
    for path, digest in receipt["preserved_historical_policy_variants"].items():
        assert file_hash(ROOT / path) == digest
    assert receipt["qualification_input"] is receipt["qualification_regraded"] is False
    assert receipt["scientific_training_launched"] is receipt["automatic_cross_source_reuse"] is False
