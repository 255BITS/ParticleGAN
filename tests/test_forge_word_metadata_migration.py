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
RECEIPT_SOURCE_REFRESH = "guarded-gram-default-receipt-source-binding-v1"
RECEIPT_SOURCE_PARENT = "90ae663551dd0fa35749d56e1859f276a4d09587"
RECEIPT_SOURCE_MEMBER = "experiments/forge/behavior_adapters.py"


def _git_blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _before_default_receipt_source_refresh(current, relative):
    """Validate the one declared source-only layer, retaining every old gate."""
    revision = current["evaluation"]["evaluator_revision"]
    assert revision["id"] == RECEIPT_SOURCE_REFRESH
    assert revision["previous_task_commit"] == RECEIPT_SOURCE_PARENT
    assert revision["previous_task_path"] == relative
    previous_bytes = subprocess.check_output([
        "git", "show", RECEIPT_SOURCE_PARENT + ":" + relative], cwd=ROOT)
    assert hashlib.sha256(previous_bytes).hexdigest() == revision["previous_task_sha256"]
    previous = json.loads(previous_bytes)
    sources = previous["evaluation"]["sources"]
    assert revision["previous_source_sha256"] == {RECEIPT_SOURCE_MEMBER: sources[RECEIPT_SOURCE_MEMBER]}
    assert revision["previous_revision"] == previous["evaluation"]["evaluator_revision"]
    assert current["evaluation"]["sources"] == {
        **sources, RECEIPT_SOURCE_MEMBER: file_hash(ROOT / RECEIPT_SOURCE_MEMBER)}
    restored = deepcopy(current)
    restored["evaluation"]["sources"] = sources
    restored["evaluation"]["evaluator_revision"] = revision["previous_revision"]
    assert restored == previous
    return previous_bytes


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
        current = read_json(ROOT / binding["path"])
        previous_bytes = _before_default_receipt_source_refresh(current, binding["path"])
        # The release-090 receipt stays immutable: its complete output is now
        # the exact previous task of this later, explicitly checked layer.
        assert hashlib.sha256(previous_bytes).hexdigest() == binding["current_sha256"]
        current = json.loads(previous_bytes)
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


def test_default_receipt_source_refresh_preserves_all_ten_current_questions():
    hosts = ("two_pole", "unused_token_hold", "ae_gan_hold", "trajectory",
             "residual_student", "unipolar", "cover_leftover", "mid_scale_identity",
             "five_word_joint_smoke", "five_word_joint_hold")
    for host in hosts:
        relative = f"configs/forge/tasks/{host}.json"
        _before_default_receipt_source_refresh(read_json(ROOT / relative), relative)
