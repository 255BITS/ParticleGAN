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
POLICY_CHECKPOINT_REFRESH = "mog-policy-extra-state-source-binding-v1"
POLICY_SOURCE_MEMBER = "particlegan/policy.py"
POLICY_MIGRATION = "reports/forge/api-model-toys/policy-checkpoint-migration.json"
REVIEW_POLICY_REFRESH = "review-policy-source-binding-20261025"


def _before_review_policy_source_refresh(current, relative):
    """Peel the review-stage policy.py rebinding; return (previous card, its sha256)."""
    revision = current["evaluation"]["evaluator_revision"]
    if revision["id"] != REVIEW_POLICY_REFRESH:
        return current, file_hash(ROOT / relative)
    assert revision["previous_task_path"] == relative
    previous_bytes = subprocess.check_output([
        "git", "show", revision["previous_task_commit"] + ":" + relative], cwd=ROOT)
    assert hashlib.sha256(previous_bytes).hexdigest() == revision["previous_task_sha256"]
    assert _git_blob(previous_bytes) == revision["previous_task_blob"]
    previous = json.loads(previous_bytes)
    sources = previous["evaluation"]["sources"]
    assert revision["previous_revision"] == previous["evaluation"]["evaluator_revision"]
    assert revision["previous_source_sha256"] == {POLICY_SOURCE_MEMBER: sources[POLICY_SOURCE_MEMBER]}
    assert current["evaluation"]["sources"] == {
        **sources, POLICY_SOURCE_MEMBER: file_hash(ROOT / POLICY_SOURCE_MEMBER)}
    restored = deepcopy(current)
    restored["evaluation"]["sources"] = sources
    restored["evaluation"]["evaluator_revision"] = revision["previous_revision"]
    assert restored == previous
    return previous, revision["previous_task_sha256"]


def _before_policy_checkpoint_source_refresh(current, relative):
    """Peel only the additive layer, preserving the exact upstream source pins."""
    current, current_sha256 = _before_review_policy_source_refresh(current, relative)
    revision = current["evaluation"]["evaluator_revision"]
    if revision["id"] != POLICY_CHECKPOINT_REFRESH:
        return current
    receipt = read_json(ROOT / POLICY_MIGRATION)
    binding = next(row for row in receipt["tasks"] if row["original_path"] == relative)
    assert revision["previous_task_commit"] == binding["upstream_commit"]
    assert revision["previous_task_path"] == binding["upstream_path"] == relative
    previous_bytes = subprocess.check_output([
        "git", "show", binding["upstream_commit"] + ":" + relative], cwd=ROOT)
    assert hashlib.sha256(previous_bytes).hexdigest() == binding["upstream_task_sha256"] == revision["previous_task_sha256"]
    assert _git_blob(previous_bytes) == binding["upstream_blob"] == revision["previous_task_blob"]
    previous = json.loads(previous_bytes)
    sources = previous["evaluation"]["sources"]
    assert sources == binding["upstream_sources"]
    assert revision["previous_revision"] == previous["evaluation"]["evaluator_revision"]
    assert revision["previous_source_sha256"] == {POLICY_SOURCE_MEMBER: sources[POLICY_SOURCE_MEMBER]}
    assert sources[POLICY_SOURCE_MEMBER] == binding["previous_policy_sha256"]
    assert current["evaluation"]["sources"] == {
        **sources, POLICY_SOURCE_MEMBER: binding["current_policy_sha256"]}
    assert current_sha256 == binding["current_task_sha256"]
    restored = deepcopy(current)
    restored["evaluation"]["sources"] = sources
    restored["evaluation"]["evaluator_revision"] = revision["previous_revision"]
    assert restored == previous
    return previous


def _git_blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _before_default_receipt_source_refresh(current, relative):
    """Validate the one declared source-only layer, retaining every old gate."""
    current = _before_policy_checkpoint_source_refresh(current, relative)
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
        assert current["evaluation"]["sources"][member] == receipt["current_bound_source"]["sha256"]
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


def test_policy_checkpoint_binding_is_additive_and_preserves_exact_original_sources():
    receipt = read_json(ROOT / POLICY_MIGRATION)
    assert receipt["schema"] == "source_only_policy_checkpoint_binding_v1"
    assert receipt["qualification_results_changed"] is receipt["training_rerun"] is False
    assert receipt["scientific_questions_changed"] is receipt["automatic_cross_source_reuse"] is False
    assert {row["task"] for row in receipt["tasks"]} == {"five_word_joint_smoke", "five_word_joint_hold"}
    for binding in receipt["tasks"]:
        relative = binding["original_path"]
        original_bytes = subprocess.check_output([
            "git", "show", binding["original_commit_full"] + ":" + relative], cwd=ROOT)
        assert binding["original_commit_full"].startswith(binding["original_commit"])
        assert hashlib.sha256(original_bytes).hexdigest() == binding["original_task_sha256"]
        assert _git_blob(original_bytes) == binding["original_blob"]
        assert json.loads(original_bytes)["evaluation"]["sources"] == binding["original_sources"]
        current = read_json(ROOT / relative)
        assert current["evaluation"]["evaluator_revision"]["id"] == REVIEW_POLICY_REFRESH
        upstream = _before_policy_checkpoint_source_refresh(current, relative)
        assert upstream["evaluation"]["evaluator_revision"]["id"] == binding["additive_previous_revision_id"] == RECEIPT_SOURCE_REFRESH
        original = json.loads(original_bytes)
        restored = deepcopy(upstream)
        restored["evaluation"]["sources"] = original["evaluation"]["sources"]
        restored["evaluation"]["evaluator_revision"] = original["evaluation"]["evaluator_revision"]
        assert restored == original
