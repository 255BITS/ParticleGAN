"""Verify current update-boundary bindings without rewriting archived task evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def receipts():
    return [json.loads(path.read_text()) for path in sorted(
        (ROOT / "reports/software-audits").glob("pr*-update-boundary-source-binding.json"))]


def before_update_boundary_refresh(current, relative):
    revision = current["evaluation"]["evaluator_revision"]
    matches = [receipt for receipt in receipts() if receipt["revision_id"] == revision["id"]]
    if not matches:
        return current, hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    receipt, = matches
    binding, = [row for row in receipt["tasks"] if row["path"] == relative]
    source = receipt["source_member"]
    assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == receipt["current_source_sha256"]
    previous_bytes = subprocess.check_output([
        "git", "show", binding["previous_task_commit"] + ":" + relative], cwd=ROOT)
    assert hashlib.sha256(previous_bytes).hexdigest() == binding["previous_task_sha256"] == revision["previous_task_sha256"]
    blob = hashlib.sha1(f"blob {len(previous_bytes)}\0".encode() + previous_bytes).hexdigest()
    assert blob == binding["previous_task_blob"] == revision["previous_task_blob"]
    assert revision["previous_task_commit"] == binding["previous_task_commit"]
    assert revision["previous_task_path"] == relative
    previous = json.loads(previous_bytes)
    assert revision["previous_revision"] == previous["evaluation"]["evaluator_revision"]
    assert revision["previous_source_sha256"] == {source: previous["evaluation"]["sources"][source]}
    assert binding["previous_policy_sha256"] == previous["evaluation"]["sources"][source]
    assert current["evaluation"]["sources"] == {
        **previous["evaluation"]["sources"], source: receipt["current_source_sha256"]}
    restored = deepcopy(current)
    restored["evaluation"]["sources"] = previous["evaluation"]["sources"]
    restored["evaluation"]["evaluator_revision"] = previous["evaluation"]["evaluator_revision"]
    assert restored == previous
    return previous, binding["previous_task_sha256"]


def previous_policy_hash():
    values = {row["previous_policy_sha256"] for receipt in receipts() for row in receipt["tasks"]}
    if not values:
        return hashlib.sha256((ROOT / "particlegan/policy.py").read_bytes()).hexdigest()
    value, = values
    return value
