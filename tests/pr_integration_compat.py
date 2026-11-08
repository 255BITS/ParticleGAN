"""Verify additive PR/develop source bindings against exact archived task bytes."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess


def before_pr_integration(root, current, relative):
    evaluation = current.get("evaluation", {})
    binding = evaluation.get("evaluator_revision")
    if not isinstance(binding, dict) or "-develop-integration-20261108" not in binding.get("id", ""):
        return current
    root = Path(root)
    raw = subprocess.check_output(["git", "show", binding["previous_task_commit"] + ":" + relative], cwd=root)
    assert binding["previous_task_path"] == relative
    assert sha256(raw).hexdigest() == binding["previous_task_sha256"]
    import hashlib
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == binding["previous_task_blob"]
    previous = json.loads(raw)
    receipt_path = root / "reports/forge/pr-integration-20261108" / (binding["id"].split("-develop-")[0] + ".json")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema"] == "pr_develop_integration_source_binding_v1"
    assert all(receipt[k] is False for k in ("training_rerun", "qualification_results_changed", "automatic_cross_source_reuse", "qualification_input"))
    row = next(row for row in receipt["tasks"] if row["previous_task_path"] == relative)
    assert sha256((root / relative).read_bytes()).hexdigest() == row["current_task_sha256"]
    assert row["sources_before"] == previous["evaluation"]["sources"]
    assert row["sources_after"] == current["evaluation"]["sources"]
    for path, expected in row["sources_after"].items():
        assert sha256((root / path).read_bytes()).hexdigest() == expected
    restored = deepcopy(current)
    if "evaluator_revision" in previous["evaluation"]:
        restored["evaluation"]["evaluator_revision"] = previous["evaluation"]["evaluator_revision"]
    else:
        restored["evaluation"].pop("evaluator_revision")
    restored["evaluation"]["sources"] = previous["evaluation"]["sources"]
    if receipt["pr"] == 461 and current["id"] in ("model_vae_hard", "model_vae_categorical"):
        assert restored["execution"].pop("transport_consumer") == "output_marginal_v1"
        assert restored["execution"].pop("observation_checkpoints") is True
    assert restored == previous
    return previous


def source_before_pr_integration(root, relative, member):
    root = Path(root)
    task = json.loads((root / relative).read_text())
    binding = task.get("evaluation", {}).get("evaluator_revision", {})
    if isinstance(binding, dict) and "-develop-integration-20261108" in binding.get("id", ""):
        task = before_pr_integration(root, task, relative)
        return task["evaluation"]["sources"][member]
    return sha256((root / member).read_bytes()).hexdigest()
