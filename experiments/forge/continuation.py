"""Independent verification of native continuation's preserved prefix."""
from pathlib import Path

import torch

from .artifacts import verify_artifacts
from .contracts import file_hash
from .state import require_optimizer_steps, require_same_formulation, state_digest


def verify_native_prefix(task, evidence):
    prefix = evidence["prefix_parity"]
    root = Path(evidence["artifact_root"])
    retained = root / "prefix"
    manifest = prefix["artifact_manifest"]
    verify_artifacts(retained, manifest)
    checkpoint = prefix["checkpoint"]
    if checkpoint["path"] not in manifest["files"]:
        raise ValueError("prefix checkpoint is not bound by the complete artifact manifest")
    path = retained / checkpoint["path"]
    if file_hash(path) != checkpoint["sha256"]:
        raise ValueError("prefix checkpoint file differs from its certified original")
    original = torch.load(path, map_location="cpu", weights_only=True)
    restored = torch.load(root / "resume-restored.pt", map_location="cpu", weights_only=True)
    digest = state_digest(original)
    if (digest != state_digest(restored) or digest != checkpoint["state_sha256"]
            or digest != prefix["reference_sha256"] or digest != prefix["continued_sha256"]):
        raise ValueError("continuation did not restore the exact prefix state")
    execution = task["execution"]
    if original["trainer"]["completed_steps"] != execution["preserve_prefix_steps"]:
        raise ValueError("prefix checkpoint update count differs from the declared continuation")
    require_optimizer_steps(original, execution["preserve_prefix_steps"])
    if original["recipe"]["total_steps"] != execution["original_schedule_horizon"]:
        raise ValueError("continuation changed the original scheduling horizon")
    parent = prefix["prerequisite"]
    if (parent["result"]["task_id"] != execution["continuation_of"]
            or parent["result"]["gate_status"] != "PASS"
            or parent["result"]["compatibility_key"] != parent["compatibility_key"]
            or parent["result"]["evidence"]["checkpoint"] != checkpoint
            or parent["result"]["evidence"]["artifact_manifest"] != manifest):
        raise ValueError("continuation prefix declaration differs from its prerequisite receipt")
    current = evidence["checkpoint"]
    if current["path"] not in evidence["artifact_manifest"]["files"]:
        raise ValueError("continued checkpoint is not certified")
    current_path = root / current["path"]
    final = torch.load(current_path, map_location="cpu", weights_only=True)
    if file_hash(current_path) != current["sha256"] or state_digest(final) != current["state_sha256"]:
        raise ValueError("continued checkpoint differs from its declaration")
    require_same_formulation(original, final)
    if (final["recipe"] != original["recipe"] or final["prior"] != original["prior"]
            or final["trainer"]["completed_steps"] != execution["steps"]
            or final["trainer"].get("max_steps") != execution["steps"]):
        raise ValueError("continued state changed formulation or lacks all declared updates")
    require_optimizer_steps(final, execution["steps"])
    verify_artifacts(retained, manifest)
