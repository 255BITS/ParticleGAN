"""Audit one existing checkpoint as saved tensor data, with no neural execution."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import file_hash, read_json, stable_hash
from reports.forge.collect_gaussian_smoke_inventory import saved_state_certificate, validate_attempt
from reports.forge.regenerate_technique_inventory import project_receipt


def audit(root, attempt_id, *, origin, digest):
    # Loading existing tensor bytes is distinct from constructing a model,
    # restoring a live learner, drawing samples or executing its forward path.
    import torch
    from experiments.forge.state import state_digest

    if torch.cuda.is_initialized():
        raise ValueError("saved-data audit requires a fresh process without a CUDA context")
    root = Path(root).resolve()
    summary = project_receipt(root, attempt_id)
    directory = root / "reports/forge/attempts" / attempt_id
    resolved, result = read_json(directory / "request.json"), read_json(directory / "result.json")
    validate_attempt(root, resolved, result, origin=origin, digest=digest)
    if result["raw"]["attempt_status"] != "completed" or len(result["task_results"]) != 1:
        raise ValueError("route probe needs one original completed task, without group pooling")
    row = result["task_results"][0]
    task_id = row["task_id"]
    if task_id not in {"gaussian1d_smoke", "two_pole", "ring16_acquisition", "five_word_joint_acquisition"}:
        raise ValueError("route probe is scoped to the four declared inventory checks")
    task = resolved["request"]["tasks"][task_id]
    certificate = saved_state_certificate(row, task)
    descriptor = row["evidence"]["provenance_checkpoint"]
    tags = {}

    def cpu_storage(storage, original_location):
        tags[storage.data_ptr()] = original_location
        return storage

    path = Path(descriptor["artifact_root"]) / descriptor["path"]
    payload = torch.load(path, weights_only=True, map_location=cpu_storage)
    if state_digest(payload) != descriptor["state_sha256"]:
        raise ValueError("saved public-state content digest differs from its certificate")
    streams = payload["streams"]
    observed_rng = row.get("rng", row.get("applied", {}).get("rng"))
    stream_hashes = {key: state_digest(value) for key, value in streams["states"].items()}
    if (streams["manifest"] != observed_rng or streams["manifest"]["seed"] != 0
            or sorted(streams["states"]) != descriptor["named_stream_keys"]
            or stream_hashes != descriptor["named_stream_state_sha256"]):
        raise ValueError("saved consumed streams differ from their manifest or content hashes")
    source = Path(resolved["request"]["source"]["snapshot_path"]) / "experiments/forge/api.py"
    if file_hash(source) != resolved["request"]["source"]["files"]["experiments/forge/api.py"]:
        raise ValueError("frozen API stream-binding source bytes changed")
    tree = ast.parse(source.read_text())
    node = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "TRAINER_STREAM_BINDINGS" for target in node.targets))
    bindings = ast.literal_eval(node)
    if task_id == "two_pole":
        models = payload["models"]
        direct = payload["role_parameters"]["prior"][0]
        if ("generator" in models or direct["name"] != "direct_particles.0"
                or direct["representation"] != "direct_sample_coordinates"
                or not tags[direct["value"].untyped_storage().data_ptr()].startswith("cuda:")
                or not direct["value"].count_nonzero().item()
                or stable_hash(payload["applied"]) != stable_hash(row["applied"]) or not payload["optimizers"]):
            raise ValueError("direct-coordinate/component provenance is incomplete or changed")
    elif task_id == "five_word_joint_acquisition":
        fixture, policy = payload["fixture"], payload["fixture"]["api_state"]
        models = policy["models"]
        data = [key for key, binding in streams["manifest"]["bindings"].items() if binding["family"] == "data"]
        if (not {"generator", "critic", "encoder", "prior"} <= set(models) or len(policy["optimizers"]) != 2
                or policy["completed_steps"] != descriptor["completed_steps"]
                or policy["completed_steps"] != row["cost"]["completed_steps"]
                or stable_hash(fixture["recipe"]) != stable_hash(row["recipe"])
                or len(data) != 1 or not torch.equal(fixture["data_generator"], streams["states"][data[0]])
                or set(policy["streams"]) != {"latent_generator", "penalty_generator", "eval_generator", "noise_generator"}):
            raise ValueError("word model/optimizer/data and policy-stream provenance is inconsistent")
        for name, value in policy["streams"].items():
            matching = [key for key, binding in streams["manifest"]["bindings"].items()
                        if (binding["family"], binding["component"], binding["purpose"]) == bindings[name]]
            if len(matching) != 1 or not torch.equal(value, streams["states"][matching[0]]):
                raise ValueError("word policy and named-stream state aliases disagree")
    else:
        trainer, models = payload["trainer"], payload["trainer"]["models"]
        if (not trainer["device"].startswith("cuda:") or trainer["completed_steps"] != descriptor["completed_steps"]
                or len(trainer["optimizers"]) != 2
                or any(stable_hash(payload[key]) != stable_hash(row[key]) for key in ("recipe", "prior", "initializer"))):
            raise ValueError("public trainer state changed its CUDA/update/recipe binding")
        required = set(bindings)
        if payload["prior"]["kind"] != "mog":
            required.remove("prior_noise_generator")
        if set(trainer["streams"]) != required:
            raise ValueError("public trainer stream set differs from its frozen API contract")
        for name, value in trainer["streams"].items():
            matching = [key for key, binding in streams["manifest"]["bindings"].items()
                        if (binding["family"], binding["component"], binding["purpose"]) == bindings[name]]
            if len(matching) != 1 or not torch.equal(value, streams["states"][matching[0]]):
                raise ValueError("public trainer and named-stream state aliases disagree")
    tensors = [value for model in models.values() for value in model.values() if isinstance(value, torch.Tensor)]
    if not tensors or any(not tags[value.untyped_storage().data_ptr()].startswith("cuda:") for value in tensors):
        raise ValueError("saved model tensor storage metadata permits a CPU neural fallback")
    if torch.cuda.is_initialized():
        raise ValueError("saved-data audit unexpectedly created a CUDA context")
    return {"task_id": task_id, "attempt_id": attempt_id, "candidate_id": summary["candidate_id"],
            "candidate_revision": summary["candidate_revision"], "recorded_gate_status": row["gate_status"],
            "source_commit": origin, "source_digest": digest,
            "original_files": summary["provenance"]["original_files"],
            "canonical_result_hash": summary["provenance"]["canonical_result_hash"],
            "saved_state_certificate": certificate,
            "recomputed_state_sha256": descriptor["state_sha256"],
            "recomputed_named_stream_state_sha256": stream_hashes,
            "physical_cuda_worker": str(resolved["worker"]["device"]),
            "original_cuda_storage_count": sum(tag.startswith("cuda:") for tag in tags.values()),
            "original_model_tensor_count": len(tensors), "all_original_model_tensor_storages_cuda": True,
            "all_state_and_stream_hashes_match": True,
            "audit_models_constructed": 0, "audit_training_updates": 0, "audit_sampling_draws": 0,
            "audit_cuda_context_created": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-digest", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.attempt, origin=args.source_commit, digest=args.source_digest)
    result["auditor_sha256"] = file_hash(__file__)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
