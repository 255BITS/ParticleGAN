"""Publish one saved-training GIF per executed task, without live execution.

Selection is independent of grades: the selected BCAP dualnorm configuration
comes first, then lexicographic candidate/revision/attempt identity. Invalid or
nonfinite selected evidence is recorded as unavailable; no better-looking
candidate is substituted. The existing Forge renderer supplies all frames.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, identifier, read_json, stable_hash

PREFERRED = "bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9"
POLICY = dict(version=1, preferred_candidate=PREFERRED,
              ordering=["preferred_candidate_first", "candidate_id", "candidate_revision", "attempt_id"],
              use_grades_for_selection=False, fallback_after_invalid_selected_evidence=False)


def selections(state, campaign):
    """Read an already completed campaign; never acquire or mutate its queue."""
    requests = {key: entry["request"] for key, entry in state["submissions"].items()
                if entry["request"]["campaign_id"] == campaign}
    if not requests:
        raise ValueError("campaign has no admitted submissions")
    active = [key for key in requests if state["submissions"][key]["status"] in {"queued", "running"}]
    if active:
        raise ValueError("campaign is still active; publish only after the complete drain")
    by_task, required = {}, set()
    for request in requests.values():
        required.update(point["task"] for point in request["view"]["assignments"]
                        if point.get("importance", "required") == "required"
                        and point["qualification_tier"] <= request["through_tier"])
    for job in state["jobs"].values():
        if not set(job.get("subscribers", [])) & requests.keys() or not job.get("result"):
            continue
        result = job["result"]
        owner = result.get("cost_owner", job.get("cost_owner"))
        request = state["submissions"][owner["request"]]["request"]
        for row in result["task_results"]:
            task = row["task_id"]
            item = dict(task_id=task, candidate_id=request["candidate"]["id"],
                        candidate_revision=result["candidate_revision"], attempt_id=result["attempt_id"],
                        recorded_grade=row["gate_status"], raw_status=row.get("raw_status"),
                        qualification_tier=next(point["qualification_tier"] for point in request["view"]["assignments"]
                                                if point["task"] == task),
                        source_commit=request["source"].get("origin_commit"), source_digest=request["source"]["digest"])
            by_task.setdefault(task, {})[(item["candidate_id"], item["candidate_revision"], item["attempt_id"])] = item
    selected = []
    for task, candidates in sorted(by_task.items()):
        chosen = min(candidates.values(), key=lambda item: (item["candidate_id"] != PREFERRED,
                     item["candidate_id"], item["candidate_revision"], item["attempt_id"]))
        selected.append({**chosen, "executed_candidate_count": len({item["candidate_id"] for item in candidates.values()})})
    return selected, sorted(required - by_task.keys())


def certified_row(directory, task_id):
    """Retain export_attempt's complete source/result/candidate certification."""
    envelope, result, certificate = [read_json(directory / (name + ".json"))
                                     for name in ("request", "result", "evidence")]
    request = envelope.get("request", envelope)
    if certificate["result_hash"] != stable_hash(result) or certificate["source"] != request["source"]:
        raise ValueError("original source/result certificate differs")
    if result["candidate_revision"] != request["candidate_revision"]:
        raise ValueError("candidate revision differs")
    if result["attempt_id"] != directory.name:
        raise ValueError("attempt directory differs from certified identity")
    local = Path(certificate["local_artifact_root"])
    if read_json(local / "result.json") != result:
        raise ValueError("local result differs from certified envelope")
    row = next(point for point in result["task_results"] if point["task_id"] == task_id)
    return request, row, local, {name + ".json": file_hash(directory / (name + ".json"))
                               for name in ("request", "result", "evidence")}


def finite_tree(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return not np.issubdtype(value.dtype, np.number) or bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(child) for child in value)
    return True


@contextmanager
def forbid_live_execution():
    """Fail visibly if a future renderer tries to construct or query a model."""
    import torch
    from particlegan import GANTrainer, ParticlePrior, MoGParticlePrior
    def forbidden(*args, **kwargs):
        raise RuntimeError("media publication forbids model construction, training and sampling")
    methods = [(torch.nn.Module, "__init__"), (torch.nn.Module, "__call__"),
               (GANTrainer, "__init__"), (GANTrainer, "step"), (GANTrainer, "sample"),
               (ParticlePrior, "sample"), (MoGParticlePrior, "sample")]
    originals = [(owner, name, getattr(owner, name)) for owner, name in methods]
    try:
        for owner, name, _ in originals:
            setattr(owner, name, forbidden)
        yield
    finally:
        for owner, name, method in reversed(originals):
            setattr(owner, name, method)


def publish(queue_root, attempts, output, *, campaign):
    import torch
    from PIL import Image
    from experiments.forge import tier1_media
    state_path = queue_root / "queue/state.json"
    state = read_json(state_path)
    chosen, unmeasured = selections(state, campaign)
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(schema_version=1, campaign=campaign, policy=POLICY, selected_tasks=chosen,
                required_unmeasured_task_ids=unmeasured, queue_state_sha256=stable_hash(state),
                optimizer_updates_added=0, sampling_draws_added=0, qualification_input=False)
    # Write the selected identities before loading saved samples or rendering.
    atomic_json(output / "selection.json", plan)
    entries = []
    renderer_hash = file_hash(Path(tier1_media.__file__))
    with forbid_live_execution():
        for choice in chosen:
            entry = dict(choice)
            identifier(choice["task_id"], "task id")
            path = output / (choice["task_id"] + ".gif")
            try:
                request, row, local, certificates = certified_row(attempts / choice["attempt_id"], choice["task_id"])
                if (request["candidate"]["id"] != choice["candidate_id"]
                        or request["candidate_revision"] != choice["candidate_revision"]
                        or request["source"]["digest"] != choice["source_digest"]
                        or request["source"].get("origin_commit") != choice["source_commit"]
                        or row["gate_status"] != choice["recorded_grade"]):
                    raise ValueError("selected queue identity differs from the certified attempt")
                entry["certificate_sha256"] = certificates
                if request["source"]["files"].get("experiments/forge/tier1_media.py") != renderer_hash:
                    raise ValueError("renderer source differs; use the executed checkout")
                evidence = row.get("evidence")
                if not isinstance(evidence, dict) or not evidence:
                    raise ValueError("executed attempt retained no renderable evidence")
                if evidence.get("guards", {}).get("all_finite") is False or not finite_tree(evidence.get("observations", [])):
                    raise FloatingPointError("nonfinite recorded observations or finite-state guard")
                task = request["tasks"][choice["task_id"]]
                samples, _ = tier1_media._scored_outputs(task, evidence, local)
                if samples is not None and not finite_tree(samples):
                    raise FloatingPointError("nonfinite retained scored outputs")
                if task["adapter"] == "clockfree_audit":
                    proof = torch.load(Path(evidence["artifact_root"]) / "comparisons.pt", map_location="cpu", weights_only=True)
                    if not finite_tree(proof):
                        raise FloatingPointError("nonfinite saved clock-free comparison states")
                receipt = tier1_media.render(task, row, local, path)
                with Image.open(path) as gif:
                    frames = gif.n_frames
                entry.update(media_status="EXPORTED", gif=path.name, gif_sha256=file_hash(path), frames=frames,
                             observation_count=receipt["observation_count"],
                             selected_observation_indices=receipt["selected_observation_indices"],
                             observations_sha256=receipt["observations_sha256"],
                             renderer_receipt=path.with_suffix(".json").name,
                             renderer_receipt_sha256=file_hash(path.with_suffix(".json")),
                             source_inputs=receipt["source_inputs"])
            except Exception as error:
                entry.update(media_status="NONFINITE_EVIDENCE" if isinstance(error, FloatingPointError) else "UNAVAILABLE_OR_INVALID",
                             error_type=type(error).__name__, reason=str(error), gif=None,
                             recorded_grade_unchanged=True, fallback_candidate_used=False)
                # Keep failed output outside publication; its gate stays visible.
                for possible in (path, path.with_suffix(".json")):
                    if possible.exists():
                        possible.unlink()
            entries.append(entry)
            print(json.dumps({key: entry.get(key) for key in ("task_id", "candidate_id", "recorded_grade", "media_status", "reason")}), flush=True)
    index = dict(schema_version=1, campaign=campaign, scope="one_actual_training_visualization_per_executed_task",
                 qualification_input=False, selection_policy=POLICY,
                 source_commits=sorted({item["source_commit"] for item in chosen if item["source_commit"]}),
                 selected_task_count=len(chosen), exported_gifs=sum(item["media_status"] == "EXPORTED" for item in entries),
                 unavailable_gifs=sum(item["media_status"] != "EXPORTED" for item in entries),
                 required_unmeasured_task_ids=unmeasured, tasks=entries,
                 optimizer_updates_added=0, sampling_draws_added=0, rescored_observations=0,
                 selection_sha256=file_hash(output / "selection.json"), script_sha256=file_hash(Path(__file__)),
                 renderer_sha256=renderer_hash, live_execution_guard="Model construction/forward, GANTrainer step/sample and prior sampling disabled during export.")
    atomic_json(output / "index.json", index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", default="gaussian-smoke-inventory-v2")
    args = parser.parse_args()
    result = publish(args.queue_root.resolve(), args.attempts.resolve(), args.output.resolve(), campaign=args.campaign)
    print(json.dumps({key: result[key] for key in ("campaign", "selected_task_count", "exported_gifs", "unavailable_gifs")}), flush=True)


if __name__ == "__main__":
    main()
