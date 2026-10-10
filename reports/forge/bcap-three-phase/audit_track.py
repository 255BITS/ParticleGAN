"""Read-only independent audit of one completed phase-three publication.

Run in a separate process per track, so its frozen public API owns imports.
This reads certified saved states; it performs no training or sampling.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authorized_retries(state, collection, authorization):
    """Validate explicitly authorized infrastructure retries without regrading."""
    from experiments.forge.contracts import stable_hash
    allowed = {item["predecessor_attempt_id"]: item for item in authorization.get("retries", [])}
    verified = []
    for job in state["jobs"].values():
        attempts = job.get("attempts", [])
        for previous, current in zip(attempts, attempts[1:]):
            predecessor = collection["attempts"][previous["attempt_id"]]["result"]
            successor = collection["attempts"][current["attempt_id"]]["result"]
            binding = successor.get("retry_of", {})
            approval = allowed.get(predecessor["attempt_id"], {})
            if not (predecessor["raw"]["attempt_status"] in {"error", "timeout", "cancelled"}
                    and all(row["gate_status"] == "INCOMPLETE" for row in predecessor["task_results"])
                    and binding.get("attempt_id") == predecessor["attempt_id"]
                    and binding.get("result_hash") == stable_hash(predecessor)
                    and approval.get("predecessor_result_hash") == stable_hash(predecessor)
                    and binding.get("reason") == approval.get("reason")
                    and predecessor["candidate_revision"] == successor["candidate_revision"]):
                raise ValueError("Retry must bind an explicitly authorized incomplete execution predecessor")
            verified.append(dict(predecessor=predecessor["attempt_id"], successor=successor["attempt_id"],
                                 reason=binding["reason"]))
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--publication", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    root, artifacts, publication = (path.resolve() for path in
                                    (options.repository, options.artifacts, options.publication))
    sys.path.insert(0, str(root))
    from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
    from experiments.forge.planning import FORMULATION_FIELDS, load_idea
    from PIL import Image

    module = load("independent_phase3", root / "reports/forge/bcap-three-phase/phase3.py")
    require = module.require
    registration = read_json(options.registration)
    result = read_json(publication / "phase3-results.json")
    progress = read_json(artifacts / "phase3-progress.json")
    require(progress["phase"] == "diagnostic_complete", "All admitted work must be terminal")
    require(result["source_commit"] == progress["source_commit"]
            and result["source_digest"] == registration["source_digest"]
            and result["registration_sha256"] == file_hash(options.registration),
            "Published source and admission identities differ")
    subprocess.run(["git", "merge-base", "--is-ancestor",
                    "9005ed73a09a740174f590fbfd543440935040eb", result["source_commit"]],
                   cwd=root, check=True)
    requests = module.resolved(root, registration)
    chosen = load_idea(root, "bcap-three-phase-incumbent-v1")
    baseline = requests["baseline"]["candidate"]
    require(all(baseline.get(field) == chosen.get(field) for field in FORMULATION_FIELDS),
            "Control formulation must exactly inherit the chosen phase-two incumbent")

    publisher = module.phase2.publisher(root)
    publisher.ROLES = module.ROLES
    publisher.required_questions = lambda _root: registration["original_requirements"]
    publisher.scopes = lambda _options: module.diagnostic_scopes(
        publisher, artifacts, artifacts / "phase3-progress.json")
    collection = publisher.collect(argparse.Namespace(repository=root, allow_partial=False))
    availability = load("independent_publication_availability", Path(__file__).with_name("publication_adapter.py"))
    availability.annotate(collection)
    require(result["task_cells"] == collection["cells"]
            and result["task_results"] == [entry["item"] for entry in collection["final"]]
            and result["accounting"] == collection["accounting"],
            "Published metrics or costs differ from certified attempts")
    audit = module.phase2.audit_saved_comparison(root, collection, module.ROLES)
    own = load("independent_own_checkpoints", root / "reports/forge/bcap-develop-integration/audit.py")
    producers = own.own_checkpoints(collection)
    require(read_json(publication / "phase3-audit.json")["own_checkpoint_producers"] == producers,
            "Published producer continuity differs from consumed checkpoint bytes")
    require(audit["status"] == "PASS", "Saved comparison audit failed")

    media = read_json(publication / "media/index.json")["media"]
    completed = {(entry["item"]["role"], entry["item"]["task_id"], entry["item"]["attempt_id"])
                 for entry in collection["final"] if entry["row"]["gate_status"] in {"PASS", "FAIL"}}
    require({(item["role"], item["task_id"], item["attempt_id"]) for item in media} == completed
            and len(media) == len(completed) == result["actual_training_gifs"],
            "Every completed gate needs exactly its own saved-training GIF")
    for item in media:
        path = (publication / item["gif"]).resolve()
        require(path.is_relative_to(publication) and file_hash(path) == item["gif_sha256"],
                "Saved-training GIF bytes differ")
        with Image.open(path) as image:
            require(image.n_frames == item["frames"] >= 2, "Saved-training frame count differs")
        require(item["optimizer_updates_added"] == item["sampling_draws_added"] == 0,
                "Publication cannot execute new updates or draws")

    state = read_json(artifacts / "phase3-queue/queue/state.json")
    require(all(not accounting["pending_work"] for accounting in collection["accounting"]), "No active work")
    authorization_path = artifacts.parents[1] / "restart-20261010/authorized-retries.json"
    authorization = read_json(authorization_path) if authorization_path.exists() else {}
    retries = authorized_retries(state, collection, authorization)
    require(sum(accounting["execution_retries"] for accounting in collection["accounting"]) == len(retries),
            "Every execution retry requires an explicit predecessor and authorization")
    paid = sum(accounting["selected_paid_seconds"] for accounting in collection["accounting"])
    require(paid <= 48000 and result["full_reservation_seconds"] == 45840
            and result["paid_ceiling_seconds"] == 48000, "Declared paid budget exceeded")
    protected = read_json(artifacts.parents[1] / "historical-publication-before.json")["files"]
    require(all(file_hash(root / path) == digest for path, digest in protected.items()),
            "Historical qualifications, memory or goal inventory changed")
    proof = dict(status="PASS", scope="independent_phase3_saved_evidence_audit", qualification_input=False,
        repository=str(root), publication=str(publication), source_commit=result["source_commit"],
        source_digest=result["source_digest"], exact_chosen_baseline=True,
        original_task_count=len(registration["task_ids"]), original_placement_counts=dict(tier1=6, tier2=10),
        outcomes={role: dict(Counter(cell["gate_status"] for cell in collection["cells"]
                                    if cell["role"] == role)) for role in module.ROLES},
        task_results_sha256=file_hash(publication / "phase3-results.json"),
        saved_state_comparisons=audit["saved_state_comparisons"],
        unavailable_complete_states=audit["unavailable_complete_states"],
        source_checks=audit["frozen_source_checks"], own_checkpoint_producers=producers,
        actual_training_gifs=len(media), paid_seconds=paid,
        accounting=collection["accounting"], protected_historical_files=len(protected),
        queue_jobs=len(state["jobs"]), authorized_execution_retries=retries,
        optimizer_updates_added=0, sampling_draws_added=0)
    atomic_json(options.output, proof)
    print(json.dumps({key: proof[key] for key in
                      ("status", "source_commit", "outcomes", "actual_training_gifs", "paid_seconds")}))


if __name__ == "__main__":
    main()
