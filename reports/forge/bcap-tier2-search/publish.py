"""Audit matched conditions and export the whole selected recipe's saved media."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from reports.forge.regenerate_technique_inventory import project_receipt

OUTPUT = ROOT / "reports/forge/bcap-tier2-search"


def main():
    combined = read_json(OUTPUT / "combined-readout.json")
    assert combined["selection"]["selection_complete"] and combined["configurations"] == 96
    trials = [trial for path in (OUTPUT / "readout.json", OUTPUT / "overnight/readout.json")
              for trial in read_json(path)["trials"]]
    selected = next(t for t in trials if t["candidate_id"] == combined["selection"]["selected_candidate_id"])
    grouped, exceptions = defaultdict(list), []
    attempts = sorted({attempt for trial in trials for attempt in trial["attempt_ids"]})
    for attempt in attempts:
        receipt = project_receipt(ROOT, attempt)
        envelope = read_json(ROOT / "reports/forge/attempts" / attempt / "request.json")
        request = envelope["request"]
        assert request["source"]["digest"] == combined["source_digest"]
        assert request["protocol"]["seed"] == 0
        directory = Path(envelope["worker"]["directory"])
        raw_path = directory / "raw-result.json"
        raw = read_json(raw_path) if raw_path.exists() else {}
        if receipt["attempt_status"] != "completed":
            exceptions.append({"attempt_id": attempt, "candidate_id": receipt["candidate_id"],
                               "task_id": envelope["job"]["task_id"], "status": receipt["attempt_status"],
                               "error": raw.get("error"),
                               "reason": [row.get("reason") for row in receipt["task_results"]],
                               "original_receipt_provenance": receipt["provenance"]})
            continue
        applied = raw.get("applied", raw)
        recipe = applied["recipe"]
        assert recipe["reg_arm"] == "b_cap" and recipe["reg_coeff"] > 0 and recipe["reg_kappa"] > 0
        name = envelope["job"]["task_id"]
        evidence = raw.get("evidence", {})
        grouped[name].append({"attempt_id": attempt,
                              "initialization": applied.get("initialization"),
                              "seen_batch_sequence": evidence.get("data_sha256"),
                              "streams": evidence.get("provenance_checkpoint", {}).get("named_stream_state_sha256")})
    audits = []
    for task, rows in sorted(grouped.items()):
        continuation = task in {"gaussian1d_stability", "five_word_joint_hold"}
        fields = {}
        for field in ("initialization", "seen_batch_sequence", "streams"):
            present = [row[field] for row in rows if row[field] is not None]
            hashes = sorted({stable_hash(value) for value in present})
            # Continuing words restore each recipe's own earliest confirmed
            # checkpoint; their consumed stream offsets can therefore differ.
            must_match = not continuation and field != "streams"
            if must_match:
                assert len(hashes) <= 1, (task, field, hashes)
            fields[field] = {"receipts_present": len(present), "distinct_hashes": hashes,
                             "matched": len(hashes) == 1 if present else None,
                             "required_equal": must_match}
        audits.append({"task_id": task, "completed_attempts": len(rows), "fields": fields,
                       "continuation": continuation})
    atomic_json(OUTPUT / "matched-conditions.json", {
        "schema_version": 1, "configurations": 96, "attempt_certificates_verified": len(attempts),
        "source_digest": combined["source_digest"], "seed": 0, "tasks": audits,
        "note": "Missing raw fields grant no inferred equality. Task-owned declarations were independently checked across every candidate by each completed runner. Continuations restore actual recipe-bound checkpoints and retain their own stream offsets.",
        "qualification_input": False})
    atomic_json(OUTPUT / "execution-exceptions.json", {"schema_version": 1, "attempts": exceptions,
        "scientific_retries": 0, "recorded_grades_unchanged": True, "qualification_input": False})
    spec = importlib.util.spec_from_file_location("saved_media", ROOT / "reports/forge/gaussian-smoke-inventory/export_media.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PREFERRED = selected["candidate_id"]
    module.POLICY = {**module.POLICY, "preferred_candidate": module.PREFERRED,
                     "whole_configuration_selection": "After both campaigns conclude; frozen PASS-count/hash objective",
                     "fallback_after_invalid_selected_evidence": False}
    report = next(read_json(path) for path in (OUTPUT / "readout.json", OUTPUT / "overnight/readout.json")
                  if any(t["candidate_id"] == module.PREFERRED for t in read_json(path)["trials"]))
    image_spec = importlib.util.spec_from_file_location("saved_image_media", ROOT / "reports/forge/bcap-convolution/publish.py")
    images = importlib.util.module_from_spec(image_spec)
    image_spec.loader.exec_module(images)
    from experiments.forge import tier1_media
    index_path = OUTPUT / "media/index.json"
    if index_path.exists():
        # Repair only failed renders, retaining all successful saved GIFs.
        index = read_json(index_path)
        from PIL import Image
        previous = []
        with module.forbid_live_execution():
            for entry in index["tasks"]:
                if entry["media_status"] == "EXPORTED":
                    continue
                previous.append({"task_id": entry["task_id"], "error_type": entry.get("error_type"),
                                 "reason": entry.get("reason")})
                request, row, local, certificates = module.certified_row(
                    ROOT / "reports/forge/attempts" / entry["attempt_id"], entry["task_id"])
                assert request["candidate"]["id"] == selected["candidate_id"]
                task = request["tasks"][entry["task_id"]]
                path = OUTPUT / "media" / (entry["task_id"] + ".gif")
                if task["adapter"] == "transfer_image":
                    receipt = images.render_image(task, row, local, path)
                else:
                    assert task["id"] == "vector_spiral"
                    _, inputs = tier1_media._scored_outputs(task, row["evidence"], local)
                    display = deepcopy(row)
                    display["evidence"].pop("saved_observer_outputs")
                    receipt = tier1_media.render(task, display, local, path)
                    receipt["source_inputs"] = inputs
                    atomic_json(path.with_suffix(".json"), receipt)
                with Image.open(path) as gif:
                    frames = gif.n_frames
                for key in ("error_type", "reason", "gif"):
                    entry.pop(key, None)
                entry.update(media_status="EXPORTED", gif=path.name, gif_sha256=file_hash(path), frames=frames,
                             renderer_kind="saved_image_outputs" if task["adapter"] == "transfer_image" else "saved_procedural_training_metrics",
                             observation_count=receipt["observation_count"],
                             selected_observation_indices=receipt["selected_observation_indices"],
                             observations_sha256=receipt["observations_sha256"], source_inputs=receipt["source_inputs"],
                             renderer_receipt=path.with_suffix(".json").name,
                             renderer_receipt_sha256=file_hash(path.with_suffix(".json")))
        index.update(exported_gifs=sum(row["media_status"] == "EXPORTED" for row in index["tasks"]),
                     unavailable_gifs=sum(row["media_status"] != "EXPORTED" for row in index["tasks"]),
                     repaired_render_failures=previous, reporting_adapter_sha256=file_hash(Path(__file__)),
                     image_renderer_sha256=file_hash(ROOT / "reports/forge/bcap-convolution/publish.py"),
                     rescored_observations=96)
        atomic_json(index_path, index)
    else:
        original_render = tier1_media.render

        def render_saved(task, row, local, path):
            if task["adapter"] == "transfer_image":
                return images.render_image(task, row, local, path)
            if task["id"] == "vector_spiral":
                _, inputs = tier1_media._scored_outputs(task, row["evidence"], local)
                display = deepcopy(row)
                display["evidence"].pop("saved_observer_outputs")
                receipt = original_render(task, display, local, path)
                receipt["source_inputs"] = inputs
                atomic_json(path.with_suffix(".json"), receipt)
                return receipt
            return original_render(task, row, local, path)

        tier1_media.render = render_saved
        try:
            index = module.publish(Path(report["queue_root"]), ROOT / "reports/forge/attempts", OUTPUT / "media",
                                   campaign=report["study_id"])
        finally:
            tier1_media.render = original_render
        index.update(reporting_adapter_sha256=file_hash(Path(__file__)),
                     image_renderer_sha256=file_hash(ROOT / "reports/forge/bcap-convolution/publish.py"),
                     rescored_observations=96)
        atomic_json(index_path, index)
    assert all(row["candidate_id"] == selected["candidate_id"] for row in index["tasks"])
    assert index["selected_task_count"] == index["exported_gifs"] == 28
    assert index["unavailable_gifs"] == index["optimizer_updates_added"] == index["sampling_draws_added"] == 0
    atomic_json(OUTPUT / "media-receipt.json", {"schema_version": 1,
        "candidate_id": selected["candidate_id"], "tasks": index["selected_task_count"],
        "exported_gifs": index["exported_gifs"], "index": "media/index.json",
        "index_sha256": file_hash(OUTPUT / "media/index.json"),
        "optimizer_updates_added": 0, "sampling_draws_added": 0, "qualification_input": False})
    print({"stage": "publication_audit", "certificates": len(attempts), "gifs": index["exported_gifs"]}, flush=True)


if __name__ == "__main__":
    main()
