"""Publish the verified standalone ring16 endpoint/GIF, without training."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
from PIL import Image

from .api_publish import _verify_grade, verify_run
from .api_ring16 import list_cases
from .api_run import file_hash, render_gif, write_json


MEDIA_ERROR = "goal media/state error: ModuleNotFoundError: No module named 'matplotlib'"


def verify_rendering_failure(receipt):
    """Validate the completed numeric FAIL; retain the original ERROR stamp."""
    if (receipt.get("status") != "ERROR" or receipt.get("verdict") != "FAIL"
            or receipt.get("passed") is not False or receipt.get("source_unchanged") is not True
            or receipt.get("default_protocol_complete") is not True
            or receipt.get("gif_frames") != 0 or "goal.gif" in receipt.get("artifacts", {})
            or receipt.get("failed_bounds", []).count(MEDIA_ERROR) != 1):
        raise ValueError("recovery requires the exact completed numeric FAIL with missing renderer dependency")
    # Validation copy only: the original receipt and its ERROR/FAIL flags never
    # change. The normal verifier rederives the declared cadence/terminal FAIL.
    numeric = deepcopy(receipt)
    numeric["failed_bounds"].remove(MEDIA_ERROR)
    numeric["gif_frames"] = len(numeric["protocol"]["media_steps"])
    return _verify_grade(numeric)


def recover_media(raw, output):
    from .api_reframe import _observation_identity, reconstruct_media, renderer_source
    receipt = json.loads((raw / "receipt.json").read_text())
    steps = verify_rendering_failure(receipt)
    before = {name: file_hash(raw / name) for name in ("receipt.json", "observations.npz", "final-state.pt")}
    for name in ("observations.npz", "final-state.pt"):
        declared = receipt["artifacts"][name]
        if before[name] != declared["sha256"] or (raw / name).stat().st_size != declared["bytes"]:
            raise ValueError("retained raw artifact identity differs from original receipt")
    with np.load(raw / "observations.npz", allow_pickle=False) as arrays:
        expected = {f"step{o['step']}_view{i}_{role}"
                    for o in receipt["observations"] for i in range(len(o["views"]))
                    for role in ("target", "samples")}
        if set(arrays.files) != expected or any(
                not np.isfinite(arrays[name]).all() or len(arrays[name]) not in (16, 4096)
                for name in arrays.files):
            raise ValueError("recovery requires all original finite target/output observations")
    records = reconstruct_media(raw, receipt, steps=steps)
    identity = _observation_identity(records)
    source = renderer_source()
    source["files_sha256"]["benchmarks/toy_audit/ring16_publish.py"] = file_hash(Path(__file__))
    source["python"] = platform.python_version()
    source["matplotlib"] = __import__("matplotlib").__version__
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "goal.gif"
    if artifact.exists():
        raise ValueError("recovery needs an unused goal GIF destination")
    annotations = render_gif(receipt["case"], records, artifact, full_budget=True,
                             requested_steps=400, final_verdict="ERROR / FAIL")
    with Image.open(artifact) as gif:
        frames = gif.n_frames
    if (frames != len(steps) or _observation_identity(records) != identity
            or before != {name: file_hash(raw / name) for name in before}):
        raise ValueError("recovery changed original evidence or lost an actual frame")
    review = dict(original_execution_status="ERROR", numeric_verdict="FAIL", displayed_verdict="ERROR / FAIL",
                  original_failed_bounds=receipt["failed_bounds"], raw_sha256=before,
                  renderer_source=source, media_steps=steps, annotations=annotations,
                  numeric_observation_sha256=identity, raw_files_unchanged=True,
                  qualification_upgrade=False, training_or_rescoring=False)
    return receipt, dict(sha256=file_hash(artifact), bytes=artifact.stat().st_size), frames, review


def publish(raw, output, *, recover_missing_media=False):
    raw, output = Path(raw), Path(output)
    if recover_missing_media:
        receipt, media, frames, review = recover_media(raw, output)
    else:
        receipt = verify_run(raw)
        media, frames, review = receipt["artifacts"]["goal.gif"], receipt["gif_frames"], None
    if receipt["case"] != list_cases()[0] or receipt["seed"] != 0:
        raise ValueError("execution differs from the frozen ring16 declaration")
    source_key = hashlib.sha256(json.dumps(receipt["source"], sort_keys=True).encode()).hexdigest()
    case = receipt["case"]
    row = {"id": case["id"], "recipe": receipt["recipe"], "runtime": receipt["runtime"],
           "source_identity": source_key, "protocol": receipt["protocol"], "seed": receipt["seed"],
           "api_components": receipt["api_components"], "initialization": case["initialization"],
           "prior_options": case["prior_options"], "sampling": case["sampling"],
           "completed_updates": receipt["completed_updates"],
           "default_protocol_complete": receipt["default_protocol_complete"],
           "metric_passed": receipt["metric_passed"],
           "sustained_metric_passed": receipt["sustained_metric_passed"],
           "verdict": receipt["verdict"], "failed_bounds": receipt["failed_bounds"],
           "final_metrics": receipt["observations"][-1]["metrics"],
           "terminal_metrics": [{key: observation[key] for key in ("step", "metrics", "passed", "failed_bounds")}
                                for observation in [o for o in receipt["observations"] if o["step"] > 0
                                    and o["step"] in receipt["protocol"]["metric_evaluation_steps"]][-5:]],
           "execution_elapsed_seconds": receipt["elapsed_seconds"],
           "gif": "goal.gif", "gif_sha256": media["sha256"],
           "frames": frames, "raw_receipt_sha256": file_hash(raw / "receipt.json"),
           "raw_artifacts": receipt["artifacts"], "qualification_input": False}
    output.mkdir(parents=True, exist_ok=True)
    target = output / "goal.gif"
    if target.exists() and file_hash(target) != row["gif_sha256"]:
        raise ValueError("refusing to replace a different published actual-training GIF")
    if not recover_missing_media:
        shutil.copyfile(raw / "goal.gif", target)
    write_json(output / "publication.json", {
        "schema": "particlegan_api_toy_supplement_v1", "cases": [case],
        "readouts": [{"id": case["id"], "goal": case["goal"], "scope": case["scope"],
                      "execution_status": receipt["status"], "verdict": receipt["verdict"],
                      "completed_updates": receipt["completed_updates"], "default_updates": case["default_steps"],
                      "failed_bounds": receipt["failed_bounds"], "gif": row["gif"],
                      "source_commit": receipt["source"]["commit"]}],
        "runs": [row], "source_identities": {source_key: receipt["source"]},
        "media_recovery": review,
        "training_or_rescoring_by_publication": False, "historical_receipts_changed": False})
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recover-missing-media", action="store_true")
    options = parser.parse_args(argv)
    row = publish(options.raw, options.output, recover_missing_media=options.recover_missing_media)
    print(json.dumps({"verdict": row["verdict"], "completed_updates": row["completed_updates"],
                      "execution_elapsed_seconds": row["execution_elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
