"""Publish the verified standalone ring16 endpoint/GIF, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .api_publish import verify_run
from .api_ring16 import list_cases
from .api_run import file_hash, write_json


def publish(raw, output):
    raw, output = Path(raw), Path(output)
    receipt = verify_run(raw)
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
                                for observation in receipt["observations"][-5:]],
           "execution_elapsed_seconds": receipt["elapsed_seconds"],
           "gif": "goal.gif", "gif_sha256": receipt["artifacts"]["goal.gif"]["sha256"],
           "frames": receipt["gif_frames"], "raw_receipt_sha256": file_hash(raw / "receipt.json"),
           "raw_artifacts": receipt["artifacts"], "qualification_input": False}
    output.mkdir(parents=True, exist_ok=True)
    target = output / "goal.gif"
    if target.exists() and file_hash(target) != row["gif_sha256"]:
        raise ValueError("refusing to replace a different published actual-training GIF")
    shutil.copyfile(raw / "goal.gif", target)
    write_json(output / "publication.json", {
        "schema": "particlegan_api_toy_supplement_v1", "cases": [case],
        "readouts": [{"id": case["id"], "goal": case["goal"], "scope": case["scope"],
                      "execution_status": receipt["status"], "verdict": receipt["verdict"],
                      "completed_updates": receipt["completed_updates"], "default_updates": case["default_steps"],
                      "failed_bounds": receipt["failed_bounds"], "gif": row["gif"],
                      "source_commit": receipt["source"]["commit"]}],
        "runs": [row], "source_identities": {source_key: receipt["source"]},
        "training_or_rescoring_by_publication": False, "historical_receipts_changed": False})
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args(argv)
    row = publish(options.raw, options.output)
    print(json.dumps({"verdict": row["verdict"], "completed_updates": row["completed_updates"],
                      "execution_elapsed_seconds": row["execution_elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
