"""Copy original certified payload bytes; retain the original self receipt."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash


def migrate(original, output):
    original, output = original.resolve(), output.resolve()
    interrupted = json.loads((original / "interruption.json").read_text())
    receipt_path = original / "smoke/adapter-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert file_hash(receipt_path) == interrupted["parent_receipt_sha256"]
    assert receipt["cost"]["completed_steps"] == 1000
    assert file_hash(original / "smoke/state.pt") == interrupted["parent_state_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    for name in ("request.json", "source.json"):
        shutil.copy2(original / name, output / name)
    certified = output / "smoke/evaluator"
    certified.mkdir(parents=True)
    manifest = receipt["evidence"]["artifact_manifest"]
    for name in manifest["files"]:
        target = certified / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / "smoke" / name, target)
    # Complete tree verification stays strict; no extra file is ignored.
    verify_artifacts(certified, manifest)
    shutil.copy2(receipt_path, output / "smoke/original-adapter-receipt.json")
    relocated = deepcopy(receipt)
    relocated["evidence"]["artifact_root"] = str(certified)
    before, after = deepcopy(receipt), deepcopy(relocated)
    before["evidence"].pop("artifact_root")
    after["evidence"].pop("artifact_root")
    assert before == after
    atomic_json(output / "smoke/adapter-receipt.json", relocated)
    proof = dict(schema_version=1, training_updates=0, repeated_prefix_updates=0,
                 original_raw=str(original), relocated_raw=str(output),
                 original_receipt_sha256=file_hash(receipt_path),
                 relocated_receipt_sha256=file_hash(output / "smoke/adapter-receipt.json"),
                 original_state_sha256=file_hash(original / "smoke/state.pt"),
                 relocated_state_sha256=file_hash(certified / "state.pt"),
                 unchanged_artifact_manifest=manifest,
                 allowed_receipt_delta="evidence.artifact_root only; payload/grades/costs unchanged",
                 strict_complete_tree_verified=True)
    atomic_json(output / "prefix-migration.json", proof)
    print(json.dumps({key: proof[key] for key in ("training_updates", "repeated_prefix_updates",
                      "original_receipt_sha256", "original_state_sha256", "strict_complete_tree_verified")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    migrate(arguments.original, arguments.output)
