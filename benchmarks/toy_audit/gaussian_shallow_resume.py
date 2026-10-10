"""Resume only the declared 5,000 updates after the artifact manifest repair."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiments.forge.contracts import file_hash
from experiments.forge.sources import inspect_source

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/gaussian-shallow/continuation-protocol.json"


def declaration(raw):
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "ready":
        raise ValueError("continuation amendment is not frozen")
    for path, digest in protocol["inputs"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError("frozen continuation input changed: " + path)
    if inspect_source(ROOT)["digest"] != protocol["scientific_source_digest"]:
        raise ValueError("frozen continuation source changed")
    migrated = json.loads((raw / "prefix-migration.json").read_text())
    if (migrated["original_receipt_sha256"] != protocol["parent_receipt_sha256"]
            or migrated["original_state_sha256"] != protocol["parent_state_sha256"]
            or file_hash(raw / "smoke/original-adapter-receipt.json") != protocol["parent_receipt_sha256"]
            or file_hash(raw / "smoke/evaluator/state.pt") != protocol["parent_state_sha256"]):
        raise ValueError("continuation must preserve the original exact prefix")
    if (raw / "stability").exists():
        raise ValueError("continuation already started; scientific retries are not admitted")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    protocol = declaration(args.output)
    if not args.device.startswith("cuda"):
        raise ValueError("CUDA is required; CPU fallback is forbidden")
    subprocess.run([sys.executable, "-u", "-m", "benchmarks.toy_audit.gaussian_smoke_study",
                    "--task", protocol["task_path"], "--candidate", protocol["candidate_path"],
                    "--stability-task", protocol["stability_task_path"], "--output", str(args.output),
                    "--device", args.device, "--through-stability", "--resume-existing"],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
