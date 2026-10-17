"""Frozen depth-only task variant on the shared Gaussian public-API host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiments.forge.contracts import file_hash
from experiments.forge.sources import inspect_source

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/gaussian-shallow/protocol.json"


def declaration():
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "ready":
        raise ValueError("depth study must be frozen before execution")
    for name, digest in {**protocol["inputs"], **protocol["scientific_implementation"]}.items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("frozen depth study input changed: " + name)
    if inspect_source(ROOT)["digest"] != protocol["scientific_source_digest"]:
        raise ValueError("frozen numerical source changed; declare a new study")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    protocol = declaration()
    if not args.device.startswith("cuda"):
        raise ValueError("the depth study requires CUDA; CPU fallback is forbidden")
    if args.output.exists():
        raise ValueError("a new raw directory is required; scientific retries are not admitted")
    command = [sys.executable, "-u", "-m", "benchmarks.toy_audit.gaussian_smoke_study",
               "--task", protocol["task_path"], "--candidate", protocol["candidate_path"],
               "--output", str(args.output), "--device", args.device, "--through-stability",
               "--stability-task", protocol["stability_task_path"]]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
