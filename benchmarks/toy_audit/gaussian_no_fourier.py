"""Frozen Fourier-only Gaussian ablation using the shared public trainer host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from experiments.forge.contracts import file_hash
from experiments.forge.sources import inspect_source

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/gaussian-no-fourier/protocol.json"


def declaration():
    protocol = json.loads(PROTOCOL.read_text())
    for name, expected in protocol["inputs"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("frozen ablation input changed: " + name)
    if inspect_source(ROOT)["digest"] != protocol["source_digest"]:
        raise ValueError("frozen scientific source changed; use the declared commit")
    return protocol


def run(output, *, device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Fourier ablation requires CUDA; no CPU fallback")
    protocol = declaration()
    if output.exists():
        raise ValueError("raw output already exists; scientific retries are forbidden")
    command = [sys.executable, "-u", "-m", "benchmarks.toy_audit.gaussian_smoke_study",
               "--task", protocol["smoke_task"], "--stability-task", protocol["stability_task"],
               "--candidate", protocol["candidate"], "--output", str(output),
               "--device", device, "--through-stability"]
    print(json.dumps({"event": "start", "protocol": protocol["id"],
                      "protocol_sha256": file_hash(PROTOCOL), "command": command}), flush=True)
    result = subprocess.run(command, cwd=ROOT, timeout=protocol["budget"]["reserved_seconds"] + 60)
    if result.returncode:
        raise RuntimeError("shared host execution failed; no retry")
    print(json.dumps({"event": "complete", "protocol": protocol["id"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    run(arguments.output, device=arguments.device)


if __name__ == "__main__":
    main()
