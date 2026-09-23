"""Hold the formulation fixed and change physical units of the ring data."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from . import mode_hold
from .baseline import Candidate, digest, protocol, run_toy, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new path")
    configs = json.loads(args.config.read_text())
    if len(configs) != 1:
        parser.error("provide one fixed candidate")
    candidate = Candidate(**configs[0])
    torch.set_num_threads(1)
    fingerprint = protocol()
    report = {"protocol": fingerprint, "protocol_sha256": digest(fingerprint),
              "config": asdict(candidate), "scales": [0.5, 1.0, 2.0],
              "description": "Scale centers and data noise together; HQ radius stays three physical sigmas. No normalization or cap retuning.",
              "rows": []}
    original_means, original_diversity = mode_hold.ring_means, mode_hold.diversity
    original_sigma = mode_hold.SIGMA
    for scale in report["scales"]:
        print(f"START scale={scale}", flush=True)
        start = time.monotonic()
        def means():
            return original_means(radius=mode_hold.RADIUS * scale)
        def diversity(samples, centers, *, detailed=False):
            return original_diversity(samples, centers, sigma=original_sigma * scale, detailed=detailed)
        with patch.object(mode_hold, "ring_means", means), patch.object(mode_hold, "SIGMA", original_sigma * scale), patch.object(mode_hold, "diversity", diversity):
            result = run_toy("mode_hold", candidate)
        report["rows"].append({"scale": scale, "radius": mode_hold.RADIUS * scale,
                               "sigma": original_sigma * scale, "result": result,
                               "seconds": time.monotonic() - start})
        write_json(args.output, report)
        print(f"DONE scale={scale} live={result['live']} convergence={result['convergence']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
