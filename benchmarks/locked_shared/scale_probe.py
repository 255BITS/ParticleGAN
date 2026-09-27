"""Hold the recipe fixed and change the physical units of the ring data.

    python -m benchmarks.locked_shared.scale_probe --output runs/locked_scale_probe/results.json

Each scale is a problem variant (ring centers and data noise scaled together;
the HQ radius stays three physical sigmas). Training comes from the ring's
recipe on ``benchmarks.toy_runner``; nothing is normalized or retuned.
Observations stream to --log as JSON lines (``tail -f``).
"""

import argparse
import json
from pathlib import Path
import time

import torch

from benchmarks.toy_runner import run
from . import mode_hold
from .baseline import BUDGETS, digest, protocol, write_json
from .observation import checkpoint, recording, sustained

SCALES = (0.5, 1.0, 2.0)
# Ring convergence requires every mode, as in the behavioral baseline.
CONVERGENCE = [("modes", ">=", mode_hold.N_MODES), ("hq", ">=", mode_hold.PASS_HQ)]


class ScaledRing(mode_hold.ModeHold):
    """The 8-mode ring with centers and noise multiplied by ``scale``."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.name = f"mode_hold_x{self.scale:g}"
        self.means = mode_hold.ring_means(radius=mode_hold.RADIUS * self.scale)
        self.sigma = mode_hold.SIGMA * self.scale

    def real(self, n, stream):
        return mode_hold.sample_ring(self.means, n, self.sigma, stream)

    def metrics(self, model):
        return mode_hold.diversity(model.sample(mode_hold.EVAL_N).x, self.means, sigma=self.sigma)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=Path("runs/toy-refactor/locked_scale_probe.log"))
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    if args.output.exists():
        parser.error("output exists; choose a new path")
    torch.set_num_threads(1)
    fingerprint = protocol()
    report = {"protocol": fingerprint, "protocol_sha256": digest(fingerprint), "scales": list(SCALES),
              "description": "Scale centers and data noise together; HQ radius stays three physical sigmas. "
                             "The ring's recipe is unchanged.",
              "rows": []}
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("")

    def log(row):
        with args.log.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False, default=float) + "\n")

    for scale in SCALES:
        print(f"START scale={scale}", flush=True)
        start = time.monotonic()
        problem = ScaledRing(scale)
        with recording(BUDGETS["mode_hold"]) as recorder:
            result = run(problem, observe_every=50, log=log, observer=checkpoint)
        row = {"scale": scale, "radius": mode_hold.RADIUS * scale, "sigma": problem.sigma,
               "live": result["live"], "ema": result["ema"], "hold": result["hold"],
               "observations": recorder.curve,
               "convergence": sustained(recorder.curve, CONVERGENCE, expected_steps=recorder.steps),
               "seconds": time.monotonic() - start}
        report["rows"].append(row)
        write_json(args.output, report)
        print(f"DONE scale={scale} live={result['live']} ema={result['ema']} "
              f"stable_from={row['convergence']['stable_from_step']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
