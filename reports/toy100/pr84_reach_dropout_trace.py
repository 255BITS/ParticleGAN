"""Read-only per-update trace of the shared continuation dropout (stall reach).

Replays the constant-rate cold ring to ``--until`` and, for updates in
``[--start, --until]``, records after each full update: clean-support mode
coverage and HQ (graded offline with the ring centers), the clean cloud's mean
translation and RMS motion, and the recorder's D slope, width and trust factors
for that same update. Training is unchanged.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared.mode_hold import PASS_HQ  # noqa: F401
from reports.toy100 import pr84_reach_candidate as reach

HQ_RADIUS = .21


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1740)
    parser.add_argument("--until", type=int, default=1800)
    parser.add_argument("--ramp", default="stall")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from benchmarks.toy100.continuous_probe import run_probe

    rows, previous = [], {}
    original = reach.ReachRecorder.phases

    def phases(self, step, opt_d, opt_g, local):
        yield from original(self, step, opt_d, opt_g, local)
        if not args.start <= self.outer_steps <= args.until:
            return
        with torch.no_grad():
            gen = local["generator"]
            y = getattr(gen, "model", gen)(local["prior"].z).detach()
            dist = torch.cdist(y, local["means"])
            near = dist.min(1).values < HQ_RADIUS
            covered = int((dist.min(0).values < HQ_RADIUS).sum())
            move = y - previous["y"] if "y" in previous else torch.zeros_like(y)
            previous["y"] = y
        rec = self.records[-1]
        rows.append(dict(step=self.outer_steps, modes=covered, hq=round(float(near.float().mean()), 3),
                         translation=round(float(move.mean(0).norm()), 4),
                         rms=round(float(move.square().sum(1).mean().sqrt()), 4),
                         sharp=round(rec.get("critic_sharpness", 0.), 3),
                         width=round(rec.get("critic_width", 0.), 3),
                         adv=round(rec["critic_advantage"], 3),
                         g_factor=round(rec["g"]["factor"], 3), g_rho=round(rec["g"]["rho"], 3),
                         d_factor=round(rec["d"]["factor"], 3), d_rho=round(rec["d"]["rho"], 3)))
        print(json.dumps(rows[-1]), flush=True)

    reach.ReachRecorder.phases = phases
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    with reach.pr84_reach_candidate(task="mode_hold", ramp=args.ramp) as (recorder, _source):
        def hook(state):
            declare = state["declare_optimizer_accounting"]
            recorder.accounting = lambda calls, outer: declare(
                calls=calls + (args.until - outer), moment_updates=args.until)
            recorder.accounting(recorder.rows[recorder.optimizers[0]]["calls"], recorder.outer_steps)
        run_probe(config, mode="constant", steps=args.until, diagnostic_every=10,
                  checkpoint_hook_step=1, checkpoint_hook=hook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows) + "\n")


if __name__ == "__main__":
    main()
