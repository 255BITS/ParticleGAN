"""The routed paired-edit example with a moving target: E22-routed versus R1-routed.

python -u examples/e22_routed_moving.py --turn-every 500
python -u examples/e22_routed_moving.py --turn-every 500 --r1

Every --turn-every updates the paired edit (target minus the frozen host) turns --degrees about the origin, on the
fitting, guard and held-out contexts alike. --r1 adds reopen_signal="optimizer" and reopen_anchor="release" to
e22_routed. Held-out RMSE is reported only; it is never a training signal. Prints one JSON line every --log-every
updates and a "period" line at the end of every period.
"""

import argparse
import json
import math
from pathlib import Path
import runpy

import torch

paired = runpy.run_path(str(Path(__file__).with_name("e22_routed_paired.py")))
R1_OVERRIDES = {"reopen_signal": "optimizer", "reopen_anchor": "release"}
HOST = torch.tensor([[1., 0., .05], [0., 1., -.07]]).bfloat16()


def host(context):
    return (context.bfloat16() @ HOST.to(context.device).T).float()


class MovingTarget:
    """Turns the paired edit of every context set in place."""

    def __init__(self, loop):
        self.loop = loop
        self.edits = {name: getattr(loop, name + "_targets") - host(getattr(loop, name + "_context"))
                      for name in ("fit", "guard", "test")}

    def turn_to(self, radians):
        c, s = math.cos(radians), math.sin(radians)
        rotation = torch.tensor([[c, -s], [s, c]], device=self.loop.policy.device)
        for name, edit in self.edits.items():
            getattr(self.loop, name + "_targets").copy_(host(getattr(self.loop, name + "_context")) + edit @ rotation.T)


def reopens(policy):
    return sum(tester.counts.get("reopens", 0) for row in policy.lr_settle.testers for tester in row if tester is not None)


def run(*, turn_every, turns=2, degrees=30., r1=False, log_every=50, emit=print):
    loop = paired["make_loop"](recipe_overrides=R1_OVERRIDES if r1 else None)
    target = MovingTarget(loop)
    periods = []
    for step in range(1, turn_every * (turns + 1) + 1):
        if step > 1 and (step - 1) % turn_every == 0:
            target.turn_to(math.radians(degrees) * ((step - 1) // turn_every))
        with torch.autograd.set_multithreading_enabled(False):
            paired["update"](loop)
        if step % log_every == 0:
            emit(json.dumps({"step": step, "heldout_rmse": paired["evaluate"](loop)["heldout_rmse"],
                             "reopens": reopens(loop.policy)}))
        if step % turn_every == 0:
            periods.append(paired["evaluate"](loop)["heldout_rmse"])
            emit(json.dumps({"event": "period", "step": step, "heldout_rmse": periods[-1]}))
    return loop, periods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turn-every", type=int, default=500)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--degrees", type=float, default=30.)
    parser.add_argument("--r1", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(turn_every=args.turn_every, turns=args.turns, degrees=args.degrees, r1=args.r1,
        log_every=args.log_every, emit=lambda line: print(line, flush=True))


if __name__ == "__main__":
    main()
