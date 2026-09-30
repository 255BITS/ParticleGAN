"""Diagnostic fork before the shared dropout: is D's lag behind G the driver?

Replays the stall-reach constant-rate ring to ``--step``; Linux ``fork`` then
continues the identical state to ``--until`` as:

- ``as_is``: unchanged (the parent);
- ``d2``: every ordinary D Adam displacement doubled (a faster critic);
- ``g05``: every ordinary G Adam displacement halved (a slower generator);
- ``game`` / ``game2``: G's trust bound also sees one / two virtual D steps answering G's proposal.

Both change only the D/G timescale ratio. Each update logs clean-support modes and
HQ (graded offline with ring centers). This is a counterfactual probe, not a candidate.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100 import pr84_reach_candidate as reach

VARIANTS = {"d2": ("d", 2.), "g05": ("g", .5), "game": ("game", 1), "game2": ("game", 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1755)
    parser.add_argument("--until", type=int, default=1900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from benchmarks.toy100.continuous_probe import run_probe

    state = dict(name="as_is", role=None, scale=1.)
    rows = []
    original_phases = reach.ReachRecorder.phases
    original_step = reach.ReachRecorder.step

    def scaled(ordinary, params):
        def run(optimizer, closure=None):
            before = [p.detach().clone() for p in params(optimizer)]
            result = ordinary(optimizer, closure=closure)
            with torch.no_grad():
                for p, b in zip(params(optimizer), before):
                    p.copy_(b + state["scale"] * (p - b))
            return result
        return run

    def step(self, optimizer, ordinary_step, closure=None):
        if state["role"] is not None and self.optimizers is not None and not self.passthrough:
            target = self.optimizers[0] if state["role"] == "d" else self.optimizers[1]
            if optimizer is target:
                ordinary_step = scaled(ordinary_step, self._params)
        return original_step(self, optimizer, ordinary_step, closure)

    def phases(self, step_index, opt_d, opt_g, local):
        if self.outer_steps == args.step and state["name"] == "as_is" and not getattr(self, "_forked", False):
            self._forked = True
            for name, (role, scale) in VARIANTS.items():
                if os.fork() == 0:
                    state.update(name=name, role=None if role == "game" else role, scale=scale)
                    self.game_bound = role == "game"
                    if self.game_bound:
                        self.game_steps = int(scale)
                    break
        yield from original_phases(self, step_index, opt_d, opt_g, local)
        if self.outer_steps > args.step and self.outer_steps % 5 == 0:
            with torch.no_grad():
                gen = local["generator"]
                y = getattr(gen, "model", gen)(local["prior"].z)
                dist = torch.cdist(y, local["means"])
            rows.append(dict(step=self.outer_steps, modes=int((dist.min(0).values < .21).sum()),
                             hq=round(float((dist.min(1).values < .21).float().mean()), 3)))
            (args.output / f"{state['name']}.json").write_text(json.dumps(rows) + "\n")
            if self.outer_steps == args.until and state["name"] != "as_is":
                os._exit(0)

    reach.ReachRecorder.phases = phases
    reach.ReachRecorder.step = step
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    with reach.pr84_reach_candidate(task="mode_hold", ramp="stall") as (recorder, _source):
        def hook(s):
            declare = s["declare_optimizer_accounting"]
            recorder.accounting = lambda calls, outer: declare(
                calls=calls + (args.until - outer), moment_updates=args.until)
            recorder.accounting(recorder.rows[recorder.optimizers[0]]["calls"], recorder.outer_steps)
        run_probe(config, mode="constant", steps=args.until, diagnostic_every=10,
                  checkpoint_hook_step=1, checkpoint_hook=hook)
    for _ in VARIANTS:
        os.wait()
    for name in ("as_is", *VARIANTS):
        data = json.loads((args.output / f"{name}.json").read_text())
        passing = sum(r["modes"] == 8 and r["hq"] >= .9 for r in data)
        print(json.dumps(dict(variant=name, passing=f"{passing}/{len(data)}",
                              min_modes=min(r["modes"] for r in data),
                              trace=[(r["step"], r["modes"], r["hq"]) for r in data if r["step"] % 10 == 0])))


if __name__ == "__main__":
    main()
