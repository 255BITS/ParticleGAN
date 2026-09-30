"""Diagnostic fork of the stuck AVX512 reach cold ring: is G's width or trust region the blocker?

The unchanged reach .5 cold ring runs to ``--step``. There, Linux ``fork``
gives each variant the identical live state; each child continues to 1200:

- ``as_is``: reach .5 unchanged (the parent);
- ``w05``: G stencil width held at .5;
- ``w05_nobound``: width .5 and G's own-curvature bound removed.

These variants are counterfactual probes of one state, not candidates.
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

VARIANTS = ("w05", "w05_nobound")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    recipe, noise, _ = declared_recipe(config)
    spec = next(job["spec"] for job in plan() if job["spec"]["name"] == "mode_hold")

    name = "as_is"
    fixed_width = None
    original_phases = reach.ReachRecorder.phases
    original_arm = reach.ReachRecorder._arm_smoothed_critic

    def arm(self):
        original_arm(self)
        if fixed_width is not None and self._smooth_on:
            self._smooth_width = fixed_width
            self.row["critic_width"] = fixed_width

    def phases(self, step, opt_d, opt_g, local):
        nonlocal name, fixed_width
        if self.outer_steps == args.step and name == "as_is" and not getattr(self, "_forked", False):
            self._forked = True
            for variant in VARIANTS:
                if os.fork() == 0:
                    name, fixed_width = variant, .5
                    if variant == "w05_nobound":
                        self.curvature_bound = float("inf")
                    break
        yield from original_phases(self, step, opt_d, opt_g, local)

    reach.ReachRecorder.phases = phases
    reach.ReachRecorder._arm_smoothed_critic = arm
    with reach.pr84_reach_candidate(task="mode_hold") as (recorder, _source):
        result, _ = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    obs = [dict(step=o["step"], modes=o["modes"], hq=round(o["hq"], 4))
           for o in result["observations"] if o["step"] >= args.step]
    late = [r for r in recorder.records if r["outer_step"] > args.step]
    row = dict(variant=name, observations=obs,
               mean_g_factor=sum(r["g"]["factor"] for r in late) / len(late),
               mean_sharp=sum(r.get("critic_sharpness", 0) for r in late) / len(late),
               mean_width=sum(r.get("critic_width", 0) for r in late) / len(late))
    (args.output / f"{name}.json").write_text(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    if name != "as_is":
        os._exit(0)
    for _ in VARIANTS:
        os.wait()


if __name__ == "__main__":
    main()
