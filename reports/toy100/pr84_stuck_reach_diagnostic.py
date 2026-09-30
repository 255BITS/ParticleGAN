"""Read-only: does a wider G critic read point stuck PR84 particles at the missing mode?

Runs the unchanged PR84 cold ring to ``--step``, saves G/D/prior, then compares
the five-point stencil gradient at each clean particle across widths. Mode
centers are used only after training, to grade directions.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from reports.toy100.pr84_reach_candidate import pr84_reach_candidate
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate, SmoothedBothBoundRecorder

FACTORIES = dict(baseline=pr84_smoothed_candidate, reach=pr84_reach_candidate)


class _Stop(Exception):
    pass


def run_capture(step, path, method):
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
    original = SmoothedBothBoundRecorder.phases

    def phases(self, s, opt_d, opt_g, local):
        if self.outer_steps == step:
            critic = local["critic"]
            while not isinstance(critic, SimpleMLPDiscriminator):
                critic = critic.model
            gen = local["generator"]
            torch.save(dict(critic=critic, generator=getattr(gen, "model", gen),
                            z=local["prior"].z.detach().clone(), means=local["means"]), path)
            raise _Stop
        yield from original(self, s, opt_d, opt_g, local)

    SmoothedBothBoundRecorder.phases = phases
    try:
        with FACTORIES[method](task="mode_hold"):
            run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    except _Stop:
        pass
    finally:
        SmoothedBothBoundRecorder.phases = original


def stencil_grad(critic, x, width):
    x = x.clone().requires_grad_(True)
    vals = [critic(x)]
    for dim in range(2):
        shift = torch.zeros_like(x)
        shift[:, dim] = width
        vals += [critic(x + shift), critic(x - shift)]
    torch.stack(vals, 0).mean(0).sum().backward()
    return x.grad


def analyse(path, widths):
    state = torch.load(path, weights_only=False)
    critic, gen, z, means = state["critic"], state["generator"], state["z"], state["means"]
    with torch.no_grad():
        y = gen(z)
    dist = torch.cdist(y, means)
    covered = (dist.min(0).values < .21)
    missing = [int(k) for k in torch.nonzero(~covered).flatten()]
    rows = dict(missing=missing, occupancy=[int((dist.argmin(1) == k).sum()) for k in range(len(means))])
    with torch.no_grad():
        rows["critic_at_centers"] = [round(float(v), 3) for v in critic(means).flatten()]
    for k in missing:
        order = torch.argsort(dist[:, k])[:3]
        to = means[k] - y[order]
        unit = to / to.norm(dim=1, keepdim=True)
        rows[f"mode{k}"] = dict(nearest_dist=[round(float(v), 3) for v in dist[order, k]])
        for w in widths:
            g = stencil_grad(critic, y[order], w) if w > 0 else stencil_grad(critic, y[order], 1e-4)
            rows[f"mode{k}"][f"w{w}"] = [round(float(v), 3) for v in (g * unit).sum(1)]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(FACTORIES), default="baseline")
    args = parser.parse_args()
    if not args.path.exists():
        run_capture(args.step, args.path, args.method)
    print(json.dumps(analyse(args.path, (0, .15, .3, .5, .75, 1.0, 1.5))), flush=True)


if __name__ == "__main__":
    main()
