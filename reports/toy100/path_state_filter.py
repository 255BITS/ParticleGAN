"""One-state check: does the path ray point at the missing ring mode?

Trains the frozen PR84 adapter, then scores clean particles. Mode centers are
used only after training, to report alignment. This does not change the update.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy
    from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
    from benchmarks.locked_shared.mode_hold import diversity, ring_means
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    from reports.toy100.path_acquisition import path_crossing_directions

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="path_state_filter", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    recipe, noise, _ = declared_recipe(config)
    spec = next(job["spec"] for job in plan() if job["spec"]["name"] == "mode_hold")
    raw = SimpleMLPDiscriminator.forward
    with pr84_smoothed_candidate(task="mode_hold") as (recorder, _):
        result, details = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        print(json.dumps(dict(event="APPLIED", applied=details["applied"])), flush=True)
        local = recorder._local
        generator, critic, prior = (local[k] for k in ("generator", "critic", "prior"))
        clean = getattr(generator, "model", generator)
        module = getattr(critic, "model", critic)
        points = clean(prior.z).detach()
        width = float(recorder.records[-1].get("critic_width") or 0.)

        def score(x):
            vals = [raw(module, x)]
            if width > 0:
                for dim in range(2):
                    shift = torch.zeros_like(x)
                    shift[:, dim] = width
                    vals.extend((raw(module, x + shift), raw(module, x - shift)))
            return torch.stack(vals).mean(0).reshape(-1)

        direction = path_crossing_directions(score, points, points)
    centers = ring_means()
    cloud = diversity(points, centers, detailed=True)
    missing = cloud["missing_modes"]
    row = dict(live=result["live"], clean=dict(modes=cloud["modes"], hq=cloud["hq"],
                                                missing=missing),
               width=width, redirected=int((direction.norm(dim=1) > 0).sum()),
               seconds=result["seconds"])
    if missing:
        target = centers[missing[0]]
        delta = target - points
        unit = delta / delta.norm(dim=1, keepdim=True).clamp_min(1e-8)
        proj = (direction * unit).sum(1)
        nearest = int(delta.norm(dim=1).argmin())
        row.update(nearest=nearest, nearest_projection=float(proj[nearest]),
                   nearest_distance=float(delta.norm(dim=1)[nearest]),
                   projections=proj.tolist(), points=points.tolist())
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
