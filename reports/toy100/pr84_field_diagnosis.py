"""Read-only field diagnosis for the frozen PR84 smoothed-critic ring run.

The original host and optimizer still execute all 1,200 updates. Afterward,
this script inspects the last same-batch G proposal and the noise-free critic
field. Mode centers are used only in the post-training diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))

    import torch
    from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy
    from reports.toy100.alternating_curvature_scratch import alternating_curvature

    torch.set_num_threads(1)
    config = json.loads((root / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="alternating_curvature_response", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    spec = next(j["spec"] for j in plan() if j["spec"]["name"] == "mode_hold")
    raw_forward = SimpleMLPDiscriminator.forward
    with alternating_curvature(task="mode_hold", bound_d=True, curvature_bound=.25,
                               d_curvature_bound=3., smooth_critic=True) as (recorder, _):
        result, _ = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        local = recorder._local
        generator, critic, prior = (local[k] for k in ("generator", "critic", "prior"))
        clean = getattr(generator, "model", generator)
        dnet = getattr(critic, "model", critic)
        opt_g = recorder.optimizers[1]
        params = recorder._params(opt_g)
        g_count = len(list(generator.parameters()))
        saved = [q.detach().clone() for q in params]

        def points(values):
            with torch.no_grad():
                for q, value in zip(params, values):
                    q.copy_(value)
                return clean(prior.z).detach().clone()

        base = points(recorder.g_base)
        proposal = points(recorder.g1)
        network_only = points(recorder.g1[:g_count] + recorder.g_base[g_count:])
        prior_only = points(recorder.g_base[:g_count] + recorder.g1[g_count:])
        accepted = points(saved)
        for q, value in zip(params, saved):
            q.data.copy_(value)

        width = float(recorder.records[-1]["critic_width"])

        def score_and_gradient(x, smooth):
            x = x.detach().clone().requires_grad_(True)
            values = [raw_forward(dnet, x)]
            if smooth:
                for dim in range(x.shape[-1]):
                    offset = torch.zeros_like(x)
                    offset[:, dim] = width
                    values.extend((raw_forward(dnet, x + offset),
                                   raw_forward(dnet, x - offset)))
            score = torch.stack(values).mean(0).reshape(-1)
            gradient, = torch.autograd.grad(score.sum(), x)
            return score.detach(), gradient.detach()

        from benchmarks.locked_shared.mode_hold import ring_means
        centers = ring_means()
        final_obs = result["observations"][-1]
        missing = final_obs["missing_modes"]
        sharp_center, _ = score_and_gradient(centers, False)
        smooth_center, _ = score_and_gradient(centers, True)
        sharp_fake, sharp_grad = score_and_gradient(base, False)
        smooth_fake, smooth_grad = score_and_gradient(base, True)
        if len(missing) != 1:
            raise RuntimeError(f"expected one missing mode, observed {missing}")
        target = centers[missing[0]]
        delta = target - base
        dist = delta.norm(dim=1)
        direction = delta / dist[:, None].clamp_min(1e-12)
        nearest = int(dist.argmin())
        move = proposal - base
        bounded = accepted - base
        net_move = network_only - base
        prior_move = prior_only - base
        sharp_component = (sharp_grad * direction).sum(1)
        smooth_component = (smooth_grad * direction).sum(1)
        candidates = sorted(range(len(dist)), key=lambda i: float(dist[i]))[:4]

        def movement(v, i):
            return {"norm": float(v[i].norm()), "toward_missing": float((v[i] * direction[i]).sum()),
                    "xy": v[i].tolist()}

        profiles = {}
        for smooth in (False, True):
            path = torch.stack([base[nearest] + t * (target - base[nearest])
                                for t in (0., .25, .5, .75, 1.)])
            scores, gradients = score_and_gradient(path, smooth)
            profiles["smooth" if smooth else "sharp"] = {
                "scores": scores.tolist(),
                "directional_derivatives": (gradients * direction[nearest]).sum(1).tolist()}

        audit = json.loads(args.audit.read_text())
        def untimed(value):
            if isinstance(value, dict):
                return {k: untimed(v) for k, v in value.items() if k != "seconds"}
            if isinstance(value, list):
                return [untimed(v) for v in value]
            return value
        parity = untimed(result) == untimed(audit["result"])
        if not parity:
            raise RuntimeError("diagnostic replay changed the archived untimed host result")
        dynamics_parity = recorder.receipt() == audit["dynamics"]
        if not dynamics_parity:
            raise RuntimeError("diagnostic replay changed the archived optimizer records")
        output = {
            "audit_host_result_exact_except_time": parity,
            "audit_optimizer_records_exact": dynamics_parity,
            "source_sha256": {str(root / "reports/toy100/alternating_curvature_scratch.py"):
                              sha(root / "reports/toy100/alternating_curvature_scratch.py"),
                              str(Path(__file__).resolve()): sha(__file__)},
            "step": 1200, "missing_mode": missing[0], "missing_center": target.tolist(),
            "stencil_width": width, "g_bound_factor": recorder.records[-1]["g"]["factor"],
            "critic_centers": {"sharp": sharp_center.tolist(), "smooth": smooth_center.tolist()},
            "critic_fake_range": {"sharp": [float(sharp_fake.min()), float(sharp_fake.max())],
                                  "smooth": [float(smooth_fake.min()), float(smooth_fake.max())]},
            "toward_missing_critic_gradient": {
                "sharp": sharp_component.tolist(), "smooth": smooth_component.tolist()},
            "candidate_particles": [{"index": i, "distance": float(dist[i]),
                                      "base": base[i].tolist(), "sharp_score": float(sharp_fake[i]),
                                      "smooth_score": float(smooth_fake[i]),
                                      "sharp_gradient_toward": float(sharp_component[i]),
                                      "smooth_gradient_toward": float(smooth_component[i]),
                                      "proposal": movement(move, i),
                                      "bounded": movement(bounded, i),
                                      "network_only": movement(net_move, i),
                                      "prior_only": movement(prior_move, i)}
                                     for i in candidates],
            "nearest_path_critic": profiles,
            "all_particle_proposal_toward_missing": (move * direction).sum(1).tolist(),
            "all_particle_bounded_toward_missing": (bounded * direction).sum(1).tolist(),
            "terminal_observations": [{"step": r["step"], "modes": r["modes"], "hq": r["hq"]}
                                      for r in result["observations"][-5:]],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("audit_host_result_exact_except_time",
                                            "audit_optimizer_records_exact", "step",
                                            "missing_mode", "terminal_observations")}))


if __name__ == "__main__":
    main()
