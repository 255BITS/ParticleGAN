"""One-state clean nonlinear pullback check after exact frozen PR84 cold replay.

This does not add the correction to training. The entire original 1,200-step
run is verified against its audited result first, then the final model/prior
is temporarily perturbed with its *last actual D real minibatch* and restored.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
from reports.toy100.coverage_pullback import centroid_pullback, prior_adam_metric
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate


def untimed(value):
    if isinstance(value, dict):
        return {key: untimed(item) for key, item in value.items() if key != "seconds"}
    if isinstance(value, list):
        return [untimed(item) for item in value]
    return value


def invariant_record(row):
    return {"outer_step": row.get("outer_step"),
            "critic_advantage": row.get("critic_advantage"),
            "critic_sharpness": row.get("critic_sharpness"),
            "critic_width": row.get("critic_width"),
            "d": {name: row["d"][name] for name in ("rho", "factor")},
            "g": {name: row["g"][name] for name in ("rho", "factor")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="alternating_curvature_response", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    spec = next(j["spec"] for j in plan() if j["spec"]["name"] == "mode_hold")
    original_sample = mode_hold.sample_ring
    observed = []
    with pr84_smoothed_candidate(task="mode_hold") as (recorder, _):
        def sampled(means, n, sigma, generator):
            points = original_sample(means, n, sigma, generator)
            if recorder.phase is not None and not recorder.passthrough:
                observed.append((recorder.outer_steps, recorder.phase, points.detach().clone()))
            return points
        with patch.object(mode_hold, "sample_ring", sampled):
            result, _ = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
        audit = json.loads(args.audit.read_text())
        if untimed(result) != untimed(audit["result"]):
            raise RuntimeError("original host replay disagrees with frozen audit")
        if ([invariant_record(row) for row in recorder.records]
                != [invariant_record(row) for row in audit["dynamics"]["records"]]):
            raise RuntimeError("optimizer replay invariants disagree with frozen audit")
        last_reals = [real for outer, phase, real in observed if outer == 1199 and phase == 2]
        if len(last_reals) != 2 or last_reals[0].shape != (mode_hold.BATCH, 2):
            tail = [(outer, phase, tuple(batch.shape)) for outer, phase, batch in observed[-12:]]
            raise RuntimeError(f"expected D and G real minibatches in final replay phase: {tail}")
        real = last_reals[0]
        local = recorder._local
        generator, prior = local["generator"], local["prior"]
        clean = getattr(generator, "model", generator)
        opt_g = recorder.optimizers[1]
        metric = prior_adam_metric(opt_g, prior.z)
        initial_z = prior.z.detach().clone()
        initial_optimizer = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                             for key, value in opt_g.state[prior.z].items()}
        rng_before = torch.get_rng_state().clone()
        width = recorder.records[-1]["critic_width"]
        correction = centroid_pullback(clean, prior.z, real, metric)
        after_z = prior.z.detach().clone()
        if not torch.equal(rng_before, torch.get_rng_state()):
            raise RuntimeError("coverage correction consumed the global RNG")
        for key, before in initial_optimizer.items():
            now = opt_g.state[prior.z][key]
            if isinstance(before, torch.Tensor) and not torch.equal(before, now):
                raise RuntimeError("coverage correction altered an Adam moment")
        with torch.no_grad():
            prior.z.copy_(initial_z)
        target = mode_hold.ring_means()[audit["result"]["observations"][-1]["missing_modes"][0]]
        before = torch.tensor(correction["output_before"])
        after = torch.tensor(correction["output_after"])
        directions = target - before
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        projection = ((after - before) * directions).sum(1)
        output = {
            "scope": "one clean final-state correction, no candidate training",
            "audit_untimed_host_parity": True, "audit_optimizer_record_invariants_parity": True,
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                              (Path(__file__), ROOT / "reports/toy100/coverage_pullback.py",
                               ROOT / "reports/toy100/pr84_smoothed_candidate.py")},
            "real_batch": mode_hold.BATCH,
            "real_batch_sha256": hashlib.sha256(real.numpy().tobytes()).hexdigest(),
            "final_stencil_width": width, "correction": correction,
            "missing_mode_for_reporting_only": int(audit["result"]["observations"][-1]["missing_modes"][0]),
            "toward_missing_output_displacement": projection.tolist(),
            "prior_restored": bool(torch.equal(prior.z, initial_z)),
            "prior_trial_change_norm": float((after_z - initial_z).norm()),
            "optimizer_moments_unchanged": True, "global_rng_unchanged": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"accepted": correction["accepted"], "alpha": correction["alpha"],
                      "coverage_before": correction["coverage_before"],
                      "coverage_after": correction["coverage_after"],
                      "toward_missing_max": float(projection.max())}))


if __name__ == "__main__":
    main()
