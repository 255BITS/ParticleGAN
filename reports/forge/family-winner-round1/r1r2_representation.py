"""Zero-update trajectory representation witness on the actual public host.

This is a constructive parameter certificate, not a trainer, GAN result, or
qualification receipt. It performs no optimizer updates and fits no parameters.
The declared 24 schedule labels only evaluate one immutable analytic state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from benchmarks.locked_shared import trajectory
from benchmarks.locked_shared.baseline import score_metrics
from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input, wrap_output
from experiments.forge.behavior_adapters import BehaviorComponents
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.sources import runtime_manifest
from experiments.forge.state import state_digest
from particlegan import ParticlePrior


def install_rotation(generator: nn.Module, frames: int, slow_speed: float,
                     fast_speed: float) -> None:
    """Use paired LeakyReLUs to realize the exact per-frame linear rotation."""
    layers = tuple(generator.net)
    if (len(layers) != 5 or not all(isinstance(layers[i], nn.Linear) for i in (0, 2, 4))
            or not all(isinstance(layers[i], nn.LeakyReLU) and layers[i].negative_slope == .2
                       for i in (1, 3))):
        raise ValueError("trajectory host changed; this construction is unresolved")
    dims = frames * 2
    first, middle, last = (layers[i] for i in (0, 2, 4))
    if (first.in_features != dims + trajectory.PROTOCOL["z_dim"]
            or first.out_features != middle.in_features
            or middle.out_features != last.in_features
            or min(first.out_features, middle.out_features) < 2 * dims
            or last.out_features != dims):
        raise ValueError("trajectory dimensions cannot realize this declared construction")
    time = torch.linspace(0, 1, frames, dtype=first.weight.dtype)
    angle = (fast_speed - slow_speed) * time
    rotation = torch.zeros(dims, dims, dtype=first.weight.dtype)
    for frame, value in enumerate(angle):
        cosine, sine = value.cos(), value.sin()
        rotation[2*frame:2*frame+2, 2*frame:2*frame+2] = torch.stack(
            (torch.stack((cosine, -sine)), torch.stack((sine, cosine))))
    with torch.no_grad():
        for parameter in generator.parameters():
            parameter.zero_()
        basis = torch.eye(dims, dtype=first.weight.dtype)
        first.weight[:dims, :dims] = basis
        first.weight[dims:2*dims, :dims] = -basis
        middle.weight[:2*dims, :2*dims] = torch.eye(2*dims, dtype=first.weight.dtype)
        last.weight[:, :dims] = rotation / (1 + .2**2)
        last.weight[:, dims:2*dims] = -rotation / (1 + .2**2)


def build_certificate() -> dict:
    candidate = read_json(ROOT / "configs/forge/ideas/r3gan-stacked-training-toy-v1.json")
    task = read_json(ROOT / "configs/forge/tasks/trajectory.json")
    seed = read_json(ROOT / "configs/forge/protocols/screening.json")["seed"]
    if (task["execution"]["host"] != "trajectory" or task["execution"]["steps"] != 400
            or task["execution"]["prior"] != {
                "kind": "particle_cloud", "sigma": 0., "standardize": False, "learnable": True,
                "exception_reason": "The conditional host enumerates prior.z in its identity objective; a stochastic latent read changes the frozen host law."}
            or task["evaluation"]["sampling_law"] != "conditional_prior_centers_with_scheduled_output_noise"
            or task["evaluation"]["scoring_weights"] != "live"
            or task["evaluation"]["thresholds"] != [["identity_mse", "<=", .02]]
            or task["evaluation"]["observations"] != 24
            or task["evaluation"]["minimum_stable_checks"] != 5
            or seed != 0):
        raise ValueError("trajectory task law changed; declare a new representation certificate")
    request = {"candidate": candidate, "protocol": {"seed": seed}}
    components = BehaviorComponents(request, task)
    if components.recipe.output_noise_std != 0 or components.recipe.input_noise_std != 0:
        raise ValueError("this analytic witness requires the declared zero-noise Modern GAN recipe")
    slow, fast = trajectory.trajectories()
    hidden = trajectory.PROTOCOL["critic_hidden"]
    generator = components.context.construct(
        lambda: trajectory._Generator(slow.shape[1], trajectory.PROTOCOL["z_dim"], fast.shape[1], hidden),
        component="generator")
    critic = components.context.construct(
        lambda: trajectory._Critic(slow.shape[1] + fast.shape[1], hidden), component="discriminator")
    served = wrap_output(generator, components.noise)
    wrapped_critic = wrap_input(critic, components.noise, data_index=1)
    prior = ParticlePrior(trajectory.PROTOCOL["n_particles"], trajectory.PROTOCOL["z_dim"],
                          init_std=.1, generator=torch.Generator().manual_seed(seed))
    # Use the exact host's public component binder. Placeholder optimizers are
    # never stepped and are replaced by the public Recipe factories in bind().
    placeholder_g = torch.optim.Adam([*generator.parameters(), *prior.parameters()], lr=.005)
    placeholder_d = torch.optim.Adam(critic.parameters(), lr=.005)
    opt_g, opt_d, _, _ = components.bind(generator=served, critic=wrapped_critic,
        priors=[prior], opt_g=placeholder_g, opt_d=placeholder_d)
    initialization = {name: state_digest(model.state_dict()) for name, model in components.models.items()}
    install_rotation(generator, trajectory.PROTOCOL["frames"],
                     trajectory.PROTOCOL["slow_speed"], trajectory.PROTOCOL["fast_speed"])
    witness_state = state_digest(generator.state_dict())
    prior_state = state_digest(prior.state_dict())
    optimizer_before = state_digest({"g": opt_g.state_dict(), "d": opt_d.state_dict()})
    observations = []
    for schedule_label in sorted({math.ceil(i * 400 / 24) for i in range(1, 25)}):
        with torch.no_grad(), components.noise.evaluation(schedule_label):
            predictions = served(slow, prior.z)
        value = trajectory.identity_mse(predictions, fast)
        observations.append({"schedule_label": schedule_label, "identity_mse": value})
    # Deliberately wrong paired and collapsed outputs discriminate the actual
    # identity gate. These are oracle controls, not optimizer trajectories.
    controls = {}
    for label, output in (("rotation_oracle", predictions), ("unchanged_slow", slow),
                          ("swapped_identity", predictions.roll(len(predictions)//2, 0)),
                          ("collapse", predictions.mean(0, keepdim=True).expand_as(predictions))):
        metric = trajectory.identity_mse(output, fast)
        controls[label] = {"identity_mse": metric,
            "bounds": score_metrics({"identity_mse": metric}, task["evaluation"]["thresholds"]),
            "passed": trajectory.passed(metric)}
    after = state_digest({"g": opt_g.state_dict(), "d": opt_d.state_dict()})
    immutable = (witness_state == state_digest(generator.state_dict())
                 and prior_state == state_digest(prior.state_dict()) and optimizer_before == after)
    expected = controls["rotation_oracle"]["passed"] and all(not controls[label]["passed"]
        for label in ("unchanged_slow", "swapped_identity", "collapse"))
    if not immutable or not expected or any(not math.isfinite(o["identity_mse"]) or o["identity_mse"] > .02
                                           for o in observations):
        raise ValueError("trajectory representation witness or discriminating controls failed")
    bound_paths = ["benchmarks/locked_shared/trajectory.py", "benchmarks/locked_shared/baseline.py",
                   "benchmarks/locked_shared/observation.py", "benchmarks/transfer_suite/protocol.py",
                   "benchmarks/transfer_suite/legacy_noise_adapters.py",
                   "experiments/forge/behavior_adapters.py", "experiments/forge/api.py",
                   "experiments/forge/rng.py", "experiments/forge/taskrecipes.py",
                   "configs/forge/tasks/trajectory.json", "configs/forge/protocols/screening.json",
                   "configs/forge/ideas/r3gan-stacked-training-toy-v1.json"]
    bound_paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "particlegan").glob("*.py"))]
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in bound_paths}
    return {"schema_version": 1, "family": "r1r2", "task": "trajectory", "status": "SUPPORTED",
        "scope": "The exact finite conditional gate is represented, independent of optimizer learning. Applies only to unchanged host/target/prior/serving and zero input/output-noise configurations.",
        "construction": "Per-frame rotation by (fast_speed-slow_speed)*t; paired LeakyReLU(.2) channels give lrelu(lrelu(x))-lrelu(lrelu(-x))=(1+.2**2)*x; latent columns zero.",
        "training_updates": 0, "fitting_updates": 0, "ordinary_qualification": False,
        "scientific_training_status": "NOT_ATTEMPTED", "convergence_time": None,
        "sample_rows": len(slow), "frames": trajectory.PROTOCOL["frames"],
        "all_required_schedule_labels_evaluated": len(observations),
        "observations_are_training_checkpoints": False,
        "minimum_identity_mse": min(o["identity_mse"] for o in observations),
        "maximum_identity_mse": max(o["identity_mse"] for o in observations),
        "maximum_coordinate_error": float((predictions-fast).abs().max()),
        "controls": controls, "models_initial_state_sha256": initialization,
        "generator_witness_state_sha256": witness_state, "prior_witness_state_sha256": prior_state,
        "observation_did_not_mutate_model_prior_or_optimizers": immutable,
        "recipe": asdict(components.recipe), "prior": components.context.prior_config,
        "sampling": task["evaluation"], "host_protocol": trajectory.PROTOCOL,
        "rng_derivation": components.context.streams.version,
        "unintended_rng_deviations": sum(a["unintended_rng_deviations"] for a in components.rng_audits),
        "source_sha256": sources, "source_manifest_hash": stable_hash(sources),
        "reproducer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": runtime_manifest(), "complete_shared_suite_representation": "UNRESOLVED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous_threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.set_num_threads(1)
            result = build_certificate()
        atomic_json(args.output, result)
        print(json.dumps({key: result[key] for key in ("task", "status", "training_updates", "maximum_identity_mse",
            "maximum_coordinate_error", "unintended_rng_deviations", "source_manifest_hash")}), flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
