"""Scratch diagnostic: cap only the particle-prior LR horizon at 4000.

The production affine/square model, G/D horizon 1600, fixed seed, 7000-step
budget, noise, and evaluation protocol come from accuracy_shared_policy.json.
This experiment is not common-22 gate evidence. Its own script and every
executed native source file are archived with the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy100 import train as runner
from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_suite
from benchmarks.toy100.gate import evaluate_suite as coverage_suite
from benchmarks.toy100.problems import PROBLEM_NAMES
from benchmarks.toy100.schedule import policy_multipliers
from particlegan.recipes import learning_rate_scale


NETWORK_HORIZON = 1600
PRIOR_HORIZON = 4000
CONFIG = ROOT / "configs/toy100/accuracy_shared_policy.json"
SOURCE_NAME = str(Path(__file__).resolve().relative_to(ROOT))


def _multipliers(trainer):
    recipe = trainer.recipe
    step = trainer.completed_steps
    network, _ = policy_multipliers(
        step, recipe.total_steps, recipe.lr_anneal_start, recipe.lr_floor,
        NETWORK_HORIZON,
    )
    prior = learning_rate_scale(
        step, min(recipe.total_steps, PRIOR_HORIZON),
        recipe.lr_anneal_start, recipe.lr_floor,
    )
    return network, prior


def _step_with_prior_horizon(trainer, real, *, network_lr_horizon_cap=None,
                             **step_kwargs):
    if network_lr_horizon_cap != NETWORK_HORIZON:
        raise ValueError("scratch probe requires the declared network horizon")
    if len(trainer.opt_g.param_groups) != 2 or len(trainer.opt_d.param_groups) != 1:
        raise RuntimeError("scratch probe expects G, prior, and D optimizer groups")
    network, prior = _multipliers(trainer)

    def set_g_rates(optimizer, args, kwargs):
        optimizer.param_groups[0]["lr"] = trainer.initial_lrs[0][0] * network
        optimizer.param_groups[1]["lr"] = trainer.initial_lrs[0][1] * prior

    def set_d_rate(optimizer, args, kwargs):
        optimizer.param_groups[0]["lr"] = trainer.initial_lrs[1][0] * network

    g_hook = trainer.opt_g.register_step_pre_hook(set_g_rates)
    try:
        d_hook = trainer.opt_d.register_step_pre_hook(set_d_rate)
        try:
            return trainer.step(real, **step_kwargs)
        finally:
            d_hook.remove()
    finally:
        g_hook.remove()


def _rate_action(trainer, completed_step, *, network_lr_horizon_cap=None):
    if (network_lr_horizon_cap != NETWORK_HORIZON
            or completed_step != trainer.completed_steps):
        raise ValueError("scratch probe update or network horizon differs")
    recipe = trainer.recipe
    network, _ = policy_multipliers(
        completed_step - 1, recipe.total_steps,
        recipe.lr_anneal_start, recipe.lr_floor, NETWORK_HORIZON,
    )
    prior = learning_rate_scale(
        completed_step - 1, min(recipe.total_steps, PRIOR_HORIZON),
        recipe.lr_anneal_start, recipe.lr_floor,
    )
    rates = {
        "lr_g": trainer.opt_g.param_groups[0]["lr"],
        "lr_prior": trainer.opt_g.param_groups[1]["lr"],
        "lr_d": trainer.opt_d.param_groups[0]["lr"],
    }
    expected = {
        "lr_g": trainer.initial_lrs[0][0] * network,
        "lr_prior": trainer.initial_lrs[0][1] * prior,
        "lr_d": trainer.initial_lrs[1][0] * network,
    }
    for role, actual in rates.items():
        if not math.isclose(actual, expected[role], rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError(f"scratch optimizer action differs: {role}")
    return {**rates,
            "network_multiplier": network,
            "prior_multiplier": prior,
            "network_lr_horizon_cap": NETWORK_HORIZON,
            "prior_lr_horizon_cap": PRIOR_HORIZON}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=PROBLEM_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_bytes = CONFIG.read_bytes()
    config = json.loads(config_bytes)
    expected = {
        "steps": 7000, "seed": 1234, "device": "cpu", "batch_size": 2048,
        "z_dim": 2, "num_particles": 20_000, "fourier": 3,
        "toy100_model": "affine_square_v1",
        "network_lr_horizon_cap": NETWORK_HORIZON,
        "output_noise_std": .029, "output_noise_warmup": .2,
        "input_noise_std": .5, "input_noise_anneal_end": .1,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("declared candidate differs from the archived three-pass core")
    config["problem"] = args.problem

    source = Path(__file__).read_bytes()
    base_provenance = runner._source_provenance

    def scratch_provenance(*, include_policy=False):
        record = (base_provenance(include_policy=True) if include_policy
                  else base_provenance())
        record["source_sha256"][SOURCE_NAME] = hashlib.sha256(source).hexdigest()
        record["declared_config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        record["trainer_factory"] = "native affine/square with scratch prior-horizon override"
        record["model_options"] = {
            "network_lr_horizon_cap": NETWORK_HORIZON,
            "prior_lr_horizon_cap": PRIOR_HORIZON,
            "prior_horizon_role": "particle_prior_only",
        }
        record["shared_gate_eligible"] = False
        return record

    runner._source_provenance = scratch_provenance
    runner.step_with_policy = _step_with_prior_horizon
    runner.policy_rate_action = _rate_action

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "declared_config.json").write_bytes(config_bytes)
    (args.output / "scratch_policy.json").write_text(json.dumps({
        "shared_gate_eligible": False,
        "network_lr_horizon_cap": NETWORK_HORIZON,
        "prior_lr_horizon_cap": PRIOR_HORIZON,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "declared_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }, indent=2) + "\n")
    summary = runner.train(config, args.output / args.problem)
    if CONFIG.read_bytes() != config_bytes:
        raise RuntimeError("declared candidate changed during scratch run")
    coverage = coverage_suite(args.output, problem=args.problem)
    accuracy = accuracy_suite(args.output, problem=args.problem)
    result = {
        "problem": args.problem,
        "shared_gate_eligible": False,
        "run_status": summary["status"],
        "coverage_status": coverage["status"],
        "accuracy_status": accuracy["status"],
        "final_live": summary["final"]["live"],
        "final_accuracy_live": summary.get("final_accuracy", {}).get("live"),
    }
    (args.output / "scratch_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
