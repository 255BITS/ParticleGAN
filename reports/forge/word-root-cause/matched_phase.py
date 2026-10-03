"""Compare the matched K3P/KA2 saved receipts and penalty phases without training.

Run from a checkout containing the frozen penalty/schedule sources:
  python matched_phase.py ROUND2_DIRECTORY
The phase calculation follows the previous completed critic step's LR record.
No model construction, optimizer update, or random sampling is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from particlegan.ka2 import S_FIX, WARMUP_CALLS
from particlegan.recipes import Recipe, learning_rate_scales


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(base):
    names = ("k3p-slow-prior-0p1", "ka2-slow-prior-0p1")
    raw = {name: json.loads((base / name / "adapter-receipt.json").read_text())
           for name in names}
    recipes = {name: item["recipe"] for name, item in raw.items()}
    recipe_differences = {
        key: {name: recipe.get(key) for name, recipe in recipes.items()}
        for key in sorted(set().union(*(recipe.keys() for recipe in recipes.values())))
        if len({json.dumps(recipe.get(key), sort_keys=True)
                for recipe in recipes.values()}) > 1
    }
    recipe = Recipe(**recipes[names[0]])
    states = {name: torch.load(base / name / "state.pt", map_location="cpu",
                               weights_only=True)["fixture"]["api_state"]
              for name in names}
    completed = {name: item["completed_steps"] for name, item in states.items()}
    if len(set(completed.values())) != 1:
        raise ValueError("matched states have different completed update counts")
    calls = next(iter(completed.values()))
    weights = []
    for call in range(1, calls + 1):
        if call == 1:
            weight = 1.0
        else:
            # call n sees the LR recorded by completed critic step n-1;
            # that step began with zero-based schedule index n-2.
            network, _ = learning_rate_scales(call - 2, recipe)
            floor = recipe.resolved_network_lr_floor
            weight = max(0.0, min(1.0, 2.0 * network) - 2.0 * floor) / (1.0 - 2.0 * floor)
        weights.append(weight)
    blended = [index + 1 for index, weight in enumerate(weights) if weight < 1.0]
    audit = {}
    for name in names:
        mechanisms = raw[name]["evidence"]["guards"]["mechanism_audit"]["mechanisms"]
        record = states[name]["optimizers"][1]["regularizer"]["record"]
        is_k3p = name.startswith("k3p")
        first_blend = blended[0] if is_k3p else WARMUP_CALLS
        blend_calls = len(blended) if is_k3p else calls - WARMUP_CALLS + 1
        if mechanisms["critic_anchor"]["eligible"] != blend_calls:
            raise ValueError(f"{name}: saved anchor eligibility disagrees with phase calculation")
        audit[name] = {
            "state_sha256": sha(base / name / "state.pt"),
            "adapter_receipt_sha256": sha(base / name / "adapter-receipt.json"),
            "first_blend_penalty_call": first_blend,
            "pure_A_calls": calls - blend_calls,
            "blended_calls": blend_calls,
            "final_blend_A_weight": weights[-1] if is_k3p else S_FIX,
            "mechanisms": mechanisms,
            "critic_record": {key: value for key, value in record.items() if key != "sur_hist"},
        }
    dimension = states[names[0]]["models"]["critic"]["net.0.weight"].shape[1]
    coeff, kappa = recipe.reg_coeff, recipe.reg_kappa
    return {
        "schema_version": 1,
        "scope": "saved-receipt and source algebra audit; no training, random sampling, or qualification",
        "recipe_differences": recipe_differences,
        "initialization_identical": raw[names[0]]["initialization"] == raw[names[1]]["initialization"],
        "rng_manifest_identical": raw[names[0]]["rng"] == raw[names[1]]["rng"],
        "prior_declaration_identical": raw[names[0]]["prior"] == raw[names[1]]["prior"],
        "completed_updates": completed,
        "states": audit,
        "penalty_algebra": {
            "joint_input_dimension": dimension,
            "reg_coeff": coeff,
            "reg_kappa": kappa,
            "pure_A_real_squared_L2_coefficient": coeff / (2.0 * dimension),
            "pure_A_fake_L2_cap_threshold": kappa * dimension ** .5,
            "pure_B_each_L2_cap_squared_coefficient": coeff / 2.0,
            "pure_B_anchor_difference_squared_L2_coefficient": coeff * recipe.reg_anchor_weight / (2.0 * dimension),
            "KA2_blend_real_A_squared_L2_coefficient": coeff * S_FIX / (2.0 * dimension),
            "KA2_blend_each_B_L2_cap_squared_coefficient": coeff * (1.0 - S_FIX) / 2.0,
            "KA2_blend_anchor_difference_squared_L2_coefficient_before_controller_W": coeff * (1.0 - S_FIX) * recipe.reg_anchor_weight / (2.0 * dimension),
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                          (ROOT / "particlegan" / filename for filename in
                           ("grad_regularizers.py", "ka2.py", "recipes.py", "policy.py"))},
        "attribution_guards": [
            "Only recipe name and critic formulation differ in this matched pair; phase/controller/guard behavior is part of the existing formulation.",
            "A2 was requested but eligible/applied zero times; direct-particle gain had no applicable host component. Synthetic checks are not training activation.",
            "The successful endpoint has small joint critic gradients, but endpoint gradients cannot establish when or why modes were acquired.",
            "Word-versus-latent gradient ratios overlap successful and failed cohorts; those ratios alone do not establish causal imbalance.",
            "Changing positive existing reg_coeff/reg_kappa preserves the penalty equations. Disabling terms or changing phase rules would alter the technique.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round2_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.round2_directory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
