"""Constant-LR grid wrapper around worker.py (worker.py itself is not modified).

Usage: lr_grid.py --lr-c MC --lr-g MG <any worker.py flags>

Scales the recipe's base LRs before the optimizers are built:
  critic LR = base * MC, generator LR = base * MG, prior LR = 2 * base * MG (prior stays 2x G).
Implemented by replacing recipe.lr = base*MG and recipe.d_lr_mult = MC/MG (prior_lr_mult unchanged);
multipliers are powers of two, so every rate is exact. worker.run already verifies after every update
that the optimizer LRs equal their initial values; this wrapper additionally checks the initial values
equal the requested ones (via the declaration written before training starts).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import worker

BASE = {"generator_0": 0.00425, "prior_1": 0.0085, "critic_0": 0.00425}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lr-c", type=float, required=True)
    pre.add_argument("--lr-g", type=float, required=True)
    mults, rest = pre.parse_known_args()
    mc, mg = mults.lr_c, mults.lr_g
    if "--d-lr-mult" in rest:
        raise SystemExit("use --lr-c instead of --d-lr-mult")
    expected = {"generator_0": BASE["generator_0"] * mg, "prior_1": BASE["prior_1"] * mg,
                "critic_0": BASE["critic_0"] * mc}

    make_recipe, describe, run, rates = worker.make_recipe, worker.describe, worker.run, worker.rates

    def make_recipe_lr(args):
        recipe = make_recipe(args)
        assert recipe.lr == BASE["generator_0"] and recipe.d_lr_mult == 1.0 and recipe.prior_lr_mult == 2.0
        return dataclasses.replace(recipe, lr=recipe.lr * mg, d_lr_mult=recipe.d_lr_mult * mc / mg)

    def describe_lr(args):
        tag = "" if (mc, mg) == (1.0, 1.0) else f" {{lr c×{mc:g} g×{mg:g}}}"
        return describe(args) + tag

    checked = {"n": 0}

    def rates_lr(opt_g, opt_d, roles):
        out = rates(opt_g, opt_d, roles)
        if out != expected:
            raise RuntimeError(f"LRs {out} != requested {expected}")
        checked["n"] += 1
        return out

    def run_lr(args):
        args.lr_c, args.lr_g = mc, mg  # recorded in declaration/result config
        result = run(args)
        # rates() is called once for initial_rates and once per update
        assert checked["n"] == result["completed_steps"] + 1 == result["constant_lr_verified_updates"] + 1
        assert result["learning_rates"] == expected
        print(json.dumps({"arm": args.arm, "lr": expected, "lr_checked_updates": checked["n"] - 1}),
              file=sys.stderr)
        return result

    worker.make_recipe, worker.describe, worker.run, worker.rates = make_recipe_lr, describe_lr, run_lr, rates_lr
    sys.argv = [sys.argv[0], *rest]
    worker.main()


if __name__ == "__main__":
    main()
