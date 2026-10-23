"""Fixed pre-admission software fixtures, not learned-quality experiments.

No source/task selection, extra data draws, seed search or paid training.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import pytest
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.process_time()
    fixtures = runpy.run_path(str(ROOT / "tests/test_bispf_mechanics_replay.py"))
    with pytest.MonkeyPatch.context() as patch:
        model = fixtures["quadratic_fixture"](patch)
        old = model.G[0].bias.detach().clone()
        result = model.step(fixtures["batch"]())
        current = model.G[0].bias.detach().clone()
        quadratic = dict(anchor_bias=old.tolist(), final_bias=current.tolist(),
                         norm_ratio=float(current.norm() / old.norm()),
                         stats=result["generator_replay"])
        assert quadratic["norm_ratio"] < 1
    with pytest.MonkeyPatch.context() as patch:
        ordinary, candidate = fixtures["trainer"]("ordinary"), fixtures["trainer"]()
        for owner in (ordinary, candidate):
            original = owner._generator_losses
            def linear(*values, owner=owner, original=original):
                losses = original(*values)
                force = next(owner.G.parameters()).sum()
                losses["loss_g"], losses["loss_gan"] = force, force
                return losses
            patch.setattr(owner, "_generator_losses", linear)
        ordinary.step(fixtures["batch"]())
        bilinear = candidate.step(fixtures["batch"]())["generator_replay"]
        fixtures["equal"](ordinary.G.state_dict(), candidate.G.state_dict())
        fixtures["equal"](ordinary.prior.state_dict(), candidate.prior.state_dict())
        assert bilinear["applied_scale"] == 1 and bilinear["consistency_lhs"] == 0
    # Rejected pre-admission protected-field operator, smoothing limit c=12.
    # At alpha*c=3 the reflected secant is zero but the iterate doubles.
    c, alpha, x = 12., .25, 1.
    p0, y = -c*x, (1-alpha*c)*x
    reflected = -c*abs(y)/2
    old_certificate = dict(c=c, alpha=alpha, predictor=y, raw_anchor=p0,
        reflected_correction=reflected, lhs=alpha*abs(reflected-p0),
        rhs=.5*abs(y-x), final=x+alpha*reflected)
    assert old_certificate["lhs"] <= old_certificate["rhs"]
    assert abs(old_certificate["final"]) > abs(x)
    receipt = dict(schema=1, cohort="fixed_public_API_software_controls",
                   protocol_seed=0, learned_quality_claim=False,
                   candidate="bispf-secant-extragradient-v1",
                   quadratic=quadratic, frozen_opponent_bilinear=bilinear,
                   rejected_protected_field_limit=old_certificate,
                   process_cpu_seconds=time.process_time()-start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output), norm_ratio=quadratic["norm_ratio"],
                          cpu_seconds=receipt["process_cpu_seconds"])))


if __name__ == "__main__":
    main()
