"""Fixed-seed comparisons of existing formulations on the two missed targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch

import torch

from particlegan import GANLoss, GradientPenalty, get_recipe
from . import mode_hold, trajectory, two_pole


def candidates():
    base = get_recipe("gan_legacy").replace(name="gan")
    return [
        ("base_core", "Stock Recipe('gan') loss + penalty; original host settings", {}, {}, base.make_loss, base.make_gradient_penalty),
        ("r1_r2", "Zero-centered R1+R2, coeff 0.02", {}, {}, None, partial(GradientPenalty, "a_r1r2", coeff=0.02)),
        ("eikonal", "Two-sided unit-slope penalty, coeff 1", {}, {}, None, partial(GradientPenalty, "c_eikonal", coeff=1.0)),
        ("interpolation_cap", "Cap between real/fake samples, coeff 1", {}, {}, None, partial(GradientPenalty, "g_interp_cap", coeff=1.0)),
        ("relativistic_average", "Ra logistic with b_cap", {}, {}, partial(GANLoss, "logistic", "ra"), None),
        ("rp_hinge", "Rp hinge with b_cap", {}, {}, partial(GANLoss, "hinge", "rp"), None),
        ("no_vicreg", "Remove host VICReg 0.05", {"vicreg_weight": 0.0}, {"vicreg_weight": 0.0}, None, None),
        ("base_prior_weight", "Stock VICReg weight 1; keep tiny host cloud", {"vicreg_weight": base.prior_reg}, {"vicreg_weight": base.prior_reg}, None, None),
        ("no_particle_l2", "Remove host particle L2 0.02", {"particle_l2": 0.0}, {"particle_l2": 0.0}, None, None),
        ("no_cover", "Remove set-cover loss (trajectory only)", {"cover_weight": 0.0}, None, None, None),
        ("music_cover", "Set-cover weight 1.0 (trajectory only)", {"cover_weight": 1.0}, None, None, None),
        ("stronger_cap", "b_cap coefficient 10", {}, {}, None, partial(GradientPenalty, "b_cap", coeff=10.0)),
        ("base_regularization", "Stock VICReg 1, no particle L2 or cover; retain host model/budget/cloud", {"vicreg_weight": 1.0, "particle_l2": 0.0, "cover_weight": 0.0}, {"vicreg_weight": 1.0, "particle_l2": 0.0}, None, None),
        ("no_prior_regularization", "Remove both VICReg and particle L2", {"vicreg_weight": 0.0, "particle_l2": 0.0}, {"vicreg_weight": 0.0, "particle_l2": 0.0}, None, None),
        ("r1_r2_0_1", "R1+R2 coeff 0.1 to test the two-pole slope tradeoff", {}, {}, None, partial(GradientPenalty, "a_r1r2", coeff=0.1)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/locked_shared/investigation.json"))
    parser.add_argument("--only", nargs="*", help="run only the named candidate(s)")
    parser.add_argument("--resume", action="store_true", help="reuse completed ring rows; fill missing diagnostics")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"torch": torch.__version__, "seed": 0, "base_recipe": asdict(get_recipe("gan_legacy").replace(name="gan")),
              "budget": {"trajectory": 400, "ring": 1200}, "rows": []}
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text())
        if report["torch"] != torch.__version__:
            parser.error("cannot resume results from a different PyTorch runtime")
    previous = {row["name"]: row for row in report["rows"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, description, traj_options, ring_options, gan_factory, cap_factory in candidates():
        if args.only and name not in args.only:
            continue
        print(f"START {name}: {description}", flush=True)
        row = previous.get(name, {"name": name, "description": description, "trajectory_options": traj_options, "ring_options": ring_options})
        if row["trajectory_options"] != traj_options or row["ring_options"] != ring_options or row["description"] != description:
            parser.error(f"candidate {name} changed since saved run")
        if traj_options is not None and "critic_gradient_median" not in row.get("trajectory", {}):
            with patch.dict(trajectory.PROTOCOL, traj_options):
                row["trajectory"] = trajectory.train(gan_factory=gan_factory, cap_factory=cap_factory, diagnostics=True)
        if ring_options is not None and "ring" not in row:
            row["ring"] = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(**ring_options),
                gan_factory=gan_factory, cap_factory=cap_factory, diagnostics=True)
        # No VICReg/cover loss is present in this host; those controls leave it unchanged.
        if "two_pole" not in row:
            row["two_pole"] = two_pole.train(gan_factory=gan_factory, cap_factory=cap_factory,
                                            particle_l2=(traj_options or {}).get("particle_l2"))
        if name not in previous:
            report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        summary = {k: v for k, v in row.items() if k not in ("trajectory_options", "ring_options")}
        if "ring" in summary:
            summary["ring"] = {k: v for k, v in summary["ring"].items() if k != "curve"}
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
