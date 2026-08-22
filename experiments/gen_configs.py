#!/usr/bin/env python
"""
gen_configs.py

Config generator for the GAN gradient-regularizer study.

Every run of the study is described by a single YAML file that is handed to
`experiments/train_arm.py --config <path>`. This script materializes those
YAML files for the three stages of the study:

  * ``main``    -- the primary grid: 5 regularizer arms x 4 coefficients x
                   5 seeds, plus the unregularized ``f_none`` control x 5
                   seeds (105 configs).
  * ``lr_sens`` -- learning-rate sensitivity for a handful of
                   (arm, coeff) winners: lr_mult in {0.5, 2.0} x 5 seeds.
                   lr_mult 1.0 is already covered by ``main``.
  * ``lazy``    -- the lazy-regularization pitfall: for each (arm, coeff),
                   (i) apply the penalty every 16 steps at the same *nominal*
                   coefficient (the trainer rescales by 16 internally), and
                   (ii) apply it every step at coeff/16. These are only
                   equivalent to first order; the point of the stage is to
                   measure how far apart they actually land.

Run names (and therefore config filenames and out_dirs) follow

    {arm}_c{coeff}[_{variant}]_s{seed}

with the coefficient formatted with 'p' in place of the decimal point
(0.02 -> ``c0p02``). For the ``lazy`` stage both variants are named with the
*nominal* coefficient so that the pair lines up in the analysis, even though
the ``c16th`` variant actually writes coeff/16 into its YAML.

Usage:

    python experiments/gen_configs.py --stage main
    python experiments/gen_configs.py --stage lr_sens \
        --arm_coeffs "a_r1r2:0.02,c_eikonal:0.1,d_asym:0.1"
    python experiments/gen_configs.py --stage lazy \
        --arm_coeffs "a_r1r2:0.02,c_eikonal:0.1"

Writing is idempotent: existing config files are silently overwritten.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# The five regularized arms plus the unregularized control.
ARMS = ["a_r1r2", "b_cap", "c_eikonal", "d_asym", "e_interp"]
CONTROL_ARM = "f_none"

COEFFS = [0.005, 0.02, 0.1, 1.0]
SEEDS = [1, 2, 3, 4, 5]

LR_MULTS = [0.5, 2.0]
LAZY_K = 16

# Shared defaults; every stage overrides only what it needs to.
BASE_CONFIG = {
    "arm": None,
    "coeff": 0.0,
    "kappa": 1.0,
    "lazy_k": 1,
    "seed": 1,
    "total_steps": 7000,
    "eval_interval": 100,
    "lr": 3e-4,
    "d_lr_mult": 1.5,
    "lr_mult": 1.0,
    "spectral": True,
    "out_dir": None,
}


def fmt_float(x: float) -> str:
    """
    Format a float for use inside a run name.

    Uses the shortest round-trippable-ish representation, always keeps at
    least one decimal place, and swaps '.' for 'p' so the result is safe in
    a directory name:

        0.005 -> '0p005'
        0.02  -> '0p02'
        1.0   -> '1p0'
        0.00125 -> '0p00125'
    """
    s = f"{x:.10g}"
    if "e" in s or "E" in s:
        # Fall back to a fixed-point form for very small coefficients so we
        # never end up with 'c1e-05' style names.
        s = f"{x:.10f}".rstrip("0")
        if s.endswith("."):
            s += "0"
    if "." not in s:
        s += ".0"
    return s.replace(".", "p").replace("-", "m")


def parse_arm_coeffs(spec: str) -> List[Tuple[str, float]]:
    """
    Parse an ``--arm_coeffs`` string like "a_r1r2:0.02,c_eikonal:0.1".

    Returns a list of (arm, coeff) pairs in the order given.
    """
    pairs: List[Tuple[str, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Malformed --arm_coeffs entry {chunk!r}; expected 'arm:coeff'"
            )
        arm, coeff_str = chunk.split(":", 1)
        arm = arm.strip()
        if arm not in ARMS and arm != CONTROL_ARM:
            raise ValueError(
                f"Unknown arm {arm!r}; expected one of {ARMS + [CONTROL_ARM]}"
            )
        pairs.append((arm, float(coeff_str)))
    if not pairs:
        raise ValueError("--arm_coeffs did not contain any entries")
    return pairs


def make_config(
    arm: str,
    coeff: float,
    seed: int,
    run_name: str,
    *,
    lazy_k: int = 1,
    lr_mult: float = 1.0,
) -> Dict:
    """Build one config dict for a single run."""
    cfg = dict(BASE_CONFIG)
    cfg["arm"] = arm
    cfg["coeff"] = float(coeff)
    cfg["lazy_k"] = int(lazy_k)
    cfg["lr_mult"] = float(lr_mult)
    cfg["seed"] = int(seed)
    cfg["out_dir"] = f"results/runs/{run_name}"
    return cfg


def write_config(cfg: Dict, run_name: str, config_dir: Path) -> Path:
    """Write one config YAML; overwrites any existing file of the same name."""
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{run_name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path


def stage_main(config_dir: Path) -> List[Path]:
    """The primary grid: arms x coeffs x seeds, plus the f_none control."""
    written: List[Path] = []
    for arm in ARMS:
        for coeff in COEFFS:
            for seed in SEEDS:
                run_name = f"{arm}_c{fmt_float(coeff)}_s{seed}"
                cfg = make_config(arm, coeff, seed, run_name)
                written.append(write_config(cfg, run_name, config_dir))

    # Unregularized control. It has no coefficient, but we still stamp
    # c0p0 into the name so every run parses with the same pattern.
    for seed in SEEDS:
        run_name = f"{CONTROL_ARM}_c{fmt_float(0.0)}_s{seed}"
        cfg = make_config(CONTROL_ARM, 0.0, seed, run_name)
        written.append(write_config(cfg, run_name, config_dir))
    return written


def stage_lr_sens(config_dir: Path, arm_coeffs: List[Tuple[str, float]]) -> List[Path]:
    """LR sensitivity around the main grid (lr_mult 1.0 lives in ``main``)."""
    written: List[Path] = []
    for arm, coeff in arm_coeffs:
        for lr_mult in LR_MULTS:
            for seed in SEEDS:
                run_name = (
                    f"{arm}_c{fmt_float(coeff)}_lr{fmt_float(lr_mult)}_s{seed}"
                )
                cfg = make_config(arm, coeff, seed, run_name, lr_mult=lr_mult)
                written.append(write_config(cfg, run_name, config_dir))
    return written


def stage_lazy(config_dir: Path, arm_coeffs: List[Tuple[str, float]]) -> List[Path]:
    """
    The lazy-regularization pitfall.

    Two variants per (arm, coeff), both named with the *nominal* coefficient:

      * ``_lazy16``: lazy_k=16 at the nominal coeff. The trainer's regularizer
        multiplies the penalty by 16 internally on the steps where it fires,
        so this is the "same expected penalty, applied in bursts" arm.
      * ``_c16th``: lazy_k=1 at coeff/16, i.e. the same time-averaged penalty
        strength applied smoothly every step.
    """
    written: List[Path] = []
    for arm, coeff in arm_coeffs:
        nominal = fmt_float(coeff)
        for seed in SEEDS:
            run_name = f"{arm}_c{nominal}_lazy16_s{seed}"
            cfg = make_config(arm, coeff, seed, run_name, lazy_k=LAZY_K)
            written.append(write_config(cfg, run_name, config_dir))

        for seed in SEEDS:
            run_name = f"{arm}_c{nominal}_c16th_s{seed}"
            cfg = make_config(arm, coeff / LAZY_K, seed, run_name, lazy_k=1)
            written.append(write_config(cfg, run_name, config_dir))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate YAML configs for the regularizer study."
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["main", "lr_sens", "lazy"],
        help="Which block of configs to generate.",
    )
    parser.add_argument(
        "--arm_coeffs",
        type=str,
        default=None,
        help=(
            "For --stage lr_sens/lazy: comma-separated arm:coeff pairs, "
            "e.g. 'a_r1r2:0.02,c_eikonal:0.1,d_asym:0.1'."
        ),
    )
    parser.add_argument(
        "--best",
        type=str,
        default=None,
        help="Alias for --arm_coeffs (kept for convenience).",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory to write YAML configs into (default: configs).",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    spec = args.arm_coeffs or args.best

    if args.stage == "main":
        written = stage_main(config_dir)
    else:
        if not spec:
            parser.error(f"--stage {args.stage} requires --arm_coeffs (or --best)")
        arm_coeffs = parse_arm_coeffs(spec)
        if args.stage == "lr_sens":
            written = stage_lr_sens(config_dir, arm_coeffs)
        else:
            written = stage_lazy(config_dir, arm_coeffs)

    print(f"[gen_configs] stage={args.stage} wrote {len(written)} configs to {config_dir}")
    for path in written[:3]:
        print(f"[gen_configs]   e.g. {path}")
    if len(written) > 3:
        print(f"[gen_configs]   ... and {len(written) - 3} more")


if __name__ == "__main__":
    main()
