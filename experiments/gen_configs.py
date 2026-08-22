#!/usr/bin/env python
"""
gen_configs.py

Config generator for the GAN gradient-regularizer study.

Every run of the study is described by a single YAML file that is handed to
`experiments/train_arm.py --config <path>`. This script materializes those
YAML files for the stages of the study:

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
  * ``audit``   -- re-runs of the key groups with the *same run names* but
                   ``out_dir`` under ``results/runs_audit/`` and the spectral
                   probe switched off. Training is deterministic, so these
                   reproduce the original models; the point is to pick up the
                   new per-mode audit metrics (per_mode_std, center_bias, ...)
                   that the original summaries predate. Written into a
                   ``audit/`` subdirectory so globs can target them alone.
  * ``anneal``  -- ``target_anneal`` in {linear, delayed} for the two arms
                   whose penalty has a moving target.
  * ``wgan``    -- the same arms under a Wasserstein objective
                   (``loss_type: wasserstein``). A different objective, so
                   these are never pooled with the logistic runs.
  * ``dualnorm``-- ``norm`` in {l1, linf} for b_cap, at lr_mult 1.0 and 2.0.
  * ``oadam``   -- ``optimizer: oadam`` (optimistic Adam) for a few groups.

Run names (and therefore config filenames and out_dirs) follow

    [wgan_]{arm}_c{coeff}[_{variant}]*_s{seed}

with the coefficient formatted with 'p' in place of the decimal point
(0.02 -> ``c0p02``). For the ``lazy`` stage both variants are named with the
*nominal* coefficient so that the pair lines up in the analysis, even though
the ``c16th`` variant actually writes coeff/16 into its YAML.

Trainer-default hygiene: the newer trainer keys (``loss_type``, ``optimizer``,
``norm``, ``target_anneal``) are written into a YAML *only* when the stage
needs a non-default value. The older keys (arm/coeff/kappa/.../out_dir) are
always written, which keeps the pre-existing configs byte-for-byte stable.

Usage:

    python experiments/gen_configs.py --stage main
    python experiments/gen_configs.py --stage lr_sens \
        --arm_coeffs "a_r1r2:0.02,c_eikonal:0.1,d_asym:0.1"
    python experiments/gen_configs.py --stage lazy \
        --arm_coeffs "a_r1r2:0.02,c_eikonal:0.1"
    python experiments/gen_configs.py --stage audit
    python experiments/gen_configs.py --stage anneal
    python experiments/gen_configs.py --stage wgan
    python experiments/gen_configs.py --stage dualnorm
    python experiments/gen_configs.py --stage oadam

Writing is idempotent: existing config files are silently overwritten.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# The five regularized arms plus the unregularized control.
ARMS = ["a_r1r2", "b_cap", "c_eikonal", "d_asym", "e_interp"]
CONTROL_ARM = "f_none"

COEFFS = [0.005, 0.02, 0.1, 1.0]
SEEDS = [1, 2, 3, 4, 5]

LR_MULTS = [0.5, 2.0]
LAZY_K = 16

# Trainer defaults for the newer config keys. A key is written into a YAML
# only when the stage asks for something other than the value below, so the
# pre-existing configs stay byte-identical when they are regenerated.
NEW_KEY_DEFAULTS = {
    "loss_type": "logistic",
    "optimizer": "adam",
    "norm": "l2",
    "target_anneal": "none",
}

# --- audit stage -----------------------------------------------------------
# (arm, coeff, lr_mult) of the groups worth re-running under the new per-mode
# metrics. Names match the originals exactly; only out_dir and spectral differ.
AUDIT_GROUPS: List[Tuple[str, float, float]] = [
    ("a_r1r2", 0.02, 1.0),
    ("a_r1r2", 0.1, 1.0),
    ("a_r1r2", 0.3, 1.0),
    ("a_r1r2", 1.0, 1.0),
    ("a_r1r2", 0.1, 2.0),
    ("b_cap", 1.0, 1.0),
    ("b_cap", 1.0, 2.0),
    ("c_eikonal", 0.1, 1.0),
    ("c_eikonal", 0.1, 2.0),
    ("d_asym", 1.0, 2.0),
    ("e_interp", 1.0, 1.0),
    (CONTROL_ARM, 0.0, 1.0),
]

# --- anneal stage ----------------------------------------------------------
ANNEAL_ARM_COEFFS: List[Tuple[str, float]] = [("b_cap", 1.0), ("c_eikonal", 0.1)]
ANNEAL_MODES = {"lin": "linear", "del": "delayed"}

# --- wgan stage ------------------------------------------------------------
WGAN_ARMS = ["a_r1r2", "b_cap", "c_eikonal", "e_interp"]
WGAN_COEFFS = [0.1, 1.0]

# --- dualnorm stage --------------------------------------------------------
DUALNORM_ARM_COEFF: Tuple[str, float] = ("b_cap", 1.0)
DUALNORM_NORMS = ["l1", "linf"]
DUALNORM_LR_MULTS = [1.0, 2.0]

# --- oadam stage -----------------------------------------------------------
OADAM_ARM_COEFFS: List[Tuple[str, float]] = [
    (CONTROL_ARM, 0.0),
    ("a_r1r2", 0.1),
    ("a_r1r2", 1.0),
    ("b_cap", 1.0),
]

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
    spectral: bool = True,
    out_root: str = "results/runs",
    extra: Optional[Dict] = None,
) -> Dict:
    """
    Build one config dict for a single run.

    ``extra`` carries the newer trainer keys (loss_type / optimizer / norm /
    target_anneal). Entries equal to the trainer default are dropped, so a
    stage that does not move a key never writes it and old-style configs keep
    exactly the key set they have always had. The surviving extras are
    inserted just before ``out_dir`` to keep the emitted YAML readable.
    """
    cfg = dict(BASE_CONFIG)
    cfg["arm"] = arm
    cfg["coeff"] = float(coeff)
    cfg["lazy_k"] = int(lazy_k)
    cfg["lr_mult"] = float(lr_mult)
    cfg["seed"] = int(seed)
    cfg["spectral"] = bool(spectral)
    cfg["out_dir"] = f"{out_root.rstrip('/')}/{run_name}"

    non_default = {
        k: v
        for k, v in (extra or {}).items()
        if NEW_KEY_DEFAULTS.get(k, object()) != v
    }
    if not non_default:
        return cfg
    out: Dict = {}
    for k, v in cfg.items():
        if k == "out_dir":
            out.update(non_default)
        out[k] = v
    return out


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


def stage_audit(config_dir: Path) -> List[Path]:
    """
    Re-run the key groups under ``results/runs_audit/`` for the new per-mode
    audit metrics.

    Run names are *identical* to the originals: training is deterministic, so
    an audit run reproduces the very same model and its extra metrics can be
    attributed to the original group. Only ``out_dir`` differs, plus
    ``spectral: false`` -- the Jacobian probe is the expensive part of the run
    and the original spectral numbers are already on disk.

    Configs land in ``<config_dir>/audit/`` so a launcher can glob them
    separately from the main study.
    """
    audit_dir = config_dir / "audit"
    written: List[Path] = []
    for arm, coeff, lr_mult in AUDIT_GROUPS:
        suffix = "" if lr_mult == 1.0 else f"_lr{fmt_float(lr_mult)}"
        for seed in SEEDS:
            run_name = f"{arm}_c{fmt_float(coeff)}{suffix}_s{seed}"
            cfg = make_config(
                arm,
                coeff,
                seed,
                run_name,
                lr_mult=lr_mult,
                spectral=False,
                out_root="results/runs_audit",
            )
            written.append(write_config(cfg, run_name, audit_dir))
    return written


def stage_anneal(config_dir: Path) -> List[Path]:
    """Annealed penalty targets: ``target_anneal`` in {linear, delayed}."""
    written: List[Path] = []
    for arm, coeff in ANNEAL_ARM_COEFFS:
        for tag, mode in ANNEAL_MODES.items():
            for seed in SEEDS:
                run_name = f"{arm}_c{fmt_float(coeff)}_ann{tag}_s{seed}"
                cfg = make_config(
                    arm, coeff, seed, run_name, extra={"target_anneal": mode}
                )
                written.append(write_config(cfg, run_name, config_dir))
    return written


def stage_wgan(config_dir: Path) -> List[Path]:
    """
    The same arms under a Wasserstein objective.

    The ``wgan_`` prefix is load-bearing: the analysis keys off it to keep
    these groups out of every logistic-objective comparison and board.
    """
    written: List[Path] = []
    extra = {"loss_type": "wasserstein"}
    for arm in WGAN_ARMS:
        for coeff in WGAN_COEFFS:
            for seed in SEEDS:
                run_name = f"wgan_{arm}_c{fmt_float(coeff)}_s{seed}"
                cfg = make_config(arm, coeff, seed, run_name, extra=extra)
                written.append(write_config(cfg, run_name, config_dir))
    for seed in SEEDS:
        run_name = f"wgan_{CONTROL_ARM}_c{fmt_float(0.0)}_s{seed}"
        cfg = make_config(CONTROL_ARM, 0.0, seed, run_name, extra=extra)
        written.append(write_config(cfg, run_name, config_dir))
    return written


def stage_dualnorm(config_dir: Path) -> List[Path]:
    """b_cap's penalty measured in the l1 and linf norms instead of l2."""
    arm, coeff = DUALNORM_ARM_COEFF
    written: List[Path] = []
    for norm in DUALNORM_NORMS:
        for lr_mult in DUALNORM_LR_MULTS:
            suffix = "" if lr_mult == 1.0 else f"_lr{fmt_float(lr_mult)}"
            for seed in SEEDS:
                run_name = (
                    f"{arm}_c{fmt_float(coeff)}_n{norm}{suffix}_s{seed}"
                )
                cfg = make_config(
                    arm,
                    coeff,
                    seed,
                    run_name,
                    lr_mult=lr_mult,
                    extra={"norm": norm},
                )
                written.append(write_config(cfg, run_name, config_dir))
    return written


def stage_oadam(config_dir: Path) -> List[Path]:
    """Optimistic Adam for the control and the strongest a_r1r2 / b_cap groups."""
    written: List[Path] = []
    for arm, coeff in OADAM_ARM_COEFFS:
        for seed in SEEDS:
            run_name = f"{arm}_c{fmt_float(coeff)}_oadam_s{seed}"
            cfg = make_config(
                arm, coeff, seed, run_name, extra={"optimizer": "oadam"}
            )
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
        choices=[
            "main",
            "lr_sens",
            "lazy",
            "audit",
            "anneal",
            "wgan",
            "dualnorm",
            "oadam",
        ],
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

    # Stages with a fixed, curated group list take no --arm_coeffs.
    fixed_stages = {
        "main": lambda: stage_main(config_dir),
        "audit": lambda: stage_audit(config_dir),
        "anneal": lambda: stage_anneal(config_dir),
        "wgan": lambda: stage_wgan(config_dir),
        "dualnorm": lambda: stage_dualnorm(config_dir),
        "oadam": lambda: stage_oadam(config_dir),
    }

    if args.stage in fixed_stages:
        written = fixed_stages[args.stage]()
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
