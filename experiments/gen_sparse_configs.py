#!/usr/bin/env python
"""
gen_sparse_configs.py

Config generator for the sparse conditional mixed-output study
(experiments/train_sparse.py). One YAML per run under

    configs/sparse/<stage>/<group>_s<seed>.yaml   ->   results/sparse/runs/<stage>/<group>_s<seed>

Only non-default keys are written (plus seed / total_steps / out_dir), so a
config keeps meaning the same thing as trainer defaults acquire new knobs.

Every stage except `recipe` is generated on top of BASE, the recipe the probe
rounds found (reports/sparse-ucd/FINDINGS.md): the class reaches G *only*
through a class-partitioned particle table (no class embedding), and the
gradient penalty is the one-sided cap on real/fake interpolates. `--base`
overrides / extends it.

Stages (each answers one question, 3 seeds per cell):

  smoke     one short run per D mode: does everything execute.
  recipe    DOES IT CONVERGE, AND WHAT IS LOAD-BEARING. The 100-Gaussians
            champion as-is (sample-point cap, class embedding in G) against
            each change in isolation and in combination, plus the
            unconditional-G floor and ceiling.
  ucd       THE UCD QUESTION. D mode in {scalar, concat, proj, ucd} with a
            lambda_1 sweep, the frozen-Gaussian prior, embedding-only and
            embedding+partition conditioning, the supervised-symbol anchor,
            and the x-only penalty.
  sparse    THE SPARSITY QUESTION. real_head in {linear, gated x lambda_sp, topk}.
  discrete  THE DISCRETE-OUTPUT QUESTION. cat_mode x tau schedule x
            lambda_sym, plus the 'split' symbol map (K = 2C).
  fewshot   THE DATA-SPARSITY QUESTION. n_train in {128, 512, 2048, 8192}
            (2 .. 128 samples per mode), and the frozen prior at 512.

Usage:
    python experiments/gen_sparse_configs.py --stage recipe
    python experiments/gen_sparse_configs.py --stage sparse --base "coeff=0.3"
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

SEEDS = (1, 2, 3)
TOTAL_STEPS = 5000
CONFIG_ROOT = Path("configs/sparse")
RUNS_ROOT = Path("results/sparse/runs")

# The recipe every non-`recipe` stage builds on (see module docstring).
BASE: Dict = {
    "emb_dim": 0,
    "prior_partition": "class",
    "arm": "g_interp_cap",
    "coeff": 1.0,
    # probe round 4/5: wider nets and a wider latent each bought ~5 modes; the
    # Fourier ramp keeps the core width honest (0.8-0.9 vs 0.5-0.6 without it)
    "hidden": 256,
    "z_dim": 16,
    "fourier_ramp_start": 0.3,
    "fourier_ramp_end": 0.7,
}
# BASE minus the capacity/schedule additions: the two structural changes alone.
BASE_MIN: Dict = {"emb_dim": 0, "prior_partition": "class", "arm": "g_interp_cap", "coeff": 1.0}

CHAMPION: Dict = {"emb_dim": 8, "prior_partition": "none", "arm": "b_cap", "coeff": 1.0}


def parse_kv(s: str) -> Dict:
    out: Dict = {}
    for kv in [p for p in s.split(",") if p.strip()]:
        k, v = kv.split("=", 1)
        out[k.strip()] = yaml.safe_load(v.strip())
    return out


def stage_groups(stage: str) -> List[Tuple[str, Dict]]:
    """(group name, overrides). `recipe` overrides are absolute; the rest sit on BASE."""
    g: List[Tuple[str, Dict]] = []
    if stage == "smoke":
        for dm in ("scalar", "concat", "proj", "ucd"):
            g.append((f"smoke_{dm}", {"d_mode": dm, "total_steps": 300, "eval_interval": 100}))
        return g

    if stage == "recipe":
        ch = dict(CHAMPION)
        g.append(("champion", dict(ch)))                                             # as shipped
        g.append(("champion_f0", {**ch, "fourier": 0}))                             # plain-MLP D
        g.append(("champion_r1r2", {**ch, "arm": "a_r1r2", "coeff": 1.0}))          # zero-centred, strong
        g.append(("champion_einterp", {**ch, "arm": "e_interp", "coeff": 1.0}))     # two-sided interpolate
        g.append(("emb_icap", {**ch, "arm": "g_interp_cap"}))                       # change penalty only
        g.append(("emb_icap_ramp", {**ch, "arm": "g_interp_cap", "fourier_ramp_start": 0.3, "fourier_ramp_end": 0.7}))
        g.append(("pp_bcap", {**ch, "emb_dim": 0, "prior_partition": "class"}))     # change conditioning only
        g.append(("pp_icap", dict(BASE_MIN)))                                       # both structural changes
        g.append(("pp_icap_ramp", {**BASE_MIN, "fourier_ramp_start": 0.3, "fourier_ramp_end": 0.7}))
        g.append(("pp_icap_f0", {**BASE_MIN, "fourier": 0}))
        g.append(("pp_icap_h256", {**BASE_MIN, "hidden": 256}))
        g.append(("pp_icap_z16", {**BASE_MIN, "z_dim": 16}))
        g.append(("base", dict(BASE)))                                              # = every other stage's base
        g.append(("base_f0", {**BASE, "fourier": 0, "fourier_ramp_start": 0.0, "fourier_ramp_end": 0.0}))
        g.append(("base_8k", {**BASE, "total_steps": 8000}))
        g.append(("uncond_floor", {**ch, "emb_dim": 0, "d_mode": "scalar"}))        # no class anywhere
        g.append(("uncond_ceiling", {**ch, "emb_dim": 0, "arm": "g_interp_cap"}))   # G unconditional, D ucd
        return g

    if stage == "ucd":
        g.append(("d_scalar", {"d_mode": "scalar"}))
        g.append(("d_concat", {"d_mode": "concat"}))
        g.append(("d_proj", {"d_mode": "proj"}))
        for lam, tag in ((0.02, "0p02"), (0.1, "0p1"), (0.5, "0p5"), (2.0, "2p0")):
            g.append((f"ucd_l{tag}", {"d_mode": "ucd", "ucd_lambda": lam}))
        g.append(("ucd_gauss", {"prior": "gaussian"}))
        g.append(("ucd_emb_only", {"emb_dim": 8, "prior_partition": "none"}))
        g.append(("ucd_emb_pp", {"emb_dim": 8}))
        g.append(("ucd_symsup", {"lambda_sym": 1.0}))
        g.append(("ucd_gpx", {"gp_on_y": False}))
        return g

    if stage == "sparse":
        g.append(("lin", {}))
        for sp, tag in ((0.0, "0"), (0.01, "0p01"), (0.1, "0p1"), (1.0, "1p0")):
            g.append((f"gated_sp{tag}", {"real_head": "gated", "lambda_sp": sp}))
        g.append(("topk", {"real_head": "topk"}))
        # gate warm-up: linear head for the first 40% of the run, then the gate engages
        g.append(("gated_sp0p01_warm", {"real_head": "gated", "lambda_sp": 0.01, "gate_start_frac": 0.4}))
        g.append(("gated_sp0_warm", {"real_head": "gated", "lambda_sp": 0.0, "gate_start_frac": 0.4}))
        g.append(("topk_warm", {"real_head": "topk", "gate_start_frac": 0.4}))
        return g

    if stage == "discrete":
        g.append(("gst_tau1", {"cat_mode": "gumbel_st"}))
        g.append(("gst_anneal", {"cat_mode": "gumbel_st", "tau_end": 0.1}))
        g.append(("gsoft_anneal", {"cat_mode": "gumbel_soft", "tau_end": 0.1}))
        g.append(("starg", {"cat_mode": "st_argmax"}))
        g.append(("soft", {"cat_mode": "soft"}))
        g.append(("soft_anneal", {"cat_mode": "soft", "tau_end": 0.1}))
        g.append(("gst_symsup", {"cat_mode": "gumbel_st", "lambda_sym": 1.0}))
        g.append(("gst_split", {"cat_mode": "gumbel_st", "symbol_map": "split", "n_symbols": 16}))
        g.append(("soft_split", {"cat_mode": "soft", "symbol_map": "split", "n_symbols": 16}))
        return g

    if stage == "fewshot":
        for n in (128, 512, 2048, 8192):
            g.append((f"n{n}", {"n_train": n}))
        g.append(("n512_gauss", {"n_train": 512, "prior": "gaussian"}))
        return g

    raise ValueError(f"unknown stage {stage!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "recipe", "ucd", "sparse", "discrete", "fewshot"])
    ap.add_argument("--base", type=str, default="", help="comma-separated key=value merged over BASE for every config")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    args = ap.parse_args()

    extra = parse_kv(args.base)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.stage == "smoke":
        seeds = seeds[:1]
    cfg_dir = CONFIG_ROOT / args.stage
    cfg_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, over in stage_groups(args.stage):
        for seed in seeds:
            cfg: Dict = {} if args.stage in ("smoke", "recipe") else dict(BASE)
            cfg.update(extra)
            cfg.update(over)
            cfg.setdefault("total_steps", args.total_steps)
            cfg["seed"] = seed
            cfg["out_dir"] = str(RUNS_ROOT / args.stage / f"{name}_s{seed}")
            with open(cfg_dir / f"{name}_s{seed}.yaml", "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=True)
            n += 1
    print(f"[gen_sparse_configs] stage={args.stage} base={ {**BASE, **extra} if args.stage not in ('smoke', 'recipe') else extra or '{}'} -> {n} configs in {cfg_dir}/")


if __name__ == "__main__":
    main()
