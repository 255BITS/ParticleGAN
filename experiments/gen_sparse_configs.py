#!/usr/bin/env python
"""
gen_sparse_configs.py

Config generator for the sparse conditional mixed-output study
(experiments/train_sparse.py). One YAML per run under

    configs/sparse/<stage>/<group>_s<seed>.yaml   ->   results/sparse/runs/<stage>/<group>_s<seed>

Only non-default keys are written (plus seed / total_steps / out_dir), so a
config keeps meaning the same thing as trainer defaults acquire new knobs.

Stages (each answers one question; later stages build on a `--base` recipe
that is meant to be updated from the earlier stage's table):

  smoke     one short run per D mode: does everything execute.
  baseline  THE UCD QUESTION. D mode in {scalar, concat, proj, ucd} with a
            lambda_1 sweep for ucd, plus the prior ablations (frozen Gaussian
            table; class-partitioned particle table) and the supervised-symbol
            anchor. Linear real head, straight-through Gumbel symbol.
  sparse    THE SPARSITY QUESTION. real_head in {linear, gated x lambda_sp,
            topk} on the base recipe.
  discrete  THE DISCRETE-OUTPUT QUESTION. cat_mode x tau schedule x
            lambda_sym, plus the 'split' symbol map (K = 2C, a real
            distribution over symbols per class).
  fewshot   THE DATA-SPARSITY QUESTION. n_train in {128, 512, 2048, 8192}
            (2 .. 128 samples per mode) on the base recipe, and the frozen
            prior at 512 for contrast.

Usage:
    python experiments/gen_sparse_configs.py --stage baseline
    python experiments/gen_sparse_configs.py --stage sparse --base "ucd_lambda=0.1"
    python experiments/gen_sparse_configs.py --stage discrete --base "ucd_lambda=0.1,real_head=gated,lambda_sp=0.01"
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

SEEDS = (1, 2, 3)
TOTAL_STEPS = 5000
CONFIG_ROOT = Path("configs/sparse")
RUNS_ROOT = Path("results/sparse/runs")


def parse_base(s: str) -> Dict:
    out: Dict = {}
    for kv in [p for p in s.split(",") if p.strip()]:
        k, v = kv.split("=", 1)
        out[k.strip()] = yaml.safe_load(v.strip())
    return out


def stage_groups(stage: str, base: Dict) -> List[Tuple[str, Dict]]:
    g: List[Tuple[str, Dict]] = []
    if stage == "smoke":
        for dm in ("scalar", "concat", "proj", "ucd"):
            g.append((f"smoke_{dm}", {"d_mode": dm, "total_steps": 300, "eval_interval": 100}))
        return g

    if stage == "baseline":
        g.append(("d_scalar", {"d_mode": "scalar"}))
        g.append(("d_concat", {"d_mode": "concat"}))
        g.append(("d_proj", {"d_mode": "proj"}))
        for lam, tag in ((0.02, "0p02"), (0.1, "0p1"), (0.5, "0p5"), (2.0, "2p0")):
            g.append((f"ucd_l{tag}", {"d_mode": "ucd", "ucd_lambda": lam}))
        g.append(("ucd_l0p1_gauss", {"d_mode": "ucd", "ucd_lambda": 0.1, "prior": "gaussian"}))
        g.append(("ucd_l0p1_pp", {"d_mode": "ucd", "ucd_lambda": 0.1, "prior_partition": "class"}))
        g.append(("ucd_l0p1_symsup", {"d_mode": "ucd", "ucd_lambda": 0.1, "lambda_sym": 1.0}))
        g.append(("ucd_l0p1_gpx", {"d_mode": "ucd", "ucd_lambda": 0.1, "gp_on_y": False}))
        return g

    if stage == "sparse":
        g.append(("lin", {}))
        for sp, tag in ((0.0, "0"), (0.01, "0p01"), (0.1, "0p1"), (1.0, "1p0")):
            g.append((f"gated_sp{tag}", {"real_head": "gated", "lambda_sp": sp}))
        g.append(("topk", {"real_head": "topk"}))
        g.append(("gated_sp0p01_pp", {"real_head": "gated", "lambda_sp": 0.01, "prior_partition": "class"}))
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
        g.append(("n512_pp", {"n_train": 512, "prior_partition": "class"}))
        return g

    raise ValueError(f"unknown stage {stage!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "baseline", "sparse", "discrete", "fewshot"])
    ap.add_argument("--base", type=str, default="", help="comma-separated key=value applied to every config in the stage")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    args = ap.parse_args()

    base = parse_base(args.base)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.stage == "smoke":
        seeds = seeds[:1]
    cfg_dir = CONFIG_ROOT / args.stage
    cfg_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, over in stage_groups(args.stage, base):
        for seed in seeds:
            cfg = dict(base)
            cfg.update(over)
            cfg.setdefault("total_steps", args.total_steps)
            cfg["seed"] = seed
            cfg["out_dir"] = str(RUNS_ROOT / args.stage / f"{name}_s{seed}")
            with open(cfg_dir / f"{name}_s{seed}.yaml", "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=True)
            n += 1
    print(f"[gen_sparse_configs] stage={args.stage} base={base or '{}'} -> {n} configs in {cfg_dir}/")


if __name__ == "__main__":
    main()
