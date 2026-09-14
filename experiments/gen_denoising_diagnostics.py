#!/usr/bin/env python
"""Targeted DDGAN diagnostics with the established optimizer/bcap recipe."""
import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_denoising import DEFAULTS, validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[24002])
    p.add_argument("--stage", choices=["diagnostics", "budget56k", "confirm28k"], default="diagnostics")
    args = p.parse_args()
    stage = args.stage
    variants = {
        "ddgan_concat_28k": {"steps": 28000},
        "ddgan_ucd_28k": {"steps": 28000, "d_mode": "ucd"},
        "ddgan_gaussian_28k": {"steps": 28000, "prior": "gaussian"},
        "ddgan_one_step_7k": {"alpha_bar": [1.0, .0001]},
        "ddgan_two_steps_14k": {"alpha_bar": [1.0, .5, .0001], "steps": 14000},
        "ddgan_class_free_28k": {"classes": 1, "steps": 28000},
        "gan_class_free_7k": {"model": "gan", "classes": 1},
        "gan_fixed_7k": {"model": "gan", "prior": "fixed"},
        "gan_learned_28k": {"model": "gan", "steps": 28000},
    }
    if stage == "budget56k":
        variants = {
            "ddgan_concat_gaussian_56k": {"prior": "gaussian"},
            "ddgan_concat_learned_56k": {},
            "ddgan_ucd_learned_56k": {"d_mode": "ucd"},
            "ddgan_ucd_learned_noise_56k": {"d_mode": "ucd", "noise": "learned"},
            "ddgan_ucd_fixed_noise_56k": {"d_mode": "ucd", "noise": "fixed"},
            "gan_learned_56k": {"model": "gan"},
        }
        variants = {name: {**cfg, "steps": 56000} for name, cfg in variants.items()}
    elif stage == "confirm28k":
        variants = {"ddgan_ucd_28k": {"steps": 28000, "d_mode": "ucd"}}
    directory = ROOT / "configs" / "denoising" / stage
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, overrides in variants.items():
        for seed in args.seeds:
            # Preserve these completed experimental designs as CLI defaults evolve.
            cfg = {**DEFAULTS, "d_mode": "concat", "steps": 7000,
                   "prior": "learned", "num_particles": 20000,
                   **overrides, "seed": seed,
                   "out_dir": f"results/denoising/{stage}/{name}_s{seed}"}
            validate(cfg)
            path = directory / f"{name}_s{seed}.yaml"
            content = yaml.safe_dump(cfg, sort_keys=True)
            if path.exists() and path.read_text() != content:
                raise FileExistsError(path)
            path.write_text(content)
            paths.append(str(path.relative_to(ROOT)))
    (directory / "manifest.json").write_text(json.dumps(paths, indent=2) + "\n")
    print(f"{len(paths)} configs: {directory / 'manifest.json'}")


if __name__ == "__main__":
    main()
