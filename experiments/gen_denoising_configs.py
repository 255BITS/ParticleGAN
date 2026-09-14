#!/usr/bin/env python
"""Materialize the declared GAN / DDGAN factorial as repeatable YAML files."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_denoising import DEFAULTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["smoke", "screen"], default="screen")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--steps", type=int)
    args = p.parse_args()
    seeds = args.seeds or ([23999] if args.stage == "smoke" else [24001, 24002, 24003])
    directory = ROOT / "configs" / "denoising" / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    cells = []
    if args.stage == "smoke":
        cells = [("gan", "concat", "learned", "gaussian"), ("ddgan", "ucd", "learned", "learned")]
    else:
        for model, disc, prior in itertools.product(["gan", "ddgan"], ["concat", "ucd"], ["gaussian", "learned"]):
            for noise in (["gaussian"] if model == "gan" else ["gaussian", "fixed", "learned"]):
                cells.append((model, disc, prior, noise))
    paths = []
    for (model, disc, prior, noise), seed in itertools.product(cells, seeds):
        name = f"{model}_{disc}_p{prior}_n{noise}_s{seed}"
        cfg = {**DEFAULTS, "model": model, "d_mode": disc, "prior": prior, "noise": noise, "seed": seed,
               # Historical initial screen; corrected 20k runs have their own configs.
               "steps": 7000, "num_particles": 4096,
               "out_dir": f"results/denoising/{args.stage}/{name}"}
        if args.stage == "smoke":
            cfg.update(steps=200, eval_interval=100, eval_samples=2048, final_samples=4096, probe_samples=64)
        if args.steps:
            cfg["steps"] = args.steps
        path = directory / f"{name}.yaml"
        if path.exists() and path.read_text() != yaml.safe_dump(cfg, sort_keys=True):
            raise FileExistsError(f"refusing to replace different config {path}")
        path.write_text(yaml.safe_dump(cfg, sort_keys=True))
        paths.append(str(path.relative_to(ROOT)))
    (directory / "manifest.json").write_text(json.dumps(paths, indent=2) + "\n")
    print(f"{len(paths)} configs: {directory / 'manifest.json'}")


if __name__ == "__main__":
    main()
