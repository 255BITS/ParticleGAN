#!/usr/bin/env python
"""Generate or run a matched three-way prior comparison on unused study seeds.

python experiments/compare_priors.py --study-dir runs/prior_comparison --run --device cuda:0

Omit --run to generate only. Generation writes one YAML config per run plus a manifest. --run runs
those configs sequentially and writes comparison.json. It does not reuse or
overwrite completed runs; choose a new study directory for a fresh execution.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_arm import DEFAULTS  # noqa: E402

PRIORS = ("particles", "frozen_gaussian", "fresh_gaussian")
SEEDS = (23001, 23002, 23003)


def generate(study_dir: Path, seeds=SEEDS, total_steps=7000):
    """Only prior, output path and the explicitly paired seed vary across runs."""
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Provide distinct seeds and at least one seed")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    study_dir = study_dir.resolve()
    if (study_dir / "manifest.json").exists():
        raise ValueError("Study manifest already exists; choose a new study directory")
    (study_dir / "configs").mkdir(parents=True, exist_ok=True)
    recipe = dict(DEFAULTS, arm="b_cap", coeff=1.0, lr=6e-4,
                  total_steps=total_steps, spectral=False)
    entries = []
    for seed in seeds:
        for prior in PRIORS:
            name = f"{prior}_seed{seed}"
            out_dir = study_dir / "runs" / name
            config_path = study_dir / "configs" / f"{name}.yaml"
            cfg = dict(recipe, prior=prior, seed=int(seed), out_dir=str(out_dir))
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=True))
            entries.append(dict(prior=prior, seed=int(seed), config=str(config_path),
                                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                                out_dir=str(out_dir)))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              check=True, text=True, capture_output=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               check=True, text=True, capture_output=True).stdout.strip())
    runtime = dict(python=sys.version, torch_cuda=torch.version.cuda,
                   packages={name: version(name) for name in
                             ("torch", "numpy", "matplotlib", "PyYAML", "POT")})
    manifest = dict(schema_version=1, source_revision=revision, source_dirty=dirty,
                    runtime=runtime,
                    rng_scheme="separate_data_latent_penalty_v1", seeds=list(seeds),
                    priors=list(PRIORS), runs=entries)
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def collect(manifest):
    rows = []
    for run in manifest["runs"]:
        config_bytes = Path(run["config"]).read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != run["config_sha256"]:
            raise ValueError(f"Config changed since generation: {run['config']}")
        expected = yaml.safe_load(config_bytes)
        summary = json.loads((Path(run["out_dir"]) / "summary.json").read_text())
        if (summary["config"] != expected or expected["prior"] != run["prior"]
                or expected["seed"] != run["seed"] or expected["out_dir"] != run["out_dir"]):
            raise ValueError(f"Summary/config mismatch for {run['out_dir']}")
        rows.append(dict(prior=run["prior"], seed=run["seed"],
                         final=summary["final"], collapse_events=summary["collapse_events"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("runs/prior_comparison"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--total-steps", type=int, default=7000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--collect", action="store_true", help="Collect an existing manifest's results")
    args = parser.parse_args()
    if args.collect:
        manifest = json.loads((args.study_dir / "manifest.json").read_text())
    else:
        # A manifest defines the original run settings and source revision.
        # Never replace it with different seeds or settings after execution.
        if (args.study_dir / "manifest.json").exists():
            parser.error("Study manifest already exists; use a new --study-dir or --collect")
        manifest = generate(args.study_dir, args.seeds, args.total_steps)
        print(f"Generated {len(manifest['runs'])} configs in {args.study_dir}")
        if args.run:
            for run in manifest["runs"]:
                if (Path(run["out_dir"]) / "summary.json").exists():
                    parser.error(f"Run already completed: {run['out_dir']}")
                command = [sys.executable, str(ROOT / "experiments/train_arm.py"),
                           "--config", run["config"]]
                if args.device:
                    command += ["--device", args.device]
                subprocess.run(command, cwd=ROOT, check=True)
    if args.run or args.collect:
        path = args.study_dir / "comparison.json"
        path.write_text(json.dumps(collect(manifest), indent=2) + "\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
