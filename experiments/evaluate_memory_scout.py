"""Evaluate a saved model using only G, its writer, and the learned particles."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.memory_scout import Config, Reader, Writer, evaluate
from particlegan import get_recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text()))
    assert cfg.resume, "config.resume must identify the checkpoint"
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    saved = torch.load(cfg.resume, map_location=args.device, weights_only=False)
    evaluation_keys = {"name", "steps", "resume", "eval_batch", "eval_steps", "log_every"}
    old = asdict(Config(**saved["config"]))
    assert all(old[k] == v for k, v in asdict(cfg).items() if k not in evaluation_keys)
    generator, writer = Reader(cfg).to(args.device), Writer(cfg.writer, cfg.memory_dim).to(args.device)
    recipe = get_recipe(num_particles=512, **cfg.recipe)
    prior = recipe.make_prior().to(args.device)
    generator.load_state_dict(saved["generator"])
    writer.load_state_dict(saved["writer"])
    prior.load_state_dict(saved["prior"])
    generator.eval(), writer.eval(), prior.eval()
    started = time.monotonic()
    # No D scoring head is constructed or used.
    metrics, paths = evaluate(generator, SimpleNamespace(writer=writer), prior, cfg, args.device)
    np.savez_compressed(args.out/"trajectories.npz", **paths)
    result = {"name": cfg.name, "steps": saved["step"], "config": asdict(cfg),
              "metrics": metrics, "seconds": time.monotonic()-started,
              "evaluation_only": True, "device": args.device}
    (args.out/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (args.out/"input.json").write_bytes(args.config.read_bytes())
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
