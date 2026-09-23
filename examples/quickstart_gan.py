#!/usr/bin/env python
"""A small runnable GANTrainer example; replace real_batch with your data."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlegan import BatchDistanceDiscriminator, LinearSkipDiscriminator, get_recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("gan", "gan_v3", "gan_v2", "gan_behavioral",
                                             "gan_v1", "gan_legacy"), default="gan")
    parser.add_argument("--steps", type=int, default=1000, help="Total budget, including updates before resume.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("quickstart.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int, help="Save early without changing the full-run LR schedule.")
    args = parser.parse_args()
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("--stop-after must be between 1 and --steps")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device(args.device)
    recipe = get_recipe(args.recipe, total_steps=args.steps)
    generator = nn.Sequential(nn.Linear(recipe.z_dim, 64), nn.LeakyReLU(.2),
                              nn.Linear(64, 64), nn.LeakyReLU(.2), nn.Linear(64, 2)).to(device)
    discriminator = (BatchDistanceDiscriminator() if args.recipe in ("gan", "gan_v3")
                     else LinearSkipDiscriminator()).to(device)
    trainer = recipe.make_trainer(generator, discriminator, seed=0)
    data_rng = torch.Generator(device=device).manual_seed(0)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        trainer.load_state_dict(checkpoint["trainer"])
        data_rng.set_state(checkpoint["data_rng"])

    def real_batch():
        # Data units matter: this example has unit-scale coordinates.
        return .2 * torch.randn(recipe.batch_size, 2, device=device, generator=data_rng) + 1

    print(json.dumps({"recipe": recipe.to_dict(), "completed_steps": trainer.completed_steps}), flush=True)
    for _ in range(trainer.completed_steps, args.stop_after or recipe.total_steps):
        stats = trainer.step(real_batch(), generator_real=real_batch)
        if stats["step"] == 1 or stats["step"] % 100 == 0 or stats["step"] == (args.stop_after or recipe.total_steps):
            print(json.dumps({key: float(value) if isinstance(value, torch.Tensor) else value
                              for key, value in stats.items()}), flush=True)
    # Live is the default. Sampling uses a separate RNG, so it cannot change training.
    samples = trainer.sample(1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trainer": trainer.state_dict(), "data_rng": data_rng.get_state(),
                "live_samples": samples.cpu()}, args.output)
    print(f"Saved step {trainer.completed_steps} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
