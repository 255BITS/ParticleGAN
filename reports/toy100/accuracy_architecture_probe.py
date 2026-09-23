"""Scratch residual generator probe under the declared shared-v3 optimizer core.

The trainer sees no mixture centers. The initial latent prior is uniform in a
generic disk; target geometry is read only by the frozen postrun evaluators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy100 import train as runner  # noqa: E402
from benchmarks.toy100.accuracy import audit_npz  # noqa: E402
from benchmarks.toy100.gate import evaluate_suite  # noqa: E402
from benchmarks.toy100.models import InputNoise, OutputNoise  # noqa: E402
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator  # noqa: E402
from particlegan import GANTrainer  # noqa: E402


class ResidualGenerator(nn.Module):
    def __init__(self, hidden_dim: int, n_hidden: int, alpha: float):
        super().__init__()
        self.residual = SimpleMLPGenerator(
            z_dim=2, hidden_dim=hidden_dim, n_hidden=n_hidden
        )
        self.alpha = alpha

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.alpha * self.residual(z)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, choices=(0.01, 0.1, 0.3, 0.5), required=True)
    parser.add_argument("--input-noise-anneal-end", type=float, choices=(0.1, 0.5),
                        default=0.5)
    parser.add_argument("--reg-coeff", type=float, choices=(2.0, 6.0), default=6.0)
    parser.add_argument("--lr-anneal-start", type=float, choices=(0.4, 0.6),
                        default=0.6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (out / "probe_source.py").write_bytes(source)
    options = {
        "generator": "z + alpha * MLP(z)",
        "alpha": args.alpha,
        "prior_init": "uniform_disk",
        "prior_radius": 6.5,
        "z_dim": 2,
        "fourier": 3,
    }
    (out / "model_options.json").write_text(json.dumps(options, indent=2) + "\n")
    config = json.loads((ROOT / "configs/toy100/accuracy_v3_base.json").read_text())
    config.update(name=f"v3_residual_a{args.alpha:g}_cap{args.reg_coeff:g}",
                  z_dim=2, fourier=3, reg_coeff=args.reg_coeff,
                  input_noise_anneal_end=args.input_noise_anneal_end,
                  lr_anneal_start=args.lr_anneal_start)
    (out / "declared_config.json").write_text(json.dumps(config, indent=2) + "\n")

    original_provenance = runner._source_provenance

    def provenance() -> dict:
        value = original_provenance()
        value["source_sha256"]["reports/toy100/accuracy_architecture_probe.py"] = (
            hashlib.sha256(source).hexdigest()
        )
        value["trainer_factory"] = "ResidualGenerator + uniform-disk learnable prior"
        value["model_options"] = options
        return value

    def make_trainer(resolved: dict, recipe) -> GANTrainer:
        device = torch.device(resolved["device"])
        devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                   if device.type == "cuda" else [])
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(resolved["seed"])
            if device.type == "cuda":
                torch.cuda.manual_seed_all(resolved["seed"])
            prior = recipe.make_prior(learnable=True).to(device)
            with torch.no_grad():
                theta = torch.rand(recipe.num_particles, device=device) * (2.0 * torch.pi)
                radius = torch.rand(recipe.num_particles, device=device).sqrt() * 6.5
                prior.z.copy_(torch.stack((theta.cos(), theta.sin()), dim=1) * radius[:, None])
            generator = ResidualGenerator(
                resolved["g_hidden"], resolved["n_hidden"], args.alpha
            ).to(device)
            discriminator = SimpleMLPDiscriminator(
                in_dim=2, hidden_dim=resolved["d_hidden"],
                n_hidden=resolved["n_hidden"], fourier=resolved["fourier"],
            ).to(device)
            runner._init_linear(generator)
            runner._init_linear(discriminator)
            if resolved["output_noise_std"]:
                generator = OutputNoise(generator, resolved["output_noise_std"])
            if resolved["input_noise_std"]:
                discriminator = InputNoise(
                    discriminator, seed=resolved["seed"] + 901, device=device
                )
            return GANTrainer(
                recipe, generator, discriminator, prior=prior, seed=resolved["seed"],
                optimizer_options={"fused": resolved["fused_adam"]},
            )

    runner.make_trainer = make_trainer
    runner._source_provenance = provenance
    runner.train({**config, "problem": "grid100"}, out / "grid100")
    gate = evaluate_suite(out, problem="grid100")
    accuracy = audit_npz(out / "grid100/final_samples.npz", "grid100")
    (out / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    (out / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    print(json.dumps({
        "gate": gate["problems"]["grid100"]["status"],
        "accuracy_pass": accuracy["accuracy_pass"],
        "accuracy_score": accuracy["accuracy_score"],
    }), flush=True)


if __name__ == "__main__":
    main()
