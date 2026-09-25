"""Matched rotated100 scratch probe: regularize the entire particle cloud.

The archived affine-identity/square-prior control uses the same recipe and
training runner. Its only training change here is replacing the large-prior
sampled-row regularizer input with all `prior.z` rows on every G update. This
is a proposed universal rule, not evidence for the existing common-22 gate.

Run with the exact archived resolved config and a fresh output directory::

    python -u reports/toy100/accuracy_full_prior_probe.py \
      --config reports/toy100/accuracy-failure/rotated100/config.json \
      --output artifacts/toy100-accuracy/affine-full-prior
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy100 import train as runner
from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_suite
from benchmarks.toy100.gate import evaluate_suite as coverage_suite
from benchmarks.toy100.models import InputNoise, OutputNoise
from lib.toy_models import SimpleMLPDiscriminator
from particlegan import GANTrainer


def install_full_cloud_prior_regularizer(trainer: GANTrainer) -> None:
    """Keep the same loss and coefficient while using every latent particle.

    GANTrainer supplies all rows for N<=1024, but only sampled unique rows for
    larger priors. The wrapper ignores that supplied view and evaluates the
    identical ParticleRegularizer on the complete trainable parameter.
    """
    original = trainer.prior_regularizer

    def full_cloud(_sampled_rows):
        return original(trainer.prior.z)

    trainer.prior_regularizer = full_cloud


def covariance(points: torch.Tensor) -> list[list[float]]:
    """Read-only sample covariance; never draws or modifies train tensors."""
    centered = points.detach() - points.detach().mean(0)
    value = centered.T @ centered / (len(points) - 1)
    return value.tolist()


def make_probe_trainer(resolved, recipe, *, output: Path, diagnostics_every=250):
    """Match the archived affine factory, then install one regularizer rule."""
    device = torch.device(resolved["device"])
    if device.type != "cpu":
        raise ValueError("this bounded scratch probe is CPU-only")
    if (resolved["problem"] != "rotated100" or resolved["z_dim"] != 2
            or resolved["batch_size"] != 2048 or resolved["num_particles"] != 20_000
            or resolved["fourier"] != 3):
        raise ValueError("probe requires the archived rotated100 affine resources")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(resolved["seed"])
        prior = recipe.make_prior(learnable=True).to(device)
        with torch.no_grad():
            prior.z.uniform_(-5.0, 5.0)
        generator = nn.Linear(2, 2)
        with torch.no_grad():
            generator.weight.copy_(torch.eye(2))
            generator.bias.zero_()
        discriminator = SimpleMLPDiscriminator(
            in_dim=2, hidden_dim=resolved["d_hidden"],
            n_hidden=resolved["n_hidden"], fourier=resolved["fourier"],
        )
        runner._init_linear(discriminator)
        if resolved["output_noise_std"]:
            generator = OutputNoise(generator, resolved["output_noise_std"])
        if resolved["input_noise_std"]:
            discriminator = InputNoise(
                discriminator, seed=resolved["seed"] + 901, device=device,
            )
        trainer = GANTrainer(
            recipe, generator, discriminator, prior=prior, seed=resolved["seed"],
            optimizer_options={"fused": False},
        )

    install_full_cloud_prior_regularizer(trainer)
    original_regularizer = trainer.prior_regularizer
    selected = torch.linspace(
        0, recipe.num_particles - 1, 1024, dtype=torch.long,
    ).round().unique()
    initial_fixed = prior.z.detach()[selected].clone()
    previous_fixed = initial_fixed.clone()
    diagnostics_file = output / "prior_diagnostics.jsonl"
    next_pre: dict = {}

    def record(step: int, pre: dict | None = None) -> None:
        nonlocal previous_fixed
        with torch.no_grad():
            fixed = prior.z.detach()[selected]
            displacement = (fixed - initial_fixed).norm(dim=1)
            since_last = (fixed - previous_fixed).norm(dim=1)
            linear = trainer.G.model if isinstance(trainer.G, OutputNoise) else trainer.G
            singular_values = torch.linalg.svdvals(linear.weight.detach()).tolist()
            row = dict(
                step=step, prior_full_covariance=covariance(prior.z),
                prior_full_mean=prior.z.detach().mean(0).tolist(),
                fixed_particle_count=len(selected),
                fixed_displacement_rms=float(displacement.square().mean().sqrt()),
                fixed_displacement_p95=float(torch.quantile(displacement, .95)),
                fixed_since_previous_rms=float(since_last.square().mean().sqrt()),
                affine_singular_values=singular_values,
                affine_bias=linear.bias.detach().tolist(),
            )
            if pre is not None:
                row["pre_update"] = pre
            with diagnostics_file.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            previous_fixed = fixed.clone()

    record(0)
    ordinary_step = trainer.step

    def measured_regularizer(sampled_rows):
        # Record the exact selected rows that the archived runner would have
        # regularized, then use the full cloud for the actual training loss.
        if (trainer.completed_steps + 1) % diagnostics_every == 0:
            with torch.no_grad():
                next_pre.update(
                    selected_count=len(sampled_rows),
                    selected_covariance=covariance(sampled_rows),
                    full_covariance=covariance(prior.z),
                    selected_regularizer=float(ordinary_regularizer(sampled_rows.detach())),
                )
        value = original_regularizer(sampled_rows)
        if (trainer.completed_steps + 1) % diagnostics_every == 0:
            next_pre["full_regularizer"] = float(value.detach())
        return value

    # Retain the same ParticleRegularizer for the counterfactual readout.
    ordinary_regularizer = recipe.make_prior_regularizer(weight=1.0)
    trainer.prior_regularizer = measured_regularizer

    def step(*args, **kwargs):
        result = ordinary_step(*args, **kwargs)
        if trainer.completed_steps % diagnostics_every == 0:
            record(trainer.completed_steps, next_pre.copy())
            print(json.dumps(dict(event="prior_diagnostic", step=trainer.completed_steps,
                                  **next_pre)), flush=True)
            next_pre.clear()
        return result

    trainer.step = step
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (output / "probe_source.py").write_bytes(source)
    config_bytes = args.config.read_bytes()
    shutil.copyfile(args.config, output / "archived_control_config.json")
    config = json.loads(config_bytes)
    if (config.get("problem") != "rotated100" or config.get("steps") != 7000
            or config.get("z_dim") != 2 or config.get("batch_size") != 2048
            or config.get("num_particles") != 20_000 or config.get("fourier") != 3):
        raise ValueError("config differs from the archived rotated100 control")
    options = dict(
        generator="trainable_affine", prior="square", scale=5.0,
        initialization="identity", prior_regularizer_scope="full_prior_z_every_update",
        prior_regularizer_implementation="same ParticleRegularizer and recipe.prior_reg",
        changed_relative_to_control="regularizer input only",
        shared_gate_eligible=False,
        archived_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )
    (output / "model_options.json").write_text(json.dumps(options, indent=2) + "\n")
    (output / "declared_config.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance_before = runner._source_provenance

    def provenance():
        record = provenance_before()
        record["source_sha256"]["reports/toy100/accuracy_full_prior_probe.py"] = (
            hashlib.sha256(source).hexdigest()
        )
        record["trainer_factory"] = "affine full-cloud prior diagnostic; not common-gate evidence"
        record["model_options"] = options
        return record

    runner.make_trainer = lambda resolved, recipe: make_probe_trainer(
        resolved, recipe, output=output,
    )
    runner._source_provenance = provenance
    summary = runner.train(config, output / "rotated100")
    coverage = coverage_suite(output, problem="rotated100")
    accuracy = accuracy_suite(output, problem="rotated100")
    print(json.dumps(dict(
        event="probe_complete", coverage=coverage["status"], accuracy=accuracy["status"],
        final=summary["final"], diagnostics=str(output / "prior_diagnostics.jsonl"),
    )), flush=True)


if __name__ == "__main__":
    main()
