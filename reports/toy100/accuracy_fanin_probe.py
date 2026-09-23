"""Scratch fan-in parameterization probe under one unchanged shared-v3 recipe.

An ordinary Xavier-initialized MLP is constructed first, preserving its RNG
order. Selected Linear weights are then represented as raw / sqrt(fan_in).
This changes the optimizer's effective function-space step while leaving its
declared Adam learning rates, betas, loss, penalty, and target sampler intact.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy100 import train as runner  # noqa: E402
from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_suite  # noqa: E402
from benchmarks.toy100.gate import evaluate_suite as coverage_suite  # noqa: E402
from benchmarks.toy100.models import InputNoise, OutputNoise  # noqa: E402
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator  # noqa: E402
from particlegan import GANTrainer  # noqa: E402


class FanInLinear(nn.Module):
    """The same linear map with its trainable weight stored at fan-in scale."""

    def __init__(self, original: nn.Linear):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.scale = math.sqrt(self.in_features)
        self.raw = nn.Parameter(original.weight.detach().clone() * self.scale)
        self.bias = (nn.Parameter(original.bias.detach().clone())
                     if original.bias is not None else None)

    @property
    def weight(self) -> torch.Tensor:
        return self.raw / self.scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)


def parameterize_linears(model: nn.Module) -> nn.Module:
    """Replace only Linear modules, preserving other layers and their order."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, FanInLinear(child))
        else:
            parameterize_linears(child)
    return model


def parity_receipt(reference: nn.Module, parameterized: nn.Module,
                   inputs: torch.Tensor) -> dict[str, float | int]:
    """Measure numerical parity of function and input Jacobian at step zero."""
    x_ref = inputs.detach().clone().requires_grad_(True)
    x_new = inputs.detach().clone().requires_grad_(True)
    y_ref, y_new = reference(x_ref), parameterized(x_new)
    g_ref = torch.autograd.grad(y_ref.square().sum(), x_ref)[0]
    g_new = torch.autograd.grad(y_new.square().sum(), x_new)[0]
    old_layers = [layer for layer in reference.modules() if isinstance(layer, nn.Linear)]
    new_layers = [layer for layer in parameterized.modules() if isinstance(layer, FanInLinear)]
    if len(old_layers) != len(new_layers):
        raise RuntimeError("linear layer count changed")
    weight_error = max(float((old.weight - new.weight).detach().abs().max())
                       for old, new in zip(old_layers, new_layers))
    bias_error = max(float((old.bias - new.bias).detach().abs().max())
                     for old, new in zip(old_layers, new_layers))
    result = {
        "layers": len(old_layers),
        "max_initial_weight_error": weight_error,
        "max_initial_bias_error": bias_error,
        "max_initial_output_error": float((y_ref - y_new).detach().abs().max()),
        "max_initial_input_gradient_error": float((g_ref - g_new).detach().abs().max()),
    }
    # The real-valued function is identical. Float32 raw-weight roundtrip can
    # differ by one ULP, so demand parity at float32 arithmetic precision.
    if (result["max_initial_weight_error"] > 3e-8
            or result["max_initial_bias_error"] != 0
            or result["max_initial_output_error"] > 2e-6
            or result["max_initial_input_gradient_error"] > 2e-6):
        raise RuntimeError(f"initial function or Jacobian changed: {result}")
    return result


def make_equalized_trainer(resolved: dict, recipe, *, parameterize_d: bool,
                           parity_output: Path | None = None) -> GANTrainer:
    """Mirror the production constructor and change only selected weight maps."""
    device = torch.device(resolved["device"])
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(resolved["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(resolved["seed"])
        prior = recipe.make_prior(learnable=True).to(device)
        generator = SimpleMLPGenerator(
            z_dim=recipe.z_dim, hidden_dim=resolved["g_hidden"],
            n_hidden=resolved["n_hidden"],
        ).to(device)
        discriminator = SimpleMLPDiscriminator(
            in_dim=2, hidden_dim=resolved["d_hidden"],
            n_hidden=resolved["n_hidden"], fourier=resolved["fourier"],
        ).to(device)
        runner._init_linear(generator)
        runner._init_linear(discriminator)
        g_original, d_original = deepcopy(generator), deepcopy(discriminator)
        rng_before = torch.random.get_rng_state().clone()
        parameterize_linears(generator)
        if parameterize_d:
            parameterize_linears(discriminator)
        if not torch.equal(rng_before, torch.random.get_rng_state()):
            raise RuntimeError("parameterization consumed the training RNG")
        g_inputs = torch.linspace(-1.0, 1.0, 31 * recipe.z_dim,
                                  device=device).reshape(31, recipe.z_dim)
        d_inputs = torch.linspace(-1.0, 1.0, 31 * 2, device=device).reshape(31, 2)
        receipt = {"generator": parity_receipt(g_original, generator, g_inputs)}
        if parameterize_d:
            receipt["discriminator"] = parity_receipt(d_original, discriminator, d_inputs)
        else:
            receipt["discriminator"] = {"parameterized": False,
                                        "initial_function_identical": True}
        if parity_output is not None:
            parity_output.write_text(json.dumps(receipt, indent=2) + "\n")
        if resolved["output_noise_std"]:
            generator = OutputNoise(generator, resolved["output_noise_std"])
        if resolved["input_noise_std"]:
            discriminator = InputNoise(
                discriminator, seed=resolved["seed"] + 901, device=device,
            )
        return GANTrainer(
            recipe, generator, discriminator, prior=prior, seed=resolved["seed"],
            optimizer_options={"fused": resolved["fused_adam"]},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameterize", choices=("g", "g_d"), required=True)
    parser.add_argument("--config", type=Path,
                        help="shared recipe manifest; defaults to the original shared-v3 probe")
    parser.add_argument("--batch-size", type=int,
                        help="toy100 resource batch size when the transfer manifest uses another size")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (output / "probe_source.py").write_bytes(source)
    source_config = (args.config if args.config is not None else
                     ROOT / "configs/toy100/accuracy_warmup_v3_base.json")
    config_bytes = source_config.read_bytes()
    config = json.loads(config_bytes)
    # The transfer manifest's per-problem resource exception is irrelevant to
    # this single-grid scratch run. All common optimizer/noise fields remain.
    config.pop("problem_overrides", None)
    config["name"] = f"shared_v3_fanin_{args.parameterize}"
    if args.config is None:
        config["lr_anneal_start"] = 0.6
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("batch size must be positive")
        config["batch_size"] = args.batch_size
    (output / "declared_config.json").write_text(json.dumps(config, indent=2) + "\n")
    options = {
        "parameterized": args.parameterize,
        "effective_weight": "raw / sqrt(fan_in)",
        "initialization": "original Xavier weight times sqrt(fan_in), original bias",
        "optimizer": "unchanged public Recipe Adam groups and learning rates",
        "effective_adam_weight_step": "approximately LR / sqrt(fan_in)",
        "source_config": str(source_config),
        "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "resource_batch_override": args.batch_size,
    }
    (output / "model_options.json").write_text(json.dumps(options, indent=2) + "\n")
    original_provenance = runner._source_provenance

    def provenance() -> dict:
        row = original_provenance()
        row["source_sha256"]["reports/toy100/accuracy_fanin_probe.py"] = (
            hashlib.sha256(source).hexdigest()
        )
        row["trainer_factory"] = "fan-in parameterized SimpleMLP G and optional D"
        row["model_options"] = options
        return row

    def factory(resolved: dict, recipe) -> GANTrainer:
        return make_equalized_trainer(
            resolved, recipe, parameterize_d=args.parameterize == "g_d",
            parity_output=output / "initial_parity.json",
        )

    runner.make_trainer = factory
    runner._source_provenance = provenance
    runner.train({**config, "problem": "grid100"}, output / "grid100")
    coverage = coverage_suite(output, problem="grid100")
    accuracy = accuracy_suite(output, problem="grid100")
    print(json.dumps({
        "coverage": coverage["status"],
        "accuracy": accuracy["status"],
        "modes": coverage["problems"]["grid100"].get("final_live", {}).get("modes"),
    }), flush=True)


if __name__ == "__main__":
    main()
