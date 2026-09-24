"""Exploratory paired input-noise probe on the saved output-noise MLP recipe.

CUDA_VISIBLE_DEVICES=1 python -u artifacts/toy100/instance_noise_probe.py \
    --variant control --device cuda:0
CUDA_VISIBLE_DEVICES=1 python -u artifacts/toy100/instance_noise_probe.py \
    --variant sigma05 --device cuda:0

The discriminator wrapper adds fresh zero-mean isotropic Gaussian noise to
every real or generated D input during both D and G updates. The peak 0.5
linearly reaches zero halfway through the 7,000-update budget. The random
stream is separate from generator, prior, and data streams. Target centers and
mode assignments are used only by the runner's post-update evaluator.
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

from benchmarks.toy100 import train as runner
from benchmarks.toy100.gate import evaluate_suite
from benchmarks.toy100.problems import PROBLEM_NAMES
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import GANTrainer


OUTPUT_STD = 0.026
PEAK_SIGMA = {"control": 0.0, "sigma05": 0.5}
DECAY_END_FRACTION = 0.5


class OutputNoise(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, latent):
        prediction = self.model(latent)
        return prediction + OUTPUT_STD * torch.randn_like(prediction)


class InputNoise(nn.Module):
    def __init__(self, model: nn.Module, *, seed: int, device: torch.device):
        super().__init__()
        self.model = model
        self.sigma = 0.0
        self.noise_stream = torch.Generator(device=device).manual_seed(seed)

    def forward(self, points):
        if self.sigma:
            noise = torch.randn(points.shape, generator=self.noise_stream,
                                device=points.device, dtype=points.dtype)
            points = points + self.sigma * noise
        return self.model(points)


class ProbeTrainer(GANTrainer):
    def __init__(self, *args, peak_sigma: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_sigma = peak_sigma

    def step(self, real, *, generator_real=None, collect_stats=False):
        fraction = self.completed_steps / (self.recipe.total_steps * DECAY_END_FRACTION)
        self.D.sigma = self.peak_sigma * max(0.0, 1.0 - fraction)
        return super().step(real, generator_real=generator_real, collect_stats=collect_stats)


def make_factory(peak_sigma: float):
    def factory(config, recipe):
        device = torch.device(config["device"])
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config["seed"])
            prior = recipe.make_prior().to(device)
            generator = SimpleMLPGenerator(
                z_dim=recipe.z_dim, hidden_dim=config["g_hidden"],
                n_hidden=config["n_hidden"],
            ).to(device)
            discriminator = SimpleMLPDiscriminator(
                in_dim=2, hidden_dim=config["d_hidden"],
                n_hidden=config["n_hidden"], fourier=config["fourier"],
            ).to(device)
            runner._init_linear(generator)
            runner._init_linear(discriminator)
            return ProbeTrainer(
                recipe, OutputNoise(generator),
                InputNoise(discriminator, seed=config["seed"] + 901, device=device),
                prior=prior, seed=config["seed"], peak_sigma=peak_sigma,
                optimizer_options={"fused": config["fused_adam"]},
            )
    return factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(PEAK_SIGMA), required=True)
    parser.add_argument("--problem", choices=PROBLEM_NAMES, default="grid100")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/toy100/instance-noise")
    args = parser.parse_args(argv)
    peak = PEAK_SIGMA[args.variant]
    root = args.output / args.variant
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / args.problem
    source = Path(__file__).read_bytes()
    (root / f"probe_source_{args.problem}.py").write_bytes(source)
    options = {"variant": args.variant, "input_noise_peak": peak,
               "input_noise_decay_end_fraction": DECAY_END_FRACTION,
               "output_noise_std": OUTPUT_STD,
               "source_sha256": hashlib.sha256(source).hexdigest()}
    (root / f"model_options_{args.problem}.json").write_text(json.dumps(options, indent=2) + "\n")
    config = json.loads((ROOT / "artifacts/toy100/noisy-mlp/grid100/config.json").read_text())
    config.update(name=f"output026_input_{args.variant}", device=args.device,
                  problem=args.problem)
    original_provenance = runner._source_provenance

    def provenance():
        value = original_provenance()
        value["source_sha256"]["artifacts/toy100/instance_noise_probe.py"] = options["source_sha256"]
        value["trainer_factory"] = "instance_noise_probe.make_factory"
        value["model_options"] = options
        return value

    runner.make_trainer = make_factory(peak)
    runner._source_provenance = provenance
    torch.manual_seed(config["seed"])
    try:
        summary = runner.train(config, run_dir)
    finally:
        # Keep exact source beside each attempt after the runner has created
        # its initially empty output directory.
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "probe_source.py").write_bytes(source)
        (run_dir / "model_options.json").write_text(json.dumps(options, indent=2) + "\n")
    gate = evaluate_suite(root, problem=args.problem)
    print(json.dumps({"variant": args.variant, "summary_status": summary["status"],
                      "gate_status": gate["status"],
                      "problem": args.problem,
                      "final_live": gate["problems"][args.problem].get("final_metrics")}), flush=True)
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
