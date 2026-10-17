"""Bounded public-API role/continuation proof; no distribution qualification."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from experiments.forge.api import FormulationContext
from experiments.forge.contracts import atomic_json
from experiments.forge.state import require_consistent_rng, require_optimizer_steps, state_digest

UPDATES = 8
CHECKS = (0, 4, 8)
WALL_SECONDS = 60


def target(n, stream):
    """Equal two-Gaussian mixture, centers (+/-1,0), isotropic std .15."""
    modes = torch.randint(2, (n,), generator=stream)
    centers = torch.stack((2. * modes - 1., torch.zeros(n)), dim=1).double()
    return centers + .15 * torch.randn(n, 2, generator=stream, dtype=torch.float64)


def construct(variant):
    context = FormulationContext(recipe_preset="halloween", seed=0,
        recipe_overrides={"adam_variant": variant, "num_particles": 16,
                          "batch_size": 16, "total_steps": UPDATES},
        prior={"kind": "mog", "sigma": .025, "standardize": False, "learnable": True})
    generator = context.construct(lambda: nn.Sequential(nn.Linear(2, 16), nn.Tanh(),
        nn.Linear(16, 2)).double(), component="generator")
    discriminator = context.construct(lambda: nn.Sequential(nn.Linear(2, 16), nn.Tanh(),
        nn.Linear(16, 1)).double(), component="discriminator")
    trainer = context.build_trainer(generator, discriminator, max_steps=UPDATES)
    data = [context.streams.generator("data", component=role, purpose="training")
            for role in ("critic", "generator")]
    reference = target(256, context.streams.generator("eval", component="target", purpose="reference"))
    return context, trainer, data, reference


def observe(trainer, update):
    values = trainer.sample(256, ema=False, output_noise=False).detach().cpu()
    if not torch.isfinite(values).all():
        raise AssertionError("nonfinite API samples")
    return {"updates": update, "samples": values}


def train(context, trainer, streams, start, deadline, output, *, frames):
    batches, saved = [], None
    for update in range(start + 1, UPDATES + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("declared software-proof wall budget exhausted")
        critic, generator = [target(16, stream) for stream in streams]
        batches.append(state_digest((critic, generator)))
        trainer.step(critic, generator_real=generator)
        if update in CHECKS:
            frames.append(observe(trainer, update))
        if update == 4:
            saved = deepcopy(context.state_dict())
            torch.save(saved, output / f"{context.recipe.adam_variant}-checkpoint.pt")
        print(f"variant={context.recipe.adam_variant} update={update}/{UPDATES}", flush=True)
    packet = context.state_dict()
    require_consistent_rng(packet)
    require_optimizer_steps(packet, UPDATES)
    return packet, saved, batches


def render(reference, frames, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    images = []
    for left, right in zip(frames["pytorch"], frames["tensorflow_v1"]):
        figure, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for axis, label, samples in zip(axes, ("Target law", "PyTorch Adam", "Dense legacy Adam"),
                                       (reference, left["samples"], right["samples"])):
            axis.scatter(samples[:, 0], samples[:, 1], s=5, alpha=.5)
            axis.set(xlim=(-2, 2), ylim=(-2, 2), title=label, aspect="equal")
        figure.suptitle(f"Actual public-API training: {left['updates']} updates; software proof, quality unqualified")
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=100)
        plt.close(figure)
        images.append(Image.open(buffer).convert("RGB"))
    images[0].save(destination, save_all=True, append_images=images[1:], duration=1200, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    deadline = started + WALL_SECONDS
    contract = {"seed": 0, "updates_per_arm": UPDATES, "evaluation_updates": CHECKS,
        "wall_seconds_ceiling": WALL_SECONDS, "execution": "public GANTrainer via FormulationContext",
        "target": "equal Gaussian mixture centers (-1,0),(1,0), std .15",
        "architecture": "G 2-16-tanh-2; D 2-16-tanh-1; float64 CPU",
        "prior": "16 learned equal-mass MoG components, sigma .025, unstandardized",
        "sampling": "256 clean live samples per observation",
        "trainer_delta": "adam_variant pytorch versus tensorflow_v1; all other settings identical",
        "gates": {"initial_models_equal": True, "seen_batches_equal": True,
                  "checkpoint_state_equal": True, "resume_samples_equal": True,
                  "finite_changed_models": True, "variant_displacement_positive": True},
        "qualification": "software consumption and continuation only; no quality claim or default adoption"}
    atomic_json(args.output / "contract.json", contract)
    finals, initials, batches, observations, receipts = {}, {}, {}, {}, {}
    for variant in ("pytorch", "tensorflow_v1"):
        context, trainer, data, reference = construct(variant)
        initials[variant] = state_digest(context.state_dict()["trainer"]["models"])
        frames = [observe(trainer, 0)]
        final, checkpoint, seen = train(context, trainer, data, 0, deadline, args.output, frames=frames)
        finals[variant], batches[variant], observations[variant] = final, seen, frames
        receipts[variant] = context.receipt()
        if variant == "tensorflow_v1":
            restored, resumed, streams, _ = construct(variant)
            restored.load_state_dict(checkpoint)
            replay_frames = []
            replay, _, suffix = train(restored, resumed, streams, 4, deadline, args.output, frames=replay_frames)
            replay_equal = state_digest(replay) == state_digest(final)
            samples_equal = torch.equal(replay_frames[-1]["samples"], frames[-1]["samples"])
            assert suffix == seen[4:]
    displacement = max((a - b).abs().max().item()
        for a, b in zip(finals["pytorch"]["trainer"]["models"]["G"].values(),
                        finals["tensorflow_v1"]["trainer"]["models"]["G"].values()))
    gates = {"initial_models_equal": len(set(initials.values())) == 1,
             "seen_batches_equal": batches["pytorch"] == batches["tensorflow_v1"],
             "checkpoint_state_equal": replay_equal, "resume_samples_equal": samples_equal,
             "finite_changed_models": all(torch.isfinite(value).all().item()
                for packet in finals.values() for model in packet["trainer"]["models"].values()
                if isinstance(model, dict) for value in model.values() if isinstance(value, torch.Tensor))
                and all(state_digest(packet["trainer"]["models"]) != initials[variant]
                        for variant, packet in finals.items()),
             "variant_displacement_positive": displacement > 0}
    source_paths = ["reports/forge/hypergan-search-audit/implementation_probe.py",
                    "experiments/forge/api.py", "experiments/forge/rng.py"]
    source_paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "particlegan").rglob("*.py"))]
    result = {"schema": "halloween_api_software_proof_v1", "contract": contract,
        "gates": gates, "pass": all(gates.values()), "elapsed_seconds": time.monotonic() - started,
        "initial_model_hashes": initials, "batch_sequence_hash": state_digest(batches["pytorch"]),
        "maximum_generator_variant_difference": displacement,
        "final_state_hashes": {name: state_digest(packet) for name, packet in finals.items()},
        "resolved_contexts": receipts, "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths},
        "artifact_directory": str(args.output.resolve())}
    atomic_json(args.output / "receipt.json", result)
    torch.save({"finals": finals, "observations": observations}, args.output / "observations.pt")
    render(reference, observations, args.output / "actual-training.gif")
    print(f"software_proof_pass={result['pass']}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
