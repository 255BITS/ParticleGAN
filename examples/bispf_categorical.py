#!/usr/bin/env python
"""Train four categorical outputs with an explicitly declared BISPF domain.

python examples/bispf_categorical.py --output runs/software/bispf-categorical
Outputs: metrics.json, checkpoint.pt and an animation of actual training samples.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlegan import GANTrainer, Recipe, get_recipe, init


TARGET = torch.tensor([0.1, 0.2, 0.3, 0.4])
TV_LIMIT = 0.15
CONFIDENCE_MIN = 0.8
DEFAULT_STEPS = 2000
EVAL_SAMPLES = 4096


def make_trainer(recipe):
    generator = nn.Sequential(nn.Linear(recipe.z_dim, 64), nn.SiLU(),
                              nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 4),
                              nn.Softmax(dim=-1))
    critic = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 1))
    init.deterministic_orthogonal_(generator, seed=0)
    init.deterministic_orthogonal_(critic, seed=1)
    prior = init.deterministic_orthogonal_(recipe.make_prior())
    return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                      penalty_options={"input_domain": "categorical"})


def score(probabilities):
    """Score category mass and concentration separately; neither can hide the other.

    TV is half the sum of absolute frequency errors. Confidence is the mean
    largest entry in each soft output. A uniform soft vector can have perfect
    average probabilities without looking like a real one-hot observation.
    """
    if (probabilities.ndim != 2 or probabilities.shape[1] != 4
            or not torch.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not torch.allclose(probabilities.sum(1), torch.ones(len(probabilities)), atol=1e-6)):
        raise ValueError("Expected finite [batch, 4] probability rows")
    frequencies = F.one_hot(probabilities.argmax(1), 4).float().mean(0)
    tv = float((frequencies - TARGET).abs().sum() / 2)
    confidence = float(probabilities.max(1).values.mean())
    return {"frequencies": frequencies.tolist(), "mass_tv": tv,
            "mean_confidence": confidence,
            "quality_pass": tv <= TV_LIMIT and confidence >= CONFIDENCE_MIN}


def observe(trainer):
    # Reusing the same evaluation draws makes the animation comparable and
    # does not consume a training/data stream. Outputs are clean live weights.
    samples = trainer.sample(EVAL_SAMPLES, generator=torch.Generator().manual_seed(0))
    return {"step": trainer.completed_steps, **score(samples)}


def render(history, destination):
    # Pillow is an examples/experiments dependency; training itself uses torch.
    from PIL import Image, ImageDraw
    frames = []
    for observation in history:
        frame = Image.new("RGB", (640, 380), "white")
        draw = ImageDraw.Draw(frame)
        draw.text((20, 15), f"Categorical BISPF: actual live outputs, update {observation['step']}", fill="black")
        draw.text((20, 40), "Blue: target frequency. Orange: generated argmax frequency.", fill="black")
        for value in (0.0, 0.5, 1.0):
            y = 300 - value * 210
            draw.line((45, y, 620, y), fill="#dddddd")
            draw.text((10, y - 5), f"{value:.1f}", fill="black")
        for category, (target, actual) in enumerate(zip(TARGET.tolist(), observation["frequencies"])):
            x = 55 + 145 * category
            draw.rectangle((x, 300 - target * 210, x + 35, 300), fill="#3569b0")
            draw.rectangle((x + 40, 300 - actual * 210, x + 75, 300), fill="#e58a2c")
            draw.text((x, 305), f"Category {category}", fill="black")
        draw.text((20, 335), f"TV {observation['mass_tv']:.3f} <= {TV_LIMIT}; "
                  f"confidence {observation['mean_confidence']:.3f} >= {CONFIDENCE_MIN}", fill="black")
        draw.text((20, 355), f"Observed quality gate: {'PASS' if observation['quality_pass'] else 'FAIL'}", fill="black")
        frames.append(frame)
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=250, loop=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Full update budget; a shorter run is only a software smoke check.")
    parser.add_argument("--output", type=Path, default=Path("runs/software/bispf-categorical"))
    parser.add_argument("--resume", type=Path, help="Restore checkpoint.pt, including data RNG and observations.")
    parser.add_argument("--stop-after", type=int, help="Checkpoint early, retaining the full --steps budget.")
    parser.add_argument("--check-quality", action="store_true", help="Exit nonzero unless the full quality gate passes.")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("--steps must be positive; --stop-after must be in [1, steps]")
    torch.set_num_threads(1)
    torch.manual_seed(0)  # Fix constructor/global state; data and training use separate streams.
    data_rng = torch.Generator().manual_seed(0)
    checkpoint = torch.load(args.resume, weights_only=True, map_location="cpu") if args.resume else None
    recipe = (Recipe.from_dict(checkpoint["trainer"]["recipe"]) if checkpoint else
              get_recipe("bispf_optimized", z_dim=4, num_particles=256,
                         batch_size=64, total_steps=args.steps))
    if recipe.total_steps != args.steps:
        parser.error("--steps must match the saved full budget")
    trainer = make_trainer(recipe)
    if checkpoint:
        trainer.load_state_dict(checkpoint["trainer"])
        data_rng.set_state(checkpoint["data_rng"])
    history = checkpoint["history"] if checkpoint else [observe(trainer)]
    args.output.mkdir(parents=True, exist_ok=True)
    # Cadence depends on the full budget, so early stopping/resume cannot alter it.
    cadence = max(1, args.steps // 20)
    print(json.dumps({"recipe": recipe.to_dict(), "input_domain": "categorical",
                      "target": TARGET.tolist(), "budget": args.steps,
                      "gates": {"mass_tv_max": TV_LIMIT, "confidence_min": CONFIDENCE_MIN}}), flush=True)

    def real_batch():
        categories = torch.multinomial(TARGET, recipe.batch_size, replacement=True, generator=data_rng)
        return F.one_hot(categories, 4).float()

    stop = args.stop_after or args.steps
    if stop < trainer.completed_steps:
        parser.error("Cannot stop before the saved step")
    last_stats = None
    for _ in range(trainer.completed_steps, stop):
        # Real rows are one-hot; generated rows are differentiable softmax
        # probabilities. argmax is used only for measurement after training.
        stats = trainer.step(real_batch(), generator_real=real_batch, collect_stats=True)
        if any(not torch.isfinite(value).all() for value in stats.values() if isinstance(value, torch.Tensor)):
            raise RuntimeError(f"Nonfinite training loss at update {trainer.completed_steps}")
        last_stats = stats["penalty_stats"]
        if last_stats["floor_pen"] != 0 or last_stats["gauge_loss"] != 0:
            raise RuntimeError("Training did not use the categorical penalty")
        if trainer.completed_steps % cadence == 0 or trainer.completed_steps == args.steps:
            history.append(observe(trainer))
            print(json.dumps(history[-1]), flush=True)
    torch.save({"trainer": trainer.state_dict(), "data_rng": data_rng.get_state(),
                "history": history}, args.output / "checkpoint.pt")
    render(history, args.output / "training.gif")
    terminal = [entry for entry in history if entry["step"] > 0][-5:]
    quality_pass = (args.steps >= DEFAULT_STEPS and trainer.completed_steps == args.steps
                    and len(terminal) == 5 and all(entry["quality_pass"] for entry in terminal))
    report = {"completed_steps": trainer.completed_steps, "budget": args.steps,
              "sampling": "clean live softmax outputs; fixed independent evaluation RNG seed 0",
              "target": TARGET.tolist(), "recipe": recipe.to_dict(),
              "penalty_options": {"input_domain": "categorical"},
              "mass_tv_max": TV_LIMIT, "confidence_min": CONFIDENCE_MIN,
              "history": history, "last_penalty_stats": last_stats,
              "quality_pass": quality_pass,
              "scope": "Four-category teaching example; no Forge qualification or native-model quality claim."}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"software_check": "PASS", "quality_pass": quality_pass,
                      "output": str(args.output)}), flush=True)
    if args.check_quality and not quality_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
