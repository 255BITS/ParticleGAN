"""A fixed 64-update CPU software illustration, not a Forge quality run.

The goal is functioning, resumable restoration during real public GAN training.
The declared software gate is 64 spring applications, the exact captured count
of unique update-owned rows, finite outputs and an exact resumed next update. Scalar fidelity is measured
against the unchanged ordinary thresholds and reported separately.
"""
from argparse import ArgumentParser
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import ndtr, ndtri
import torch
from torch import nn

from particlegan import get_recipe
from experiments.forge.api import FormulationContext
from experiments.forge.state import state_digest
from benchmarks.toy_audit.gaussian1d_quality import score_samples
from benchmarks.toy_audit.reproducibility import reproducible_execution


TARGET = dict(kind="gaussian_mixture", means=[[2.]], covariances=[[[.25]]], masses=[1.])
THRESHOLDS = dict(cdf_ks=["<=", .05], mean_error_sigma=["<=", .2], std_ratio=["range", [.8, 1.2]])
STEPS, SAMPLES, BATCH, PARTICLES = 64, 512, 32, 32


def build(strength):
    # All BCAP-leader trainer fields remain fixed; this explicit software
    # cohort changes only task resources and uses width8/depth1 networks.
    settings = asdict(get_recipe("bcap"))
    settings.pop("prior_proximal_strength", None)  # Runs on the pinned zero baseline too.
    settings.update(num_particles=PARTICLES, z_dim=2, batch_size=BATCH, total_steps=STEPS,
                    prior_kind="mog", standardize=False)
    if strength:
        settings["prior_proximal_strength"] = strength
    context = FormulationContext(recipe_overrides=settings, prior_contract="recipe_owned_v1",
        prior=dict(kind="mog", sigma=.1, standardize=False), seed=0)
    generator = context.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)), component="generator")
    critic = context.construct(lambda: nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1)), component="discriminator")
    return context, context.build_trainer(generator, critic)


def matched_state(context):
    checkpoint = context.state_dict()
    trained = checkpoint["trainer"]
    return dict(models=trained["models"], optimizers=trained["optimizers"],
                trainer_streams=trained["streams"], cpu_rng=trained["cpu_rng"],
                cuda_rng=trained["cuda_rng"], streams=checkpoint["streams"])


def gate(metrics):
    return (metrics["finite_fraction"] == 1 and metrics["cdf_ks"] <= .05
            and metrics["mean_error_sigma"] <= .2 and .8 <= metrics["std_ratio"] <= 1.2)


def frame(step, observations):
    figure, axes = plt.subplots(1, 3, figsize=(10, 3))
    grid = np.linspace(-1, 4.5, 512)
    for axis, (strength, samples, metrics, table, initial) in zip(axes[:2], observations):
        axis.plot(grid, ndtr((grid - 2) / .5), color="black", label="Target N(2, .5²)")
        ordered = np.sort(samples[:, 0])
        axis.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label="Actual API samples")
        axis.set(xlim=(-1, 4.5), ylim=(0, 1), title=f"λ={strength:g}; fidelity {'PASS' if gate(metrics) else 'FAIL'}\nKS {metrics['cdf_ks']:.3f}/≤.05", xlabel="Output", ylabel="CDF")
        axis.legend(fontsize=7)
        axes[2].scatter(table[:, 0], table[:, 1], s=12, label=f"Actual prior λ={strength:g}")
    initial = observations[0][4]
    axes[2].scatter(initial[:, 0], initial[:, 1], marker="x", color="black", s=14, label="Initial reference")
    axes[2].set(xlim=(-2, 2), ylim=(-2, 2), title="Desired stable latent positions\nActual learned locations", xlabel="z₁", ylabel="z₂")
    axes[2].legend(fontsize=7)
    figure.suptitle(f"Public BCAP training, update {step}/64; software illustration, seed 0")
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=90)
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


@reproducible_execution
def run(output, strengths=(0., 1.), *, seed=0, device="cpu"):
    if seed != 0 or device != "cpu":
        raise ValueError("this fixed software cohort uses protocol seed0 on CPU")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    quantiles = 2 + .5 * ndtri((np.arange(SAMPLES) + .5) / SAMPLES)
    oracle = score_samples(quantiles[:, None], TARGET)
    destructive = score_samples(quantiles[:, None] + 2, TARGET)
    assert gate(oracle) and not gate(destructive)
    arms = [(value, *build(value)) for value in strengths]
    owned_rows = {strength: 0 for strength in strengths}
    captured_update = {strength: None for strength in strengths}
    def count_rows(strength, trained):
        def before_step(optimizer, args, kwargs):
            # Optimizer inheritance can dispatch the same Torch hook around
            # both the composed step and its base; count one trainer update.
            if captured_update[strength] == trained.completed_steps:
                return
            captured_update[strength] = trained.completed_steps
            rows = optimizer.sampled_rows_for(trained.prior.z)
            if rows is not None:
                owned_rows[strength] += len(rows)
        trained.opt_g.register_step_pre_hook(before_step)
    for strength, _, trained in arms:
        count_rows(strength, trained)
    initial = [trained.prior.z.detach().clone() for _, _, trained in arms]
    base_streams = state_digest(arms[0][1].streams.state_dict())
    assert all(state_digest(context.streams.state_dict()) == base_streams for _, context, _ in arms)
    frames, last, resumes = [], {}, {}
    for step in range(STEPS + 1):
        observations = []
        for index, (strength, context, trained) in enumerate(arms):
            if step:
                stream = context.streams.generator("data", component="target", purpose="batches")
                real = 2 + .5 * torch.randn(BATCH, 1, generator=stream)
                trained.step(real)
                if step == 33:
                    resumes[str(strength)] = state_digest(matched_state(context)) == state_digest(matched_state(restored[strength]))
            if step % 8 == 0:
                samples = trained.sample(SAMPLES).detach().cpu().numpy()
                metrics = score_samples(samples, TARGET)
                last[str(strength)] = metrics
                observations.append((strength, samples, metrics, trained.prior.z.detach().numpy().copy(), initial[index].numpy()))
            if step == 32:
                # Restore each arm's own prefix including spring reference and
                # consumed named streams, then replay only the next update.
                saved = context.state_dict()
                resume_context, resume_trainer = build(strength)
                resume_context.load_state_dict(saved)
                stream = resume_context.streams.generator("data", component="target", purpose="batches")
                resume_trainer.step(2 + .5 * torch.randn(BATCH, 1, generator=stream))
                if index == 0:
                    restored = {}
                restored[strength] = resume_context
        if observations:
            if len(observations) == 2:
                frames.append(frame(step, observations))
            print(json.dumps(dict(event="observation", step=step, metrics=last)), flush=True)
    metrics, passed = {}, all(resumes.values())
    for strength, context, trained in arms:
        stats = getattr(trained.opt_g, "prior_proximal_stats", {})
        passed = passed and last[str(strength)]["finite_fraction"] == 1
        if strength:
            passed = (passed and stats["steps"] == STEPS and stats["rows"] == owned_rows[strength]
                      and owned_rows[strength] > 0 and stats["max_restoration"] > 0)
        metrics[str(strength)] = dict(final=last[str(strength)], fidelity_pass=gate(last[str(strength)]),
                                    spring=stats, matched_state_sha256=state_digest(matched_state(context)),
                                    recipe=trained.recipe.to_dict(), exact_resume=resumes[str(strength)],
                                    captured_owned_rows=owned_rows[strength])
        torch.save(context.state_dict(), output / f"lambda-{strength:g}.pt")
    if frames:
        frames[0].save(output / "goal.gif", save_all=True, append_images=frames[1:], duration=400, loop=0)
    receipt = dict(schema_version=1, scope="public_api_software_illustration", protocol_seed=0,
        question="Does the initial-position spring consume actual owned rows and resume exactly during public BCAP training?",
        target=TARGET, steps=STEPS, batch_size=BATCH, eval_samples=SAMPLES, prior=dict(kind="mog", particles=PARTICLES, z_dim=2, sigma=.1, standardize=False),
        architecture="G 2→8→1, D 1→8→1, tanh, public deterministic initializer", cadence=8,
        software_gate=dict(spring_steps=STEPS, spring_owned_row_updates="equals captured unique owned rows; >0", finite_fraction=1, exact_resume=True, max_restoration=">0"),
        fidelity_thresholds=THRESHOLDS, oracle=oracle, destructive=destructive, metrics=metrics,
        status="PASS" if passed else "FAIL", qualification=False)
    (output / "metrics.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(event="complete", status=receipt["status"])), flush=True)
    if not passed:
        raise RuntimeError("prior spring software illustration failed")
    return receipt


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    run(args.output, strengths=(0.,) if args.baseline_only else (0., 1.))
