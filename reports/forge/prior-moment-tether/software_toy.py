"""Scoped public-API restoring-force diagnostic; grants no Tier 2 qualification."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn

from experiments.forge.api import FormulationContext
from experiments.forge.state import state_digest


ROOT = Path(__file__).resolve().parents[3]
STEPS = 80
WEIGHTS = (.001, .01, .1)
GATE_RATIO = .1


def run(output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    leader = json.loads((ROOT / "configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json").read_text())
    settings = dict(leader["recipe_overrides"], num_particles=64, z_dim=2,
                    total_steps=STEPS, batch_size=64,
                    prior_regularizer="moment_tether", prior_reg=.1)
    ctx = FormulationContext(recipe_preset="bcap", recipe_overrides=settings,
        prior=dict(kind="mog", sigma=.025, standardize=False),
        prior_contract="recipe_owned_v1", seed=0)
    generator = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)), component="generator")
    critic = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)), component="discriminator")
    trainer = ctx.build_trainer(generator, critic)
    initial = trainer.prior.z.detach().clone()
    references = {name: value.clone() for name, value in trainer.prior.state_dict().items()
                  if name.startswith("moment_tether_")}
    initial_models = [deepcopy(module.state_dict()) for module in (generator, critic)]
    loss = trainer.prior_regularizer
    oracle = float(loss(initial))
    # Explicit perturbation cohort, after fixed-reference capture; no hidden baseline substitution.
    with torch.no_grad():
        trainer.prior.z.mul_(1.35).add_(torch.tensor([.4, -.3]))
    initial_loss = float(loss(trainer.prior.z).detach())
    # Coefficient units: gradient after the declared deformation, before DualNorm.
    gradient = torch.autograd.grad(loss(trainer.prior.z), trainer.prior.z)[0]
    calibration = {str(weight): dict(max_row_gradient=float((weight * gradient).norm(dim=1).max()),
                                    mean_row_gradient=float((weight * gradient).norm(dim=1).mean()))
                   for weight in WEIGHTS}
    target_rng = ctx.streams.generator("data", component="target", purpose="training")
    evaluation_rng = ctx.streams.generator("eval", component="target", purpose="display")
    target = .5 * torch.randn(512, 2, generator=evaluation_rng)
    frames, observed = [], []
    checks = set(range(0, STEPS + 1, 4))
    for step in range(STEPS + 1):
        if step:
            # Explicit target law: unconditional isotropic N(0, .5^2 I), separate data stream.
            trainer.step(.5 * torch.randn(64, 2, generator=target_rng))
        if step not in checks:
            continue
        penalty = float(loss(trainer.prior.z).detach())
        samples = trainer.sample(512).cpu()
        row = dict(step=step, moment_discrepancy=penalty,
                   ratio_to_perturbed_initial=penalty / initial_loss,
                   finite_samples=bool(torch.isfinite(samples).all()))
        observed.append(row)
        print(json.dumps(row), flush=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        live = trainer.prior.z.detach().cpu()
        axes[0].scatter(initial[:, 0], initial[:, 1], s=12, label="Initial reference", alpha=.6)
        axes[0].scatter(live[:, 0], live[:, 1], s=12, label="Actual trained prior", alpha=.7)
        axes[0].set_title(f"Prior moments: ratio {row['ratio_to_perturbed_initial']:.3f} <= {GATE_RATIO}")
        axes[1].scatter(target[:, 0], target[:, 1], s=6, label="Target Gaussian", alpha=.3)
        axes[1].scatter(samples[:, 0], samples[:, 1], s=6, label="Actual API outputs", alpha=.3)
        axes[1].set_title("Concurrent BCAP G / D training")
        for axis in axes:
            axis.set_xlim(-2, 2)
            axis.set_ylim(-2, 2)
            axis.set_aspect("equal")
            axis.legend(fontsize=7, loc="lower left")
        fig.suptitle(f"Moment tether software diagnostic | seed 0 | update {step}/{STEPS}")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=90)
        plt.close(fig)
        frames.append(Image.open(buffer).convert("RGB"))
    gif = output / "training.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=150, loop=0)
    unchanged = all(torch.equal(value, trainer.prior.state_dict()[name]) for name, value in references.items())
    changed = all(any(not torch.equal(old[name], value) for name, value in model.state_dict().items())
                  for old, model in zip(initial_models, (generator, critic)))
    final = observed[-1]
    passed = final["ratio_to_perturbed_initial"] <= GATE_RATIO and final["finite_samples"] and unchanged and changed
    receipt = dict(schema_version=1, scope="software_perturbed_initial_moment_restoration_only",
        scientific_qualification=False, protocol_seed=0, public_initializer=ctx.initializer,
        target_law="N(0, .5^2 I) in two dimensions", sampling="clean live MoG API outputs; fixed width .025",
        perturbation="after initial reference capture: z <- 1.35*z + [.4, -.3]",
        steps=STEPS, resolved_recipe=trainer.recipe.to_dict(),
        gate=dict(final_discrepancy_ratio_at_most=GATE_RATIO, finite_outputs=True,
                  initial_references_unchanged=True, both_networks_trained=True),
        scorer_controls=dict(initial_oracle_discrepancy=oracle, perturbed_initial_discrepancy=initial_loss),
        coefficient_gradient_calibration=calibration, final=final,
        references_unchanged=unchanged, both_networks_trained=changed,
        status="PASS" if passed else "FAIL", streams_sha256=state_digest(ctx.streams.state_dict()),
        gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest())
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    torch.save(ctx.state_dict(), output / "state.pt")
    print(json.dumps(dict(status=receipt["status"], output=str(output))), flush=True)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
