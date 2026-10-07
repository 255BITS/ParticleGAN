"""CUDA saved-state/representation audit; zero training, no qualification credit."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.rng import NamedStreams
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models
from .api_vectors import _bounds
from .gaussian1d_quality import score_samples
from .reproducibility import reproducible_execution

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/gaussian1d-diagnosis/protocol.json"


def shape_metrics(samples):
    values = samples.detach().cpu().numpy().astype(np.float64).ravel()
    standardized = (values - values.mean()) / values.std()
    ordered = np.sort(standardized)
    cdf = ndtr(ordered)
    ranks = np.arange(len(values)) / len(values)
    return dict(fitted_normal_ks=float(max(np.max(cdf - ranks), np.max(ranks + 1 / len(values) - cdf))),
                skewness=float(np.mean(standardized ** 3)),
                excess_kurtosis=float(np.mean(standardized ** 4) - 3),
                central_99_percent_range=np.quantile(values, [.005, .995]).tolist())


def context_from_state(protocol, path, device):
    task = json.loads((ROOT / protocol["task_path"]).read_text())
    task["execution"]["prior"] = deepcopy(protocol["diagnostic_prior"])
    candidate = json.loads((ROOT / protocol["candidate_path"]).read_text())
    saved = torch.load(path, map_location="cpu", weights_only=False)
    context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
    generator, critic = build_vector_models(context, task["execution"]["host_definition"])
    trainer = context.build_trainer(generator, critic, max_steps=saved["trainer"].get("max_steps", 1000))
    context.load_state_dict(saved)
    if state_digest(context.state_dict()) != state_digest(saved):
        raise ValueError("saved context did not restore exactly")
    assert all(p.device.type == "cuda" for m in (trainer.G, trainer.D, trainer.prior) for p in m.parameters())
    return task, context, trainer


@torch.no_grad()
def affine_fixture(generator, slope, intercept):
    """Explicit target-informed capacity control, separate from trained evidence.

    Two signed channels carry z0 through both LeakyReLU(.2) layers. Their
    difference after two activations is (1 + .2**2) * z0.
    """
    for parameter in generator.parameters():
        parameter.zero_()
    first, second, last = generator.net[0], generator.net[2], generator.net[4]
    first.weight[0, 0], first.weight[1, 0] = 1., -1.
    second.weight[0, 0], second.weight[1, 1] = 1., 1.
    last.weight[0, 0], last.weight[0, 1] = slope / 1.04, -slope / 1.04
    last.bias[0] = intercept


def component_audit(trainer, streams, count, original_locations):
    """Equal draws per component, separate diagnostic law (not gate sampling)."""
    prior, generator = trainer.prior, trainer.G
    z = prior.z.detach()
    rng = streams.generator("eval", component="component", purpose="jitter")
    noise = torch.randn(len(z), count, z.shape[1], device=z.device, generator=rng) * prior.sigma
    with torch.no_grad():
        values = generator((z[:, None, :] + noise).flatten(0, 1)).reshape(len(z), count)
        centers = generator(z).ravel()
    within = values.var(dim=1, unbiased=False).mean()
    between = values.mean(dim=1).var(unbiased=False)
    inputs = z.clone().requires_grad_(True)
    derivatives = torch.autograd.grad(generator(inputs).sum(), inputs)[0].norm(dim=1)
    delta = (z - original_locations.to(z.device)).norm(dim=1)
    return dict(within_component_variance=float(within), between_component_variance=float(between),
                within_variance_fraction=float(within / (within + between)),
                center_only_metrics=score_samples(centers[:, None], SPEC),
                prior_coordinate_mean=z.mean(dim=0).tolist(),
                prior_coordinate_std=z.std(dim=0, unbiased=False).tolist(),
                prior_displacement_from_initial_mean=float(delta.mean()),
                prior_displacement_from_initial_max=float(delta.max()),
                center_latent_jacobian_norm_median=float(derivatives.median()),
                center_latent_jacobian_norm_max=float(derivatives.max()))


def critic_audit(trainer):
    # Deterministic grid: no target, training or evaluation stream consumption.
    x = torch.linspace(0., 4., 1025, device=trainer.device)[:, None].requires_grad_(True)
    scores = trainer.D(x)
    slope = torch.autograd.grad(scores.sum(), x)[0].ravel()
    core = (x.detach().ravel() >= 1.) & (x.detach().ravel() <= 3.)
    return dict(grid=[0., 4., 1025], score_min=float(scores.min().detach()),
                score_max=float(scores.max().detach()),
                abs_input_gradient_median=float(slope.abs().median()),
                abs_input_gradient_max=float(slope.abs().max()),
                fraction_grid_above_bcap_kappa=float((slope.abs() > 1.).float().mean()),
                core_signed_input_gradient_mean=float(slope[core].mean()),
                core_positive_input_gradient_fraction=float((slope[core] > 0).float().mean()))


SPEC = {"kind": "gaussian_mixture", "means": [[2.]], "covariances": [[[.25]]], "masses": [1.]}


@reproducible_execution
def run(output, *, device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Gaussian diagnosis requires CUDA; CPU fallback is forbidden")
    protocol = json.loads(PROTOCOL.read_text())
    for path, expected in protocol["inputs"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"frozen evidence changed or absent: {path}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    streams = NamedStreams(0, device=device)
    raw_rows, seen = [], set()
    # Read the longest saved trajectory once; its prefix is identical to parent.
    observations = torch.load(ROOT / protocol["observations_path"], map_location="cpu", weights_only=True)
    curve = json.loads((ROOT / protocol["curve_path"]).read_text())
    task = json.loads((ROOT / protocol["task_path"]).read_text())
    bounds = task["evaluation"]["thresholds"]
    for row in observations:
        if row["step"] in seen:
            raise ValueError("duplicate observation")
        seen.add(row["step"])
        metrics = score_samples(row["samples"], SPEC)
        if metrics != row["metrics"]:
            raise ValueError("saved samples do not reproduce original metrics")
        raw_rows.append(dict(step=row["step"], metrics=metrics, **shape_metrics(row["samples"]),
                             failed_bounds=_bounds(metrics, bounds)))
    by_step = {row["step"]: row for row in raw_rows}
    for row in curve:
        if row["metrics"] != by_step[row["step"]]["metrics"] or row["full_pass"] != (not by_step[row["step"]]["failed_bounds"]):
            raise ValueError("saved curve disagrees with samples/gate")
    atomic_json(output / "observations-analysis.json", raw_rows)
    initial = torch.load(ROOT / protocol["states"][0]["path"], map_location="cpu", weights_only=False)
    original_locations = initial["trainer"]["models"]["prior"]["z"]
    states = []
    for binding in protocol["states"]:
        _, context, trainer = context_from_state(protocol, ROOT / binding["path"], device)
        digest = state_digest(context.state_dict())
        trainer.G.eval(); trainer.D.eval()
        rng = streams.generator("eval", component="live", purpose="large_sample")
        samples = trainer.sample(protocol["fresh_sample_count"], generator=rng, output_noise=False)
        result = dict(step=trainer.completed_steps, restored_exactly=True,
                      large_sample_metrics=score_samples(samples, SPEC), **shape_metrics(samples),
                      components=component_audit(trainer, streams, protocol["draws_per_component"], original_locations),
                      critic=critic_audit(trainer),
                      parameter_counts={name: sum(p.numel() for p in model.parameters()) for name, model in
                                        (("generator", trainer.G), ("critic", trainer.D), ("prior", trainer.prior))},
                      effective_rates=[[g["lr"] for g in opt.param_groups] for opt in (trainer.opt_g, trainer.opt_d)])
        if state_digest(context.state_dict()) != digest:
            raise ValueError("read-only probes mutated saved models/optimizer/training streams")
        states.append(result)
        print(f"state={trainer.completed_steps} KS={result['large_sample_metrics']['cdf_ks']:.6f} "
              f"within_var_fraction={result['components']['within_variance_fraction']:.4f}", flush=True)
    _, context, trainer = context_from_state(protocol, ROOT / protocol["states"][0]["path"], device)
    z = trainer.prior.z.detach().cpu().numpy()[:, 0].astype(np.float64)
    # Match exact mixture moments; the initial centers, width, masses stay fixed.
    slope = .5 / np.sqrt(z.var() + .1 ** 2)
    intercept = 2. - slope * z.mean()
    affine_fixture(trainer.G, slope, intercept)
    grid = torch.linspace(-10, 10, 1001, device=device)
    latent = torch.stack((grid, torch.zeros_like(grid)), dim=1)
    with torch.no_grad():
        error = float((trainer.G(latent).ravel() - (slope * grid + intercept)).abs().max())
    if error > 2e-6:
        raise ValueError("fixed affine fixture implementation failed")
    control = []
    rng = streams.generator("eval", component="affine_capacity", purpose="live")
    for check in range(protocol["control_checks"]):
        samples = trainer.sample(4096, generator=rng, output_noise=False)
        metrics = score_samples(samples, SPEC)
        control.append(dict(check=check + 1, metrics=metrics, passed=not _bounds(metrics, bounds)))
    x = np.linspace(-2., 6., 32769)
    exact_cdf = ndtr((x[:, None] - (intercept + slope * z)[None, :]) / (slope * .1)).mean(axis=1)
    exact_gap = np.abs(exact_cdf - ndtr((x - 2.) / .5))
    # CDF difference Lipschitz constant: sum of density upper bounds. This
    # covers unsampled x between grid nodes, plus negligible [-2,6] tails.
    grid_upper = float(exact_gap.max() + (1 / (slope * .1) + 1 / .5) / np.sqrt(2 * np.pi) * np.diff(x)[0] / 2)
    affine = dict(scope="fixed_target_informed_representation_control_not_training", training_updates=0,
                  unchanged_initial_prior=True, slope=float(slope), intercept=float(intercept),
                  affine_max_abs_error=error, checks_passed=sum(row["passed"] for row in control),
                  checks_total=len(control), exact_cdf_grid_max_error=float(exact_gap.max()),
                  exact_cdf_grid_lipschitz_upper_bound=grid_upper, observations=control)
    atomic_json(output / "affine-control.json", affine)
    torch.save(streams.state_dict(), output / "diagnostic-rng.pt")
    counts = {bound: sum(bound in row["failed_bounds"] for row in raw_rows[1:])
              for bound in sorted({b for row in raw_rows[1:] for b in row["failed_bounds"]})}
    selected_steps = sorted({0, 1000, 2000, 3000, 3334, 4000} | {row["step"] for row in raw_rows[1:] if not row["failed_bounds"]})
    result = dict(schema_version=1, id=protocol["id"], scope=protocol["scope"], training_updates=0,
                  qualification_input=False, seed=0, device=device, gpu=torch.cuda.get_device_name(device),
                  protocol_sha256=file_hash(PROTOCOL), inputs=protocol["inputs"],
                  analysis_source_sha256=file_hash(Path(__file__)), original_recipe_sha256=stable_hash(initial["recipe"]),
                  saved_observations_scored=len(raw_rows), failure_counts=counts,
                  saved_full_pass_checks=sum(not row["failed_bounds"] for row in raw_rows[1:]),
                  selected_saved_observations=[by_step[s] for s in selected_steps], saved_states=states,
                  affine_capacity={k: v for k, v in affine.items() if k != "observations"},
                  rng_manifest=streams.manifest(), rng_final_states=streams.audit(),
                  elapsed_seconds=time.monotonic() - started)
    atomic_json(output / "results.json", result)
    print(f"affine_capacity={affine['checks_passed']}/{affine['checks_total']} "
          f"exact_KS_upper={grid_upper:.6f}; zero training updates", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.output, device=args.device)


if __name__ == "__main__":
    main()
