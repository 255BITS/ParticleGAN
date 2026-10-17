"""A bounded Tier 1 scalar acquisition caller using the shared public trainer."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import gaussian1d_quality
from .api_vectors import VectorFixture, _case, _digest, _law

TASK_PATH = Path(__file__).resolve().parents[2] / "configs/forge/tasks/gaussian1d_acquisition.json"
CASE_ID = "api-gaussian1d-acquisition"


def list_cases():
    task = json.loads(TASK_PATH.read_text())
    spec = deepcopy(task["execution"]["host_definition"])
    spec["thresholds"] = deepcopy(task["evaluation"]["thresholds"])
    prior = task["execution"]["prior"]
    law = _law(spec)
    return [_case(
        CASE_ID, ["develop-gaussian1d_acquisition"], "1-D Gaussian: histogram matching",
        task["description"], steps=spec["steps"], batch=spec["batch"], evaluation=4096,
        default_recipe="k3p", particles=spec["particles"], z_dim=spec["z_dim"],
        metric_family="gaussian1d_acquisition", spec=spec, thresholds=spec["thresholds"],
        sample_evaluator="benchmarks.toy_audit.gaussian1d_quality:score_samples",
        task_sha256=hashlib.sha256(TASK_PATH.read_bytes()).hexdigest(),
        law=law, law_sha256=_digest(law), protocol_seed=0,
        evaluation_observations=24, terminal_observations=5,
        recipe_overrides=dict(prior_kind="mog", sigma_rel=0., standardize=prior["standardize"]),
        prior_options=dict(sigma=prior["sigma"]), profile=dict(init_std=.5),
        initialization=dict(generator="deterministic_orthogonal seed0", discriminator="deterministic_orthogonal seed1", prior="deterministic_orthogonal R2Normal init_std=.5"),
        sampling="Live public GANTrainer.sample(output_noise=False); uniformly sampled learned MoG locations with fixed sigma=.025 latent Gaussian noise; standardize=False; no EMA.",
        scope="Acquire N(2, .5^2) in 1,000 updates from deterministic initialization; exact Gaussian CDF, location and width gates must pass at five terminal checks. Tier 1 remains provisional; this standalone API cohort grants no whole-view qualification.",
        adaptation="New scalar question. Reuses vector MLPs, explicit learned MoG and GANTrainer; output dimension is one. All prior 2-D targets and recorded results retain their identities.")]


class Gaussian1DFixture(VectorFixture):
    api_components = ("particlegan.Recipe", "particlegan.GANTrainer", "particlegan.MoGParticlePrior")

    def observe(self, n=4096, seed=713):
        observed = super().observe(n=n, seed=seed)
        view = observed["views"][0]
        # The raw draws have already been scored; these counts are display-only.
        edges = np.linspace(-2., 5., 57)
        width = np.diff(edges)
        def density(points):
            values = np.asarray(points)[:, 0]
            return np.histogram(values, edges)[0] / len(values) / width
        observed["views"] = [dict(
            kind="bar", title="Desired and learned 1-D Gaussian histograms",
            target=density(view["target"]), samples=density(view["samples"]),
            bin_centers=(edges[:-1] + edges[1:]) / 2, bin_width=float(width[0]),
            xlim=[-2., 5.], ylim=[0., 8.5], xlabel="x", ylabel="density",
            caption="N(2, 0.5²); fixed bins/axes. The gate uses all samples and the exact CDF, including tails outside the plot.")]
        return observed


def build_case(id, *, device="cpu", seed=0, recipe_name="k3p", max_steps=None):
    if id != CASE_ID:
        raise ValueError(f"unknown scalar case: {id}")
    return Gaussian1DFixture(list_cases()[0], device=device, seed=seed,
                             recipe_name=recipe_name, max_steps=max_steps)
