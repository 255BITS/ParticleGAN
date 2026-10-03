"""Standalone acquisition caller for the provisional Forge sixteen-mode ring.

The shared vector fixture owns all training, scoring and actual-state views.
This named caller retains its own initializer/RNG cohort; its media supplies no
automatic Forge qualification or scientific tier calibration.
"""
from copy import deepcopy
import json
import hashlib
from pathlib import Path

from .api_vectors import VectorFixture, _case, _digest, _law
from . import ring16_quality  # Bind the declared scorer before run source capture.


TASK_PATH = Path(__file__).resolve().parents[2] / "configs/forge/tasks/ring16_acquisition.json"
CASE_ID = "api-ring16-acquisition"


def list_cases():
    task = json.loads(TASK_PATH.read_text())
    spec = deepcopy(task["execution"]["host_definition"])
    spec["thresholds"] = deepcopy(task["evaluation"]["thresholds"])
    prior = task["execution"]["prior"]
    law = _law(spec)
    return [_case(
        CASE_ID, ["develop-ring16_acquisition"], "Sixteen Gaussian clusters: acquisition",
        task["description"], steps=spec["steps"], batch=spec["batch"], evaluation=4096,
        default_recipe="k3p", particles=spec["particles"], z_dim=spec["z_dim"],
        metric_family="ring16_acquisition",
        sample_evaluator=ring16_quality.score_samples.__module__ + ":score_samples",
        task_sha256=hashlib.sha256(TASK_PATH.read_bytes()).hexdigest(),
        spec=spec, thresholds=spec["thresholds"], law=law, law_sha256=_digest(law),
        protocol_seed=0, evaluation_observations=24, terminal_observations=5,
        recipe_overrides=dict(prior_kind="mog", sigma_rel=0., standardize=prior["standardize"]),
        prior_options=dict(sigma=prior["sigma"]), profile=dict(init_std=.5),
        sampling="Live generator; uniform row draws from the learned MoG plus fixed sigma=.025 Gaussian latent noise; standardize=False; output_noise=False; no EMA or policy serving.",
        scope="From random initialization, acquire sixteen equal radius-three sigma-.1 Gaussian clusters within 400 updates. Five terminal acquisition checks add no hold phase. Tier 1 is provisional; this standalone K3P/API initializer and RNG cohort supplies no Forge promotion credit.",
        adaptation="New retained question develop-ring16_acquisition; shared transfer_vector target/scorer and public GANTrainer, with an explicit nonzero-width MoG. Existing ring8 acquisition/hold evidence is unchanged.",
        initialization=dict(generator="deterministic_orthogonal seed0", discriminator="deterministic_orthogonal seed1", prior="Gaussian init_std=.5 seed0"))]


def build_case(id, *, device="cpu", seed=0, recipe_name="k3p", max_steps=None):
    if id != CASE_ID:
        raise ValueError(f"unknown ring16 case: {id}")
    return VectorFixture(list_cases()[0], device=device, seed=seed,
                         recipe_name=recipe_name, max_steps=max_steps)
