"""Standalone acquisition caller for the provisional Forge sixteen-mode ring.

The shared vector fixture owns all training, scoring and actual-state views.
This named caller retains its own initializer/RNG cohort; its media supplies no
automatic Forge qualification or scientific tier calibration.
"""
from copy import deepcopy
import json
import hashlib
import math
from pathlib import Path

import torch

from .api_vectors import VectorFixture, _case, _digest, _law, _view, score_case
from . import ring16_quality  # Bind the declared scorer before run source capture.


TASK_PATH = Path(__file__).resolve().parents[2] / "configs/forge/tasks/ring16_acquisition.json"
CASE_ID = "api-ring16-acquisition"
CURRENT_CASE_ID = "api-ring16-acquisition-v2"
ROOT = TASK_PATH.parents[3]
LEGACY_TASK_PATH = ROOT / "reports/forge/tier1-prior-smoke/frozen-tasks/ring16_acquisition.json"
CANDIDATE_PATH = ROOT / "configs/forge/configurations/bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9.json"


def _legacy_cases():
    task = json.loads(LEGACY_TASK_PATH.read_text())
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
        task_sha256=hashlib.sha256(LEGACY_TASK_PATH.read_bytes()).hexdigest(),
        spec=spec, thresholds=spec["thresholds"], law=law, law_sha256=_digest(law),
        protocol_seed=0, evaluation_observations=24, terminal_observations=5,
        recipe_overrides=dict(prior_kind="mog", sigma_rel=0., standardize=prior["standardize"]),
        prior_options=dict(sigma=prior["sigma"]), profile=dict(init_std=.5),
        sampling="Live generator; uniform row draws from the learned MoG plus fixed sigma=.025 Gaussian latent noise; standardize=False; output_noise=False; no EMA or policy serving.",
        scope="From deterministic initialization, acquire sixteen equal radius-three sigma-.1 Gaussian clusters within 400 updates. Five terminal acquisition checks add no hold phase. Tier 1 is provisional; this standalone K3P/API initializer and RNG cohort supplies no Forge promotion credit.",
        adaptation="New retained question develop-ring16_acquisition; shared transfer_vector target/scorer and public GANTrainer, with an explicit nonzero-width MoG. Existing ring8 acquisition/hold evidence is unchanged.",
        initialization=dict(generator="deterministic_orthogonal seed0", discriminator="deterministic_orthogonal seed1", prior="deterministic_orthogonal R2Normal init_std=.5"))]


def list_cases():
    task = json.loads(TASK_PATH.read_text())
    spec = deepcopy(task["execution"]["host_definition"])
    spec["thresholds"] = deepcopy(task["evaluation"]["thresholds"])
    current = deepcopy(_legacy_cases()[0])
    current.update(
        id=CURRENT_CASE_ID, title="Sixteen Gaussian clusters: GPU smoke",
        goal=task["description"], default_steps=spec["steps"], spec=spec,
        default_recipe="bcap", evaluation_observations=task["evaluation"]["observations"],
        task_sha256=hashlib.sha256(TASK_PATH.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest(),
        trainer_configuration=str(CANDIDATE_PATH.relative_to(ROOT)),
        prior_options=dict(sigma=task["execution"]["prior"]["sigma"]),
        profile=dict(init_std=1.),
        initialization=dict(policy="deterministic_orthogonal", streams="Forge named constructor/init streams", prior="R2Normal init_std=1"),
        sampling="Clean live public MoG: 256 learned locations, uniform weights, sigma .1, standardize=False; checkpointed evaluation stream; no output noise or EMA.",
        scope="GPU acquisition smoke at seed 0: the selected whole BCAP dualnorm configuration, 1600 updates, 96 checks and five terminal full passes. Preserves the numerical bounds; scientific calibration remains provisional.",
        adaptation="User-selected ring16 conditions from the saved duration diagnostic. The original 400-update sigma-.025 K3P caller remains a distinct legacy case.")
    return [current, *_legacy_cases()]


class RingSmokeFixture(VectorFixture):
    """Use the proven Forge context; reuse only the vector observer geometry."""
    api_components = ("particlegan.Recipe", "particlegan.GANTrainer", "particlegan.MoGParticlePrior", "particlegan.init")

    def __init__(self, case, *, device, seed, recipe_name, max_steps):
        from experiments.forge.api import task_formulation_context
        from experiments.forge.vectorprofiles import build_vector_models
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("ring16 smoke requires CUDA; CPU fallback is forbidden")
        if seed != 0 or recipe_name not in (None, "auto", "bcap"):
            raise ValueError("ring16 smoke is bound to seed 0 and the declared BCAP configuration")
        self.metadata, self.case_id = deepcopy(case), case["id"]
        self.device, self.seed = torch.device(device), seed
        self.execution_steps = case["default_steps"] if max_steps is None else max_steps
        if type(self.execution_steps) is not int or not 1 <= self.execution_steps <= case["default_steps"]:
            raise ValueError("max_steps must be a positive prefix of the declared full budget")
        task = json.loads(TASK_PATH.read_text())
        candidate = json.loads(CANDIDATE_PATH.read_text())
        self.context = task_formulation_context(candidate, task, {"seed": seed}, device=device, root=ROOT)
        g, d = build_vector_models(self.context, task["execution"]["host_definition"])
        self.trainer = self.context.build_trainer(g, d, max_steps=self.execution_steps)
        self.recipe, self.completed_steps = self.trainer.recipe, 0
        self.data_rng = self.context.streams.generator("data", component="target", purpose="training", device="cpu")
        self.eval_rng = self.context.streams.generator("eval", component="live", purpose="samples")
        self.checks = {math.ceil(i * case["default_steps"] / case["evaluation_observations"])
                       for i in range(1, case["evaluation_observations"] + 1)}
        self.batch_sequence_sha256 = hashlib.sha256(b"ring16-batches-v2").hexdigest()
        self.cached_samples = None

    def step(self):
        from benchmarks.transfer_suite.vector_tasks import sample_target
        if self.completed_steps >= self.execution_steps:
            raise RuntimeError("declared execution prefix is complete")
        real = sample_target(self.metadata["spec"], self.recipe.batch_size, self.data_rng, self.completed_steps)
        self.batch_sequence_sha256 = hashlib.sha256(bytes.fromhex(self.batch_sequence_sha256) + real.numpy().tobytes()).hexdigest()
        result = self.trainer.step(real.to(self.device))
        self.completed_steps = self.trainer.completed_steps
        self.cached_samples = None
        if self.completed_steps in self.checks:
            self.cached_samples = self.trainer.sample(4096, generator=self.eval_rng, output_noise=False).detach().cpu()
        return result

    @torch.no_grad()
    def observe(self, n=4096, seed=10000):
        from benchmarks.transfer_suite.vector_tasks import sample_target
        if type(n) is not int or n < 1 or seed != 10000:
            raise ValueError("observer requires a positive sample count and the declared evaluation seed 10000")
        samples = self.cached_samples if n == 4096 else None
        if samples is None:
            with self.context.streams.preserve():
                samples = self.trainer.sample(n, generator=self.eval_rng, output_noise=False).detach().cpu()
        target = sample_target(self.metadata["spec"], n, torch.Generator().manual_seed(78013), self.completed_steps)
        scored = score_case(self.metadata, samples, self.completed_steps)
        centers, sigma = self._centers(), self._zoom_sigma()
        masses = lambda points: torch.bincount(torch.cdist(points, centers).argmin(1), minlength=16).float() / len(points)
        scored["views"] = [
            _view("Whole target law", target, samples, self.metadata),
            dict(kind="bar", title="All mode masses", target=masses(target), samples=masses(samples)),
            dict(kind="scatter", title="Fixed mode0: local width", target=target, samples=samples,
                 xlim=[float(centers[0, 0]-4*sigma), float(centers[0, 0]+4*sigma)],
                 ylim=[float(centers[0, 1]-4*sigma), float(centers[0, 1]+4*sigma)])]
        return scored

    def state_dict(self):
        return dict(case_sha256=_digest(self.metadata), context=self.context.state_dict(),
                    completed_steps=self.completed_steps, cached_samples=self.cached_samples,
                    batch_sequence_sha256=self.batch_sequence_sha256)

    def load_state_dict(self, state):
        if state["case_sha256"] != _digest(self.metadata):
            raise ValueError("checkpoint case differs")
        self.context.load_state_dict(state["context"])
        if state["completed_steps"] != self.trainer.completed_steps:
            raise ValueError("checkpoint caller and trainer clocks differ")
        self.completed_steps = self.trainer.completed_steps
        self.cached_samples = deepcopy(state["cached_samples"])
        self.batch_sequence_sha256 = state["batch_sequence_sha256"]


def build_case(id, *, device="cpu", seed=0, recipe_name=None, max_steps=None):
    if id == CURRENT_CASE_ID:
        return RingSmokeFixture(list_cases()[0], device=device, seed=seed, recipe_name=recipe_name, max_steps=max_steps)
    if id != CASE_ID:
        raise ValueError(f"unknown ring16 case: {id}")
    return VectorFixture(_legacy_cases()[0], device=device, seed=seed,
                         recipe_name=recipe_name or "k3p", max_steps=max_steps)
