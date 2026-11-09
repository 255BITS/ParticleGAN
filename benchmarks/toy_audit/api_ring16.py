"""Standalone acquisition caller for the provisional Forge sixteen-mode ring.

The shared vector fixture owns all training, scoring and actual-state views.
This named caller retains its own initializer/RNG cohort; its media supplies no
automatic Forge qualification or scientific tier calibration.
"""
from copy import deepcopy
from contextlib import contextmanager
from functools import wraps
import json
import hashlib
import math
from pathlib import Path

import torch

import torch.nn as nn
import torch.nn.functional as F

from particlegan import (
    ManipulatedPrior,
    MoGParticlePrior,
    Recipe,
    get_recipe,
    init,
    scale_learning_rates,
    unwrap_prior,
)
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from .api_vectors import VectorFixture, _case, _digest, _law, _view, score_case
from . import ring16_quality  # Bind the declared scorer before run source capture.


TASK_PATH = Path(__file__).resolve().parents[2] / "configs/forge/tasks/ring16_acquisition.json"
BIGAN_TASK_PATH = Path(__file__).resolve().parents[2] / "configs/forge/tasks/ring16_bigan_continuous.json"
ALICE_SPIRAL_TASK_PATH = Path(__file__).resolve().parents[2] / "configs/forge/tasks/alice_spiral_bigan.json"
CASE_ID = "api-ring16-acquisition"
CURRENT_CASE_ID = "api-ring16-acquisition-v2"
BIGAN_CASE_ID = "api-ring16-bigan-continuous"
ALICE_SPIRAL_CASE_ID = "api-alice-spiral-bigan"
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
        recipe_overrides={},
        prior_options=dict(prior_kind="mog", sigma_rel=0., standardize=prior["standardize"],
                           sigma=prior["sigma"]), profile=dict(init_std=.5),
        sampling="Live generator; uniform row draws from the learned MoG plus fixed sigma=.025 Gaussian latent noise; standardize=False; output_noise=False; no EMA or policy serving.",
        scope="From deterministic initialization, acquire sixteen equal radius-three sigma-.1 Gaussian clusters within 400 updates. Five terminal acquisition checks add no hold phase. Tier 1 is provisional; this standalone K3P/API initializer and RNG cohort supplies no Forge promotion credit.",
        adaptation="New retained question develop-ring16_acquisition; shared transfer_vector target/scorer and public GANTrainer, with an explicit nonzero-width MoG. Existing ring8 acquisition/hold evidence is unchanged.",
        initialization=dict(generator="deterministic_orthogonal seed0", discriminator="deterministic_orthogonal seed1", prior="deterministic_orthogonal R2Normal init_std=.5"))]


def _bigan_case():
    task = json.loads(BIGAN_TASK_PATH.read_text())
    return {
        "id": BIGAN_CASE_ID,
        "legacy_ids": ["develop-ring16-bigan-continuous"],
        "title": "Sixteen Gaussian clusters: continuous multimodal joint BiGAN",
        "goal": task["description"],
        "kind": "vector",
        "default_steps": task["execution"]["steps"],
        "batch_size": task["execution"]["host_definition"]["batch"],
        "eval_samples": task["evaluation"]["eval_samples"],
        "default_recipe": "k3p",
        "supported_recipes": ["k3p", "ka2"],
        "terminal_observations": 5,
        "evaluation_observations": 24,
        "protocol_seed": 0,
        "thresholds": {
            "modes_min": 16,
            "hq_min": 0.85,
            "mass_tv_max": 0.15,
            "recon_rmse_max": 0.12,
            "mode_consistency_min": 0.98,
        },
        "sampling": "Continuous 16-Gaussian ring (R=3.0, sigma=0.1); MoGParticlePrior(sigma=0.1); paired round-trip x -> G(E(x))",
        "scope": "Tier 4 continuous 2D multimodal joint BiGAN (G(z), E(x), D(x, z)).",
    }


def _alice_spiral_case():
    task = json.loads(ALICE_SPIRAL_TASK_PATH.read_text())
    return {
        "id": ALICE_SPIRAL_CASE_ID,
        "legacy_ids": ["develop-alice-spiral-bigan"],
        "title": "Archimedes spiral: continuous 2D manifold ALICE / BiGAN",
        "goal": task["description"],
        "kind": "vector",
        "default_steps": task["execution"]["steps"],
        "batch_size": task["execution"]["host_definition"]["batch"],
        "eval_samples": task["evaluation"]["eval_samples"],
        "default_recipe": "k3p",
        "supported_recipes": ["k3p", "ka2"],
        "terminal_observations": 5,
        "evaluation_observations": 24,
        "protocol_seed": 0,
        "thresholds": {
            "arc_bins_min": 15,
            "hq_min": 0.85,
            "mass_tv_max": 0.16,
            "recon_rmse_max": 0.14,
            "latent_recon_rmse_max": 0.25,
            "arc_consistency_min": 0.90,
        },
        "sampling": "Continuous 2D Archimedes spiral (theta in [pi/2, 3.5*pi], r=0.5+0.25*theta, sigma=0.06); MoGParticlePrior(sigma=0.06); paired ALICE cycle x -> G(E(x)) and z -> E(G(z))",
        "scope": "Tier 4 continuous 2D manifold joint ALICE/BiGAN (G(z), E(x), D(x, z)) with cycle-consistency regularization.",
    }


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
        prior_options=dict(prior_kind="mog", sigma_rel=0.,
                           standardize=task["execution"]["prior"]["standardize"],
                           sigma=task["execution"]["prior"]["sigma"]),
        profile=dict(init_std=1.),
        initialization=dict(policy="deterministic_orthogonal", streams="Forge named constructor/init streams", prior="R2Normal init_std=1"),
        sampling="Clean live public MoG: 256 learned locations, uniform weights, sigma .1, standardize=False; checkpointed evaluation stream; no output noise or EMA.",
        scope="GPU acquisition smoke at seed 0: the selected whole BCAP dualnorm configuration, 1600 updates, 96 checks and five terminal full passes. Preserves the numerical bounds; scientific calibration remains provisional.",
        adaptation="User-selected ring16 conditions from the saved duration diagnostic. The original 400-update sigma-.025 K3P caller remains a distinct legacy case.")
    return [current, *_legacy_cases(), _bigan_case(), _alice_spiral_case()]


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


@torch.no_grad()
def score_bigan(samples, target, reconstructed, centers=None, sigma=0.10):
    if centers is None:
        angles = torch.linspace(0, 2 * math.pi, 17, device=samples.device)[:16]
        centers = torch.stack([3.0 * torch.cos(angles), 3.0 * torch.sin(angles)], dim=1).float()
    else:
        centers = centers.to(samples.device).float()

    if not torch.all(torch.isfinite(samples)) or not torch.all(torch.isfinite(reconstructed)):
        metrics = {
            "modes": 0,
            "hq": 0.0,
            "mass_tv": 1.0,
            "recon_rmse": float("inf"),
            "mode_consistency": 0.0,
            "consistency": 0.0,
        }
        return {
            "passed": False,
            "failed_bounds": ["modes", "hq", "mass_tv", "recon_rmse", "mode_consistency", "nonfinite_output"],
            "metrics": metrics,
            "views": [],
        }

    dist, nearest = torch.cdist(samples, centers).min(dim=1)
    hq = (dist <= 3.0 * sigma).float().mean()
    counts = torch.bincount(nearest[dist <= 3.0 * sigma], minlength=16)
    modes = int((counts >= 10).sum())
    mass_tv = 0.5 * (counts.float() / counts.sum().clamp_min(1) - 1.0 / 16.0).abs().sum()

    recon_rmse = torch.sqrt(((target - reconstructed) ** 2).sum(dim=1).mean())
    target_modes = torch.cdist(target, centers).argmin(dim=1)
    rec_modes = torch.cdist(reconstructed, centers).argmin(dim=1)
    consistency = (target_modes == rec_modes).float().mean()

    metrics = {
        "modes": modes,
        "hq": float(hq),
        "mass_tv": float(mass_tv),
        "recon_rmse": float(recon_rmse),
        "mode_consistency": float(consistency),
        "consistency": float(consistency),
    }

    failed_bounds = []
    if metrics["modes"] < 16:
        failed_bounds.append("modes")
    if metrics["hq"] < 0.85:
        failed_bounds.append("hq")
    if metrics["mass_tv"] > 0.15:
        failed_bounds.append("mass_tv")
    if metrics["recon_rmse"] > 0.12:
        failed_bounds.append("recon_rmse")
    if metrics["mode_consistency"] < 0.98:
        failed_bounds.append("mode_consistency")

    passed = len(failed_bounds) == 0
    views = [
        dict(kind="scatter", title="Forward generation vs 16-mode ring", target=target.detach().cpu(), samples=samples.detach().cpu()),
        dict(kind="scatter", title="Paired round-trip reconstruction x -> G(E(x))", target=target.detach().cpu(), samples=reconstructed.detach().cpu()),
        dict(kind="bar", title="Mode mass distribution", target=torch.full((16,), 1.0 / 16.0), samples=counts.float().cpu() / counts.sum().clamp_min(1).cpu()),
    ]
    return dict(metrics=metrics, passed=passed, failed_bounds=sorted(set(failed_bounds)), views=views)


@torch.no_grad()
def _spiral_reference_grid(device, n_grid=1024, theta_min=0.5 * math.pi, theta_max=3.5 * math.pi, r0=0.5, r_slope=0.25):
    theta_grid = torch.linspace(theta_min, theta_max, n_grid + 1, device=device)[:-1] + 0.5 * (theta_max - theta_min) / n_grid
    r_grid = r0 + r_slope * theta_grid
    curve_grid = torch.stack([r_grid * torch.cos(theta_grid), r_grid * torch.sin(theta_grid)], dim=1).float()
    return theta_grid, curve_grid


@torch.no_grad()
def score_alice_spiral(samples, target, reconstructed, z=None, latent_reconstructed=None, sigma=0.06):
    tensors = [samples, target, reconstructed]
    if z is not None:
        tensors.append(z)
    if latent_reconstructed is not None:
        tensors.append(latent_reconstructed)

    if any(not torch.all(torch.isfinite(t)) for t in tensors):
        metrics = {
            "arc_bins": 0,
            "hq": 0.0,
            "mass_tv": 1.0,
            "recon_rmse": float("inf"),
            "latent_recon_rmse": float("inf"),
            "arc_consistency": 0.0,
        }
        return {
            "passed": False,
            "failed_bounds": [
                "arc_bins",
                "hq",
                "mass_tv",
                "recon_rmse",
                "latent_recon_rmse",
                "arc_consistency",
                "nonfinite_output",
            ],
            "metrics": metrics,
            "views": [],
        }

    n_grid = 1024
    _, curve_grid = _spiral_reference_grid(samples.device, n_grid=n_grid)
    pts_per_bin = n_grid // 16

    dist, nearest_idx = torch.cdist(samples.float(), curve_grid).min(dim=1)
    hq_mask = dist <= 3.0 * sigma
    hq = hq_mask.float().mean()
    sample_bins = nearest_idx // pts_per_bin
    counts = torch.bincount(sample_bins[hq_mask], minlength=16)
    arc_bins = int((counts >= 20).sum())
    mass_tv = 0.5 * (counts.float() / counts.sum().clamp_min(1) - 1.0 / 16.0).abs().sum()

    recon_rmse = torch.sqrt(F.mse_loss(reconstructed.float(), target.float()))
    if z is not None and latent_reconstructed is not None:
        latent_recon_rmse = float(torch.sqrt(F.mse_loss(latent_reconstructed.float(), z.float())))
    else:
        latent_recon_rmse = 0.0

    _, target_idx = torch.cdist(target.float(), curve_grid).min(dim=1)
    _, rec_idx = torch.cdist(reconstructed.float(), curve_grid).min(dim=1)
    target_bins = target_idx // pts_per_bin
    rec_bins = rec_idx // pts_per_bin
    arc_consistency = ((target_bins - rec_bins).abs() <= 1).float().mean()

    metrics = {
        "arc_bins": arc_bins,
        "hq": float(hq),
        "mass_tv": float(mass_tv),
        "recon_rmse": float(recon_rmse),
        "latent_recon_rmse": float(latent_recon_rmse),
        "arc_consistency": float(arc_consistency),
    }

    failed_bounds = []
    if metrics["arc_bins"] < 15:
        failed_bounds.append("arc_bins")
    if metrics["hq"] < 0.85:
        failed_bounds.append("hq")
    if metrics["mass_tv"] > 0.16:
        failed_bounds.append("mass_tv")
    if metrics["recon_rmse"] > 0.14:
        failed_bounds.append("recon_rmse")
    if metrics["latent_recon_rmse"] > 0.25:
        failed_bounds.append("latent_recon_rmse")
    if metrics["arc_consistency"] < 0.90:
        failed_bounds.append("arc_consistency")

    passed = len(failed_bounds) == 0
    views = [
        dict(kind="scatter", title="Forward generation vs Archimedes spiral", target=target.detach().cpu(), samples=samples.detach().cpu()),
        dict(kind="scatter", title="Paired ALICE cycle reconstruction x -> G(E(x))", target=target.detach().cpu(), samples=reconstructed.detach().cpu()),
        dict(kind="bar", title="Spiral arc-bin mass distribution", target=torch.full((16,), 1.0 / 16.0), samples=counts.float().cpu() / counts.sum().clamp_min(1).cpu()),
    ]
    return dict(metrics=metrics, passed=passed, failed_bounds=sorted(set(failed_bounds)), views=views)


class _RNGStreams:
    def __init__(self, seed: int, device: torch.device):
        self.seed = int(seed)
        self.device = torch.device(device)
        self._offsets = {
            "constructor": 100,
            "data": 101,
            "latent": 102,
            "eval": 103,
            "noise": 104,
            "model": 105,
            "constructor_cpu": 106,
        }
        self._streams = {}

    def fork(self, name: str) -> torch.Generator:
        if name not in self._offsets:
            raise ValueError(f"undeclared BiGAN RNG stream: {name}")
        if name not in self._streams:
            offset = self._offsets[name]
            device = "cpu" if name == "constructor_cpu" else self.device
            self._streams[name] = torch.Generator(device=device).manual_seed(self.seed + offset)
        return self._streams[name]


@contextmanager
def _owned_torch_rng(stream, device):
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        if device.type == "cuda":
            torch.cuda.set_rng_state(stream.get_state(), device)
        else:
            torch.set_rng_state(stream.get_state())
        try:
            yield
        finally:
            stream.set_state(torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state())


def _bigan_update(method):
    @wraps(method)
    def update(self):
        if not self._checkpoint_ready:
            raise RuntimeError("interrupted BiGAN update requires a complete checkpoint restore")
        if self.completed_steps >= self.limit:
            raise RuntimeError("declared execution cap reached")
        self._checkpoint_ready = False
        with _owned_torch_rng(self.model_rng, self.device), torch.autograd.set_multithreading_enabled(False):
            result = method(self)
        self._checkpoint_ready = True
        return result
    return update


class BiGANGenerator(nn.Module):
    def __init__(self, z_dim=2, out_dim=2, hidden=128, layers=3):
        super().__init__()
        mods = [nn.Linear(z_dim, hidden), nn.LeakyReLU(0.2)]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)])
        mods.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, z):
        return self.net(z)


class BiGANEncoder(nn.Module):
    def __init__(self, in_dim=2, z_dim=2, hidden=128, layers=3, fourier=0):
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.fourier = fourier
        if fourier > 0:
            freqs = torch.pi * (2.0 ** torch.arange(fourier, dtype=torch.float32))
            self.register_buffer("freqs", freqs)
            feat_dim = in_dim * (1 + 2 * fourier)
        else:
            feat_dim = in_dim
        mods = [nn.Linear(feat_dim, hidden), nn.LeakyReLU(0.2)]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)])
        mods.append(nn.Linear(hidden, z_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        if self.fourier > 0:
            xf = x.unsqueeze(-1) * self.freqs
            h = torch.cat([x, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
        else:
            h = x
        return self.net(h)


class BiGANJointCritic(nn.Module):
    def __init__(self, in_dim=2, z_dim=2, hidden=128, layers=3, fourier=0):
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.fourier = fourier
        if fourier > 0:
            freqs = torch.pi * (2.0 ** torch.arange(fourier, dtype=torch.float32))
            self.register_buffer("freqs", freqs)
            feat_dim = in_dim * (1 + 2 * fourier)
        else:
            feat_dim = in_dim
        mods = [nn.Linear(feat_dim + z_dim, hidden), nn.LeakyReLU(0.2)]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)])
        mods.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x_or_pair, z=None):
        if z is not None:
            x, z_code = x_or_pair, z
            true_coords = torch.cat([x, z_code], dim=1)
        else:
            x, z_code = x_or_pair[:, :self.in_dim], x_or_pair[:, self.in_dim:]
            true_coords = x_or_pair
        if self.fourier > 0:
            xf = x.unsqueeze(-1) * self.freqs
            hx = torch.cat([x, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
        else:
            hx = x
        pair = torch.cat([hx, z_code], dim=1)
        out = self.net(pair).squeeze(-1)
        out._bispf_input = true_coords
        return out


class Ring16BiGANFixture:
    api_components = (
        "particlegan.Recipe",
        "particlegan.Recipe.make_optimizers",
        "particlegan.Recipe.make_critic_penalty",
        "particlegan.MoGParticlePrior",
        "particlegan.GANLoss.joint_g_loss",
        "particlegan.init",
    )
    sigma = 0.10
    cycle_weight = 2.0
    default_recipe = "k3p"
    supported_recipes = ("k3p", "ka2")

    def __init__(
        self,
        case=None,
        *,
        recipe_name="k3p",
        recipe=None,
        prior=None,
        device="cpu",
        seed=0,
        steps=4000,
        max_steps=None,
        batch_size=128,
        particles=256,
        z_dim=2,
    ):
        if case is None:
            case = _bigan_case()
        self.metadata = deepcopy(case)
        self.case = self.metadata
        self.case_id = case["id"]
        self.device = torch.device(device)
        self.seed = int(seed)
        self.completed_steps = 0
        default_steps = case.get("default_steps", steps)
        if max_steps is not None:
            if type(max_steps) is not int or not 0 < max_steps <= default_steps:
                raise ValueError("max_steps must be in the declared budget")
            self.limit = max_steps
        else:
            if type(steps) is not int or steps <= 0:
                raise ValueError("steps must be positive")
            self.limit = steps
        self.steps = self.limit
        self.batch_size = int(case.get("batch_size", batch_size) if batch_size == 128 else batch_size)
        self.particles = int(particles)
        self.z_dim = int(z_dim)

        self._rng_streams = _RNGStreams(self.seed, self.device)
        self.data_rng = self._rng_streams.fork("data")
        self.latent_rng = self._rng_streams.fork("latent")
        self.eval_rng = self._rng_streams.fork("eval")
        self.noise_rng = self._rng_streams.fork("noise")
        self.model_rng = self._rng_streams.fork("model")
        self._rng_streams.fork("constructor")

        angles = torch.linspace(0, 2 * math.pi, 17, device=self.device)[:16]
        self.centers = torch.stack([3.0 * torch.cos(angles), 3.0 * torch.sin(angles)], dim=1).float()

        self._build_models(fourier=3)

        init.deterministic_orthogonal_(self.G, seed=self.seed + 1)
        init.deterministic_orthogonal_(self.E, seed=self.seed + 2)
        init.deterministic_orthogonal_(self.D, seed=self.seed + 3)

        self.prior = None
        self.rebind_trainer(recipe if recipe is not None else (recipe_name or "k3p"), prior=prior)
        self.last_losses = {}
        self._checkpoint_ready = True

    def _build_models(self, *, fourier):
        # Linear constructors consume CPU randomness even for a CUDA fixture.
        # Keep those discarded initial weights off the caller's global stream.
        stream = self._rng_streams.fork("constructor_cpu")
        with _owned_torch_rng(stream, torch.device("cpu")):
            self.G = BiGANGenerator(z_dim=self.z_dim, out_dim=2, hidden=128, layers=3).to(self.device)
            self.E = BiGANEncoder(in_dim=2, z_dim=self.z_dim, hidden=128, layers=3, fourier=fourier).to(self.device)
            self.D = BiGANJointCritic(in_dim=2, z_dim=self.z_dim, hidden=128, layers=3, fourier=fourier).to(self.device)

    def _resolve_recipe(self, recipe_or_name):
        if isinstance(recipe_or_name, Recipe):
            return recipe_or_name
        name = recipe_or_name or "k3p"
        base = get_recipe(name)
        overrides = dict(
            batch_size=self.batch_size,
            z_dim=self.z_dim,
            num_particles=self.particles,
            sigma_rel=0.0,
            standardize=False,
        )
        if base.continuous_policy is None:
            overrides["total_steps"] = self.limit
            if not base.particle_birth_death:
                overrides["prior_kind"] = "mog"
        return base.replace(**overrides)

    def rebind_trainer(self, recipe, prior=None):
        if self.completed_steps:
            raise ValueError("BiGAN rebinding requires a fresh untrained fixture")
        resolved = self._resolve_recipe(recipe)
        self._validate_supported_recipe(resolved)
        self.recipe = resolved
        if prior is not None:
            self.prior = prior.to(self.device)
            self.recipe.apply_prior_policy(self.prior)
        elif getattr(self, "prior", None) is None:
            self.prior = self.recipe.replace(
                prior_update="learned",
                num_particles=self.particles,
                z_dim=self.z_dim,
            ).make_prior(
                sigma=self.sigma,
                sigma_rel=0.0,
                device=self.device,
                generator=self._rng_streams.fork("constructor"),
            )
            init.deterministic_orthogonal_(unwrap_prior(self.prior), seed=self.seed + 4)
            self.recipe.apply_prior_policy(self.prior)
        else:
            self.recipe.apply_prior_policy(self.prior)

        self._prior_configuration()

        self.ema_critic = deepcopy(self.D) if self.recipe.effective_critic_formulation != "bispf" else None
        self.opt_g, self.opt_d = self.recipe.make_optimizers(
            self.G,
            self.D,
            self.prior,
            encoder=self.E,
            ema_critic=self.ema_critic,
            foreach=False,
            fused=False,
        )
        self.base_rates = [[group["lr"] for group in opt.param_groups] for opt in (self.opt_g, self.opt_d)]
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)
        self.regularizer = self.recipe.make_prior_regularizer(prior=self.prior)
        self.loss = self.recipe.make_loss()
        if self.recipe.effective_critic_formulation == "bispf":
            from particlegan.bispf import bind_thermal_source
            bind_thermal_source(self.opt_d.record, self.loss, self.regularizer)

    def _validate_supported_recipe(self, recipe):
        unsupported = [name for name in (
            "continuous_policy", "row_evidence_gate", "particle_birth_death", "serve_average",
            "input_noise_std", "output_noise_std", "kinetic_transport_weight", "kinetic_transport_local_weight",
            "reopen_guard"
        ) if getattr(recipe, name)]
        if recipe.lr_control == "stationarity":
            unsupported.append("lr_control=stationarity")
        if recipe.output_noise_mode != "fixed":
            unsupported.append("output_noise_mode")
        if recipe.generator_gradient_response != "none":
            unsupported.append("generator_gradient_response")
        if recipe.row_policy != "independent" or recipe.prior_code_law != "table":
            unsupported.append("prior sampling policy")
        for name, value in (("batch_size", self.batch_size), ("num_particles", self.particles), ("z_dim", self.z_dim)):
            if getattr(recipe, name) != value:
                unsupported.append(f"{name} differs from task resources")
        if recipe.prior_kind != "mog" or recipe.sigma_rel != 0. or recipe.standardize:
            unsupported.append("prior law differs from task MoG")
        if unsupported:
            from .api_contract import UnsupportedRecipeOverrides
            raise UnsupportedRecipeOverrides("BiGAN host cannot execute the intact recipe mechanisms: "
                + ", ".join(unsupported) + "; declare a separate supported software cohort")

    def _prior_configuration(self):
        from .api_contract import UnsupportedRecipeOverrides
        prior = self.prior
        if (type(prior) is not MoGParticlePrior or prior.z.shape != (self.particles, self.z_dim)
                or float(prior.sigma) != float(torch.tensor(self.sigma, dtype=prior.sigma.dtype))
                or prior.sigma_rel != 0. or prior.standardize or prior.code_law != "table"):
            raise UnsupportedRecipeOverrides("BiGAN prior must match the declared unwrapped fixed-width MoG task law")
        return {"kind": "mog", "sigma": float(prior.sigma), "sigma_rel": prior.sigma_rel,
                "standardize": prior.standardize, "code_law": prior.code_law,
                "particles": self.particles, "z_dim": self.z_dim}

    def _sample_prior(self, n, generator=None):
        rng = self.latent_rng if generator is None else generator
        if isinstance(self.prior, (MoGParticlePrior, ManipulatedPrior)):
            noise_rng = self.noise_rng if generator is None else generator
            return self.prior.sample(n, generator=rng, noise_generator=noise_rng)
        return self.prior.sample(n, generator=rng)

    def _generate(self, z_fake):
        x_fake = self.G(z_fake)
        if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
            x_fake = self.prior.postprocess_generated(x_fake)
        return x_fake

    def sample_target(self, n, generator=None):
        rng = self.data_rng if generator is None else generator
        mode_ids = torch.randint(0, 16, (n,), generator=rng, device=self.device)
        eps = torch.randn(n, 2, generator=rng, device=self.device)
        return self.centers[mode_ids] + self.sigma * eps

    @_bigan_update
    def step(self):
        if self.completed_steps >= self.limit:
            raise RuntimeError("declared execution cap reached")

        if self.recipe.continuous_policy is None and getattr(self.recipe, "lr_control", "mobility") != "stationarity":
            scale_learning_rates(self.completed_steps, self.recipe, [self.opt_g, self.opt_d], self.base_rates, self.prior)
        real_x = self.sample_target(self.batch_size)
        self.G.train(); self.E.train(); self.D.train()

        with torch.no_grad():
            z_fake, _ = self._sample_prior(len(real_x))
            x_fake = self._generate(z_fake)
            z_real = self.E(real_x)
            pair_real = torch.cat([real_x, z_real], dim=1)
            pair_fake = torch.cat([x_fake, z_fake], dim=1)

        self.opt_d.zero_grad()
        loss_d = self.loss.d_loss(self.D(pair_real), self.D(pair_fake))
        reg_d = self.penalty(self.D, pair_real, pair_fake)
        (loss_d + reg_d).backward()
        self.opt_d.step()

        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            z_fake, idx_fake = self._sample_prior(len(real_x))
            x_fake = self._generate(z_fake)
            z_real = self.E(real_x)
            pair_fake = torch.cat([x_fake, z_fake], dim=1)
            pair_real = torch.cat([real_x, z_real], dim=1)

            loss_joint = self.loss.joint_g_loss(self.D(pair_fake), self.D(pair_real))
            loss_recon = F.mse_loss(self.G(z_real), real_x) + F.mse_loss(self.E(x_fake), z_fake.detach())
            reg_prior = self.recipe.prior_regularization(
                unwrap_prior(self.prior).z, regularizer=self.regularizer, prior=self.prior
            )
            if isinstance(self.prior, ManipulatedPrior) and self.prior.has_regularizer:
                reg_prior = reg_prior + self.prior.regularize(
                    unwrap_prior(self.prior).z, sampled_codes=z_fake, indices=idx_fake, generated=x_fake
                )
            loss_g = loss_joint + self.cycle_weight * loss_recon + reg_prior

            self.opt_g.zero_grad()
            if hasattr(self.opt_g, "set_sampled_rows") and unwrap_prior(self.prior).z.requires_grad:
                self.opt_g.set_sampled_rows(unwrap_prior(self.prior).z, idx_fake)
            if self.recipe.constraint_geometry_mode != "none":
                constraint_geometry_backward(loss_g, self.opt_g, (loss_joint,))
            else:
                loss_g.backward()
            self.opt_g.step()
        finally:
            for p, f in zip(self.D.parameters(), flags):
                p.requires_grad_(f)

        self.completed_steps += 1
        self.last_losses = {
            "loss_d": float(loss_d.detach()),
            "loss_g": float(loss_g.detach()),
            "reg_d": float(reg_d.detach()),
        }
        return dict(step=self.completed_steps, **self.last_losses)

    @torch.no_grad()
    def observe(self, n=4096, seed=10000):
        self.G.eval(); self.E.eval(); self.D.eval()
        eval_gen = torch.Generator(device=self.device).manual_seed(int(seed))
        target_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        z, _ = self._sample_prior(n, generator=eval_gen)
        samples = self._generate(z)
        target = self.sample_target(n, generator=target_gen)
        reconstructed = self.G(self.E(target))
        return score_bigan(samples, target, reconstructed, self.centers, sigma=self.sigma)

    def state_dict(self):
        if not self._checkpoint_ready:
            raise RuntimeError("BiGAN checkpoints require a completed update boundary")
        return self._checkpoint_state()

    def _checkpoint_modules(self):
        modules = {"generator": self.G, "encoder": self.E, "critic": self.D,
                   "ema_critic": self.ema_critic, "prior": self.prior, "regularizer": self.regularizer}
        return {name: module for name, module in modules.items() if module is not None}

    def _checkpoint_state(self):
        modules = self._checkpoint_modules()
        cooler = getattr(self.penalty.regularizer, "cooler", None)
        return deepcopy({
            "version": "bigan-complete-v1",
            "case_id": self.case_id,
            "case": self.metadata,
            "seed": self.seed,
            "limit": self.limit,
            "device": str(self.device),
            "host": {"batch_size": self.batch_size, "particles": self.particles, "z_dim": self.z_dim,
                     "sigma": self.sigma, "cycle_weight": self.cycle_weight,
                     "target": {key: getattr(self, key) for key in
                                ("theta_min", "theta_max", "r0", "r_slope") if hasattr(self, key)}},
            "prior_configuration": self._prior_configuration(),
            "target_state": {name: getattr(self, name) for name in ("centers", "theta_grid", "curve_grid")
                             if hasattr(self, name)},
            "completed_steps": self.completed_steps,
            "recipe": self.recipe.to_dict(),
            "generator": self.G.state_dict(),
            "encoder": self.E.state_dict(),
            "critic": self.D.state_dict(),
            "ema_critic": self.ema_critic.state_dict() if self.ema_critic is not None else None,
            "prior": self.prior.state_dict(),
            "regularizer": self.regularizer.state_dict() if self.regularizer is not None else None,
            "prior_regularizer": self.recipe.prior_regularizer_checkpoint(self.regularizer).get("prior_regularizer"),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "penalty": {"record": self.opt_d.record.state_dict(),
                        "cooler": None if cooler is None else cooler.state_dict(),
                        "coeff": self.penalty.regularizer.coeff, "last_stats": self.penalty.last_stats},
            "streams": {name: stream.get_state() for name, stream in self._rng_streams._streams.items()},
            "stream_offsets": self._rng_streams._offsets,
            "base_rates": self.base_rates,
            "module_modes": {name: {key: sub.training for key, sub in module.named_modules()}
                             for name, module in modules.items()},
            "requires_grad": {name: {key: p.requires_grad for key, p in module.named_parameters()}
                              for name, module in modules.items()},
            "last_losses": self.last_losses,
        })

    def load_state_dict(self, state):
        from experiments.forge.state import state_digest
        from particlegan.policy import _validate_optimizer_state
        state = deepcopy(state)  # Native optimizer loaders can borrow input tensors.
        expected = self._checkpoint_state()
        if not isinstance(state, dict) or state.get("version") != "bigan-complete-v1":
            raise ValueError("incomplete historical BiGAN checkpoint; use its original package source")
        if state.keys() != expected.keys():
            raise ValueError("incomplete BiGAN checkpoint fields")
        for key in ("case_id", "case", "seed", "limit", "device", "host", "prior_configuration",
                    "recipe", "requires_grad", "base_rates", "stream_offsets"):
            if state[key] != expected[key]:
                raise ValueError(f"BiGAN checkpoint {key} differs from this declared host")
        if state_digest(state["target_state"]) != state_digest(expected["target_state"]):
            raise ValueError("BiGAN checkpoint target law differs from this declared host")
        steps = state["completed_steps"]
        if type(steps) is not int or not 0 <= steps <= self.limit:
            raise ValueError("invalid BiGAN checkpoint update count")
        modules = self._checkpoint_modules()
        try:
            for name, module in modules.items():
                saved, current = state[name], module.state_dict()
                if not isinstance(saved, dict) or saved.keys() != current.keys():
                    raise ValueError("model fields differ")
                for key, value in current.items():
                    other = saved[key]
                    if isinstance(value, torch.Tensor):
                        if (not isinstance(other, torch.Tensor) or other.shape != value.shape
                                or other.dtype != value.dtype or not torch.isfinite(other).all()):
                            raise ValueError("invalid model tensor")
                    elif other != value:
                        raise ValueError("model declaration differs")
                deepcopy(module).load_state_dict(saved)
            if state["ema_critic"] is None and self.ema_critic is not None:
                raise ValueError("missing EMA critic history")
            if self.ema_critic is None and state["ema_critic"] is not None:
                raise ValueError("unexpected EMA critic history")
            if self.regularizer is None and state["regularizer"] is not None:
                raise ValueError("unexpected regularizer state")
            self.recipe.restore_prior_regularizer(self.regularizer, state["prior_regularizer"], validate_only=True)
            public_reg = self.recipe.prior_regularizer_checkpoint(self.regularizer)
            if public_reg and state_digest(state["prior_regularizer"]) != state_digest(state["regularizer"]):
                raise ValueError("conflicting regularizer histories")
            for name, optimizer in (("opt_g", self.opt_g), ("opt_d", self.opt_d)):
                _validate_optimizer_state(optimizer, state[name])
                pending = [state[name]]
                while pending:
                    value = pending.pop()
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        raise ValueError("nonfinite optimizer-owned history")
                    if isinstance(value, dict):
                        pending.extend(value.values())
                    elif isinstance(value, (list, tuple)):
                        pending.extend(value)
                parameter_ids = {key for group in state[name]["param_groups"] for key in group["params"]}
                if steps and state[name]["state"].keys() != parameter_ids:
                    raise ValueError("incomplete trained optimizer history")
                for values in state[name]["state"].values():
                    if steps and float(values["step"]) != steps:
                        raise ValueError("optimizer clock differs from completed update count")
            penalty = state["penalty"]
            if not isinstance(penalty, dict) or penalty.keys() != expected["penalty"].keys():
                raise ValueError("incomplete penalty state")
            if (penalty["record"] != state["opt_d"].get("regularizer", {}).get("record")
                    or type(penalty["record"].get("observed_steps")) is not int
                    or penalty["record"].get("observed_steps") != steps
                    or type(penalty["record"].get("calls")) is not int or penalty["record"]["calls"] < 0):
                raise ValueError("conflicting critic histories")
            if (getattr(self.opt_d, "ema_critic", None) is not None
                    and state_digest(state["ema_critic"]) != state_digest(state["opt_d"]["regularizer"]["ema"])):
                raise ValueError("conflicting EMA critic histories")
            cooler = getattr(self.penalty.regularizer, "cooler", None)
            if cooler is None:
                if penalty["cooler"] is not None:
                    raise ValueError("unexpected cooler history")
            else:
                saved, current = penalty["cooler"], cooler.state_dict()
                if not isinstance(saved, dict) or saved.keys() != current.keys():
                    raise ValueError("incomplete cooler history")
                for key in current.keys() - {"initialized", "ema_residual"}:
                    if saved[key] != current[key]:
                        raise ValueError("cooler declaration differs")
                if (type(saved["initialized"]) is not bool or type(saved["ema_residual"]) not in (int, float)
                        or not math.isfinite(saved["ema_residual"]) or saved["ema_residual"] < 0):
                    raise ValueError("invalid cooler history")
                deepcopy(cooler).load_state_dict(saved)
            if type(penalty["coeff"]) not in (int, float) or not math.isfinite(penalty["coeff"]) or penalty["coeff"] < 0:
                raise ValueError("invalid penalty coefficient")
            if not isinstance(penalty["last_stats"], dict):
                raise TypeError("invalid penalty diagnostics")
            if not isinstance(state["streams"], dict) or state["streams"].keys() != expected["streams"].keys():
                raise ValueError("incomplete RNG state")
            for name, values in state["streams"].items():
                torch.Generator(device=self._rng_streams._streams[name].device).set_state(values.cpu())
            if state["module_modes"].keys() != expected["module_modes"].keys():
                raise ValueError("incomplete module modes")
            for name, modes in state["module_modes"].items():
                if modes.keys() != expected["module_modes"][name].keys() or any(type(v) is not bool for v in modes.values()):
                    raise ValueError("invalid module modes")
            if (not isinstance(state["last_losses"], dict) or any(type(v) not in (int, float) or not math.isfinite(v)
                                                               for v in state["last_losses"].values())):
                raise ValueError("invalid loss diagnostics")
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid complete BiGAN checkpoint state") from error
        # All native loaders were preflighted on independent objects. Preserve
        # caller-visible model/prior identities while restoring the live graph.
        self.G.load_state_dict(state["generator"])
        self.E.load_state_dict(state["encoder"])
        self.D.load_state_dict(state["critic"])
        if self.ema_critic is not None and state.get("ema_critic") is not None:
            self.ema_critic.load_state_dict(state["ema_critic"])
        self.prior.load_state_dict(state["prior"])
        if self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])
        self.recipe.restore_prior_regularizer(self.regularizer, state["prior_regularizer"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        if state["penalty"]["cooler"] is not None:
            self.penalty.regularizer.cooler.load_state_dict(state["penalty"]["cooler"])
        self.penalty.regularizer.coeff = state["penalty"]["coeff"]
        self.penalty.last_stats = deepcopy(state["penalty"]["last_stats"])
        for name, values in state["streams"].items():
            self._rng_streams._streams[name].set_state(values.cpu())
        for name, module in modules.items():
            for key, sub in module.named_modules():
                sub.training = state["module_modes"][name][key]
        self.completed_steps = steps
        self.last_losses = deepcopy(state["last_losses"])
        self._checkpoint_ready = True


class ALICESpiralBiGANFixture(Ring16BiGANFixture):
    sigma = 0.06
    cycle_weight = 2.0
    default_recipe = "k3p"
    supported_recipes = ("k3p", "ka2")
    theta_min = 0.5 * math.pi
    theta_max = 3.5 * math.pi
    r0 = 0.5
    r_slope = 0.25

    def __init__(
        self,
        case=None,
        *,
        recipe_name="k3p",
        recipe=None,
        prior=None,
        device="cpu",
        seed=0,
        steps=4000,
        max_steps=None,
        batch_size=256,
        particles=256,
        z_dim=2,
    ):
        if case is None:
            case = _alice_spiral_case()
        self.metadata = deepcopy(case)
        self.case = self.metadata
        self.case_id = case["id"]
        self.device = torch.device(device)
        self.seed = int(seed)
        self.completed_steps = 0
        default_steps = case.get("default_steps", steps)
        if max_steps is not None:
            if type(max_steps) is not int or not 0 < max_steps <= default_steps:
                raise ValueError("max_steps must be in the declared budget")
            self.limit = max_steps
        else:
            if type(steps) is not int or steps <= 0:
                raise ValueError("steps must be positive")
            self.limit = steps
        self.steps = self.limit
        self.batch_size = int(case.get("batch_size", batch_size) if batch_size == 256 else batch_size)
        self.particles = int(particles)
        self.z_dim = int(z_dim)

        self._rng_streams = _RNGStreams(self.seed, self.device)
        self.data_rng = self._rng_streams.fork("data")
        self.latent_rng = self._rng_streams.fork("latent")
        self.eval_rng = self._rng_streams.fork("eval")
        self.noise_rng = self._rng_streams.fork("noise")
        self.model_rng = self._rng_streams.fork("model")
        self._rng_streams.fork("constructor")

        self.theta_grid, self.curve_grid = _spiral_reference_grid(
            self.device, n_grid=1024, theta_min=self.theta_min, theta_max=self.theta_max, r0=self.r0, r_slope=self.r_slope
        )

        self._build_models(fourier=0)

        init.deterministic_orthogonal_(self.G, seed=self.seed + 1)
        init.deterministic_orthogonal_(self.E, seed=self.seed + 2)
        init.deterministic_orthogonal_(self.D, seed=self.seed + 3)

        self.prior = None
        self.rebind_trainer(recipe if recipe is not None else (recipe_name or "k3p"), prior=prior)
        self.last_losses = {}
        self._checkpoint_ready = True

    def sample_target(self, n, generator=None):
        rng = self.data_rng if generator is None else generator
        theta = self.theta_min + (self.theta_max - self.theta_min) * torch.rand(n, generator=rng, device=self.device)
        r = self.r0 + self.r_slope * theta
        eps = torch.randn(n, 2, generator=rng, device=self.device) * self.sigma
        return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1).float() + eps

    @torch.no_grad()
    def observe(self, n=4096, seed=10000):
        self.G.eval(); self.E.eval(); self.D.eval()
        eval_gen = torch.Generator(device=self.device).manual_seed(int(seed))
        target_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        z, _ = self._sample_prior(n, generator=eval_gen)
        samples = self._generate(z)
        target = self.sample_target(n, generator=target_gen)
        reconstructed = self.G(self.E(target))
        latent_reconstructed = self.E(samples)
        return score_alice_spiral(
            samples, target, reconstructed, z=z, latent_reconstructed=latent_reconstructed, sigma=self.sigma
        )


def build_case(id, *, device="cpu", seed=0, recipe_name=None, max_steps=None):
    if id == BIGAN_CASE_ID:
        return Ring16BiGANFixture(_bigan_case(), device=device, seed=seed, recipe_name=recipe_name or "k3p", max_steps=max_steps)
    if id == ALICE_SPIRAL_CASE_ID:
        return ALICESpiralBiGANFixture(_alice_spiral_case(), device=device, seed=seed, recipe_name=recipe_name or "k3p", max_steps=max_steps)
    if id == CURRENT_CASE_ID:
        return RingSmokeFixture(list_cases()[0], device=device, seed=seed, recipe_name=recipe_name, max_steps=max_steps)
    if id != CASE_ID:
        raise ValueError(f"unknown ring16 case: {id}")
    return VectorFixture(_legacy_cases()[0], device=device, seed=seed,
                         recipe_name=recipe_name or "k3p", max_steps=max_steps)

