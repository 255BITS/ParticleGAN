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

import torch.nn as nn
import torch.nn.functional as F

from particlegan import Recipe, get_recipe, init, MoGParticlePrior, scale_learning_rates
from .api_vectors import VectorFixture, _case, _digest, _law, _view, score_case
from . import ring16_quality  # Bind the declared scorer before run source capture.


TASK_PATH = Path(__file__).resolve().parents[2] / "configs/forge/tasks/ring16_acquisition.json"
BIGAN_TASK_PATH = Path(__file__).resolve().parents[2] / "configs/forge/tasks/ring16_bigan_continuous.json"
CASE_ID = "api-ring16-acquisition"
CURRENT_CASE_ID = "api-ring16-acquisition-v2"
BIGAN_CASE_ID = "api-ring16-bigan-continuous"
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
        "default_recipe": "bcap",
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
    return [current, *_legacy_cases(), _bigan_case()]


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


class BiGANGenerator(nn.Module):
    def __init__(self, z_dim=2, out_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class BiGANEncoder(nn.Module):
    def __init__(self, in_dim=2, z_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, x):
        return self.net(x)


class BiGANJointCritic(nn.Module):
    def __init__(self, in_dim=2, z_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + z_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_or_pair, z=None):
        pair = torch.cat([x_or_pair, z], dim=1) if z is not None else x_or_pair
        return self.net(pair).squeeze(-1)


class Ring16BiGANFixture:
    api_components = (
        "particlegan.Recipe",
        "particlegan.GANTrainer",
        "particlegan.MoGParticlePrior",
        "particlegan.GANLoss.joint_g_loss",
        "particlegan.init",
    )

    def __init__(self, case, *, device="cpu", seed=0, recipe_name=None, max_steps=None):
        self.metadata = deepcopy(case)
        self.case = self.metadata
        self.case_id = case["id"]
        self.device = torch.device(device)
        self.seed = int(seed)
        self.completed_steps = 0
        self.limit = case["default_steps"] if max_steps is None else max_steps
        if type(self.limit) is not int or not 0 < self.limit <= case["default_steps"]:
            raise ValueError("max_steps must be in the declared budget")

        self.recipe = get_recipe(
            recipe_name or "bcap",
            batch_size=case.get("batch_size", 128),
            total_steps=case["default_steps"],
            z_dim=2,
            num_particles=256,
            prior_kind="mog",
            sigma_rel=0.1,
            standardize=False,
            lr=0.002,
        )

        angles = torch.linspace(0, 2 * math.pi, 17, device=self.device)[:16]
        self.centers = torch.stack([3.0 * torch.cos(angles), 3.0 * torch.sin(angles)], dim=1).float()

        self.G = BiGANGenerator(z_dim=2, out_dim=2, hidden=64).to(self.device)
        self.E = BiGANEncoder(in_dim=2, z_dim=2, hidden=64).to(self.device)
        self.D = BiGANJointCritic(in_dim=2, z_dim=2, hidden=128).to(self.device)
        self.prior = MoGParticlePrior(num_particles=256, z_dim=2, sigma=0.1, standardize=False).to(self.device)

        init.deterministic_orthogonal_(self.G, seed=self.seed + 1)
        init.deterministic_orthogonal_(self.E, seed=self.seed + 2)
        init.deterministic_orthogonal_(self.D, seed=self.seed + 3)
        init.deterministic_orthogonal_(self.prior, seed=self.seed + 4)

        groups = [
            dict(params=list(self.G.parameters()), lr=self.recipe.lr),
            dict(params=list(self.E.parameters()), lr=self.recipe.lr),
        ]
        self.opt_g = self.recipe.make_generator_optimizer(groups)
        self.opt_d = self.recipe.make_critic_optimizer(self.D, ema_critic=deepcopy(self.D))
        self.base_rates = [[group["lr"] for group in opt.param_groups] for opt in (self.opt_g, self.opt_d)]
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)

        self.data_rng = torch.Generator(device=self.device).manual_seed(self.seed + 101)
        self.latent_rng = torch.Generator(device=self.device).manual_seed(self.seed + 102)
        self.eval_rng = torch.Generator(device=self.device).manual_seed(self.seed + 103)
        self.last_losses = {}

    def sample_target(self, n, generator=None):
        rng = self.data_rng if generator is None else generator
        mode_ids = torch.randint(0, 16, (n,), generator=rng, device=self.device)
        eps = torch.randn(n, 2, generator=rng, device=self.device)
        return self.centers[mode_ids] + 0.10 * eps

    def step(self):
        if self.completed_steps >= self.limit:
            raise RuntimeError("declared execution cap reached")

        scale_learning_rates(self.completed_steps, self.recipe, [self.opt_g, self.opt_d], self.base_rates, self.prior)
        real_x = self.sample_target(self.recipe.batch_size)
        self.G.train(); self.E.train(); self.D.train()

        with torch.no_grad():
            z_fake, _ = self.prior.sample(len(real_x), generator=self.latent_rng)
            x_fake = self.G(z_fake)
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
            z_fake, _ = self.prior.sample(len(real_x), generator=self.latent_rng)
            x_fake = self.G(z_fake)
            z_real = self.E(real_x)
            pair_fake = torch.cat([x_fake, z_fake], dim=1)
            pair_real = torch.cat([real_x, z_real], dim=1)

            loss_joint = self.loss.joint_g_loss(self.D(pair_fake), self.D(pair_real))
            loss_recon = F.mse_loss(self.G(z_real), real_x) + F.mse_loss(self.E(x_fake), z_fake)
            loss_g = loss_joint + 0.5 * loss_recon

            self.opt_g.zero_grad()
            if self.recipe.constraint_geometry_mode != "none":
                from particlegan.optim.constraint_geometry import constraint_geometry_backward
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
        z, _ = self.prior.sample(n, generator=eval_gen)
        samples = self.G(z)
        target = self.sample_target(n, generator=target_gen)
        reconstructed = self.G(self.E(target))
        return score_bigan(samples, target, reconstructed, self.centers, sigma=0.10)

    def state_dict(self):
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "limit": self.limit,
            "completed_steps": self.completed_steps,
            "recipe": self.recipe.to_dict(),
            "generator": self.G.state_dict(),
            "encoder": self.E.state_dict(),
            "critic": self.D.state_dict(),
            "prior": self.prior.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "last_losses": self.last_losses,
        }

    def load_state_dict(self, state):
        self.completed_steps = state["completed_steps"]
        self.G.load_state_dict(state["generator"])
        self.E.load_state_dict(state["encoder"])
        self.D.load_state_dict(state["critic"])
        self.prior.load_state_dict(state["prior"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.last_losses = state["last_losses"]


def build_case(id, *, device="cpu", seed=0, recipe_name=None, max_steps=None):
    if id == BIGAN_CASE_ID:
        return Ring16BiGANFixture(list_cases()[-1], device=device, seed=seed, recipe_name=recipe_name, max_steps=max_steps)
    if id == CURRENT_CASE_ID:
        return RingSmokeFixture(list_cases()[0], device=device, seed=seed, recipe_name=recipe_name, max_steps=max_steps)
    if id != CASE_ID:
        raise ValueError(f"unknown ring16 case: {id}")
    return VectorFixture(_legacy_cases()[0], device=device, seed=seed,
                         recipe_name=recipe_name or "k3p", max_steps=max_steps)

