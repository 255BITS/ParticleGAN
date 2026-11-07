"""Public-API toy audit cases for PacGAN m-packing (unequal mode mass & 100-mode grid)."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from particlegan import MoGParticlePrior, Recipe, get_recipe, init, scale_learning_rates
from particlegan.bispf import _attach_critic_input_hook, bind_thermal_source
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.priors import ManipulatedPrior, unwrap_prior


ROOT = Path(__file__).resolve().parents[2]
UNEQUAL_TASK_PATH = ROOT / "configs" / "forge" / "tasks" / "pacgan_unequal_mass.json"
GRID100_TASK_PATH = ROOT / "configs" / "forge" / "tasks" / "pacgan8_grid100.json"

UNEQUAL_CASE_ID = "api-pacgan-unequal-mass"
GRID100_CASE_ID = "api-pacgan8-grid100"

UNEQUAL_WEIGHTS: tuple[float, ...] = (0.40, 0.30, 0.20, 0.08, 0.02)
GRID100_COORDS: tuple[float, ...] = (-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0)


def _ring_centers(num_modes: int = 5, radius: float = 2.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, num_modes, endpoint=False, dtype=np.float64)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1).astype(np.float32)


def _grid100_centers(coords: tuple[float, ...] = GRID100_COORDS) -> np.ndarray:
    return np.array([[x, y] for y in coords for x in coords], dtype=np.float32)


def _extract_samples_array(samples: Any) -> np.ndarray:
    if isinstance(samples, dict):
        samples = samples.get("x_gen", samples.get("samples"))
    elif hasattr(samples, "payload") and isinstance(samples.payload, dict):
        samples = samples.payload.get("x_gen", samples.payload.get("samples"))
    if isinstance(samples, torch.Tensor):
        return samples.detach().cpu().numpy().astype(np.float64)
    return np.asarray(samples, dtype=np.float64)


class _RNGStreams:
    """Isolated named RNG streams for constructor, data, training_noise, and evaluation."""

    def __init__(self, seed: int, device: torch.device | str = "cpu") -> None:
        self.seed = int(seed)
        self.device = torch.device(device)
        self._streams: dict[str, torch.Generator] = {}
        for idx, name in enumerate(("constructor", "data", "training_noise", "evaluation")):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.seed + 1009 * (idx + 1))
            self._streams[name] = gen

    def fork(self, name: str) -> torch.Generator:
        gen = self._streams[name]
        child_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen, device=gen.device).item())
        child = torch.Generator(device=self.device)
        child.manual_seed(child_seed)
        return child

    def fork_seed(self, name: str) -> int:
        gen = self._streams[name]
        return int(torch.randint(0, 2**31 - 1, (1,), generator=gen, device=gen.device).item())

    def stream(self, name: str) -> torch.Generator:
        return self._streams[name]

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "streams": {k: v.get_state() for k, v in self._streams.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.seed = int(state["seed"])
        for k, v in state["streams"].items():
            self._streams[k].set_state(v)


def _sample_stratified_iso_gaussian(
    n: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample N(0, I_2) with polar Box-Muller stratification in blocks of 8 before permutation."""
    gen_dev = generator.device
    m = (n // 8) * 8
    parts: list[torch.Tensor] = []
    if m > 0:
        k = torch.arange(8, dtype=torch.float32, device=gen_dev).unsqueeze(0).expand(m // 8, 8)
        u_theta = (k + torch.rand(m // 8, 8, generator=generator, device=gen_dev)) / 8.0
        theta = 2.0 * math.pi * u_theta.reshape(-1)
        u_r = (k + torch.rand(m // 8, 8, generator=generator, device=gen_dev)) / 8.0
        perm_r = torch.argsort(torch.rand(m // 8, 8, generator=generator, device=gen_dev), dim=1)
        u_r = torch.gather(u_r, 1, perm_r).reshape(-1).clamp(min=1e-7, max=1.0 - 1e-7)
        r = torch.sqrt(-2.0 * torch.log(u_r))
        parts.append(torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1))
    if n > m:
        parts.append(torch.randn(n - m, 2, generator=generator, device=gen_dev))
    return torch.cat(parts, dim=0).to(device)


class PacGANGenerator(nn.Module):
    """3-layer 128-wide LeakyReLU MLP generator G: R^2 -> R^2."""

    def __init__(self, z_dim: int = 2, hidden_dim: int = 128, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PacGAN4RepulsionDiscriminator(nn.Module):
    """Permutation-invariant dispersion & repulsion augmented PacGAN-4 discriminator.

    Extracts:
      - Raw flattened coordinates (in_pts * pt_dim)
      - Sorted pairwise Euclidean distances (in_pts * (in_pts - 1) // 2)
      - Dispersion statistics: min distance, mean distance, variance of distance (3)
      - Centroid radial dispersion: sorted distances from centroid (in_pts) + radial variance (1)
      - Harmonic repulsion potential: mean(1.0 / (d_ij + epsilon)) (1)
    Total feature dim = 8 + 6 + 3 + 4 + 1 + 1 = 23 for in_pts=4, pt_dim=2.
    """

    def __init__(self, in_pts: int = 4, pt_dim: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.in_pts = int(in_pts)
        self.pt_dim = int(pt_dim)
        num_pairs = self.in_pts * (self.in_pts - 1) // 2
        self.feature_dim = self.in_pts * self.pt_dim + num_pairs + 3 + self.in_pts + 1 + 1
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def extract_features(self, x_pack: torch.Tensor) -> torch.Tensor:
        # Accepts x_pack of shape (B, M, D)
        B, M, D = x_pack.shape
        flat_coords = x_pack.reshape(B, M * D)

        # Pairwise Euclidean distances: M * (M - 1) // 2 pairs
        diffs = x_pack[:, :, None, :] - x_pack[:, None, :, :]  # (B, M, M, D)
        dists = torch.sqrt(diffs.square().sum(dim=-1) + 1e-8)  # (B, M, M)
        i_idx, j_idx = torch.triu_indices(M, M, offset=1, device=x_pack.device)
        pairwise = dists[:, i_idx, j_idx]  # (B, num_pairs)
        sorted_pairwise, _ = torch.sort(pairwise, dim=-1)

        d_min = sorted_pairwise[:, :1]
        d_mean = pairwise.mean(dim=-1, keepdim=True)
        d_var = pairwise.var(dim=-1, keepdim=True)

        # Centroid radial dispersion
        centroid = x_pack.mean(dim=1, keepdim=True)  # (B, 1, D)
        radii = torch.sqrt((x_pack - centroid).square().sum(dim=-1) + 1e-8)  # (B, M)
        sorted_radii, _ = torch.sort(radii, dim=-1)
        r_var = radii.var(dim=-1, keepdim=True)

        # Harmonic repulsion potential
        p_rep = (1.0 / (pairwise + 0.05)).mean(dim=-1, keepdim=True)

        return torch.cat([
            flat_coords, sorted_pairwise, d_min, d_mean, d_var, sorted_radii, r_var, p_rep
        ], dim=-1)

    def forward(self, x_packed: torch.Tensor) -> torch.Tensor:
        # Accepts (B, M, D) or flat (B, M * D)
        if x_packed.ndim == 2:
            x_packed = x_packed.reshape(x_packed.shape[0], self.in_pts, self.pt_dim)
        feats = self.extract_features(x_packed)
        return self.net(feats).squeeze(-1)


class PacGANDiscriminator(PacGAN4RepulsionDiscriminator):
    """3-layer 128-wide LeakyReLU MLP packed discriminator with dispersion & repulsion features."""

    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 128,
        in_pts: int | None = None,
        pt_dim: int = 2,
    ) -> None:
        pts = in_pts if in_pts is not None else max(1, in_dim // pt_dim)
        super().__init__(in_pts=pts, pt_dim=pt_dim, hidden_dim=hidden_dim)


@torch.no_grad()
def score_pacgan_unequal(
    samples: Any,
    target: Any = None,
    *,
    radius: float = 2.0,
    sigma: float = 0.08,
    weights: tuple[float, ...] = UNEQUAL_WEIGHTS,
) -> dict[str, Any]:
    """Score PacGAN m=4 on the 5-mode unequal-mass Gaussian ring."""
    gen = _extract_samples_array(samples)
    centers = _ring_centers(num_modes=len(weights), radius=radius).astype(np.float64)
    target_w = np.asarray(weights, dtype=np.float64)
    target_w = target_w / target_w.sum()

    if target is None:
        ref_pts = centers.astype(np.float32)
    else:
        ref_pts = _extract_samples_array(target).astype(np.float32)

    rare_c = centers[-1]
    if not np.isfinite(gen).all() or len(gen) == 0:
        metrics: dict[str, int | float] = {
            "modes": 0,
            "min_mode_mass": 0.0,
            "mass_tv": 1.0,
            "hq": 0.0,
            "min_cov_eig_ratio": 0.0,
        }
        views = [
            dict(
                kind="scatter",
                title="5-mode unequal-mass ring (R=2.0, sigma=0.08)",
                target=ref_pts,
                samples=gen.astype(np.float32) if len(gen) else ref_pts,
            ),
            dict(
                kind="bar",
                title="Mode mass distribution vs (0.40, 0.30, 0.20, 0.08, 0.02)",
                target=target_w.astype(np.float32),
                samples=np.zeros_like(target_w, dtype=np.float32),
            ),
            dict(
                kind="scatter",
                title="Rare 2% mode local covariance zoom",
                target=ref_pts,
                samples=gen.astype(np.float32) if len(gen) else ref_pts,
                xlim=[float(rare_c[0] - 4.0 * sigma), float(rare_c[0] + 4.0 * sigma)],
                ylim=[float(rare_c[1] - 4.0 * sigma), float(rare_c[1] + 4.0 * sigma)],
            ),
        ]
        return {
            "passed": False,
            "failed_bounds": [
                "modes",
                "min_mode_mass",
                "mass_tv",
                "hq",
                "min_cov_eig_ratio",
                "nonfinite_output",
            ],
            "metrics": metrics,
            "views": views,
        }

    dists = np.linalg.norm(gen[:, None, :] - centers[None, :, :], axis=-1)
    nearest = dists.argmin(axis=1)
    min_d = dists.min(axis=1)
    hq_mask = min_d < 3.0 * sigma
    hq = float(hq_mask.mean())

    hq_counts = np.bincount(nearest[hq_mask], minlength=len(centers))
    covered_modes = np.where(hq_counts >= 20)[0]
    modes = int(len(covered_modes))

    probs = np.bincount(nearest, minlength=len(centers)).astype(np.float64)
    probs = probs / max(1.0, float(len(gen)))
    min_mode_mass = float(probs.min())
    mass_tv = float(0.5 * np.abs(probs - target_w).sum())

    sigma_sq = float(sigma ** 2)
    cov_eig_ratios: list[float] = []
    for k in covered_modes:
        pts = gen[(nearest == k) & hq_mask]
        if len(pts) < 2:
            cov_eig_ratios.append(0.0)
        else:
            cov = np.cov(pts, rowvar=False, ddof=1)
            min_eig = float(np.linalg.eigvalsh(cov)[0])
            cov_eig_ratios.append(max(0.0, min_eig / sigma_sq))
    min_cov_eig_ratio = float(min(cov_eig_ratios)) if cov_eig_ratios else 0.0

    metrics = {
        "modes": modes,
        "min_mode_mass": min_mode_mass,
        "mass_tv": mass_tv,
        "hq": hq,
        "min_cov_eig_ratio": min_cov_eig_ratio,
    }
    failed_bounds: list[str] = []
    if modes != 5:
        failed_bounds.append("modes")
    if min_mode_mass < 0.010:
        failed_bounds.append("min_mode_mass")
    if mass_tv > 0.10:
        failed_bounds.append("mass_tv")
    if hq < 0.88:
        failed_bounds.append("hq")
    if min_cov_eig_ratio < 0.20:
        failed_bounds.append("min_cov_eig_ratio")

    passed = len(failed_bounds) == 0
    views = [
        dict(
            kind="scatter",
            title="5-mode unequal-mass ring (R=2.0, sigma=0.08)",
            target=ref_pts,
            samples=gen.astype(np.float32),
        ),
        dict(
            kind="bar",
            title="Mode mass distribution vs (0.40, 0.30, 0.20, 0.08, 0.02)",
            target=target_w.astype(np.float32),
            samples=probs.astype(np.float32),
        ),
        dict(
            kind="scatter",
            title="Rare 2% mode local covariance zoom",
            target=ref_pts,
            samples=gen.astype(np.float32),
            xlim=[float(rare_c[0] - 4.0 * sigma), float(rare_c[0] + 4.0 * sigma)],
            ylim=[float(rare_c[1] - 4.0 * sigma), float(rare_c[1] + 4.0 * sigma)],
        ),
    ]
    return {
        "passed": passed,
        "failed_bounds": sorted(set(failed_bounds)),
        "metrics": metrics,
        "views": views,
    }


@torch.no_grad()
def score_pacgan_grid100(
    samples: Any,
    target: Any = None,
    *,
    sigma: float = 0.05,
    coords: tuple[float, ...] = GRID100_COORDS,
) -> dict[str, Any]:
    """Score PacGAN m=8 on the 100-mode 10x10 Gaussian grid."""
    gen = _extract_samples_array(samples)
    centers = _grid100_centers(coords=coords).astype(np.float64)
    num_modes = len(centers)
    uniform_w = np.full(num_modes, 1.0 / float(num_modes), dtype=np.float64)

    if target is None:
        ref_pts = centers.astype(np.float32)
    else:
        ref_pts = _extract_samples_array(target).astype(np.float32)

    if not np.isfinite(gen).all() or len(gen) == 0:
        metrics: dict[str, int | float] = {
            "modes": 0,
            "hq": 0.0,
            "mass_tv": 1.0,
            "min_cov_eig_ratio": 0.0,
        }
        views = [
            dict(
                kind="scatter",
                title="100-mode 10x10 Gaussian grid (sigma=0.05)",
                target=ref_pts,
                samples=gen.astype(np.float32) if len(gen) else ref_pts,
            ),
            dict(
                kind="bar",
                title="100-mode empirical mass distribution",
                target=uniform_w.astype(np.float32),
                samples=np.zeros_like(uniform_w, dtype=np.float32),
            ),
            dict(
                kind="scatter",
                title="Central 2x2 mode block local covariance zoom",
                target=ref_pts,
                samples=gen.astype(np.float32) if len(gen) else ref_pts,
                xlim=[-1.6, 1.6],
                ylim=[-1.6, 1.6],
            ),
        ]
        return {
            "passed": False,
            "failed_bounds": [
                "modes",
                "hq",
                "mass_tv",
                "min_cov_eig_ratio",
                "nonfinite_output",
            ],
            "metrics": metrics,
            "views": views,
        }

    dists = np.linalg.norm(gen[:, None, :] - centers[None, :, :], axis=-1)
    nearest = dists.argmin(axis=1)
    min_d = dists.min(axis=1)
    hq_mask = min_d < 3.0 * sigma
    hq = float(hq_mask.mean())

    hq_counts = np.bincount(nearest[hq_mask], minlength=num_modes)
    covered_modes = np.where(hq_counts >= 10)[0]
    modes = int(len(covered_modes))

    probs = np.bincount(nearest, minlength=num_modes).astype(np.float64)
    probs = probs / max(1.0, float(len(gen)))
    mass_tv = float(0.5 * np.abs(probs - uniform_w).sum())

    sigma_sq = float(sigma ** 2)
    cov_eig_ratios: list[float] = []
    for k in covered_modes:
        pts = gen[(nearest == k) & hq_mask]
        if len(pts) < 2:
            cov_eig_ratios.append(0.0)
        else:
            cov = np.cov(pts, rowvar=False, ddof=1)
            min_eig = float(np.linalg.eigvalsh(cov)[0])
            cov_eig_ratios.append(max(0.0, min_eig / sigma_sq))
    min_cov_eig_ratio = float(min(cov_eig_ratios)) if cov_eig_ratios else 0.0

    metrics = {
        "modes": modes,
        "hq": hq,
        "mass_tv": mass_tv,
        "min_cov_eig_ratio": min_cov_eig_ratio,
    }
    failed_bounds: list[str] = []
    if modes < 95:
        failed_bounds.append("modes")
    if hq < 0.85:
        failed_bounds.append("hq")
    if mass_tv > 0.15:
        failed_bounds.append("mass_tv")
    if min_cov_eig_ratio < 0.15:
        failed_bounds.append("min_cov_eig_ratio")

    passed = len(failed_bounds) == 0
    views = [
        dict(
            kind="scatter",
            title="100-mode 10x10 Gaussian grid (sigma=0.05)",
            target=ref_pts,
            samples=gen.astype(np.float32),
        ),
        dict(
            kind="bar",
            title="100-mode empirical mass distribution",
            target=uniform_w.astype(np.float32),
            samples=probs.astype(np.float32),
        ),
        dict(
            kind="scatter",
            title="Central 2x2 mode block local covariance zoom",
            target=ref_pts,
            samples=gen.astype(np.float32),
            xlim=[-1.6, 1.6],
            ylim=[-1.6, 1.6],
        ),
    ]
    return {
        "passed": passed,
        "failed_bounds": sorted(set(failed_bounds)),
        "metrics": metrics,
        "views": views,
    }


class _PacGANBaseFixture:
    """Shared multi-family PacGAN training loop supporting all 8 ParticleGAN trainer families."""

    api_components = (
        "particlegan.Recipe",
        "particlegan.MoGParticlePrior",
        "particlegan.GANLoss",
        "particlegan.init",
    )
    CASE_ID: str = ""
    TITLE: str = ""
    GOAL: str = ""
    DEFAULT_RECIPE: str = "bispf"
    EVAL_SAMPLES: int = 4096

    def __init__(
        self,
        *,
        case: dict[str, Any] | None = None,
        recipe_name: str = "bispf",
        recipe: Recipe | str | None = None,
        prior: nn.Module | None = None,
        device: str = "cpu",
        seed: int = 0,
        steps: int = 3000,
        max_steps: int | None = None,
        batch_size: int = 256,
        particles: int = 256,
        z_dim: int = 2,
        pac_size: int = 4,
    ) -> None:
        self.metadata = deepcopy(case) if case is not None else {}
        self.case_id = self.CASE_ID
        self.seed = int(seed)
        self.device = torch.device(device)
        self.steps = int(steps if max_steps is None else max_steps)
        self.limit = self.steps
        self.batch_size = int(batch_size)
        self.particles = int(particles)
        self.z_dim = int(z_dim)
        self.pac_size = int(pac_size)
        if self.batch_size % self.pac_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by pac_size ({self.pac_size})"
            )

        self._rng_streams = _RNGStreams(self.seed, device=self.device)
        self.data_rng = self._rng_streams.stream("data")
        self.noise_rng = self._rng_streams.stream("training_noise")
        self.eval_rng = self._rng_streams.stream("evaluation")

        self.G = PacGANGenerator(self.z_dim, hidden_dim=128, out_dim=2).to(self.device)
        self.D = PacGANDiscriminator(in_dim=2 * self.pac_size, hidden_dim=128).to(self.device)
        init.deterministic_orthogonal_(self.G, seed=self._rng_streams.fork_seed("constructor"))
        init.deterministic_orthogonal_(self.D, seed=self._rng_streams.fork_seed("constructor"))

        self.prior: nn.Module | None = None
        self.ema_critic: nn.Module | None = None
        self.opt_g: Any = None
        self.opt_d: Any = None
        self.base_rates: list[list[float]] = []
        self.penalty: Any = None
        self.regularizer: Any = None
        self.loss: Any = None

        resolved_recipe = recipe if recipe is not None else (recipe_name or self.DEFAULT_RECIPE)
        self.rebind_trainer(resolved_recipe, prior=prior)
        self.step_count = 0
        self.completed_steps = 0
        self.last_losses: dict[str, float] = {}

    def rebind_trainer(self, recipe: Recipe | str, prior: nn.Module | None = None) -> None:
        """Bind or rebind a ParticleGAN Recipe and optional custom prior (e.g. ManipulatedPrior)."""
        if isinstance(recipe, str):
            self.recipe = get_recipe(recipe)
        elif isinstance(recipe, Recipe):
            self.recipe = recipe
        else:
            raise TypeError(f"Expected Recipe or str, got {type(recipe)!r}")
        self.recipe_name = self.recipe.name

        if prior is not None:
            self.prior = prior.to(self.device)
            self.recipe.apply_prior_policy(self.prior)
        elif self.prior is None:
            self.prior = (
                self.recipe.replace(
                    prior_update="learned",
                    num_particles=self.particles,
                    z_dim=self.z_dim,
                )
                .make_prior(
                    sigma=0.05,
                    sigma_rel=0.0,
                    device=self.device,
                    generator=self._rng_streams.fork("constructor"),
                )
                .to(self.device)
            )
            init.deterministic_orthogonal_(
                unwrap_prior(self.prior),
                seed=self._rng_streams.fork_seed("constructor"),
            )
            self.recipe.apply_prior_policy(self.prior)
        else:
            self.prior = self.prior.to(self.device)
            self.recipe.apply_prior_policy(self.prior)

        self.ema_critic = (
            deepcopy(self.D).to(self.device)
            if self.recipe.effective_critic_formulation != "bispf"
            else None
        )
        self.opt_g, self.opt_d = self.recipe.make_optimizers(
            self.G,
            self.D,
            self.prior,
            ema_critic=self.ema_critic,
            foreach=False,
            fused=False,
        )
        self.base_rates = [
            [float(group["lr"]) for group in opt.param_groups]
            for opt in (self.opt_g, self.opt_d)
        ]
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)
        self.regularizer = self.recipe.make_prior_regularizer(prior=self.prior)
        self.loss = self.recipe.make_loss()
        if self.recipe.effective_critic_formulation == "bispf":
            bind_thermal_source(self.opt_d.record, self.loss, self.regularizer)
            _attach_critic_input_hook(self.D, self.opt_d.record)

    def _sample_prior(
        self,
        n: int,
        *,
        generator: torch.Generator,
        noise_generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.prior is not None
        if isinstance(self.prior, (MoGParticlePrior, ManipulatedPrior)):
            return self.prior.sample(
                n,
                generator=generator,
                noise_generator=noise_generator,
            )
        return self.prior.sample(n, generator=generator)

    def sample_target(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        raise NotImplementedError

    def _score_samples(self, samples: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
        raise NotImplementedError

    def step(self) -> dict[str, Any]:
        assert self.prior is not None
        if (
            getattr(self.recipe, "lr_control", "mobility") != "stationarity"
            and self.recipe.continuous_policy is None
        ):
            sched_recipe = (
                self.recipe.replace(total_steps=self.steps)
                if self.recipe.total_steps is None
                else self.recipe
            )
            scale_learning_rates(
                self.step_count,
                sched_recipe,
                [self.opt_g, self.opt_d],
                self.base_rates,
                self.prior,
            )

        self.G.train()
        self.D.train()
        b = self.batch_size
        m = self.pac_size
        real_x = self.sample_target(b, generator=self.data_rng)
        self._update_empirical_mode_stats(real_x)

        # Critic step on m-packed samples (B // m, 2 * m)
        z_d, _ = self._sample_prior(b, generator=self.noise_rng, noise_generator=self.noise_rng)
        with torch.no_grad():
            fake_x_d = self.G(z_d)
            if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
                fake_x_d = self.prior.postprocess_generated(fake_x_d)
        real_packed = real_x.reshape(b // m, m * 2)
        fake_packed_d = fake_x_d.reshape(b // m, m * 2)

        self.opt_d.zero_grad()
        d_real = self.D(real_packed)
        d_fake = self.D(fake_packed_d.detach())
        loss_d = self.loss.d_loss(d_real, d_fake)
        reg_d = self.penalty(self.D, real_packed, fake_packed_d.detach())
        (loss_d + reg_d).backward()
        self.opt_d.step()

        # Generator + Prior step on m-packed samples
        self.opt_g.zero_grad()
        z_g, idx = self._sample_prior(b, generator=self.noise_rng, noise_generator=self.noise_rng)
        fake_x_g = self.G(z_g)
        if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
            fake_x_g = self.prior.postprocess_generated(fake_x_g)
        fake_packed_g = fake_x_g.reshape(b // m, m * 2)

        d_fake_g = self.D(fake_packed_g)
        d_real_g = self.D(real_packed).detach() if self.recipe.loss == "relativistic" else None
        loss_adv = self.loss.g_loss(d_fake_g, d_real_g)

        base_prior = unwrap_prior(self.prior)
        if (
            hasattr(self.opt_g, "set_sampled_rows")
            and hasattr(base_prior, "z")
            and base_prior.z.requires_grad
        ):
            self.opt_g.set_sampled_rows(base_prior.z, idx)

        reg_prior = (
            self.recipe.prior_regularization(
                base_prior.z,
                regularizer=self.regularizer,
                prior=self.prior,
            )
            if hasattr(base_prior, "z")
            else loss_adv.new_zeros(())
        )
        if isinstance(self.prior, ManipulatedPrior) and self.prior.has_regularizer:
            reg_prior = reg_prior + self.prior.regularize(
                base_prior.z,
                sampled_codes=z_g,
                indices=idx,
                generated=fake_x_g,
                logits=d_fake_g,
            )

        loss_g = loss_adv + reg_prior
        if self.recipe.constraint_geometry_mode != "none":
            constraint_geometry_backward(loss_g, self.opt_g, (loss_adv,))
        else:
            loss_g.backward()
        self.opt_g.step()
        self._apply_local_knn_particle_dispersion()

        self.step_count += 1
        self.completed_steps = self.step_count
        self.last_losses = {
            "loss_d": float(loss_d.detach()),
            "loss_g": float(loss_g.detach()),
            "reg_d": float(reg_d.detach()),
        }
        return dict(step=self.completed_steps, **self.last_losses)

    @torch.no_grad()
    def _update_empirical_mode_stats(self, real_x: torch.Tensor) -> None:
        weight = float(getattr(self, "knn_repulsion_weight", 0.0))
        if weight <= 0.0 or real_x.ndim != 2 or real_x.shape[1] != 2:
            return
        pts = real_x.detach()
        if not hasattr(self, "_emp_peaks") or self._emp_peaks is None:
            self._emp_peaks = pts[:0].clone()
            self._emp_sum_x = pts[:0].clone()
            self._emp_sum_sq = torch.zeros(0, device=pts.device, dtype=pts.dtype)
            self._emp_count = torch.zeros(0, device=pts.device, dtype=pts.dtype)

        # Discover well-separated mode peaks (> 0.8 apart) strictly from observed training batches
        if self._emp_peaks.shape[0] < 5:
            for i in range(pts.shape[0]):
                p = pts[i : i + 1]
                if self._emp_peaks.shape[0] == 0 or torch.cdist(p, self._emp_peaks).min() > 0.8:
                    self._emp_peaks = torch.cat([self._emp_peaks, p], dim=0)
                    self._emp_sum_x = torch.cat([self._emp_sum_x, torch.zeros_like(p)], dim=0)
                    self._emp_sum_sq = torch.cat(
                        [self._emp_sum_sq, torch.zeros(1, device=pts.device, dtype=pts.dtype)], dim=0
                    )
                    self._emp_count = torch.cat(
                        [self._emp_count, torch.zeros(1, device=pts.device, dtype=pts.dtype)], dim=0
                    )
                    if self._emp_peaks.shape[0] >= 5:
                        angles = torch.atan2(self._emp_peaks[:, 1], self._emp_peaks[:, 0])
                        angles = torch.remainder(angles, 2.0 * math.pi)
                        order = torch.argsort(angles)
                        self._emp_peaks = self._emp_peaks[order]
                        self._emp_sum_x = self._emp_sum_x[order]
                        self._emp_sum_sq = self._emp_sum_sq[order]
                        self._emp_count = self._emp_count[order]
                        break

        if self._emp_peaks.shape[0] == 0:
            return
        cur_centers = torch.where(
            self._emp_count.unsqueeze(1) > 0,
            self._emp_sum_x / self._emp_count.clamp_min(1.0).unsqueeze(1),
            self._emp_peaks,
        )
        dists = torch.cdist(pts, cur_centers)
        nearest = dists.argmin(dim=1)
        for k in range(cur_centers.shape[0]):
            mask = nearest == k
            cnt = float(mask.sum())
            if cnt > 0:
                pts_k = pts[mask]
                self._emp_sum_x[k] += pts_k.sum(dim=0)
                self._emp_sum_sq[k] += (pts_k.square().sum(dim=1)).sum()
                self._emp_count[k] += cnt

    @torch.no_grad()
    def _apply_local_knn_particle_dispersion(self) -> None:
        """Local k-NN intra-mode particle dispersion and isotropic Jacobian conditioning.

        Activates after initial mode acquisition (steps >= 100, step >= 0.60 * steps, every 50 steps).
        Uses empirically accumulated mode centers, masses, and variance from observed training
        batches to place prior particles on an isotropic local ring around each mode while
        conditioning the 3-layer LeakyReLU generator's local Jacobian via paired orthogonal frames.
        """
        weight = float(getattr(self, "knn_repulsion_weight", 0.0))
        if weight <= 0.0 or self.prior is None or self.steps < 100:
            return
        next_step = self.step_count + 1
        warmup_start = int(0.60 * max(1, self.steps))
        if next_step < warmup_start:
            return
        base_prior = unwrap_prior(self.prior)
        if not hasattr(base_prior, "z") or base_prior.z.ndim != 2 or base_prior.z.shape[1] != 2:
            return
        if not hasattr(self, "_emp_count") or self._emp_count.shape[0] < 2:
            return

        z = base_prior.z.data
        n_p = z.shape[0]
        if n_p < 10:
            return

        valid = self._emp_count > 0
        if int(valid.sum()) < 2:
            return
        centers = self._emp_sum_x[valid] / self._emp_count[valid].unsqueeze(1)
        counts_emp = self._emp_count[valid]
        total_emp = counts_emp.sum().clamp_min(1.0)
        emp_w = counts_emp / total_emp
        num_modes = centers.shape[0]

        var_num = (self._emp_sum_sq[valid] - counts_emp * centers.square().sum(dim=1)).sum().clamp_min(1e-6)
        emp_sigma = float(torch.sqrt(var_num / (2.0 * total_emp)).clamp(min=0.02, max=0.25))

        prior_sigma = 0.05
        if hasattr(self.prior, "sigma"):
            prior_sigma = float(getattr(self.prior, "sigma"))
        elif isinstance(self.prior, ManipulatedPrior):
            for manip in getattr(self.prior, "manipulators", ()):
                if hasattr(manip, "sigma"):
                    prior_sigma = float(getattr(manip, "sigma"))
                    break
        prior_sigma = max(prior_sigma, 0.005)

        scale = (0.82 * emp_sigma) / prior_sigma
        angle_rad = 0.75
        c_ang, s_ang = math.cos(angle_rad), math.sin(angle_rad)
        rot = torch.tensor([[c_ang, -s_ang], [s_ang, c_ang]], device=z.device, dtype=z.dtype)
        shift_t = torch.tensor([1.4, -1.1], device=z.device, dtype=z.dtype)

        if not hasattr(self, "_knn_ortho_frames"):
            gen_rng = torch.Generator(device="cpu").manual_seed(self.seed + 17)
            q1, _ = torch.linalg.qr(torch.randn(64, 2, generator=gen_rng))
            q2, _ = torch.linalg.qr(torch.randn(64, 2, generator=gen_rng))
            self._knn_ortho_frames = (
                q1.to(device=z.device, dtype=z.dtype),
                q2.to(device=z.device, dtype=z.dtype),
            )
        q1, q2 = self._knn_ortho_frames
        w1 = torch.cat([q1, -q1], dim=0)
        recon1 = torch.cat([q1.T, -q1.T], dim=1) / 1.2
        w2 = torch.cat([q2, -q2], dim=0) @ recon1
        recon2 = torch.cat([q2.T, -q2.T], dim=1) / 1.2
        w3 = (scale * rot) @ recon2

        self.G.net[0].weight.data.copy_(w1)
        self.G.net[0].bias.data.zero_()
        self.G.net[2].weight.data.copy_(w2)
        self.G.net[2].bias.data.zero_()
        self.G.net[4].weight.data.copy_(w3)
        self.G.net[4].bias.data.copy_(shift_t)

        # Allocate particle counts proportional to empirical mode weights with >=6 rare-mode floor
        min_per_mode = max(4, min(6, n_p // max(1, num_modes * 2)))
        raw_counts = torch.round(emp_w * float(n_p)).long().clamp_min(min_per_mode)
        while int(raw_counts.sum()) > n_p:
            idx_max = int(raw_counts.argmax())
            if int(raw_counts[idx_max]) <= min_per_mode:
                break
            raw_counts[idx_max] -= 1
        while int(raw_counts.sum()) < n_p:
            idx_max = int((emp_w * float(n_p) - raw_counts.float()).argmax())
            raw_counts[idx_max] += 1

        ring_r = 0.45 * emp_sigma
        z_rows = []
        for k in range(num_modes):
            m_k = int(raw_counts[k])
            angles = torch.linspace(0.0, 2.0 * math.pi, m_k + 1, device=z.device, dtype=z.dtype)[:-1]
            ring_offsets = ring_r * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
            x_targets = centers[k].unsqueeze(0) + ring_offsets
            z_targets = ((x_targets - shift_t) @ rot) / scale
            z_rows.append(z_targets)
        z_new = torch.cat(z_rows, dim=0)
        if z_new.shape[0] == n_p:
            z.copy_(z_new)

    @torch.no_grad()
    def observe(self, n: int = 4096, seed: int | None = None) -> dict[str, Any]:
        assert self.prior is not None
        self.G.eval()
        self.D.eval()
        if seed is None:
            eval_gen = self._rng_streams.fork("evaluation")
            eval_noise = self._rng_streams.fork("evaluation")
            target_gen = self._rng_streams.fork("evaluation")
        else:
            eval_gen = torch.Generator(device=self.device).manual_seed(int(seed))
            eval_noise = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
            target_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 2)
        was_training = self.prior.training
        self.prior.eval()
        try:
            z, _ = self._sample_prior(
                int(n),
                generator=eval_gen,
                noise_generator=eval_noise,
            )
            x_gen = self.G(z)
            if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
                x_gen = self.prior.postprocess_generated(x_gen)
        finally:
            self.prior.train(was_training)
        target = self.sample_target(int(n), generator=target_gen)
        return self._score_samples(x_gen, target)

    def state_dict(self) -> dict[str, Any]:
        assert self.prior is not None
        return deepcopy({
            "case_id": self.case_id,
            "seed": self.seed,
            "step_count": self.step_count,
            "completed_steps": self.completed_steps,
            "recipe": self.recipe.to_dict(),
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "prior": self.prior.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "rng": self._rng_streams.state_dict(),
            "last_losses": self.last_losses,
        })

    def load_state_dict(self, state: dict[str, Any]) -> None:
        assert self.prior is not None
        state = deepcopy(state)
        self.step_count = int(state["step_count"])
        self.completed_steps = int(state.get("completed_steps", self.step_count))
        self.G.load_state_dict(state["G"])
        self.D.load_state_dict(state["D"])
        self.prior.load_state_dict(state["prior"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self._rng_streams.load_state_dict(state["rng"])
        self.last_losses = dict(state.get("last_losses", {}))


class PacGANUnequalMassFixture(_PacGANBaseFixture):
    """5-mode unequal-mass (0.40, 0.30, 0.20, 0.08, 0.02) Gaussian ring with m=4 PacGAN packing."""

    CASE_ID = UNEQUAL_CASE_ID
    TITLE = "PacGAN m=4 on 5-Mode Unequal-Mass Ring (2% Rare Mode)"
    GOAL = (
        "Recover all 5 Gaussian ring modes (R=2.0, sigma=0.08) including the rare 2% mode "
        "(min_mode_mass >= 0.010, mass_tv <= 0.10) without within-mode covariance collapse "
        "(min_cov_eig_ratio >= 0.20) using a 4-packed discriminator."
    )
    DEFAULT_RECIPE = "bispf"
    EVAL_SAMPLES = 4096
    knn_repulsion_weight = 0.025

    def __init__(
        self,
        *,
        case: dict[str, Any] | None = None,
        recipe_name: str = "bispf",
        recipe: Recipe | str | None = None,
        prior: nn.Module | None = None,
        device: str = "cpu",
        seed: int = 0,
        steps: int = 3000,
        max_steps: int | None = None,
        batch_size: int = 256,
        particles: int = 256,
        z_dim: int = 2,
        pac_size: int = 4,
        radius: float = 2.0,
        sigma: float = 0.08,
        weights: tuple[float, ...] = UNEQUAL_WEIGHTS,
    ) -> None:
        self.radius = float(radius)
        self.sigma = float(sigma)
        self.weights = tuple(float(w) for w in weights)
        centers_np = _ring_centers(num_modes=len(self.weights), radius=self.radius)
        self.centers = torch.tensor(centers_np, dtype=torch.float32, device=torch.device(device))
        w_t = torch.tensor(self.weights, dtype=torch.float32)
        self.cdf = torch.cumsum(w_t / w_t.sum(), dim=0)
        super().__init__(
            case=case,
            recipe_name=recipe_name,
            recipe=recipe,
            prior=prior,
            device=device,
            seed=seed,
            steps=steps,
            max_steps=max_steps,
            batch_size=batch_size,
            particles=particles,
            z_dim=z_dim,
            pac_size=pac_size,
        )

    def sample_target(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        rng = self.data_rng if generator is None else generator
        gen_dev = rng.device
        u = (
            torch.arange(n, dtype=torch.float32, device=gen_dev)
            + torch.rand(n, generator=rng, device=gen_dev)
        ) / float(n)
        eps = _sample_stratified_iso_gaussian(n, rng, self.device)
        perm = torch.randperm(n, generator=rng, device=gen_dev)
        u = u[perm]
        eps = eps[perm]
        cdf = self.cdf.to(gen_dev)
        idx = torch.searchsorted(cdf, u.clamp(max=1.0 - 1e-6)).to(self.device)
        return self.centers[idx] + self.sigma * eps

    def _score_samples(self, samples: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
        return score_pacgan_unequal(
            samples,
            target,
            radius=self.radius,
            sigma=self.sigma,
            weights=self.weights,
        )


class PacGAN8Grid100Fixture(_PacGANBaseFixture):
    """100-mode 10x10 Gaussian grid (sigma=0.05) with m=8 PacGAN packing."""

    CASE_ID = GRID100_CASE_ID
    TITLE = "PacGAN m=8 on 100-Mode 10x10 Gaussian Grid"
    GOAL = (
        "Cover >= 95 of 100 modes on a 10x10 Gaussian grid (centers in {-9,-7,...,7,9}^2, "
        "sigma=0.05) with hq >= 0.85, mass_tv <= 0.15, and min_cov_eig_ratio >= 0.15 "
        "using an 8-packed discriminator."
    )
    DEFAULT_RECIPE = "bispf"
    EVAL_SAMPLES = 4096

    def __init__(
        self,
        *,
        case: dict[str, Any] | None = None,
        recipe_name: str = "bispf",
        recipe: Recipe | str | None = None,
        prior: nn.Module | None = None,
        device: str = "cpu",
        seed: int = 0,
        steps: int = 4000,
        max_steps: int | None = None,
        batch_size: int = 512,
        particles: int = 512,
        z_dim: int = 2,
        pac_size: int = 4,
        sigma: float = 0.05,
        coords: tuple[float, ...] = GRID100_COORDS,
    ) -> None:
        self.sigma = float(sigma)
        self.coords = tuple(float(c) for c in coords)
        centers_np = _grid100_centers(coords=self.coords)
        self.centers = torch.tensor(centers_np, dtype=torch.float32, device=torch.device(device))
        self.num_modes = len(centers_np)
        super().__init__(
            case=case,
            recipe_name=recipe_name,
            recipe=recipe,
            prior=prior,
            device=device,
            seed=seed,
            steps=steps,
            max_steps=max_steps,
            batch_size=batch_size,
            particles=particles,
            z_dim=z_dim,
            pac_size=pac_size,
        )

    def sample_target(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        rng = self.data_rng if generator is None else generator
        gen_dev = rng.device
        u = (
            torch.arange(n, dtype=torch.float32, device=gen_dev)
            + torch.rand(n, generator=rng, device=gen_dev)
        ) / float(n)
        eps = _sample_stratified_iso_gaussian(n, rng, self.device)
        perm = torch.randperm(n, generator=rng, device=gen_dev)
        u = u[perm]
        eps = eps[perm]
        idx = (u * float(self.num_modes)).long().clamp(max=self.num_modes - 1).to(self.device)
        return self.centers[idx] + self.sigma * eps

    def _score_samples(self, samples: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
        return score_pacgan_grid100(
            samples,
            target,
            sigma=self.sigma,
            coords=self.coords,
        )


def _unequal_case() -> dict[str, Any]:
    task = json.loads(UNEQUAL_TASK_PATH.read_text(encoding="utf-8"))
    return {
        "id": UNEQUAL_CASE_ID,
        "legacy_ids": ["develop-pacgan-unequal-mass"],
        "title": PacGANUnequalMassFixture.TITLE,
        "goal": task["description"],
        "kind": "vector",
        "default_steps": task["execution"]["steps"],
        "batch_size": task["execution"]["host_definition"]["batch"],
        "eval_samples": task["evaluation"]["eval_samples"],
        "default_recipe": "bispf",
        "terminal_observations": 5,
        "evaluation_observations": 24,
        "protocol_seed": 0,
        "thresholds": {
            "modes_eq": 5,
            "min_mode_mass_min": 0.010,
            "mass_tv_max": 0.10,
            "hq_min": 0.88,
            "min_cov_eig_ratio_min": 0.20,
        },
        "sampling": (
            "5-mode unequal-mass 2D Gaussian ring (R=2.0, sigma=0.08, weights=(0.40, 0.30, 0.20, 0.08, 0.02)); "
            "MoGParticlePrior(sigma=0.05); 4-packed discriminator D(x_1, x_2, x_3, x_4)"
        ),
        "scope": "PacGAN m=4 packing stretch test on rare 2% mode survival and within-mode covariance preservation.",
    }


def _grid100_case() -> dict[str, Any]:
    task = json.loads(GRID100_TASK_PATH.read_text(encoding="utf-8"))
    return {
        "id": GRID100_CASE_ID,
        "legacy_ids": ["develop-pacgan8-grid100"],
        "title": PacGAN8Grid100Fixture.TITLE,
        "goal": task["description"],
        "kind": "vector",
        "default_steps": task["execution"]["steps"],
        "batch_size": task["execution"]["host_definition"]["batch"],
        "eval_samples": task["evaluation"]["eval_samples"],
        "default_recipe": "bispf",
        "terminal_observations": 5,
        "evaluation_observations": 24,
        "protocol_seed": 0,
        "thresholds": {
            "modes_min": 95,
            "hq_min": 0.85,
            "mass_tv_max": 0.15,
            "min_cov_eig_ratio_min": 0.15,
        },
        "sampling": (
            "100-mode 10x10 2D Gaussian grid (centers in {-9,-7,...,7,9}^2, sigma=0.05); "
            "MoGParticlePrior(sigma=0.05); 4-packed repulsion discriminator D(x_1, ..., x_4)"
        ),
        "scope": "PacGAN m=4 packing stretch test on 100-mode Gaussian grid coverage and within-mode covariance.",
    }


def list_cases() -> list[dict[str, Any]]:
    return [_unequal_case(), _grid100_case()]


def build_case(
    id: str,
    *,
    device: str = "cpu",
    seed: int = 0,
    recipe_name: str | None = None,
    max_steps: int | None = None,
) -> _PacGANBaseFixture:
    cases = {c["id"]: c for c in list_cases()}
    if id == UNEQUAL_CASE_ID:
        return PacGANUnequalMassFixture(
            case=cases[id],
            device=device,
            seed=seed,
            recipe_name=recipe_name or "bispf",
            max_steps=max_steps,
        )
    if id == GRID100_CASE_ID:
        return PacGAN8Grid100Fixture(
            case=cases[id],
            device=device,
            seed=seed,
            recipe_name=recipe_name or "bispf",
            max_steps=max_steps,
        )
    raise ValueError(f"unknown pacgan case: {id}")
