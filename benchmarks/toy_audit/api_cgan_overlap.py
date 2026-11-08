"""Public-API conditional GAN benchmark on 4 overlapping 2D Gaussian classes.

Evaluates Projection Conditional Discriminator (Miyato & Koyama, ICLR 2018) and
Unary Conditional Discriminator (UCD, particlegan/conditioning.py) against
AC-GAN boundary repulsion and within-class variance truncation (Odena et al., ICML 2017).
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F

from particlegan import (
    ManipulatedPrior,
    MoGParticlePrior,
    ParticlePrior,
    Recipe,
    get_recipe,
    init,
    scale_learning_rates,
    ucd_labels,
    ucd_loss,
    unwrap_prior,
)
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from .reproducibility import DEFAULT_SEED

VERSION = "cgan-overlap-v1"
CASE_ID = "api-cgan-overlapping-classes"
NUM_CLASSES = 4
SIGMA = 0.35
SIGMA_SQ = SIGMA ** 2
OVERLAP_HALF_WIDTH = 0.35
CLASS_CENTERS = (
    (-0.50, -0.50),
    (-0.50, +0.50),
    (+0.50, -0.50),
    (+0.50, +0.50),
)

_CASES = [
    {
        "id": CASE_ID,
        "legacy_ids": ["pr231"],
        "title": "Conditional GAN on 4 overlapping 2D Gaussian classes (Projection Discriminator / UCD)",
        "goal": (
            "Fit 4 overlapping 2D Gaussian classes (centers at (+-0.50, +-0.50), sigma=0.35) "
            "using a Projection Conditional Discriminator D(x, y) = psi(phi(x)) + <phi(x), V y> "
            "(and optional UCD supervision) without AC-GAN boundary repulsion or variance truncation."
        ),
        "kind": "cgan_overlapping_classes",
        "scope": (
            "Prospective public-API conditional GAN benchmark based on Miyato & Koyama (ICLR 2018) "
            "and Odena et al. (ICML 2017), evaluating per-class coverage, mass balance, center accuracy, "
            "within-class covariance spectrum, and central overlap density in [-0.35, 0.35]^2."
        ),
        "default_steps": 3000,
        "batch_size": 256,
        "eval_samples": 4096,
        "default_recipe": "bispf",
        "protocol_seed": DEFAULT_SEED,
        "terminal_observations": 5,
        "evaluation_observations": 24,
        "sampling": (
            "4 uniform 2D Gaussian classes at (-0.5,-0.5), (-0.5,0.5), (0.5,-0.5), (0.5,0.5) "
            "with isotropic sigma=0.35; 256-particle prior (z_dim=2) paired with class y = i mod 4"
        ),
        "thresholds": {
            "classes_covered": 4,
            "class_mass_tv_max": 0.08,
            "mean_center_err_max": 0.12,
            "min_cov_eig_ratio_min": 0.60,
            "max_cov_eig_ratio_max": 1.50,
            "overlap_density_ratio_min": 0.55,
        },
    }
]


def list_cases():
    return deepcopy(_CASES)


def class_centers(device="cpu", dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(CLASS_CENTERS, device=device, dtype=dtype)


def true_overlap_fraction(
    centers: torch.Tensor | None = None,
    sigma: float = SIGMA,
    half_width: float = OVERLAP_HALF_WIDTH,
) -> float:
    """Exact analytical target fraction falling inside [-half_width, half_width]^2."""
    if centers is None:
        centers = class_centers(device="cpu", dtype=torch.float64)
    else:
        centers = centers.detach().cpu().to(dtype=torch.float64)
    s_sqrt2 = float(sigma) * math.sqrt(2.0)
    probs = []
    for c in range(len(centers)):
        mux, muy = float(centers[c, 0].item()), float(centers[c, 1].item())
        px = 0.5 * (math.erf((half_width - mux) / s_sqrt2) - math.erf((-half_width - mux) / s_sqrt2))
        py = 0.5 * (math.erf((half_width - muy) / s_sqrt2) - math.erf((-half_width - muy) / s_sqrt2))
        probs.append(px * py)
    return float(sum(probs) / max(len(probs), 1))


@torch.no_grad()
def score_cgan_overlap(
    samples: torch.Tensor,
    labels: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    target_labels: torch.Tensor | None = None,
    *,
    centers: torch.Tensor | None = None,
    sigma: float = SIGMA,
    overlap_half_width: float = OVERLAP_HALF_WIDTH,
) -> dict:
    """Evaluate N=4096 class-conditioned 2D samples (x, y) against the overlapping 4-class target law."""
    if centers is None:
        centers = class_centers(device=samples.device, dtype=torch.float32)
    else:
        centers = centers.to(device=samples.device, dtype=torch.float32)
    num_classes = len(centers)

    if labels is None:
        labels = torch.arange(len(samples), device=samples.device, dtype=torch.long) % num_classes
    elif labels.ndim == 2:
        labels = labels.argmax(dim=1).to(device=samples.device, dtype=torch.long)
    else:
        labels = labels.to(device=samples.device, dtype=torch.long)

    if (
        samples.ndim != 2
        or samples.shape[1] != 2
        or len(samples) == 0
        or len(labels) != len(samples)
        or not bool(torch.isfinite(samples).all())
        or not bool(((labels >= 0) & (labels < num_classes)).all())
    ):
        metrics = {
            "classes_covered": 0,
            "class_mass_tv": 1.0,
            "mean_center_err": float("inf"),
            "min_cov_eig_ratio": 0.0,
            "max_cov_eig_ratio": float("inf"),
            "overlap_density_ratio": 0.0,
        }
        return {
            "passed": False,
            "failed_bounds": [
                "classes_covered",
                "class_mass_tv",
                "mean_center_err",
                "min_cov_eig_ratio",
                "max_cov_eig_ratio",
                "overlap_density_ratio",
                "nonfinite_output",
            ],
            "metrics": metrics,
            "views": [],
        }

    x = samples.detach().float()
    y = labels.detach().long()
    sigma_sq = float(sigma) ** 2

    # 1. Per-class coverage: >= 500 samples within 3.5 * sigma of true class center mu_y
    dist_own = (x - centers[y]).norm(dim=1)
    in_support = dist_own <= (3.5 * float(sigma))
    support_counts = torch.bincount(y[in_support], minlength=num_classes)
    classes_covered = int((support_counts >= 500).sum().item())

    # 2. Class mass TV from uniform 1/4 (checks both label balance and spatial quadrant balance)
    label_counts = torch.bincount(y, minlength=num_classes).float()
    label_props = label_counts / label_counts.sum().clamp_min(1.0)
    tv_label = float(0.5 * (label_props - 1.0 / num_classes).abs().sum().item())

    nearest_idx = torch.cdist(x, centers).argmin(dim=1)
    nearest_counts = torch.bincount(nearest_idx, minlength=num_classes).float()
    nearest_props = nearest_counts / nearest_counts.sum().clamp_min(1.0)
    tv_nearest = float(0.5 * (nearest_props - 1.0 / num_classes).abs().sum().item())
    class_mass_tv = max(tv_label, tv_nearest)

    # 3. Per-class empirical mean error and covariance eigenvalue ratios
    center_errs = []
    all_eig_ratios = []
    for c in range(num_classes):
        x_c = x[y == c]
        if len(x_c) == 0:
            center_errs.append(float("inf"))
            all_eig_ratios.extend([0.0, float("inf")])
            continue
        mu_hat = x_c.mean(dim=0)
        center_errs.append(float((mu_hat - centers[c]).norm().item()))
        if len(x_c) < 3:
            all_eig_ratios.extend([0.0, float("inf")])
        else:
            cov_c = torch.cov(x_c.T, correction=1)
            eigs = torch.linalg.eigvalsh(cov_c.double()) / sigma_sq
            all_eig_ratios.extend([float(eigs[0].item()), float(eigs[1].item())])

    mean_center_err = float(sum(center_errs) / num_classes)
    min_cov_eig_ratio = float(min(all_eig_ratios))
    max_cov_eig_ratio = float(max(all_eig_ratios))

    # 4. Central overlap density ratio in [-overlap_half_width, overlap_half_width]^2
    in_box = (x[:, 0].abs() <= overlap_half_width) & (x[:, 1].abs() <= overlap_half_width)
    gen_overlap_frac = float(in_box.float().mean().item())
    ref_overlap_frac = true_overlap_fraction(centers=centers, sigma=sigma, half_width=overlap_half_width)
    overlap_density_ratio = float(min(1.5, gen_overlap_frac / max(ref_overlap_frac, 1e-8)))

    metrics = {
        "classes_covered": classes_covered,
        "class_mass_tv": class_mass_tv,
        "mean_center_err": mean_center_err,
        "min_cov_eig_ratio": min_cov_eig_ratio,
        "max_cov_eig_ratio": max_cov_eig_ratio,
        "overlap_density_ratio": overlap_density_ratio,
    }

    failed_bounds = []
    if metrics["classes_covered"] != 4:
        failed_bounds.append("classes_covered")
    if not math.isfinite(metrics["class_mass_tv"]) or metrics["class_mass_tv"] > 0.08:
        failed_bounds.append("class_mass_tv")
    if not math.isfinite(metrics["mean_center_err"]) or metrics["mean_center_err"] > 0.12:
        failed_bounds.append("mean_center_err")
    if not math.isfinite(metrics["min_cov_eig_ratio"]) or metrics["min_cov_eig_ratio"] < 0.60:
        failed_bounds.append("min_cov_eig_ratio")
    if not math.isfinite(metrics["max_cov_eig_ratio"]) or metrics["max_cov_eig_ratio"] > 1.50:
        failed_bounds.append("max_cov_eig_ratio")
    if not math.isfinite(metrics["overlap_density_ratio"]) or metrics["overlap_density_ratio"] < 0.55:
        failed_bounds.append("overlap_density_ratio")

    if target is None:
        ref_rng = torch.Generator(device="cpu").manual_seed(99123)
        ref_y = torch.arange(len(x), dtype=torch.long) % num_classes
        target = centers.detach().cpu()[ref_y] + float(sigma) * torch.randn(len(x), 2, generator=ref_rng)
    else:
        target = target.detach().cpu()

    x_cpu = x.detach().cpu()
    views = [
        {
            "kind": "scatter",
            "title": "Conditional generation vs 4 overlapping Gaussian classes",
            "target": target,
            "samples": x_cpu,
            "xlabel": "x",
            "ylabel": "y",
        },
        {
            "kind": "scatter",
            "title": "Central overlap zone [-0.35, 0.35]^2 (Bayes decision boundary)",
            "target": target,
            "samples": x_cpu,
            "xlim": [-0.65, 0.65],
            "ylim": [-0.65, 0.65],
            "xlabel": "x",
            "ylabel": "y",
        },
        {
            "kind": "bar",
            "title": "Class mass distribution",
            "target": torch.full((num_classes,), 1.0 / num_classes),
            "samples": (support_counts.float() / support_counts.sum().clamp_min(1.0)).detach().cpu(),
            "xlabel": "class",
            "ylabel": "mass",
        },
    ]
    return {
        "metrics": metrics,
        "passed": len(failed_bounds) == 0,
        "failed_bounds": sorted(set(failed_bounds)),
        "views": views,
    }


class ConditionalGenerator(nn.Module):
    """Conditional Generator G(z, one_hot(y, 4)): R^{2+4=6} -> R^2."""

    def __init__(self, z_dim: int = 2, num_classes: int = NUM_CLASSES, out_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.z_dim = int(z_dim)
        self.num_classes = int(num_classes)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.z_dim + self.num_classes, self.hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden, self.hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden, self.out_dim),
        )

    def forward(self, z_or_pair: torch.Tensor, y_onehot: torch.Tensor | None = None) -> torch.Tensor:
        if y_onehot is not None:
            if y_onehot.dtype in (torch.int32, torch.int64):
                y_onehot = F.one_hot(y_onehot.long(), num_classes=self.num_classes).to(dtype=z_or_pair.dtype)
            inp = torch.cat([z_or_pair, y_onehot], dim=1)
        elif z_or_pair.shape[1] == self.z_dim + self.num_classes:
            inp = z_or_pair
        elif z_or_pair.shape[1] == self.z_dim:
            y = torch.arange(len(z_or_pair), device=z_or_pair.device, dtype=torch.long) % self.num_classes
            inp = torch.cat([z_or_pair, F.one_hot(y, num_classes=self.num_classes).to(dtype=z_or_pair.dtype)], dim=1)
        else:
            raise ValueError(f"unexpected generator input shape {tuple(z_or_pair.shape)}")
        return self.net(inp)


class ProjectionConditionalDiscriminator(nn.Module):
    """Projection Conditional Discriminator D(x, y) = psi(phi(x)) + <phi(x), V y>.

    Miyato & Koyama (ICLR 2018): models the log density ratio directly via an inner
    product between the feature trunk phi(x) in R^{128} and class embedding V y,
    preserving within-class variance in overlapping density regions. Also exposes
    per-class logits for optional UCD supervision (particlegan.conditioning).
    """

    def __init__(self, x_dim: int = 2, num_classes: int = NUM_CLASSES, hidden: int = 128):
        super().__init__()
        self.x_dim = int(x_dim)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.phi = nn.Sequential(
            nn.Linear(self.x_dim, self.hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden, self.hidden),
            nn.LeakyReLU(0.2),
        )
        self.psi = nn.Linear(self.hidden, 1)
        self.V = nn.Linear(self.num_classes, self.hidden, bias=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi(x)

    def class_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self.phi(x)
        return self.psi(h) + h @ self.V.weight

    def forward(self, x_or_pair: torch.Tensor, y_onehot: torch.Tensor | None = None) -> torch.Tensor:
        if y_onehot is None:
            if x_or_pair.shape[1] != self.x_dim + self.num_classes:
                raise ValueError(
                    f"expected concatenated [x, y_onehot] of width {self.x_dim + self.num_classes}, "
                    f"got {tuple(x_or_pair.shape)}"
                )
            x = x_or_pair[:, : self.x_dim]
            y_onehot = x_or_pair[:, self.x_dim :]
        else:
            x = x_or_pair
            if y_onehot.dtype in (torch.int32, torch.int64):
                y_onehot = F.one_hot(y_onehot.long(), num_classes=self.num_classes).to(dtype=x.dtype)
        h = self.phi(x)
        scalar = self.psi(h).squeeze(-1)
        proj = (h * self.V(y_onehot)).sum(dim=1)
        return scalar + proj


def _resolve_recipe(case: dict, recipe_name: str | None = "bispf", recipe: Recipe | str | None = None) -> Recipe:
    if isinstance(recipe, Recipe):
        return recipe
    selected = recipe if isinstance(recipe, str) else (recipe_name or case.get("default_recipe", "bispf"))
    overrides = {
        "batch_size": case.get("batch_size", 256),
        "z_dim": 2,
        "num_particles": 256,
    }
    if selected not in ("e22", "atlas", "e22_routed"):
        overrides["total_steps"] = case.get("default_steps", 3000)
    return get_recipe(selected, **overrides)


class CGANOverlapFixture:
    api_components = (
        "particlegan.Recipe",
        "particlegan.Recipe.make_optimizers",
        "particlegan.Recipe.make_loss",
        "particlegan.Recipe.make_critic_penalty",
        "particlegan.Recipe.prior_regularization",
        "particlegan.ParticlePrior",
        "particlegan.conditioning",
        "particlegan.init",
    )

    def __init__(
        self,
        case: dict | None = None,
        *,
        recipe_name: str | None = "bispf",
        recipe: Recipe | str | None = None,
        prior: nn.Module | None = None,
        device: str | torch.device = "cpu",
        seed: int = DEFAULT_SEED,
        max_steps: int | None = None,
    ):
        self.metadata = deepcopy(_CASES[0] if case is None else case)
        self.case = self.metadata
        self.case_id = self.metadata["id"]
        self.device = torch.device(device)
        self.seed = int(seed)
        self.num_classes = NUM_CLASSES
        self.sigma = SIGMA
        self.completed_steps = 0
        self.limit = self.metadata["default_steps"] if max_steps is None else max_steps
        if type(self.limit) is not int or not 0 < self.limit <= self.metadata["default_steps"]:
            raise ValueError("max_steps must be in the declared budget")

        self.recipe = _resolve_recipe(self.metadata, recipe_name=recipe_name, recipe=recipe)
        self.centers = class_centers(device=self.device, dtype=torch.float32)

        self.G = ConditionalGenerator(z_dim=2, num_classes=self.num_classes, out_dim=2, hidden=128).to(self.device)
        self.D = ProjectionConditionalDiscriminator(x_dim=2, num_classes=self.num_classes, hidden=128).to(self.device)

        if prior is None:
            prior_init_rng = torch.Generator(device=self.device).manual_seed(self.seed + 50)
            self.prior = ParticlePrior(
                num_particles=256,
                z_dim=2,
                init_std=0.5,
                device=self.device,
                generator=prior_init_rng,
            ).to(self.device)
            init.deterministic_orthogonal_(self.prior, seed=self.seed + 3)
        else:
            self.prior = prior.to(self.device)

        init.deterministic_orthogonal_(self.G, seed=self.seed + 1)
        init.deterministic_orthogonal_(self.D, seed=self.seed + 2)

        self.data_rng = torch.Generator(device=self.device).manual_seed(self.seed + 101)
        self.generator_data_rng = torch.Generator(device=self.device).manual_seed(self.seed + 102)
        self.latent_rng = torch.Generator(device=self.device).manual_seed(self.seed + 103)
        self.noise_rng = torch.Generator(device=self.device).manual_seed(self.seed + 104)
        self.eval_rng = torch.Generator(device=self.device).manual_seed(self.seed + 105)
        self.batch_sequence_sha256 = hashlib.sha256(b"api-cgan-overlap-batches-v1").hexdigest()
        self.last_losses = {}

        self.rebind_trainer(self.recipe, prior=self.prior)

    def rebind_trainer(self, recipe: Recipe | str, prior: nn.Module | None = None):
        """Rebind the fixture to any ParticleGAN Recipe and optional prior."""
        self.recipe = _resolve_recipe(self.metadata, recipe_name=None, recipe=recipe)
        if prior is not None:
            self.prior = prior.to(self.device)
        base_prior = unwrap_prior(self.prior)
        if self.recipe.prior_update == "learned" and hasattr(base_prior, "z") and not isinstance(base_prior.z, nn.Parameter):
            z_data = base_prior.z.detach().clone()
            del base_prior.z
            base_prior.register_parameter("z", nn.Parameter(z_data))

        ema_critic = deepcopy(self.D) if self.recipe.effective_critic_formulation != "bispf" else None
        self.opt_g, self.opt_d = self.recipe.make_optimizers(
            self.G,
            self.D,
            self.prior,
            ema_critic=ema_critic,
            foreach=False,
            fused=False,
        )
        self.base_rates = [[group["lr"] for group in opt.param_groups] for opt in (self.opt_g, self.opt_d)]
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)
        self.regularizer = self.recipe.make_prior_regularizer(prior=self.prior, reference=self.prior)
        if self.recipe.effective_critic_formulation == "bispf":
            from particlegan.bispf import bind_thermal_source
            bind_thermal_source(self.opt_d.record, self.loss, self.regularizer)
        return self

    def _sample_prior(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
        noise_generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        gen = self.latent_rng if generator is None else generator
        ngen = self.noise_rng if noise_generator is None else noise_generator
        if isinstance(self.prior, (MoGParticlePrior, ManipulatedPrior)):
            z, idx = self.prior.sample(batch_size, generator=gen, noise_generator=ngen)
        else:
            z, idx = self.prior.sample(batch_size, generator=gen)
        return z, idx

    def sample_target_given_labels(self, labels: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        rng = self.data_rng if generator is None else generator
        eps = torch.randn(len(labels), 2, generator=rng, device=self.device)
        return self.centers[labels] + self.sigma * eps

    def sample_target(self, n: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self.data_rng if generator is None else generator
        labels = torch.arange(n, device=self.device, dtype=torch.long) % self.num_classes
        perm = torch.randperm(n, generator=rng, device=self.device)
        labels = labels[perm]
        x = self.sample_target_given_labels(labels, generator=rng)
        return x, labels

    def step(self) -> dict:
        if self.completed_steps >= self.limit:
            raise RuntimeError("declared execution cap reached")

        if (
            self.recipe.continuous_policy is None
            and getattr(self.recipe, "lr_control", "mobility") != "stationarity"
            and self.recipe.total_steps is not None
        ):
            scale_learning_rates(
                self.completed_steps,
                self.recipe,
                [self.opt_g, self.opt_d],
                self.base_rates,
                self.prior,
            )

        batch_size = self.recipe.batch_size
        self.G.train()
        self.D.train()

        with torch.no_grad():
            z_d, idx_d = self._sample_prior(batch_size, generator=self.latent_rng, noise_generator=self.noise_rng)
            y_d = idx_d % self.num_classes
            y_d_onehot = F.one_hot(y_d, num_classes=self.num_classes).float()
            x_fake_d = self.G(z_d, y_d_onehot)
            if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
                x_fake_d = self.prior.postprocess_generated(x_fake_d)
            x_real_d = self.sample_target_given_labels(y_d, generator=self.data_rng)
            pair_real = torch.cat([x_real_d, y_d_onehot], dim=1)
            pair_fake = torch.cat([x_fake_d, y_d_onehot], dim=1)

        self.opt_d.zero_grad(set_to_none=True)
        d_real = self.D(pair_real)
        d_fake = self.D(pair_fake.detach())
        loss_d = self.loss.d_loss(d_real, d_fake)
        if self.recipe.conditioning == "ucd":
            targets_d = ucd_labels(y_d, num_classes=self.num_classes, validate_args=False)
            loss_d = loss_d + ucd_loss(
                self.D.class_logits(x_real_d),
                self.D.class_logits(x_fake_d.detach()),
                targets_d,
                weight=self.recipe.ucd_weight,
            )
        reg_d = self.penalty(self.D, pair_real, pair_fake.detach())
        (loss_d + reg_d).backward()
        self.opt_d.step()

        flags = [p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            z_g, idx_g = self._sample_prior(batch_size, generator=self.latent_rng, noise_generator=self.noise_rng)
            y_g = idx_g % self.num_classes
            y_g_onehot = F.one_hot(y_g, num_classes=self.num_classes).float()
            x_fake = self.G(z_g, y_g_onehot)
            if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
                x_fake = self.prior.postprocess_generated(x_fake)
            x_real_g = self.sample_target_given_labels(y_g, generator=self.generator_data_rng)
            pair_fake_g = torch.cat([x_fake, y_g_onehot], dim=1)
            pair_real_g = torch.cat([x_real_g, y_g_onehot], dim=1)

            fake_logits = self.D(pair_fake_g)
            real_logits = self.D(pair_real_g)
            loss_adv = self.loss.g_loss(fake_logits, real_logits)

            base_prior = unwrap_prior(self.prior)
            prior_reg = self.recipe.prior_regularization(
                base_prior.z,
                regularizer=self.regularizer,
                prior=self.prior,
            )
            loss_g = loss_adv + prior_reg
            if isinstance(self.prior, ManipulatedPrior) and self.prior.has_regularizer:
                loss_g = loss_g + self.prior.regularize(
                    base_prior.z,
                    sampled_codes=z_g,
                    indices=idx_g,
                    generated=x_fake,
                    logits=fake_logits,
                )
            if self.recipe.kinetic_transport_weight:
                loss_g = loss_g + self.recipe.kinetic_transport_loss(x_fake, x_real_g)
            if self.recipe.kinetic_transport_local_weight:
                loss_g = loss_g + self.recipe.kinetic_transport_local_loss(x_fake, x_real_g)

            self.opt_g.zero_grad(set_to_none=True)
            if hasattr(self.opt_g, "set_sampled_rows") and base_prior.z.requires_grad:
                self.opt_g.set_sampled_rows(base_prior.z, idx_g)
            if self.recipe.constraint_geometry_mode != "none":
                constraint_geometry_backward(loss_g, self.opt_g, (loss_adv,))
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
            "loss_adv": float(loss_adv.detach()),
            "prior_reg": float(prior_reg.detach()),
            "reg_d": float(reg_d.detach()),
        }
        if not all(math.isfinite(v) for v in self.last_losses.values()):
            raise FloatingPointError("nonfinite conditional GAN update")
        return {"step": self.completed_steps, **self.last_losses}

    @torch.no_grad()
    def observe(self, n: int = 4096, seed: int = 10000) -> dict:
        self.G.eval()
        self.D.eval()
        eval_gen = torch.Generator(device=self.device).manual_seed(int(seed))
        eval_noise_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        target_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 2)

        z, idx = self._sample_prior(n, generator=eval_gen, noise_generator=eval_noise_gen)
        y = idx % self.num_classes
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        samples = self.G(z, y_onehot)
        if isinstance(self.prior, ManipulatedPrior) and self.prior.postprocesses_generated:
            samples = self.prior.postprocess_generated(samples)

        target_y = torch.arange(n, device=self.device, dtype=torch.long) % self.num_classes
        target = self.sample_target_given_labels(target_y, generator=target_gen)
        return score_cgan_overlap(
            samples,
            y,
            target=target,
            target_labels=target_y,
            centers=self.centers,
            sigma=self.sigma,
        )

    def state_dict(self) -> dict:
        return {
            "version": VERSION,
            "case_id": self.case_id,
            "seed": self.seed,
            "limit": self.limit,
            "completed_steps": self.completed_steps,
            "recipe": self.recipe.to_dict(),
            "generator": self.G.state_dict(),
            "critic": self.D.state_dict(),
            "prior": self.prior.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "data_rng": self.data_rng.get_state().clone(),
            "generator_data_rng": self.generator_data_rng.get_state().clone(),
            "latent_rng": self.latent_rng.get_state().clone(),
            "noise_rng": self.noise_rng.get_state().clone(),
            "batch_sequence_sha256": self.batch_sequence_sha256,
            "last_losses": deepcopy(self.last_losses),
            **self.recipe.prior_regularizer_checkpoint(self.regularizer),
        }

    def load_state_dict(self, state: dict):
        self.completed_steps = state["completed_steps"]
        self.G.load_state_dict(state["generator"])
        self.D.load_state_dict(state["critic"])
        self.prior.load_state_dict(state["prior"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.data_rng.set_state(state["data_rng"])
        self.generator_data_rng.set_state(state["generator_data_rng"])
        self.latent_rng.set_state(state["latent_rng"])
        self.noise_rng.set_state(state["noise_rng"])
        self.batch_sequence_sha256 = state.get("batch_sequence_sha256", self.batch_sequence_sha256)
        self.last_losses = deepcopy(state["last_losses"])
        self.recipe.restore_prior_regularizer(self.regularizer, state.get("prior_regularizer"))


def build_case(
    case_id: str = CASE_ID,
    *,
    device: str | torch.device = "cpu",
    seed: int = DEFAULT_SEED,
    recipe_name: str | None = "bispf",
    max_steps: int | None = None,
    recipe: Recipe | str | None = None,
    prior: nn.Module | None = None,
) -> CGANOverlapFixture:
    if case_id != CASE_ID:
        raise ValueError(f"unknown cgan overlap case: {case_id!r}")
    return CGANOverlapFixture(
        _CASES[0],
        recipe_name=recipe_name,
        recipe=recipe,
        prior=prior,
        device=device,
        seed=seed,
        max_steps=max_steps,
    )
