"""Versioned, inspectable recipes selected in the repository's toy studies."""
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class Recipe:
    name: str = "100gaussians"
    model: str = "gan"
    z_dim: int = 4
    num_particles: int = 20_000
    num_classes: int | None = None
    conditioning: str = "scalar"
    ucd_target: str = "class"
    ucd_weight: float = 0.02
    alpha_bar: tuple[float, ...] = (1.0, 0.9, 0.5, 0.05, 0.0001)
    batch_size: int = 256
    total_steps: int = 7_000
    lr: float = 6e-4
    d_lr_mult: float = 1.5
    prior_lr_mult: float = 10.0
    betas: tuple[float, float] = (0.0, 0.999)
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_every: int = 1
    reg_method: str = "autograd"
    prior_reg: float = 1.0
    ema_decay: float = 0.995
    lr_anneal_start: float = 0.6
    lr_floor: float = 0.05

    def __post_init__(self):
        object.__setattr__(self, "betas", tuple(self.betas))
        object.__setattr__(self, "alpha_bar", tuple(self.alpha_bar))
        for key in ("z_dim", "num_particles", "batch_size", "total_steps", "reg_every"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.model not in ("gan", "ddgan"):
            raise ValueError("model must be 'gan' or 'ddgan'")
        if self.conditioning not in ("scalar", "conditional", "ucd"):
            raise ValueError("conditioning must be scalar, conditional, or ucd")
        if self.num_classes is not None and (type(self.num_classes) is not int or self.num_classes < 1):
            raise ValueError("num_classes must be a positive integer or None")
        if self.conditioning != "scalar" and self.num_classes is None:
            raise ValueError("conditional and UCD recipes require num_classes")
        if self.ucd_target not in ("class", "time_class"):
            raise ValueError("ucd_target must be class or time_class")
        if self.ucd_target == "time_class" and (self.model != "ddgan" or self.conditioning != "ucd"):
            raise ValueError("time_class requires DDGAN + UCD")
        if not 0 <= self.ema_decay < 1 or not 0 <= self.lr_anneal_start < 1 or not 0 <= self.lr_floor <= 1:
            raise ValueError("invalid EMA decay or LR schedule")
        for key in ("lr", "d_lr_mult", "prior_lr_mult"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("reg_coeff", "reg_kappa", "prior_reg", "ucd_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        # Validate resolved component settings at construction, not later in training.
        self.make_loss()
        self.make_gradient_penalty()
        if (len(self.alpha_bar) < 2 or self.alpha_bar[0] != 1
                or any(not math.isfinite(a) or a <= 0 for a in self.alpha_bar)
                or any(b >= a for a, b in zip(self.alpha_bar, self.alpha_bar[1:]))):
            raise ValueError("alpha_bar must start at 1 and decrease strictly, remaining positive")

    def replace(self, **overrides):
        """Return a modified copy; unknown options raise TypeError."""
        return replace(self, **overrides)

    def to_dict(self):
        return asdict(self)

    def make_prior(self, **overrides):
        """Construct the learned prior; explicit keyword overrides are local to this call."""
        from .particle_prior import ParticlePrior
        return ParticlePrior(**{"num_particles": self.num_particles, "z_dim": self.z_dim, **overrides})

    def make_loss(self, **overrides):
        from .gan_loss import GANLoss
        return GANLoss(**{"loss_type": self.loss_type, "mode": self.gan_mode, **overrides})

    def make_gradient_penalty(self, **overrides):
        from .grad_regularizers import GradientPenalty
        return GradientPenalty(**{"arm": self.reg_arm, "coeff": self.reg_coeff,
                                  "kappa": self.reg_kappa, "lazy_k": self.reg_every,
                                  "method": self.reg_method, **overrides})

    def make_prior_regularizer(self, **overrides):
        from .vicreg_loss import ParticleRegularizer
        return ParticleRegularizer(**{"weight": self.prior_reg, **overrides})

    def make_optimizers(self, generator, discriminator, prior=None):
        """Return ordinary ``(Adam(G + prior), Adam(D))`` optimizers.

        Move modules to their desired device before calling. Frozen parameters
        are excluded, and a Gaussian/frozen prior adds no optimizer group.
        """
        from torch.optim import Adam
        prior_params = [] if prior is None else [p for p in prior.parameters() if p.requires_grad]
        prior_ids = {id(p) for p in prior_params}
        g_params = [p for p in generator.parameters() if p.requires_grad and id(p) not in prior_ids]
        d_params = [p for p in discriminator.parameters() if p.requires_grad]
        groups = []
        if g_params:
            groups.append({"params": g_params, "lr": self.lr})
        if prior_params:
            groups.append({"params": prior_params, "lr": self.lr * self.prior_lr_mult})
        return (Adam(groups, lr=self.lr, betas=self.betas),
                Adam(d_params, lr=self.lr * self.d_lr_mult, betas=self.betas))


def get_recipe(name="100gaussians", **overrides):
    if name == "100gaussians":
        recipe = Recipe()
    elif name == "denoising":
        recipe = Recipe(name=name, model="ddgan", num_classes=4,
                        conditioning="ucd", total_steps=56_000)
    else:
        raise ValueError(f"Unknown recipe {name!r}; choose '100gaussians' or 'denoising'")
    return recipe.replace(**overrides)


def learning_rate_scale(step, total_steps, start=0.6, floor=0.05):
    """Scale by completed updates: full LR for 60%, then cosine to the floor.

    Pass zero before the first optimizer update, as in the reference trainers.
    """
    if total_steps <= 0 or not 0 <= start < 1 or not 0 <= floor <= 1:
        raise ValueError("invalid learning-rate schedule")
    fraction = min(1.0, max(0.0, (step - start * total_steps) / ((1 - start) * total_steps)))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * fraction))
