"""Shared hyperparameters and small component factories; callers own control flow."""
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class Recipe:
    name: str = "gan_v3"
    model: str = "gan"
    z_dim: int = 4
    num_particles: int = 20_000
    prior_kind: str = "particles"
    sigma_rel: float = 0.0
    standardize: bool = True
    num_classes: int | None = None
    conditioning: str = "scalar"
    ucd_target: str = "class"
    ucd_weight: float = 0.02
    alpha_bar: tuple[float, ...] = (1.0, 0.9, 0.5, 0.05, 0.0001)
    batch_size: int = 256
    total_steps: int = 7_000
    lr: float = 0.00425
    d_lr_mult: float = 1.0
    prior_lr_mult: float = 2.0
    betas: tuple[float, float] = (0.0, 0.99)
    prior_betas: tuple[float, float] | None = None
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    reg_arm: str = "b_cap"
    reg_coeff: float = 6.0
    reg_kappa: float = 1.25
    reg_every: int = 1
    reg_method: str = "autograd"
    prior_reg: float = 0.05
    ema_decay: float = 0.995
    lr_anneal_start: float = 0.6
    lr_floor: float = 0.05
    encoder_mode: str = "none"
    routing_temperature: float = 0.25
    distance_reduction: str = "sum"
    observation_sigma: float = 0.03
    reconstruction_weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "betas", tuple(self.betas))
        if self.prior_betas is not None:
            object.__setattr__(self, "prior_betas", tuple(self.prior_betas))
        object.__setattr__(self, "alpha_bar", tuple(self.alpha_bar))
        for key in ("z_dim", "num_particles", "batch_size", "total_steps", "reg_every"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.encoder_mode not in ("none", "ae", "categorical", "hard"):
            raise ValueError("encoder_mode must be none, ae, categorical, or hard")
        if self.encoder_mode != "none" and self.prior_kind != "mog":
            raise ValueError("particle encoders require prior_kind='mog'")
        if self.distance_reduction not in ("sum", "mean"):
            raise ValueError("distance_reduction must be sum or mean")
        if self.model not in ("gan", "ddgan"):
            raise ValueError("model must be 'gan' or 'ddgan'")
        if self.prior_kind not in ("particles", "mog"):
            raise ValueError("prior_kind must be 'particles' or 'mog'")
        if not math.isfinite(self.sigma_rel) or self.sigma_rel < 0:
            raise ValueError("sigma_rel must be finite and nonnegative")
        if type(self.standardize) is not bool:
            raise ValueError("standardize must be a boolean")
        if self.prior_kind == "particles" and self.sigma_rel != 0:
            raise ValueError("nonzero sigma_rel requires prior_kind='mog'")
        if self.prior_kind == "mog" and self.num_particles < 2:
            raise ValueError("MoG calibration requires at least two particles")
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
        for key in ("lr", "d_lr_mult", "prior_lr_mult", "routing_temperature", "observation_sigma"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("reg_coeff", "reg_kappa", "prior_reg", "ucd_weight", "reconstruction_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.prior_betas is not None and (len(self.prior_betas) != 2 or any(not 0 <= b < 1 for b in self.prior_betas)):
            raise ValueError("prior_betas must contain two values in [0, 1) or be None")
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
        from .particle_prior import MoGParticlePrior, ParticlePrior
        options = {"num_particles": self.num_particles, "z_dim": self.z_dim,
                   "sigma_rel": self.sigma_rel, "standardize": self.standardize, **overrides}
        kind = options.pop("prior_kind", self.prior_kind)
        if kind == "mog":
            return MoGParticlePrior(**options)
        if kind != "particles":
            raise ValueError("prior_kind must be 'particles' or 'mog'")
        if options.pop("sigma_rel") != 0:
            raise ValueError("nonzero sigma_rel requires prior_kind='mog'")
        options.pop("standardize")
        return ParticlePrior(**options)

    def encode(self, query, prior, *, offset=None, draws=2, generator=None):
        """Route caller-produced queries; returns a ParticleEncoding.

        AE requires raw offsets and returns one deterministic draw. VAE rejects
        offsets because local posterior and prior must match for this recipe.
        """
        from .autoencoder import particle_ae, particle_vae
        options = dict(temperature=self.routing_temperature,
                       distance_reduction=self.distance_reduction)
        if self.encoder_mode == "ae":
            if offset is None:
                raise ValueError("AE encoding requires offset")
            return particle_ae(query, offset, prior, **options)
        if self.encoder_mode in ("categorical", "hard"):
            if offset is not None:
                raise ValueError("prior-matching VAE does not accept an offset")
            return particle_vae(query, prior, draws=draws, generator=generator,
                                hard=self.encoder_mode == "hard", **options)
        raise ValueError("this recipe has no encoder")

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

    def make_optimizers(self, generator, discriminator, prior=None, *, encoder=None, **adam_kwargs):
        """Return ordinary ``(Adam(G + optional E + prior), Adam(D))`` optimizers.

        Move modules to their desired device before calling. Frozen parameters
        are excluded, and a Gaussian/frozen prior adds no optimizer group.
        Additional Adam options, such as ``fused`` or ``eps``, apply to both
        optimizers. Set learning rates and betas on the recipe.
        """
        from torch.optim import Adam
        prior_params = [] if prior is None else [p for p in prior.parameters() if p.requires_grad]
        prior_ids = {id(p) for p in prior_params}
        g_params = []
        seen = set(prior_ids)
        for module in (generator, encoder):
            if module is not None:
                for p in module.parameters():
                    if p.requires_grad and id(p) not in seen:
                        g_params.append(p)
                        seen.add(id(p))
        d_params = [p for p in discriminator.parameters() if p.requires_grad]
        groups = []
        if g_params:
            groups.append({"params": g_params, "lr": self.lr})
        if prior_params:
            groups.append({"params": prior_params, "lr": self.lr * self.prior_lr_mult,
                           "betas": self.prior_betas if self.prior_betas is not None else self.betas})
        return (Adam(groups, lr=self.lr, betas=self.betas, **adam_kwargs),
                Adam(d_params, lr=self.lr * self.d_lr_mult, betas=self.betas, **adam_kwargs))


def get_recipe(name="gan", **overrides):
    """Select a model family with current shared defaults and explicit overrides.

    Names configure components, never training control flow or historical
    optimizer versions. Use ``Recipe(**saved_fields)`` for resolved checkpoints
    and ``recipe.replace(name=...)`` for custom report labels.
    """
    families = {
        "gan": {},
        "mog": dict(prior_kind="mog", sigma_rel=.025, num_particles=400),
        "ddgan": dict(model="ddgan", conditioning="ucd", num_classes=4),
        "ddgan_mog": dict(model="ddgan", conditioning="ucd", num_classes=4,
                          prior_kind="mog", sigma_rel=.025, num_particles=400),
        "ae_gan": dict(encoder_mode="ae", prior_kind="mog", sigma_rel=.025,
                       num_particles=400, z_dim=2),
        "vae_gan": dict(encoder_mode="hard", prior_kind="mog", sigma_rel=.025,
                        num_particles=400, z_dim=2),
        "ae_ddgan": dict(model="ddgan", encoder_mode="ae", prior_kind="mog",
                         sigma_rel=.025, num_particles=1024, z_dim=64,
                         batch_size=64, routing_temperature=.125,
                         distance_reduction="mean"),
    }
    if name not in families:
        raise ValueError(f"Unknown recipe {name!r}; choose {', '.join(families)}")
    options = families[name]
    if name != "gan":
        options = {"name": name, **options}
    return Recipe(**{**options, **overrides})


def learning_rate_scale(step, total_steps, start=0.6, floor=0.05):
    """Scale by completed updates: full LR for 60%, then cosine to the floor.

    Pass zero before the first optimizer update, as in the reference trainers.
    """
    if total_steps <= 0 or not 0 <= start < 1 or not 0 <= floor <= 1:
        raise ValueError("invalid learning-rate schedule")
    fraction = min(1.0, max(0.0, (step - start * total_steps) / ((1 - start) * total_steps)))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * fraction))
