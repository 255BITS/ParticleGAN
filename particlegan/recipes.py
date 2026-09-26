"""Shared hyperparameters and small component factories; callers own control flow."""
import math
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Recipe:
    """Resolved hyperparameters plus role-named factories (``make_*``) for one model.

    Fields are validated at construction; ``replace`` returns a modified copy.
    Two fields exist only for ablations; their defaults are the shipped
    formulation:

    * ``reg_anchor_weight`` (1.0) scales the critic penalty's EMA-anchor term,
      which ties the critic's input gradient to its parameter EMA once the
      controller anchors the critic. 0 removes the term (no ``ema_critic`` needed).
    * ``direct_particle_gain`` (True) lets a direct sample-particle group
      (``make_generator_optimizer(direct_particles=...)``) raise its LR by up
      to 2x while successive centered gradients agree. False keeps that
      group's LR at the scheduled value (``direct_particle_betas`` still apply).
    """
    name: str = "ka2"
    model: str = "gan"
    z_dim: int = 2
    num_particles: int = 20_000
    prior_kind: str = "particles"
    sigma_rel: float = 0.0
    standardize: bool = True
    num_classes: int | None = None
    conditioning: str = "scalar"
    ucd_target: str = "class"
    ucd_weight: float = 0.02
    alpha_bar: tuple[float, ...] = (1.0, 0.9, 0.5, 0.05, 0.0001)
    batch_size: int = 2048
    total_steps: int = 7_000
    lr: float = 0.00425
    d_lr_mult: float = 1.0
    prior_lr_mult: float = 2.0
    betas: tuple[float, float] = (0.0, 0.999)
    prior_betas: tuple[float, float] | None = None
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_every: int = 1
    prior_reg: float = 0.0
    ema_decay: float = 0.995
    lr_anneal_start: float = 0.6
    lr_floor: float = 0.05
    # G/D follow their own cosine over min(total, horizon cap)
    # down to network_lr_floor; the prior keeps the full-budget cosine above.
    # None means "same as lr_floor"; a None horizon cap means the full budget.
    network_lr_floor: float | None = 0.01
    network_lr_horizon_cap: int | None = 1600
    reg_anchor_min_decay: float = 0.90
    # Ablation switches; see the class docstring.
    reg_anchor_weight: float = 1.0
    direct_particle_gain: bool = True
    # Critic spike guard (d_guard_ratio=0 disables it).
    d_guard_ratio: float = 5.0
    d_guard_min_steps: int = 200
    # A2 sparse latent-row damping (0 disables it).
    latent_damping_max_rate: float = 0.5
    # Direct sample-particle response betas (make_generator_optimizer).
    direct_particle_betas: tuple[float, float] = (0.0, 0.9)
    # Critic input noise: peak std at the first update, linear to 0 by
    # input_noise_anneal_end * total_steps. Generator output noise: linear
    # warmup from 0 to output_noise_std over output_noise_warmup * total_steps.
    input_noise_std: float = 0.5
    input_noise_anneal_end: float = 0.1
    output_noise_std: float = 0.029
    output_noise_warmup: float = 0.2
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
        object.__setattr__(self, "direct_particle_betas", tuple(float(b) for b in self.direct_particle_betas))
        if self.network_lr_horizon_cap is not None and (
                type(self.network_lr_horizon_cap) is not int or self.network_lr_horizon_cap <= 0):
            raise ValueError("network_lr_horizon_cap must be a positive integer or None")
        if type(self.d_guard_min_steps) is not int or self.d_guard_min_steps < 0:
            raise ValueError("d_guard_min_steps must be a nonnegative integer")
        floor = self.network_lr_floor
        if floor is not None and (isinstance(floor, bool) or not math.isfinite(floor) or not 0 <= floor <= 1):
            raise ValueError("network_lr_floor must be None or in [0, 1]")
        if isinstance(self.reg_anchor_min_decay, bool) or not 0 <= self.reg_anchor_min_decay < 1:
            raise ValueError("reg_anchor_min_decay must be in [0, 1)")
        for key in ("d_guard_ratio", "input_noise_std", "output_noise_std"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("latent_damping_max_rate", "output_noise_warmup"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if not math.isfinite(self.input_noise_anneal_end) or not 0 < self.input_noise_anneal_end <= 1:
            raise ValueError("input_noise_anneal_end must be in (0, 1]")
        if len(self.direct_particle_betas) != 2 or any(not 0 <= b < 1 for b in self.direct_particle_betas):
            raise ValueError("direct_particle_betas must contain two values in [0, 1)")
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
        if type(self.direct_particle_gain) is not bool:
            raise ValueError("direct_particle_gain must be a boolean")
        for key in ("reg_coeff", "reg_kappa", "reg_anchor_weight", "prior_reg", "ucd_weight",
                    "reconstruction_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.prior_betas is not None and (len(self.prior_betas) != 2 or any(not 0 <= b < 1 for b in self.prior_betas)):
            raise ValueError("prior_betas must contain two values in [0, 1) or be None")
        # JSON commonly writes a zero moment as 0. Adam requires homogeneous
        # floating-point moment values, even when an integer passes our range
        # checks. Normalize only after validating the original values.
        object.__setattr__(self, "betas", tuple(float(b) for b in self.betas))
        if self.prior_betas is not None:
            object.__setattr__(self, "prior_betas", tuple(float(b) for b in self.prior_betas))
        # Validate resolved component settings at construction, not later in training.
        self._penalty_options()
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
        """Construct the prior; overrides are local to this call.

        MoG recipes calibrate spacing on the initialized
        means (potentially expensive). Pass ``sigma=...`` to skip calibration,
        including ``sigma=0`` when restoring a checkpoint.
        """
        from .particle_prior import MoGParticlePrior, ParticlePrior, calibrate_mog_sigma
        options = {"num_particles": self.num_particles, "z_dim": self.z_dim,
                   "sigma_rel": self.sigma_rel, "standardize": self.standardize, **overrides}
        kind = options.pop("prior_kind", self.prior_kind)
        if kind == "mog":
            sigma_rel = options.pop("sigma_rel")
            if "sigma" in options:
                return MoGParticlePrior(**options)
            prior = MoGParticlePrior(sigma=0, **options)
            sigma, d0 = calibrate_mog_sigma(prior.means(), sigma_rel)
            prior.set_sigma(sigma)
            prior.d0.copy_(d0)
            prior.sigma_rel = float(sigma_rel)
            return prior
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

    def make_loss(self):
        """The adversarial loss (RpGAN logistic): ``d_loss(real, fake)``, ``g_loss(fake, real)``."""
        from .gan_loss import GANLoss
        return GANLoss()

    def make_critic_penalty(self, optimizer, *, output=None, collect_stats=False, **penalty_overrides):
        """The critic gradient penalty paired with one critic optimizer.

        ``optimizer`` comes from ``make_optimizers`` or ``make_critic_optimizer``;
        the penalty reads the state it needs (EMA critic, LR record, step count)
        from it. Call it like a loss: ``penalty(D, real, fake, *condition,
        **condition_kwargs)`` returns a scalar tensor; the conditioning goes to
        the critic and its EMA. ``output`` selects the logits from the critic's
        output (default: first element of a tuple/list); ``collect_stats``
        fills ``penalty.last_stats``. ``penalty_overrides`` replace the
        recipe's ``coeff``, ``kappa``, ``lazy_k`` or
        ``anchor_weight`` for this penalty.
        """
        from .ka2 import CriticPenalty
        return CriticPenalty(self, optimizer, output=output, collect_stats=collect_stats,
                             **penalty_overrides)

    def _penalty_options(self, **overrides):
        """Resolved kernel settings for ``make_critic_penalty``."""
        options = {"coeff": self.reg_coeff, "kappa": self.reg_kappa, "lazy_k": self.reg_every,
                   "anchor_weight": self.reg_anchor_weight, **overrides}
        unknown = set(options) - {"coeff", "kappa", "lazy_k", "anchor_weight"}
        if unknown:
            raise TypeError(f"unknown critic penalty options: {sorted(unknown)}")
        from .ka2 import KA2GradientPenalty
        KA2GradientPenalty(**options)  # validate
        return options

    def make_critic_optimizer(self, critic, *, ema_critic=None, **adam_kwargs):
        """Adam over ``critic``'s trainable parameters whose ``step()`` does the
        recipe's critic-side work (KA2: spike guard, moment surprise, adaptive
        EMA-critic update and guarded reseed).

        ``ema_critic`` is a caller-allocated copy of ``critic`` (e.g.
        ``copy.deepcopy(critic)``) that becomes the EMA; the critic penalty
        requires it unless ``reg_anchor_weight == 0``. Checkpoint with ``optimizer.state_dict()``: it holds the
        EMA critic and all counters. ``adam_kwargs`` override the recipe's
        ``lr * d_lr_mult`` and ``betas`` or add options such as ``fused``.
        """
        from .ka2 import KA2CriticAdam
        options = {"lr": self.lr * self.d_lr_mult, "betas": self.betas, **adam_kwargs}
        return KA2CriticAdam([p for p in critic.parameters() if p.requires_grad], critic=critic,
                             ema_critic=ema_critic, anchor_min_decay=self.reg_anchor_min_decay,
                             guard_ratio=self.d_guard_ratio, guard_min_steps=self.d_guard_min_steps, **options)

    def make_generator_optimizer(self, params, *, latent_table=None, direct_particles=None, **adam_kwargs):
        """Adam over ``params`` (tensors or param groups) whose ``step()`` does the
        recipe's generator-side work (A2 damping of the sparse
        ``latent_table``, e.g. ``prior.z`` alone in its group with beta1 == 0,
        and the direct-particle response for the param group ``direct_particles``).

        With neither, ``step()`` is exactly ``Adam.step()``. ``adam_kwargs``
        override the recipe's ``lr`` and ``betas`` or add Adam options.
        """
        from .k3p import K3PGeneratorAdam
        options = {"lr": self.lr, "betas": self.betas, **adam_kwargs}
        return K3PGeneratorAdam(params, latent_table=latent_table, direct_particles=direct_particles,
                                latent_max_rate=self.latent_damping_max_rate,
                                direct_betas=self.direct_particle_betas,
                                direct_gain=self.direct_particle_gain, **options)

    @property
    def resolved_network_lr_floor(self):
        """G/D LR floor: ``network_lr_floor`` or ``lr_floor``."""
        return self.lr_floor if self.network_lr_floor is None else self.network_lr_floor

    def make_prior_regularizer(self, **overrides):
        from .vicreg_loss import ParticleRegularizer
        return ParticleRegularizer(**{"weight": self.prior_reg, **overrides})

    def make_optimizers(self, generator, discriminator, prior=None, *, encoder=None, ema_critic=None,
                        **adam_kwargs):
        """Return ``(opt_g, opt_d)``: Adam optimizers whose ``step()`` does the recipe's work.

        ``opt_g`` covers G + optional E + prior (``make_generator_optimizer``;
        a learnable ``ParticlePrior`` table gets A2 damping) and ``opt_d`` the
        critic (``make_critic_optimizer``, with the caller-allocated
        ``ema_critic``). Use them like any Adam: ``zero_grad``/``step``,
        ``state_dict``/``load_state_dict`` (which carry all regularization
        state), LR schedulers. Move modules to their desired device before
        calling. Frozen parameters are excluded, and a Gaussian/frozen prior
        adds no optimizer group. Additional Adam options, such as ``fused`` or
        ``eps``, apply to both optimizers. Set learning rates and betas on the
        recipe. The generator optimizer's groups are ``[generator/encoder,
        prior]`` (either may be absent); scale the prior group by the prior
        multiplier of ``learning_rate_scales`` and everything else by the
        network one.
        """
        from .particle_prior import ParticlePrior
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
        groups = []
        if g_params:
            groups.append({"params": g_params, "lr": self.lr})
        if prior_params:
            groups.append({"params": prior_params, "lr": self.lr * self.prior_lr_mult,
                           "betas": self.prior_betas if self.prior_betas is not None else self.betas})
        # A2 acts on a plain particle table (not MoG means or Gaussian priors).
        latent_table = prior.z if type(prior) is ParticlePrior and prior.z.requires_grad else None
        return (self.make_generator_optimizer(groups, latent_table=latent_table, **adam_kwargs),
                self.make_critic_optimizer(discriminator, ema_critic=ema_critic, **adam_kwargs))


def get_recipe(name="gan", **overrides):
    """Select a model family with current shared defaults and explicit overrides.

    Every family trains with the same formulation; names configure model
    components only. Use ``Recipe(**saved_fields)`` for resolved checkpoints
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


class NetworkLRTransition:
    """Caller-triggered network LR decay for a custom training loop.

    The caller decides when validation has plateaued and marks the number of
    completed updates. G/D then cosine-decay to the recipe's network floor over
    ``decay_steps``; the particle prior keeps its ordinary full-budget schedule.
    Save this object's state alongside the optimizers for exact continuation.
    """

    def __init__(self, decay_steps, start_step=None):
        if type(decay_steps) is not int or decay_steps <= 0:
            raise ValueError("decay_steps must be a positive integer")
        if start_step is not None and (type(start_step) is not int or start_step < 0):
            raise ValueError("start_step must be a nonnegative integer or None")
        self.decay_steps = decay_steps
        self.start_step = start_step

    def mark_plateau(self, completed_steps):
        """Start decay after this many updates; repeating the same mark is safe."""
        if type(completed_steps) is not int or completed_steps < 0:
            raise ValueError("completed_steps must be a nonnegative integer")
        if self.start_step is not None and self.start_step != completed_steps:
            raise ValueError("network LR transition is already marked")
        self.start_step = completed_steps

    def state_dict(self):
        return {"decay_steps": self.decay_steps, "start_step": self.start_step}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"decay_steps", "start_step"}:
            raise ValueError("invalid network LR transition state")
        restored = type(self)(**state)
        if restored.decay_steps != self.decay_steps:
            raise ValueError("network LR transition decay_steps differ from the configured schedule")
        self.start_step = restored.start_step


def learning_rate_scales(step, recipe, *, network_transition=None):
    """Return ``(network, prior)`` LR multipliers after ``step`` completed updates.

    Generator and critic ("network") follow ``learning_rate_scale`` over
    ``min(total_steps, network_lr_horizon_cap)`` down to ``network_lr_floor``
    and then hold; the particle prior follows it over the full budget down to
    ``lr_floor``. When ``network_transition`` is supplied, G/D instead hold at
    full LR until its caller-marked plateau, then decay over its ``decay_steps``.
    KA2's critic controller is independent of these multipliers.
    """
    total = recipe.total_steps
    if network_transition is None:
        horizon = min(total, recipe.network_lr_horizon_cap or total)
        network = learning_rate_scale(step, horizon, recipe.lr_anneal_start, recipe.resolved_network_lr_floor)
    elif isinstance(network_transition, NetworkLRTransition):
        network = (1.0 if network_transition.start_step is None else
                   learning_rate_scale(step - network_transition.start_step,
                                       network_transition.decay_steps, 0.0,
                                       recipe.resolved_network_lr_floor))
    else:
        raise TypeError("network_transition must be a NetworkLRTransition or None")
    prior = learning_rate_scale(step, total, recipe.lr_anneal_start, recipe.lr_floor)
    return network, prior


def scale_learning_rates(step, recipe, optimizers, base_rates, prior=None, *, network_transition=None):
    """Set every group's LR from ``learning_rate_scales(step, recipe)``.

    ``base_rates`` holds each optimizer's unscaled group LRs (read them once
    after construction). Groups whose parameters all belong to ``prior`` get
    the prior multiplier; every other group gets the network one, so a custom
    loop follows the recipe's split network/prior schedules.
    Pass a ``NetworkLRTransition`` to choose the network decay from validation
    while retaining the ordinary prior schedule. Returns ``(network, prior)``
    multipliers.
    """
    network, prior_scale = learning_rate_scales(step, recipe, network_transition=network_transition)
    prior_ids = set() if prior is None else {id(p) for p in prior.parameters()}
    for optimizer, rates in zip(optimizers, base_rates):
        for group, rate in zip(optimizer.param_groups, rates):
            is_prior = bool(prior_ids) and all(id(p) in prior_ids for p in group["params"])
            group["lr"] = rate * (prior_scale if is_prior else network)
    return network, prior_scale
