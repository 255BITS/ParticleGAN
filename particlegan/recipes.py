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
    critic_formulation: str = "ka2"
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
    total_steps: int | None = 7_000
    continuous_policy: str | None = None
    lr: float = 0.00425
    d_lr_mult: float = 1.0
    prior_lr_mult: float = 2.0
    betas: tuple[float, float] = (0.0, 0.999)
    prior_betas: tuple[float, float] | None = None
    # None uses critic_formulation. Explicit legacy arms use the K3P optimizer;
    # fixed R1/R2 and BCap retain their released L2 units.
    reg_arm: str | None = None
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
    # Opt-in label-free transport signal on the same G-phase target batch.
    kinetic_transport_weight: float = 0.0
    kinetic_transport_local_weight: float = 0.0
    kinetic_transport_projections: int = 32
    # AMSGrad for every recipe optimizer (G, prior and critic). Intended for a
    # G/D LR that stays high: the Adam step then shrinks with the gradient at
    # equilibrium instead of creeping up as the second moment decays. The
    # default keeps plain Adam (bit-identical to the release without it).
    amsgrad: bool = False
    # Ablation switches (appended; defaults reproduce the release bit-for-bit).
    # critic_r1_real=False drops only the real-data squared-gradient (R1) term
    # from the KA2 penalty's part A (fake cap, part-B caps and proximity stay).
    critic_r1_real: bool = True
    # critic_payoff_damping=False drops the continuous controller's D-only
    # 1/(1+payoff_error^2) LR factor (G/prior scales unchanged).
    critic_payoff_damping: bool = True
    # Generator output noise: "fixed" (output_noise_std), "learnable" (a
    # trainer-owned log-sigma trained with the generator loss through the
    # reparameterized noise, initialized at log(output_noise_std)) or
    # "mobility" (output_noise_std times the continuous controller mobility).
    output_noise_mode: str = "fixed"
    # LR control of a continuous (dv12) recipe: "mobility" (DV12: mobility and
    # game_trust scales, bit-identical to the release) or "stationarity" (per
    # optimizer group Pflug/SASA-style settling test in intrinsic time; see
    # continuous.SettleTest). Mobility is still updated (it drives
    # output_noise_mode="mobility") and game_trust still scales the KA2 anchor.
    lr_control: str = "mobility"
    # Fisher-Rao birth-death moves on the particle table (kNN density ratio,
    # sequential evidence, BH at q=.05; see birth_death.py). Off by default.
    particle_birth_death: bool = False
    # Per-row force-persistence evidence (row_evidence.py): rows whose own gradient history has a significant mean
    # keep the full table rate and do not vote in the table tester; descent is held while many rows are unbalanced.
    row_evidence_gate: bool = False
    # Table stationarity tester: which evidence may raise the table rate again ("any" | "both" | "never" | "anchor"), see continuous.py.
    table_release_rule: str = "any"
    # Parts of the row-evidence gate (ablation switches; all True is the declared mechanism).
    row_evidence_hot: bool = True
    row_evidence_exclude: bool = True
    row_evidence_hold: bool = True
    # Space in which birth-death compares the table's samples with the real reservoir: "data" (the samples themselves) or
    # "critic" (the critic's penultimate features: no data-space metric is needed). Locality uses the table's own latents.
    birth_death_space: str = "data"
    # Served model: 0 = the live iterate (the fast parameters) is what samples are drawn from. m > 0 = the averaged model is the model of
    # record: samples, evaluations and checkpoints' served parameters are an exponential average of the training iterate whose window is
    # m table-tester blocks (steps per block = b / s of the table tester, both controller state); the fast iterate continues to train.
    serve_average: float = 0.0
    # What re-opens the learning-rate ladders when the game changes under them: "data" = the base's real-batch drift
    # statistic (fast vs slow mean of random features of standardized RAW real batches, z-score > 3: data space);
    # "none" = no re-open and no statistic of the real batch is read (the ladders can still release on their own evidence);
    # "optimizer" = continuous.OptimizerSurprise: a sustained jump of the optimizers' own Adam step signal (|g| / sqrt(v_hat))
    # against its calm level re-opens every ladder once and restarts the generator-side Adam moments (hysteresis re-arm).
    reopen_signal: str = "data"
    # What an "optimizer" re-open does to the KA2 critic anchor: "hold" = nothing (a blind recipe forces the anchor on, W = 1,
    # and damps its EMA by game_trust, so after a re-open the anchor keeps pulling the critic toward the pre-event EMA critic);
    # "release" = the re-open counts as drift evidence for KA2 (evidence = 1) from the fire until KA2's own surprise ratio has
    # risen above REL_HI and fallen back below REL_LO: KA2's native release / EMA tracking / reseed rules then apply. Event-gated.
    reopen_anchor: str = "hold"
    # Optional acquisition guard for optimizer re-opens. None retains R1's
    # exact legacy law; "settled" requires a contracted network ladder at
    # the start of the surprise excursion and rebases known KA2 loss epochs.
    reopen_guard: str | None = None
    # Null law of the row-evidence gate: "theory" = the exact-law p-value of the statistic as is; "scaled" = the same law applied to t2 / c, where
    # c >= 1 is the smallest scale that makes the median p-value over the tested rows .5 (the typical row is the null: a slowly varying force
    # shared by neighbouring rows, correlated gradients or a different noise level scale every row's statistic alike; only rows that are extreme
    # relative to the bulk are flagged).
    row_evidence_null: str = "theory"
    # Birth-death also re-draws table rows that have no real support. With birth_death_space="critic": a split-conformal isolation test of every
    # row against the real reservoir in the critic's feature space (k-th neighbour distance over the local real scale; valid when the fake
    # law equals the real law, conservative for rows scored at their clean centres), Benjamini-Hochberg at the birth-death level over the table; flagged rows become clones of uniformly drawn unflagged
    # rows, and only while the flagged fraction is at most that same level (a table that is still mostly in transit is not resampled).
    birth_death_isolation: bool = False
    # Scale of the critic's feature space in critic-space birth-death and the support test: "none" = the features as the critic delivers them;
    # "std" = every feature divided by its standard deviation on the reference half (even rows) of the real reservoir. The critic function does not fix
    # the scale of a hidden unit (rescale a ReLU-type unit and its outgoing weight inversely: same function, different Euclidean distances), so the
    # raw metric depends on an arbitrary parametrisation; "std" is invariant to that symmetry and is a function of the reference half only.
    birth_death_feature_scale: str = "none"
    # Opt-in bounded feature law. Auto is conservatively scoped to a complete
    # raw-output moment frame and finite conformal/BH resolution.
    birth_death_backend: str = "knn"
    birth_death_cells: int = 64
    birth_death_metric_rank: int = 8
    birth_death_chunk: int = 256
    birth_death_parent_policy: str = "real_anchor"
    # Independent E22 uses its original equal-mass particle statistics.
    # routed_paired is a distinct conditional dense-bank adaptation, bound to
    # an explicit RoutedRows contract in the caller-owned policy API.
    row_policy: str = "independent"
    # Explicit optimizer family: the default retains KA2/K3P interventions.
    # adam uses the declared plain Adam law with only observer bookkeeping. Its fixed
    # penalties and optional moment/coefficient cosine schedules are declared
    # independently of any candidate name. Non-None endpoints interpolate from
    # betas[1]/reg_coeff over anneal_end * total_steps completed updates, then
    # hold. External execution limits do not change that declared horizon.
    optimizer_family: str = "formulation"
    # Dual-norm experiments retain the fixed loss/penalty and LR multiplier.
    # lr is their step size; isolation arms can pin native Adam groups to the
    # original baseline rate independently of the normalized-step sweep.
    optimizer_momentum: float = 0.0
    # Fixed gradient scale for smoothed polar, bias and sampled-prior updates.
    # epsilon in the smoothed-polar paper is optimizer_smoothing ** 2.
    optimizer_smoothing: float = 0.0
    # Explicit convolution adaptation; dense/default checkpoint packets stay unchanged.
    optimizer_convolution: str = "none"
    # Opt-in protection of existing G/encoder/prior objectives. Component
    # callers bind their protected losses before backward only when enabled.
    constraint_geometry_mode: str = "none"
    optimizer_adam_lr: float | None = None
    eps: float = 1e-8
    beta2_end: float | None = None
    beta2_anneal_end: float = 0.2
    reg_coeff_end: float | None = None
    reg_coeff_anneal_end: float = 0.2
    # Objective on raw scores; regularization is selected independently.
    loss: str = "relativistic"
    # G/E use betas/eps. D inherits unless these are explicit; prior inherits
    # G/E except for its existing prior_betas and optional prior_eps.
    d_betas: tuple[float, float] | None = None
    d_eps: float | None = None
    prior_eps: float | None = None
    loss_labels: tuple[float, float, float] = (0.0, 1.0, 1.0)
    adam_variant: str = "pytorch"
    # Exponential decay counts completed whole training updates, not role
    # applications or sampling calls. Its horizon is independent of total_steps.
    lr_schedule: str = "cosine"
    lr_decay_rate: float = 0.96
    lr_decay_steps: int = 50_000
    lr_decay_staircase: bool = False

    def __post_init__(self):
        from .gan_loss import GANLoss
        objective = GANLoss(self.loss, labels=self.loss_labels)
        object.__setattr__(self, "loss_labels", objective.labels)
        from .optim.dualnorm import NORMALIZED_FAMILIES
        if self.optimizer_family not in ("formulation", "adam", *NORMALIZED_FAMILIES):
            raise ValueError("unknown optimizer_family")
        if isinstance(self.optimizer_momentum, bool) or self.optimizer_momentum not in (0., .5, .9):
            raise ValueError("optimizer_momentum must be 0, 0.5 or 0.9")
        if self.optimizer_momentum != 0 and self.optimizer_family not in ("dualnorm", "dualnorm_D_only"):
            raise ValueError("optimizer_momentum requires a dualnorm optimizer family")
        if (type(self.optimizer_smoothing) not in (int, float)
                or not math.isfinite(self.optimizer_smoothing) or self.optimizer_smoothing < 0):
            raise ValueError("optimizer_smoothing must be finite and nonnegative")
        if self.optimizer_smoothing and self.optimizer_family != "dualnorm":
            raise ValueError("optimizer_smoothing requires optimizer_family='dualnorm'")
        if self.constraint_geometry_mode not in ("none", "nonascent", "strict_progress", "direction_blend"):
            raise ValueError("constraint_geometry_mode must be none, nonascent, strict_progress or direction_blend")
        if self.constraint_geometry_mode != "none" and (self.optimizer_family != "dualnorm" or self.optimizer_momentum):
            raise ValueError("constraint_geometry requires zero-momentum full DualNorm")
        if self.optimizer_convolution not in ("none", "per_offset"):
            raise ValueError("optimizer_convolution must be none or per_offset")
        if self.optimizer_convolution != "none" and self.optimizer_family != "dualnorm":
            raise ValueError("optimizer_convolution requires optimizer_family='dualnorm'")
        if self.optimizer_adam_lr is not None:
            if (isinstance(self.optimizer_adam_lr, bool) or not math.isfinite(self.optimizer_adam_lr)
                    or self.optimizer_adam_lr <= 0):
                raise ValueError("optimizer_adam_lr must be None or finite and positive")
            if self.optimizer_family not in ("dualnorm_D_only", "particle_rownorm_only"):
                raise ValueError("optimizer_adam_lr is supported only by isolation arms")
        if self.d_betas is not None:
            if (not isinstance(self.d_betas, (list, tuple)) or len(self.d_betas) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1
                           for v in self.d_betas)):
                raise ValueError("d_betas must contain two finite numbers in [0, 1) or be None")
            object.__setattr__(self, "d_betas", tuple(float(v) for v in self.d_betas))
        for key in ("d_eps", "prior_eps"):
            value = getattr(self, key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive or None")
        if self.adam_variant not in ("pytorch", "tensorflow_v1"):
            raise ValueError("adam_variant must be pytorch or tensorflow_v1")
        if self.adam_variant == "tensorflow_v1" and (
                self.optimizer_family != "adam" or self.amsgrad or self.beta2_end is not None):
            raise ValueError("tensorflow_v1 requires plain Adam, no AMSGrad and constant betas")
        if self.lr_schedule not in ("cosine", "constant", "exponential"):
            raise ValueError("lr_schedule must be cosine, constant or exponential")
        if (type(self.lr_decay_steps) is not int or self.lr_decay_steps <= 0
                or type(self.lr_decay_rate) not in (int, float)
                or not math.isfinite(self.lr_decay_rate) or not 0 < self.lr_decay_rate <= 1
                or type(self.lr_decay_staircase) is not bool):
            raise ValueError("invalid exponential LR rate, steps or staircase")
        if self.lr_schedule != "cosine" and (self.continuous_policy is not None or self.network_lr_horizon_cap is not None):
            raise ValueError("explicit constant/exponential LR requires a scheduled recipe without a horizon cap")
        if isinstance(self.eps, bool) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        for name in ("beta2_anneal_end", "reg_coeff_anneal_end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.beta2_end is not None and (
                isinstance(self.beta2_end, bool) or not math.isfinite(self.beta2_end) or not 0 <= self.beta2_end < 1):
            raise ValueError("beta2_end must be None or in [0, 1)")
        if self.reg_coeff_end is not None and (
                isinstance(self.reg_coeff_end, bool) or not math.isfinite(self.reg_coeff_end) or self.reg_coeff_end < 0):
            raise ValueError("reg_coeff_end must be None or finite and nonnegative")
        if self.optimizer_family != "formulation":
            if self.reg_arm not in ("a_r1r2", "b_cap"):
                raise ValueError("plain optimizers require an explicit fixed R1/R2 or BCap reg_arm")
            if (self.d_guard_ratio != 0 or self.reg_anchor_weight != 0
                    or self.latent_damping_max_rate != 0 or self.direct_particle_gain):
                raise ValueError("plain optimizers require disabled guard, anchor, latent damping and direct particle gain")
            if self.continuous_policy is not None:
                raise ValueError("plain optimizers do not implement a continuous update policy")
        if self.optimizer_family == "ada_nsgda" and (
                self.betas[0] != 0 or (self.d_betas is not None and self.d_betas[0] != 0)
                or (self.prior_betas is not None and self.prior_betas[0] != 0)):
            raise ValueError("ada_nsgda requires beta1=0 for networks and prior")
        if self.beta2_end is not None and self.optimizer_family != "adam":
            raise ValueError("scheduled beta2 requires optimizer_family='adam'")
        if self.reg_coeff_end is not None and self.reg_arm not in ("a_r1r2", "b_cap"):
            raise ValueError("scheduled reg_coeff requires a fixed R1/R2 or BCap reg_arm")
        if (self.beta2_end is not None or self.reg_coeff_end is not None) and self.total_steps is None:
            raise ValueError("recipe cosine schedules require a declared total_steps horizon")
        if self.critic_formulation not in ("ka2", "k3p", "bcap"):
            raise ValueError("critic_formulation must be ka2, k3p or bcap")
        if self.critic_formulation == "bcap" and (self.optimizer_family == "formulation" or self.reg_arm != "b_cap"):
            raise ValueError("bcap formulation requires a plain optimizer and reg_arm='b_cap'")
        if self.reg_arm is not None and self.critic_formulation != "bcap":
            # Keep resolved recipe/Forge provenance truthful about the optimizer
            # family selected by an explicit legacy arm.
            object.__setattr__(self, "critic_formulation", "k3p")
        if self.continuous_policy not in (None, "dv1", "dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            raise ValueError("unknown continuous_policy")
        if (self.total_steps is None) != (self.continuous_policy is not None):
            raise ValueError("continuous policies require total_steps=None")
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
            if key == "total_steps" and value is None:
                continue
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
        if (type(self.kinetic_transport_weight) not in (int, float)
                or not math.isfinite(self.kinetic_transport_weight) or self.kinetic_transport_weight < 0):
            raise ValueError("kinetic_transport_weight must be finite and nonnegative")
        if (type(self.kinetic_transport_local_weight) not in (int, float)
                or not math.isfinite(self.kinetic_transport_local_weight) or self.kinetic_transport_local_weight < 0):
            raise ValueError("kinetic_transport_local_weight must be finite and nonnegative")
        if type(self.kinetic_transport_projections) is not int or self.kinetic_transport_projections < 1:
            raise ValueError("kinetic_transport_projections must be a positive integer")
        if type(self.amsgrad) is not bool:
            raise ValueError("amsgrad must be a boolean")
        for key in ("critic_r1_real", "critic_payoff_damping"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        if self.output_noise_mode not in ("fixed", "learnable", "mobility"):
            raise ValueError("output_noise_mode must be fixed, learnable or mobility")
        if self.output_noise_mode == "learnable" and not self.output_noise_std > 0:
            raise ValueError("learnable output noise needs output_noise_std > 0 (its initial value)")
        if self.output_noise_mode != "fixed" and self.continuous_policy is None:
            raise ValueError("learnable/mobility output noise requires a continuous_policy (no warmup schedule)")
        if self.lr_control not in ("mobility", "stationarity"):
            raise ValueError("lr_control must be mobility or stationarity")
        if self.lr_control == "stationarity" and self.continuous_policy != "dv12":
            raise ValueError("lr_control='stationarity' requires continuous_policy='dv12'")
        if type(self.particle_birth_death) is not bool:
            raise ValueError("particle_birth_death must be a boolean")
        if self.particle_birth_death and self.prior_kind != "particles":
            raise ValueError("particle_birth_death requires prior_kind='particles' (a trainable table)")
        if (isinstance(self.serve_average, bool) or not isinstance(self.serve_average, (int, float))
                or not math.isfinite(self.serve_average) or self.serve_average < 0):
            raise ValueError("serve_average must be a finite number >= 0")
        if self.reopen_signal not in ("data", "none", "optimizer"):
            raise ValueError("reopen_signal must be data, none or optimizer")
        if self.reopen_anchor not in ("hold", "release"):
            raise ValueError("reopen_anchor must be hold or release")
        if self.reopen_anchor != "hold" and self.reopen_signal != "optimizer":
            raise ValueError("reopen_anchor release requires reopen_signal optimizer")
        if self.reopen_guard not in (None, "settled"):
            raise ValueError("reopen_guard must be None or settled")
        if self.reopen_guard is not None and (
                self.reopen_signal != "optimizer" or self.lr_control != "stationarity"):
            raise ValueError("reopen_guard settled requires optimizer re-open and stationarity control")
        if self.row_evidence_null not in ("theory", "scaled"):
            raise ValueError("row_evidence_null must be theory or scaled")
        if self.birth_death_space not in ("data", "critic"):
            raise ValueError("birth_death_space must be data or critic")
        if self.birth_death_feature_scale not in ("none", "std"):
            raise ValueError("birth_death_feature_scale must be none or std")
        if (self.row_policy == "independent" and self.birth_death_feature_scale == "std"
                and not (self.particle_birth_death and self.birth_death_space == "critic")):
            raise ValueError("birth_death_feature_scale='std' needs particle_birth_death with birth_death_space='critic'")
        if type(self.birth_death_isolation) is not bool:
            raise ValueError("birth_death_isolation must be a boolean")
        if (self.row_policy == "independent" and self.birth_death_isolation
                and not (self.particle_birth_death and self.birth_death_space == "critic")):
            raise ValueError("birth_death_isolation needs particle_birth_death with birth_death_space='critic' (a support test on raw samples is not allowed)")
        for _name in ("row_evidence_hot", "row_evidence_exclude", "row_evidence_hold"):
            if type(getattr(self, _name)) is not bool:
                raise ValueError(f"{_name} must be a boolean")
        if self.table_release_rule not in ("any", "both", "never", "anchor"):
            raise ValueError("table_release_rule must be any, both, never or anchor")
        if self.birth_death_backend not in ("knn", "feature_cells", "auto"):
            raise ValueError("birth_death_backend must be knn, feature_cells or auto")
        for key in ("birth_death_cells", "birth_death_metric_rank", "birth_death_chunk"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.birth_death_parent_policy != "real_anchor":
            raise ValueError("birth_death_parent_policy must be real_anchor")
        if self.birth_death_backend != "knn":
            if (not self.particle_birth_death or self.birth_death_space != "critic"
                    or self.birth_death_feature_scale != "std" or self.continuous_policy != "dv12"
                    or self.lr_control != "stationarity"):
                raise ValueError("feature/auto requires standardized critic birth/death with DV12 stationarity")
        elif (self.birth_death_cells, self.birth_death_metric_rank, self.birth_death_chunk) != (64, 8, 256):
            raise ValueError("feature-cell options require feature_cells or auto")
        if type(self.row_evidence_gate) is not bool:
            raise ValueError("row_evidence_gate must be a boolean")
        if self.row_policy not in ("independent", "routed_paired"):
            raise ValueError("row_policy must be independent or routed_paired")
        if self.row_policy == "routed_paired" and (
                self.continuous_policy != "dv12" or self.lr_control != "stationarity"):
            raise ValueError("row_policy='routed_paired' requires DV12 and stationarity control")
        if self.row_evidence_gate and self.lr_control != "stationarity":
            raise ValueError("row_evidence_gate requires lr_control='stationarity'")
        if (self.row_policy == "independent" and (self.particle_birth_death or self.row_evidence_gate)
                and self.conditioning != "scalar"):
            raise ValueError("particle birth/death and row evidence require independently sampled "
                             "unconditional rows; use row_policy='routed_paired' with RoutedRows "
                             "for conditional dense banks")
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
        from .recipe_compat import without_default_additions
        result = without_default_additions(asdict(self))
        if self.loss == "relativistic":
            # Older recipes/checkpoints had this one fixed objective. Preserve
            # their packets while recording every alternative explicitly.
            result.pop("loss")
        if self.optimizer_family == "formulation":
            result.pop("optimizer_family")
        if self.optimizer_momentum == 0:
            result.pop("optimizer_momentum")
        if self.optimizer_convolution == "none":
            result.pop("optimizer_convolution", None)
        if self.optimizer_adam_lr is None:
            result.pop("optimizer_adam_lr")
        if self.eps == 1e-8:
            result.pop("eps")
        if self.beta2_end is None:
            result.pop("beta2_end")
            if self.beta2_anneal_end == 0.2:
                result.pop("beta2_anneal_end")
        if self.reg_coeff_end is None:
            result.pop("reg_coeff_end")
            if self.reg_coeff_anneal_end == 0.2:
                result.pop("reg_coeff_anneal_end")
        if self.critic_formulation == "ka2":
            result.pop("critic_formulation")
        if self.reg_arm is None:
            result.pop("reg_arm")
        if self.reopen_guard is None:
            result.pop("reopen_guard")
        if self.birth_death_backend == "knn":
            # Keep current reference Recipe/config/checkpoint dictionaries.
            for key in ("birth_death_backend", "birth_death_cells", "birth_death_metric_rank",
                        "birth_death_chunk", "birth_death_parent_policy"):
                result.pop(key)
        return result

    def make_prior(self, **overrides):
        """Construct the prior; overrides are local to this call.

        Learnable tables take the ordinary random draw; use
        ``particlegan.init.deterministic_orthogonal_(prior)`` for R2 points (it
        recalibrates MoG spacing). MoG recipes calibrate spacing on the
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
        """The selected objective: ``d_loss(real, fake)``, ``g_loss(fake, real=None)``."""
        from .gan_loss import GANLoss
        return GANLoss(self.loss, labels=self.loss_labels)

    def kinetic_transport_loss(self, fake, real):
        """Weighted empirical transport on caller-owned G-phase output batches.

        Enabling this objective requires a host consumer. It draws no samples,
        detaches real targets and retains gradients through fake outputs. Hosts
        with zero weight retain their original objective without calling it.
        """
        from .kinetic_transport import kinetic_transport_loss
        return self.kinetic_transport_weight * kinetic_transport_loss(
            fake, real, projections=self.kinetic_transport_projections)

    def kinetic_transport_local_loss(self, fake, real):
        """Weighted local-v2 residual on the same caller-owned output batches.

        Anchors, bandwidths and normalizers come from detached real targets.
        This auxiliary term does not become a protected projection objective.
        """
        from .kinetic_transport import kinetic_transport_local_loss
        return self.kinetic_transport_local_weight * kinetic_transport_local_loss(fake, real)

    def make_critic_penalty(self, optimizer, *, output=None, collect_stats=False, **penalty_overrides):
        """The critic gradient penalty paired with one critic optimizer.

        ``optimizer`` comes from ``make_optimizers`` or ``make_critic_optimizer``;
        the penalty reads the state it needs (EMA critic, LR record, step count)
        from it. Call it like a loss: ``penalty(D, real, fake, *condition,
        **condition_kwargs)`` returns a scalar tensor; the conditioning goes to
        the critic and its EMA. ``output`` selects the logits from the critic's
        output (default: first element of a tuple/list); ``collect_stats``
        fills ``penalty.last_stats``. ``penalty_overrides`` replace the
        recipe's ``coeff``, ``kappa``, ``lazy_k``,
        ``anchor_weight`` or ``r1_real`` for this penalty.
        """
        if self.effective_critic_formulation in ("k3p", "bcap"):
            from .k3p import CriticPenalty
        else:
            from .ka2 import CriticPenalty
        return CriticPenalty(self, optimizer, output=output, collect_stats=collect_stats,
                             **penalty_overrides)

    def _penalty_options(self, **overrides):
        """Resolved kernel settings for ``make_critic_penalty``."""
        if self.critic_formulation not in ("ka2", "k3p", "bcap"):
            raise ValueError("critic_formulation must be ka2, k3p or bcap")
        if self.effective_critic_formulation in ("k3p", "bcap"):
            if not self.critic_r1_real:
                raise ValueError("critic_r1_real=False requires the KA2 formulation")
            floor = self.resolved_network_lr_floor
            options = {"arm": self.reg_arm or "k3p", "coeff": self.reg_coeff,
                       "kappa": self.reg_kappa, "lazy_k": self.reg_every,
                       "anchor_weight": self.reg_anchor_weight,
                       "lr_floor": floor if floor < .5 else 0., **overrides}
            from .grad_regularizers import GradientPenalty
            GradientPenalty(**options)
            return options
        options = {"coeff": self.reg_coeff, "kappa": self.reg_kappa, "lazy_k": self.reg_every,
                   "anchor_weight": self.reg_anchor_weight, "r1_real": self.critic_r1_real, **overrides}
        unknown = set(options) - {"coeff", "kappa", "lazy_k", "anchor_weight", "r1_real"}
        if unknown:
            raise TypeError(f"unknown critic penalty options: {sorted(unknown)}")
        from .ka2 import KA2GradientPenalty
        KA2GradientPenalty(**options)  # validate
        return options

    def make_critic_optimizer(self, critic, *, ema_critic=None, **adam_kwargs):
        """Optimizer over ``critic``'s trainable parameters whose ``step()`` does the
        recipe's critic-side work (KA2: spike guard, moment surprise, adaptive
        EMA-critic update and guarded reseed).

        ``ema_critic`` is a caller-allocated copy of ``critic`` (e.g.
        ``copy.deepcopy(critic)``) that becomes the EMA; the critic penalty
        requires it unless ``reg_anchor_weight == 0``. Checkpoint with ``optimizer.state_dict()``: it holds the
        EMA critic and all counters. ``adam_kwargs`` override the recipe's
        ``lr * d_lr_mult``, ``betas`` and ``amsgrad`` or add options such as ``fused``.
        ``optimizer_family='adam'`` returns the declared Adam law with disabled intervention
        metadata and an observation-only checkpointed critic step counter.
        Normalized families use the same fixed penalty and step observer;
        ``lr * d_lr_mult`` is their scheduled critic step size.
        """
        options = {"lr": self.lr * self.d_lr_mult, "betas": self.d_betas or self.betas,
                   "amsgrad": self.amsgrad, "eps": self.eps if self.d_eps is None else self.d_eps,
                   **adam_kwargs}
        if self.optimizer_family == "adam":
            from .recipe_schedules import make_plain_adam
            return make_plain_adam(self, [p for p in critic.parameters() if p.requires_grad], critic=critic, **options)
        if self.optimizer_family != "formulation":
            from .optim.dualnorm import convolution_parameter_groups, make_normalized_optimizer
            if self.optimizer_family == "particle_rownorm_only" and self.optimizer_adam_lr is not None:
                options["lr"] = self.optimizer_adam_lr * self.d_lr_mult
            params = [{"params": [p for p in critic.parameters() if p.requires_grad], "role": "critic"}]
            if self.optimizer_convolution == "per_offset":
                params = convolution_parameter_groups(params, [critic], family=self.optimizer_family)
            return make_normalized_optimizer(self, params, critic=critic, **options)
        if self.effective_critic_formulation == "k3p":
            from .k3p import K3PCriticAdam
            return K3PCriticAdam([p for p in critic.parameters() if p.requires_grad], critic=critic,
                                 ema_critic=ema_critic, anchor_decay=.999,
                                 guard_ratio=self.d_guard_ratio, guard_min_steps=self.d_guard_min_steps, **options)
        from .ka2 import KA2CriticAdam
        return KA2CriticAdam([p for p in critic.parameters() if p.requires_grad], critic=critic,
                             ema_critic=ema_critic, anchor_min_decay=self.reg_anchor_min_decay,
                             guard_ratio=self.d_guard_ratio, guard_min_steps=self.d_guard_min_steps, **options)

    def make_generator_optimizer(self, params, *, latent_table=None, direct_particles=None, **adam_kwargs):
        """Optimizer over ``params`` (module, tensors or param groups) whose ``step()`` does the
        recipe's generator-side work (A2 damping of the sparse
        ``latent_table``, e.g. ``prior.z`` alone in its group with beta1 == 0,
        and the direct-particle response for the param group ``direct_particles``).

        With neither, ``step()`` is exactly ``Adam.step()``. ``adam_kwargs``
        override the recipe's ``lr``, ``betas`` and ``amsgrad`` or add Adam options.
        Plain ``optimizer_family='adam'`` returns the declared Adam law without A2 or
        direct-particle response, including when role annotations are supplied.
        Normalized families accept role-named groups. Their row-normalized
        prior requires ``set_sampled_rows(prior.z, indices)`` before stepping;
        the public trainer/policy records generator-side sample IDs automatically.
        A supplied module binds Conv2d/ConvTranspose2d weights automatically when
        ``optimizer_convolution='per_offset'``. Bare high-rank tensor iterables
        lack this layout contract and are rejected by DualNorm.

        With ``constraint_geometry_mode != 'none'``, call
        ``particlegan.optim.constraint_geometry.constraint_geometry_backward``
        before ``step()`` with the host's existing protected scalar losses.
        Protection must cover the joint generator/encoder/prior optimizer.
        The default mode requires no protected-loss hook and retains the
        original optimizer type and checkpoint format.
        """
        from torch import nn
        module = params if isinstance(params, nn.Module) else None
        if module is not None:
            params = [p for p in module.parameters() if p.requires_grad]
        options = {"lr": self.lr, "betas": self.betas, "amsgrad": self.amsgrad, "eps": self.eps, **adam_kwargs}
        if self.optimizer_family == "adam":
            from .recipe_schedules import make_plain_adam
            return make_plain_adam(self, params, **options)
        if self.optimizer_family != "formulation":
            from .optim.dualnorm import convolution_parameter_groups, make_normalized_optimizer
            params = list(params)
            if params and not isinstance(params[0], dict):
                params = [{"params": params}]
            groups = []
            direct_ids = set() if direct_particles is None else {id(p) for p in direct_particles}
            for original in params:
                group = dict(original)
                group["params"] = list(group["params"])
                group.setdefault("role", group.get("forge_role", "generator"))
                if latent_table is not None and any(p is latent_table for p in group["params"]):
                    group["role"] = "prior"
                if direct_ids and any(id(p) in direct_ids for p in group["params"]):
                    # Direct generated coordinates are the generator player;
                    # this fixture has no sampled latent table.
                    group["role"] = "generator"
                is_prior = group["role"] in ("prior", "table")
                uses_adam = (self.optimizer_family == "dualnorm_D_only"
                             or (self.optimizer_family == "particle_rownorm_only" and not is_prior))
                if uses_adam and self.optimizer_adam_lr is not None:
                    group["lr"] = self.optimizer_adam_lr * (self.prior_lr_mult if is_prior else 1.)
                groups.append(group)
            if module is not None and self.optimizer_convolution == "per_offset":
                groups = convolution_parameter_groups(groups, [module], family=self.optimizer_family)
            return make_normalized_optimizer(self, groups, **options)
        from .k3p import K3PGeneratorAdam
        return K3PGeneratorAdam(params, latent_table=latent_table, direct_particles=direct_particles,
                                latent_max_rate=self.latent_damping_max_rate,
                                direct_betas=self.direct_particle_betas,
                                direct_gain=self.direct_particle_gain, **options)

    @property
    def effective_critic_formulation(self):
        """Explicit legacy arms retain their K3P optimizer and penalty family."""
        if self.critic_formulation == "bcap":
            return "bcap"
        return "k3p" if self.reg_arm is not None else self.critic_formulation

    @property
    def resolved_network_lr_floor(self):
        """G/D LR floor: ``network_lr_floor`` or ``lr_floor``."""
        return self.lr_floor if self.network_lr_floor is None else self.network_lr_floor

    def make_prior_regularizer(self, **overrides):
        from .vicreg_loss import ParticleRegularizer
        return ParticleRegularizer(**{"weight": self.prior_reg, **overrides})

    def make_optimizers(self, generator, discriminator, prior=None, *, encoder=None, ema_critic=None,
                        require_latent_damping=False, **adam_kwargs):
        """Return ``(opt_g, opt_d)`` selected by ``optimizer_family``.

        ``opt_g`` covers G + optional E + prior (``make_generator_optimizer``;
        a learnable ``ParticlePrior`` table gets A2 damping) and ``opt_d`` the
        critic (``make_critic_optimizer``, with the caller-allocated
        ``ema_critic``). Use them like any Adam: ``zero_grad``/``step``,
        ``state_dict``/``load_state_dict`` (which carry all regularization
        state), LR schedulers. Move modules to their desired device before
        calling. Frozen parameters are excluded, and a Gaussian/frozen prior
        adds no optimizer group. Additional Adam options, such as ``fused`` or
        ``eps``, apply to both optimizers. Set learning rates, betas and
        ``amsgrad`` on the recipe (``amsgrad`` reaches every group: G/E,
        prior and critic). The generator optimizer's groups are ``[generator/encoder,
        prior]`` (either may be absent); scale the prior group by the prior
        multiplier of ``learning_rate_scales`` and everything else by the
        network one.

        Parameters are used as supplied; initialize fresh networks first (e.g.
        ``particlegan.init.deterministic_orthogonal_``).

        ``opt_g.prior_mechanisms`` records prior capabilities and resolved A2
        status. Nonstandardized MoG locations are row-local and support A2;
        standardized reads do not. ``require_latent_damping=True`` rejects an
        unavailable or disabled hook before constructing optimizers. The default
        preserves historical component callers that did not apply A2 to MoG.
        Fixed-penalty normalized families share the caller's training loop and
        schedule; G and E have one global normalization norm, with the prior
        separate. Isolation arms use ``optimizer_adam_lr`` for native Adam
        groups and ``lr`` for normalized groups (the same role multipliers).
        """
        from .capabilities import prior_mechanisms
        prior_params = [] if prior is None else [p for p in prior.parameters() if p.requires_grad]
        prior_ids = {id(p) for p in prior_params}
        g_params = []
        seen = set(prior_ids)
        normalized = self.optimizer_family not in ("adam", "formulation")
        groups = []
        for role, module in (("generator", generator), ("encoder", encoder)):
            role_params = []
            if module is not None:
                for p in module.parameters():
                    if p.requires_grad and id(p) not in seen:
                        g_params.append(p)
                        role_params.append(p)
                        seen.add(id(p))
            if normalized and role_params:
                role_groups = [{"params": role_params, "lr": self.lr, "role": role}]
                if self.optimizer_convolution == "per_offset":
                    from .optim.dualnorm import convolution_parameter_groups
                    role_groups = convolution_parameter_groups(role_groups, [generator, encoder],
                                                               family=self.optimizer_family)
                groups.extend(role_groups)
        if g_params and not normalized:
            groups.append({"params": g_params, "lr": self.lr})
        if prior_params:
            groups.append({"params": prior_params, "lr": self.lr * self.prior_lr_mult,
                           "betas": self.prior_betas if self.prior_betas is not None else self.betas,
                           **({"role": "prior"} if normalized else {}),
                           **({"eps": self.prior_eps} if self.prior_eps is not None else {})})
        mechanisms = prior_mechanisms(prior,
            latent_damping_max_rate=self.latent_damping_max_rate,
            prior_beta1=(self.prior_betas or self.betas)[0])
        if require_latent_damping and not mechanisms["a2"]["enabled"]:
            raise ValueError("required A2 latent damping unavailable: " + mechanisms["a2"]["reason"])
        # Plain and nonstandardized MoG reads have the same row-local gradient
        # ownership. Standardized MoG keeps the component API's historic policy
        # (no A2), now explicit in the optimizer receipt and optionally required.
        latent_table = prior.z if mechanisms["prior"]["a2_eligible"] else None
        opt_g = self.make_generator_optimizer(groups, latent_table=latent_table, **adam_kwargs)
        opt_g.prior_mechanisms = mechanisms
        return opt_g, self.make_critic_optimizer(discriminator, ema_critic=ema_critic, **adam_kwargs)



def get_recipe(name="gan", **overrides):
    """Select a model family or a policy preset with explicit overrides.

    Model families share the default KA2 formulation. ``"e22"`` selects the
    schedule-free DV12/stationarity policy, row evidence, critic-feature
    birth/death, learned output noise and served averaging. Its default
    dimensions are the native 100-Gaussian task's 20,000 particles, latent
    dimension 2 and batch size 2,048. Set ``num_particles``, ``z_dim``,
    ``batch_size`` and ``output_noise_std`` explicitly for another task.
    ``"e22_routed"`` selects the conditional dense-bank ``routed_paired``
    adaptation. It retains E22's controls and requires an explicit RoutedRows
    binding, paired observations and separate guard contexts. Its evidence
    and restructuring law differ from the independent-row formulation.
    ``"atlas"`` adds automatic feature-cell selection (128 cells) and the
    settled optimizer-reopen guard to E22. ``"ka2"`` names the default;
    ``"k3p"`` explicitly selects the earlier critic formulation.
    ``"bcap"`` selects zero-momentum dualnorm with G/E step .012, D step
    .018 and sampled-prior row step .03, non-saturating loss, smoothing .001,
    per-offset convolution updates, fixed real/fake input-gradient caps
    and constant rates. ``"bcap_adam"`` retains the earlier native-Adam
    preset. Both disable the guard, anchor, latent damping, extra
    regularization, EMA serving and additive training noise.
    No research configuration file is read at runtime.

    Use ``Recipe(**saved_fields)`` for resolved checkpoints and
    ``recipe.replace(name=...)`` for custom report labels.
    """
    families = {
        "gan": {},
        "ka2": {},
        "k3p": dict(critic_formulation="k3p"),
        "bcap_adam": dict(critic_formulation="bcap", optimizer_family="adam", reg_arm="b_cap",
                     reg_coeff=1., reg_kappa=1., reg_every=1,
                     lr=.00425, d_lr_mult=1., prior_lr_mult=2., betas=(0., .999),
                     lr_floor=1., network_lr_floor=1., network_lr_horizon_cap=None,
                     beta2_end=None, reg_coeff_end=None, d_guard_ratio=0., reg_anchor_weight=0.,
                     latent_damping_max_rate=0., direct_particle_gain=False,
                     prior_reg=0., ema_decay=0., input_noise_std=0., output_noise_std=0.,
                     output_noise_warmup=0.),
        "e22": dict(continuous_policy="dv12", total_steps=None,
                    lr_control="stationarity", amsgrad=True, reg_coeff=3.0,
                    input_noise_std=0.0, output_noise_warmup=0.0,
                    output_noise_mode="learnable", particle_birth_death=True,
                    row_evidence_gate=True, table_release_rule="anchor",
                    birth_death_space="critic", birth_death_feature_scale="std",
                    birth_death_isolation=True, row_evidence_null="scaled",
                    serve_average=4.0, reopen_signal="optimizer",
                    reopen_anchor="release"),
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
    families["bcap"] = {
        **families["bcap_adam"], "optimizer_family": "dualnorm",
        "lr": .012, "d_lr_mult": 1.5, "prior_lr_mult": 2.5,
        "optimizer_momentum": 0.,
        "loss": "non_saturating", "optimizer_smoothing": .001,
        "optimizer_convolution": "per_offset",
    }
    families["atlas"] = {**families["e22"], "birth_death_backend": "auto",
                         "birth_death_cells": 128, "reopen_guard": "settled"}
    families["e22_routed"] = {**families["e22"], "row_policy": "routed_paired"}
    families["halloween"] = {
        **families["bcap_adam"], "reg_coeff": 0.0,
        "lr": 0.008020980209802098, "d_lr_mult": 0.4922016232592352,
        "betas": (0.6119661196611966, 0.5612256122561226),
        "d_betas": (0.1051710517105171, 0.7203172031720317),
        "eps": 0.3087330873308733, "d_eps": 0.007500075000750007,
        "prior_betas": (0.0, 0.999), "prior_eps": 1e-8,
        "adam_variant": "tensorflow_v1", "loss": "least_squares",
        "loss_labels": (-1.0, 1.0, 1.0), "lr_schedule": "constant",
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
    if floor == 1:
        return 1.0
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


def learning_rate_scales(step, recipe, *, network_transition=None, controller=None):
    """Return ``(network, prior)`` LR multipliers after ``step`` completed updates.

    Generator and critic ("network") follow ``learning_rate_scale`` over
    ``min(total_steps, network_lr_horizon_cap)`` down to ``network_lr_floor``
    and then hold; the particle prior follows it over the full budget down to
    ``lr_floor``. When ``network_transition`` is supplied, G/D instead hold at
    full LR until its caller-marked plateau, then decay over its ``decay_steps``.
    KA2's critic controller is independent of these multipliers.

    A continuous recipe instead reads the current scales from its matching
    ``controller``; ``step`` is ignored. GANTrainer updates that controller
    from ordinary training signals. A custom loop must do that explicitly.
    The network value is the shared base multiplier; DV7 additionally applies
    ``controller.critic_scale()`` to D. Use ``scale_learning_rates(..., critic=)``
    to apply all three parameter roles.
    """
    if getattr(recipe, "lr_control", "mobility") == "stationarity":
        raise ValueError("lr_control='stationarity' requires per-group policy state; use "
                         "E22Policy/UpdatePolicy.begin_step() or GANTrainer instead of learning_rate_scales")
    if recipe.continuous_policy is not None:
        from .continuous import DataDriftController
        if not isinstance(controller, DataDriftController) or controller.variant != recipe.continuous_policy:
            raise ValueError("continuous LR scales require the matching controller; use GANTrainer or pass controller=")
        if network_transition is not None:
            raise ValueError("continuous policies do not accept scheduled network transitions")
        return controller.current_scales()
    if controller is not None:
        raise ValueError("controller requires a continuous recipe")
    if recipe.lr_schedule != "cosine":
        if type(step) is not int or step < 0:
            raise ValueError("LR schedule clock must be completed nonnegative training updates")
        if network_transition is not None:
            raise ValueError("explicit constant/exponential LR does not accept network transitions")
        exponent = step / recipe.lr_decay_steps
        if recipe.lr_decay_staircase:
            exponent = math.floor(exponent)
        scale = recipe.lr_decay_rate**exponent if recipe.lr_schedule == "exponential" else 1.0
        return scale, scale
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


def scale_learning_rates(step, recipe, optimizers, base_rates, prior=None, *,
                         network_transition=None, controller=None, critic=None):
    """Set every group's LR from ``learning_rate_scales(step, recipe)``.

    ``base_rates`` holds each optimizer's unscaled group LRs (read them once
    after construction). Groups whose parameters all belong to ``prior`` get
    the prior multiplier; every other group gets the network one, so a custom
    loop follows the recipe's split network/prior schedules.
    Pass a ``NetworkLRTransition`` to choose the network decay from validation
    while retaining the ordinary prior schedule. Returns ``(network, prior)``
    multipliers. Continuous recipes require their matching ``controller``.
    DV7 also requires ``critic``: its module or the exact optimized critic
    parameters. Critic and prior groups must contain only their own role.
    D receives the additional critic multiplier; the returned pair remains
    the shared network/prior multipliers. This helper never observes data or
    advances the controller: use the same observe_game/observe_real ordering
    as GANTrainer before applying rates.
    """
    network, prior_scale = learning_rate_scales(step, recipe, network_transition=network_transition, controller=controller)
    optimizers = tuple(optimizers)
    from .recipe_schedules import apply_training_schedules
    apply_training_schedules(step, recipe, optimizers)
    prior_ids = set() if prior is None else {id(p) for p in prior.parameters()}
    if recipe.continuous_policy in ("dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
        from torch import Tensor, nn
        if critic is None:
            raise ValueError("asymmetric continuous rates require critic=module or optimized critic parameters")
        try:
            parameters = list(critic.parameters() if isinstance(critic, nn.Module) else critic)
        except TypeError as error:
            raise TypeError("critic must be a module or iterable of parameters") from error
        if not parameters or any(not isinstance(p, Tensor) for p in parameters):
            raise ValueError("critic must identify nonempty optimized parameters")
        critic_ids = {id(p) for p in parameters}
        if critic_ids & prior_ids:
            raise ValueError("critic and prior parameter roles overlap")
        optimizers, base_rates = tuple(optimizers), tuple(base_rates)
        if len(optimizers) != len(base_rates):
            raise ValueError("base_rates must match every optimizer")
        pending, found = [], set()
        critic_scale = controller.critic_scale() if recipe.critic_payoff_damping else 1.
        for optimizer, rates in zip(optimizers, base_rates):
            if len(optimizer.param_groups) != len(rates):
                raise ValueError("base_rates must match every optimizer group")
            for group, rate in zip(optimizer.param_groups, rates):
                ids = {id(p) for p in group["params"]}
                if not ids or (ids & critic_ids and not ids <= critic_ids):
                    raise ValueError("critic parameters must occupy complete nonempty groups")
                if ids & prior_ids and not ids <= prior_ids:
                    raise ValueError("prior parameters must occupy complete groups")
                if getattr(optimizer, "critic", None) is not None and not ids <= critic_ids:
                    raise ValueError("critic role does not match the critic optimizer")
                found.update(ids & critic_ids)
                value = rate * (prior_scale if ids <= prior_ids else network)
                if ids <= critic_ids:
                    value *= critic_scale
                pending.append((group, value))
        if found != critic_ids:
            raise ValueError("critic parameters must all belong to the supplied optimizers")
        for group, value in pending:
            group["lr"] = value
        return network, prior_scale
    for optimizer, rates in zip(optimizers, base_rates):
        for group, rate in zip(optimizer.param_groups, rates):
            is_prior = bool(prior_ids) and all(id(p) in prior_ids for p in group["params"])
            group["lr"] = rate * (prior_scale if is_prior else network)
    return network, prior_scale
