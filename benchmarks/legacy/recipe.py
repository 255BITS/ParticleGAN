"""Pinned pre-removal ``Recipe`` with the formulation switches benchmarks replay.

``LegacyRecipe`` is ``particlegan.Recipe`` plus the fields ParticleGAN no
longer ships (``loss_type``, ``gan_mode``, ``reg_arm``, ``reg_method``) and
the factories that honored them, built from the pinned copies in this
package. With default switches it trains exactly like ``particlegan``'s
recipe; archived GAN v3 / locked_shared / arm-study configurations resolve
through it so their receipts stay reproducible. Benchmarks only.
"""
from copy import copy, deepcopy
from dataclasses import dataclass, fields

import torch.nn as nn

from particlegan import Recipe
from particlegan.k3p import K3PCriticAdam

from .critic_optimizer import LegacyCriticAdam
from .gan_loss import GANLoss
from .grad_regularizers import GradientPenalty

__all__ = ["LegacyRecipe", "get_recipe", "CriticPenalty"]

# Field order of the recipe these receipts were recorded with.
_RECORDED_ORDER = (
    'name', 'model', 'z_dim', 'num_particles', 'prior_kind', 'sigma_rel', 'standardize', 'num_classes',
    'conditioning', 'ucd_target', 'ucd_weight', 'alpha_bar', 'batch_size', 'total_steps', 'lr', 'd_lr_mult',
    'prior_lr_mult', 'betas', 'prior_betas', 'loss_type', 'gan_mode', 'reg_arm', 'reg_coeff', 'reg_kappa',
    'reg_every', 'reg_method', 'prior_reg', 'ema_decay', 'lr_anneal_start', 'lr_floor', 'network_lr_floor',
    'network_lr_horizon_cap', 'reg_anchor_decay', 'd_guard_ratio', 'd_guard_min_steps',
    'latent_damping_max_rate', 'direct_particle_betas', 'input_noise_std', 'input_noise_anneal_end',
    'output_noise_std', 'output_noise_warmup', 'encoder_mode', 'routing_temperature', 'distance_reduction',
    'observation_sigma', 'reconstruction_weight')
# Fields added after those receipts, with the values that reproduce them.
_ADDED = {"reg_anchor_weight": 1.0, "direct_particle_gain": True, "amsgrad": False}


@dataclass(frozen=True)
class LegacyRecipe(Recipe):
    # The package defaults these receipts were recorded with (the package has
    # since moved to constant-LR AMSGrad without noise; see docs/k3p.md).
    lr: float = 0.00425
    d_lr_mult: float = 1.0
    reg_coeff: float = 1.0
    lr_floor: float = 0.05
    network_lr_floor: float | None = 0.01
    network_lr_horizon_cap: int | None = 1600
    input_noise_std: float = 0.5
    output_noise_std: float = 0.029
    amsgrad: bool = False
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    reg_arm: str = "k3p"
    reg_method: str = "autograd"
    # Removed from the package with the direct sample-particle response (no
    # benchmark used it); kept so recorded dicts round-trip.
    direct_particle_gain: bool = True
    direct_particle_betas: tuple[float, float] = (0.0, 0.9)
    # Removed from the package with the EMA-critic anchor. The legacy K3P
    # penalty evaluates the EMA critic that LegacyCriticAdam keeps at this
    # decay (reg_anchor_weight only round-trips recorded dicts).
    reg_anchor_decay: float = 0.999
    reg_anchor_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "direct_particle_betas", tuple(float(b) for b in self.direct_particle_betas))
        if len(self.direct_particle_betas) != 2 or any(not 0 <= b < 1 for b in self.direct_particle_betas):
            raise ValueError("direct_particle_betas must contain two values in [0, 1)")
        if type(self.direct_particle_gain) is not bool:
            raise ValueError("direct_particle_gain must be a boolean")
        if isinstance(self.reg_anchor_decay, bool) or not 0 <= self.reg_anchor_decay < 1:
            raise ValueError("reg_anchor_decay must be in [0, 1)")
        self.make_loss()
        self.make_gradient_penalty()

    def to_dict(self):
        """The recorded dict: recorded field order; added fields only when changed."""
        values = super().to_dict()
        out = {key: values[key] for key in _RECORDED_ORDER}
        out.update({key: values[key] for key, neutral in _ADDED.items() if values[key] != neutral})
        return out

    def make_loss(self, **overrides):
        return GANLoss(**{"loss_type": self.loss_type, "mode": self.gan_mode, **overrides})

    def make_gradient_penalty(self, **overrides):
        options = {"arm": self.reg_arm, "coeff": self.reg_coeff,
                   "kappa": self.reg_kappa, "lazy_k": self.reg_every,
                   "method": self.reg_method, **overrides}
        if options["arm"] == "k3p":
            floor = self.resolved_network_lr_floor
            options.setdefault("lr_floor", floor if floor < 0.5 else 0.0)
        return GradientPenalty(**options)

    def make_critic_penalty(self, optimizer, *, output=None, generator=None, collect_stats=False,
                            **penalty_overrides):
        return CriticPenalty(self, optimizer, output=output, generator=generator,
                             collect_stats=collect_stats, **penalty_overrides)

    def make_critic_optimizer(self, critic, *, ema_critic=None, **adam_kwargs):
        """The package critic optimizer plus the EMA critic the legacy K3P penalty reads.

        A K3P recipe allocates ``copy.deepcopy(critic)`` as the EMA when
        ``ema_critic`` is not given (GANTrainer and the package factories no
        longer pass one).
        """
        if ema_critic is None and self.reg_arm == "k3p":
            ema_critic = deepcopy(critic)
        options = {"lr": self.lr * self.d_lr_mult, "betas": self.betas, "amsgrad": self.amsgrad,
                   **adam_kwargs}
        return LegacyCriticAdam([p for p in critic.parameters() if p.requires_grad], critic=critic,
                                ema_critic=ema_critic, anchor_decay=self.reg_anchor_decay,
                                guard_ratio=self.d_guard_ratio, guard_min_steps=self.d_guard_min_steps,
                                **options)

    def make_optimizers(self, generator, discriminator, prior=None, *, encoder=None, ema_critic=None,
                        **adam_kwargs):
        """``Recipe.make_optimizers`` with ``make_critic_optimizer``'s EMA critic (``ema_critic`` optional)."""
        opt_g, opt_d = super().make_optimizers(generator, discriminator, prior, encoder=encoder, **adam_kwargs)
        if ema_critic is not None:
            opt_d = self.make_critic_optimizer(discriminator, ema_critic=ema_critic, **adam_kwargs)
        return opt_g, opt_d


def get_recipe(name="gan", **overrides):
    """``particlegan.get_recipe`` returning a ``LegacyRecipe`` (on the pinned legacy defaults)."""
    from particlegan import get_recipe as current
    base, plain = current(name), Recipe()
    family = {f.name: getattr(base, f.name) for f in fields(Recipe)
              if getattr(base, f.name) != getattr(plain, f.name)}
    return LegacyRecipe(**{**family, **overrides})


def _first_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


class CriticPenalty:
    """The recipe's critic penalty, paired with one critic optimizer.

    Built by ``recipe.make_critic_penalty(opt_d)``; call it like a loss::

        d_loss = adv + penalty(D, real, fake)                      # plain critic
        d_loss = adv + penalty(D, x, fake, labels, xt=xt, t=t)     # conditional critic

    Extra positional/keyword arguments are forwarded as conditioning to the
    critic and to the paired EMA critic. ``D`` is the optimizer's critic, one of
    its submodules (a role of a shared module; the same-named EMA submodule is
    used), or a module wrapping one of those (e.g. ``InputNoise(D)``; the EMA
    is evaluated through a shallow copy of the wrapper). A tuple/list output
    uses its first element unless ``output=`` selects the logits. The step
    used for lazy application is the optimizer's completed step count + 1, so
    several calls per critic step (roles, views) share one step. Returns the
    scalar penalty; ``last_stats`` holds the stats of the last call when
    ``collect_stats`` is set, ``diagnostics()`` host scalars.
    """

    def __init__(self, recipe, optimizer, *, output=None, generator=None, collect_stats=False,
                 **penalty_overrides):
        if not isinstance(optimizer, K3PCriticAdam):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")
        self.optimizer, self.critic = optimizer, optimizer.critic
        k3p = penalty_overrides.get("arm", recipe.reg_arm) == "k3p"
        if k3p and getattr(optimizer, "anchor", None) is None:
            raise ValueError("this penalty needs the critic's EMA: pass ema_critic=copy.deepcopy(critic) "
                             "to recipe.make_optimizers / recipe.make_critic_optimizer")
        options = {"record": optimizer.record} if k3p else {}
        self.regularizer = recipe.make_gradient_penalty(**options, **penalty_overrides)
        self.output = _first_output if output is None else output
        self.generator, self.collect_stats = generator, bool(collect_stats)
        self.last_stats = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}

    @property
    def ema_critic(self):
        """The paired optimizer's EMA critic module (None without one)."""
        return getattr(self.optimizer, "ema_critic", None)

    def _ema_view(self, critic):
        """``m -> module`` mapping the EMA root to the EMA counterpart of ``critic``."""
        if isinstance(critic, nn.Module):
            name = self._names.get(id(critic))
            if name is not None:
                return lambda m: m.get_submodule(name)
            for key, child in critic._modules.items():
                inner = None if child is None else self._names.get(id(child))
                if inner is not None:
                    def view(m, key=key, inner=inner):
                        clone = copy(critic)
                        clone._modules = dict(critic._modules)
                        clone._modules[key] = m.get_submodule(inner)
                        return clone
                    return view
        raise TypeError("pass the critic paired with this penalty's optimizer, one of its submodules, "
                        "or a module wrapping one of those")

    def __call__(self, critic, x_real, x_fake, *condition, **condition_kwargs):
        output = self.output

        def live(x):
            return output(critic(x, *condition, **condition_kwargs))
        options = {}
        anchor = getattr(self.optimizer, "anchor", None)
        if anchor is not None and self.regularizer.arm == "k3p":
            view = self._ema_view(critic)
            options["ema_critic"] = lambda x: anchor.forward(
                lambda m, inputs: output(view(m)(inputs, *condition, **condition_kwargs)), x)
        step = self.optimizer.record.observed_steps + 1
        penalty, stats = self.regularizer.penalty(live, x_real, x_fake, step, self.generator,
                                                  self.collect_stats, **options)
        self.last_stats = stats
        return penalty

    def diagnostics(self):
        """Host-side scalars for logging (K3P: blend weight; guard clip count)."""
        out = {}
        if self.regularizer.arm == "k3p":
            out["blend_weight"] = float(self.regularizer.blend_weight())
        if self.optimizer.guard is not None:
            out["clipped_tensors"] = self.optimizer.guard.clipped_tensors
        return out
