"""DV12: the state-driven learning-rate controller shared by the recipe optimizers.

Every update each role trains at a fraction of its peak (the param group's
``lr``), computed from training signals only -- there is no horizon, no
schedule and no step clock::

    G     = lr_G     * (.01 + .99 m) * gt
    prior = lr_prior * (.05 + .95 m) * gt
    D     = lr_D     * (.01 + .99 m) * gt / (1 + pe^2)

* ``pe`` (payoff error): EMA (.02) of how far the generator is losing to the
  critic, ``max(0, g_loss - d_loss) / log 2``, read from the recipe's loss.
* ``data_drive``: RMS z-score of the drift between fast (.1) and slow (.01)
  EMA means of fixed random Fourier features of the real batches, mapped to
  [0, 1] as ``clip((z - 3) / 3, 0, 1)``. Real statistics only set this
  scalar; they never enter a loss, sample or parameter.
* ``m`` (mobility) moves toward ``max(data_drive, min(1, pe^2))``, rising at
  rate .05 and falling at .005.
* ``gt`` (game trust) shrinks all rates when the critic's Adam moment surprise
  (from the KA2 record) rises above its birth baseline without data movement.

Users never call this directly. ``recipe.make_optimizers`` builds one
controller per (generator, critic) pair: the critic penalty feeds it the real
batch, the recipe's loss feeds it the payoff, and the optimizers' ``step()``
apply its rates. Its state travels with the critic optimizer's
``state_dict()``.
"""
from copy import deepcopy
import math
import warnings

import torch


class DV12Controller:
    """Mobility / game-trust / payoff state and the per-role LR fractions it implies."""

    _TENSORS = ("location", "scale", "projection", "reference", "fast_reference",
                "fast_mean_variance", "slow_mean_variance", "mean_covariance")
    _FLOATS = ("mobility", "game_trust", "game_ratio", "payoff_error", "data_score", "data_drive")
    KEYS = _TENSORS + _FLOATS + ("updates", "observed_at", "network_scale", "prior_scale")

    def __init__(self):
        self.mobility = 1.0
        self.game_trust = 1.0
        self.game_ratio = 1.0
        self.payoff_error = 0.0
        self.data_score = 0.0
        self.data_drive = 0.0
        for key in self._TENSORS:
            setattr(self, key, None)
        self.updates = 0
        # Critic step count at which the current rates were observed (-1: none).
        self.observed_at = -1
        self.network_scale = self.prior_scale = 1.0
        # Per-step losses for the payoff (not checkpointed: consumed within a step).
        self._critic_loss = self._penalty = self._generator_loss = None
        self._generator_pending = False
        self._warned = False

    # -- per-step observations -------------------------------------------------

    def begin_critic_step(self, real, record):
        """Observe one critic step's real batch (once per step; the penalty calls this)."""
        if self.observed_at == record.observed_steps:
            return
        self._observe_game(record)
        self.network_scale, self.prior_scale = self._observe_real(real)
        self.observed_at = record.observed_steps
        self._generator_pending = True

    def _observe_game(self, record):
        ratio = 1.0
        if record.sur_base and record.last_sur is not None:
            ratio = record.last_sur / record.sur_base
        self.game_ratio = ratio
        unexplained = max(0., ratio - 1.) * (1. - self.data_drive)
        self.game_trust = 1. / (1. + unexplained * unexplained)

    @torch.no_grad()
    def _observe_real(self, real):
        x = real.detach().flatten(1)
        if self.location is None:
            self.location = x.mean(0)
            self.scale = x.std(0, unbiased=False).clamp_min(1e-6)
            # A private stream: the features never advance training RNG.
            stream = torch.Generator(device="cpu").manual_seed(1729)
            self.projection = (torch.randn(x.shape[1], 32, generator=stream, device="cpu")
                               / math.sqrt(x.shape[1])).to(x)
        u = ((x - self.location) / self.scale) @ self.projection
        features = torch.cat((u, torch.cos(u), torch.sin(u), torch.cos(2*u), torch.sin(2*u)), 1)
        mean = features.mean(0)
        var = features.var(0, unbiased=False).clamp_min(1e-6)
        if self.reference is None:
            self.reference = mean.clone()
        batch_variance = var / len(x)
        if self.fast_reference is None:
            self.fast_reference = mean.clone()
            self.fast_mean_variance = batch_variance.clone()
            self.slow_mean_variance = batch_variance.clone()
            self.mean_covariance = batch_variance.clone()
        self.fast_reference.lerp_(mean, .1)
        self.reference.lerp_(mean, .01)
        self.fast_mean_variance.mul_(.9**2).add_(batch_variance, alpha=.1**2)
        self.slow_mean_variance.mul_(.99**2).add_(batch_variance, alpha=.01**2)
        self.mean_covariance.mul_(.9*.99).add_(batch_variance, alpha=.1*.01)
        variance_of_difference = (self.fast_mean_variance + self.slow_mean_variance
                                  - 2*self.mean_covariance).clamp_min(1e-10)
        self.data_score = float(((self.fast_reference - self.reference).square()
                                 / variance_of_difference).mean().sqrt())
        self.data_drive = min(1., max(0., (self.data_score - 3.) / 3.))
        target = max(self.data_drive, min(1., self.payoff_error ** 2))
        speed = .05 if target > self.mobility else .005
        self.mobility += speed * (target - self.mobility)
        self.updates += 1
        return ((.01 + .99 * self.mobility) * self.game_trust,
                (.05 + .95 * self.mobility) * self.game_trust)

    def critic_scale(self):
        """Extra critic fraction ``1 / (1 + pe^2)``: a losing generator slows the critic."""
        return 1. / (1. + self.payoff_error ** 2)

    def record_critic_loss(self, value):
        self._critic_loss = value.detach()

    def record_penalty(self, value):
        self._penalty = value.detach()

    def record_generator_loss(self, value):
        self._generator_loss = value.detach()

    def observe_payoff(self):
        """Update ``pe`` from this step's losses (the generator optimizer calls this)."""
        if self._critic_loss is None or self._generator_loss is None:
            raise RuntimeError(
                "the recipe's LR controller reads the game payoff from its loss: build it with "
                "recipe.make_loss(opt_d) and use loss.d_loss and loss.g_loss in every step")
        critic = self._critic_loss
        if self._penalty is not None:
            # The critic's adversarial term measured as (total - penalty), as
            # the reference trainer did; bit-identical to the recorded runs.
            critic = (critic + self._penalty) - self._penalty
        error = max(0., float(self._generator_loss - critic) / math.log(2.))
        self.payoff_error += .02 * (error - self.payoff_error)
        self._critic_loss = self._penalty = self._generator_loss = None

    def critic_step_rates(self, record):
        """``(network, critic)`` fractions for the critic step about to run.

        A critic step the penalty did not see (no real batch observed) keeps
        the current rates and warns once: the shipped update calls
        ``recipe.make_critic_penalty(opt_d)`` every critic step.
        """
        if self.observed_at != record.observed_steps and not self._warned:
            self._warned = True
            warnings.warn("a recipe critic optimizer stepped without its critic penalty; the LR "
                          "controller keeps its last rates (call recipe.make_critic_penalty(opt_d) "
                          "every critic step)", RuntimeWarning, stacklevel=3)
        return self.network_scale, self.critic_scale()

    def take_generator_step(self):
        """Rates for one generator step; reads the payoff when it follows a critic step.

        A generator step without a new critic step (e.g. a generator-only
        phase) reuses the current rates and observes no payoff.
        """
        if self._generator_pending:
            self._generator_pending = False
            self.observe_payoff()
        return self.network_scale, self.prior_scale

    # -- reporting and checkpoints ---------------------------------------------

    def diagnostics(self):
        return {"mobility": self.mobility, "game_trust": self.game_trust, "game_ratio": self.game_ratio,
                "payoff_error": self.payoff_error, "data_score": self.data_score,
                "data_drive": self.data_drive, "network_lr_scale": self.network_scale,
                "prior_lr_scale": self.prior_scale, "critic_lr_scale": self.network_scale * self.critic_scale()}

    def state_dict(self):
        state = {key: getattr(self, key) for key in self.KEYS}
        state["generator_pending"] = self._generator_pending
        return deepcopy(state)

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != set(self.KEYS) | {"generator_pending"}:
            raise ValueError("invalid LR controller state")
        for key in self._TENSORS:
            if state[key] is not None and not isinstance(state[key], torch.Tensor):
                raise ValueError(f"invalid LR controller {key}")
        for key in (*self._FLOATS, "network_scale", "prior_scale"):
            value = state[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid LR controller {key}")
        if type(state["updates"]) is not int or type(state["observed_at"]) is not int:
            raise ValueError("invalid LR controller counters")
        state = deepcopy(state)
        self._generator_pending = bool(state.pop("generator_pending"))
        for key, value in state.items():
            setattr(self, key, value)
