"""Opt-in generator idle rule: skip the G step while the critic gap is unresolved.

``Recipe(generator_idle_se=k)`` wraps the generator optimizer. Before each
``step()`` the caller records the critic's paired real/fake scores, either
through ``Recipe.make_loss(generator_optimizer=opt_g)`` or explicitly with
``opt_g.record_critic_gap``. When the batch mean of ``real - fake`` is within
``k`` unbiased standard errors of zero, ``step()`` returns without touching any
parameter, moment, prior row or pending row ownership. The critic still steps.
``generator_idle_se=None`` (the default) constructs the unwrapped optimizer.
"""
import math

import torch

from ..gan_loss import GANLoss


def generator_is_idle(real_logits, fake_logits, standard_errors):
    """Whether the paired real-fake critic gap is unresolved at this threshold.

    Idle iff ``|mean(real - fake)| <= standard_errors * SE`` with the unbiased
    float64 standard error. Fewer than two pairs or a non-finite statistic is
    not idle. A constant nonzero gap has SE zero, so it is not idle; a constant
    zero gap is idle.
    """
    gap = (real_logits.detach().reshape(-1).to(torch.float64)
           - fake_logits.detach().reshape(-1).to(torch.float64))
    count = gap.numel()
    if count < 2:
        return False
    mean, variance = gap.mean(), gap.var(unbiased=True)
    if not bool(torch.isfinite(mean)) or not bool(torch.isfinite(variance)):
        return False
    standard_error = torch.sqrt(variance / count)
    return bool(mean.abs() <= float(standard_errors) * standard_error)


class _GeneratorIdle:
    """Mixin over a recipe generator optimizer. See the module docstring."""

    def _init_generator_idle(self, standard_errors):
        if (isinstance(standard_errors, bool) or type(standard_errors) not in (int, float)
                or not math.isfinite(standard_errors) or standard_errors < 0):
            raise ValueError("generator_idle_se must be a finite nonnegative number")
        self.generator_idle_se = float(standard_errors)
        self.idle_steps = 0
        self.idle_decisions = 0
        self.last_idle = None
        self._idle_gaps = []
        self._idle_in_step = False

    def record_critic_gap(self, real_logits, fake_logits):
        """Record this G phase's paired scores; several calls pool their pairs.

        Calls made while ``step()`` runs (for example finite-probe evaluators)
        are ignored, so a decision only uses scores from before the update.
        """
        if self._idle_in_step:
            return
        if real_logits is None:
            raise ValueError("generator idle rule requires real critic scores")
        real = real_logits.detach().reshape(-1)
        fake = fake_logits.detach().reshape(-1)
        if real.numel() != fake.numel():
            raise ValueError("generator idle rule requires paired real and fake scores")
        self._idle_gaps.append(real.to(torch.float64) - fake.to(torch.float64))

    def _discard_pending_update(self):
        clear = getattr(self, "clear_sampled_rows", None)
        if clear is not None:
            clear()
        if hasattr(self, "_protected"):
            self._protected = None

    def step(self, closure=None):
        if not self._idle_gaps:
            raise RuntimeError("generator idle rule requires record_critic_gap before step(); "
                               "bind it with Recipe.make_loss(generator_optimizer=opt_g)")
        gap = torch.cat(self._idle_gaps)
        self._idle_gaps = []
        idle = generator_is_idle(gap, torch.zeros_like(gap), self.generator_idle_se)
        self.idle_decisions += 1
        self.last_idle = idle
        if idle:
            self.idle_steps += 1
            self._discard_pending_update()
            return None
        self._idle_in_step = True
        try:
            return super().step() if closure is None else super().step(closure)
        finally:
            self._idle_in_step = False

    def _idle_meta(self):
        return dict(schema=1, standard_errors=self.generator_idle_se, idle_steps=self.idle_steps,
                    decisions=self.idle_decisions, last_idle=self.last_idle)

    def state_dict(self):
        saved = super().state_dict()
        saved["generator_idle"] = self._idle_meta()
        return saved

    def _check_idle_meta(self, meta):
        if (not isinstance(meta, dict) or set(meta) != set(self._idle_meta())
                or meta["schema"] != 1 or meta["standard_errors"] != self.generator_idle_se
                or type(meta["idle_steps"]) is not int or type(meta["decisions"]) is not int
                or not 0 <= meta["idle_steps"] <= meta["decisions"]
                or meta["last_idle"] not in (None, True, False)):
            raise ValueError("invalid generator idle checkpoint")

    def validate_state_dict(self, saved):
        base = dict(saved)
        meta = base.pop("generator_idle", None)
        if meta is not None:
            self._check_idle_meta(meta)
        parent = getattr(super(), "validate_state_dict", None)
        if parent is not None:
            parent(base)

    def load_state_dict(self, saved):
        if not isinstance(saved, dict):
            raise ValueError("invalid generator idle checkpoint")
        base = dict(saved)
        meta = base.pop("generator_idle", None)
        self._check_idle_meta(meta)
        super().load_state_dict(base)
        self.idle_steps, self.idle_decisions = meta["idle_steps"], meta["decisions"]
        self.last_idle = meta["last_idle"]
        self._idle_gaps = []


# torch.optim.Optimizer re-wraps ``type(self).step`` with its hook profiler on
# deepcopy/unpickle; the base step already carries those hooks.
_GeneratorIdle.step.hooked = True


class IdleRecordingLoss(GANLoss):
    """``recipe.make_loss(generator_optimizer=opt_g)`` with the idle rule on.

    Same objective values as the unbound loss; each generator call also
    records its paired scores on ``opt_g`` and therefore requires real scores.
    """

    def __init__(self, loss, optimizer):
        if not hasattr(optimizer, "record_critic_gap"):
            raise ValueError("generator_optimizer must come from this recipe's generator optimizer factory")
        super().__init__(loss.loss, labels=loss.labels)
        self._plain, self.generator_optimizer = loss, optimizer

    def g_loss(self, fake_logits, real_logits=None):
        self.generator_optimizer.record_critic_gap(real_logits, fake_logits)
        return self._plain.g_loss(fake_logits, real_logits)

    def joint_g_loss(self, fake_logits, real_logits):
        self.generator_optimizer.record_critic_gap(real_logits, fake_logits)
        return self._plain.joint_g_loss(fake_logits, real_logits)


_CLASSES = {}


def generator_idle_optimizer_class(base_class):
    """The idle-rule subclass of a generator optimizer class (cached)."""
    if issubclass(base_class, _GeneratorIdle):
        raise ValueError("generator idle rule is already applied")
    if base_class not in _CLASSES:
        _CLASSES[base_class] = type(f"GeneratorIdle{base_class.__name__}", (_GeneratorIdle, base_class),
                                    {"__module__": __name__, "__doc__": base_class.__doc__})
    return _CLASSES[base_class]


def enable_generator_idle(optimizer, standard_errors):
    """Apply the rule to a freshly constructed recipe generator optimizer."""
    if getattr(optimizer, "state", None):
        raise ValueError("enable the generator idle rule before any optimizer update")
    optimizer.__class__ = generator_idle_optimizer_class(type(optimizer))
    optimizer._init_generator_idle(standard_errors)
    return optimizer


def generator_idle_counts(optimizers):
    """(idle_steps, decisions) shared by every wrapped part, or None when off."""
    counts = {(o.idle_steps, o.idle_decisions) for o in optimizers if isinstance(o, _GeneratorIdle)}
    if not counts:
        return None
    if len(counts) != 1:
        raise RuntimeError("generator optimizer parts disagree on idle decisions")
    return counts.pop()
