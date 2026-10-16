"""Explicit recipe schedules and observation-only bookkeeping for plain Adam.

The schedules use completed optimizer updates against the declared recipe
horizon. Sampling, penalty calls and externally shortened execution budgets do
not advance or rescale that horizon. Plain Adam uses the declared update law;
its critic observer only counts steps for penalties and checkpoint receipts.
"""
from copy import copy
import math

import torch

from .grad_regularizers import CriticStepRecord


def cosine_value(completed_steps, start, end, anneal_end, total_steps):
    """Cosine interpolation over a declared initial fraction, then hold."""
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("completed_steps must be a nonnegative integer")
    fraction = min(1.0, completed_steps / (anneal_end * total_steps))
    return float(end + (start - end) * .5 * (1.0 + math.cos(math.pi * fraction)))


def apply_optimizer_schedule(completed_steps, recipe, optimizer):
    """Apply configured beta2 to every group before its next Adam update."""
    if getattr(recipe, "beta2_end", None) is None:
        return
    for group in optimizer.param_groups:
        initial = group.get("_recipe_initial_betas")
        if initial is None:
            raise ValueError("scheduled beta2 requires an optimizer from the recipe factories")
        group["betas"] = (initial[0], cosine_value(completed_steps, initial[1], recipe.beta2_end,
                                                 recipe.beta2_anneal_end, recipe.total_steps))


def apply_penalty_schedule(completed_steps, recipe, penalty):
    """Set the fixed penalty coefficient before evaluating its loss."""
    if getattr(recipe, "reg_coeff_end", None) is None or penalty is None:
        return
    penalty.regularizer.coeff = cosine_value(
        completed_steps, penalty.initial_coeff, recipe.reg_coeff_end,
        recipe.reg_coeff_anneal_end, recipe.total_steps)


def apply_training_schedules(completed_steps, recipe, optimizers, penalty=None):
    """Shared schedule boundary for public policy and caller-owned loops."""
    for optimizer in optimizers:
        apply_optimizer_schedule(completed_steps, recipe, optimizer)
    apply_penalty_schedule(completed_steps, recipe, penalty)


def _observe_critic_step(optimizer, args, kwargs):
    optimizer.record.record_step(optimizer)


def _save_critic_observer(optimizer, state):
    state["regularizer"] = {"optimizer_family": "adam", "record": optimizer.record.state_dict(),
                            "ema": None, "guard": None}


def validate_plain_adam_state(optimizer, state):
    """Validate observer/schedule metadata without mutating an optimizer.

    Native Adam's deepcopy retains only its standard tensors and groups. Public
    checkpoint preflight therefore checks these extras on the live observer
    before making that temporary copy.
    """
    if getattr(optimizer, "recipe_optimizer_family", None) != "adam":
        return
    if not isinstance(state, dict):
        raise ValueError("invalid plain Adam checkpoint")
    if type(optimizer) is torch.optim.Adam and "tensorflow_v1" in state:
        raise ValueError("TensorFlow-v1 checkpoint cannot load into native PyTorch Adam")
    if hasattr(optimizer, "record"):
        extra = state.get("regularizer")
        if (not isinstance(extra, dict) or set(extra) != {"optimizer_family", "record", "ema", "guard"}
                or extra["optimizer_family"] != "adam" or extra["ema"] is not None or extra["guard"] is not None):
            raise ValueError("invalid plain Adam critic observer checkpoint")
        record = extra["record"]
        if not isinstance(record, dict) or set(record) != set(CriticStepRecord.KEYS):
            raise ValueError("invalid plain Adam critic step record")
        for key in ("calls", "observed_steps"):
            if type(record[key]) is not int or record[key] < 0:
                raise ValueError("invalid plain Adam critic step count")
        if record["anchor_started"] is not False:
            raise ValueError("plain Adam cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            value = record[key]
            if key == "lr_last" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid plain Adam critic LR record")
        copy(optimizer.record).load_state_dict(record)
    groups = state.get("param_groups", [])
    for actual, saved in zip(optimizer.param_groups, groups):
        if ("_recipe_initial_betas" in actual
                and tuple(saved.get("_recipe_initial_betas", ())) != actual["_recipe_initial_betas"]):
            raise ValueError("plain Adam beta2 schedule initialization differs from its recipe")


def _load_critic_before(optimizer, state):
    validate_plain_adam_state(optimizer, state)
    state = dict(state)
    optimizer._plain_pending_record = state.pop("regularizer")["record"]
    return state


def _load_native_before(optimizer, state):
    validate_plain_adam_state(optimizer, state)


def _load_critic_after(optimizer):
    optimizer.record.load_state_dict(optimizer.__dict__.pop("_plain_pending_record"))


def make_plain_adam(recipe, params, *, critic=None, **options):
    """Declared Adam law with disabled interventions and a critic observer."""
    if recipe.adam_variant == "tensorflow_v1":
        from .tensorflow_adam import TensorFlowV1Adam
        optimizer = TensorFlowV1Adam(params, **options)
    else:
        optimizer = torch.optim.Adam(params, **options)
    optimizer.recipe_optimizer_family = "adam"
    if type(optimizer) is torch.optim.Adam:
        optimizer.register_load_state_dict_pre_hook(_load_native_before)
    if recipe.beta2_end is not None:
        for group in optimizer.param_groups:
            group["_recipe_initial_betas"] = tuple(group["betas"])
    if critic is None:
        optimizer.latent_damping = optimizer.latent_history = None
        optimizer.direct_response = optimizer.direct_history = None
    else:
        optimizer.critic = critic
        optimizer.ema_critic = optimizer.anchor = optimizer.guard = None
        optimizer.record = CriticStepRecord()
        optimizer.register_step_post_hook(_observe_critic_step)
        optimizer.register_state_dict_post_hook(_save_critic_observer)
        optimizer.register_load_state_dict_pre_hook(_load_critic_before)
        optimizer.register_load_state_dict_post_hook(_load_critic_after)
    return optimizer


def progress_fraction_scale(progress, start, floor):
    """The scheduled cosine LR shape over a progress fraction instead of step/horizon."""
    if floor == 1:
        return 1.0
    fraction = min(1.0, max(0.0, (progress - start) / (1 - start)))
    return float(floor + (1 - floor) * .5 * (1 + math.cos(math.pi * fraction)))


class ProgressClock:
    """Opt-in, horizon-free schedule driver (``Recipe.progress_schedule``).

    ``progress`` replaces ``completed_steps / total_steps`` for the scheduled
    LR, input-noise and output-noise shapes. It depends only on observed
    critic outputs and its own saved state, never on the step label, the
    declared horizon or evaluation calls. It is monotone: annealing is never
    undone. ``critic_order`` advances by ``rate * clamp((target - a) /
    (target - 0.5), 0, 1)`` where ``a`` is an EMA of the paired order accuracy
    mean[D(real_i) > D(fake_i)]; ``updates`` advances by ``rate`` every update
    (a diagnostic internal counter).
    """

    KEYS = ("progress", "order_ema", "observed")

    def __init__(self, recipe):
        self.mode, self.rate = recipe.progress_schedule, float(recipe.progress_rate)
        self.target, self.decay = float(recipe.progress_target), float(recipe.progress_ema)
        self.progress, self.order_ema, self.observed = 0.0, None, 0

    def observe(self, real_logits, fake_logits):
        real = real_logits.detach().reshape(len(real_logits), -1).mean(1)
        fake = fake_logits.detach().reshape(len(fake_logits), -1).mean(1)
        count = min(len(real), len(fake))
        accuracy = float((real[:count] > fake[:count]).float().mean())
        self.order_ema = (accuracy if self.order_ema is None
                          else self.decay * self.order_ema + (1 - self.decay) * accuracy)
        gate = (1.0 if self.mode == "updates"
                else min(1.0, max(0.0, (self.target - self.order_ema) / (self.target - .5))))
        self.progress = min(1.0, self.progress + self.rate * gate)
        self.observed += 1

    def scales(self, recipe):
        """(network, prior) LR multipliers at the current progress."""
        return (progress_fraction_scale(self.progress, recipe.lr_anneal_start, recipe.resolved_network_lr_floor),
                progress_fraction_scale(self.progress, recipe.lr_anneal_start, recipe.lr_floor))

    def input_noise(self, recipe):
        return float(recipe.input_noise_std * max(0.0, 1.0 - self.progress / recipe.input_noise_anneal_end))

    def output_noise(self, recipe):
        if recipe.output_noise_warmup == 0:
            return float(recipe.output_noise_std)
        return float(recipe.output_noise_std * min(1.0, self.progress / recipe.output_noise_warmup))

    def state_dict(self):
        return {"progress": self.progress, "order_ema": self.order_ema, "observed": self.observed}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != set(self.KEYS):
            raise ValueError("invalid progress schedule checkpoint")
        progress, ema, observed = state["progress"], state["order_ema"], state["observed"]
        if (type(progress) is not float or not 0 <= progress <= 1 or type(observed) is not int or observed < 0
                or (ema is not None and (type(ema) is not float or not 0 <= ema <= 1))):
            raise ValueError("invalid progress schedule checkpoint values")
        self.progress, self.order_ema, self.observed = progress, ema, observed
