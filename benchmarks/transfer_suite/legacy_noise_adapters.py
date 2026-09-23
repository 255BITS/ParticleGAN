"""Explicit native-data-space noise adapters for the nine custom GAN hosts.

The hosts keep their original targets, resources, architectures, losses and
metric formulas. This optional policy adds the same output and discriminator
input noise used by the 100-mode GANTrainer path. A zero policy is an identity
and consumes no random numbers. Each host must call ``set_step`` before every
update and place its checkpoints in ``evaluation`` to isolate training RNG.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import time

import torch
from torch import nn

from benchmarks.toy100.models import (
    LearnableOutputScale, linear_input_noise, linear_output_noise,
)


EVAL_SCOPES = {
    "two_pole": "learned_particles_and_critic_gradient",
    "trajectory": "generated_samples",
    "residual_student": "generated_samples",
    "unipolar": "learned_residual_parameters",
    "ae_gan_hold": "generated_and_reconstructed_samples",
    "cover_leftover": "learned_residual_parameters",
    "unused_token_hold": "learned_embedding_parameters",
    "mid_scale_identity": "learned_residual_parameters",
    "mode_hold": "generated_samples",
}


class NoisePolicy:
    """One target-agnostic noise rule with separate D and evaluation streams."""

    def __init__(
        self, output_std: float, input_std: float, input_anneal_end: float,
        total_steps: int, *, seed: int = 0, output_noise_warmup: float = 0.0,
        output_noise_learnable: bool = False,
    ) -> None:
        if not math.isfinite(output_std) or output_std < 0:
            raise ValueError("output_std must be finite and nonnegative")
        if not math.isfinite(input_std) or input_std < 0:
            raise ValueError("input_std must be finite and nonnegative")
        if (isinstance(output_noise_warmup, bool)
                or not math.isfinite(output_noise_warmup)
                or not 0.0 <= output_noise_warmup <= 1.0):
            raise ValueError("output_noise_warmup must be in [0, 1]")
        if type(seed) is not int or type(total_steps) is not int or total_steps <= 0:
            raise ValueError("seed and total_steps must be integers; total_steps > 0")
        if type(output_noise_learnable) is not bool:
            raise ValueError("output_noise_learnable must be a boolean")
        if output_noise_learnable and output_std <= 0:
            raise ValueError("output_noise_learnable requires output_std > 0")
        # Validate the fraction with the same function as the trainer hosts.
        linear_input_noise(input_std, 0, total_steps, input_anneal_end)
        self.output_std = float(output_std)
        self.output_noise_learnable = output_noise_learnable
        self.output_scale = (LearnableOutputScale(output_std)
                             if output_noise_learnable else None)
        self.generator_base_parameters: int | None = None
        self._output_scale_optimizer_owned = False
        self._final_live_scale: float | None = None
        self._final_ema_scale: float | None = None
        self._detach_output_scale = False
        self._effective_step_trace: list[float] = []
        self.output_noise_warmup = float(output_noise_warmup)
        self.output_sigma = linear_output_noise(
            output_std, 0, total_steps, output_noise_warmup,
        )
        self.input_std = float(input_std)
        self.input_anneal_end = float(input_anneal_end)
        self.total_steps = total_steps
        self.seed = seed
        self.input_stream = torch.Generator(device="cpu").manual_seed(seed + 901)
        self.input_sigma = 0.0
        self._evaluating = False
        self._step_calls = 0
        self._first_sigma: float | None = None
        self._last_sigma: float | None = None
        self._nonzero_steps = 0
        self._first_output_sigma: float | None = None
        self._last_output_sigma: float | None = None
        self._nonzero_output_steps = 0
        self._counts = {
            "output_train_calls": 0, "output_eval_calls": 0,
            "input_train_calls": 0, "input_eval_calls": 0,
            "output_train_elements": 0, "output_eval_elements": 0,
            "input_train_elements": 0, "input_eval_elements": 0,
        }

    def _output_sigma_for(self, completed_steps: int) -> float:
        return linear_output_noise(
            self.output_std, completed_steps, self.total_steps,
            self.output_noise_warmup,
        )

    def scale_parameters(self) -> list[nn.Parameter]:
        """The single G-owned scalar, or an empty list in fixed-noise mode."""
        return [] if self.output_scale is None else list(self.output_scale.parameters())

    def register_generator_base(self, model: nn.Module | nn.Parameter | int) -> None:
        if isinstance(model, nn.Module):
            count = sum(parameter.numel() for parameter in model.parameters())
        elif isinstance(model, nn.Parameter):
            count = model.numel()
        elif type(model) is int:
            count = model
        else:
            raise TypeError("generator base must be a module, parameter, or integer")
        if count <= 0 or self.generator_base_parameters is not None:
            raise ValueError("generator base parameters must be positive and registered once")
        self.generator_base_parameters = count

    def register_generator_optimizer(
        self, generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
    ) -> None:
        if self.output_scale is None:
            return
        scalar = self.output_scale.raw_scale
        g_count = sum(parameter is scalar for group in generator_optimizer.param_groups
                      for parameter in group["params"])
        d_count = sum(parameter is scalar for group in discriminator_optimizer.param_groups
                      for parameter in group["params"])
        if g_count != 1 or d_count:
            raise ValueError("learnable output scale must occur exactly once in G and never D")
        self._output_scale_optimizer_owned = True

    def _scale_value(self) -> float:
        return (self.output_std if self.output_scale is None
                else float(self.output_scale().detach()))

    def _effective_sigma(self, base_sigma: float, scale: float | None = None) -> float:
        if self.output_scale is None:
            return base_sigma
        return base_sigma / self.output_std * (self._scale_value() if scale is None else scale)

    def capture_final_live(self) -> None:
        """Preserve the learned live scalar before a host copies EMA weights."""
        self._final_live_scale = self._scale_value()

    def capture_final_ema(self) -> None:
        self._final_ema_scale = self._scale_value()

    @contextmanager
    def discriminator(self):
        """Detach the noise scalar when drawing a fake batch for a D update."""
        previous = self._detach_output_scale
        self._detach_output_scale = True
        try:
            yield
        finally:
            self._detach_output_scale = previous

    def set_step(self, completed_steps: int) -> float:
        """Set both noise amplitudes for the next D and G update."""
        sigma = linear_input_noise(
            self.input_std, completed_steps, self.total_steps,
            self.input_anneal_end,
        )
        self.input_sigma = sigma
        self.output_sigma = self._output_sigma_for(completed_steps)
        self._step_calls += 1
        if self._first_sigma is None:
            self._first_sigma = sigma
        self._last_sigma = sigma
        self._nonzero_steps += int(sigma > 0.0)
        if self._first_output_sigma is None:
            self._first_output_sigma = self.output_sigma
        self._last_output_sigma = self.output_sigma
        self._nonzero_output_steps += int(self.output_sigma > 0.0)
        self._effective_step_trace.append(self._effective_sigma(self.output_sigma))
        return sigma

    def output(self, generated: torch.Tensor, *, generator_step: bool | None = None) -> torch.Tensor:
        """Add fresh Gaussian noise after one generated data batch is formed."""
        key = "output_eval" if self._evaluating else "output_train"
        self._counts[key + "_calls"] += 1
        if self.output_sigma == 0.0:
            return generated
        self._counts[key + "_elements"] += generated.numel()
        # Match the ordinary OutputNoise wrapper's global training stream.
        if self.output_scale is None:
            sigma = self.output_sigma
        else:
            sigma = self.output_scale() * (self.output_sigma / self.output_std)
            if (self._detach_output_scale or self._evaluating
                    or generator_step is False):
                sigma = sigma.detach()
        return generated + sigma * torch.randn_like(generated)

    def input(self, data: torch.Tensor) -> torch.Tensor:
        """Perturb only the critic's data coordinates, never its conditioning."""
        key = "input_eval" if self._evaluating else "input_train"
        self._counts[key + "_calls"] += 1
        if self.input_sigma == 0.0:
            return data
        self._counts[key + "_elements"] += data.numel()
        noise = torch.randn(
            data.shape, generator=self.input_stream,
            device=data.device, dtype=data.dtype,
        )
        return data + self.input_sigma * noise

    @contextmanager
    def evaluation(self, step: int):
        """Give recorded draws fixed RNGs without advancing either train stream."""
        if self.output_std == 0.0 and self.input_std == 0.0:
            yield
            return
        if type(step) is not int or step < 0:
            raise ValueError("evaluation step must be a nonnegative integer")
        saved_d = self.input_stream.get_state()
        previous = self._evaluating
        previous_detach = self._detach_output_scale
        previous_output_sigma = self.output_sigma
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed + 402 + step)
                self.input_stream.manual_seed(self.seed + 1402 + step)
                self.output_sigma = self._output_sigma_for(step)
                self._evaluating = True
                self._detach_output_scale = True
                yield
        finally:
            self._evaluating = previous
            self._detach_output_scale = previous_detach
            self.output_sigma = previous_output_sigma
            self.input_stream.set_state(saved_d)

    def receipt(self) -> dict:
        """Report actual host call coverage, not merely configured intent."""
        live_scale = (self._scale_value() if self._final_live_scale is None
                      else self._final_live_scale)
        ema_scale = self._final_ema_scale
        return {
            "output_std": self.output_std,
            "output_noise_learnable": self.output_noise_learnable,
            "output_scale_parameter_count": len(self.scale_parameters()),
            "output_scale_optimizer_owned": self._output_scale_optimizer_owned,
            "generator_base_parameters": self.generator_base_parameters,
            "generator_total_parameters": (
                None if self.generator_base_parameters is None else
                self.generator_base_parameters + len(self.scale_parameters())
            ),
            "output_scale_initial": self.output_std,
            "output_scale_final": live_scale,
            "output_scale_ema_final": ema_scale,
            "output_sigma_effective_first": (
                self._effective_step_trace[0] if self._effective_step_trace else None
            ),
            "output_sigma_effective_last": (
                self._effective_step_trace[-1] if self._effective_step_trace else None
            ),
            "output_sigma_effective_step_trace": self._effective_step_trace,
            "output_sigma_effective_final_evaluation": self._effective_sigma(
                self._output_sigma_for(self.total_steps), live_scale,
            ),
            "output_sigma_effective_ema_final_evaluation": (
                None if ema_scale is None else self._effective_sigma(
                    self._output_sigma_for(self.total_steps), ema_scale,
                )
            ),
            "output_noise_warmup": self.output_noise_warmup,
            "output_sigma_first": self._first_output_sigma,
            "output_sigma_last": self._last_output_sigma,
            "output_sigma_final_evaluation": self._output_sigma_for(self.total_steps),
            "output_nonzero_steps": self._nonzero_output_steps,
            "input_std": self.input_std,
            "input_anneal_end": self.input_anneal_end,
            "total_steps": self.total_steps,
            "seed": self.seed,
            "d_noise_seed": self.seed + 901,
            "step_calls": self._step_calls,
            "input_sigma_first": self._first_sigma,
            "input_sigma_last": self._last_sigma,
            "input_nonzero_steps": self._nonzero_steps,
            **self._counts,
            "train_output_applied": bool(self._counts["output_train_elements"]),
            "train_input_applied": bool(self._counts["input_train_elements"]),
            "eval_output_applied": bool(self._counts["output_eval_elements"]),
        }


class _OutputAdapter(nn.Module):
    def __init__(self, model: nn.Module, policy: NoisePolicy) -> None:
        super().__init__()
        self.model = model
        self.policy = policy
        policy.register_generator_base(model)
        # Register the one learnable scalar on the G wrapper before the host
        # constructs its optimizer. The policy itself is deliberately not a
        # Module, so the scalar appears exactly once in generator.parameters().
        self.output_scale = policy.output_scale

    def forward(self, *args, **kwargs):
        return self.policy.output(self.model(*args, **kwargs))


class _InputAdapter(nn.Module):
    def __init__(self, model: nn.Module, policy: NoisePolicy, data_index: int) -> None:
        super().__init__()
        self.model = model
        self.policy = policy
        self.data_index = data_index

    def forward(self, *args, **kwargs):
        values = list(args)
        values[self.data_index] = self.policy.input(values[self.data_index])
        return self.model(*values, **kwargs)

    def features(self, *args, **kwargs):
        values = list(args)
        values[self.data_index] = self.policy.input(values[self.data_index])
        return self.model.features(*values, **kwargs)


def wrap_output(model: nn.Module, policy: NoisePolicy | None) -> nn.Module:
    return model if policy is None else _OutputAdapter(model, policy)


def wrap_input(
    model: nn.Module, policy: NoisePolicy | None, *, data_index: int = 0,
) -> nn.Module:
    return model if policy is None else _InputAdapter(model, policy, data_index)


def run_legacy(spec: dict, recipe, noise: dict, *, model_policy: dict | None = None) -> tuple[dict, dict]:
    """Run one frozen custom host with the common recipe and explicit noise."""
    from benchmarks import learned_lr_evaluation as bridge
    from benchmarks.locked_shared import baseline
    from benchmarks.smart_descent import evaluate
    from .compare_defaults import candidate, optimizer_defaults
    from . import vector_tasks

    started = time.perf_counter()
    applied = []
    policy = NoisePolicy(
        noise["output_noise_std"], noise["input_noise_std"],
        noise["input_noise_anneal_end"], spec["steps"], seed=0,
        output_noise_warmup=noise.get("output_noise_warmup", 0.0),
        output_noise_learnable=noise.get("output_noise_learnable", False),
    )
    schedule = vector_tasks.fixed_policy("cosine")
    cap = (model_policy or {}).get("network_lr_horizon_cap")
    network_floor = (model_policy or {}).get("network_lr_floor")
    with optimizer_defaults(recipe, applied, network_lr_horizon_cap=cap,
                            network_lr_floor=network_floor):
        control = evaluate.FixedControl(schedule, spec["steps"])
        with bridge.control_host_schedules(control):
            result = baseline.run_toy(
                spec["name"], candidate(recipe), noise_policy=policy,
            )
    result["actions"] = control.trace
    result["seconds"] = time.perf_counter() - started
    receipt = policy.receipt()
    receipt["eval_scope"] = EVAL_SCOPES[spec["name"]]
    receipt["training_mechanism_verified"] = bool(
        receipt["step_calls"] == spec["steps"]
        and receipt["train_output_applied"]
        and receipt["train_input_applied"]
    )
    context = {
        "applied": applied,
        "shapes": {"host": "legacy auxiliary custom loop"},
        "host_recipe": recipe,
        "noise_receipt": receipt,
        "eval_scope": receipt["eval_scope"],
    }
    return result, context
