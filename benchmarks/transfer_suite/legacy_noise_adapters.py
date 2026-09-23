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

from benchmarks.toy100.models import linear_input_noise, linear_output_noise


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
        # Validate the fraction with the same function as the trainer hosts.
        linear_input_noise(input_std, 0, total_steps, input_anneal_end)
        self.output_std = float(output_std)
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
        return sigma

    def output(self, generated: torch.Tensor) -> torch.Tensor:
        """Add fresh Gaussian noise after one generated data batch is formed."""
        key = "output_eval" if self._evaluating else "output_train"
        self._counts[key + "_calls"] += 1
        if self.output_sigma == 0.0:
            return generated
        self._counts[key + "_elements"] += generated.numel()
        # Match the ordinary OutputNoise wrapper's global training stream.
        return generated + self.output_sigma * torch.randn_like(generated)

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
        previous_output_sigma = self.output_sigma
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed + 402 + step)
                self.input_stream.manual_seed(self.seed + 1402 + step)
                self.output_sigma = self._output_sigma_for(step)
                self._evaluating = True
                yield
        finally:
            self._evaluating = previous
            self.output_sigma = previous_output_sigma
            self.input_stream.set_state(saved_d)

    def receipt(self) -> dict:
        """Report actual host call coverage, not merely configured intent."""
        return {
            "output_std": self.output_std,
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


def run_legacy(spec: dict, recipe, noise: dict) -> tuple[dict, dict]:
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
    )
    schedule = vector_tasks.fixed_policy("cosine")
    with optimizer_defaults(recipe, applied):
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
