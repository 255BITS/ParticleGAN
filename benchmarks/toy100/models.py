"""Optional benchmark-local noise wrappers for 100-Gaussian models.

These wrappers do not use target centers or component assignments. The public
``GANTrainer`` and ``Recipe`` remain responsible for the ordinary GAN updates.
"""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

import particlegan.sample_stream as sample_stream


# A fixed namespace, separate from GANTrainer's +2/+3/+4 streams and the
# discriminator input-noise stream at +901. This is a policy constant, never
# an experiment parameter or a substitute for the declared training seed.
OUTPUT_NOISE_SEED_OFFSET = 1901


class LearnableOutputScale(nn.Module):
    """One positive output-noise standard deviation learned with the generator."""

    def __init__(self, initial_std: float):
        super().__init__()
        if (isinstance(initial_std, bool) or not math.isfinite(initial_std)
                or initial_std <= 0):
            raise ValueError("learnable output noise requires a positive finite initial std")
        self.initial_std = float(initial_std)
        self.raw_scale = nn.Parameter(torch.tensor(
            initial_std + math.log(-math.expm1(-initial_std)),
        ))

    def forward(self) -> torch.Tensor:
        return F.softplus(self.raw_scale)


class OutputNoise(nn.Module):
    """Add fresh isotropic output noise in both training and sampling."""

    def __init__(self, model: nn.Module, std: float, learnable: bool = False):
        super().__init__()
        if not math.isfinite(std) or std < 0:
            raise ValueError("output noise std must be finite and nonnegative")
        if type(learnable) is not bool:
            raise ValueError("learnable must be a boolean")
        self.model = model
        self.std = float(std)
        self.initial_std = float(std)
        self.output_scale = LearnableOutputScale(std) if learnable else None
        if self.output_scale is not None:
            reference = next(model.parameters(), None)
            if reference is None:
                reference = next(model.buffers(), None)
            if reference is not None:
                dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
                self.output_scale.to(device=reference.device, dtype=dtype)

    def effective_std(self) -> float | torch.Tensor:
        """Apply the common warmup amplitude to the learned base scale."""
        if self.output_scale is None:
            return self.std
        return self.output_scale() * (self.std / self.initial_std)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        prediction = self.model(latent)
        if self.std == 0:
            return prediction
        # GANTrainer.sample forks the global RNG, so evaluation leaves this
        # training noise stream untouched and fixed-seed frames replay exactly.
        return prediction + self.effective_std() * self._randn(prediction)

    def _randn(self, prediction: torch.Tensor) -> torch.Tensor:
        if sample_stream.replacing():
            return sample_stream.normal(
                "output", prediction.shape, device=prediction.device, dtype=prediction.dtype,
            )
        return torch.randn_like(prediction)


class IsolatedOutputNoise(OutputNoise):
    """Output noise with an independent, checkpointable random stream.

    The private generator is not a parameter or buffer. Its ByteTensor state
    is the module's ``_extra_state`` entry, which is compatible with the
    public trainer's tensor-only checkpoint validator. Counters are ordinary
    persistent buffers, so a resumed run also retains its draw receipts.
    GANTrainer copies those buffers to EMA after each update; EMA counts thus
    inherit the live training receipt and are not EMA training calls.
    """

    def __init__(self, model: nn.Module, std: float, *, seed: int,
                 device: torch.device, learnable: bool = False):
        if type(seed) is not int or seed < 0:
            raise ValueError("output noise seed must be a nonnegative integer")
        if not math.isfinite(std) or std <= 0:
            raise ValueError("isolated output noise requires positive std")
        super().__init__(model, std, learnable=learnable)
        device = torch.device(device)
        self.noise_stream = torch.Generator(device=device).manual_seed(
            seed + OUTPUT_NOISE_SEED_OFFSET,
        )
        self.register_buffer("noise_draw_calls", torch.zeros((), dtype=torch.int64, device=device))
        self.register_buffer("noise_draw_elements", torch.zeros((), dtype=torch.int64, device=device))
        self._output_rng_scope_active = False

    def _randn(self, prediction: torch.Tensor) -> torch.Tensor:
        if sample_stream.replacing():
            noise = sample_stream.normal(
                "output", prediction.shape, device=prediction.device, dtype=prediction.dtype,
            )
        else:
            noise = torch.randn(
                prediction.shape, generator=self.noise_stream,
                device=prediction.device, dtype=prediction.dtype,
            )
        self.noise_draw_calls.add_(1)
        self.noise_draw_elements.add_(prediction.numel())
        return noise

    def get_extra_state(self) -> torch.Tensor:
        return self.noise_stream.get_state()

    def set_extra_state(self, state: torch.Tensor) -> None:
        self.noise_stream.set_state(state.cpu())

    def draw_receipt(self) -> dict[str, int]:
        return {"calls": int(self.noise_draw_calls),
                "elements": int(self.noise_draw_elements)}


@contextmanager
def paired_output_noise(models, *, seed: int):
    """Pair evaluation draws without advancing either training stream.

    ``seed`` is the caller's existing output-evaluation seed. Both models use
    its fixed output-noise namespace; their original states and counters are
    restored even if evaluation raises. A declared fixed-noise model is a
    no-op, preserving the historical global-RNG path.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("output-noise evaluation seed must be a nonnegative integer")
    wrappers = tuple(dict.fromkeys(
        model for model in models if isinstance(model, IsolatedOutputNoise)
    ))
    saved = tuple((model.noise_stream.get_state(), model.noise_draw_calls.clone(),
                   model.noise_draw_elements.clone(), model._output_rng_scope_active)
                  for model in wrappers)
    try:
        for model in wrappers:
            model.noise_stream.manual_seed(seed + OUTPUT_NOISE_SEED_OFFSET)
            model.noise_draw_calls.zero_()
            model.noise_draw_elements.zero_()
            model._output_rng_scope_active = True
        yield
    finally:
        for model, (state, calls, elements, active) in zip(wrappers, saved):
            model.noise_stream.set_state(state)
            model.noise_draw_calls.copy_(calls)
            model.noise_draw_elements.copy_(elements)
            model._output_rng_scope_active = active


class InputNoise(nn.Module):
    """Add noise to each discriminator input from its own training stream."""

    def __init__(self, model: nn.Module, *, seed: int, device: torch.device):
        super().__init__()
        self.model = model
        self.sigma = 0.0
        self.noise_stream = torch.Generator(device=device).manual_seed(seed)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if self.sigma:
            noise = torch.randn(
                points.shape, generator=self.noise_stream,
                device=points.device, dtype=points.dtype,
            )
            points = points + self.sigma * noise
        return self.model(points)


class StatefulInputNoise(InputNoise):
    """Keep the existing input-noise stream in new-policy checkpoints only."""

    def get_extra_state(self) -> torch.Tensor:
        return self.noise_stream.get_state()

    def set_extra_state(self, state: torch.Tensor) -> None:
        self.noise_stream.set_state(state.cpu())


def linear_input_noise(
    peak: float, completed_steps: int, total_steps: int, end_fraction: float,
) -> float:
    """Peak at the first update, linearly reaching zero by ``end_fraction``."""
    if not math.isfinite(peak) or peak < 0:
        raise ValueError("input noise std must be finite and nonnegative")
    if (type(completed_steps) is not int or completed_steps < 0
            or type(total_steps) is not int or total_steps <= 0):
        raise ValueError("invalid input noise step count")
    if not math.isfinite(end_fraction) or not 0 < end_fraction <= 1:
        raise ValueError("input noise anneal end must be in (0, 1]")
    return float(peak * max(0.0, 1.0 - completed_steps / (total_steps * end_fraction)))


def linear_output_noise(
    peak: float, completed_steps: int, total_steps: int, warmup_fraction: float = 0.0,
) -> float:
    """Rise from zero to ``peak`` over a fraction of completed updates.

    A zero warmup retains the original constant-noise recipe. The first
    update of a positive warmup runs with zero output noise.
    """
    if isinstance(peak, bool) or not math.isfinite(peak) or peak < 0:
        raise ValueError("output noise std must be finite and nonnegative")
    if (type(completed_steps) is not int or completed_steps < 0
            or type(total_steps) is not int or total_steps <= 0):
        raise ValueError("invalid output noise step count")
    if (isinstance(warmup_fraction, bool) or not math.isfinite(warmup_fraction)
            or not 0 <= warmup_fraction <= 1):
        raise ValueError("output noise warmup must be a finite fraction in [0, 1]")
    if warmup_fraction == 0:
        return float(peak)
    return float(peak * min(1.0, completed_steps / (warmup_fraction * total_steps)))
