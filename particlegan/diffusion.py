"""Gaussian DDGAN transitions and independent noise sources."""
import torch
from torch import nn


class DiffusionSchedule(nn.Module):
    """Gaussian transitions with a caller-owned reverse loop.

    ``validate_args=False`` skips tensor bounds checks for caller-validated
    timesteps, avoiding their host synchronization during CUDA training.
    Shape, dtype, and device checks remain enabled.
    """

    def __init__(self, alpha_bar=(1.0, 0.9, 0.5, 0.05, 0.0001), *, device=None, dtype=None, validate_args=True):
        super().__init__()
        self.validate_args = bool(validate_args)
        ab = torch.as_tensor(alpha_bar, device=device, dtype=dtype or torch.float32).detach().clone()
        if not ab.is_floating_point() or ab.ndim != 1 or len(ab) < 2:
            raise ValueError("alpha_bar must be a one-dimensional floating schedule with at least two entries")
        if not bool(torch.isfinite(ab).all()) or ab[0] != 1 or not bool(((ab[1:] > 0) & (ab[1:] < ab[:-1])).all()):
            raise ValueError("alpha_bar must start at 1 and decrease strictly, remaining positive")
        al = ab[1:] / ab[:-1]
        beta = 1 - al
        self.steps = len(ab) - 1
        self.register_buffer("ab", ab)
        self.register_buffer("alpha", torch.cat([ab.new_ones(1), al]))
        self.register_buffer("beta", torch.cat([ab.new_zeros(1), beta]))
        self.register_buffer("A", torch.cat([ab.new_zeros(1), ab[:-1].sqrt() * beta / (1 - ab[1:])]))
        self.register_buffer("B", torch.cat([ab.new_zeros(1), al.sqrt() * (1 - ab[:-1]) / (1 - ab[1:])]))
        self.register_buffer("posterior_var", torch.cat([ab.new_zeros(1), beta * (1 - ab[:-1]) / (1 - ab[1:])]))

    def _validate(self, x, t):
        if x.ndim < 2 or not x.is_floating_point():
            raise ValueError("data must be floating point with shape [batch, ...]")
        if t.dtype != torch.long or t.shape != (len(x),):
            raise ValueError("timesteps must be LongTensor of shape [batch]")
        if x.device != self.ab.device or t.device != x.device:
            raise ValueError("schedule, data, and timesteps must share a device; use .to(device)")
        if self.validate_args and bool(((t < 1) | (t > self.steps)).any()):
            raise ValueError(f"timesteps must be in [1, {self.steps}]")

    def _coefficient(self, values, t, x):
        return values[t].to(dtype=x.dtype).reshape((-1,) + (1,) * (x.ndim - 1))

    def forward_pair(self, x0, t, rng=None, *, generator=None):
        """Draw coupled ``(x_(t-1), x_t)``; pass a generator to isolate RNG state."""
        self._validate(x0, t)
        if rng is not None and generator is not None:
            raise ValueError("pass either rng or generator, not both")
        rng = generator if generator is not None else rng
        prev_a = self._coefficient(self.ab, t - 1, x0)
        prev = prev_a.sqrt() * x0 + (1 - prev_a).sqrt() * torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=rng)
        xt = self._coefficient(self.alpha, t, x0).sqrt() * prev + self._coefficient(self.beta, t, x0).sqrt() * torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=rng)
        return prev, xt

    def reverse(self, x0, xt, t, eta):
        """Construct a reverse transition from predicted clean data and supplied noise."""
        self._validate(x0, t)
        if any(v.shape != x0.shape or v.device != x0.device or v.dtype != x0.dtype for v in (xt, eta)):
            raise ValueError("x0, xt, and eta must have identical shape, device, and dtype")
        return self._coefficient(self.A, t, x0) * x0 + self._coefficient(self.B, t, x0) * xt + self._coefficient(self.posterior_var, t, x0).sqrt() * eta


DDGAN = DiffusionSchedule


class DrawSource(nn.Module):
    """Fresh Gaussian or uniform draws from a fixed/learned independent table."""
    def __init__(self, kind, count, dim, seed, device):
        super().__init__()
        if kind not in ("gaussian", "fixed", "learned", "zero"):
            raise ValueError(f"unknown source {kind}")
        self.kind, self.dim = kind, dim
        init_rng = torch.Generator(device=device).manual_seed(seed)
        table = torch.randn((count, dim), generator=init_rng, device=device)
        if kind == "learned":
            self.table = nn.Parameter(table)
        else:
            self.register_buffer("table", table)

    def sample(self, n, rng):
        if self.kind == "gaussian":
            return torch.randn((n, self.dim), device=self.table.device, dtype=self.table.dtype, generator=rng), None
        if self.kind == "zero":
            return self.table.new_zeros(n, self.dim), None
        ids = torch.randint(len(self.table), (n,), device=self.table.device, generator=rng)
        return self.table[ids], ids
