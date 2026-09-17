"""
Particle prior module.

This defines a learnable latent particle cloud that can be shared across
different training scripts. It is intentionally minimal and fully
vectorized so it plays nicely with Accelerate / DDP and large particle
counts (e.g. 100k+).
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn


class ParticlePrior(nn.Module):
    r"""
    Learnable latent particle cloud.

    This module holds a parameter matrix z \in R^{M x D} where each row is a
    latent "particle". Sampling is implemented as pure indexing into this
    matrix, so it is:

      * fully vectorized (no Python-side loops),
      * data-parallel / multi-GPU friendly (z is just a regular Parameter),
      * easy to plug into EP-style regularizers that operate on the full cloud.

    Pass `learnable=False` for the frozen-Gaussian control: identical interface,
    identical sampling, but the cloud is a buffer rather than a Parameter, so it
    never moves and `parameters()` comes back empty.

    Typical usage (with Accelerate):

        prior = ParticlePrior(num_particles=100_000, z_dim=256)
        prior, ... = accelerator.prepare(prior, ...)

        # later in the training loop
        idx = accelerator.unwrap_model(prior).sample_indices(batch_size)
        z = prior(idx)  # Keep the DDP forward path so gradients synchronize.
    """

    def __init__(
        self,
        num_particles: int = 20_000,
        z_dim: int = 4,
        init_std: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        learnable: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()

        if type(num_particles) is not int or num_particles <= 0:
            raise ValueError(f"num_particles must be positive, got {num_particles}")
        if type(z_dim) is not int or z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        if not math.isfinite(init_std) or init_std < 0:
            raise ValueError("init_std must be finite and nonnegative")

        factory_kwargs = {"device": device, "dtype": dtype}

        # One tensor of particles.
        # When learnable (the default) it is a single Parameter, so DDP /
        # Accelerate treat it like any other weight matrix.
        #
        # `learnable=False` registers the same tensor as a *buffer* instead: the
        # cloud is then a frozen draw from N(0, init_std^2) that no optimizer can
        # move. That is the control condition for the whole premise of this repo
        # -- a fixed Gaussian prior, with G left to do all the warping on its own
        # -- while every call site below (`sample`, `forward`, `z[idx]`) keeps
        # working unchanged. Sampling still consumes exactly the same randomness
        # either way, so flipping this flag does not shift any other RNG stream.
        z = torch.empty(num_particles, z_dim, **factory_kwargs)
        if learnable:
            self.z = nn.Parameter(z)
        else:
            self.register_buffer("z", z)
        with torch.no_grad():
            self.z.normal_(mean=0.0, std=init_std, generator=generator)

    @property
    def num_particles(self) -> int:
        return self.z.shape[0]

    @property
    def z_dim(self) -> int:
        return self.z.shape[1]

    @torch.no_grad()
    def sample_indices(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.LongTensor:
        """
        Sample integer indices into the particle table.

        This is fully vectorized and runs entirely on the same device
        as `self.z`, which keeps it efficient in multi-GPU setups.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        return torch.randint(
            0,
            self.num_particles,
            (batch_size,),
            device=self.z.device,
            generator=generator,
        )

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """
        Index into the particle table.

        Args:
            idx: LongTensor of indices on any device; if necessary it will be
                 moved to `self.z.device` before indexing.

        Returns:
            z_batch: (B, z_dim) subset of the particle cloud.
        """
        if idx.device != self.z.device:
            idx = idx.to(self.z.device)
        return self.z[idx]

    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        *,
        fixed_first_n: bool = False,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Convenience wrapper returning both latent codes and their indices.

        By default, this samples indices uniformly at random from the full
        particle table using `sample_indices`.

        If `fixed_first_n` is True, the call becomes deterministic and returns
        a contiguous block of particles:

            idx = [offset, offset + 1, ..., offset + batch_size - 1]

        This is handy for evaluation snapshots where you want to keep a fixed
        latent grid over the course of training (e.g. for videos).

        Returns:
            z_batch: (B, z_dim)
            idx: (B,) LongTensor of indices on the same device as `self.z`.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if fixed_first_n:
            if offset < 0:
                raise ValueError(f"offset must be non-negative, got {offset}")
            if offset + batch_size > self.num_particles:
                raise ValueError(
                    f"Requested batch_size={batch_size} with offset={offset}, "
                    f"but num_particles={self.num_particles}"
                )
            with torch.no_grad():
                idx = torch.arange(
                    offset,
                    offset + batch_size,
                    device=self.z.device,
                    dtype=torch.long,
                )
        else:
            with torch.no_grad():
                idx = self.sample_indices(batch_size, generator=generator)
        z_batch = self.z[idx]
        return z_batch, idx


class MoGParticlePrior(ParticlePrior):
    """Uniform Gaussian mixture with learned means and fixed calibrated noise.

    Each draw is ``means()[idx] + sigma * eps``. ``sigma_rel`` multiplies the
    initial median nearest-neighbor distance; sigma stays fixed while training.
    Optional per-dimension read standardization is differentiable. Regularize
    raw ``z``, not noisy draws or standardized means. ``forward`` samples noise
    for supplied indices, so it can be used through a DDP wrapper.

    Calibration uses SciPy when installed (``pip install particlegan[mog]``),
    otherwise exact, memory-bounded Torch distances. The Torch fallback is
    quadratic in table size; SciPy is recommended for large low-dimensional tables.
    """

    def __init__(
        self,
        num_particles: int = 400,
        z_dim: int = 4,
        init_std: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        learnable: bool = True,
        generator: Optional[torch.Generator] = None,
        *,
        sigma_rel: float = 1 / 40,
        standardize: bool = True,
    ) -> None:
        if not math.isfinite(sigma_rel) or sigma_rel < 0:
            raise ValueError("sigma_rel must be finite and nonnegative")
        if type(standardize) is not bool:
            raise ValueError("standardize must be a boolean")
        super().__init__(num_particles, z_dim, init_std, device, dtype, learnable, generator)
        if self.num_particles < 2:
            raise ValueError("MoG calibration requires at least two particles")
        self.sigma_rel = float(sigma_rel)
        self.standardize = standardize
        self.register_buffer("sigma", self.z.new_zeros(()))
        self.register_buffer("d0", self.z.new_zeros(()))
        self.calibrate()

    def means(self):
        """Return all differentiable component centers, without sampling noise."""
        if not self.standardize:
            return self.z
        return (self.z - self.z.mean(0)) / (self.z.std(0) + 1e-6)

    @torch.no_grad()
    def calibrate(self):
        """Reset d0 and sigma from current means; called once during construction.

        Calling this again explicitly changes the fixed noise scale. Training,
        EMA copies and checkpoint loading do not recalibrate.
        """
        points = self.means().detach().cpu().double()
        if not torch.isfinite(points).all():
            raise ValueError("component means must be finite for calibration")
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            # At most ~32 MiB of distances; no full N x N matrix is retained.
            chunk_size = max(1, min(1024, 4_000_000 // self.num_particles))
            nearest = []
            for start in range(0, self.num_particles, chunk_size):
                chunk = points[start:start + chunk_size]
                distances = torch.cdist(chunk, points, compute_mode="donot_use_mm_for_euclid_dist")
                rows = torch.arange(len(chunk))
                distances[rows, rows + start] = float("inf")
                nearest.append(distances.min(dim=1).values)
            # quantile averages the two middle distances for an even table.
            distance = torch.cat(nearest).quantile(.5).item()
        else:
            # Preserve the experimental calibration, including even-N median.
            import numpy as np
            distance = float(np.median(cKDTree(points.numpy()).query(points.numpy(), k=2)[0][:, 1]))
        self.d0.fill_(distance)
        if not torch.isfinite(self.d0) or self.d0 <= 0:
            raise ValueError("median nearest-neighbor distance must be finite and positive")
        self.sigma.copy_(self.d0 * self.sigma_rel)
        if not torch.isfinite(self.sigma):
            raise ValueError("calibrated sigma must be finite")
        self._noise_enabled = bool(self.sigma > 0)

    def forward(self, idx, generator=None, *, eps=None):
        """Draw from the indexed components; use means()[idx] for centers only.

        Explicit eps must match the output shape/device/dtype and replaces the
        Gaussian RNG draw. At sigma=0 no noise RNG is consumed.
        """
        z = self.means()[idx.to(self.z.device)]
        if self._noise_enabled:
            if eps is None:
                eps = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
            elif eps.shape != z.shape or eps.device != z.device or eps.dtype != z.dtype:
                raise ValueError("eps must match sampled codes' shape, device and dtype")
            z = z + self.sigma * eps
        return z

    def sample(self, batch_size, generator=None, *, fixed_first_n=False,
               offset=0, eps=None):
        """Return noisy codes and component indices, uniformly with replacement.

        fixed_first_n fixes indices only; also supply fixed eps for reproducible
        positive-noise snapshots. A generator controls both indices and noise.
        """
        # Reuse index validation and RNG consumption exactly, including r=0.
        _, idx = super().sample(batch_size, generator,
                                fixed_first_n=fixed_first_n, offset=offset)
        return self(idx, generator=generator, eps=eps), idx

    def get_extra_state(self):
        return {"sigma_rel": self.sigma_rel, "standardize": self.standardize}

    def set_extra_state(self, state):
        sigma_rel, standardize = state["sigma_rel"], state["standardize"]
        if not math.isfinite(sigma_rel) or sigma_rel < 0 or type(standardize) is not bool:
            raise ValueError("invalid MoG checkpoint configuration")
        self.sigma_rel, self.standardize = float(sigma_rel), standardize

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Pre-public experiment checkpoints saved just z/sigma/d0. Their read
        # standardization must still be supplied by the constructor/config.
        key = prefix + "_extra_state"
        if key not in state_dict:
            state = self.get_extra_state()
            if prefix + "sigma" in state_dict and prefix + "d0" in state_dict:
                state["sigma_rel"] = float(state_dict[prefix + "sigma"] / state_dict[prefix + "d0"])
            state_dict[key] = state
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        self._noise_enabled = bool(self.sigma > 0)


class GaussianPrior(nn.Module):
    """Fresh Gaussian latent draws, with no table or trainable parameters.

    An empty buffer tracks device and dtype through normal ``.to(...)`` calls.
    ``sample`` returns ``(codes, None)`` because draws have no particle indices.
    """

    def __init__(self, z_dim=4, init_std=1.0, device=None, dtype=None):
        super().__init__()
        if type(z_dim) is not int or z_dim <= 0:
            raise ValueError("z_dim must be a positive integer")
        if not math.isfinite(init_std) or init_std < 0:
            raise ValueError("init_std must be finite and nonnegative")
        self.z_dim, self.init_std = z_dim, float(init_std)
        self.register_buffer("_anchor", torch.empty(0, device=device, dtype=dtype))

    def forward(self, batch_size, generator=None):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return torch.randn((batch_size, self.z_dim), device=self._anchor.device,
                           dtype=self._anchor.dtype, generator=generator) * self.init_std

    def sample(self, batch_size, generator=None):
        return self(batch_size, generator=generator), None


class FreshGaussianPrior(ParticlePrior):
    """Fresh Gaussian noise, with a fixed reference batch for snapshots only.

    Ordinary ``sample`` calls draw from N(0, init_std**2), independently of the
    reference buffer ``z``. ``fixed_first_n=True`` instead selects that buffer
    for reproducible plots. The buffer also keeps initialization RNG consumption
    identical across prior controls. This prior has no trainable parameters;
    ordinary samples return ``None`` for indices because they are not table rows.
    """

    def __init__(self, num_particles=100_000, z_dim=256, init_std=1.0,
                 device=None, dtype=None):
        super().__init__(num_particles, z_dim, init_std, device, dtype, learnable=False)
        self.init_std = float(init_std)

    def sample(self, batch_size, generator=None, *, fixed_first_n=False, offset=0):
        if fixed_first_n:
            return super().sample(batch_size, generator, fixed_first_n=True, offset=offset)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        z = torch.randn(batch_size, self.z_dim, device=self.z.device,
                        dtype=self.z.dtype, generator=generator) * self.init_std
        return z, None


PRIOR_KINDS = ("mog", "particles", "frozen_gaussian", "fresh_gaussian", "gaussian")


def canonical_prior_kind(kind: str) -> str:
    """Keep historical ``gaussian`` configs as frozen finite-table controls."""
    kind = str(kind).lower()
    if kind not in PRIOR_KINDS:
        raise ValueError(f"Unknown prior: {kind!r} (expected one of {PRIOR_KINDS})")
    return "frozen_gaussian" if kind == "gaussian" else kind


def make_prior(kind: str, **kwargs) -> ParticlePrior:
    kind = canonical_prior_kind(kind)
    if kind == "mog":
        return MoGParticlePrior(**kwargs)
    if kind == "fresh_gaussian":
        return FreshGaussianPrior(**kwargs)
    return ParticlePrior(**kwargs, learnable=(kind == "particles"))
