"""
Particle prior module.

This defines a learnable latent particle cloud that can be shared across
different training scripts. It is intentionally minimal and fully
vectorized so it plays nicely with Accelerate / DDP and large particle
counts (e.g. 100k+).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ParticlePrior(nn.Module):
    """
    Learnable latent particle cloud.

    This module holds a parameter matrix z \in R^{M x D} where each row is a
    latent "particle". Sampling is implemented as pure indexing into this
    matrix, so it is:

      * fully vectorized (no Python-side loops),
      * data-parallel / multi-GPU friendly (z is just a regular Parameter),
      * easy to plug into EP-style regularizers that operate on the full cloud.

    Typical usage (with Accelerate):

        prior = ParticlePrior(num_particles=100_000, z_dim=256)
        prior, ... = accelerator.prepare(prior, ...)

        # later in the training loop
        z, idx = accelerator.unwrap_model(prior).sample(batch_size)
    """

    def __init__(
        self,
        num_particles: int = 100_000,
        z_dim: int = 256,
        init_std: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if num_particles <= 0:
            raise ValueError(f"num_particles must be positive, got {num_particles}")
        if z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {z_dim}")

        factory_kwargs = {"device": device, "dtype": dtype}

        # Big-ass tensor of learnable particles.
        # Kept as a single Parameter so DDP / Accelerate treat it like any
        # other weight matrix.
        z = torch.empty(num_particles, z_dim, **factory_kwargs)
        self.z = nn.Parameter(z)
        with torch.no_grad():
            self.z.normal_(mean=0.0, std=init_std)

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
