"""Composable prior manipulation architecture.

This module provides base interfaces and implementations for swappable prior
manipulation, enabling custom index sampling, in-flight latent transformations,
generator post-processing autograd hooks, and intrinsic prior penalties.
"""
from typing import Any, List, Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn


class PriorManipulator(nn.Module):
    """Base class for composable prior manipulators.

    A PriorManipulator intercepts or augments the sampling, code transformation,
    generator output postprocessing, or prior regularization of ParticlePrior
    or MoGParticlePrior modules without altering the base prior storage.
    """

    row_local_gradients: bool = True

    def sample_indices(
        self,
        base_prior: nn.Module,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[torch.Tensor]:
        """Custom particle index selection.

        Args:
            base_prior: The underlying base prior module (e.g. ParticlePrior).
            batch_size: Positive integer number of particles to sample.
            generator: Optional torch.Generator for isolated RNG control.

        Returns:
            Optional 1D LongTensor of shape (batch_size,) on base_prior device,
            or None to delegate index sampling to subsequent manipulators or base.
        """
        return None

    def transform(
        self,
        codes: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        *,
        base_prior: nn.Module,
        generator: Optional[torch.Generator] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """Transforms sampled latent particle rows.

        Args:
            codes: Float tensor of sampled codes, shape (B, D) or intermediate transform shape.
            idx: Optional LongTensor of shape (B,) containing sampled particle indices.
            base_prior: The underlying base prior module.
            generator: Optional torch.Generator for isolated RNG control.
            training: Boolean indicating training (True) vs evaluation (False) mode.

        Returns:
            Transformed tensor, typically of shape (B, D_out).
        """
        return codes

    def postprocess_generated(
        self,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        """Attaches backward gradient hooks or post-processes generator outputs.

        Runs during the G-step after fake_g = G(latent) and before critic evaluation.

        Args:
            generated: Float tensor of generator outputs, typically (B, C, H, W).

        Returns:
            Post-processed or autograd-hooked tensor of identical shape and dtype.
        """
        return generated

    def regularize(
        self,
        raw_rows: torch.Tensor,
        *,
        sampled_codes: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        generated: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluates intrinsic manipulator loss penalties.

        Added directly to loss_g inside GANTrainer._step. When non-zero, GANTrainer
        records the detached penalty in result["prior_manipulation"].

        Args:
            raw_rows: Raw particle table rows (prior.z or prior.z[unique(indices)]), shape (N_sub, D).
            sampled_codes: Optional transformed latent codes used for generation, shape (B, D_lat).
            indices: Optional particle indices drawn for this batch, shape (B,).
            generated: Optional generator outputs fake_g, shape (B, C, H, W).
            logits: Optional critic logits critic(fake_g), shape (B, ...).

        Returns:
            Scalar loss tensor () with device and dtype matching raw_rows.
        """
        return torch.zeros((), device=raw_rows.device, dtype=raw_rows.dtype)


class ManipulatedPrior(nn.Module):
    """Wraps a base ParticlePrior or MoGParticlePrior and chains PriorManipulators.

    Delegates all core properties (.z, .num_particles, .z_dim, .init_std, .sigma,
    .standardize, .means(), .d0) to self.base so that external consumers
    (UpdatePolicy, GANTrainer, ServedModel, Recipe) continue working seamlessly.
    """

    def __init__(
        self,
        base: nn.Module,
        manipulators: Optional[Sequence[Any]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Module):
            raise TypeError(f"base prior must be an nn.Module, got {type(base)}")
        self.base = base

        flat_manipulators: List[PriorManipulator] = []
        if manipulators is not None:
            for item in manipulators:
                if isinstance(item, (list, tuple)):
                    flat_manipulators.extend(item)
                else:
                    flat_manipulators.append(item)

        self.manipulators = nn.ModuleList(flat_manipulators)
        for m in self.manipulators:
            if not isinstance(m, PriorManipulator):
                raise TypeError(f"manipulator must be an instance of PriorManipulator, got {type(m)}")
            if hasattr(m, "_ensure_initialized") and hasattr(self.base, "z_dim"):
                dev = getattr(self.base, "z", torch.empty(0)).device
                dt = getattr(self.base, "z", torch.empty(0)).dtype
                m._ensure_initialized(self.base.z_dim, device=dev, dtype=dt)

    def manipulate(self, *manipulators: Any) -> "ManipulatedPrior":
        """Chain additional manipulators, returning a new ManipulatedPrior."""
        return ManipulatedPrior(self.base, [*self.manipulators, *manipulators])

    # ------------------------------------------------------------------
    # Delegated Base Prior Attributes
    # ------------------------------------------------------------------

    @property
    def z(self) -> torch.Tensor:
        return self.base.z

    @z.setter
    def z(self, value: torch.Tensor) -> None:
        self.base.z = value

    @z.deleter
    def z(self) -> None:
        del self.base.z

    @property
    def num_particles(self) -> int:
        return self.base.num_particles

    @property
    def z_dim(self) -> int:
        return self.base.z_dim

    @property
    def init_std(self) -> float:
        return self.base.init_std

    @property
    def sigma(self) -> Any:
        return getattr(self.base, "sigma", None)

    @property
    def standardize(self) -> bool:
        return getattr(self.base, "standardize", False)

    @property
    def d0(self) -> Optional[torch.Tensor]:
        return getattr(self.base, "d0", None)

    @property
    def sigma_rel(self) -> float:
        return getattr(self.base, "sigma_rel", 0.0)

    @sigma_rel.setter
    def sigma_rel(self, value: float) -> None:
        if hasattr(self.base, "sigma_rel"):
            self.base.sigma_rel = value

    def means(self) -> torch.Tensor:
        if hasattr(self.base, "means"):
            return self.base.means()
        return self.base.z

    def set_sigma(self, sigma: float) -> None:
        if hasattr(self.base, "set_sigma"):
            self.base.set_sigma(sigma)

    @property
    def row_local_gradients(self) -> bool:
        base_local = getattr(self.base, "row_local_gradients", True)
        if hasattr(self.base, "standardize") and self.base.standardize:
            base_local = False
        return base_local and all(getattr(m, "row_local_gradients", True) for m in self.manipulators)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.__dict__.get("_modules", {}).get("base")
            if base is not None and hasattr(base, name):
                return getattr(base, name)
            raise

    # ------------------------------------------------------------------
    # Sampling & Execution
    # ------------------------------------------------------------------

    def sample_indices(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.LongTensor:
        """Sample indices by delegating through manipulators, falling back to base."""
        for m in self.manipulators:
            idx = m.sample_indices(self.base, batch_size, generator=generator)
            if idx is not None:
                return idx
        if hasattr(self.base, "sample_indices"):
            return self.base.sample_indices(batch_size, generator=generator)
        return torch.randint(0, self.base.num_particles, (batch_size,),
                             device=self.base.z.device, generator=generator)

    def forward(
        self,
        idx: torch.LongTensor,
        generator: Optional[torch.Generator] = None,
        *,
        eps: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass indexing into base prior and applying transform chain."""
        if hasattr(self.base, "forward"):
            import inspect
            sig = inspect.signature(self.base.forward)
            if "eps" in sig.parameters:
                codes = self.base(idx, generator=generator, eps=eps)
            else:
                codes = self.base(idx)
        else:
            codes = self.base.z[idx]

        for m in self.manipulators:
            codes = m.transform(
                codes, idx, base_prior=self.base, generator=generator, training=self.training
            )
        return codes

    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        *,
        fixed_first_n: bool = False,
        offset: int = 0,
        eps: Optional[torch.Tensor] = None,
        noise_generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Sample latent codes and particle indices."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if fixed_first_n:
            if hasattr(self.base, "sample"):
                import inspect
                sig = inspect.signature(self.base.sample)
                if "noise_generator" in sig.parameters:
                    base_codes, idx = self.base.sample(
                        batch_size, generator=generator, fixed_first_n=True,
                        offset=offset, eps=eps, noise_generator=noise_generator
                    )
                else:
                    base_codes, idx = self.base.sample(
                        batch_size, generator=generator, fixed_first_n=True,
                        offset=offset
                    )
            else:
                idx = torch.arange(offset, offset + batch_size, device=self.base.z.device, dtype=torch.long)
                base_codes = self.base.z[idx]

            codes = base_codes
            for m in self.manipulators:
                codes = m.transform(
                    codes, idx, base_prior=self.base, generator=generator, training=self.training
                )
            return codes, idx

        idx = self.sample_indices(batch_size, generator=generator)
        gen = noise_generator if noise_generator is not None else generator
        codes = self.forward(idx, generator=gen, eps=eps)
        return codes, idx

    def postprocess_generated(self, generated: torch.Tensor) -> torch.Tensor:
        """Chain postprocess_generated hooks across all manipulators."""
        for m in self.manipulators:
            generated = m.postprocess_generated(generated)
        return generated

    def regularize(
        self,
        raw_rows: torch.Tensor,
        *,
        sampled_codes: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        generated: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sum regularize losses across all manipulators."""
        total = torch.zeros((), device=raw_rows.device, dtype=raw_rows.dtype)
        for m in self.manipulators:
            loss = m.regularize(
                raw_rows,
                sampled_codes=sampled_codes,
                indices=indices,
                generated=generated,
                logits=logits,
            )
            total = total + loss
        return total


class StratifiedSamplingManipulator(PriorManipulator):
    """Epoch-stratified without-replacement permutation sampler with optional local k-NN jitter.

    Buffers:
        _perm: LongTensor of shape (num_particles,) storing the current permutation.
        _cursor: Scalar LongTensor tracking the consumption cursor in [0, num_particles].
        thermal_ratio: Scalar Float32 tensor in [0, 1] scaling optional k-NN jitter.

    Both _perm and _cursor are persisted in state_dict() for bit-exact checkpoint continuation.
    """

    def __init__(
        self,
        num_particles: int,
        *,
        jitter_std: float = 0.0,
        k_nn: int = 4,
        initial_cursor: int = 0,
    ) -> None:
        super().__init__()
        if type(num_particles) is not int or num_particles <= 0:
            raise ValueError(f"num_particles must be positive integer, got {num_particles}")
        if jitter_std < 0 or not math.isfinite(jitter_std):
            raise ValueError(f"jitter_std must be finite nonnegative float, got {jitter_std}")
        if type(k_nn) is not int or k_nn < 1:
            raise ValueError(f"k_nn must be positive integer, got {k_nn}")

        self.num_particles = num_particles
        self.jitter_std = float(jitter_std)
        self.k_nn = int(k_nn)

        self.register_buffer("_perm", torch.arange(num_particles, dtype=torch.long))
        self.register_buffer("_cursor", torch.tensor(initial_cursor, dtype=torch.long))
        self.register_buffer("thermal_ratio", torch.tensor(1.0, dtype=torch.float32))

    def set_thermal_ratio(self, ratio: Any) -> None:
        with torch.no_grad():
            val = float(ratio.item()) if isinstance(ratio, torch.Tensor) else float(ratio)
            self.thermal_ratio.fill_(max(0.0, min(1.0, val)))

    def sample_indices(
        self,
        base_prior: nn.Module,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        device = self._perm.device

        # In evaluation mode, draw uniform random indices without advancing the epoch cursor
        if not self.training:
            return torch.randint(0, self.num_particles, (batch_size,), device=device, generator=generator)

        # Batch exceeds total particles: generate fresh permutation with repeats
        if batch_size >= self.num_particles:
            perm = torch.randperm(self.num_particles, device=device, generator=generator)
            reps = (batch_size + self.num_particles - 1) // self.num_particles
            return perm.repeat(reps)[:batch_size]

        c = int(self._cursor.item())
        if c + batch_size >= self.num_particles:
            rem = self.num_particles - c
            tail = self._perm[c:].clone() if rem > 0 else self._perm[:0]
            self._perm.copy_(torch.randperm(self.num_particles, device=device, generator=generator))
            need = batch_size - rem
            idx = torch.cat((tail, self._perm[:need]), dim=0) if need > 0 else tail
            self._cursor.fill_(need)
            return idx
        else:
            idx = self._perm[c : c + batch_size]
            self._cursor.fill_(c + batch_size)
            return idx

    def transform(
        self,
        codes: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        *,
        base_prior: nn.Module,
        generator: Optional[torch.Generator] = None,
        training: bool = True,
    ) -> torch.Tensor:
        if not training or self.jitter_std <= 0.0 or len(codes) <= 1:
            return codes

        t_ratio = float(self.thermal_ratio.item())
        if t_ratio <= 1e-5:
            return codes

        # Detached k-NN local distance computation keeps row_local_gradients = True
        with torch.no_grad():
            k = min(self.k_nn, len(codes) - 1)
            dists = torch.cdist(codes.detach(), codes.detach(), p=2.0)
            eye_inf = torch.eye(len(codes), device=codes.device, dtype=codes.dtype) * 1e6
            knn_dists, _ = torch.topk(dists + eye_inf, k=k, dim=1, largest=False)
            local_scale = knn_dists.median(dim=1, keepdim=True).values.clamp_min(1e-4)

        noise = torch.randn(codes.shape, device=codes.device, dtype=codes.dtype, generator=generator)
        return codes + (self.jitter_std * t_ratio) * local_scale * noise


class SimplexRoutingManipulator(PriorManipulator):
    """Maps particle codes to simplex mode weights w(z) in Delta^{K-1} and optional residuals.

    Formulation:
        w(z) = softmax(P(z) / T) in Delta^{K-1}
        where P(z) is either a 2D linear projection matrix Parameter[z_dim, num_modes]
        (mode="project") or coordinate slicing z[:, :num_modes] (mode="slice").

    All learnable parameters are 2D Parameter matrices (ndim == 2) so DualNorm
    optimizes them natively with role="router".
    """

    def __init__(
        self,
        num_modes: int,
        *,
        temperature: float = 1.0,
        z_dim: Optional[int] = None,
        routing_mode: str = "project",
        residual_dim: Optional[int] = None,
        learnable_projection: bool = True,
    ) -> None:
        super().__init__()
        if type(num_modes) is not int or num_modes < 2:
            raise ValueError(f"num_modes must be integer >= 2, got {num_modes}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be positive finite float, got {temperature}")
        if routing_mode not in ("project", "slice"):
            raise ValueError(f"routing_mode must be 'project' or 'slice', got {routing_mode!r}")

        self.num_modes = num_modes
        self.temperature = float(temperature)
        self.routing_mode = routing_mode
        self.z_dim = z_dim
        self.residual_dim = residual_dim
        self.learnable_projection = bool(learnable_projection)

        if routing_mode == "project" and z_dim is not None:
            proj = torch.empty(z_dim, num_modes)
            nn.init.orthogonal_(proj)
            if self.learnable_projection:
                self.projection = nn.Parameter(proj)
            else:
                self.register_buffer("projection", proj)
        else:
            self.projection = None

        self.last_weights: Optional[torch.Tensor] = None

    def _ensure_initialized(self, z_dim: int, device: Any = None, dtype: Any = None) -> None:
        if self.routing_mode == "project" and self.projection is None:
            self.z_dim = z_dim
            proj = torch.empty(z_dim, self.num_modes, device=device, dtype=dtype)
            nn.init.orthogonal_(proj)
            if self.learnable_projection:
                self.projection = nn.Parameter(proj)
            else:
                self.register_buffer("projection", proj)

    def transform(
        self,
        codes: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        *,
        base_prior: nn.Module,
        generator: Optional[torch.Generator] = None,
        training: bool = True,
    ) -> torch.Tensor:
        if self.routing_mode == "project":
            if self.projection is None:
                self._ensure_initialized(codes.shape[-1], device=codes.device, dtype=codes.dtype)
            if self.projection.device != codes.device or self.projection.dtype != codes.dtype:
                self.projection.data = self.projection.data.to(device=codes.device, dtype=codes.dtype)
            logits = torch.matmul(codes, self.projection)
            w = torch.softmax(logits / self.temperature, dim=-1)
            if self.residual_dim is not None and self.residual_dim > 0:
                res = codes[:, :self.residual_dim]
                out = torch.cat([w, res], dim=-1)
            else:
                out = w
        else:
            if codes.shape[-1] < self.num_modes:
                raise ValueError(
                    f"codes dimension {codes.shape[-1]} smaller than num_modes {self.num_modes}"
                )
            logits = codes[:, :self.num_modes]
            w = torch.softmax(logits / self.temperature, dim=-1)
            res = codes[:, self.num_modes:]
            out = torch.cat([w, res], dim=-1) if res.shape[-1] > 0 else w

        self.last_weights = w.detach()
        try:
            out.weights = w
        except Exception:
            pass
        return out

    @staticmethod
    def split(codes: torch.Tensor, num_modes: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Utility separating simplex weights from residual codes."""
        w = codes[:, :num_modes]
        res = codes[:, num_modes:] if codes.shape[-1] > num_modes else None
        return w, res


from .wavefront import (  # noqa: E402
    GhostFreeDeMixingManipulator,
    SobolevWavefrontManipulator,
    SplatCarrierManipulator,
)
from .tomographic import (  # noqa: E402
    CosineRepulsionManipulator,
    MarginalEntropyManipulator,
)

__all__ = [
    "PriorManipulator",
    "ManipulatedPrior",
    "StratifiedSamplingManipulator",
    "SimplexRoutingManipulator",
    "GhostFreeDeMixingManipulator",
    "SobolevWavefrontManipulator",
    "SplatCarrierManipulator",
    "CosineRepulsionManipulator",
    "MarginalEntropyManipulator",
]

