"""SGDA, normalized steps and spectral dual-norm steps for fixed GAN recipes.

Learning rates are step sizes in each algorithm's own units. G and E belong
to one player for global normalization; the learned prior is a separate
player. The graft uses the norm of Adam's *unit-learning-rate* direction,
so the scheduled learning rate is applied exactly once.
"""
from copy import deepcopy
from dataclasses import asdict
import math

import torch
from torch.optim import Optimizer

from ..grad_regularizers import CriticStepRecord
from .convolution_frobenius import frobenius_direction
from .gram_polar import GRAM_DENSE_PROFILE, guarded_gram_polar
from .gram_svd_fallback import VERSION as FALLBACK_DENSE_PROFILE, Counters, svd_fallback_polar


NORMALIZED_FAMILIES = (
    "sgda", "nsgda_global", "nsgda_layer", "ada_nsgda", "dualnorm",
    "dualnorm_D_only", "particle_rownorm_only",
)
_ROLES = {"generator", "encoder", "router", "noise", "critic", "prior", "table"}
_CONVOLUTION_VERSION = "per_offset_polar_v1"
_FROBENIUS_CONVOLUTION_VERSION = "per_offset_frobenius_hybrid_v1"
_QUINTIC_CONVOLUTION_VERSION = "per_offset_quintic_hybrid_v1"
_NEWTON_SCHULZ_VERSION = 3
_BATCHED_SVD_VERSION = 1


def _convolution_version(rule):
    if rule == "frobenius":
        return _FROBENIUS_CONVOLUTION_VERSION
    if rule == "quintic":
        return _QUINTIC_CONVOLUTION_VERSION
    return _CONVOLUTION_VERSION


def convolution_parameter_groups(groups, modules, *, family, convolution_rule="polar"):
    """Bind kernel updates to module semantics, never infer layout from shape.

    Only DualNorm kernel groups are split. Groups without eligible kernels
    retain their original parameters and fields, including checkpoint layout.
    A bare parameter iterable has no module contract and cannot use this hook.
    """
    if convolution_rule not in ("polar", "frobenius", "quintic"):
        raise ValueError("optimizer convolution rule must be polar, frobenius or quintic")
    if family not in ("dualnorm", "dualnorm_D_only"):
        return groups
    kernels = {}
    for root in modules:
        if root is None:
            continue
        for module in root.modules():
            if not isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                continue
            metadata = {
                "update_version": _convolution_version(convolution_rule),
                "layout": ("conv_transpose2d_in_out" if isinstance(module, torch.nn.ConvTranspose2d)
                           else "conv2d_out_in"),
                "groups": module.groups, "in_channels": module.in_channels,
                "out_channels": module.out_channels, "kernel_size": list(module.kernel_size),
            }
            identifier = id(module.weight)
            if identifier in kernels and kernels[identifier] != metadata:
                raise ValueError("shared convolution weight has incompatible module layouts")
            kernels[identifier] = metadata
    result = []
    for original in groups:
        role = original.get("role", original.get("forge_role", "generator"))
        eligible = role not in ("prior", "table") and (family == "dualnorm" or role == "critic")
        parameters = list(original["params"])
        if not eligible or not any(id(p) in kernels for p in parameters):
            result.append({**original, "params": parameters})
            continue
        if "dualnorm_convolution" in original:
            raise ValueError("module convolution metadata cannot replace explicit group metadata")
        pending = []
        for parameter in parameters:
            metadata = kernels.get(id(parameter))
            if metadata is None:
                pending.append(parameter)
                continue
            if pending:
                result.append({**original, "params": pending})
                pending = []
            result.append({**original, "params": [parameter],
                           "dualnorm_convolution": deepcopy(metadata)})
        if pending:
            result.append({**original, "params": pending})
    return result


def _validate_convolution_group(group, convolution_rule="polar"):
    metadata = group["dualnorm_convolution"]
    fields = {"update_version", "layout", "groups", "in_channels", "out_channels", "kernel_size"}
    if (group["algorithm"] != "dualnorm" or len(group["params"]) != 1
            or not isinstance(metadata, dict) or set(metadata) != fields
            or metadata["update_version"] != _convolution_version(convolution_rule)
            or metadata["layout"] not in ("conv2d_out_in", "conv_transpose2d_in_out")):
        raise ValueError("invalid DualNorm convolution group or update version")
    for key in ("groups", "in_channels", "out_channels"):
        if type(metadata[key]) is not int or metadata[key] <= 0:
            raise ValueError("invalid DualNorm convolution channels or groups")
    groups, inputs, outputs = (metadata[key] for key in ("groups", "in_channels", "out_channels"))
    kernel = metadata["kernel_size"]
    if (inputs % groups or outputs % groups or not isinstance(kernel, list) or len(kernel) != 2
            or any(type(size) is not int or size <= 0 for size in kernel)):
        raise ValueError("invalid DualNorm convolution kernel or channel groups")
    channels = ((outputs, inputs // groups) if metadata["layout"] == "conv2d_out_in"
                else (inputs, outputs // groups))
    parameter = group["params"][0]
    if not parameter.is_floating_point() or tuple(parameter.shape) != (*channels, *kernel):
        raise ValueError("DualNorm convolution metadata differs from the kernel shape")


def _convolution_matrices(tensor, metadata):
    """Yield writable [output/group, input/group] views in storage order."""
    transposed = metadata["layout"] == "conv_transpose2d_in_out"
    channels = metadata["in_channels"] if transposed else metadata["out_channels"]
    per_group = channels // metadata["groups"]
    for group in range(metadata["groups"]):
        for row in range(metadata["kernel_size"][0]):
            for column in range(metadata["kernel_size"][1]):
                matrix = tensor[group * per_group:(group + 1) * per_group, :, row, column]
                yield matrix.T if transposed else matrix


def _convolution_batch(tensor, metadata):
    """Return channel matrices in [group, row, column] order."""
    groups = metadata["groups"]
    inputs = metadata["in_channels"] // groups
    outputs = metadata["out_channels"] // groups
    kernel = metadata["kernel_size"]
    transposed = metadata["layout"] == "conv_transpose2d_in_out"
    channels = (inputs, outputs) if transposed else (outputs, inputs)
    storage = tensor.reshape(groups, *channels, *kernel)
    return storage.permute(0, 3, 4, 2, 1) if transposed else storage.permute(0, 3, 4, 1, 2)


def _polar_initial(oriented, smoothing):
    width = oriented.shape[-1]
    identity = torch.eye(width, device=oriented.device, dtype=oriented.dtype)
    normalizer = oriented.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(smoothing)
    scaled = oriented / normalizer
    regularizer = smoothing / normalizer
    trace = scaled.square().sum(dim=(-2, -1), keepdim=True) + width * regularizer.square()
    result = scaled / trace.sqrt()
    lower = regularizer * identity / trace.sqrt()
    return scaled, regularizer, result, lower, identity


def _polar_block(result, lower, identity):
    for _ in range(4):
        correction = .5 * (3 * identity - result.mT @ result - lower.mT @ lower)
        result = result @ correction
        lower = lower @ correction
    return result, lower


def _polar_failed(result, lower, scaled, regularizer, identity):
    residual = (identity - result.mT @ result - lower.mT @ lower).abs().sum(-1).amax(-1)
    # Orthogonality alone is insufficient: shared right updates also preserve
    # smoothing * result == M * lower. Check that original-input relation.
    denominator = regularizer.flatten() * result.norm(dim=(-2, -1))
    relation = ((regularizer * result - scaled @ lower).norm(dim=(-2, -1))
                / denominator.clamp_min(torch.finfo(result.dtype).tiny))
    return (~torch.isfinite(result).all(dim=(-2, -1)) | ~torch.isfinite(residual)
            | (residual > 1e-3) | ~torch.isfinite(relation) | (relation > 1e-3))


class _PolarGraphs:
    """Pure tensor blocks; no parameter, optimizer state or RNG is captured."""

    def __init__(self, oriented, smoothing, *, stream, pool):
        self.scaled, self.regularizer, self.result, self.lower, self.identity = [
            value.clone() for value in _polar_initial(oriented, smoothing)]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Establish cuBLAS workspaces outside capture on the shared stream.
            for _ in range(3):
                _polar_block(self.result, self.lower, self.identity)
                _polar_failed(self.result, self.lower, self.scaled, self.regularizer, self.identity)
        torch.cuda.current_stream().wait_stream(stream)
        self.block = torch.cuda.CUDAGraph()
        self.guard = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.block, stream=stream, pool=pool):
            result, lower = _polar_block(self.result, self.lower, self.identity)
            self.result.copy_(result)
            self.lower.copy_(lower)
        with torch.cuda.graph(self.guard, stream=stream, pool=pool):
            self.failed = _polar_failed(self.result, self.lower, self.scaled,
                                        self.regularizer, self.identity)
        torch.cuda.current_stream().wait_stream(stream)

    def solve(self, oriented, smoothing):
        initial = _polar_initial(oriented, smoothing)
        for target, source in zip((self.scaled, self.regularizer, self.result, self.lower), initial):
            target.copy_(source)
        for iteration in range(4, 33, 4):
            self.block.replay()
            if iteration >= 8:
                self.guard.replay()
                if not bool(self.failed.any()):
                    break
        return self.result, self.failed


class _PolarGraphCache:
    """Optimizer-local graph ownership, including one workspace stream/device.

    Live entries retain their shared pool. Recreating a pool handle after all
    graphs have been freed is required by the CUDA allocator; this entire cache
    is discarded on checkpoint restore/deepcopy rather than partially evicted.
    """

    def __init__(self):
        self.entries, self.resources = {}, {}

    def solve(self, oriented, smoothing):
        key = (oriented.device, oriented.dtype, tuple(oriented.shape), smoothing,
               torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision(),
               torch.are_deterministic_algorithms_enabled(),
               torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        with torch.cuda.device(oriented.device):
            if key not in self.entries:
                if oriented.device not in self.resources:
                    self.resources[oriented.device] = (torch.cuda.Stream(device=oriented.device),
                                                       torch.cuda.graph_pool_handle())
                stream, pool = self.resources[oriented.device]
                self.entries[key] = _PolarGraphs(oriented, smoothing, stream=stream, pool=pool)
            return self.entries[key].solve(oriented, smoothing)


def _newton_schulz_polar(matrices, *, smoothing, cache=None):
    """Batched regularized polar approximation, bounded at 32 cubic iterations.

    Orthogonalize [M; smoothing I] using the smaller channel Gram coordinates.
    Check orthogonality and the original-input relation every four iterations
    starting at eight; each residual limit is 1e-3. Failed matrices fall back
    individually to native SVD. Optimizer-owned CUDA graphs replay the same
    four-iteration tensor block and guards without recording model state or RNG.

    This explicitly different numerical rule retains absolute smoothing but
    has iteration/rounding error and omits native SVD's hard rank cutoff on its
    successful branch. For ill-conditioned directions that cutoff difference
    can be material; quality qualification is separate from solver convergence.
    """
    original_shape, original_dtype = matrices.shape, matrices.dtype
    value = matrices.reshape(-1, *original_shape[-2:])
    if original_dtype not in (torch.float32, torch.float64):
        value = value.float()
    oriented = value.mT if value.shape[-2] < value.shape[-1] else value
    oriented = oriented.contiguous()
    if cache is not None:
        result, failed = cache.solve(oriented, smoothing)
    else:
        scaled, regularizer, result, lower, identity = _polar_initial(oriented, smoothing)
        for iteration in range(4, 33, 4):
            result, lower = _polar_block(result, lower, identity)
            if iteration >= 8:
                failed = _polar_failed(result, lower, scaled, regularizer, identity)
                if not bool(failed.any()):
                    break
    # Host decisions are made per kernel batch rather than per spatial offset.
    # Every rejected slice retains the original exact SVD update.
    for index in failed.nonzero(as_tuple=False).flatten().tolist():
        result[index] = polar_factor(oriented[index], smoothing=smoothing)
    if value.shape[-2] < value.shape[-1]:
        result = result.mT
    return result.reshape(original_shape).to(dtype=original_dtype)


def polar_factor(matrix, *, smoothing=0., backend="native"):
    """Return the polar direction with optional fixed-scale spectral smoothing.

    tau = max(rows, columns) * eps * s_max uses the SVD computation dtype.
    With smoothing > 0, singular weights are s / hypot(s, smoothing).
    Otherwise retained weights are one. Weights at or below tau stay zero.
    Float16/bfloat16 inputs retain the existing float32 computation policy;
    float32/float64 inputs are not cast. Exact SVD at every size makes the
    cutoff independent of an iterative polar approximation. No RNG is used.
    The opt-in CPU backend computes the entire polar operation on CPU, then
    returns to the input device/dtype. Accelerator rounding can differ; this
    is a declared numerical trainer change rather than a parity optimization.
    """
    if type(smoothing) not in (int, float) or not math.isfinite(smoothing) or smoothing < 0:
        raise ValueError("polar smoothing must be finite and nonnegative")
    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError("polar_factor requires a floating-point matrix")
    if backend not in ("native", "cpu"):
        raise ValueError("polar backend must be native or cpu")
    if backend == "cpu" and matrix.device.type != "cpu":
        return polar_factor(matrix.cpu(), smoothing=smoothing).to(matrix)
    original_dtype = matrix.dtype
    value = matrix if original_dtype in (torch.float32, torch.float64) else matrix.float()
    if 0 in value.shape:
        return torch.zeros_like(matrix)
    left, singular, right = torch.linalg.svd(value, full_matrices=False)
    threshold = max(value.shape) * torch.finfo(value.dtype).eps * singular[0]
    if smoothing:
        weights = singular / torch.hypot(singular, torch.full_like(singular, smoothing))
        weights = weights * (singular > threshold)
        return ((left * weights) @ right).to(dtype=original_dtype)
    return ((left * (singular > threshold)) @ right).to(dtype=original_dtype)


def _batched_polar(matrices, *, smoothing):
    """Native smoothed SVD law in a single batch of small channel matrices.

    CUDA may choose a different SVD kernel for batched inputs. This is an
    explicit backend, with the same per-matrix rank cutoff and absolute
    smoothing, rather than a bitwise equivalence promise.
    """
    dtype = matrices.dtype
    value = matrices if dtype in (torch.float32, torch.float64) else matrices.float()
    left, singular, right = torch.linalg.svd(value, full_matrices=False)
    threshold = max(value.shape[-2:]) * torch.finfo(value.dtype).eps * singular[..., :1]
    weights = singular > threshold
    if smoothing:
        weights = weights * (singular / torch.hypot(singular, torch.full_like(singular, smoothing)))
    return ((left * weights.unsqueeze(-2)) @ right).to(dtype=dtype)


class NormalizedOptimizer(Optimizer):
    """Role-aware optimizer, with native Adam only in the isolation arms.

    Groups use ``role='generator'|'encoder'|'critic'|'prior'``. A prior table
    must be alone in its group. For row-normalized prior updates, call
    ``set_sampled_rows(table, indices)`` with the actual generator-side draws
    before ``step``. Unsampled rows stay fixed even when standardization or
    a regularizer creates dense gradients. Multiple calls union sampled rows.
    A deliberately direct table group can set ``sampled_rows_required=False``:
    every direct row participates rather than being sampled. Recipe's ordinary
    ``direct_particles`` fixture instead belongs to the generator player.
    """

    def __init__(self, params, *, family="dualnorm", lr=.01, betas=(0., .999),
                 eps=1e-8, momentum=0., amsgrad=False, smoothing=0., convolution="none",
                 svd_backend="native", convolution_rule="polar", dense_profile="default", **options):
        if family not in NORMALIZED_FAMILIES:
            raise ValueError("unknown normalized optimizer family")
        if isinstance(momentum, bool) or momentum not in (0., .5, .9):
            raise ValueError("optimizer momentum must be 0, 0.5 or 0.9")
        if momentum != 0 and family not in ("dualnorm", "dualnorm_D_only"):
            raise ValueError("momentum is supported only by dualnorm arms")
        if type(smoothing) not in (int, float) or not math.isfinite(smoothing) or smoothing < 0:
            raise ValueError("optimizer smoothing must be finite and nonnegative")
        if smoothing and family != "dualnorm":
            raise ValueError("optimizer smoothing requires the dualnorm family")
        if convolution not in ("none", "per_offset"):
            raise ValueError("optimizer convolution must be none or per_offset")
        if convolution != "none" and family != "dualnorm":
            raise ValueError("optimizer convolution requires the dualnorm family")
        if svd_backend not in ("native", "cpu", "newton_schulz", "batched"):
            raise ValueError("optimizer SVD backend must be native, cpu, newton_schulz or batched")
        if svd_backend != "native" and family != "dualnorm":
            raise ValueError("optimizer SVD backend requires the dualnorm family")
        if svd_backend == "newton_schulz" and (convolution != "per_offset" or not smoothing):
            raise ValueError("newton_schulz requires per_offset convolution and positive smoothing")
        if svd_backend == "batched" and convolution != "per_offset":
            raise ValueError("batched SVD requires per_offset convolution")
        if type(dense_profile) is not str or dense_profile not in ("default", GRAM_DENSE_PROFILE, FALLBACK_DENSE_PROFILE):
            raise ValueError("unknown optimizer dense_profile")
        if dense_profile != "default" and (family != "dualnorm" or momentum != 0
                or smoothing != .001 or svd_backend not in ("native", "batched")):
            raise ValueError("dense_profile requires zero-momentum full DualNorm, smoothing .001 "
                             "and native or batched SVD backend")
        if convolution_rule not in ("polar", "frobenius", "quintic"):
            raise ValueError("optimizer convolution rule must be polar, frobenius or quintic")
        if convolution_rule != "polar" and (
                family != "dualnorm" or convolution != "per_offset" or svd_backend not in ("native", "batched")):
            raise ValueError("nonpolar rules require full dualnorm, per_offset convolution and native or batched SVD backend")
        if isinstance(lr, bool) or not math.isfinite(lr) or lr < 0:
            raise ValueError("optimizer step size must be finite and nonnegative")
        if isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("optimizer epsilon must be finite and positive")
        betas = tuple(float(b) for b in betas)
        if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        for key, value in options.items():
            if key not in {"weight_decay", "maximize", "differentiable", "capturable", "foreach", "fused"}:
                raise TypeError(f"unsupported normalized optimizer option: {key}")
            if value not in (None, False, 0):
                raise ValueError(f"normalized optimizers require disabled {key}")
        self.family, self.momentum = family, float(momentum)
        self.smoothing = float(smoothing)
        self.convolution = convolution
        self.convolution_rule = convolution_rule
        self.svd_backend = svd_backend
        self.dense_profile = dense_profile
        self._dense_profile_leaf_counters = {}
        self.recipe_optimizer_family = family
        self._sampled_rows = {}
        defaults = {"lr": float(lr), "betas": betas, "eps": float(eps), "amsgrad": bool(amsgrad),
                    "role": "generator", "weight_decay": 0., "maximize": False,
                    "differentiable": False, "capturable": False, "foreach": None,
                    "fused": None, **options}
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._configure_group(group)
        self._validate_dense_profile_parameters()
        self._adam = None
        adam_groups = [group for group in self.param_groups if group["algorithm"] == "adam"]
        if adam_groups:
            self._adam = torch.optim.Adam(adam_groups, lr=lr, betas=betas, eps=eps,
                                          amsgrad=amsgrad, **options)
            self._refresh_adam()
        self.latent_damping = self.latent_history = None
        self.direct_response = self.direct_history = None

    def _validate_dense_profile_parameters(self):
        if self.dense_profile == "default":
            return
        for group in self.param_groups:
            if group["algorithm"] != "dualnorm" or "dualnorm_convolution" in group:
                continue
            for parameter in group["params"]:
                if parameter.ndim == 2 and (parameter.dtype != torch.float32
                        or parameter.device.type not in ("cpu", "cuda")
                        or parameter.layout != torch.strided):
                    raise ValueError("dense_profile requires ordinary dense FP32 CPU/CUDA parameters")

    def _configure_group(self, group):
        role = group["role"]
        if role not in _ROLES:
            raise ValueError(f"unsupported optimizer role: {role}")
        if isinstance(group["lr"], bool) or not math.isfinite(group["lr"]) or group["lr"] < 0:
            raise ValueError("optimizer group step size must be finite and nonnegative")
        if isinstance(group["eps"], bool) or not math.isfinite(group["eps"]) or group["eps"] <= 0:
            raise ValueError("optimizer group epsilon must be finite and positive")
        group["betas"] = tuple(float(beta) for beta in group["betas"])
        if len(group["betas"]) != 2 or any(not 0 <= beta < 1 for beta in group["betas"]):
            raise ValueError("optimizer group betas must contain two values in [0, 1)")
        for key in ("weight_decay", "maximize", "differentiable", "capturable", "foreach", "fused"):
            if group[key] not in (None, False, 0):
                raise ValueError(f"normalized optimizer groups require disabled {key}")
        if self.family == "dualnorm_D_only":
            algorithm = "dualnorm" if role == "critic" else "adam"
        elif self.family == "particle_rownorm_only":
            algorithm = "rownorm" if role in ("prior", "table") else "adam"
        else:
            algorithm = self.family
        if algorithm == "dualnorm" and role in ("prior", "table"):
            algorithm = "rownorm"
        group["algorithm"] = algorithm
        if algorithm == "rownorm":
            if len(group["params"]) != 1 or group["params"][0].ndim != 2:
                raise ValueError("the prior table requires its own two-dimensional parameter group")
            group.setdefault("sampled_rows_required", True)
        if algorithm == "ada_nsgda" and group["betas"][0] != 0:
            raise ValueError("ada_nsgda requires beta1=0 in every group")
        if "dualnorm_convolution" in group:
            if self.convolution != "per_offset":
                raise ValueError("DualNorm convolution metadata requires convolution='per_offset'")
            _validate_convolution_group(group, self.convolution_rule)
        elif algorithm == "dualnorm" and any(p.ndim > 2 for p in group["params"]):
            raise ValueError("dualnorm high-rank weights require explicit Conv2d/ConvTranspose2d module metadata")

    def _refresh_adam(self):
        adam_groups = [g for g in self.param_groups if g["algorithm"] == "adam"]
        if adam_groups and self._adam is None:
            # An isolation arm may start with only a row-normalized table and
            # acquire trainable network/noise parameters later. Construct its
            # native component on first use, without introducing Adam into a
            # family whose groups all follow normalized rules.
            keys = ("lr", "betas", "eps", "amsgrad", "weight_decay", "maximize",
                    "differentiable", "capturable", "foreach", "fused")
            self._adam = torch.optim.Adam(adam_groups, **{key: self.defaults[key] for key in keys})
        if self._adam is not None:
            self._adam.param_groups = adam_groups
            self._adam.state = self.state

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        self._configure_group(self.param_groups[-1])
        if hasattr(self, "_adam"):
            self._refresh_adam()

    def __getstate__(self):
        # Optimizer's default pickle omits the role family, observer and native
        # Adam component; public checkpoint preflight also deep-copies us.
        return {key: value for key, value in self.__dict__.items()
                if key not in ("_newton_graphs", "_quintic_graphs", "_dense_profile_leaf_counters")}

    @property
    def dense_profile_stats(self):
        """Detached runtime snapshot; not checkpoint identity/state.

        Keys identify zero-based group:param positions. A successful public load
        resets these diagnostics; rejected loads retain them. HOLD may follow
        earlier leaf writes/clock advances and requires whole-state recovery.
        Primary timing is host wall; fallback timing includes selected-device
        transfer completion. Timers are not independent additive GPU costs.
        """
        per_leaf = {key: asdict(value) for key, value in
                    getattr(self, "_dense_profile_leaf_counters", {}).items()}
        aggregate = {key: value for key, value in asdict(Counters()).items()
                     if key != "last_fallback"}
        for counters in per_leaf.values():
            for key in aggregate:
                aggregate[key] += counters[key]
        return {"profile": self.dense_profile, "aggregate": aggregate,
                "per_leaf": per_leaf, "serialized": False,
                "reset_on_successful_load": True}

    def set_sampled_rows(self, table, rows):
        """Record actual sampled indices, never infer ownership from gradients."""
        group = next((g for g in self.param_groups if any(p is table for p in g["params"])), None)
        if group is None:
            raise ValueError("sampled table is not owned by this optimizer")
        if group["algorithm"] != "rownorm":
            return
        if not isinstance(rows, torch.Tensor) or rows.dtype != torch.long or rows.ndim != 1:
            raise ValueError("sampled rows must be a one-dimensional int64 tensor")
        rows = rows.detach().to(device=table.device)
        if rows.numel() and (bool((rows < 0).any()) or bool((rows >= len(table)).any())):
            raise ValueError("sampled row index outside the prior table")
        previous = self._sampled_rows.get(table)
        self._sampled_rows[table] = torch.unique(rows if previous is None else torch.cat((previous, rows)))

    def clear_sampled_rows(self):
        """Discard an aborted update's row ownership."""
        self._sampled_rows.clear()

    def sampled_rows_for(self, table):
        """Return a read-only copy of the pending step's sampled row IDs."""
        rows = self._sampled_rows.get(table)
        return None if rows is None else rows.detach().clone()

    @staticmethod
    def _player(role):
        return "prior" if role in ("prior", "table") else ("critic" if role == "critic" else "generator")

    @torch.no_grad()
    def step(self, closure=None):
        self._validate_dense_profile_parameters()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Fail before mutating any parameters if ownership is absent or the
        # chosen rule cannot consume a gradient.
        for group in self.param_groups:
            if "dualnorm_convolution" in group:
                _validate_convolution_group(group, self.convolution_rule)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse or parameter.grad.is_complex():
                    raise ValueError("normalized optimizers require dense real gradients")
                if (group["algorithm"] == "rownorm" and group["sampled_rows_required"]
                        and parameter not in self._sampled_rows):
                    raise ValueError("row-normalized prior updates require actual sampled rows")
        norms = {}
        for group in self.param_groups:
            if group["algorithm"] == "nsgda_global":
                player = self._player(group["role"])
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        squared = parameter.grad.square().sum()
                        norms[player] = squared if player not in norms else norms[player] + squared
        norms = {player: value.sqrt() for player, value in norms.items()}
        for group_index, group in enumerate(self.param_groups):
            algorithm, rate, eps = group["algorithm"], group["lr"], group["eps"]
            for parameter_index, parameter in enumerate(group["params"]):
                gradient = parameter.grad
                if gradient is None or algorithm == "adam":
                    continue
                state = self.state[parameter]
                state["step"] = state.get("step", 0) + 1
                if algorithm == "sgda":
                    parameter.add_(gradient, alpha=-rate)
                elif algorithm == "nsgda_global":
                    parameter.add_(gradient / (norms[self._player(group["role"])] + eps), alpha=-rate)
                elif algorithm == "nsgda_layer":
                    parameter.add_(gradient / (gradient.norm() + eps), alpha=-rate)
                elif algorithm == "ada_nsgda":
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(parameter)
                    beta2 = group["betas"][1]
                    second = state["exp_avg_sq"]
                    second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                    if group["amsgrad"]:
                        torch.maximum(state["max_exp_avg_sq"], second, out=state["max_exp_avg_sq"])
                        second = state["max_exp_avg_sq"]
                    adam_direction = gradient / ((second / (1 - beta2 ** state["step"])).sqrt() + eps)
                    parameter.add_(gradient / (gradient.norm() + eps) * adam_direction.norm(), alpha=-rate)
                elif algorithm == "rownorm":
                    rows = (self._sampled_rows[parameter] if group["sampled_rows_required"]
                            else torch.arange(len(parameter), device=parameter.device))
                    selected = gradient[rows]
                    norm = selected.norm(dim=1, keepdim=True)
                    denominator = (torch.hypot(norm, torch.full_like(norm, self.smoothing))
                                   if self.smoothing else norm + eps)
                    update = selected / denominator
                    parameter.index_add_(0, rows, update, alpha=-rate)
                elif algorithm == "dualnorm":
                    direction = gradient
                    if self.momentum:
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(parameter)
                        direction = state["momentum_buffer"]
                        direction.mul_(self.momentum).add_(gradient)
                    if "dualnorm_convolution" in group:
                        metadata = group["dualnorm_convolution"]
                        factor = math.sqrt(metadata["out_channels"] / metadata["in_channels"])
                        factor /= math.prod(metadata["kernel_size"])
                        largest_channel = max(metadata["in_channels"], metadata["out_channels"]) // metadata["groups"]
                        if (self.svd_backend == "batched" and parameter.device.type == "cuda"
                                and largest_channel <= 32):
                            gradient_batch = _convolution_batch(gradient, metadata)
                            direction_batch = _convolution_batch(direction, metadata)
                            update = _batched_polar(direction_batch, smoothing=self.smoothing)
                            active = ((gradient_batch.norm(dim=(-2, -1)) >= eps)
                                      & (direction_batch.norm(dim=(-2, -1)) >= eps))
                            update = torch.where(active[..., None, None], update, 0.)
                            if metadata["layout"] == "conv_transpose2d_in_out":
                                update = update.permute(0, 4, 3, 1, 2)
                            else:
                                update = update.permute(0, 3, 4, 1, 2)
                            parameter.add_(update.reshape_as(parameter), alpha=-rate * factor)
                            continue
                        if (self.convolution_rule != "polar" and parameter.device.type == "cuda"
                                and largest_channel > 32):
                            gradient_batch = _convolution_batch(gradient, metadata)
                            direction_batch = _convolution_batch(direction, metadata)
                            if self.convolution_rule == "frobenius":
                                update = frobenius_direction(direction_batch, smoothing=self.smoothing)
                            else:
                                from .convolution_quintic import QuinticGraphCache, quintic_direction
                                if self.svd_backend == "batched":
                                    if getattr(self, "_quintic_graphs", None) is None:
                                        self._quintic_graphs = QuinticGraphCache()
                                    update = self._quintic_graphs.direction(direction_batch, smoothing=self.smoothing)
                                else:
                                    update = quintic_direction(direction_batch, smoothing=self.smoothing)
                            active = ((gradient_batch.norm(dim=(-2, -1)) >= eps)
                                      & (direction_batch.norm(dim=(-2, -1)) >= eps))
                            update = torch.where(active[..., None, None], update, 0.)
                            if metadata["layout"] == "conv_transpose2d_in_out":
                                update = update.permute(0, 4, 3, 1, 2)
                            else:
                                update = update.permute(0, 3, 4, 1, 2)
                            parameter.add_(update.reshape_as(parameter), alpha=-rate * factor)
                            continue
                        if (self.svd_backend == "newton_schulz" and parameter.device.type == "cuda"
                                and largest_channel > 32):
                            gradient_batch = _convolution_batch(gradient, metadata)
                            direction_batch = _convolution_batch(direction, metadata)
                            if getattr(self, "_newton_graphs", None) is None:
                                self._newton_graphs = _PolarGraphCache()
                            update = _newton_schulz_polar(direction_batch, smoothing=self.smoothing,
                                                        cache=self._newton_graphs)
                            active = ((gradient_batch.norm(dim=(-2, -1)) >= eps)
                                      & (direction_batch.norm(dim=(-2, -1)) >= eps))
                            # A skipped slice must remain fixed even when an
                            # extreme, dtype-underflowing smoothing value makes
                            # its unused solver output undefined (NaN * 0 is NaN).
                            update = torch.where(active[..., None, None], update, 0.)
                            if metadata["layout"] == "conv_transpose2d_in_out":
                                update = update.permute(0, 4, 3, 1, 2)
                            else:
                                update = update.permute(0, 3, 4, 1, 2)
                            parameter.add_(update.reshape_as(parameter), alpha=-rate * factor)
                            continue
                        matrices = zip(_convolution_matrices(parameter, metadata),
                                       _convolution_matrices(gradient, metadata),
                                       _convolution_matrices(direction, metadata))
                        for weight_matrix, gradient_matrix, direction_matrix in matrices:
                            # Mirror the dense rule per channel matrix: a zero
                            # current slice does not move on stale momentum.
                            if bool(gradient_matrix.norm() < eps) or bool(direction_matrix.norm() < eps):
                                continue
                            update = (polar_factor(direction_matrix, smoothing=self.smoothing)
                                      if self.svd_backend in ("native", "newton_schulz", "batched") else
                                      polar_factor(direction_matrix, smoothing=self.smoothing,
                                                   backend=self.svd_backend))
                            weight_matrix.add_(update, alpha=-rate * factor)
                    elif parameter.ndim == 2:
                        if bool(gradient.norm() < eps) or bool(direction.norm() < eps):
                            continue
                        factor = math.sqrt(max(1., parameter.shape[0] / parameter.shape[1]))
                        if self.dense_profile == GRAM_DENSE_PROFILE:
                            update = guarded_gram_polar(direction, smoothing=self.smoothing)
                        elif self.dense_profile == FALLBACK_DENSE_PROFILE:
                            if not hasattr(self, "_dense_profile_leaf_counters"):
                                self._dense_profile_leaf_counters = {}
                            counters = self._dense_profile_leaf_counters.setdefault(
                                f"{group_index}:{parameter_index}", Counters())
                            update = svd_fallback_polar(direction, smoothing=self.smoothing,
                                                            counters=counters)
                        elif self.svd_backend in ("native", "newton_schulz", "batched"):
                            update = (polar_factor(direction, smoothing=self.smoothing)
                                      if self.smoothing else polar_factor(direction))
                        else:
                            update = polar_factor(direction, smoothing=self.smoothing,
                                                  backend=self.svd_backend)
                        parameter.add_(update, alpha=-rate * factor)
                    else:
                        norm = direction.norm()
                        denominator = (torch.hypot(norm, norm.new_tensor(self.smoothing))
                                       if self.smoothing else norm + eps)
                        parameter.add_(direction / denominator, alpha=-rate)
        if self._adam is not None:
            self._adam.step()
        self.clear_sampled_rows()
        if hasattr(self, "record"):
            self.record.record_step(self)
        return loss

    def state_dict(self):
        result = super().state_dict()
        result["dualnorm"] = {
            "schema": 1, "family": self.family, "momentum": self.momentum,
            "sampled_rows": [self._sampled_rows.get(group["params"][0])
                             if group["algorithm"] == "rownorm" else None
                             for group in self.param_groups],
        }
        if self.smoothing:
            result["dualnorm"]["smoothing"] = self.smoothing
        if self.convolution != "none":
            result["dualnorm"]["convolution"] = self.convolution
        if self.convolution_rule != "polar":
            result["dualnorm"]["convolution_rule"] = self.convolution_rule
        if self.svd_backend != "native":
            result["dualnorm"]["svd_backend"] = self.svd_backend
        if self.svd_backend == "newton_schulz":
            result["dualnorm"]["solver_version"] = _NEWTON_SCHULZ_VERSION
        if self.svd_backend == "batched":
            result["dualnorm"]["solver_version"] = _BATCHED_SVD_VERSION
        if self.dense_profile != "default":
            result["dualnorm"]["dense_profile"] = self.dense_profile
        if hasattr(self, "record"):
            result["regularizer"] = {"optimizer_family": self.family,
                                     "record": self.record.state_dict(), "ema": None, "guard": None}
        return result

    def validate_state_dict(self, saved):
        """Fail closed on recipe/role changes and malformed update histories."""
        expected_keys = {"state", "param_groups", "dualnorm"}
        if hasattr(self, "record"):
            expected_keys.add("regularizer")
        if not isinstance(saved, dict) or saved.keys() != expected_keys:
            raise ValueError("invalid normalized optimizer checkpoint schema")
        meta = saved["dualnorm"]
        if isinstance(meta, dict) and meta.get("dense_profile", "default") != self.dense_profile:
            raise ValueError("checkpoint optimizer dense profile differs")
        expected_meta = {"schema", "family", "momentum", "sampled_rows"}
        if self.smoothing:
            expected_meta.add("smoothing")
        if self.convolution != "none":
            expected_meta.add("convolution")
        if self.convolution_rule != "polar":
            expected_meta.add("convolution_rule")
        if self.svd_backend != "native":
            expected_meta.add("svd_backend")
        if self.svd_backend == "newton_schulz":
            expected_meta.add("solver_version")
            if not isinstance(meta, dict) or meta.get("solver_version") != _NEWTON_SCHULZ_VERSION:
                raise ValueError("checkpoint Newton-Schulz solver version differs")
        if self.svd_backend == "batched":
            expected_meta.add("solver_version")
            if not isinstance(meta, dict) or meta.get("solver_version") != _BATCHED_SVD_VERSION:
                raise ValueError("checkpoint batched SVD version differs")
        if self.dense_profile != "default":
            expected_meta.add("dense_profile")
        if (not isinstance(meta, dict) or set(meta) != expected_meta
                or meta["schema"] != 1 or meta["family"] != self.family or meta["momentum"] != self.momentum):
            raise ValueError("checkpoint normalized optimizer family or momentum differs")
        if meta.get("smoothing", 0.) != self.smoothing:
            raise ValueError("checkpoint optimizer smoothing differs")
        if meta.get("convolution", "none") != self.convolution:
            raise ValueError("checkpoint optimizer convolution differs")
        if meta.get("convolution_rule", "polar") != self.convolution_rule:
            raise ValueError("checkpoint optimizer convolution rule differs")
        if meta.get("svd_backend", "native") != self.svd_backend:
            raise ValueError("checkpoint optimizer SVD backend differs")
        if meta.get("dense_profile", "default") != self.dense_profile:
            raise ValueError("checkpoint optimizer dense profile differs")
        groups, values = saved["param_groups"], saved["state"]
        if (not isinstance(groups, list) or len(groups) != len(self.param_groups)
                or not isinstance(values, dict) or not isinstance(meta["sampled_rows"], list)
                or len(meta["sampled_rows"]) != len(groups)):
            raise ValueError("invalid normalized optimizer groups")
        seen = set()
        for index, (actual, group) in enumerate(zip(self.param_groups, groups)):
            if "dualnorm_convolution" in actual:
                _validate_convolution_group(actual, self.convolution_rule)
            if not isinstance(group, dict) or set(group) != set(actual) or not isinstance(group.get("params"), list):
                raise ValueError("invalid normalized optimizer group fields")
            if len(group["params"]) != len(actual["params"]):
                raise ValueError("checkpoint normalized optimizer group sizes differ")
            for key in set(actual) - {"params", "lr"}:
                if group[key] != actual[key]:
                    raise ValueError(f"checkpoint normalized optimizer group {key} differs")
            if isinstance(group["lr"], bool) or not math.isfinite(group["lr"]) or group["lr"] < 0:
                raise ValueError("invalid normalized optimizer step size")
            rows = meta["sampled_rows"][index]
            if rows is not None:
                if (actual["algorithm"] != "rownorm" or not isinstance(rows, torch.Tensor)
                        or rows.dtype != torch.long or rows.ndim != 1
                        or (rows.numel() and (bool((rows < 0).any()) or bool((rows >= len(actual["params"][0])).any())))
                        or len(torch.unique(rows)) != len(rows)):
                    raise ValueError("invalid checkpoint sampled rows")
            for identifier, parameter in zip(group["params"], actual["params"]):
                if type(identifier) is not int or identifier in seen:
                    raise ValueError("invalid normalized optimizer parameter IDs")
                seen.add(identifier)
                state = values.get(identifier, {})
                if not isinstance(state, dict):
                    raise ValueError("invalid normalized optimizer parameter history")
                if not state:
                    continue
                algorithm = actual["algorithm"]
                required = {"step"}
                if algorithm == "adam":
                    required = {"step", "exp_avg", "exp_avg_sq"}
                elif algorithm == "ada_nsgda":
                    required = {"step", "exp_avg_sq"}
                elif algorithm == "dualnorm" and self.momentum:
                    required.add("momentum_buffer")
                if algorithm in ("adam", "ada_nsgda") and actual["amsgrad"]:
                    required.add("max_exp_avg_sq")
                if state.keys() != required:
                    raise ValueError("invalid normalized optimizer history fields")
                for key in required - {"step"}:
                    tensor = state[key]
                    if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                            or tensor.dtype != parameter.dtype or not bool(torch.isfinite(tensor).all())):
                        raise ValueError(f"invalid normalized optimizer {key} tensor")
                    if key in ("exp_avg_sq", "max_exp_avg_sq") and bool((tensor < 0).any()):
                        raise ValueError("invalid negative second-moment history")
                if "step" in state:
                    step = state["step"]
                    if isinstance(step, torch.Tensor):
                        if step.numel() != 1 or not bool(torch.isfinite(step).all()) or bool(step < 0) or bool(step != step.floor()):
                            raise ValueError("invalid normalized optimizer step count")
                    elif type(step) is not int or step < 0:
                        raise ValueError("invalid normalized optimizer step count")
        if not values.keys() <= seen:
            raise ValueError("unknown normalized optimizer parameter history")
        if hasattr(self, "record"):
            regularizer = saved["regularizer"]
            if (not isinstance(regularizer, dict) or set(regularizer) != {"optimizer_family", "record", "ema", "guard"}
                    or regularizer["optimizer_family"] != self.family
                    or regularizer["ema"] is not None or regularizer["guard"] is not None):
                raise ValueError("invalid normalized critic observer")
            record = regularizer["record"]
            if hasattr(self.record, "base_state"):
                record = self.record.base_state(record)
            if not isinstance(record, dict) or set(record) != set(CriticStepRecord.KEYS):
                raise ValueError("invalid normalized critic record")
            if record["anchor_started"] is not False or any(type(record[k]) is not int or record[k] < 0 for k in ("calls", "observed_steps")):
                raise ValueError("invalid normalized critic counters")
            for key in ("lr_max", "lr_last"):
                rate = record[key]
                if key == "lr_last" and rate is None:
                    continue
                if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
                    raise ValueError("invalid normalized critic learning-rate record")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        state = deepcopy(state)
        metadata = state.pop("dualnorm")
        regularizer = state.pop("regularizer", None)
        super().load_state_dict(state)
        self.__dict__.pop("_newton_graphs", None)
        self.__dict__.pop("_quintic_graphs", None)
        self._refresh_adam()
        self.clear_sampled_rows()
        for group, rows in zip(self.param_groups, metadata["sampled_rows"]):
            if rows is not None:
                self.set_sampled_rows(group["params"][0], rows)
        if regularizer is not None:
            self.record.load_state_dict(regularizer["record"])
            if hasattr(self.record, "restore_device_state"):
                self.record.restore_device_state(self.param_groups[0]["params"][0].device)

        self._dense_profile_leaf_counters = {}


def make_normalized_optimizer(recipe, params, *, critic=None, **options):
    """Recipe factory plus the same observation-only penalty metadata as Adam."""
    optimizer_class = NormalizedOptimizer
    if critic is not None and recipe.critic_step_mode == "finite_cap":
        from .critic_cap import CriticCapOptimizer
        optimizer_class = CriticCapOptimizer
    if recipe.constraint_geometry_mode == "nonascent" and critic is None:
        from .constraint_geometry import ConstraintGeometryOptimizer
        optimizer_class = ConstraintGeometryOptimizer
    elif recipe.constraint_geometry_mode == "strict_progress" and critic is None:
        from .strict_progress import StrictProgressOptimizer
        optimizer_class = StrictProgressOptimizer
    elif recipe.constraint_geometry_mode == "direction_blend" and critic is None:
        from .direction_blend import DirectionBlendOptimizer
        optimizer_class = DirectionBlendOptimizer
    if recipe.prior_proximal_strength and recipe.prior_update == "learned" and critic is None:
        from .prior_proximal import proximal_optimizer_class
        if any(group.get("role", "generator") in ("prior", "table") for group in params):
            optimizer_class = proximal_optimizer_class(optimizer_class)
            options["prior_proximal_strength"] = recipe.prior_proximal_strength
    optimizer = optimizer_class(params, family=recipe.optimizer_family,
                                    momentum=recipe.optimizer_momentum,
                                    smoothing=recipe.optimizer_smoothing,
                                    convolution=recipe.optimizer_convolution,
                                    convolution_rule=recipe.optimizer_convolution_rule,
                                    svd_backend=recipe.optimizer_svd_backend,
                                    dense_profile=recipe.optimizer_dense_profile, **options)
    if critic is not None:
        optimizer.critic = critic
        optimizer.ema_critic = optimizer.anchor = optimizer.guard = None
        if recipe.effective_critic_formulation == "bispf":
            from ..bispf import _BISPFObserverStepRecord
            optimizer.record = _BISPFObserverStepRecord()
        else:
            optimizer.record = CriticStepRecord()
    return optimizer
