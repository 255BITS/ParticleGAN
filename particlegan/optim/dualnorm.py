"""SGDA, normalized steps and spectral dual-norm steps for fixed GAN recipes.

Learning rates are step sizes in each algorithm's own units. G and E belong
to one player for global normalization; the learned prior is a separate
player. The graft uses the norm of Adam's *unit-learning-rate* direction,
so the scheduled learning rate is applied exactly once.
"""
from copy import deepcopy
import math

import torch
from torch.optim import Optimizer

from ..grad_regularizers import CriticStepRecord


NORMALIZED_FAMILIES = (
    "sgda", "nsgda_global", "nsgda_layer", "ada_nsgda", "dualnorm",
    "dualnorm_D_only", "particle_rownorm_only",
)
_ROLES = {"generator", "encoder", "router", "noise", "critic", "prior", "table"}
_CONVOLUTION_VERSION = "per_offset_polar_v1"


def convolution_parameter_groups(groups, modules, *, family):
    """Bind kernel updates to module semantics, never infer layout from shape.

    Only DualNorm kernel groups are split. Groups without eligible kernels
    retain their original parameters and fields, including checkpoint layout.
    A bare parameter iterable has no module contract and cannot use this hook.
    """
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
                "update_version": _CONVOLUTION_VERSION,
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


def _validate_convolution_group(group):
    metadata = group["dualnorm_convolution"]
    fields = {"update_version", "layout", "groups", "in_channels", "out_channels", "kernel_size"}
    if (group["algorithm"] != "dualnorm" or len(group["params"]) != 1
            or not isinstance(metadata, dict) or set(metadata) != fields
            or metadata["update_version"] != _CONVOLUTION_VERSION
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


def polar_factor(matrix, *, smoothing=0.):
    """Return the polar direction with optional fixed-scale spectral smoothing.

    tau = max(rows, columns) * eps * s_max uses the SVD computation dtype.
    With smoothing > 0, singular weights are s / hypot(s, smoothing).
    Otherwise retained weights are one. Weights at or below tau stay zero.
    Float16/bfloat16 inputs retain the existing float32 computation policy;
    float32/float64 inputs are not cast. Exact SVD at every size makes the
    cutoff independent of an iterative polar approximation. No RNG is used.
    """
    if type(smoothing) not in (int, float) or not math.isfinite(smoothing) or smoothing < 0:
        raise ValueError("polar smoothing must be finite and nonnegative")
    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError("polar_factor requires a floating-point matrix")
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
                 eps=1e-8, momentum=0., amsgrad=False, smoothing=0., convolution="none", **options):
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
        self.recipe_optimizer_family = family
        self._sampled_rows = {}
        defaults = {"lr": float(lr), "betas": betas, "eps": float(eps), "amsgrad": bool(amsgrad),
                    "role": "generator", "weight_decay": 0., "maximize": False,
                    "differentiable": False, "capturable": False, "foreach": None,
                    "fused": None, **options}
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._configure_group(group)
        self._adam = None
        adam_groups = [group for group in self.param_groups if group["algorithm"] == "adam"]
        if adam_groups:
            self._adam = torch.optim.Adam(adam_groups, lr=lr, betas=betas, eps=eps,
                                          amsgrad=amsgrad, **options)
            self._refresh_adam()
        self.latent_damping = self.latent_history = None
        self.direct_response = self.direct_history = None

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
            _validate_convolution_group(group)
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
        return dict(self.__dict__)

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
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Fail before mutating any parameters if ownership is absent or the
        # chosen rule cannot consume a gradient.
        for group in self.param_groups:
            if "dualnorm_convolution" in group:
                _validate_convolution_group(group)
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
        for group in self.param_groups:
            algorithm, rate, eps = group["algorithm"], group["lr"], group["eps"]
            for parameter in group["params"]:
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
                        matrices = zip(_convolution_matrices(parameter, metadata),
                                       _convolution_matrices(gradient, metadata),
                                       _convolution_matrices(direction, metadata))
                        for weight_matrix, gradient_matrix, direction_matrix in matrices:
                            # Mirror the dense rule per channel matrix: a zero
                            # current slice does not move on stale momentum.
                            if bool(gradient_matrix.norm() < eps) or bool(direction_matrix.norm() < eps):
                                continue
                            update = polar_factor(direction_matrix, smoothing=self.smoothing)
                            weight_matrix.add_(update, alpha=-rate * factor)
                    elif parameter.ndim == 2:
                        if bool(gradient.norm() < eps) or bool(direction.norm() < eps):
                            continue
                        factor = math.sqrt(max(1., parameter.shape[0] / parameter.shape[1]))
                        update = (polar_factor(direction, smoothing=self.smoothing)
                                  if self.smoothing else polar_factor(direction))
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
        expected_meta = {"schema", "family", "momentum", "sampled_rows"}
        if self.smoothing:
            expected_meta.add("smoothing")
        if self.convolution != "none":
            expected_meta.add("convolution")
        if (not isinstance(meta, dict) or set(meta) != expected_meta
                or meta["schema"] != 1 or meta["family"] != self.family or meta["momentum"] != self.momentum):
            raise ValueError("checkpoint normalized optimizer family or momentum differs")
        if meta.get("smoothing", 0.) != self.smoothing:
            raise ValueError("checkpoint optimizer smoothing differs")
        if meta.get("convolution", "none") != self.convolution:
            raise ValueError("checkpoint optimizer convolution differs")
        groups, values = saved["param_groups"], saved["state"]
        if (not isinstance(groups, list) or len(groups) != len(self.param_groups)
                or not isinstance(values, dict) or not isinstance(meta["sampled_rows"], list)
                or len(meta["sampled_rows"]) != len(groups)):
            raise ValueError("invalid normalized optimizer groups")
        seen = set()
        for index, (actual, group) in enumerate(zip(self.param_groups, groups)):
            if "dualnorm_convolution" in actual:
                _validate_convolution_group(actual)
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
        self._refresh_adam()
        self.clear_sampled_rows()
        for group, rows in zip(self.param_groups, metadata["sampled_rows"]):
            if rows is not None:
                self.set_sampled_rows(group["params"][0], rows)
        if regularizer is not None:
            self.record.load_state_dict(regularizer["record"])


def make_normalized_optimizer(recipe, params, *, critic=None, **options):
    """Recipe factory plus the same observation-only penalty metadata as Adam."""
    optimizer_class = NormalizedOptimizer
    if recipe.constraint_geometry_mode == "nonascent" and critic is None:
        from .constraint_geometry import ConstraintGeometryOptimizer
        optimizer_class = ConstraintGeometryOptimizer
    optimizer = optimizer_class(params, family=recipe.optimizer_family,
                                    momentum=recipe.optimizer_momentum,
                                    smoothing=recipe.optimizer_smoothing,
                                    convolution=recipe.optimizer_convolution, **options)
    if critic is not None:
        optimizer.critic = critic
        optimizer.ema_critic = optimizer.anchor = optimizer.guard = None
        optimizer.record = CriticStepRecord()
    return optimizer
