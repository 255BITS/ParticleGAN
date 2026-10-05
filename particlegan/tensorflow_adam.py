"""Dense TensorFlow-v1 Adam update law with explicit optimizer application clocks.

This is a PyTorch implementation of the inspected dense legacy formula, not a
TensorFlow dependency or a promise about an unbound historical training run.
"""
from copy import deepcopy
import math

import torch


class TensorFlowV1Adam(torch.optim.Adam):
    """Adam whose epsilon is added before second-moment bias correction.

Each optimizer has its own application clock, shared by its parameter groups.
Skipped parameter gradients do not stop that clock. Sparse/complex gradients,
weight decay, AMSGrad and accelerated/differentiable modes are unsupported.
All clocks, powers and moments travel in ``state_dict``.
"""

    def __init__(self, params, **options):
        forbidden = {"amsgrad": False, "weight_decay": 0, "maximize": False,
                     "capturable": False, "differentiable": False,
                     "foreach": None, "fused": None, "decoupled_weight_decay": False}
        if any(options.get(name, default) not in (default, False) for name, default in forbidden.items()):
            raise ValueError("tensorflow_v1 supports only ordinary dense Adam without weight decay or AMSGrad")
        super().__init__(params, **options)
        for group in self.param_groups:
            group.update(_tf_step=0, _tf_beta1_power=1.0, _tf_beta2_power=1.0,
                         _tf_initial_betas=tuple(group["betas"]))
        self._validate_groups(self.param_groups)

    @staticmethod
    def _validate_groups(groups):
        steps = set()
        for group in groups:
            step = group.get("_tf_step")
            betas = tuple(group.get("betas", ()))
            if (type(step) is not int or step < 0 or len(betas) != 2
                    or tuple(group.get("_tf_initial_betas", ())) != betas
                    or any(type(b) not in (int, float) or not math.isfinite(b) or not 0 <= b < 1 for b in betas)):
                raise ValueError("invalid tensorflow_v1 clock or changed betas")
            steps.add(step)
            if any(type(group.get(name)) not in (int, float)
                   or not math.isfinite(group[name]) or group[name] < 0 for name in ("lr", "eps")):
                raise ValueError("tensorflow_v1 requires finite nonnegative scalar rates and epsilon")
            for name, beta in zip(("_tf_beta1_power", "_tf_beta2_power"), betas):
                power = group.get(name)
                if (type(power) not in (int, float) or not math.isfinite(power)
                        or not 0 <= power <= 1 or not math.isclose(power, beta**step, rel_tol=1e-10, abs_tol=1e-15)):
                    raise ValueError("invalid tensorflow_v1 bias-correction power")
            if any(group.get(key, False) for key in ("amsgrad", "maximize", "capturable", "differentiable",
                                                      "fused", "foreach", "decoupled_weight_decay")) or group.get("weight_decay", 0):
                raise ValueError("unsupported tensorflow_v1 parameter-group option")
        if len(steps) != 1:
            raise ValueError("tensorflow_v1 parameter groups must share one application clock")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._validate_groups(self.param_groups)
        # Reject unsupported inputs before any parameter, moment or clock moves.
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.is_complex() or (parameter.grad is not None and parameter.grad.is_sparse):
                    raise ValueError("tensorflow_v1 requires dense real gradients")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            group["_tf_step"] += 1
            group["_tf_beta1_power"] *= beta1
            group["_tf_beta2_power"] *= beta2
            rate = group["lr"] * math.sqrt(1 - group["_tf_beta2_power"]) / (1 - group["_tf_beta1_power"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state.update(step=torch.tensor(0.), exp_avg=torch.zeros_like(parameter),
                                 exp_avg_sq=torch.zeros_like(parameter))
                state["step"].fill_(group["_tf_step"])
                state["exp_avg"].mul_(beta1).add_(parameter.grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(parameter.grad, parameter.grad, value=1 - beta2)
                denominator = state["exp_avg_sq"].sqrt().add_(group["eps"])
                parameter.addcdiv_(state["exp_avg"], denominator, value=-rate)
        return loss

    def state_dict(self):
        state = super().state_dict()
        state["tensorflow_v1"] = {"schema_version": 1, "gradient_law": "dense_real"}
        return state

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("tensorflow_v1") != {"schema_version": 1, "gradient_law": "dense_real"}:
            raise ValueError("missing or incompatible tensorflow_v1 optimizer checkpoint")
        groups = state.get("param_groups", [])
        if len(groups) != len(self.param_groups):
            raise ValueError("incompatible tensorflow_v1 parameter groups")
        self._validate_groups(groups)
        for actual, saved in zip(self.param_groups, groups):
            if (tuple(actual["_tf_initial_betas"]) != tuple(saved["_tf_initial_betas"])
                    or actual["eps"] != saved["eps"]):
                raise ValueError("tensorflow_v1 checkpoint role settings differ")
        # Validate moments before the standard loader can mutate the live state.
        for actual, saved in zip(self.param_groups, groups):
            if len(actual["params"]) != len(saved["params"]):
                raise ValueError("incompatible tensorflow_v1 parameters")
            for parameter, key in zip(actual["params"], saved["params"]):
                values = state.get("state", {}).get(key, {})
                if not values:
                    continue
                for name in ("exp_avg", "exp_avg_sq"):
                    value = values.get(name)
                    if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                            or value.is_complex() or not torch.isfinite(value).all()
                            or (name == "exp_avg_sq" and (value < 0).any())):
                        raise ValueError("invalid tensorflow_v1 moment tensor")
                step = values.get("step")
                if (not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all()
                        or step.item() < 1 or step.item() > saved["_tf_step"] or step.item() != int(step.item())):
                    raise ValueError("invalid tensorflow_v1 parameter clock")
        packet = deepcopy(state)
        packet.pop("tensorflow_v1")
        super().load_state_dict(packet)
