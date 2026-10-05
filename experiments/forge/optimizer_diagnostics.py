"""Opt-in, RNG-free measurements of actual optimizer steps, outside grading.

The BCAP study enables these observers for every arm, including Adam. Only
declared observation steps are retained. No gradients, losses or updates change.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from .contracts import file_hash


class OptimizerDiagnostics:
    def __init__(self, optimizers, critic, output, checkpoints, *, prior=None):
        self.optimizers, self.critic, self.prior = optimizers, critic, prior
        self.checkpoints = set(checkpoints)
        self.path = Path(output) / "optimizer-diagnostics.jsonl"
        self.pending, self.inputs, self.handles = {}, [], []
        self.counts = {role: max((int(state.get("step", 0)) for state in optimizer.state.values()), default=0)
            for role, optimizer in optimizers.items()}
        self.measuring = False
        self.rows = 0
        for role, optimizer in optimizers.items():
            self.handles.append(optimizer.register_step_pre_hook(self._before(role)))
            self.handles.append(optimizer.register_step_post_hook(self._after(role)))
        self.handles.append(critic.register_forward_pre_hook(self._capture_inputs))

    def _step(self, optimizer):
        return optimizer.record.observed_steps + 1

    def _capture_inputs(self, module, args):
        if self.measuring or len(self.inputs) >= 2:
            return
        if self._step(self.optimizers["D"]) in self.checkpoints and args and isinstance(args[0], torch.Tensor):
            self.inputs.append(args[0][:64].detach().clone())

    def _before(self, role):
        def before(optimizer, args, kwargs):
            step = self.counts[role] + 1
            if step not in self.checkpoints:
                return
            self.pending[role] = (step, [(group, parameter, parameter.detach().clone(),
                getattr(optimizer, "sampled_rows_for", lambda p: None)(parameter))
                for group in optimizer.param_groups for parameter in group["params"]])
        return before

    def _after(self, role):
        def after(optimizer, args, kwargs):
            self.counts[role] += 1
            pending = self.pending.pop(role, None)
            if pending is None:
                if role == "D":
                    self.inputs.clear()
                return
            step, parameters = pending
            row = {"step": step, "optimizer": role, "layers": [], "players": {}}
            for index, (group, parameter, previous, sampled) in enumerate(parameters):
                delta = parameter.detach() - previous
                weight_norm, update_norm = float(previous.norm()), float(delta.norm())
                component = group.get("role", "generator" if role == "G" else "critic")
                player = "prior" if component in ("prior", "table") else "D" if component == "critic" else "G"
                if self.prior is not None and parameter is self.prior.z:
                    player = "prior"
                    gradients = parameter.grad
                    # Native Adam has no sampled-row metadata: gradient support
                    # is reported separately and never called sampled support.
                    support = (gradients.detach().norm(dim=1) > 0) if gradients is not None else torch.zeros(len(parameter), dtype=torch.bool, device=parameter.device)
                    sampled_mask = None
                    if sampled is not None:
                        sampled_mask = torch.zeros_like(support)
                        sampled_mask[sampled] = True
                    norms = delta.norm(dim=1)
                    row["prior"] = {"row_support_kind": "sampled_indices" if sampled_mask is not None else "nonzero_gradient_support",
                        "support_rows": int((sampled_mask if sampled_mask is not None else support).sum()),
                        "mean_support_row_displacement": float(norms[sampled_mask if sampled_mask is not None else support].mean()) if bool((sampled_mask if sampled_mask is not None else support).any()) else 0.,
                        "max_outside_support_row_displacement": float(norms[~(sampled_mask if sampled_mask is not None else support)].max()) if bool((~(sampled_mask if sampled_mask is not None else support)).any()) else 0.}
                relative = update_norm / max(weight_norm, 1e-12)
                row["layers"].append({"index": index, "shape": list(parameter.shape), "player": player, "component": component,
                    "weight_norm": weight_norm, "update_norm": update_norm, "relative_update": relative,
                    "gradient_norm": float(parameter.grad.norm()) if parameter.grad is not None else 0.})
                aggregate = row["players"].setdefault(player, {"relative_update_sum": 0., "weight_norm_squared": 0., "update_norm_squared": 0.})
                aggregate["relative_update_sum"] += relative
                aggregate["weight_norm_squared"] += weight_norm ** 2
                aggregate["update_norm_squared"] += update_norm ** 2
            if role == "D":
                row.update(self._critic_metrics())
                self.inputs.clear()
            with self.path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            self.rows += 1
        return after

    def _critic_metrics(self):
        norms = [float(torch.linalg.matrix_norm(p.detach().float(), ord=2))
            for p in self.critic.parameters() if p.ndim == 2]
        log_product = sum(math.log(max(value, 1e-30)) for value in norms)
        result = {"critic_weight_spectral_norms": norms, "critic_log_spectral_product": log_product,
            "critic_spectral_product": math.exp(log_product) if log_product < 700 else None,
            "spectral_proxy_scope": "matrix-weight product; excludes Fourier/input maps and nonlinearities"}
        modes = [(module, module.training) for module in self.critic.modules()]
        self.measuring = True
        try:
            self.critic.eval()
            for label, inputs in zip(("real", "fake"), self.inputs):
                with torch.enable_grad():
                    inputs = inputs.clone().requires_grad_(True)
                    outputs = self.critic(inputs)
                    gradient = torch.autograd.grad(outputs.sum(), inputs)[0]
                    result["critic_input_gradient_mean_" + label] = float(gradient.flatten(1).norm(dim=1).mean())
        finally:
            for module, mode in modes:
                module.training = mode
            self.measuring = False
        return result

    def receipt(self):
        for handle in self.handles:
            handle.remove()
        return {"path": self.path.name, "sha256": file_hash(self.path), "rows": self.rows,
            "qualification_input": False, "sampling_draws_added": 0, "optimizer_updates_added": 0,
            "contract": "actual-step norms and deterministic critic probes at declared observation steps"}


def attach(optimizers, critic, output, checkpoints, *, prior=None):
    if os.environ.get("PARTICLEGAN_FORGE_OPTIMIZER_DIAGNOSTICS") != "1":
        return None
    return OptimizerDiagnostics(optimizers, critic, output, checkpoints, prior=prior)
