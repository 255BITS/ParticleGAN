"""Explicit constant instance-noise policy for frozen legacy-host research.

The policy changes only sigma(t): sigma(t) = input_noise_std. It does not
reinterpret input_noise_anneal_end as persistence. The original noise adapter
still draws from its independent critic-noise stream and preserves evaluation
streams. Output noise, alternating Adam, and all host code remain unchanged.
The Adam wrapper observes ordinary updates without changing their calculation.
This scratch policy cannot enter the production common gate.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import math
from unittest.mock import patch

import torch

from benchmarks.toy100.models import linear_input_noise as _linear_input_noise
from benchmarks.transfer_suite import legacy_noise_adapters as legacy

PAPER = "https://arxiv.org/html/1801.04406v4"
POLICY = "constant_instance_noise_v1"


def constant_input_noise(std, completed_steps, total_steps, anneal_end):
    """Validate the ordinary fields, then explicitly use constant sigma."""
    _linear_input_noise(std, completed_steps, total_steps, anneal_end)
    return float(std)


class NoiseRecorder:
    def __init__(self):
        self.policy = None
        self.clock = []
        self.inputs = []
        self.updates = []
        self.optimizers = []
        self.since_update = 0

    def set_step(self, policy, completed_steps, original):
        if self.policy is None:
            self.policy = policy
        if self.policy is not policy or completed_steps != len(self.clock):
            raise ValueError("constant-noise clock must be one complete ordered host")
        if len(self.updates) != 2 * completed_steps:
            raise ValueError("host must make one alternating D/G update per clock")
        value = original(policy, completed_steps)
        self.clock.append(dict(step=completed_steps, input_sigma=policy.input_sigma,
                               output_sigma=policy.output_sigma))
        return value

    def input(self, policy, data, original):
        sigma = float(policy.input_sigma)
        evaluated = bool(policy._evaluating)
        result = original(policy, data)
        if policy.input_sigma != sigma:
            raise ValueError("noise amplitude changed inside an application")
        with torch.no_grad():
            delta = result.detach() - data.detach()
            rms = math.sqrt(float(delta.double().square().mean()))
        self.inputs.append(dict(step=len(self.clock) - 1, evaluating=evaluated,
                                sigma=sigma, elements=data.numel(), applied_rms=rms))
        if not evaluated:
            if not self.clock or policy is not self.policy:
                raise ValueError("training input noise applied without a clock")
            self.since_update += 1
        return result

    def step(self, optimizer, original, closure=None):
        if closure is not None or not self.clock:
            raise ValueError("observer requires ordinary explicit-gradient Adam")
        role = "d" if len(self.updates) % 2 == 0 else "g"
        position = 0 if role == "d" else 1
        if len(self.optimizers) <= position:
            if optimizer in self.optimizers:
                raise ValueError("D and G optimizers must be distinct")
            self.optimizers.append(optimizer)
        if self.optimizers[position] is not optimizer or self.since_update <= 0:
            raise ValueError("alternating update order or noisy critic inputs differ")
        groups, saved = [], {}
        with torch.no_grad():
            for group in optimizer.param_groups:
                if (group.get("weight_decay", 0) or any(group.get(key, False) for key in
                        ("amsgrad", "maximize", "differentiable", "capturable", "fused"))):
                    raise ValueError("observer requires the declared ordinary Adam")
                parameters = group["params"]
                if any(p.grad is None for p in parameters):
                    raise ValueError("an Adam parameter has no explicit gradient")
                count = sum(p.numel() for p in parameters)
                square = sum(float(p.grad.detach().double().square().sum()) for p in parameters)
                saved.update({p: p.detach().clone() for p in parameters})
                groups.append(dict(
                    role=("d" if role == "d" else "prior" if group.get("_comparison_prior") else "g"),
                    parameters=count, lr=float(group["lr"]), betas=list(group["betas"]),
                    gradient_rms=math.sqrt(square / count),
                ))
        # Exactly the original PyTorch Adam call, with the original gradients.
        result = original(optimizer)
        with torch.no_grad():
            for group, observed in zip(optimizer.param_groups, groups):
                square = sum(float((p.detach() - saved[p]).double().square().sum())
                             for p in group["params"])
                observed["update_rms"] = math.sqrt(square / observed["parameters"])
                observed["moment_steps"] = [int(optimizer.state[p]["step"])
                                              for p in group["params"]]
        self.updates.append(dict(step=len(self.clock) - 1, role=role,
                                 input_sigma=float(self.policy.input_sigma),
                                 input_calls=self.since_update, groups=groups))
        self.since_update = 0
        return result

    def receipt(self):
        return dict(policy=POLICY, paper=PAPER, common_gate_eligible=False,
                    optimizer="ordinary_alternating_torch_Adam",
                    extra_gradient_evaluations_per_player_per_outer_step=0,
                    schedule="sigma(t) = input_noise_std; input_noise_anneal_end is superseded",
                    clock=self.clock, inputs=self.inputs, updates=self.updates)


@contextmanager
def persistent_noise():
    recorder = NoiseRecorder()
    original_clock, original_input = legacy.NoisePolicy.set_step, legacy.NoisePolicy.input
    original_adam = torch.optim.Adam.step
    with ExitStack() as stack:
        stack.enter_context(patch.object(legacy, "linear_input_noise", constant_input_noise))
        stack.enter_context(patch.object(legacy.NoisePolicy, "set_step",
                                        lambda policy, step: recorder.set_step(policy, step, original_clock)))
        stack.enter_context(patch.object(legacy.NoisePolicy, "input",
                                        lambda policy, data: recorder.input(policy, data, original_input)))
        stack.enter_context(patch.object(torch.optim.Adam, "step",
                                        lambda optimizer, closure=None: recorder.step(optimizer, original_adam, closure)))
        yield recorder


@contextmanager
def persistent_noise_regrade():
    """Change only the expected sigma rule in the frozen scratch verifier.

    Raw constant-noise receipts are retained as generated. Every remaining
    source, target, optimizer, budget, live-checkpoint and metric check runs.
    The production verifier always rejects these explicitly marked episodes.
    """
    from benchmarks import toy_suite
    with patch.object(toy_suite, "linear_input_noise", constant_input_noise):
        yield
