"""Damp Adam's generator proposal in its empirical clean-output metric.

For network Adam metric P and batch-output Jacobian J, solve
  (P^-1 + J.T J / (batch_size * output_step)) delta = -gradient.
This is a penalty on the proposed displacement, not a pull on the parameters
or critic gradients. The prior and D retain ordinary Adam. The implementation
uses Adam's rounded proposal on the right-hand side, preserving that proposal
exactly in observation-only mode. Deterministic, buffer-free MLP scope only.
"""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.func import functional_call, jacrev


def metric_correction(proposal, diagonal, jacobian, ridge):
    """Woodbury solve; tensors are flattened, ridge is batch_size*output_step."""
    if ridge <= 0 or not math.isfinite(ridge):
        raise ValueError("positive finite ridge required")
    if not bool((diagonal > 0).all()):
        raise ValueError("positive Adam metric required")
    jp = jacobian * diagonal.unsqueeze(0)
    kernel = jp @ jacobian.T
    kernel.diagonal().add_(ridge)
    response = torch.linalg.solve(kernel, jacobian @ proposal)
    return proposal - jp.T @ response


class FunctionalMetric:
    def __init__(self, output_step=.029, observe_only=False):
        if not math.isfinite(output_step) or output_step <= 0:
            raise ValueError("output_step must be finite and positive")
        self.output_step, self.observe_only = output_step, observe_only
        self.records, self.hooks, self.measuring = [], [], False
        self.receipt = dict(policy="generator_functional_metric_v1",
            shared_gate_eligible=False, output_step=output_step, observe_only=observe_only,
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope="deterministic buffer-free G network; prior and D ordinary Adam", updates=[])

    def register(self, model):
        if any(True for _ in model.buffers()):
            raise ValueError("buffer-free generator required")
        if any(isinstance(m, (nn.Dropout, nn.BatchNorm1d)) for m in model.modules()):
            raise ValueError("deterministic generator required")
        record = dict(model=model, ids={id(p) for p in model.parameters()}, inputs=None)
        self.records.append(record)
        def capture(module, inputs):
            if not self.measuring and torch.is_grad_enabled():
                if not inputs or any(not isinstance(x, torch.Tensor) or x.ndim != 2 for x in inputs):
                    raise ValueError("matrix generator inputs required")
                record['inputs'] = tuple(x.detach().clone() for x in inputs)
        self.hooks.append(model.register_forward_pre_hook(capture))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def step(self, optimizer, original_step, closure=None):
        owned = {id(p) for g in optimizer.param_groups for p in g['params']}
        matches = [r for r in self.records if r['ids'] & owned]
        if not matches:
            return original_step(optimizer, closure=closure)
        if closure is not None or len(matches) != 1:
            raise ValueError("one explicit generator per optimizer required")
        record = matches[0]; model = record['model']; inputs = record['inputs']
        if inputs is None:
            raise RuntimeError("missing training generator inputs")
        parameters = dict(model.named_parameters())
        groups = {id(p):g for g in optimizer.param_groups for p in g['params']}
        if set(groups) & record['ids'] != record['ids']:
            raise RuntimeError("generator only partially owned")
        if any(groups[id(p)]['betas'][0] != 0 or groups[id(p)].get('weight_decay',0) != 0
               for p in parameters.values()):
            raise ValueError("ordinary zero-momentum, zero-decay Adam required")
        before = {name:p.detach().clone() for name,p in parameters.items()}
        result = original_step(optimizer)
        after = {name:p.detach().clone() for name,p in parameters.items()}
        if self.observe_only:
            self.receipt['updates'].append(dict(observe_only=True,
                nominal_rates=[float(g['lr']) for g in optimizer.param_groups],
                moment_steps=[sorted({int(optimizer.state[p]['step']) for p in g['params']})
                              for g in optimizer.param_groups], jacobian_rows=0))
            return result
        proposal = torch.cat([(after[name]-before[name]).flatten().double() for name in parameters])
        diagonal=[]
        for p in parameters.values():
            group=groups[id(p)]; state=optimizer.state[p]; step=int(state['step'])
            variance=state['exp_avg_sq']/(1-group['betas'][1]**step)
            diagonal.append((group['lr']/(variance.sqrt()+group['eps'])).flatten().double())
        diagonal=torch.cat(diagonal)
        try:
            self.measuring=True
            with torch.no_grad():
                for name,p in parameters.items(): p.copy_(before[name])
            widths=[x.shape[1] for x in inputs]; batch=inputs[0].shape[0]
            unique,counts=torch.unique(torch.cat(inputs,dim=1),dim=0,return_counts=True)
            unique_inputs=tuple(unique.split(widths,dim=1))
            def forward(weights):
                return functional_call(model,weights,unique_inputs)
            rng=torch.get_rng_state().clone()
            derivative=jacrev(forward)(before)
            with torch.no_grad():
                base_outputs=model(*unique_inputs)
                jacobian=torch.cat([derivative[name].reshape(base_outputs.numel(),-1)
                                    for name in parameters],dim=1).double()
                weights=counts.sqrt().repeat_interleave(base_outputs.shape[1]).double()
                weighted_jacobian=jacobian*weights[:,None]
                delta=metric_correction(proposal,diagonal,weighted_jacobian,batch*self.output_step)
                offset=0
                for name,p in parameters.items():
                    n=p.numel(); p.copy_(before[name]+delta[offset:offset+n].reshape(p.shape).to(p.dtype)); offset+=n
                actual_outputs=model(*unique_inputs)
                movement=((actual_outputs-base_outputs).square().sum(1)*counts).sum().div(batch).sqrt()
                proposed_linear=(weighted_jacobian@proposal).square().sum().div(batch).sqrt()
                corrected_linear=(weighted_jacobian@delta).square().sum().div(batch).sqrt()
                if not torch.isfinite(movement):
                    raise FloatingPointError("nonfinite functional movement")
            if not torch.equal(rng,torch.get_rng_state()):
                raise RuntimeError("functional metric consumed training RNG")
            self.receipt['updates'].append(dict(batch=batch,unique_inputs=len(unique),
                jacobian_rows=base_outputs.numel(),jacobian_parameter_elements=proposal.numel(),
                clean_forwards=3, nominal_rates=[float(g['lr']) for g in optimizer.param_groups],
                moment_steps=[sorted({int(optimizer.state[p]['step']) for p in g['params']})
                              for g in optimizer.param_groups],
                proposed_linear_output_rms=float(proposed_linear),
                corrected_linear_output_rms=float(corrected_linear),actual_output_rms=float(movement)))
        finally:
            self.measuring=False
        return result


@contextmanager
def functional_metric(*, output_step=.029, observe_only=False, state=None):
    controller=FunctionalMetric(output_step,observe_only)
    original_step=torch.optim.Adam.step if state is None else state['base_adam_step']
    def step(optimizer,closure=None): return controller.step(optimizer,original_step,closure)
    if state is not None:
        controller.register(state['generator'].model)
        state['set_step_delegate'](step)
        try: yield controller.receipt
        finally:
            state['set_step_delegate'](original_step); controller.close()
    else:
        from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
        original_register=NoisePolicy.register_generator_base
        def register(policy,model):
            original_register(policy,model); controller.register(model)
        try:
            with patch.object(NoisePolicy,'register_generator_base',register),patch.object(torch.optim.Adam,'step',step):
                yield controller.receipt
        finally: controller.close()
