"""Joint G-network/prior version of the fixed functional displacement metric.

The earlier network-only correction leaves the learned input cloud's Adam
proposal uncorrected. Here J differentiates clean outputs with respect to both
roles and the same Woodbury solve corrects their combined proposal. D then G
ordering, losses, nominal rates and once-per-update Adam moments are retained.
No target centers, quality scores or elapsed-time feedback are read.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch
from torch.func import functional_call, jacrev

from reports.toy100.functional_metric_scratch import FunctionalMetric, metric_correction


class JointFunctionalMetric(FunctionalMetric):
    def __init__(self, output_step=.029, observe_only=False):
        super().__init__(output_step, observe_only)
        self.receipt.update(policy="joint_generator_prior_functional_metric_v1",
            scope="deterministic buffer-free G network plus learned prior; D ordinary Adam",
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def step(self, optimizer, original_step, closure=None):
        owned = {id(p) for g in optimizer.param_groups for p in g['params']}
        matches = [r for r in self.records if r['ids'] & owned]
        if not matches:
            return original_step(optimizer, closure=closure)
        if closure is not None or len(matches) != 1:
            raise ValueError("one explicit generator per optimizer required")
        record = matches[0]
        model, inputs = record['model'], record['inputs']
        if inputs is None:
            raise RuntimeError("missing training generator inputs")
        network = dict(model.named_parameters())
        groups = {id(p): g for g in optimizer.param_groups for p in g['params']}
        prior = [p for g in optimizer.param_groups if g.get('_comparison_prior') for p in g['params']]
        if len(prior) != 1 or prior[0].ndim != 2:
            raise ValueError("one explicitly identified prior matrix required")
        if any(g['betas'][0] != 0 or g.get('weight_decay', 0) != 0 for g in optimizer.param_groups):
            raise ValueError("zero first momentum and weight decay required")
        params = dict(network, __prior__=prior[0])
        if {id(p) for p in params.values()} != owned:
            raise ValueError("optimizer must own exactly the G network and prior")
        before = {name: p.detach().clone() for name, p in params.items()}
        # Exact input matching recovers particle identity without consulting data.
        # Refuse duplicate prior rows: their distinct derivatives are ambiguous.
        widths = [x.shape[1] for x in inputs]
        unique, counts = torch.unique(torch.cat(inputs, dim=1), dim=0, return_counts=True)
        unique_inputs = tuple(unique.split(widths, dim=1))
        match = (unique_inputs[-1][:, None, :] == before['__prior__'][None, :, :]).all(-1)
        if not bool((match.sum(1) == 1).all()):
            raise ValueError("training latent rows must exactly identify distinct prior particles")
        indices = match.long().argmax(1)
        result = original_step(optimizer)
        after = {name: p.detach().clone() for name, p in params.items()}
        if self.observe_only:
            self.receipt['updates'].append(dict(observe_only=True))
            return result
        proposal = torch.cat([(after[n]-before[n]).flatten().double() for n in params])
        diagonal = []
        for p in params.values():
            group, state = groups[id(p)], optimizer.state[p]
            variance = state['exp_avg_sq'] / (1-group['betas'][1]**int(state['step']))
            diagonal.append((group['lr']/(variance.sqrt()+group['eps'])).flatten().double())
        diagonal = torch.cat(diagonal)
        batch = inputs[0].shape[0]
        try:
            self.measuring = True
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(before[n])
            def forward(weights):
                args = (*unique_inputs[:-1], weights['__prior__'][indices])
                return functional_call(model, {n: weights[n] for n in network}, args)
            rng = torch.get_rng_state().clone()
            derivative = jacrev(forward)(before)
            with torch.no_grad():
                base_outputs = forward(before)
                jacobian = torch.cat([derivative[n].reshape(base_outputs.numel(), -1) for n in params], 1).double()
                weights = counts.sqrt().repeat_interleave(base_outputs.shape[1]).double()
                weighted_jacobian = jacobian * weights[:, None]
                delta = metric_correction(proposal, diagonal, weighted_jacobian, batch*self.output_step)
                offset = 0
                for n, p in params.items():
                    size = p.numel()
                    p.copy_(before[n] + delta[offset:offset+size].reshape(p.shape).to(p.dtype))
                    offset += size
                actual = forward(params)
                movement = ((actual-base_outputs).square().sum(1)*counts).sum().div(batch).sqrt()
                linear = (weighted_jacobian@delta).square().sum().div(batch).sqrt()
                if not torch.isfinite(movement):
                    raise FloatingPointError("nonfinite joint functional movement")
            if not torch.equal(rng, torch.get_rng_state()):
                raise RuntimeError("joint metric consumed RNG")
            self.receipt['updates'].append(dict(batch=batch, unique_inputs=len(unique),
                jacobian_rows=base_outputs.numel(), jacobian_parameter_elements=proposal.numel(),
                nominal_rates=[float(g['lr']) for g in optimizer.param_groups],
                moment_steps=[sorted({int(optimizer.state[p]['step']) for p in g['params']})
                              for g in optimizer.param_groups],
                proposed_linear_output_rms=float((weighted_jacobian@proposal).square().sum().div(batch).sqrt()),
                corrected_linear_output_rms=float(linear), actual_output_rms=float(movement)))
        finally:
            self.measuring = False
        return result


@contextmanager
def joint_functional_metric(*, output_step=.029, observe_only=False, state=None):
    from reports.toy100 import functional_metric_scratch as base
    with patch.object(base, 'FunctionalMetric', JointFunctionalMetric):
        with base.functional_metric(output_step=output_step, observe_only=observe_only, state=state) as receipt:
            yield receipt
