"""Finite-displacement generator trust bound; scoped research adapter.

Retains ordinary Adam moments, all nominal rates and every prior update.
Interpolates only G's proposed parameter step until its clean output motion
on that step's existing input batch fits a fixed data-space RMS radius.
No horizon, elapsed step, evaluation draw, or target center enters the rule.
"""
from contextlib import contextmanager
import math
from unittest.mock import patch
import torch
from torch import nn


@contextmanager
def output_trust(radius: float):
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be positive and finite')
    original_register = NoisePolicy.register_generator_base
    original_step = torch.optim.Adam.step
    models, hooks, receipt = [], [], []
    measuring = False

    def capture(model, inputs):
        if not measuring:
            for record in models:
                if record['model'] is model:
                    record['inputs'] = tuple(x.detach().clone() if isinstance(x,torch.Tensor) else x for x in inputs)

    def register(self, model):
        original_register(self, model)
        if not isinstance(model,nn.Module):
            raise ValueError('output trust requires an explicit generator module')
        models.append(dict(model=model, ids={id(p) for p in model.parameters()}, inputs=None))
        hooks.append(model.register_forward_pre_hook(capture))

    def step(self, *args, **kwargs):
        nonlocal measuring
        ids={id(p) for group in self.param_groups for p in group['params']}
        owned=[r for r in models if r['ids'] & ids]
        if not owned:
            return original_step(self,*args,**kwargs)
        if len(owned)!=1:
            raise ValueError('ambiguous generator ownership')
        record=owned[0]
        model=record['model']
        inputs=record['inputs']
        if inputs is None:
            raise RuntimeError('G had no training forward')
        parameters=[p for p in model.parameters() if id(p) in ids]
        before=[p.detach().clone() for p in parameters]
        try:
            measuring=True
            with torch.no_grad():
                output_before=model(*inputs)
            answer=original_step(self,*args,**kwargs)
            with torch.no_grad():
                delta=[p.detach()-old for p,old in zip(parameters,before)]
                output_proposal=model(*inputs)
                displacement=float((output_proposal-output_before).flatten(1).square().sum(1).mean().sqrt())
                if not math.isfinite(displacement):
                    raise FloatingPointError('Nonfinite clean output proposal; this run cannot be resumed')
                scale=min(1.,radius/max(displacement,1e-30))
                final_displacement=displacement
                extra=0
                if scale < 1.:
                    for backtrack in range(17):
                        for p,old,change in zip(parameters,before,delta):
                            p.copy_(old+scale*change)
                        actual=model(*inputs)
                        extra+=1
                        final_displacement=float((actual-output_before).flatten(1).square().sum(1).mean().sqrt())
                        if not math.isfinite(final_displacement):
                            raise FloatingPointError('Nonfinite clean output trial; this run cannot be resumed')
                        if final_displacement<=radius*(1+1e-6):
                            break
                        scale*=.5
                    else:
                        raise RuntimeError('Nonlinear trust bound failed after 17 trials')
            receipt.append(dict(step=len(receipt)+1,radius=radius,scale=scale,
                proposed_output_rms=displacement,actual_output_rms=final_displacement,
                extra_clean_forwards=2+extra, nominal_lrs=[g['lr'] for g in self.param_groups]))
            return answer
        finally:
            measuring=False

    try:
        with patch.object(NoisePolicy,'register_generator_base',register),patch.object(torch.optim.Adam,'step',step):
            yield receipt
    finally:
        for hook in hooks:
            hook.remove()
