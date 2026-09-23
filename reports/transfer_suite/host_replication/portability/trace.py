"""Trace the rare-mode winner under one numerical path, without changing its numerics.

Saves step-1 layer outputs, D gradients before its first Adam step, D after that
step, and all G/particle/D parameters after every G update (optionally only the
first TRACE_STEPS). Compare two traces with analyze.py.
  python3 -m reports.transfer_suite.host_replication.portability.trace OUT.pt
"""
import gzip
import json
import os
import sys
from unittest.mock import patch

import torch

from benchmarks.transfer_suite import vector_tasks
from benchmarks.transfer_suite.linear_skip_refinement_research import ARCHITECTURES, constructor
from .probe import REFERENCE


class Stop(BaseException):
    pass


def main():
    spec = json.loads(gzip.decompress(REFERENCE.read_bytes()))['spec']
    card = next(c for c in ARCHITECTURES if c['name'] == 'linear_skip_d96_beta5')
    limit = int(os.environ.get('TRACE_STEPS', '0')) or None
    models, first, trace, counts = {}, {}, [], dict(g=0, d=0)
    make_d, make_g, make_p = constructor(card), vector_tasks.SimpleMLPGenerator, vector_tasks.ParticlePrior

    def record(key):
        def hook(module, inputs, output):
            if counts['d'] == 0 and key not in first:
                first[key] = output.detach().clone()
        return hook

    def hooked(name, create):
        def build(*args, **kwargs):
            model = models.setdefault(name, create(*args, **kwargs))
            if name != 'p':
                for label, module in model.named_modules():
                    if label:
                        module.register_forward_hook(record(f'{name}.{label}'))
            return model
        return build

    flat = lambda name: torch.cat([p.detach().reshape(-1) for p in models[name].parameters()])
    adam_step = torch.optim.Adam.step

    def step(self, *args, **kwargs):
        role = 'g' if len(self.param_groups) == 2 else 'd'
        if role == 'd' and counts['d'] == 0:
            first['d_grad_before_first_update'] = torch.cat([p.grad.reshape(-1) for p in self.param_groups[0]['params']]).clone()
        out = adam_step(self, *args, **kwargs)
        counts[role] += 1
        if role == 'd' and counts['d'] == 1:
            first['d_after_first_update'] = flat('d').clone()
        if role == 'g':
            trace.append(torch.cat([flat('g'), flat('p'), flat('d')]).clone())
            if limit and len(trace) == limit:
                raise Stop
        return out
    with patch.object(vector_tasks, 'SimpleMLPDiscriminator', hooked('d', make_d)), \
            patch.object(vector_tasks, 'SimpleMLPGenerator', hooked('g', make_g)), \
            patch.object(vector_tasks, 'ParticlePrior', hooked('p', make_p)), patch.object(torch.optim.Adam, 'step', step):
        try:
            result = vector_tasks.run_episode(spec, vector_tasks.fixed_policy('cosine'), fixed=True)
        except Stop:
            result = dict(observations=[])
    torch.save(dict(trace=torch.stack(trace), first=first, capability=torch.backends.cpu.get_cpu_capability(),
                    sizes={k: sum(p.numel() for p in models[k].parameters()) for k in ('g', 'p', 'd')},
                    observations=result['observations']), sys.argv[1])
    print('saved', len(trace), 'G updates', flush=True)


if __name__ == '__main__':
    main()
