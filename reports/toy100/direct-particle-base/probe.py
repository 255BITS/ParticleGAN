"""Compare the original recipe across backends; CPU RNG is diagnostic only."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--task', required=True)
p.add_argument('--backend', choices=['cpu', 'cuda'], required=True)
p.add_argument('--cpu-random', action='store_true')
p.add_argument('--init-only', action='store_true', help='capture initialization before the first optimizer update')
p.add_argument('--initial-state', type=Path, help='initialize GPU parameters from an audited CPU fixture')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(a.repo.resolve()))
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

device = 'cuda:0' if a.backend == 'cuda' else 'cpu'
if a.backend == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA required')
torch.set_default_device(device)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class RandomAudit(TorchDispatchMode):
    """Move only random draws to CPU, leaving model math and updates on CUDA."""
    def __init__(self, cpu_random):
        super().__init__()
        self.cpu_random = cpu_random
        self.calls = 0
        self.elements = 0
        self.prefix = []
        self.hash = hashlib.sha256()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if torch.Tag.nondeterministic_seeded not in func.tags:
            return func(*args, **kwargs)
        name = func._schema.name.split('::')[-1]
        if name not in {'rand', 'randn', 'randint', 'rand_like', 'randn_like',
                        'randint_like', 'randperm', 'uniform_', 'normal_',
                        'random_', 'multinomial', 'normal', 'bernoulli', 'bernoulli_'}:
            raise RuntimeError('Unreviewed random operation: ' + str(func))
        if self.cpu_random:
            generator = kwargs.get('generator')
            if generator is not None and generator.device.type != 'cpu':
                raise RuntimeError('CPU RNG diagnostic requires CPU generators')
            target = next((v.device for v in args if isinstance(v, torch.Tensor)),
                          torch.device(kwargs.get('device') or device))
            cpu_args = tree_map(lambda v: v.detach().cpu() if isinstance(v, torch.Tensor) else v, args)
            cpu_kwargs = dict(kwargs)
            if 'device' in cpu_kwargs:
                cpu_kwargs['device'] = torch.device('cpu')
            result = func(*cpu_args, **cpu_kwargs)
            raw = result
            if name.endswith('_'):
                args[0].copy_(result)
                result = args[0]
            else:
                result = result.to(target)
        else:
            result = func(*args, **kwargs)
            raw = result
        self.calls += 1
        self.elements += raw.numel()
        # Full stream digest distinguishes input randomness from model arithmetic.
        h = digest(raw)
        self.hash.update((str(func) + str(tuple(raw.shape)) + h).encode())
        if len(self.prefix) < 64:
            self.prefix.append(dict(op=str(func), shape=list(raw.shape), sha256=h))
        return result


mobility = []
mobility_previous = {}
proof = {'adam_calls': 0, 'initial_optimizers': [], 'optimizers': {}}
initial_values = []
initial_fixture = (torch.load(a.initial_state, map_location='cpu', weights_only=True)
                   if a.initial_state else None)
original_init = torch.optim.Adam.__init__
original_step = torch.optim.Adam.step


def init(opt, *args, **kwargs):
    kwargs.setdefault('foreach', False)
    kwargs.setdefault('fused', False)
    original_init(opt, *args, **kwargs)
    values = [v for g in opt.param_groups for v in g['params']]
    if initial_fixture is not None:
        expected = initial_fixture[len(initial_values)]
        assert len(expected) == len(values)
        with torch.no_grad():
            for parameter, value in zip(values, expected):
                assert parameter.shape == value.shape and parameter.dtype == value.dtype
                parameter.copy_(value)
    initial_values.append([v.detach().cpu().clone() for v in values])
    proof['initial_optimizers'].append([
        dict(shape=list(v.shape), sha256=digest(v)) for v in values])


class InitializationCaptured(Exception):
    pass


def step(opt, *args, **kwargs):
    if a.init_only:
        raise InitializationCaptured()
    values = [v for g in opt.param_groups for v in g['params']]
    assert all(v.device.type == a.backend for v in values)
    assert all(v.grad is None or v.grad.device == v.device for v in values)
    item = proof['optimizers'].setdefault(str(id(opt)), dict(calls=0, device=device))
    item['calls'] += 1
    proof['adam_calls'] += 1
    for group in opt.param_groups:
        for v in group['params']:
            if v.grad is not None and not opt.state[v]:
                opt.state[v]['step'] = torch.zeros((), dtype=torch.float32, device=v.device)
                opt.state[v]['exp_avg'] = torch.zeros_like(v, memory_format=torch.preserve_format)
                opt.state[v]['exp_avg_sq'] = torch.zeros_like(v, memory_format=torch.preserve_format)
                if group.get('amsgrad', False):
                    opt.state[v]['max_exp_avg_sq'] = torch.zeros_like(v, memory_format=torch.preserve_format)
    before = [v.detach().clone() for v in values]
    grads = [None if v.grad is None else v.grad.detach().clone() for v in values]
    previous = mobility_previous.get(id(opt))
    response_saved = response.begin(opt)
    result = original_step(opt, *args, **kwargs)
    if item['calls'] <= 80 or item['calls'] % 50 == 0:
        displacement = torch.cat([(v.detach()-b).flatten() for v,b in zip(values,before)])
        gradient = torch.cat([g.flatten() for g in grads if g is not None])
        dot = None
        if previous is not None:
            old = torch.cat([g.flatten() for g in previous if g is not None])
            dot = float(torch.nn.functional.cosine_similarity(gradient,old,dim=0))
        row = dict(optimizer=list(proof['optimizers']).index(str(id(opt))), step=item['calls'],
                   gradient_rms=float(gradient.square().mean().sqrt()),
                   gradient_mean=float(gradient.mean()), gradient_cosine_previous=dot,
                   displacement_rms=float(displacement.square().mean().sqrt()),
                   displacement_mean=float(displacement.mean()),
                   parameter_mean_abs=float(torch.cat([v.detach().flatten() for v in values]).abs().mean()),
                   group_lrs=[g['lr'] for g in opt.param_groups],
                   group_betas=[list(g['betas']) for g in opt.param_groups])
        mobility.append(row)
        if item['calls'] == 1 or item['calls'] % 20 == 0:
            print(json.dumps(dict(mobility=row)),flush=True)
    mobility_previous[id(opt)] = grads
    response.end(response_saved)
    if item['calls'] == 1:
        for state in opt.state.values():
            for key in ('step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state:
                    assert state[key].device.type == a.backend
    item['state_devices'] = sorted({str(t.device) for state in opt.state.values() for t in state.values() if isinstance(t, torch.Tensor)})
    item['parameter_dtypes'] = sorted({str(v.dtype) for v in values})
    return result


import mechanism
import response


config = json.loads(a.config.read_text())
config['device'] = device
audit = RandomAudit(a.cpu_random)
started = time.perf_counter()
record = dict(task=a.task, backend=a.backend, cpu_random=a.cpu_random,
              init_only=a.init_only,
              initialization_fixture_sha256=(hashlib.sha256(a.initial_state.read_bytes()).hexdigest()
                                             if a.initial_state else None),
              torch=torch.__version__, config=config,
              cpu=os.popen("lscpu | sed -n 's/^Model name: *//p'").read().strip(),
              environment={k: os.environ.get(k) for k in (
                  'ATEN_CPU_CAPABILITY', 'MKL_ENABLE_INSTRUCTIONS', 'LD_PRELOAD',
                  'CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG')},
              worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
try:
    from benchmarks.transfer_suite.toy100_compatibility import (
        declared_recipe, declared_model_policy, run_vector, run_image)
    from benchmarks.transfer_suite.public_default_verification import load_declaration, declared_spec
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    recipe, noise, _ = declared_recipe(config)
    jobs, profile = load_declaration()
    job = next(j for j in jobs if j['spec']['name'] == a.task)
    spec, card, _ = declared_spec(job, profile, recipe)
    with audit, patch.object(torch.optim.Adam, '__init__', init), patch.object(torch.optim.Adam, 'step', step):
        if spec['runner'] == 'vector':
            result, _ = run_vector(spec, card, recipe, noise, model_policy=declared_model_policy(config))
        elif spec['runner'] == 'image':
            result, _ = run_image(spec, recipe, noise, model_policy=declared_model_policy(config))
        else:
            result, _ = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    if result.get('error'):
        raise RuntimeError(result['error'])
    verdict = test_verdict(spec, result)
    record.update(status=verdict['status'], spec=spec, result=result, verdict=verdict)
except InitializationCaptured:
    record.update(status='INITIALIZATION_CAPTURED')
except Exception:
    record.update(status='ERROR', error=traceback.format_exc())
record.update(response_receipt=response.receipt, mobility=mobility, regularizer_receipt=mechanism.receipt, seconds=time.perf_counter()-started, proof=proof,
              randomness=dict(calls=audit.calls, elements=audit.elements,
                              sha256=audit.hash.hexdigest(), prefix=audit.prefix))
torch.save(initial_values, a.output / 'initial-values.pt')
(a.output / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: record[k] for k in ('task', 'backend', 'cpu_random', 'status', 'seconds')}
                 | {'live': record.get('result', {}).get('live'), 'error': record.get('error')}), flush=True)
raise SystemExit(2 if record['status'] == 'ERROR' else 0)
