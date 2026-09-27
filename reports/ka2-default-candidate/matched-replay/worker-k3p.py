"""Replay frozen mode-hold through research hooks or public recipe factories.

Each arm runs in its own process. The frozen host owns models, initialization,
batches, noise, schedules, evaluations and the shift. The public arm replaces
only optimizer/penalty construction. A common source transform observes full
optimizer steps without replacing their implementation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import fields
import difflib
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
FROZEN = Path('/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda')
SOURCE = ROOT / 'reports/toy100/gap-fill-20260925/sources/k3p'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm', choices=('research', 'public'), required=True)
p.add_argument('--variant', choices=('k3p', 'ka2'), default='k3p')
p.add_argument('--repo', type=Path, default=FROZEN)
p.add_argument('--source', type=Path)
p.add_argument('--config', type=Path)
p.add_argument('--public-repo', type=Path)
p.add_argument('--initial-state', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--steps', type=int, default=3600)
p.add_argument('--network-floor', type=float, default=.01)
p.add_argument('--prior-floor', type=float, default=.05)
p.add_argument('--anneal-start', type=float, default=.6)
a = p.parse_args()
a.source = a.source or (SOURCE if a.variant == 'k3p' else ROOT / 'reports/ka2-default-candidate/source')
a.config = a.config or a.source / 'config.json'
a.public_repo = a.public_repo or (Path('/tmp/particlegan-public-080') if a.variant == 'k3p' else ROOT)
if a.steps < 1200:
    p.error('--steps must be at least 1200; the full comparison uses 3600')
a.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(a.repo.resolve()))
sys.path.insert(0, str(a.source.resolve()))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.utils._python_dispatch import TorchDispatchMode

if not torch.cuda.is_available():
    raise RuntimeError('This replay requires the frozen CUDA backend')
torch.set_default_device('cuda:0')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_signature(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=digest(tensor))


def stream_digest(items):
    value = hashlib.sha256()
    for name, tensor in items:
        value.update(name.encode())
        value.update(json.dumps(None if tensor is None else tensor_signature(tensor), sort_keys=True).encode())
    return value.hexdigest()


random_file = (a.output / 'randomness.jsonl').open('w', buffering=1)
step_file = (a.output / 'updates.jsonl').open('w', buffering=1)


class RandomAudit(TorchDispatchMode):
    """The frozen worker's complete CUDA seeded-operation stream digest."""
    def __init__(self):
        super().__init__()
        self.calls = self.elements = 0
        self.prefix = []
        self.hash = hashlib.sha256()
        self.seeded = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        seeded = self.seeded.get(func)
        if seeded is None:
            seeded = self.seeded[func] = torch.Tag.nondeterministic_seeded in func.tags
        result = func(*args, **(kwargs or {}))
        if seeded:
            name = func._schema.name.split('::')[-1]
            if name not in {'rand', 'randn', 'randint', 'rand_like', 'randn_like',
                            'randint_like', 'randperm', 'uniform_', 'normal_',
                            'random_', 'multinomial', 'normal', 'bernoulli', 'bernoulli_'}:
                raise RuntimeError('Unreviewed random operation: ' + str(func))
            self.calls += 1
            self.elements += result.numel()
            h = digest(result)
            self.hash.update((str(func) + str(tuple(result.shape)) + h).encode())
            row = dict(op=str(func), shape=list(result.shape), sha256=h)
            if len(self.prefix) < 64:
                self.prefix.append(row)
            random_file.write(json.dumps(dict(call=self.calls, **row)) + '\n')
        return result

    def summary(self):
        return dict(calls=self.calls, elements=self.elements, sha256=self.hash.hexdigest(), prefix=self.prefix)


config = json.loads(a.config.read_text())
config.update(device='cuda:0', network_lr_floor=a.network_floor,
              lr_floor=a.prior_floor, lr_anneal_start=a.anneal_start)
initial_fixture = (torch.load(a.initial_state, map_location='cpu', weights_only=True)
                   if a.initial_state else None)
initial_values, initial_signatures, optimizers = [], [], []
original_init = torch.optim.Adam.__init__
original_step = torch.optim.Adam.step
audit = RandomAudit()
counts = {'g': 0, 'd': 0}
public_recipe = None
mechanism = latent = response = None


def audited_init(opt, *args, **kwargs):
    kwargs.setdefault('foreach', False)
    kwargs.setdefault('fused', False)
    original_init(opt, *args, **kwargs)
    values = [v for group in opt.param_groups for v in group['params']]
    if initial_fixture is not None:
        expected = initial_fixture[len(initial_values)]
        assert len(expected) == len(values)
        with torch.no_grad():
            for parameter, value in zip(values, expected):
                assert parameter.shape == value.shape and parameter.dtype == value.dtype
                parameter.copy_(value)
    initial_values.append([v.detach().cpu().clone() for v in values])
    # Same shape/hash structure as canonical shift.py's initial_optimizers.
    initial_signatures.append([dict(shape=list(v.shape), sha256=digest(v)) for v in values])
    optimizers.append(opt)


def initialize_adam_state(opt):
    # Canonical shift.py puts scalar Adam counters on CUDA as well as moments.
    for group in opt.param_groups:
        for value in group['params']:
            if value.grad is not None and not opt.state[value]:
                opt.state[value]['step'] = torch.zeros((), dtype=torch.float32, device=value.device)
                opt.state[value]['exp_avg'] = torch.zeros_like(value, memory_format=torch.preserve_format)
                opt.state[value]['exp_avg_sq'] = torch.zeros_like(value, memory_format=torch.preserve_format)
                if group.get('amsgrad', False):
                    opt.state[value]['max_exp_avg_sq'] = torch.zeros_like(value, memory_format=torch.preserve_format)


def base_step(opt, *args, **kwargs):
    # cp's Adam-step accounting wraps this callable in BOTH arms. Public
    # optimizer subclasses reach it through their own _adam_step helper.
    initialize_adam_state(opt)
    if a.arm == 'public':
        return original_step(opt, *args, **kwargs)
    latent_saved = latent.begin(opt)
    response_saved = response.begin(opt)
    result = original_step(opt, *args, **kwargs)
    response.end(response_saved)
    latent.end(latent_saved)
    return result


def optimizer_signature(opt):
    params, grads, moments = [], [], []
    for gi, group in enumerate(opt.param_groups):
        for pi, value in enumerate(group['params']):
            key = f'{gi}:{pi}'
            params.append((key, value))
            grads.append((key, value.grad))
            state = opt.state.get(value, {})
            for name in ('step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                moments.append((key + ':' + name, state.get(name)))
    return dict(parameters=stream_digest(params), gradients=stream_digest(grads),
                adam=stream_digest(moments),
                rates=[float(group['lr']) for group in opt.param_groups],
                betas=[list(group['betas']) for group in opt.param_groups])


def controller_signature(opt, role):
    if role != 'd':
        return None
    if a.arm == 'research':
        state = mechanism._state
        anchor = state['ema']
        out = dict(lr_last=state['lr_last'], lr_max=state['lr_max'],
                   blend=mechanism.handover_weight(), calls=mechanism.receipt['calls'],
                   anchor=None if anchor is None else stream_digest((str(i), t) for i, t in enumerate(anchor)))
        out.update({key: state[key] for key in ('w', 'alpha', 'last_sur', 'sur_base', 'last_ratio', 'low_streak') if key in state})
        return out
    record = opt.record
    anchor = (None if not record.anchor_started else
              stream_digest((str(i), t) for i, t in enumerate(opt.ema_critic.parameters())))
    out = dict(lr_last=record.lr_last, lr_max=record.lr_max, calls=record.calls, anchor=anchor)
    if hasattr(public_penalty.regularizer, 'blend_weight'):
        out['blend'] = public_penalty.regularizer.blend_weight()
    out.update({key: getattr(record, key) for key in ('w', 'alpha', 'last_sur', 'sur_base', 'last_ratio', 'low_streak') if hasattr(record, key)})
    return out


def audit_step(opt, role, loss):
    counts[role] += 1
    before = optimizer_signature(opt)
    # State initialization precedes the public guard/A2 just as canonical
    # shift.py initializes state before entering its research step hooks.
    initialize_adam_state(opt)
    result = opt.step()
    row = dict(role=role, step=counts[role], loss=float(loss.detach()), loss_sha256=digest(loss), before=before,
               after=optimizer_signature(opt), controller=controller_signature(opt, role),
               randomness={k: v for k, v in audit.summary().items() if k != 'prefix'})
    step_file.write(json.dumps(row, allow_nan=False) + '\n')
    if counts[role] == 1 or (role == 'g' and counts[role] % 100 == 0):
        print(json.dumps(dict(event='update', role=role, step=counts[role],
                              parameters=row['after']['parameters'], random_calls=audit.calls)), flush=True)
    return result


def capture_host(generator, critic, prior, stream, noise_policy):
    value = dict(modules={name: {key: tensor_signature(tensor) for key, tensor in module.state_dict().items()}
                          for name, module in [('g', generator), ('d', critic), ('prior', prior)]},
                 batch=128, particles=prior.z.shape[0], training_stream=digest(stream.get_state()),
                 global_cuda=digest(torch.cuda.get_rng_state()), global_cpu=digest(torch.get_rng_state()),
                 input_stream=digest(noise_policy.input_stream.get_state()))
    (a.output / 'initial-host.json').write_text(json.dumps(value, indent=2) + '\n')
    torch.save(initial_values, a.output / 'initial-values.pt')


public_penalty = None


def public_optimizers(generator, critic, prior, host_recipe):
    global public_penalty
    assert public_recipe is not None
    # The alias-loaded public package has a different ParticlePrior class;
    # pass latent_table explicitly through the documented public factory.
    groups = [dict(params=list(generator.parameters()), _comparison_prior=False),
              dict(params=list(prior.parameters()), _comparison_prior=True)]
    opt_g = public_recipe.make_generator_optimizer(
        groups, latent_table=prior.z, lr=.002,
        betas=(host_recipe.beta1, host_recipe.beta2), foreach=False, fused=False)
    # Own the bare module. Public penalty's wrapper adapter retains the live
    # noise policy while substituting the EMA inner module for anchor forwards.
    owner = critic.model if hasattr(critic, 'policy') else critic
    opt_d = public_recipe.make_critic_optimizer(
        owner, ema_critic=deepcopy(owner), lr=.002 * host_recipe.d_lr_mult,
        betas=(host_recipe.beta1, host_recipe.beta2), foreach=False, fused=False)
    assert opt_g.latent_damping is not None
    public_penalty = public_recipe.make_critic_penalty(opt_d)
    return opt_g, opt_d


def penalty_bridge(critic, real, fake, step):
    # Public penalty determines the step from its paired optimizer state.
    assert step == public_penalty.optimizer.record.observed_steps + 1
    return public_penalty(critic, real, fake)


def load_public():
    package = a.public_repo / 'particlegan'
    spec = importlib.util.spec_from_file_location('_matched_public_particlegan', package / '__init__.py',
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    allowed = {field.name for field in fields(module.Recipe)}
    values = {key: value for key, value in config.items() if key in allowed}
    settings = dict(name=a.variant, reg_anchor_decay=.999, d_guard_ratio=5.,
                    d_guard_min_steps=200, latent_damping_max_rate=.5)
    values.update({key: value for key, value in settings.items() if key in allowed})
    return module.Recipe(**values)


def transform_host(mode_hold):
    source = inspect.getsource(mode_hold.train_mode_hold)
    transformed = source
    if a.arm == 'public':
        begin = transformed.index('    opt_g = torch.optim.Adam(')
        end = transformed.index('    if training_recipe is not None:', begin)
        transformed = (transformed[:begin]
                       + '    opt_g, opt_d = _matched_public_optimizers(generator, critic, prior, recipe)\n'
                       + '    regularizer = _matched_penalty\n' + transformed[end:])
        old = '    regularizer = (cap_factory or (training_recipe.make_gradient_penalty if training_recipe else make_b_cap))()'
        assert transformed.count(old) == 1
        transformed = transformed.replace(old, '    regularizer = None  # paired public penalty is built with opt_d below')
    for role in ('g', 'd'):
        old = f'        opt_{role}.step()'
        assert transformed.count(old) == 1
        transformed = transformed.replace(old, f'        _matched_audit_step(opt_{role}, "{role}", {role}_loss)')
    marker = '    batch = BATCH if training_recipe is None else training_recipe.batch_size'
    assert transformed.count(marker) == 1
    transformed = transformed.replace(marker, '    _matched_capture(generator, critic, prior, stream, noise_policy)\n' + marker)
    (a.output / 'host.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), transformed.splitlines(True), fromfile='frozen/train_mode_hold', tofile=a.arm + '/train_mode_hold')))
    (a.output / 'host.py').write_text(transformed)
    mode_hold.__dict__.update(_matched_audit_step=audit_step, _matched_capture=capture_host,
                              _matched_public_optimizers=public_optimizers, _matched_penalty=penalty_bridge)
    exec(compile(transformed, inspect.getsourcefile(mode_hold.train_mode_hold), 'exec'), mode_hold.__dict__)
    return dict(original=hashlib.sha256(source.encode()).hexdigest(), transformed=hashlib.sha256(transformed.encode()).hexdigest())


record = dict(arm=a.arm, variant=a.variant, config=config, steps=a.steps, noise_horizon=1200,
              shift_step=2400 if a.steps > 2400 else None, torch=torch.__version__,
              repo=str(a.repo), source=str(a.source), public_repo=str(a.public_repo),
              worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              initialization_fixture_sha256=(None if a.initial_state is None else hashlib.sha256(a.initial_state.read_bytes()).hexdigest()),
              environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS')})
started = time.perf_counter()
try:
    if a.arm == 'research':
        import mechanism
        import response
        import latent
    else:
        public_recipe = load_public()
        record['public_recipe'] = public_recipe.to_dict()
        record['public_files'] = {str(path.relative_to(a.public_repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((a.public_repo / 'particlegan').glob('*.py'))}
    from benchmarks.toy100 import continuous_probe as cp
    record['host_transform'] = transform_host(cp.mode_hold)

    def log(event):
        if event.get('event') == 'shift' or event.get('step', 0) % 100 == 0:
            print(json.dumps({key: event.get(key) for key in ('event', 'step', 'modes', 'hq')}), flush=True)

    with audit, patch.object(torch.optim.Adam, '__init__', audited_init), patch.object(torch.optim.Adam, 'step', base_step):
        out = cp.run_probe(config, mode='scheduled', steps=a.steps, noise_horizon=1200,
                           diagnostic_every=10, shift_step=record['shift_step'], shift=(1.0, 0.0), log=log)
    record.update(out)
except Exception:
    record.update(status='ERROR', error=traceback.format_exc())
finally:
    record.update(seconds=time.perf_counter()-started, randomness=audit.summary(),
                  proof=dict(initial_optimizers=initial_signatures, adam_calls=sum(counts.values()), updates=counts))
    if mechanism is not None:
        mechanism._write_receipt()
        record.update(regularizer_receipt=mechanism.receipt, latent_receipt=latent.receipt,
                      response_receipt=response.receipt)
    if public_penalty is not None:
        record['public_diagnostics'] = public_penalty.diagnostics()
    torch.save(initial_values, a.output / 'initial-values.pt')
    (a.output / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False, default=str) + '\n')
    random_file.close()
    step_file.close()
print(json.dumps({key: record.get(key) for key in ('arm', 'status', 'seconds', 'final', 'error')}), flush=True)
raise SystemExit(2 if record['status'] == 'ERROR' else 0)
