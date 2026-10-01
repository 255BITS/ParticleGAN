"""Default-CPU RA15/RA16 forward law and the original allocation failure."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
OLD = STUDY / 'pkg-RA15-partial-recovery'
NEW = STUDY / 'pkg-RA16-portability'
FIXTURE = STUDY / 'integration-prep/ra16-portability/tests/fixtures/feature-auto-base.json'
BASE = json.loads(FIXTURE.read_text())
SEED = 1234


def package(path, name):
    spec = importlib.util.spec_from_file_location(name, path / 'particlegan/__init__.py',
                                                 submodule_search_locations=[str(path / 'particlegan')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.device == b.device and a.dtype == b.dtype
        assert torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):equal(x, y)
    elif isinstance(a, float) and math.isnan(a):assert math.isnan(b)
    else:assert a == b, (a, b)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def make(api, width=2):
    torch.manual_seed(SEED)
    G = torch.nn.Sequential(torch.nn.Linear(2, 8, device='cpu'), torch.nn.Tanh(),
                            torch.nn.Linear(8, width, device='cpu'))
    D = torch.nn.Sequential(torch.nn.Linear(width, 8, device='cpu'), torch.nn.Tanh(),
                            torch.nn.Linear(8, 1, device='cpu'))
    prior = api.ParticlePrior(1024, 2, device='cpu',
                             generator=torch.Generator(device='cpu').manual_seed(SEED + 1))
    recipe = api.Recipe(**dict(BASE, z_dim=2, num_particles=1024, batch_size=128))
    return api.GANTrainer(recipe, G, D, prior=prior, seed=SEED, serial_backward=True)


assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
torch.set_default_device('cpu')
torch.use_deterministic_algorithms(True)
old, new = package(OLD, 'ra15_reference'), package(NEW, 'ra16_candidate')
# The timing field is a diagnostic elapsed clock, not learner state.
for alias in ('ra15_reference', 'ra16_candidate'):
    __import__(alias + '.feature_cells')
    sys.modules[alias + '.feature_cells'].time = SimpleNamespace(perf_counter=lambda: 0.)

allocation_calls = []
changed = []
for original in sorted((OLD / 'particlegan').rglob('*.py')):
    candidate = NEW / 'particlegan' / original.name
    if original.read_bytes() == candidate.read_bytes():continue
    changed.append(original.name)
    a, b = ast.parse(original.read_text()), ast.parse(candidate.read_text())
    if original.name == 'feature_policy.py':
        # Remove the sole new pre-mutation validator from an AST copy.
        restore = next(n for n in ast.walk(b) if isinstance(n, ast.FunctionDef) and n.name == 'prepare_restore')
        index = next(i for i, n in enumerate(restore.body)
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'birth_state' for t in n.targets))
        assert isinstance(restore.body[index + 1], ast.If)
        assert isinstance(restore.body[index + 2], ast.Assign)
        assert restore.body[index + 2].targets[0].id == 'controls'
        del restore.body[index:index + 2]
    else:
        a_calls = [n for n in ast.walk(a) if isinstance(n, ast.Call)]
        b_calls = [n for n in ast.walk(b) if isinstance(n, ast.Call)]
        assert len(a_calls) == len(b_calls)
        for ac, bc in zip(a_calls, b_calls):
            if ast.dump(ac, include_attributes=False) == ast.dump(bc, include_attributes=False):continue
            added = [k for k in bc.keywords if k.arg == 'device' and not any(k0.arg == 'device' for k0 in ac.keywords)]
            # Nested polygamma calls differ too; strip only the direct factory.
            if not added:continue
            assert len(added) == 1 and isinstance(added[0].value, ast.Constant) and added[0].value.value == 'cpu'
            assert isinstance(bc.func, ast.Attribute) and isinstance(bc.func.value, ast.Name)
            assert bc.func.value.id == 'torch' and bc.func.attr in ('zeros', 'empty', 'tensor')
            allocation_calls.append(dict(module=original.name, line=bc.lineno, factory=bc.func.attr))
            bc.keywords.remove(added[0])
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False), original.name
assert len(allocation_calls) == 7
assert changed == ['feature_policy.py', 'feature_reference.py', 'mean_transport.py',
                   'output_moments.py', 'population_continuity.py']

sample = torch.arange(600 * 10, dtype=torch.float64, device='cpu').reshape(600, 10).sin()
old_moments = __import__('ra15_reference.output_moments', fromlist=['fit_projection'])
new_moments = __import__('ra16_candidate.output_moments', fromlist=['fit_projection'])
failure = None
with torch.device('meta'):
    try:old_moments.fit_projection(sample)
    except RuntimeError as error:failure = dict(type=type(error).__name__, message=str(error))
assert failure is not None
projection_old, reason_old = old_moments.fit_projection(sample)
with torch.device('meta'):projection_new, reason_new = new_moments.fit_projection(sample)
assert reason_old is reason_new is None
equal(projection_old.__dict__, projection_new.__dict__)

shape_source = make(old)
shape_batch = torch.arange(256, dtype=torch.float32, device='cpu').reshape(128, 2).sin()
shape_source.step(shape_batch)
invalid_shape = shape_source.state_dict()
invalid_shape['birth_death']['sample_shape'] = (1, 2)
shape_target = make(old)
shape_target.load_state_dict(invalid_shape)
shape_failure = None
try:
    shape_target.birth_death._capture_generated(shape_target.policy._feature_selection.facade,
                                                shape_target.prior.z.detach())
except ValueError as error:shape_failure = dict(type=type(error).__name__, message=str(error))
assert shape_failure is not None
shape_candidate = make(new)
shape_before = shape_candidate.state_dict()
shape_rejection = None
try:shape_candidate.load_state_dict(invalid_shape)
except ValueError as error:shape_rejection = dict(type=type(error).__name__, message=str(error))
assert shape_rejection is not None
equal(shape_before, shape_candidate.state_dict())

traces = []
for width, steps in ((2, 18), (9, 2)):
    batch = torch.arange(128 * width, dtype=torch.float32, device='cpu').reshape(128, width).sin()
    reference = make(old, width)
    states, losses = [], []
    for step in range(steps):
        losses.append(reference.step(batch + step * .001))
        states.append(reference.state_dict())
    reference_sample = reference.sample(64, output_noise=True,
        generator=torch.Generator(device='cpu').manual_seed(SEED + 100))
    candidate = make(new, width)
    for step in range(steps):
        equal(losses[step], candidate.step(batch + step * .001))
        equal(states[step], candidate.state_dict())
    candidate_sample = candidate.sample(64, output_noise=True,
        generator=torch.Generator(device='cpu').manual_seed(SEED + 100))
    equal(reference_sample, candidate_sample)
    saved = reference.state_dict()
    restored = make(new, width)
    restored.load_state_dict(saved)
    equal(saved, restored.state_dict())
    equal(candidate.step(batch + steps * .001), restored.step(batch + steps * .001))
    equal(candidate.state_dict(), restored.state_dict())
    # Restore also crosses back to the original schema and validation law.
    back = make(old, width)
    back.load_state_dict(candidate.state_dict())
    equal(candidate.state_dict(), back.state_dict())
    traces.append(dict(width=width, updates=steps,
        actual_backend=saved['backend_selection']['actual_backend'],
        feature_reactions=getattr(candidate.birth_death, 'snapshot_serial', None),
        mean_forward_rows=candidate.birth_death.counters.get('mean_forward_rows'),
        exact_losses_every_step=True, exact_all_checkpoint_state_every_step=True,
        exact_samples=True, valid_checkpoint_roundtrip_both_directions=True,
        resumed_next_update_exact=True))
assert traces[0]['feature_reactions'] >= 2 and traces[0]['mean_forward_rows'] > 0
assert not torch.cuda.is_initialized()
result = dict(status='CPU_PASS', seed=SEED, noseedexperiment=True,
              source_AST_only_7_CPU_factory_keywords_plus_pre_mutation_shape_validator=True,
              CPU_allocation_sites=allocation_calls, changed_modules=changed,
              original_meta_failure=failure, meta_projection_exact_normal_CPU=True,
              original_shape_failure=dict(restore_accepted=True, output_shape=[2],
                  fifo_sample_shape=[1, 2], generated_observation_failure=shape_failure),
              candidate_shape_rejection=shape_rejection,
              candidate_shape_rejection_state_atomic=True,
              default_CPU_forward_law=traces, CUDA_visible=True, CUDA_initialized=False,
              fixture_sha256=sha(FIXTURE))
(ROOT / 'default-cpu-bridge.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True), flush=True)
