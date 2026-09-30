"""Paired fixed-input birth profile; CUDA mode is for the root GPU owner only."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument('--package-root', type=Path, required=True)
parser.add_argument('--reference-package-root', type=Path, required=True)
parser.add_argument('--inputs', type=Path, default=Path(__file__).resolve().parent / 'inputs.pt')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
assert 1 <= args.repeats <= 5 and not args.output.exists()
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if args.device == 'cuda' else ''
import torch
from fixture_utils import convert, load_package, nested_equal, network, output_hash, sha

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
if args.device == 'cuda':
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.2, 0)


def synchronize():
    if args.device == 'cuda':
        torch.cuda.synchronize(0)


@contextmanager
def traced_math():
    original_jacobian, original_svd = torch.autograd.functional.jacobian, torch.linalg.svd
    def jacobian(*a, **k):
        with torch.profiler.record_function('anchor.jacobian'):
            return original_jacobian(*a, **k)
    def svd(*a, **k):
        with torch.profiler.record_function('anchor.svd'):
            return original_svd(*a, **k)
    torch.autograd.functional.jacobian, torch.linalg.svd = jacobian, svd
    try:
        yield
    finally:
        torch.autograd.functional.jacobian, torch.linalg.svd = original_jacobian, original_svd


def run_side(package, alias, case):
    module, birth = load_package(package, alias)
    data = convert(case, device)
    G, D, ema_G = (network(data['models'][name]) for name in ('G', 'D', 'ema_G'))
    trainer = SimpleNamespace(G=G, D=D)
    controller = SimpleNamespace(_heads=[D[4]], sample_shape=(2,))
    callbacks = [birth.learned_latent_features(controller, trainer, model) for model in (G, ema_G)]
    calls = dict(live=0, ema=0)
    def callback(index):
        name = ('live', 'ema')[index]
        def captured(latents):
            calls[name] += 1
            with torch.profiler.record_function('anchor.features.' + name):
                return callbacks[index](latents)
        return captured
    current, average = callback(0), callback(1)
    snapshot = module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
    snapshot.__dict__.update(data['snapshot'])
    initial_work = dict(snapshot.work)
    def plan():
        snapshot.work = dict(initial_work)
        value = birth.plan_real_anchor_births(snapshot, data['q'], data['flags'], data['pvalues'],
            data['comparison'], data['latents'], current, ema_latents=data['ema_latents'],
            ema_feature_of_latent=average, previous_children=data['previous_children'],
            previous_copy_parents=data['previous_copy_parents'], supported_counts=data['supported_counts'],
            max_moves=data['max_moves'])
        return value, dict(snapshot.work)
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(0).clone() if args.device == 'cuda' else None
    value, work = plan()
    synchronize()
    timings = []
    for _ in range(args.repeats):
        synchronize()
        start = time.perf_counter()
        repeated, _ = plan()
        synchronize()
        timings.append(time.perf_counter() - start)
        assert nested_equal(value, repeated)
    calls.update(live=0, ema=0)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile, traced_math():
        profiled, profiled_work = plan()
        synchronize()
    assert nested_equal(value, profiled) and profiled_work == work
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(0))
    assert all(p.grad is None for model in (G, D, ema_G) for p in model.parameters())
    operators = {}
    for event in profile.key_averages():
        if (event.key.startswith('anchor.') or event.key in ('aten::_local_scalar_dense', 'aten::item',
                'aten::nonzero', 'aten::linalg_svd', 'aten::_linalg_svd', 'aten::mm', 'aten::bmm', 'aten::std')):
            operators[event.key] = dict(count=event.count, self_cpu_ms=event.self_cpu_time_total / 1000,
                cpu_total_ms=event.cpu_time_total / 1000,
                device_total_ms=getattr(event, 'device_time_total', 0) / 1000)
    summary = dict(moves=value['moves'], attempted_cells=value['attempted_cells'],
        linearizations=[(a['current']['linearizations'], a['average']['linearizations']) for a in value['attempts']],
        callback_calls=calls, complete_output_sha256=output_hash(value), snapshot_work=work,
        operators=operators, median_wall_ms=statistics.median(timings) * 1000,
        wall_ms=[t * 1000 for t in timings], RNG_unchanged=True, parameter_gradients_untouched=True)
    return value, summary


inputs = torch.load(args.inputs, map_location='cpu', weights_only=False)
source_paths = [Path(__file__), Path(__file__).resolve().parent / 'fixture_utils.py', args.inputs,
    *sorted((args.package_root / 'particlegan').rglob('*.py')),
    *sorted((args.reference_package_root / 'particlegan').rglob('*.py'))]
before = {str(p): sha(p) for p in source_paths}
records = []
for index, case in enumerate(inputs['cases']):
    original, baseline = run_side(args.reference_package_root, f'anchor_reference_{index}', case)
    proposed, candidate = run_side(args.package_root, f'anchor_candidate_{index}', case)
    exact = nested_equal(original, proposed)
    assert exact, case['step']
    records.append(dict(step=case['step'], complete_plan_bit_exact=exact, baseline=baseline, candidate=candidate))
assert before == {str(p): sha(p) for p in source_paths}
if args.device == 'cpu':
    assert not torch.cuda.is_initialized()
result = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), device=str(device),
    source_sha256=before, records=records, inputs_sha256=sha(args.inputs),
    scope='paired fixed saved clean-table planner mechanical/profile proof; no historical actions or quality trajectory',
    timing_scope='isolated microprofile wall time; GPU shared-process timing is separate from exactness',
    optimizer_updates=0, new_seeds=0, quality_verdict=None)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status='PASS', device=result['device'], cases=len(records), output=str(args.output))))
