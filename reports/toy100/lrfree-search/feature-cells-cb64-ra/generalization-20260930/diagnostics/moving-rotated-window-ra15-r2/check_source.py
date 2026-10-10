"""CPU preparation checks: exact original host/scorer AST and passive tracing."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paired_preparation', HERE / 'run_pair.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
original = ast.parse((run.ORIGINAL / 'adapted_runner.py').read_text())


def dump(node):
    return ast.dump(node, include_attributes=False)


functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
original_loop = next(node for node in original.body if isinstance(node, ast.For) and node.target.id == 'step')
for variant, package in [('control', run.CONTROL), ('candidate', run.CANDIDATE)]:
    source = run.adapted(package, variant)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            assert dump(node) == dump(functions[node.name]), node.name
    loop = next(node for node in tree.body if isinstance(node, ast.For) and node.target.id == 'step')
    loop.body = [node for node in loop.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'support'
        and node.value.func.attr == 'note_update')]
    assert [dump(node) for node in loop.body] == [dump(node) for node in original_loop.body]
    assert isinstance(loop.iter.args[0], ast.Constant) and loop.iter.args[0].value == 1001
    assert dump(loop.iter.args[1]) == dump(original_loop.iter.args[1])
    # Everything after the original loop (draw, target scoring, gate and output)
    # is unchanged; only a receipt/check call is appended.
    old_tail = original.body[original.body.index(original_loop) + 1:]
    new_tail = tree.body[tree.body.index(loop) + 1:-1]
    assert [dump(node) for node in old_tail] == [dump(node) for node in new_tail]
    assert 'seed, device, batch, n_rows = 1234' in source
    assert "serial_backward=True" in source
    assert "optimizer_options={'foreach': False, 'fused': False}" in source
    compile(source, f'<{variant}-source-check>', 'exec')

# Check the return-profiler adapter without importing a package, constructing a
# model, loading a checkpoint, advancing RNG or initializing CUDA.
import torch
assert not torch.cuda.is_initialized()
def reject_cuda(*args, **kwargs):
    raise AssertionError('CUDA forbidden during source preparation')
torch.cuda._lazy_init = reject_cuda
spec = importlib.util.spec_from_file_location('window_support_check', HERE / 'window_support.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
support._trainer = SimpleNamespace(completed_steps=1000)
before = torch.get_rng_state().clone()
fixed = SimpleNamespace(weights=torch.tensor([.25, 0., .5], dtype=torch.float64))
snapshot = SimpleNamespace(mass_groups=3)
def dummy_freeze(snapshot, fixed):
    counts = torch.tensor([2, 3, 4])
    ema_counts = torch.tensor([8, 0, 9])
    return fixed, None
result = support._observe_return(dummy_freeze, 'freeze_moment')(snapshot, fixed)
assert result[0] is fixed and result[1] is None
assert sys.getprofile() is None
record = support._pending.pop()
assert record['ema_counts_missing_groups'] == [1]
assert record['initial_inactive_groups'] == [1]
assert record['retained_even_mass'] == .75 and record['step'] == 1001
assert torch.equal(before, torch.get_rng_state())
assert not torch.cuda.is_initialized()
# Exact byte semantics include intentional NaNs. Distinct finite values,
# NaN positions/payloads, signedzero bits, dtypes and shapes remain differences.
same = torch.tensor([1., float('nan'), -0.], dtype=torch.float32)
assert not support.compare(same, same.clone())
assert not support.compare(float('nan'), float('nan'))
assert support.compare(same, torch.tensor([2., float('nan'), -0.]))
assert support.compare(same, torch.tensor([float('nan'), 1., -0.]))
assert support.compare(same, torch.tensor([1., 2., -0.]))
assert support.compare(same, same.double())
assert support.compare(same, same.reshape(1, 3))
assert support.compare(torch.tensor([0.]), torch.tensor([-0.]))
assert support.compare(float('nan'), 1.)
assert support.compare(1., float('nan'))
payload = same.clone()
payload.view(torch.int32)[1] ^= 1
assert support.compare(same, payload)
scalar = torch.tensor(float('nan'))
assert not support.compare(scalar, scalar.clone())
checkpoint = torch.load(run.ORIGINAL / 'checkpoint-001000.pt', map_location='cpu', weights_only=False)
cloned = copy.deepcopy(checkpoint)
assert not support.compare(checkpoint, cloned)
nan_tensors = []
def scan(value, path=''):
    if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
        count = int(torch.isnan(value).sum())
        if count:
            twin = cloned
            for key in path.split('.'):
                twin = twin[int(key)] if isinstance(twin, (list, tuple)) else twin[int(key)] if key.isdigit() and int(key) in twin else twin[key]
            assert value.dtype == twin.dtype and value.shape == twin.shape
            assert torch.equal(torch.isnan(value), torch.isnan(twin))
            assert not support.compare(value, twin)
            nan_tensors.append(dict(path=path, shape=list(value.shape), dtype=str(value.dtype),
                nan_count=count, raw_bytes_identical=True, nan_positions_identical=True))
    elif isinstance(value, dict):
        for key, item in value.items():scan(item, f'{path}.{key}' if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):scan(item, f'{path}.{index}')
scan(checkpoint)
assert nan_tensors
assert torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized()
revision = json.loads((HERE / 'PREPARATION-REVISION.json').read_text())
for path, expected in revision['read_only_file_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
    (HERE / 'run_pair.py', HERE / 'window_support.py', HERE / 'check_source.py',
     run.ORIGINAL / 'adapted_runner.py')}
inputs.update(revision['read_only_file_sha256'])
receipt = dict(status='CPU_PASS', source_ast_host_functions_identical=list(functions),
    original_update_body_identical_except_passive_note=True,
    original_terminal_draw_scorer_gate_ast_identical=True,
    paired_window_only=([1001, 1500]), original_seed=1234,
    passive_profiler_reads_actual_locals=True, profiler_restored=True,
    cpu_rng_unchanged=True, cuda_initialized=False, gpu_operations=0,
    model_calls=0, checkpoint_loads=1, training_updates=0, input_sha256=inputs,
    strict_nan_comparator=dict(matching_tensor_raw_bytes_pass=True, matching_scalar_NaN_pass=True,
        finite_changes_rejected=True, changed_NaN_positions_rejected=True, changed_NaN_payload_rejected=True,
        finite_to_NaN_rejected=True, changed_shape_or_dtype_rejected=True, signedzero_bits_rejected=True,
        real_checkpoint1000_CPU_copy_exact=True, real_checkpoint_NaN_tensors=nan_tensors),
    limitation='The failed worker saved no restored-state artifact. CPU-copy byte and NaN audit is exact; CUDA-restored state equality remains a required runtime gate.')
run.write(HERE / 'SOURCE-ADAPTER-CHECK-V2.json', receipt)
print(json.dumps(dict(status='CPU_PASS', host_functions=list(functions),
    original_loop_body=True, original_gate=True, passive_profiler=True, cuda_initialized=False)), flush=True)
