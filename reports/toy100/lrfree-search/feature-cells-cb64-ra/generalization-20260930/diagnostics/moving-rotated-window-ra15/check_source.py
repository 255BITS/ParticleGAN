"""CPU preparation checks: exact original host/scorer AST and passive tracing."""
import ast
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
inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
    (HERE / 'run_pair.py', HERE / 'window_support.py', HERE / 'check_source.py',
     run.ORIGINAL / 'adapted_runner.py')}
receipt = dict(status='CPU_PASS', source_ast_host_functions_identical=list(functions),
    original_update_body_identical_except_passive_note=True,
    original_terminal_draw_scorer_gate_ast_identical=True,
    paired_window_only=([1001, 1500]), original_seed=1234,
    passive_profiler_reads_actual_locals=True, profiler_restored=True,
    cpu_rng_unchanged=True, cuda_initialized=False, gpu_operations=0,
    model_calls=0, checkpoint_loads=0, training_updates=0, input_sha256=inputs,
    limitation='Source/observation validation only; exact CUDA restoration and control reproduction remain runtime gates.')
run.write(HERE / 'SOURCE-ADAPTER-CHECK-V2.json', receipt)
print(json.dumps(dict(status='CPU_PASS', host_functions=list(functions),
    original_loop_body=True, original_gate=True, passive_profiler=True, cuda_initialized=False)), flush=True)
