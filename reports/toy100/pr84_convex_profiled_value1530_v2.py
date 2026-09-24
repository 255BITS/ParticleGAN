"""Bound-certificate continuation of the unchanged V1 readout diagnostic.

Neither fit budget nor optimality tolerance changes. The only mathematical
gate change permits a value trial when the base has a valid finite upper
bound, and accepts only when the trial's global lower bound exceeds it.
Inner NOT_CERTIFIED statuses remain visible even after a verified decrease.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100 import pr84_convex_profiled_value1530 as v1


def transformed_main():
    source = inspect.getsource(v1.main)
    tree = ast.parse(source)
    changes = dict(base=0, trial=0, method=0, status=0, declaration=0)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
            first = ast.unparse(node.test.values[0])
            if first == "base_fit['status'] == 'CERTIFIED_GAP'":
                node.test.values[0] = ast.parse("base_fit['certificate']['available']", mode='eval').body
                changes['base'] += 1
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'verified':
                assert isinstance(node.value, ast.BoolOp)
                assert ast.unparse(node.value.values[0]) == "trial_fit['status'] == 'CERTIFIED_GAP'"
                node.value.values = node.value.values[1:]
                changes['trial'] += 1
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'declaration':
                assert isinstance(node.value, ast.Call)
                node.value.keywords.extend([
                    ast.keyword(arg='generated_runner_sha256', value=ast.Name(id='_generated_sha', ctx=ast.Load())),
                    ast.keyword(arg='v1_result_sha256', value=ast.Name(id='_base_result_sha', ctx=ast.Load())),
                    ast.keyword(arg='continuation_scope', value=ast.Constant(
                        value='same fit budget/tolerance and exact repeated base; accept only disjoint objective bounds; retain nonconverged inner statuses')),
                ])
                changes['declaration'] += 1
        if isinstance(node, ast.Constant):
            if node.value == 'saved1530_convex_readout_profiled_sharp_D_value':
                node.value = 'saved1530_convex_readout_profiled_value_bound_continuation'
                changes['method'] += 1
            if node.value == 'VERIFIED_PROFILED_VALUE_DECREASE':
                node.value = 'VERIFIED_PROFILED_VALUE_DECREASE_WITH_DECLARED_INNER_GAPS'
                changes['status'] += 1
    if changes != dict(base=1, trial=1, method=1, status=1, declaration=1):
        raise RuntimeError(f'unexpected frozen V1 source: {changes}')
    ast.fix_missing_locations(tree)
    return tree, ast.unparse(tree)+'\n'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--base-result', type=Path, required=True)
    own, remaining = parser.parse_known_args()
    base_bytes = own.base_result.read_bytes()
    base = json.loads(base_bytes)
    if base['status'] != 'UNRESOLVED_BASE_OPTIMALITY':
        raise RuntimeError('expected the retained V1 nonconverged result')
    for name, digest in base['declaration']['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('frozen V1 source changed')
    tree, source = transformed_main()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    calls = 0
    original_fit = v1.fit_readout
    def checked_fit(problem, initial):
        nonlocal calls
        result = original_fit(problem, initial)
        calls += 1
        if calls == 1 and result[1] != base['base_fit']:
            raise RuntimeError('V2 base solve differs from the frozen V1 receipt')
        return result
    namespace = dict(vars(v1))
    namespace.update(_generated_sha=source_hash,
        _base_result_sha=hashlib.sha256(base_bytes).hexdigest(), fit_readout=checked_fit,
        SOURCES=v1.SOURCES[:-1]+('reports/toy100/pr84_convex_profiled_value1530_v2.py',)+v1.SOURCES[-1:])
    exec(compile(tree, '<convex-value-bound-continuation>', 'exec'), namespace)
    original_arguments = sys.argv
    try:
        sys.argv = [sys.argv[0]]+remaining
        namespace['main']()
    finally:
        sys.argv = original_arguments
    if calls < 1:
        raise RuntimeError('continuation did not reproduce its base solve')
    output = Path(remaining[remaining.index('--output')+1])
    (output/'source/generated_value_continuation.py').write_text(source)
    result = json.loads((output/'result.json').read_text())
    result['base_fit_exact_to_v1'] = True
    result['readout_fit_attempts'] = calls
    (output/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
