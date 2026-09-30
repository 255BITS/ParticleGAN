"""Finish only the two blocked CPU updates; reuse closed passing API evidence."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    inputs = json.loads(args.inputs.read_text())
    assert inputs['status'] == 'PRE_EXECUTION_CONTINUATION_ONLY_FROZEN'
    for path, digest in inputs['protected_sha256'].items():
        assert sha(path) == digest, path
    previous = json.loads(Path(inputs['previous_failure']).read_text())
    assert previous['status'] == 'FAIL'
    assert previous['error'] == 'TypeError("\'<\' not supported between instances of \'int\' and \'NoneType\'")'
    assert len(previous['completed_records']) == 2
    assert [record['case'] for record in previous['completed_records']] == ['grid', 'toy']
    assert all(record['status'] == 'PASS' for record in previous['completed_records'])
    assert sum(len(record['rejected_controls']) for record in previous['completed_records']) == 17
    args.output.mkdir(parents=True)
    try:
        # Guard all files before either module can import Torch or interpret PT.
        import importlib.util
        def load(path, name):
            spec = importlib.util.spec_from_file_location(name, path)
            result = importlib.util.module_from_spec(spec)
            sys.modules[name] = result
            spec.loader.exec_module(result)
            return result
        original = load(HERE / 'check_api.py', 'ra10_original_qualified_cases')
        original.raw_guards(json.loads(Path(inputs['previous_input_seal']).read_text()))
        skeleton = load(HERE / 'check_api_skeleton.py', 'ra10_unchanged_skeleton_continuation')
        api = skeleton.load_api_after_guard()
        assert not api.torch.cuda.is_initialized()
        package = api.load_package(Path(inputs['package_root']), 'ra10_continuation_composed')
        # Resolve only the erroneous helper guard. Preserve the frozen function
        # body and all sampling/update/comparison statements exactly.
        tree = ast.parse((HERE / 'check_api_skeleton.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'two_updates')
        guards = [n for n in function.body if isinstance(n, ast.Assert)
                  and ast.unparse(n.test) == "state['completed_steps'] < state['recipe']['total_steps']"]
        assert len(guards) == 1
        replacement = ast.parse("assert state['completed_steps'] < 2000 and (state['recipe']['total_steps'] is None or state['completed_steps'] < state['recipe']['total_steps'])").body[0]
        function.body[function.body.index(guards[0])] = replacement
        namespace = dict(construct=skeleton.construct)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(HERE / 'check_api_skeleton.py'), 'exec'), namespace)
        state = api.torch.load(inputs['toy_after'], map_location='cpu', weights_only=False)
        assert state['completed_steps'] == 1 and state['recipe']['total_steps'] is None
        before = api.fingerprint(state)
        global_before = api.torch.get_rng_state().clone()
        updates = namespace['two_updates'](api, package, state)
        assert api.fingerprint(state) == before
        api.torch.set_rng_state(global_before)
        assert not api.torch.cuda.is_initialized()
        for path, digest in inputs['protected_sha256'].items():
            assert sha(path) == digest, path
        prior_inputs = json.loads(Path(inputs['previous_input_seal']).read_text())
        receipt = dict(status='PASS', scope='Independent CPU full checkpoint/API/resume/typed-state contracts; continuation-only after retained helper guard failure',
            backend_schema=9, trainer_schema=5, package_sha256=prior_inputs['package_sha256'],
            config_sha256=prior_inputs['config_sha256'], records=previous['completed_records'],
            continuation=updates, previous_failure=str(Path(inputs['previous_failure'])),
            earlier_passing_cases_repeated=False, earlier_API_samples_repeated=False,
            original_recipe_horizon=None, declared_original_toy_schedule=2000,
            input_seal_sha256=sha(args.inputs), source_and_input_sha256=inputs['protected_sha256'],
            cuda_initialized=False, API_sample_calls=sum(r['serving']['sample_calls'] for r in previous['completed_records']),
            API_rows_per_sample=17, CPU_updates_total=2, new_quality_emissions=0, quality_verdict=None,
            limitation='CPU fresh-law mechanics, not historical CUDA replay; inherited served-load rejection restores semantic/view bytes while advancing parameter versions.')
        (args.output / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(status='PASS', receipt_sha256=sha(args.output / 'receipt.json'), CPU_updates_total=2)), flush=True)
    except Exception as error:
        (args.output / 'FAILURE.json').write_text(json.dumps(dict(status='FAIL', error=repr(error), traceback=traceback.format_exc()), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
