"""The unchanged1530 value-bound assay at declared independent1325/cold472.

Cold472's original fit failed before G. Its reference G endpoint and metric
come from the previously archived exact guarded-recovery reconstruction,
not an invented original post-G capture. No solve/line-search budget changes.
"""
import argparse
import ast
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100 import pr84_convex_profiled_value1530 as v1
from reports.toy100.pr84_convex_profiled_value1530_v2 import transformed_main

COLD_SHA = '19ceb38a92ba95ffaccd3aafbda55bbd77917765612666d923e986e5523d6965'
REFERENCE = ROOT/'reports/toy100/continuous-evidence/profiled-field472'
ALGORITHM_REFERENCE = ROOT/'reports/toy100/continuous-evidence/convex-profiled-value1530/v1/result.json.gz'


def read(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def assay_main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--case', type=int, choices=(1325, 472), required=True)
    own, remaining = parser.parse_known_args()
    reference_bytes = read(ALGORITHM_REFERENCE)
    algorithm_reference = json.loads(reference_bytes)
    for name, digest in algorithm_reference['declaration']['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('original1530 source changed')
    case_sha = v1.fit.STATE_SHA if own.case == 1325 else COLD_SHA
    input_path = Path(remaining[remaining.index('--states')+1])
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != case_sha:
        raise RuntimeError('wrong declared state input')
    raw = torch.load(input_path, weights_only=True)
    native_control = v1.native_control
    reference_hashes = {}
    if own.case == 1325:
        captured = raw[own.case]
        metric_scope = 'literal captured post-bounded-G Adam diagonal; fixed; no candidate moment updates'
        critic_scope = 'native accepted-D nonlinear features fixed;96-dimensional readout and fixed bias gauge'
        endpoint_scope = 'exact captured original full G/D/prior and both optimizer dictionaries'
    else:
        reference_result_path = REFERENCE/'result.json.gz'
        reference_tensors_path = REFERENCE/'tensors.pt.gz'
        old = json.loads(read(reference_result_path))
        if not all(old['endpoint_parity'].values()):
            raise RuntimeError('cold reference did not reproduce guarded G endpoint')
        tensors = torch.load(BytesIO(read(reference_tensors_path)), weights_only=True)
        reference_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (reference_result_path, reference_tensors_path)}
        pre, after_d = deepcopy(raw['pre_step']), deepcopy(raw['post_accepted_d'])
        after_d['critic'] = deepcopy(raw['best'])
        after_g = deepcopy(after_d)
        endpoint = tensors['accepted']
        after_g.update(generator=deepcopy(endpoint['generator']), prior=deepcopy(endpoint['prior']),
                       optimizer_g=deepcopy(endpoint['optimizer']))
        captured = dict(pre_step=pre, post_accepted_d=after_d, post_bounded_g=after_g)
        metric_scope = ('post-Adam G metric from archived exact guarded G472 reconstruction; fixed; '
                        'original failed fit never produced this G endpoint; no candidate moments')
        critic_scope = ('captured best finite D used by repaired G472; freeze its nonlinear features; '
                        'fit final96-dimensional readout with fixed bias gauge')
        endpoint_scope = ('native D first gradient exact; full guarded G/prior/Adam reconstruction '
                          'must equal separately archived accepted tensors exactly')

        def guarded_reference(pre_state, d_rows, g_rows, recipe):
            _, critic, _ = v1.fit.modules(after_d)
            critic.load_state_dict(raw['best'])
            receipt, accepted, _ = v1.fit.g_proposal(after_d, critic, g_rows[0], recipe.make_loss(), own.case)
            if v1._sha(accepted) != v1._sha(endpoint):
                raise RuntimeError('cold reference full G/prior/Adam differs from archived reconstruction')
            if not torch.equal(g_rows[0]['real'], tensors['g_batch']['real']):
                raise RuntimeError('cold next G data differs from original recovered draw')
            return receipt, dict(generator=accepted['generator'], critic=deepcopy(raw['best']),
                prior=accepted['prior'], optimizer_g=accepted['optimizer'],
                optimizer_d=deepcopy(after_d['optimizer_d']))
        native_control = SimpleNamespace(mean_adam_update=guarded_reference)

    tree, _ = transformed_main()
    changes = dict(input_hash=0, capture=0, method=0, provenance=0, endpoint=0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == 'fit' and node.attr == 'STATE_SHA':
                node.value.id = '_input_metadata'
                changes['input_hash'] += 1
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'captured':
                assert ast.unparse(node.value) == 'torch.load(args.states, weights_only=True)[STEP]'
                node.value = ast.Call(func=ast.Name(id='_get_capture', ctx=ast.Load()), args=[], keywords=[])
                changes['capture'] += 1
            if node.targets[0].id == 'declaration':
                node.value.keywords.extend([
                    ast.keyword(arg='reference_endpoint_scope', value=ast.Name(id='_endpoint_scope', ctx=ast.Load())),
                    ast.keyword(arg='reference_inputs_sha256', value=ast.Name(id='_reference_hashes', ctx=ast.Load())),
                ])
                changes['provenance'] += 1
        if isinstance(node, ast.keyword):
            if node.arg == 'v1_result_sha256':
                node.arg = 'algorithm_reference1530_result_sha256'
            if node.arg == 'native_single_bank_full_model_Adam_exact':
                node.arg = 'native_D_and_declared_reference_G_endpoint_exact'
                changes['endpoint'] += 1
            if node.arg == 'metric' and isinstance(node.value, ast.Constant):
                node.value = ast.Name(id='_metric_scope', ctx=ast.Load())
            if node.arg == 'critic' and isinstance(node.value, ast.Constant):
                node.value = ast.Name(id='_critic_scope', ctx=ast.Load())
        if isinstance(node, ast.Constant) and node.value == 'saved1530_convex_readout_profiled_value_bound_continuation':
            node.value = 'independent_saved_state_convex_readout_profiled_value_bound_assay'
            changes['method'] += 1
    if changes != dict(input_hash=2, capture=1, method=1, provenance=1, endpoint=1):
        raise RuntimeError(f'changed frozen source shape: {changes}')
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)+'\n'
    namespace = dict(vars(v1))
    namespace.update(STEP=own.case, native_control=native_control,
        _input_metadata=SimpleNamespace(STATE_SHA=case_sha), _get_capture=lambda: deepcopy(captured),
        _generated_sha=hashlib.sha256(source.encode()).hexdigest(),
        _base_result_sha=hashlib.sha256(reference_bytes).hexdigest(),
        _endpoint_scope=endpoint_scope, _metric_scope=metric_scope, _critic_scope=critic_scope,
        _reference_hashes=reference_hashes,
        SOURCES=v1.SOURCES[:-1]+('reports/toy100/pr84_convex_profiled_value1530_v2.py',
                               'reports/toy100/pr84_convex_profiled_value_assays.py')+v1.SOURCES[-1:])
    exec(compile(tree, '<independent-profiled-value-assay>', 'exec'), namespace)
    arguments = sys.argv
    try:
        sys.argv = [arguments[0]]+remaining
        namespace['main']()
    finally:
        sys.argv = arguments
    output = Path(remaining[remaining.index('--output')+1])
    (output/'source/generated_assay.py').write_text(source)


if __name__ == '__main__':
    assay_main()
