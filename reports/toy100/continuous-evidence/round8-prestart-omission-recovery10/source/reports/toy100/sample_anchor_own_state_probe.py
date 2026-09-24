"""Qualified own-state hold and same-target response for the sample-anchor guard.

This is a thin source-bound caller of the reviewed exact mode_hold resumer.
No training may begin until the separate cold trajectory and ring both pass.
The response starts from the candidate's completed own-acquired hold state.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
REST_FACTORY = 'reports.toy100.sample_anchor_rest_candidate:sample_anchor_rest_candidate'
PRESTART_FACTORY = 'reports.toy100.sample_anchor_prestart_candidate:sample_anchor_prestart_candidate'
NOISE_HORIZON = 1200
HOLD_END = 2400
RESPONSE_END = 2450
SOURCE_NAMES = (
    'reports/toy100/sample_anchor_own_state_probe.py',
    'reports/toy100/pr84_model_error_recovery.py',
    'reports/toy100/pr84_critic_refinement_resume.py',
    'reports/toy100/pr84_critic_refinement_capture.py',
    'benchmarks/transfer_suite/legacy_noise_adapters.py',
    'benchmarks/locked_shared/mlp.py',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_factory(root, entrypoint):
    sys.path.insert(0, str(root))
    module_name, separator, function_name = entrypoint.partition(':')
    if (not separator or not module_name.startswith('reports.toy100.')
            or not function_name.isidentifier()):
        raise ValueError('factory must be an explicit reports.toy100.module:context_name')
    module = importlib.import_module(module_name)
    filename = str(Path(module.__file__).resolve().relative_to(root))
    if (not filename.startswith('reports/toy100/') or not filename.endswith('.py')
            or not isinstance(module.METHOD, str) or not module.METHOD):
        raise RuntimeError('loaded factory lacks a repository source or method')
    function = getattr(module, function_name)
    if not callable(function):
        raise RuntimeError('declared context factory is not callable')
    return function, module.METHOD, filename


def require_cold(cold, root, factory, entrypoint, method, factory_file):
    """Read and verify the complete cold gate before creating any output."""
    from reports.toy100 import pr84_model_error_recovery as recovery
    from reports.toy100.pr84_critic_refinement_capture import _sha
    declaration = json.loads((cold / 'declaration.json').read_text())
    summary = json.loads((cold / 'summary.json').read_text())
    if (declaration.get('phase') != 'cold' or summary.get('phase') != 'cold'
            or declaration.get('factory') != entrypoint or summary.get('factory') != entrypoint
            or declaration.get('method') != method or summary.get('method') != method):
        raise RuntimeError('cold gate is not for the explicit factory and method')
    if factory_file not in declaration['source'] or sha((root / factory_file).read_bytes()) != declaration['source'][factory_file]:
        raise RuntimeError('cold declaration does not bind the current factory bytes')
    with patch.object(recovery, 'METHOD', method), patch.object(recovery, 'ROOT', root):
        saved, _, _, ring = recovery.require_qualified_cold(cold)
    if (ring.get('saved_state_stage') != 'after final host evaluation, live weights restored; before next set_step'
            or ring['final_snapshot_sha256'] != _sha(saved)
            or ring['spec']['steps'] != NOISE_HORIZON
            or saved['snapshot_scope']['version'] != 1
            or saved['noise_policy']['total_steps'] != NOISE_HORIZON):
        raise RuntimeError('cold snapshot is not the exact complete update1200 boundary')
    config = json.loads((cold / 'config.json').read_text())
    if (config.get('name') != method or config.get('lr_floor') != 1.
            or config.get('lr_anneal_start') != 0.
            or 'network_lr_horizon_cap' in config or 'network_lr_floor' in config):
        raise RuntimeError('cold recipe did not retain constant rates')
    with factory(task='mode_hold', correction=True) as (_, generated):
        if sha(generated.encode()) != ring['dynamics']['host_source']['generated_function_sha256']:
            raise RuntimeError('cold ring and current factory generate different mode_hold host source')
    return saved, declaration, ring, config


def bind_sources(output, cold_source, root):
    names = set(cold_source) | set(SOURCE_NAMES)
    bound = {}
    for name in sorted(names):
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError(f'source file left the repository: {name}')
        raw = path.read_bytes()
        if name in cold_source and sha(raw) != cold_source[name]:
            raise RuntimeError(f'cold source changed: {name}')
        bound[name] = sha(raw)
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return bound


def verify_loaded_sources(root, bound):
    """Catch an imported sibling-worktree module whose bytes differ from the gate."""
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename or not filename.endswith('.py'):
            continue
        path = Path(filename).resolve()
        for checkout in (root, ROOT):
            try:
                relative = str(path.relative_to(checkout))
            except ValueError:
                continue
            if relative in bound and sha(path.read_bytes()) != bound[relative]:
                raise RuntimeError(f'loaded source differs from bound bytes: {relative}')
            break


@contextmanager
def bound_factory(recovery, factory):
    def selected():
        return factory(task='mode_hold', start_step=0, correction=True)
    with patch.object(recovery, 'pr84_critic_refinement_finite', selected):
        yield


def run_bound(saved, recipe, noise, factory, *, completed, target, perturb=False,
              fail_fast=False, log=None):
    from reports.toy100 import pr84_model_error_recovery as recovery
    with bound_factory(recovery, factory):
        result = recovery.run_continuation(saved, recipe, noise,
            completed_steps=completed, target_steps=target, perturb=perturb,
            fail_fast=fail_fast, log=log)
    from reports.toy100.pr84_critic_refinement_resume import resumed_source
    with factory(task='mode_hold', start_step=0, correction=True) as (_, generated):
        _, continued = resumed_source(generated, completed, target)
    if sha(continued.encode()) != result['receipt']['source_sha256']:
        raise RuntimeError('actual resumed host source differs from exact AST source')
    return result, continued


def require_hold(hold, cold, declaration):
    """Accept only a completed own-acquired dense hold from this exact cold state."""
    from reports.toy100.pr84_critic_refinement_capture import _sha
    value = json.loads((hold / 'hold.json').read_text())
    if (value.get('status') != 'PASS' or value.get('method') != declaration['method']
            or value.get('factory') != declaration['factory']
            or value.get('cold_ring_state_file_sha256') != declaration['cold_ring_state_file_sha256']
            or value.get('source') != declaration['source']
            or value.get('receipt', {}).get('pass_all') is not True
            or value['receipt'].get('completed') is not True
            or value['receipt'].get('completed_before_resume') != NOISE_HORIZON
            or value['receipt'].get('target_steps') != HOLD_END
            or value['receipt'].get('updates') != HOLD_END - NOISE_HORIZON
            or value['receipt'].get('first_failure_step') is not None
            or value['receipt'].get('noise_horizon') != NOISE_HORIZON
            or value['receipt'].get('actual_adam_updates') != {'d': 1200, 'g': 1200}
            or value['receipt'].get('optimizer_callbacks') != {'d': 3600, 'g': 3600}
            or value['generated_resume_source_sha256'] != value['receipt'].get('source_sha256')
            or [r['step'] for r in value['receipt']['checkpoints']] != list(range(NOISE_HORIZON + 1, HOLD_END + 1))
            or any(row['modes'] != 8 or row['hq'] < .9 for row in value['receipt']['checkpoints'])):
        raise RuntimeError('complete own-acquired dense1200 hold did not pass')
    if sha((cold / declaration['cold_ring_state_file']).read_bytes()) != declaration['cold_ring_state_file_sha256']:
        raise RuntimeError('cold snapshot changed after hold')
    path = hold / value['final_state_file']
    if sha(path.read_bytes()) != value['final_state_file_sha256']:
        raise RuntimeError('hold final state bytes changed')
    saved = torch.load(path, weights_only=True, map_location='cpu')
    if (_sha(saved) != value['final_snapshot_sha256']
            or saved['noise']['step_calls'] != HOLD_END
            or saved['snapshot_scope']['host_loop_step'] not in (HOLD_END - 1, HOLD_END)
            or saved['noise_policy']['total_steps'] != NOISE_HORIZON):
        raise RuntimeError('hold state is not the completed update2400 snapshot')
    return saved, value


def slim(branch):
    from reports.toy100.pr84_critic_refinement_capture import _sha
    return dict(result=branch['result'], receipt=branch['receipt'],
                dynamics=branch['dynamics'], applied=branch['applied'],
                policy=branch['policy'], final_state_sha256=_sha(branch['state']),
                host_source_sha256=branch['host_source_sha256'])


def run_hold(output, saved, recipe, noise, factory, declaration):
    from reports.toy100.pr84_critic_refinement_capture import _sha
    events = []
    def observe(row):
        events.append(row)
        if row['step'] % 20 == 0 or not row['passed']:
            print(json.dumps(dict(event='OWN_HOLD_PROGRESS', step=row['step'],
                                  modes=row['modes'], hq=row['hq'], passed=row['passed'])), flush=True)
    branch, generated = run_bound(saved, recipe, noise, factory, completed=NOISE_HORIZON,
        target=HOLD_END, fail_fast=True, log=observe)
    receipt = branch['receipt']
    if len(events) != receipt['updates'] or any(row['step'] != NOISE_HORIZON + i + 1
                                                for i, row in enumerate(events)):
        raise RuntimeError('hold log omitted a completed update')
    if receipt['actual_adam_updates'] != {'d': receipt['updates'], 'g': receipt['updates']}:
        raise RuntimeError('hold Adam accounting differs')
    (output / 'generated-resumed-mode_hold.py').write_text(generated)
    final_file = output / 'hold-final-state.pt'
    torch.save(branch['state'], final_file)
    status = 'PASS' if receipt['pass_all'] and receipt['completed'] and receipt['updates'] == 1200 else 'FAIL_FIRST_CHECK'
    value = dict(status=status, method=declaration['method'], factory=declaration['factory'],
                 source=declaration['source'],
                 cold_ring_state_file_sha256=declaration['cold_ring_state_file_sha256'],
                 receipt=receipt, dynamics=branch['dynamics'], applied=branch['applied'],
                 policy=branch['policy'], result=branch['result'],
                 final_state_file=final_file.name, final_state_file_sha256=sha(final_file.read_bytes()),
                 final_snapshot_sha256=_sha(branch['state']),
                 generated_resume_source_sha256=sha(generated.encode()),
                 shared_gate_eligible=False,
                 scope='own-acquired fixed-target dense update1201..2400; stop at first failure')
    write_json(output / 'hold.json', value)
    print(json.dumps(dict(event='OWN_HOLD_DONE', status=status,
                          completed=receipt['updates'], first_failure=receipt['first_failure_step'])), flush=True)
    return value


def run_response(output, saved, recipe, noise, factory, declaration):
    from reports.toy100 import pr84_model_error_recovery as recovery
    from reports.toy100.pr84_critic_refinement_capture import _sha
    control, generated = run_bound(saved, recipe, noise, factory,
                                   completed=HOLD_END, target=RESPONSE_END)
    perturbed, second = run_bound(saved, recipe, noise, factory,
                                  completed=HOLD_END, target=RESPONSE_END, perturb=True)
    if generated != second:
        raise RuntimeError('paired arms used different resumed host source')
    if (control['receipt']['restored_snapshot_sha256'] != _sha(saved)
            or perturbed['receipt']['restored_snapshot_sha256'] != _sha(saved)
            or any(arm['receipt']['updates'] != 50 or not arm['receipt']['completed']
                   for arm in (control, perturbed))):
        raise RuntimeError('paired arms did not restore the same completed update2400 state')
    frozen = recovery.frozen_perturbed_points(perturbed['perturbed_start'], noise,
        completed_steps=HOLD_END, target_steps=RESPONSE_END)
    for arm in (control, perturbed):
        if arm['state']['noise_policy']['_effective_step_trace'][HOLD_END:] != frozen['effective_step_trace']:
            raise RuntimeError('paired arms did not share the original absolute noise horizon')
    if (_sha(control['state']['rng']) != _sha(perturbed['state']['rng'])
            or control['state']['noise_policy']['_counts'] != perturbed['state']['noise_policy']['_counts']):
        raise RuntimeError('trained response arms used different RNG or noise counts')
    verdict = recovery.grade_response(control, perturbed, frozen)
    (output / 'generated-resumed-mode_hold.py').write_text(generated)
    for name, arm in (('control', control), ('perturbed', perturbed)):
        torch.save(arm['state'], output / f'{name}-final-state.pt')
    def rms(tensor):
        return float(tensor.square().sum(-1).mean().sqrt())
    clean_initial = perturbed['model_error']['clean_before']
    clean_shifted = perturbed['model_error']['clean_perturbed']
    functional = dict(initial_bias_shift_rms=rms(clean_shifted-clean_initial),
        perturbed_training_move_rms=rms(perturbed['final_clean_support']-clean_shifted),
        final_indexed_distance_to_original_rms=rms(perturbed['final_clean_support']-clean_initial),
        control_training_move_rms=rms(control['final_clean_support']-clean_initial),
        interpretation='indexed movement is diagnostic; particle permutations can preserve the data law')
    injection = {key: value.tolist() if isinstance(value, torch.Tensor) else value
                 for key, value in perturbed['model_error'].items()}
    value = dict(method=declaration['method'], factory=declaration['factory'],
        source=declaration['source'],
        cold_ring_state_file_sha256=declaration['cold_ring_state_file_sha256'],
        own_hold_state_file_sha256=declaration['own_hold_state_file_sha256'],
        verdict=verdict, control=slim(control), perturbed=slim(perturbed), frozen=frozen,
        perturbation=injection, functional_displacement=functional,
        control_state_file_sha256=sha((output/'control-final-state.pt').read_bytes()),
        perturbed_state_file_sha256=sha((output/'perturbed-final-state.pt').read_bytes()),
        generated_resume_source_sha256=sha(generated.encode()),
        shared_gate_eligible=False, same_target=True,
        scope='own-acquired update2400 state; paired 2401..2450 fixed +0.35 G output-bias response')
    write_json(output / 'response.json', value)
    print(json.dumps(dict(event='OWN_RESPONSE_DONE', **verdict)), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('hold', 'response'), required=True)
    parser.add_argument('--factory', required=True)
    parser.add_argument('--cold', type=Path, required=True)
    parser.add_argument('--hold', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.root.resolve()
    factory, method, filename = load_factory(root, args.factory)
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    saved, cold_declaration, ring, config = require_cold(
        args.cold, root, factory, args.factory, method, filename)
    verify_loaded_sources(root, cold_declaration['source'])
    cold_hash = ring['final_state_file_sha256']
    if args.phase == 'response':
        if args.hold is None:
            raise ValueError('response requires the passed own-acquired hold directory')
        hold_source = json.loads((args.hold/'declaration.json').read_text())
        if (hold_source['method'] != method or hold_source['factory'] != args.factory
                or hold_source['cold_ring_state_file_sha256'] != cold_hash
                or hold_source['cold_declaration_sha256'] != sha((args.cold/'declaration.json').read_bytes())):
            raise RuntimeError('response hold declaration differs from the qualified cold ring')
        saved, hold = require_hold(args.hold, args.cold, hold_source)
        for name, digest in hold_source['source'].items():
            if sha((root/name).read_bytes()) != digest or sha((args.hold/'source'/name).read_bytes()) != digest:
                raise RuntimeError(f'hold source changed: {name}')
    recipe, noise, _ = declared_recipe(config)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sources = bind_sources(args.output, cold_declaration['source'], root)
    verify_loaded_sources(root, sources)
    declaration = dict(method=method, factory=args.factory, phase=args.phase, source=sources,
                       cold_ring_state_file_sha256=cold_hash,
                       cold_ring_state_file=ring['final_state_file'],
                       cold_declaration_sha256=sha((args.cold/'declaration.json').read_bytes()),
                       own_hold_state_file_sha256=(None if args.phase == 'hold' else
                           hold['final_state_file_sha256']),
                       noise_horizon=NOISE_HORIZON,
                       nominal_rates={'d': [.00425], 'g_prior': [.00425, .0085]},
                       response_bias=[.35, 0.], response_updates=50,
                       same_target=True, shared_gate_eligible=False)
    write_json(args.output/'declaration.json', declaration)
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    try:
        if args.phase == 'hold':
            run_hold(args.output, saved, recipe, noise, factory, declaration)
        else:
            run_response(args.output, saved, recipe, noise, factory, declaration)
    except BaseException as error:
        write_json(args.output/'error.json', dict(status='ERROR_INCOMPLETE', error=repr(error),
                                                 shared_gate_eligible=False))
        raise


if __name__ == '__main__':
    main()
