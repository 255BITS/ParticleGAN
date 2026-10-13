"""Pure saved-auditor fixtures; no trained grade, model, optimizer or RNG draw.

Run with the declared venv, PYTHONPATH=., no bytecode and one CPU thread.
All tensors and source/clock identities below are explicit synthetic fixtures.
Git subprocess responses are mocked; no qualification or source freeze occurs.
"""
from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def load(name):
    path = OUT / name
    ast.parse(path.read_text(), filename=str(path))
    compile(path.read_text(), str(path), 'exec')
    spec = importlib.util.spec_from_file_location('fixture_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Checks:
    def __init__(self):
        self.names = []

    def passed(self, name):
        self.names.append(name)

    def reject(self, name, function):
        try:
            function()
        except (ValueError, RuntimeError):
            self.passed(name)
        else:
            raise AssertionError(name)


def signed_and_source_fixtures(a, density, checks):
    state = dict(schema_version=1, consumer='output_energy_u_v1', weight=1., calls=3, active_calls=3,
                 loss_sum=-.75, minimum_loss=-.5, maximum_loss=0.)
    def binding(word=False):
        return dict(version=1, consumer='output_energy_u_v1',
            native_loop='benchmarks.toy_audit.api_images.WordFixture.step' if word else 'experiments.forge.behavior_adapters.run_behavior',
            source='experiments/forge/energy_adapters.py', source_sha256=a.file_hash(ROOT / 'experiments/forge/energy_adapters.py'),
            objective_source='particlegan/energy_distance.py', objective_source_sha256=a.file_hash(ROOT / 'particlegan/energy_distance.py'))
    helper = SimpleNamespace(role_groups=lambda saved, task: ([], saved['test_recipe']))
    for name, value in [('negative', state),
            ('zero', dict(state, calls=0, active_calls=0, loss_sum=0., minimum_loss=0., maximum_loss=0.)),
            ('mixed', dict(state, loss_sum=-.25, minimum_loss=-.5, maximum_loss=.5)),
            ('positive', dict(state, loss_sum=.75, minimum_loss=0., maximum_loss=.5))]:
        assert a.validate_energy_state(value) == value
        checks.passed('signed_state_' + name)
    bad = [('extra_field', dict(state, transport_loss_sum=1)),
           ('missing_field', {k: v for k, v in state.items() if k != 'minimum_loss'}),
           ('schema_bool', dict(state, schema_version=True)), ('wrong_consumer', dict(state, consumer='transport')),
           ('weight_bool', dict(state, weight=True)), ('wrong_weight', dict(state, weight=.5)),
           ('calls_bool', dict(state, calls=True)), ('negative_calls', dict(state, calls=-1, active_calls=-1)),
           ('count_mismatch', dict(state, active_calls=2)), ('loss_nan', dict(state, loss_sum=float('nan'))),
           ('min_inf', dict(state, minimum_loss=-float('inf'))), ('bool_sum', dict(state, loss_sum=True)),
           ('wrong_extrema', dict(state, minimum_loss=.5, maximum_loss=0.)),
           ('sum_below', dict(state, loss_sum=-2.)), ('sum_above', dict(state, loss_sum=1.)),
           ('nonzero_uncalled', dict(state, calls=0, active_calls=0))]
    for name, value in bad:
        checks.reject('reject_' + name, lambda value=value: a.validate_energy_state(value))
    checks.reject('reject_full_step_mismatch', lambda: a.validate_energy_state(state, steps=4))
    for role, weight in [('non_saturating', 0.), ('energy_u', 1.)]:
        native = dict(test_recipe=dict(energy_distance_weight=weight), trainer=dict(completed_steps=3))
        receipt = a.energy_receipt(native, dict(evidence={}), a.CLOCK, role, helper, steps=3)
        assert receipt['scalar_counter_available'] is False and receipt['weight'] == weight
        checks.passed('native_stateless_' + role)
    checks.reject('reject_fabricated_native_counter', lambda: a.energy_receipt(
        dict(native, component_energy=state), dict(evidence={}), a.CLOCK, 'energy_u', helper))
    behavior = dict(test_recipe=dict(energy_distance_weight=1.), component_energy=deepcopy(state),
                    applied=dict(component_energy=deepcopy(state), energy_binding=binding()))
    row = dict(evidence=dict(component_energy=deepcopy(state), guards=dict(
        component_energy_requested=True, component_energy_active_calls=3, hooks_exercised=True)))
    assert a.energy_receipt(behavior, row, 'two_pole', 'energy_u', helper, steps=3)['loss_sum'] == -.75
    checks.passed('behavior_signed_and_source_binding')
    word = dict(test_recipe=dict(energy_distance_weight=1.), fixture=dict(
        component_energy=deepcopy(state), api_state=dict(completed_steps=3)))
    word_row = dict(evidence={**deepcopy(row['evidence']), 'energy_binding': binding(True)})
    assert a.energy_receipt(word, word_row, 'five_word_joint_smoke', 'energy_u', helper, steps=3)['minimum_loss'] == -.5
    checks.passed('word_signed_and_native_reverse_loop_binding')
    for name in ('guard', 'state', 'source'):
        saved, evidence = deepcopy(behavior), deepcopy(row)
        if name == 'guard': evidence['evidence']['guards']['hooks_exercised'] = False
        elif name == 'state': evidence['evidence']['component_energy']['loss_sum'] = -1.
        else: saved['applied']['energy_binding']['source_sha256'] = '0' * 64
        checks.reject('reject_behavior_' + name, lambda saved=saved, evidence=evidence:
                      a.energy_receipt(saved, evidence, 'two_pole', 'energy_u', helper, steps=3))
    control = dict(test_recipe=dict(energy_distance_weight=0.), applied={})
    assert a.energy_receipt(control, dict(evidence={}), 'two_pole', 'non_saturating', helper)['weight'] == 0.
    checks.passed('weight_zero_original_path')
    checks.reject('reject_weight_zero_energy_state', lambda: a.energy_receipt(
        dict(control, component_energy=state), dict(evidence={}), 'two_pole', 'non_saturating', helper))
    checks.reject('reject_weight_zero_energy_binding', lambda: a.energy_receipt(
        control, dict(evidence=dict(energy_binding=binding())), 'two_pole', 'non_saturating', helper))
    checks.reject('reject_weight_zero_energy_guard', lambda: a.energy_receipt(
        control, dict(evidence=dict(guards=dict(component_energy_requested=False))), 'two_pole', 'non_saturating', helper))
    for module in (a, density):
        result = module.complete_recipe(dict(energy_distance_weight=1.))
        assert result['energy_distance_weight'] == 1. and result['reg_coeff'] == dict((f.name, f.default) for f in a.fields(a.Recipe))['reg_coeff']
        checks.passed('static_recipe_without_constructor_' + module.__name__)
    for name, function in [('recipe_constructor', lambda: a.Recipe()), ('nn_constructor', lambda: torch.nn.Linear(2, 2)),
            ('optimizer_constructor', lambda: torch.optim.Optimizer([torch.tensor(0.)], dict())),
            ('torch_draw', lambda: torch.rand(1)), ('numpy_draw', lambda: np.random.normal()), ('python_draw', lambda: random.random())]:
        checks.reject('reject_' + name, function)
    files, oid = {'a.py': 'a' * 64, 'path with space.py': 'b' * 64}, 'c' * 40
    def git(args, **kwargs):
        if args[1] == 'ls-tree':
            return b'100644 blob ' + oid.encode() + b'\ta.py\0' + b'100644 blob ' + oid.encode() + b'\tpath with space.py\0'
        assert args == ['git', 'hash-object', '--stdin-paths'] and kwargs['input'] == b'"a.py"\n"path with space.py"\n'
        return (oid + '\n' + oid + '\n').encode()
    hash_fixture = lambda path: files[str(path.relative_to(ROOT))]
    with patch.object(a.subprocess, 'check_output', git), patch.object(a, 'file_hash', hash_fixture):
        assert a.committed_sources(ROOT, 'd' * 40, files) == 2
        checks.passed('git_blob_binding_and_space_paths')
        checks.reject('reject_source_sha_change', lambda: a.committed_sources(ROOT, 'd' * 40, dict(files, **{'a.py': '0' * 64})))
    def absent(args, **kwargs): return b'' if args[1] == 'ls-tree' else (oid + '\n' + oid + '\n').encode()
    with patch.object(a.subprocess, 'check_output', absent), patch.object(a, 'file_hash', hash_fixture):
        checks.reject('reject_uncommitted_source', lambda: a.committed_sources(ROOT, 'd' * 40, files))
    def changed(args, **kwargs): return git(args, **kwargs) if args[1] == 'ls-tree' else ('e' * 40 + '\n' + oid + '\n').encode()
    with patch.object(a.subprocess, 'check_output', changed), patch.object(a, 'file_hash', hash_fixture):
        checks.reject('reject_changed_execution_blob', lambda: a.committed_sources(ROOT, 'd' * 40, files))


def clock_and_revision_fixtures(a, checks):
    candidate = dict(id='same-original-ns-card', recipe_overrides=dict(loss='non_saturating'),
        resolved_recipe=dict(loss='non_saturating', energy_distance_weight=0.), prior=dict(kind='mog', sigma=.1))
    old_revision, new_revision = (a.candidate_revision_for(source * 64, candidate) for source in ('b', 'a'))
    old = dict(candidates=dict(non_saturating=dict(candidate=candidate['id'], candidate_revision=old_revision)),
               declarations={f"configs/forge/ideas/{candidate['id']}.json": 'card-bytes'})
    new = dict(candidates=dict(non_saturating=dict(candidate=candidate['id'], candidate_revision=new_revision)))
    assert old_revision != new_revision
    checks.passed('candidate_revision_changes_with_scientific_source')
    with patch.object(a, 'file_hash', lambda path: 'card-bytes'):
        a.fresh_control_identity(ROOT, new, old)
        checks.passed('same_card_new_source_revision_accepted')
        checks.reject('old_source_revision_rejected', lambda: a.fresh_control_identity(ROOT,
            dict(candidates=dict(non_saturating=deepcopy(old['candidates']['non_saturating']))), old))
        changed = deepcopy(new); changed['candidates']['non_saturating']['candidate'] = 'different-card'
        checks.reject('substituted_control_card_rejected', lambda: a.fresh_control_identity(ROOT, changed, old))
    with patch.object(a, 'file_hash', lambda path: 'changed-bytes'):
        checks.reject('changed_control_card_bytes_rejected', lambda: a.fresh_control_identity(ROOT, new, old))
    semantics = [('data', 'clock_probe', 'training'), ('data', 'target', 'training'), ('eval', 'clock_probe', 'samples'),
        ('eval', 'sampler', 'samples'), ('eval', 'target', 'reference'), ('init', 'discriminator', 'construction'),
        ('init', 'generator', 'construction'), ('init', 'prior', 'locations'), ('noise', 'critic', 'input'),
        ('noise', 'generator', 'output'), ('noise', 'models', 'stochastic_layers'), ('noise', 'penalty', 'training'),
        ('noise', 'prior', 'gaussian'), ('prior', 'latent', 'indices')]
    trainer_keys = ['latent_generator', 'penalty_generator', 'eval_generator', 'noise_generator',
                    'input_noise_generator', 'prior_noise_generator', 'model_generator']
    branches = ['reference', 'step_label', 'horizon', 'evaluation_cadence', 'restart']
    def proof(warm, device):
        keys = [json.dumps([*part, device]) for part in semantics]
        initial = dict(initializer='deterministic_orthogonal', initialization={part: dict(initial_state_sha256='f' * 64)
            for part in ('generator', 'discriminator', 'prior')}, streams=dict(
                states={key: torch.tensor([i, 0], dtype=torch.uint8) for i, key in enumerate(keys)},
                manifest=dict(seed=0, bindings={key: dict(seed=i) for i, key in enumerate(keys)})),
            trainer=dict(models=dict(G=dict(weight=torch.tensor([float(warm)])), D=dict(weight=torch.tensor([-float(warm)])),
                P=dict(z=torch.tensor([.2]))), optimizers=[dict(step=2, moment=torch.tensor([float(warm)]))],
                streams={key: torch.tensor([0, 1], dtype=torch.uint8) for key in trainer_keys}))
        return dict(initial=initial, branch_initial={branch: deepcopy(initial) for branch in branches},
                    trajectories={branch: [deepcopy(initial) for _ in range(4)] for branch in branches})
    control, treatment = proof(1, 'cuda:0'), proof(2, 'cuda:1')
    c, e = a.clock_receipt(control), a.clock_receipt(treatment)
    assert c['saved_states_checked'] == e['saved_states_checked'] == 26 and c['own_warmed_tensors_sha256'] != e['own_warmed_tensors_sha256']
    a.matched_clock_receipts(dict(non_saturating=c, energy_u=e))
    checks.passed('different_own_warm_states_and_device_labels_accepted')
    for name, field in [('construction', 'constructor_initialization_sha256'), ('external_stream', 'initial_streams_sha256'),
                         ('trainer_stream', 'initial_trainer_streams_sha256')]:
        changed = deepcopy(e); changed[field] = '0' * 64
        checks.reject('changed_cross_role_' + name + '_rejected', lambda changed=changed:
                      a.matched_clock_receipts(dict(non_saturating=c, energy_u=changed)))
    checks.reject('missing_clock_peer_rejected', lambda: a.matched_clock_receipts(dict(non_saturating=c)))
    for name in ('branch_model', 'branch_optimizer', 'branch_stream', 'missing_stream', 'missing_trainer_stream',
                 'missing_initializer', 'short_branch', 'nonbyte_stream'):
        changed = deepcopy(treatment)
        if name == 'branch_model': changed['branch_initial']['restart']['trainer']['models']['G']['weight'] = torch.tensor([8.])
        elif name == 'branch_optimizer': changed['branch_initial']['restart']['trainer']['optimizers'][0]['step'] = 3
        elif name == 'branch_stream': changed['branch_initial']['restart']['trainer']['streams']['latent_generator'] = torch.tensor([1, 1], dtype=torch.uint8)
        elif name == 'missing_stream': changed['trajectories']['reference'][0]['streams']['states'].pop(next(iter(changed['initial']['streams']['states'])))
        elif name == 'missing_trainer_stream': changed['trajectories']['reference'][0]['trainer']['streams'].pop('latent_generator')
        elif name == 'missing_initializer': changed['initial']['initialization'].pop('prior')
        elif name == 'short_branch': changed['trajectories']['restart'].pop()
        else: changed['trajectories']['reference'][0]['streams']['states'][next(iter(changed['initial']['streams']['states']))] = torch.tensor([0., 1.])
        checks.reject(name + '_rejected', lambda changed=changed: a.clock_receipt(changed))


def ordinary_device_fixtures(a, checks):
    parent = dict(initial_tensors_sha256=dict(G='a' * 64), actual_target_batches=dict(
        critic=dict(sha256='f' * 64, batches=1, samples=2), generator=dict(sha256='e' * 64, batches=1, samples=2)),
        initial_tensor_metadata=dict(G=dict(weight=dict(shape=[2, 2], dtype='torch.float32', device='cuda:0'))))
    actual = deepcopy(parent)
    actual['initial_tensor_metadata']['G']['weight']['device'] = 'cuda:1'
    a.matched_ordinary_comparisons(parent, actual)
    checks.passed('ordinary_different_cuda_worker_ordinals_accepted')
    for name, field, value in [('backend', 'device', 'cpu'), ('shape', 'shape', [2, 3]),
                              ('dtype', 'dtype', 'torch.float64')]:
        changed = deepcopy(actual); changed['initial_tensor_metadata']['G']['weight'][field] = value
        checks.reject('ordinary_changed_' + name + '_rejected', lambda changed=changed:
                      a.matched_ordinary_comparisons(parent, changed))
    for name in ('initial_tensor', 'batch_hash', 'batch_count'):
        changed = deepcopy(actual)
        if name == 'initial_tensor': changed['initial_tensors_sha256']['G'] = '0' * 64
        elif name == 'batch_hash': changed['actual_target_batches']['generator']['sha256'] = '0' * 64
        else: changed['actual_target_batches']['generator']['batches'] = 2
        checks.reject('ordinary_changed_' + name + '_rejected', lambda changed=changed:
                      a.matched_ordinary_comparisons(parent, changed))


def main():
    a, density = load('audit_collection.py'), load('density_audit.py')
    helper, checks = a.load_helper(), Checks()
    torch.set_num_threads(1)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with a.forbid_live_execution(helper), torch.no_grad():
        signed_and_source_fixtures(a, density, checks)
        assert len(checks.names) == 45
        clock_and_revision_fixtures(a, checks)
        assert len(checks.names) == 63
        ordinary_device_fixtures(a, checks)
        assert len(checks.names) == 70
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0] and before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:] and torch.equal(before[2], after[2])
    print(json.dumps(dict(status='PASS', scope='explicit_pure_software_fixtures_not_qualification',
        checks=len(checks.names), names=checks.names, source_hashes={str((OUT / name).relative_to(ROOT)): a.file_hash(OUT / name)
            for name in ('audit_collection.py', 'density_audit.py', 'software_audit_probe.py')},
        python_numpy_torch_cpu_rng_unchanged=True, model_optimizer_recipe_construction_training_sampling_draws=0,
        archived_grades_used=0), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
