"""Continuation software controls; no original acquisition or CUDA job runs."""
from copy import deepcopy
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'policy_hold_continuation',
    ROOT / 'reports/forge/continuous-baseline-20261003/run_hold_continuation.py')
hold = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hold)


def prefix():
    flags = [False, False, False, False, True, False, True, True, True,
             False, False, False, False, True, True, True, False, False,
             True, True, True, True, True, True, True]
    return [{'step': i * 50, 'passed': flag} for i, flag in enumerate(flags)]


def appended(flags=(True, True, True)):
    return [{'step': s, 'passed': flag} for s, flag in zip(hold.BOUNDARIES, flags)]


def test_original_prefix_is_not_relabelled_as_complete_hold():
    actual = hold.compound_gate(prefix(), [], completed=1200)
    assert actual['status'] == 'INCOMPLETE'
    assert actual['first_confirmed_step'] == 1100
    assert actual['hold_checks'] == actual['hold_passed'] == 2
    assert actual['original_gate'] == 'PASS'
    assert actual['original_study_gate'] == 'INCOMPLETE'


def test_only_three_actual_new_checks_complete_the_named_hold():
    actual = hold.compound_gate(prefix(), appended(), completed=1350)
    assert actual['status'] == 'PASS'
    assert actual['hold_checks'] == actual['hold_passed'] == 5
    assert actual['speed_eligible'] is False
    assert actual['original_study_gate'] == 'INCOMPLETE'


@pytest.mark.parametrize('failed', range(3))
def test_any_one_new_check_failure_rejects_even_if_later_checks_pass(failed):
    flags = [True] * 3; flags[failed] = False
    result = hold.compound_gate(prefix(), appended(flags), completed=1350)
    assert result['status'] == 'FAIL'
    assert result['passed'] is False
    assert result['full_protocol_complete'] is True


def test_interruption_never_zero_fills_future_hold():
    for cursor, count in [(1200, 0), (1227, 0), (1250, 1), (1317, 2), (1350, 2)]:
        result = hold.compound_gate(prefix(), appended()[:count], completed=cursor)
        assert result['status'] == 'INCOMPLETE'
        assert result['hold_checks'] == 2 + count


@pytest.mark.parametrize('mutation', ['missing_prefix', 'early_confirmation', 'restarted_hold',
                                    'nonbinary', 'wrong_boundary', 'duplicate', 'future', 'overrun'])
def test_distinct_or_fabricated_protocol_is_rejected(mutation):
    old, new, cursor = prefix(), appended(), 1350
    if mutation == 'missing_prefix': old.pop(1)
    elif mutation == 'early_confirmation':
        for observation in old[1:6]: observation['passed'] = True
    elif mutation == 'restarted_hold': old[-2]['passed'] = False
    elif mutation == 'nonbinary': new[0]['passed'] = 1
    elif mutation == 'wrong_boundary': new[0]['step'] = 1249
    elif mutation == 'duplicate': new[1]['step'] = 1250
    elif mutation == 'future': cursor = 1300
    else: cursor = 1351
    with pytest.raises(ValueError):
        hold.compound_gate(old, new, completed=cursor)


def test_source_guard_rejects_new_commit_changed_bytes_and_path_escape(tmp_path):
    path = tmp_path / 'particlegan.py'; path.write_text('original = True\n')
    source = {'commit': hold.ORIGINAL_COMMIT, 'files_sha256': {'particlegan.py': hold.sha(path)}}
    hold.verify_sources(tmp_path, source)
    with pytest.raises(ValueError, match='8021'):
        hold.verify_sources(tmp_path, {**source, 'commit': '1' * 40})
    path.write_text('original = False\n')
    with pytest.raises(ValueError, match='changed'):
        hold.verify_sources(tmp_path, source)
    outside = tmp_path.parent / (tmp_path.name + '-outside.py'); outside.write_text('same = True\n')
    with pytest.raises(ValueError, match='changed'):
        hold.verify_sources(tmp_path, {'commit': hold.ORIGINAL_COMMIT,
                                      'files_sha256': {'../' + outside.name: hold.sha(outside)}})


def test_same_module_bytes_from_maintained_namespace_are_not_accepted(tmp_path):
    old, maintained = tmp_path / 'old', tmp_path / 'maintained'
    old.mkdir(); maintained.mkdir()
    for directory in (old, maintained):
        (directory / 'particlegan.py').write_text('original = True\n')
    source = {'files_sha256': {'particlegan.py': hold.sha(old / 'particlegan.py')}}
    hold.guard_imports(old, source, modules={'particlegan': SimpleNamespace(__file__=str(old / 'particlegan.py'))})
    with pytest.raises(ValueError, match='wrong scientific import path'):
        hold.guard_imports(old, source, modules={'particlegan': SimpleNamespace(__file__=str(maintained / 'particlegan.py'))})


def namespace_module(directory, *, paths=None, spec_paths=None, origin=None):
    paths = [str(directory)] if paths is None else paths
    return SimpleNamespace(__file__=None, __path__=paths,
                           __spec__=SimpleNamespace(origin=origin,
                                                    submodule_search_locations=paths if spec_paths is None else spec_paths))


@pytest.mark.parametrize('mutation', ['missing_path', 'multiple_paths', 'current_path',
                                    'spec_path', 'not_namespace', 'no_descendants', 'changed_child'])
def test_namespace_lib_needs_one_exact_original_path_and_pinned_children(tmp_path, mutation):
    old = tmp_path / 'old'; lib = old / 'lib'; lib.mkdir(parents=True)
    child = lib / 'toy_models.py'; child.write_text('original = True\n')
    source = {'files_sha256': {'lib/toy_models.py': hold.sha(child)}}
    module = namespace_module(lib)
    children = {'lib': module, 'lib.toy_models': SimpleNamespace(__file__=str(child))}
    hold.guard_imports(old, source, modules=children)
    current = tmp_path / 'current/lib'; current.mkdir(parents=True)
    (current / 'toy_models.py').write_text(child.read_text())
    if mutation == 'missing_path': module.__path__ = []
    elif mutation == 'multiple_paths': module.__path__.append(str(current))
    elif mutation == 'current_path': module = namespace_module(current); children['lib'] = module
    elif mutation == 'spec_path': module.__spec__.submodule_search_locations = [str(current)]
    elif mutation == 'not_namespace': module.__spec__.origin = str(lib)
    elif mutation == 'no_descendants': source['files_sha256'] = {}
    else: children['lib.toy_models'].__file__ = str(current / 'toy_models.py')
    with pytest.raises(ValueError): hold.guard_imports(old, source, modules=children)


def test_actual_public_providers_and_selected_registry_import_in_a_fresh_cpu_namespace(tmp_path):
    # A self-contained software source export keeps this control portable to
    # shallow CI; it supplies no historical scientific-source qualification.
    export = tmp_path / 'export'; export.mkdir()
    files = {}
    for root_name in ('benchmarks', 'particlegan', 'experiments', 'lib'):
        for path in (ROOT / root_name).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            target = export / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target); files[relative] = hold.sha(path)
    source = {'commit': hold.ORIGINAL_COMMIT, 'files_sha256': files}
    declaration = tmp_path / 'software-source.json'; hold.write(declaration, source)
    command = (
        'import importlib.util,json; '
        f's=importlib.util.spec_from_file_location("hold",{str(Path(hold.__file__))!r}); '
        'm=importlib.util.module_from_spec(s);s.loader.exec_module(m); '
        f'source=json.load(open({str(declaration)!r}));m.activate_original({str(export)!r},source); '
        'from benchmarks.toy_audit import api_vectors,api_contract,api_run; '
        'case=api_vectors._registry()[m.CASE]; '
        'assert case["id"]==m.CASE and case["default_steps"]==1200; '
        'case["provider"]="api_vectors"; '
        'assert api_contract.validate_recipe_overrides(case,"atlas",m.KNOBS)==m.KNOBS; '
        f'm.guard_imports({str(export)!r},source); '
        'import lib,torch; '
        f'assert list(lib.__path__)==[{str(export / "lib")!r}]; '
        'assert not torch.cuda.is_initialized(); '
        'print("public provider/registry source preflight PASS; zero model/optimizer/sampler updates")'
    )
    environment = dict(os.environ); environment.pop('PYTHONPATH', None)
    environment['CUDA_VISIBLE_DEVICES'] = ''
    completed = subprocess.run([sys.executable, '-B', '-c', command], cwd=tmp_path,
                               env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert 'public provider/registry source preflight PASS' in completed.stdout


def test_fresh_child_namespace_switch_does_not_depend_on_historical_git_objects(tmp_path):
    old = tmp_path / 'old'; maintained = tmp_path / 'maintained'
    for directory, value in ((old, 'original'), (maintained, 'new')):
        (directory / 'particlegan').mkdir(parents=True)
        (directory / 'particlegan/__init__.py').write_text(f'origin = {value!r}\n')
    source = {'commit': hold.ORIGINAL_COMMIT,
              'files_sha256': {'particlegan/__init__.py': hold.sha(old / 'particlegan/__init__.py')}}
    command = (
        'import importlib.util,json,sys; '
        f's=importlib.util.spec_from_file_location("hold",{str(Path(hold.__file__))!r}); '
        'm=importlib.util.module_from_spec(s);s.loader.exec_module(m); '
        f'sys.path.insert(0,{str(maintained)!r}); '
        f'm.activate_original({str(old)!r},json.loads({json.dumps(source)!r})); '
        'import particlegan; '
        f'm.guard_imports({str(old)!r},json.loads({json.dumps(source)!r})); '
        'assert particlegan.origin=="original";print("original namespace verified")'
    )
    completed = subprocess.run([sys.executable, '-B', '-c', command], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert 'original namespace verified' in completed.stdout


def test_an_already_imported_current_scientific_namespace_is_refused():
    with pytest.raises(ValueError, match='fresh scientific namespace'):
        hold.activate_original(ROOT, {'commit': hold.ORIGINAL_COMMIT, 'files_sha256': {}})


@pytest.mark.parametrize('family', ['atlas', 'e22'])
def test_actual_public_checkpoint_restore_cap_only_extension_and_two_new_updates(family):
    from benchmarks.toy_audit.api_vectors import build_case
    with torch.random.fork_rng(devices=[]):
        fixture = build_case(hold.CASE, device='cpu', seed=24002, recipe_name=family,
                             max_steps=2, recipe_overrides=hold.KNOBS)
        fixture.step(); fixture.step()
        state = fixture.state_dict()
        immutable_input = deepcopy(state)
        before_prior = state['trainer']['models']['prior']
        resumed = build_case(hold.CASE, device='cpu', seed=24002, recipe_name=family,
                             max_steps=2, recipe_overrides=hold.KNOBS)
        proof = hold.restore(resumed, state, initial=2, final=4)
        assert proof['appended_update_limit'] == 2
        assert resumed.recipe.to_dict() == state['recipe']
        assert hold.tree_equal(resumed.state_dict()['trainer']['models']['prior'], before_prior)
        assert resumed.completed_steps == resumed.trainer.completed_steps == 2
        # Observation uses independent RNG; it cannot mutate optimizer, model,
        # caller data cursor, named/global RNG, policy or EMA state.
        before = resumed.state_dict()
        hold.observe_pure(resumed, n=128, seed=34002)
        assert hold.tree_equal(before, resumed.state_dict())
        assert resumed.step()['step'] == 3
        assert resumed.step()['step'] == 4
        assert resumed.completed_steps == resumed.trainer.completed_steps == 4
        observed_trajectory = resumed.state_dict()
        # Independent restored baseline takes the same two updates with no
        # observer. This checks update trajectory, not only self-state purity.
        unobserved = build_case(hold.CASE, device='cpu', seed=24002, recipe_name=family,
                                max_steps=2, recipe_overrides=hold.KNOBS)
        hold.restore(unobserved, state, initial=2, final=4)
        unobserved.step(); unobserved.step()
        assert hold.tree_equal(observed_trajectory, unobserved.state_dict())
        assert hold.tree_equal(state, immutable_input)
        with pytest.raises(RuntimeError, match='complete'):
            resumed.step()
        assert state['completed_steps'] == state['trainer']['completed_steps'] == 2


@pytest.mark.parametrize('mutation', ['cursor', 'cap', 'recipe', 'nonfinite', 'schedule'])
def test_actual_checkpoint_identity_failures_cannot_extend(mutation):
    from benchmarks.toy_audit.api_vectors import build_case
    fixture = build_case(hold.CASE, device='cpu', recipe_name='e22', max_steps=1,
                         recipe_overrides=hold.KNOBS)
    fixture.step(); state = fixture.state_dict()
    resumed = build_case(hold.CASE, device='cpu', recipe_name='e22', max_steps=1,
                         recipe_overrides=hold.KNOBS)
    if mutation == 'cursor': state['completed_steps'] = 0
    elif mutation == 'cap': state['trainer']['max_steps'] = 2
    elif mutation == 'recipe': state['recipe']['lr'] *= 2
    elif mutation == 'nonfinite':
        next(v for v in state['trainer']['models']['G'].values() if v.is_floating_point()).flatten()[0] = float('nan')
    else:
        from particlegan import get_recipe
        resumed.recipe = get_recipe('ka2', total_steps=3)
    with pytest.raises(ValueError):
        hold.restore(resumed, state, initial=1, final=3)
    assert resumed.completed_steps == 0
    assert resumed.execution_steps == resumed.trainer.max_steps == 1


def test_baseline_prerequisite_refuses_absent_negative_partial_and_unbound_positive(tmp_path):
    path = tmp_path / 'baseline.json'
    for candidate in [[], [{'id': 'atlas-original19-portability-img_intensity2', 'status': 'FAIL'}],
                      [{'id': 'atlas-original19-portability-img_intensity2', 'status': 'PASS',
                        'original_gate': 'PASS', 'full_protocol_complete': False}]]:
        hold.write(path, {'rows': candidate})
        with pytest.raises(ValueError): hold.baseline_control(path)


def test_incomplete_child_and_exit_zero_cannot_be_complete(tmp_path):
    row = {'id': 'c6-atlas-broad-hold-1200-to-1350', 'family': 'atlas'}
    packet = {'parents': {'atlas': {'receipt_sha256': 'receipt'}}, 'original_source': {},
              'source': {}, 'lane_runtime': {}}
    assert hold.certify(packet, row, tmp_path, 0)['status'] == 'INCOMPLETE'
    result = {'case_id': row['id'], 'family': 'atlas', 'original_receipt_sha256': 'receipt',
              'original_source': {}, 'maintained_source': {}, 'runtime': {}, 'status': 'INCOMPLETE'}
    hold.write(tmp_path / 'result.json', result)
    assert hold.certify(packet, row, tmp_path, 0)['full_protocol_complete'] is False


def test_cli_does_not_launch_without_explicit_verified_baseline():
    with pytest.raises(SystemExit) as error:
        hold.main(['--output', '/tmp/never-launched-hold-test'])
    assert error.value.code == 2


def test_direct_parent_cli_from_outside_checkout_imports_real_forge_without_pythonpath(tmp_path):
    output = tmp_path / 'never-launched-hold'
    snapshot = tmp_path / 'invalid-source'; snapshot.mkdir()
    preparation = output.parent / f'.{output.name}.hold-preparation.json'
    hold.write(preparation, {'execution_source': {
        'snapshot_path': str(snapshot), 'files': {}, 'digest': '0' * 64}})
    before = preparation.read_bytes()
    environment = dict(os.environ); environment.pop('PYTHONPATH', None)
    completed = subprocess.run(
        [sys.executable, '-B', str(Path(hold.__file__)), '--prepare-only', '--output', str(output),
         '--queue-root', str(tmp_path / 'never-created-queue')],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=20)
    assert completed.returncode != 0
    assert 'invalid source manifest digest' in completed.stderr
    assert 'experiments/forge/sources.py' in completed.stderr
    assert 'ModuleNotFoundError' not in completed.stderr
    assert preparation.read_bytes() == before
    assert not output.exists()
    assert not (tmp_path / 'never-created-queue').exists()


def recorded_clouds():
    from benchmarks.toy_audit.api_vectors import list_cases, score_case
    case = next(c for c in list_cases() if c['id'] == hold.CASE)
    arrays = {}
    observations = []
    for step in hold.BOUNDARIES:
        # A collapsed negative cloud. Forging a PASS scalar must never turn it
        # into passing evidence. Only the pure original scorer is exercised.
        values = np.zeros((hold.SAMPLES, 2), dtype=np.float32)
        score = score_case(case, torch.from_numpy(values), step)
        observations.append({**score, 'failed_bounds': sorted(set(score['failed_bounds'])),
                             'step': step, 'elapsed_seconds': (step - 1200) / 50,
                             'views': [{'kind': 'scatter'}, {'kind': 'bar'}, {'kind': 'scatter'}]})
        for index in range(3):
            for role in ('target', 'samples'):
                arrays[f'step{step}_view{index}_{role}'] = np.zeros(
                    (2,) if index == 1 else (hold.SAMPLES, 2), dtype=np.float32)
    return {'observations': observations}, arrays, case, score_case


def test_retained_numeric_fail_clouds_are_valid_negative_evidence_without_sampling():
    result, arrays, case, scorer = recorded_clouds()
    hold.validate_arrays(result, arrays, case, scorer)
    assert all(o['passed'] is False for o in result['observations'])
    assert all(len(o['failed_bounds']) > 1 for o in result['observations'])


@pytest.mark.parametrize('mutation', ['forged_pass', 'changed_metric', 'nonfinite', 'short_cloud',
                                    'missing_array', 'extra_array', 'late_timing', 'equal_timing'])
def test_source_bound_new_cloud_and_timing_controls(mutation):
    result, arrays, case, scorer = recorded_clouds()
    if mutation == 'forged_pass':
        result['observations'][0]['passed'] = True
        result['observations'][0]['failed_bounds'] = []
    elif mutation == 'changed_metric': result['observations'][0]['metrics']['mass_tv'] = 0.
    elif mutation == 'nonfinite': arrays['step1250_view0_samples'][0, 0] = float('nan')
    elif mutation == 'short_cloud': arrays['step1250_view0_samples'] = arrays['step1250_view0_samples'][:-1]
    elif mutation == 'missing_array': arrays.pop('step1250_view0_target')
    elif mutation == 'extra_array': arrays['step1400_view0_samples'] = arrays['step1250_view0_samples']
    elif mutation == 'late_timing': result['observations'][-1]['elapsed_seconds'] = 180.01
    else: result['observations'][1]['elapsed_seconds'] = result['observations'][0]['elapsed_seconds']
    with pytest.raises(ValueError): hold.validate_arrays(result, arrays, case, scorer)


def test_finished_child_cost_cannot_be_zeroed_or_reduced_coherently():
    cost = {'status': 'PASS', 'full_protocol_complete': True, 'paid_wall_seconds': 12.,
            'charged_seconds': 12., 'acquisition_seconds': 8., 'export_seconds': 2.}
    assert hold.verify_cost(cost) == 12.
    for paid in (0., 9., -1., float('nan')):
        with pytest.raises(ValueError):
            hold.verify_cost({**cost, 'paid_wall_seconds': paid, 'charged_seconds': paid})


def test_interrupt_retains_original_allowance_without_creating_a_retry_budget():
    assert hold.verify_cost({'status': 'INCOMPLETE', 'paid_wall_seconds': 7.,
                             'unmeasured_interrupt_reserved_seconds': 233., 'charged_seconds': 240.}) == 240.
    with pytest.raises(ValueError):
        hold.verify_cost({'status': 'INCOMPLETE', 'paid_wall_seconds': 7.,
                          'unmeasured_interrupt_reserved_seconds': 200., 'charged_seconds': 207.})


def test_recovery_keeps_source_bound_engineering_prefix_and_paid_debit(tmp_path):
    pins = deepcopy(hold.PREVIOUS_STUDIES)
    for family, pin in pins.items():
        name = f'c6-{family}-broad-hold-1200-to-1350'
        directory = tmp_path / family / name; directory.mkdir(parents=True)
        study = {'source': hold.PREVIOUS_SOURCE, 'spent_seconds': pin['paid_wall_seconds'],
                 'rows': [{'id': name, 'status': 'INCOMPLETE', 'child_returncode': 1,
                           'full_protocol_complete': False, 'result_path': None, 'result_sha256': None,
                           'timeout_seconds': hold.CAP, 'allowance_seconds': hold.CAP + hold.GRACE,
                           'paid_wall_seconds': pin['paid_wall_seconds'],
                           'charged_seconds': pin['paid_wall_seconds']}]}
        hold.write(directory.parent / 'study.json', study)
        pin['study_sha256'] = hold.sha(directory.parent / 'study.json')
        for filename, key in [('request.json', 'request_sha256'), ('run.log', 'log_sha256')]:
            (directory / filename).write_text('synthetic software fixture\n')
            pin[key] = hold.sha(directory / filename)
    result = hold.engineering_history(tmp_path, pins=pins)
    assert result['paid_seconds'] == 6.8186783420387655
    assert result['ordinary_updates'] == 0
    assert result['automatic_retry'] is False
    assert all(row['status'] == 'INCOMPLETE' for row in result['records'])
    # Even coherently rewriting the local costs/hash cannot erase the pinned
    # engineering debit under the frozen combined 480-second allowance.
    path = tmp_path / 'atlas/study.json'; modified = hold.read(path)
    modified['spent_seconds'] = 0.
    modified['rows'][0].update(paid_wall_seconds=0., charged_seconds=0.)
    hold.write(path, modified); pins['atlas']['study_sha256'] = hold.sha(path)
    with pytest.raises(ValueError, match='cost contract'):
        hold.engineering_history(tmp_path, pins=pins)


def test_old_engineering_charge_limits_new_attempt_reservations():
    debit = 6.8186783420387655
    assert hold.can_reserve(debit, 0.)
    assert hold.can_reserve(debit, 20.)
    assert not hold.can_reserve(debit, 234.)
    assert hold.TOTAL_CAP == 480.
    for paid in (-1., float('nan'), float('inf')):
        with pytest.raises(ValueError): hold.can_reserve(debit, paid)


def test_source_defined_unavailable_diagnostics_roundtrip_without_waiving_model_health():
    state = {'trainer': {'lr_settle': [[{'r_b': [torch.tensor([float('nan'), .2])],
                                        'last_look': {'t_2b': float('nan')},
                                        'last': {'t_b': float('inf'), 'log_bf_b': float('nan')}}]],
                         'models': {'G': {'weight': torch.ones(1)}},
                         'optimizers': [{'moment': torch.ones(1)}]}, 'data_rng': torch.ones(10, dtype=torch.uint8)}
    assert hold.tree_equal(state, deepcopy(state))
    assert hold.finite_state(state)
    different = deepcopy(state); different['trainer']['lr_settle'][0][0]['r_b'][0][0] = 0
    assert not hold.tree_equal(state, different)
    for protected in ('models', 'optimizers'):
        invalid = deepcopy(state)
        if protected == 'models': invalid['trainer']['models']['G']['weight'][0] = float('nan')
        else: invalid['trainer']['optimizers'][0]['moment'][0] = float('nan')
        assert not hold.finite_state(invalid)
    invalid = deepcopy(state); invalid['trainer']['lr_settle'][0][0]['tau'] = float('nan')
    assert not hold.finite_state(invalid)
    invalid = deepcopy(state); invalid['trainer']['lr_settle'][0][0]['r_b'][0][0] = float('inf')
    assert not hold.finite_state(invalid)


@pytest.mark.parametrize('container', ['last', 'last_look', 'log'])
@pytest.mark.parametrize('leaf', ['log_bf_b', 'log_bf_2b'])
def test_only_source_defined_negative_infinity_log_bf_sentinels_are_admitted(container, leaf):
    diagnostic = {leaf: float('-inf')}
    state = {'trainer': {'lr_settle': [[{container: [diagnostic] if container == 'log' else diagnostic}]]}}
    assert hold.finite_state(state)
    assert hold.tree_equal(state, deepcopy(state))
    diagnostic[leaf] = float('inf')
    assert not hold.finite_state(state)
    diagnostic[leaf] = 0.
    diagnostic['unknown_log_bf'] = float('-inf')
    assert not hold.finite_state(state)
