"""Original19 orchestration controls; no model, scientific training or GPU use."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import pytest

from experiments.forge import policy_execution as execution
from experiments.forge.contracts import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'reports/forge/continuous-baseline-20261003/run_atlas_baseline.py'
spec = importlib.util.spec_from_file_location('atlas_baseline_software', MODULE)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

COVERAGE = [('modes', '>=', 100), ('precision', '>=', .97),
            ('min_hq_mode_mass', '>=', .005), ('mass_tv', '<=', .1),
            ('max_mode_mass', '<=', .02), ('min_cov_eig_ratio', '>=', .4),
            ('max_cov_eig_ratio', '<=', 1.7), ('min_radial_median_ratio', '>=', .65),
            ('max_radial_median_ratio', '<=', 1.4)]
ACCURACY = [('acc_mass_tv', '<=', .06), ('acc_center_rms_sigma', '<=', .2),
            ('acc_abs_cov_trace_bias', '<=', .1), ('acc_radial_ks', '<=', .04)]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    (root / 'particlegan').mkdir(parents=True)
    (root / 'particlegan/__init__.py').write_text('VALUE = 1\n')
    helper = root / baseline.RELATIVE_SELF
    helper.parent.mkdir(parents=True)
    shutil.copyfile(MODULE, helper)
    config = root / baseline.CONFIG
    config.parent.mkdir(parents=True)
    atomic_json(config, {'lr': .00425, 'prior_lr_mult': 2., 'recipe_name': 'atlas'})
    harness = tmp_path / 'original-harness'
    initializer = tmp_path / 'original-initializer'
    native_root = tmp_path / 'original-native'
    for folder in (harness / 'tasks', initializer, native_root / 'particlegan'):
        folder.mkdir(parents=True)
    (initializer / 'initialization.py').write_text('VALUE = 2\n')
    (native_root / 'particlegan/__init__.py').write_text('VALUE = 3\n')
    (harness / 'native100_score.py').write_text("from pathlib import Path\nROOT = Path(FIXTURE['frozen_repo'])\n")
    adapter = root / baseline.ADAPTER
    adapter.mkdir(parents=True)
    (adapter / 'screen_current.py').write_text(
        f'NATIVE_COVERAGE_THRESHOLDS = {COVERAGE!r}\nNATIVE_ACCURACY_THRESHOLDS = {ACCURACY!r}\n')
    (adapter / 'current_api_fixtures.py').write_text('VALUE = 4\n')
    rotate = tmp_path / 'rotate_gate.py'
    rotate.write_text('VALUE = 5\n')
    image_specs = {task: {'steps': 600, 'z_dim': 8, 'batch_size': 32, 'particles': 32}
                   for task in baseline.PORTS if task.startswith('img_')}
    vector_specs = {task: {'spec': {'steps': 1600 if task == 'vector_spiral' else 1200,
                                   'z_dim': 2, 'batch': 128, 'particles': 256}}
                    for task in baseline.PORTS if task.startswith('vector_')}
    atomic_json(harness / 'tasks/image_task_specs.json', image_specs)
    atomic_json(harness / 'tasks/vector_task_specs.json', vector_specs)
    records = []
    for group, task in baseline.ordered_cases():
        if group == 'native': steps, observations = 7000, 34
        elif group == 'moving': steps, observations = 1500, 3
        elif task.startswith('img_'): steps, observations = 600, 24
        elif task.startswith('vector_'): steps, observations = vector_specs[task]['spec']['steps'], 24
        elif task == 'mode_hold': steps, observations = 1200, 24
        else: steps, observations = (4600, 460) if task == 'ring_shift' else (7500, 750)
        records.append({'group': group, 'task': task, 'completed_steps': steps,
                        'observations': observations, 'portability': {'requirements': [['hq', '>=', .9]]}})
    reference = {'results': records, 'original_protocol_sources': {'verified_source_sha256': {}}}
    refpath = root / baseline.REFERENCE
    refpath.parent.mkdir(parents=True)
    atomic_json(refpath, reference)
    files = {str(p): baseline.sha(p) for base in (harness, initializer, native_root, adapter)
             for p in base.rglob('*') if p.is_file()}
    files[str(rotate)] = baseline.sha(rotate)
    inputs = {'reference': reference, 'files': files, 'native_root': str(native_root),
              'harness': str(harness), 'initializer': str(initializer), 'rotate': str(rotate)}
    monkeypatch.setattr(baseline, 'CONFIG_SHA', baseline.sha(config))
    monkeypatch.setattr(baseline, 'REFERENCE_SHA', baseline.sha(refpath))
    monkeypatch.setattr(baseline, 'original_inputs', lambda _: deepcopy(inputs))
    real_check_output = subprocess.check_output
    def source_identity(command, *args, **kwargs):
        if command == ['git', 'rev-parse', 'HEAD']:
            return 'a' * 40 + '\n'
        return real_check_output(command, *args, **kwargs)
    monkeypatch.setattr(baseline.subprocess, 'check_output', source_identity)
    monkeypatch.setattr(baseline, 'runtime', lambda: {'python': 'software', 'torch': 'software', 'cuda': 'software',
                                                   'device': 'cuda:0', 'cuda_device_model': 'mock GPU', 'torch_threads': 1})
    monkeypatch.setattr(baseline, 'gpu_readiness', lambda: {'ready': True, 'reason': None})
    monkeypatch.setattr(execution, 'host_capacity', lambda: {'cpu_threads': 8, 'memory_mb': 65536,
                                                          'available_memory_mb': 65536})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    return types.SimpleNamespace(root=root, inputs=inputs, output=tmp_path / 'output', queue=tmp_path / 'queue')


def prepare(setup):
    return baseline.prepare(setup.output, root=setup.root, queue_root=setup.queue)


def evidence(packet, row, target, *, status='PASS', steps=None):
    target.mkdir(parents=True, exist_ok=True)
    definition = packet['case_definitions'][row['id']]
    final_steps = definition['original_host']['steps'] if steps is None else steps
    header = {'options': baseline.OPTIONS, 'overrides': {
        **read_json(Path(packet['snapshot_locations']['package']) / baseline.CONFIG),
        'initialization': 'batch_feature_zero'}, 'device': 'cuda:0', 'cuda_visible_devices': '1',
        'package_root': packet['snapshot_locations']['package'],
        'package_sha256': baseline.package_digest(packet['snapshot_locations']['package'])}
    result = {'status': status, 'task': row['task'], 'completed_steps': final_steps,
              'stream_deviations': 0, 'header': header, 'final': {'hq': .96}}
    atomic_json(target / 'result.json', result)
    values = [{'step': s, 'hq': .96, 'modes': 100, 'precision': .98, 'mass_tv': .03,
               'acc_center_rms_sigma': .1, 'acc_abs_cov_trace_bias': .03, 'acc_radial_ks': .02}
              for s in definition['observation_steps'] if s <= final_steps]
    (target / 'metrics.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in values))
    (target / 'final-state.pt').write_bytes(b'software-only complete state control')
    return result


def fake_launch(monkeypatch, *, status='PASS', steps=None, paid=1.):
    calls = []
    def launch(self, command, packet, log, leases, allowance):
        request = read_json(command[-1]); row = request['row']; target = Path(request['target'])
        calls.append(row['id'])
        evidence(packet, row, target, status=status, steps=steps)
        log.write_text('mocked public scientific child\n')
        result = subprocess.CompletedProcess(command, 0)
        result.paid_wall_seconds = paid
        return result
    monkeypatch.setattr(execution.PolicyCoordinator, 'launch', launch)
    return calls


def run(setup, **kwargs):
    return baseline.run(setup.output, root=setup.root, queue_root=setup.queue, **kwargs)


def no_media(monkeypatch):
    def render(packet, row, target):
        path = target / 'software-media-control'
        path.write_bytes(b'No scientific frames: synthetic hash control')
        return {'path': str(path), 'sha256': baseline.sha(path), 'bytes': path.stat().st_size}
    monkeypatch.setattr(baseline, 'render_metric_gif', render)


def test_direct_cli_imports_owned_checkout_without_pythonpath(tmp_path):
    environment={**os.environ,'PYTHONPATH':''}
    result=subprocess.run([sys.executable,'-B',str(MODULE),'--help'],cwd=tmp_path,
                          env=environment,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert '--max-new-attempts' in result.stdout


def test_original19_scope_order_resources_and_exact_native_thresholds(setup):
    packet = baseline.plan(setup.root)
    assert len(packet['rows']) == packet['required'] == 19
    assert [r['task'] for r in packet['rows'][:13]] == list(baseline.PORTS)
    assert sum(d['original_host']['steps'] for d in packet['case_definitions'].values()) == 48800
    native = packet['case_definitions']['atlas-original19-native-grid100']
    assert native['original_requirements'] == [list(v) for v in COVERAGE + ACCURACY]
    assert native['observation_steps'] == [0, 1, 10, 25, 50, 100] + list(range(250, 7001, 250))
    assert native['original_host']['holdout_samples'] == 100000
    assert native['original_host']['terminal_steps'] == [6000, 6250, 6500, 6750, 7000]
    assert packet['case_definitions']['atlas-original19-portability-stationary']['observation_steps'] == list(range(10, 7501, 10))
    assert packet['case_definitions']['atlas-original19-portability-ring_shift']['original_options']['ring_frozen_control'] is False
    assert all(not d['added_policy_hold_gate'] for d in packet['case_definitions'].values())
    assert [r['timeout_seconds'] for r in packet['rows']] == [1800.] * 16 + [2400.] * 3
    assert not packet['qualification_input']


def test_real_prepare_register_admit_complete_and_scientific_fail_exit0(setup, monkeypatch):
    prepared = prepare(setup)
    assert baseline.preparation_path(setup.output).is_file()
    assert not setup.output.exists()  # Real register must accept this fresh archive.
    calls = fake_launch(monkeypatch, status='FAIL')
    no_media(monkeypatch)
    packet = run(setup, max_new_attempts=1)
    first = packet['rows'][0]
    assert first['status'] == first['original_gate'] == 'FAIL'
    assert first['full_protocol_complete'] and first['child_returncode'] == 0
    assert calls == [first['id']]
    assert packet['completed'] == 1 and packet['required'] == 19
    ledger = execution.PolicyCoordinator(setup.queue).retained(first['attempt_key'])
    assert ledger['charged_seconds'] == first['charged_seconds'] == 1.
    assert packet['spent_seconds'] == packet['new_paid_seconds'] == 1.
    packet = run(setup, max_new_attempts=1)
    assert calls == [prepared['rows'][0]['id'], prepared['rows'][1]['id']]
    assert packet['rows'][0]['status'] == 'FAIL'  # Independent diagnostics continue; no retry.


def test_partial_exit0_is_incomplete_and_never_retried(setup, monkeypatch):
    calls = fake_launch(monkeypatch, steps=599)
    packet = run(setup, max_new_attempts=1)
    assert packet['rows'][0]['status'] == 'INCOMPLETE'
    assert not packet['rows'][0].get('full_protocol_complete')
    assert not packet['rows'][0].get('media')
    run(setup, max_new_attempts=1)
    assert calls.count('atlas-original19-portability-img_intensity2') == 1


def test_second_output_attaches_to_same_physical_attempt(setup, monkeypatch):
    calls = fake_launch(monkeypatch)
    no_media(monkeypatch)
    first = run(setup, max_new_attempts=1)
    second_output = setup.output.parent / 'alias'
    second = baseline.run(second_output, root=setup.root, queue_root=setup.queue, max_new_attempts=1)
    assert second['coordinator']['attached'] is True
    assert second['rows'][0]['attempt_key'] == first['rows'][0]['attempt_key']
    assert calls.count(first['rows'][0]['id']) == 1


def test_preparation_is_immutable_after_report_source_edits(setup):
    packet = prepare(setup)
    (setup.root / baseline.RELATIVE_SELF).write_text('# future implementation\n')
    again = prepare(setup)
    assert again == packet
    source = Path(packet['execution_source']['snapshot_path']) / baseline.RELATIVE_SELF
    source.write_text('# tampered frozen implementation\n')
    with pytest.raises(ValueError, match='changed'):
        prepare(setup)


@pytest.mark.parametrize('mutation', ['source', 'seed', 'options', 'gate', 'cadence', 'config', 'runtime', 'cap', 'frames'])
def test_attempt_identity_binds_every_scientific_protocol_axis(setup, mutation):
    packet = prepare(setup); packet['lane_runtime'] = baseline.runtime(); row = packet['rows'][0]
    key = baseline.attempt_identity(packet, row)
    other = deepcopy(packet); altered_row = deepcopy(row); definition = other['case_definitions'][row['id']]
    if mutation == 'source': other['execution_source']['digest'] = 'different'
    elif mutation == 'seed': definition['original_host']['seed'] += 1
    elif mutation == 'options': definition['original_options']['eval_output_noise'] = False
    elif mutation == 'gate': definition['original_requirements'][0][2] = .8
    elif mutation == 'cadence': definition['observation_steps'][0] += 1
    elif mutation == 'config': other['recipe_overrides']['original_config_sha256'] = 'different'
    elif mutation == 'runtime': other['lane_runtime']['cuda_device_model'] = 'different'
    elif mutation == 'cap': altered_row['timeout_seconds'] += 1
    else: definition['frame_contract']['maximum_frames'] = 8
    assert baseline.attempt_identity(other, altered_row) != key


def test_hot_gpu_and_complete_next_reservation_stop_without_launch(setup, monkeypatch):
    calls = fake_launch(monkeypatch)
    monkeypatch.setattr(baseline, 'gpu_readiness', lambda: {'ready': False, 'reason': 'GPU1 hot: 83 C'})
    packet = run(setup, max_new_attempts=1)
    assert not calls and packet['waiting_reason'] == 'GPU1 hot: 83 C'
    saved = read_json(setup.output / 'study.json')
    saved['spent_seconds'] = baseline.TOTAL_CAP - saved['rows'][0]['allowance_seconds'] + 1
    atomic_json(setup.output / 'study.json', saved)
    monkeypatch.setattr(baseline, 'gpu_readiness', lambda: {'ready': True})
    packet = run(setup, max_new_attempts=1)
    assert not calls and 'complete next original task' in packet['waiting_reason']
    assert packet['rows'][0]['status'] == 'UNKNOWN'


@pytest.mark.parametrize('kind,expected', [('measured_timeout', 3.), ('unmeasured_timeout', 1860.), ('failed_certification', 2.), ('unmeasured_error', 1860.)])
def test_actual_or_explicit_unmeasured_cost_matches_central_ledger(setup, monkeypatch, kind, expected):
    def launch(self, command, packet, log, leases, allowance):
        if kind == 'failed_certification':
            request = read_json(command[-1]); target = Path(request['target'])
            result = evidence(packet, request['row'], target)
            result['stream_deviations'] = 1; atomic_json(target / 'result.json', result)
            done = subprocess.CompletedProcess(command, 0); done.paid_wall_seconds = 2.; return done
        if kind.endswith('timeout'):
            error = subprocess.TimeoutExpired(command, allowance)
            if kind.startswith('measured'): error.paid_wall_seconds = 3.
            raise error
        raise RuntimeError('no durable child terminal')
    monkeypatch.setattr(execution.PolicyCoordinator, 'launch', launch)
    packet = run(setup, max_new_attempts=1); row = packet['rows'][0]
    saved = execution.PolicyCoordinator(setup.queue).retained(row['attempt_key'])
    assert row['charged_seconds'] == saved['charged_seconds'] == expected
    assert row['charged_seconds'] == row['paid_wall_seconds'] + row.get('unmeasured_interrupt_reserved_seconds', 0.)


def test_media_failure_preserves_original_gate_and_required_evidence_is_incomplete(setup, monkeypatch):
    fake_launch(monkeypatch)
    monkeypatch.setattr(baseline, 'render_metric_gif', lambda *args: (_ for _ in ()).throw(ValueError('media export failed')))
    packet = run(setup, max_new_attempts=1)
    assert packet['rows'][0]['status'] == packet['rows'][0]['original_gate'] == 'PASS'
    assert packet['rows'][0]['media_error'] == 'ValueError: media export failed'
    assert packet['media_completed'] == 0 and not packet['required_evidence_complete']


def test_gif_uses_actual_observations_and_preserves_all_raw_bytes(setup):
    packet = prepare(setup); row = packet['rows'][0]; row['status'] = 'FAIL'
    target = setup.output / 'plot'; evidence(packet, row, target, status='FAIL')
    before = {p.name: baseline.sha(p) for p in target.iterdir()}
    media = baseline.render_metric_gif(packet, row, target)
    assert media['actual_steps'][0] == 25 and media['actual_steps'][-1] == 600
    assert media['frames'] == 9 and media['metric_only'] and not media['new_draws']
    assert media['training_updates'] == 0
    with Image.open(media['path']) as gif: assert gif.n_frames == 9
    assert before == {name: baseline.sha(target / name) for name in before}
    (target / 'metrics.jsonl').write_text((target / 'metrics.jsonl').read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='complete actual cadence'):
        baseline.render_metric_gif(packet, row, target)


def test_native_gif_preserves_unavailable_accuracy_and_binds_its_renderer(setup):
    packet = prepare(setup); row = packet['rows'][16]; row['status'] = 'PASS'
    target = setup.output / 'native-media'; evidence(packet, row, target)
    path = target / 'metrics.jsonl'
    records = baseline._jsonl(path)
    for record in records[:2]:
        record['acc_center_rms_sigma'] = None
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    before = {p.name: baseline.sha(p) for p in target.iterdir()}
    media = baseline.render_metric_gif(packet, row, target)
    assert media['frames'] == 9 and media['training_updates'] == 0
    assert media['unavailable_observations'] == {'acc_center_rms_sigma': [0, 1]}
    assert media['renderer_source']['sha256'] == baseline.sha(MODULE)
    assert media['renderer_source']['scientific_source_digest'] == packet['source']['execution_digest']
    assert before == {name: baseline.sha(target / name) for name in before}
    assert baseline._jsonl(path)[:2] == records[:2]


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_native_gif_rejects_nonfinite_values_without_treating_them_as_unavailable(setup, invalid):
    packet = prepare(setup); row = packet['rows'][16]
    target = setup.output / 'invalid-media'; evidence(packet, row, target)
    path = target / 'metrics.jsonl'; records = baseline._jsonl(path)
    records[0]['acc_center_rms_sigma'] = invalid
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    with pytest.raises(ValueError, match='nonfinite retained plot metric'):
        baseline.render_metric_gif(packet, row, target)
    assert not (target / 'goal-metrics.gif').exists()


@pytest.mark.parametrize('coverage,accuracy,expected', [('FAIL', 'PASS', 'FAIL'), ('PASS', 'PASS', 'PASS'), ('PASS', 'FAIL', 'FAIL')])
def test_native_original_gate_requires_both_coverage_and_accuracy(setup, coverage, accuracy, expected):
    packet = prepare(setup); row = packet['rows'][16]; target = setup.output / 'native'
    evidence(packet, row, target, status=accuracy)
    definition = packet['case_definitions'][row['id']]
    for law in ('noisy', 'clean'):
        folder = target / f'native-{law}'; folder.mkdir()
        atomic_json(folder / 'summary.json', {'completed_steps': 7000, 'eval_steps': definition['observation_steps'], 'accuracy': {'holdout_samples': 100000}})
        atomic_json(folder / 'verdict.json', {'coverage': {'status': coverage}, 'accuracy': {'status': accuracy}, 'sources': {}})
        for relative, count in [('final_samples.npz', 20000), ('holdout_samples.npz', 100000)] + [(f'quality_checks/step_{s:06d}.npz', 20000) for s in definition['original_host']['terminal_steps']]:
            path = folder / relative; path.parent.mkdir(exist_ok=True)
            values = np.zeros((count, 2), dtype=np.float32)
            np.savez_compressed(path, live=values, ema=values, target=values)
    result = baseline.certify(packet, row, target, 0)
    assert result['status'] == result['original_protocol_gate'] == expected
    assert result['reported_original_status'] == accuracy
    assert result['full_protocol_complete']
    assert result['native_gates']['noisy'] == {'coverage': coverage, 'accuracy': accuracy}
