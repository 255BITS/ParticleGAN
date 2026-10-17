"""Saved-array media controls; no training, model, scorer or GPU execution."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'reports/forge/continuous-baseline-20261003/export_moving_goal.py'
spec = importlib.util.spec_from_file_location('moving_goal_software', MODULE)
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)


def save_study(archive):
    media.write(archive.study_path, archive.study)


def repin(archive, name):
    item = media.identity(archive.run / name)
    archive.row['artifacts'][name] = {k: item[k] for k in ('bytes', 'sha256')}
    if name == 'frames.npz.verdict.json':
        archive.row['result_sha256'] = item['sha256']
    save_study(archive)


def change_result(archive, mutate):
    path = archive.run / 'frames.npz.verdict.json'
    value = media.read(path)
    mutate(value)
    path.write_text(json.dumps(value) + '\n')
    repin(archive, path.name)


def change_arrays(archive, mutate):
    path = archive.run / 'frames.npz'
    with np.load(path, allow_pickle=False) as loaded:
        values = {name: loaded[name].copy() for name in loaded.files}
    mutate(values)
    np.savez_compressed(path, **values)
    repin(archive, path.name)


def change_definition(archive, mutate):
    mutate(archive.definition)
    archive.request['row']['case_sha256'] = archive.row['case_sha256'] = media.stable(archive.definition)
    media.write(archive.run / 'request.json', archive.request)
    repin(archive, 'request.json')


@pytest.fixture
def archive(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline'
    run = baseline / 'moving/grid100'
    run.mkdir(parents=True)
    snapshot = tmp_path / 'frozen-source'
    rotate = snapshot / 'atlas19-external/rotate_gate.py'
    rotate.parent.mkdir(parents=True)
    rotate.write_text('# Synthetic producer identity control; never executed.\n')
    driver = snapshot / media.BASELINE
    driver.parent.mkdir(parents=True)
    driver.write_text('# Synthetic wrapper identity control; never executed.\n')
    config = snapshot / media.CONFIG
    config.parent.mkdir(parents=True)
    media.write(config, {'recipe_name': 'atlas', 'lr': .00425, 'prior_lr_mult': 2})
    native = snapshot / 'particlegan/__init__.py'
    native.parent.mkdir()
    native.write_text('# Synthetic package identity control; never imported.\n')
    monkeypatch.setattr(media, 'PRODUCER_SHA', media.sha(rotate))
    monkeypatch.setattr(media, 'BASELINE_SHA', media.sha(driver))
    monkeypatch.setattr(media, 'CONFIG_SHA', media.sha(config))
    files = {str(p.relative_to(snapshot)): media.sha(p) for p in (rotate, driver, config, native)}
    source = {'snapshot_path': str(snapshot), 'files': files, 'digest': media.stable(files),
              'origin_commit': 'a' * 40, 'schema_version': 1}
    case_id = 'atlas-original19-moving-grid100'
    definition = {'id': case_id, 'group': 'moving', 'task': 'grid100',
                  'original_host': dict(steps=1500, seed=1234, num_particles=20000, z_dim=2,
                                        batch_size=2048, period_steps=500, rotate_degrees=30,
                                        turns=2, captured_points=4096, evaluation_samples=20000),
                  'observation_steps': [500, 1000, 1500],
                  'original_requirements': [['modes', '>=', 95], ['hq', '>=', '0.9 * observed pre_turn_hq']],
                  'config_sha256': media.CONFIG_SHA, 'reference_sha256': media.REFERENCE_SHA,
                  'original_options': deepcopy(media.OPTIONS),
                  'original_terminal_gate': True, 'added_policy_hold_gate': False}
    packet = {'family': 'atlas', 'source': {'commit': 'a' * 40, 'execution_digest': source['digest'],
                                         'files_sha256': {media.CONFIG: media.CONFIG_SHA, media.BASELINE: media.BASELINE_SHA}},
              'execution_source': source, 'snapshot_locations': {'rotate': str(rotate), 'package': str(snapshot)},
              'case_definitions': {case_id: definition},
              'recipe_overrides': {'original_config_sha256': media.CONFIG_SHA, 'original_options': deepcopy(media.OPTIONS)}}
    requested_row = {'id': case_id, 'group': 'moving', 'task': 'grid100', 'status': 'UNKNOWN',
                     'case_sha256': media.stable(definition)}
    request = {'packet': packet, 'row': requested_row, 'target': str(run)}
    media.write(run / 'request.json', request)
    (run / 'runner.py').write_text('# Synthetic executed-wrapper artifact; never executed.\n')
    for step in media.STEPS[1:]:
        (run / f'frames.npz.checkpoint-{step:06d}.pt').write_bytes(f'software checkpoint {step}'.encode())
    centers = np.stack(np.meshgrid(np.arange(10) - 4.5, np.arange(10) - 4.5, indexing='ij'), -1).reshape(100, 2).astype(np.float32)
    # Synthetic numerical controls, explicitly unrelated to scientific training.
    frames = np.stack([centers[np.arange(4096) % 100] + .01 * i for i in range(4)]).astype(np.float16)
    np.savez_compressed(run / 'frames.npz', frames=frames, steps=np.asarray(media.STEPS), centers=centers,
                        angles=np.asarray(media.ANGLES), final=json.dumps({'modes': 100, 'hq': .93}))
    result = {'task': 'grid100', 'status': 'PASS', 'rule': media.RULE, 'turns': 2,
              'pre_turn_hq': .95, 'passed_periods': 2,
              'periods': [{'task': 'grid100', 'period_end': step, 'target_deg': degrees, 'modes': 100, 'hq': hq}
                          for step, degrees, hq in zip([500, 1000, 1500], [0, 30, 60], [.95, .94, .93])]}
    media.write(run / 'frames.npz.verdict.json', result)
    images = [Image.new('RGB', (50, 50), color) for color in ('red', 'green', 'blue')]
    images[0].save(run / 'goal-metrics.gif', save_all=True, append_images=images[1:], duration=100)
    original_media = {**media.identity(run / 'goal-metrics.gif'), 'actual_steps': [500, 1000, 1500],
                      'frames': 3, 'metric_only': True, 'training_updates': 0, 'new_draws': False}
    media.write(run / 'media-receipt.json', original_media)
    row = {**requested_row, 'status': 'PASS', 'original_gate': 'PASS', 'full_protocol_complete': True,
           'completed_steps': 1500, 'child_returncode': 0, 'metric_observations': 3,
           'result_path': str(run / 'frames.npz.verdict.json'), 'result_sha256': media.sha(run / 'frames.npz.verdict.json'),
           'artifacts': {p.name: {'bytes': p.stat().st_size, 'sha256': media.sha(p)}
                         for p in run.iterdir() if p.is_file()}, 'media': original_media}
    study = {'rows': [row], 'case_definitions': {case_id: definition},
             'source': packet['source'], 'execution_source': source}
    fixture = SimpleNamespace(run=run, baseline=baseline, snapshot=snapshot, source=source, row=row,
                              packet=packet, request=request, definition=definition, study=study,
                              study_path=baseline / 'study.json', output=tmp_path / 'new-media')
    save_study(fixture)
    return fixture


def renderer_oracle(monkeypatch):
    monkeypatch.setattr(media, 'renderer_identity', lambda: {'commit': 'b' * 40, 'sha256': 'c' * 64,
                                                          'git_path': media.SELF, 'software_fixture': True})


def test_load_actual_contract_and_rotation_timing_without_evaluation(archive):
    data = media.load_run(archive.run)
    assert data['frames'].dtype == np.float16
    assert data['steps'].tolist() == [0, 500, 1000, 1500]
    assert data['angles'].tolist() == media.ANGLES
    assert data['result']['status'] == 'PASS'
    expected = archive.definition['original_requirements']
    assert expected == [['modes', '>=', 95], ['hq', '>=', '0.9 * observed pre_turn_hq']]


@pytest.mark.parametrize('mutation', ['steps', 'shape', 'dtype', 'nonfinite', 'angles', 'final_modes', 'final_hq', 'missing_key'])
def test_rebound_array_corruptions_reject_before_media(archive, mutation):
    def mutate(values):
        if mutation == 'steps': values['steps'][2] = 1001
        elif mutation == 'shape': values['frames'] = values['frames'][:3]
        elif mutation == 'dtype': values['frames'] = values['frames'].astype(np.float32)
        elif mutation == 'nonfinite': values['frames'][1, 0, 0] = np.nan
        elif mutation == 'angles': values['angles'][1] = np.pi / 6
        elif mutation == 'final_modes': values['final'] = np.asarray(json.dumps({'modes': 99, 'hq': .93}))
        elif mutation == 'final_hq': values['final'] = np.asarray(json.dumps({'modes': 100, 'hq': .1}))
        else: del values['centers']
    change_arrays(archive, mutate)
    with pytest.raises(ValueError, match='saved|schema|terminal'):
        media.export(archive.run, archive.output)
    assert not archive.output.exists()


@pytest.mark.parametrize('mutation', ['preturn', 'period_end', 'degrees', 'gate_status', 'nan_hq'])
def test_rebound_recorded_metrics_reject_mismatch_without_regating_cloud(archive, mutation):
    def mutate(value):
        if mutation == 'preturn': value['pre_turn_hq'] = .9
        elif mutation == 'period_end': value['periods'][1]['period_end'] = 1001
        elif mutation == 'degrees': value['periods'][0]['target_deg'] = 30
        elif mutation == 'gate_status': value['status'] = 'FAIL'
        else: value['periods'][1]['hq'] = float('nan')
    change_result(archive, mutate)
    with pytest.raises(ValueError, match='HQ|timing|internally'):
        media.load_run(archive.run)


@pytest.mark.parametrize('mutation', ['partial', 'unknown', 'missing_checkpoint', 'raw_tamper'])
def test_partial_or_missing_evidence_never_creates_goal_gif(archive, mutation):
    if mutation == 'partial': archive.row['completed_steps'] = 1000
    elif mutation == 'unknown': archive.row['status'] = 'UNKNOWN'
    elif mutation == 'missing_checkpoint': (archive.run / 'frames.npz.checkpoint-001500.pt').unlink()
    else: (archive.run / 'frames.npz').write_bytes(b'tampered array control')
    save_study(archive)
    with pytest.raises((ValueError, FileNotFoundError)):
        media.export(archive.run, archive.output)
    assert not archive.output.exists()


@pytest.mark.parametrize('mutation', ['seed', 'extra_math_field', 'noise', 'new_hold', 'recipe'])
def test_rebound_definition_preserves_original_math_and_recipe(archive, mutation):
    def mutate(value):
        if mutation == 'seed': value['original_host']['seed'] = 4321
        elif mutation == 'extra_math_field': value['original_host']['new_prior'] = 'other'
        elif mutation == 'noise': value['original_options']['eval_output_noise'] = False
        elif mutation == 'new_hold': value['added_policy_hold_gate'] = True
        else: value['config_sha256'] = 'd' * 64
    change_definition(archive, mutate)
    with pytest.raises(ValueError, match='resource|scope'):
        media.load_run(archive.run)


def test_changed_snapshot_package_is_detected_even_without_import(archive):
    (archive.snapshot / 'particlegan/__init__.py').write_text('# changed source control\n')
    with pytest.raises(ValueError, match='source changed'):
        media.load_run(archive.run)


def test_rebound_changed_config_still_rejects_original_identity(archive):
    path = archive.snapshot / media.CONFIG
    path.write_text('{"lr": 999}\n')
    archive.source['files'][media.CONFIG] = media.sha(path)
    archive.source['digest'] = media.stable(archive.source['files'])
    archive.packet['source']['execution_digest'] = archive.source['digest']
    media.write(archive.run / 'request.json', archive.request)
    repin(archive, 'request.json')
    with pytest.raises(ValueError, match='config/driver'):
        media.load_run(archive.run)


def test_source_manifest_escape_is_rejected(tmp_path):
    root = tmp_path / 'source'; root.mkdir()
    outside = tmp_path / 'outside.py'; outside.write_text('# control')
    files = {'../outside.py': media.sha(outside)}
    with pytest.raises(ValueError, match='source changed'):
        media.verify_source({'snapshot_path': str(root), 'files': files, 'digest': media.stable(files)})


@pytest.mark.parametrize('original_status', ['PASS', 'FAIL'])
def test_export_preserves_all_raw_bytes_and_original_full_verdict(archive, monkeypatch, original_status):
    renderer_oracle(monkeypatch)
    if original_status == 'FAIL':
        def fail_period(value):
            value['periods'][-1]['modes'] = 90
            value.update(status='FAIL', passed_periods=1)
        change_result(archive, fail_period)
        change_arrays(archive, lambda values: values.update(final=np.asarray(json.dumps({'modes': 90, 'hq': .93}))))
        archive.row.update(status='FAIL', original_gate='FAIL')
        save_study(archive)
    before = {str(p): p.read_bytes() for base in (archive.baseline, archive.snapshot) for p in base.rglob('*') if p.is_file()}
    receipt = media.export(archive.run, archive.output)
    after = {str(p): p.read_bytes() for base in (archive.baseline, archive.snapshot) for p in base.rglob('*') if p.is_file()}
    assert before == after
    assert receipt['original_gate'] == original_status
    assert receipt['training_updates'] == receipt['model_forwards'] == receipt['new_draws'] == 0
    assert receipt['rescoring'] is receipt['qualification_input'] is False
    assert receipt['raw_inputs_unchanged'] and receipt['original_metric_gif_unchanged']
    assert receipt['actual_steps'] == [0, 500, 1000, 1500]
    assert receipt['rotation_transition_updates'] == [501, 1001]
    assert receipt['displayed_observations'][0]['original_20k_period_observation'] is None
    assert receipt['display_samples']['count_per_frame'] == 4096 and receipt['gate_samples'] == 20000
    assert receipt['complete_execution_snapshot_verified']
    assert media.read(archive.output / 'moving-media-receipt.json') == receipt
    with Image.open(archive.output / 'moving-goal.gif') as image:
        assert image.n_frames == 4
    # A second posthoc render must not overwrite an earlier reviewed artifact.
    with pytest.raises(ValueError, match='empty output'):
        media.export(archive.run, archive.output)


@pytest.mark.parametrize('destination', ['same', 'child', 'ancestor'])
def test_output_cannot_change_original_archive(archive, monkeypatch, destination):
    renderer_oracle(monkeypatch)
    output = archive.run if destination == 'same' else archive.run / 'new-media' if destination == 'child' else archive.baseline
    with pytest.raises(ValueError, match='outside'):
        media.export(archive.run, output)


def test_source_bound_camera_zoom_ignores_apparent_success_and_keeps_full_extent():
    centers = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]] + [[8, 8]] * 96, dtype=np.float32)
    frames = np.zeros((4, 4096, 2), dtype=np.float16)
    targets, full, zoom, modes = media.camera(centers, media.ANGLES, frames)
    assert modes == [0, 1, 2, 3]
    changed = frames.copy(); changed[3, 0] = [-100, 100]
    targets2, full2, zoom2, modes2 = media.camera(centers, media.ANGLES, changed)
    assert np.array_equal(targets, targets2) and zoom == zoom2 and modes == modes2
    assert full2[0] < -100 and full2[3] > 100 and full != full2
    assert np.isclose(full[1] - full[0], full[3] - full[2])
    assert np.isclose(zoom[1] - zoom[0], zoom[3] - zoom[2])
    assert np.allclose(targets[1], centers)
    assert np.allclose(targets[2, 0], media.target_centers(centers, np.pi / 6)[0])


def test_renderer_commit_must_resolve_to_exact_file_bytes(tmp_path, monkeypatch):
    root = tmp_path / 'renderer'; path = root / media.SELF
    path.parent.mkdir(parents=True); path.write_text('# committed exporter control\n')
    monkeypatch.setattr(media, 'ROOT', root)
    calls = []
    def git(command, **kwargs):
        calls.append((command, kwargs['cwd']))
        return 'f' * 40 + '\n' if command[1] == 'rev-parse' else b'# committed exporter control\n'
    monkeypatch.setattr(media.subprocess, 'check_output', git)
    result = media.renderer_identity()
    assert result['sha256'] == media.sha(path) and result['commit'] == 'f' * 40
    assert calls[-1][0] == ['git', 'show', 'f' * 40 + ':' + media.SELF]
    path.write_text('# uncommitted changed exporter\n')
    with pytest.raises(ValueError, match='commit the exact'):
        media.renderer_identity()


def test_cli_help_requires_no_scientific_source_or_gpu(tmp_path):
    environment = {**os.environ, 'PYTHONPATH': '', 'CUDA_VISIBLE_DEVICES': ''}
    result = subprocess.run([sys.executable, '-B', str(MODULE), '--help'], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert '--run' in result.stdout and '--output' in result.stdout
