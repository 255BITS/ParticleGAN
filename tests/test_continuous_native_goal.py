"""Native media integrity controls on synthetic arrays; no scientific execution."""
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
MODULE = ROOT / 'reports/forge/continuous-baseline-20261003/export_native_goal.py'
spec = importlib.util.spec_from_file_location('native_goal_software', MODULE)
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)

COVERAGE = [('modes', '>=', 100), ('precision', '>=', .97), ('min_hq_mode_mass', '>=', .005),
            ('mass_tv', '<=', .1), ('max_mode_mass', '<=', .02), ('min_cov_eig_ratio', '>=', .4),
            ('max_cov_eig_ratio', '<=', 1.7), ('min_radial_median_ratio', '>=', .65), ('max_radial_median_ratio', '<=', 1.4)]
ACCURACY = [('acc_mass_tv', '<=', .06), ('acc_center_rms_sigma', '<=', .2),
            ('acc_abs_cov_trace_bias', '<=', .1), ('acc_radial_ks', '<=', .04)]


def repin(archive, name):
    file = archive.run / name
    archive.row['artifacts'][name] = {'bytes': file.stat().st_size, 'sha256': media.sha(file)}
    if name == 'result.json': archive.row['result_sha256'] = media.sha(file)
    media.write(archive.study_path, archive.study)


def change_json(archive, name, mutation):
    path = archive.run / name; value = media.read(path); mutation(value)
    media.write(path, value); repin(archive, name)


def change_arrays(archive, name, mutation):
    path = archive.run / name
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    mutation(arrays); np.savez_compressed(path, **arrays); repin(archive, name)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    run = tmp_path / 'baseline/native/grid100'; run.mkdir(parents=True)
    snapshot = tmp_path / 'frozen'; snapshot.mkdir()
    adapter = snapshot / 'atlas19-external/adapter'; adapter.mkdir(parents=True)
    screen = adapter / 'screen_current.py'
    screen.write_text(f'NATIVE_COVERAGE_THRESHOLDS = {COVERAGE!r}\nNATIVE_ACCURACY_THRESHOLDS = {ACCURACY!r}\n')
    monkeypatch.setattr(media, 'SCREEN_SHA', media.sha(screen))
    driver = snapshot / media.core.BASELINE; driver.parent.mkdir(parents=True)
    driver.write_text('# Synthetic source identity; never executed\n')
    monkeypatch.setattr(media.core, 'BASELINE_SHA', media.sha(driver))
    config = snapshot / media.core.CONFIG; config.parent.mkdir(parents=True)
    media.write(config, {'recipe_name': 'atlas', 'lr': .00425, 'prior_lr_mult': 2})
    monkeypatch.setattr(media.core, 'CONFIG_SHA', media.sha(config))
    harness = snapshot / 'atlas19-external/harness'; (harness / 'tasks').mkdir(parents=True)
    native_root = snapshot / 'atlas19-external/native_root'; native_root.mkdir(parents=True)
    scorer_file = native_root / 'score.py'; scorer_file.write_text('# Synthetic source; never imported\n')
    native_sources = {'score.py': media.sha(scorer_file)}
    host_config = dict(steps=7000, seed=1234, num_particles=20000, z_dim=2, batch_size=2048,
                       toy100_model='affine_square_v1', g_hidden=128, d_hidden=128, n_hidden=3,
                       fourier=3, eval_samples=20000, snapshot_samples=4096, eval_interval=250,
                       snapshot_interval=250, stable_evals=5, early_eval_steps=[0, 1, 10, 25, 50, 100], threads=1)
    media.write(harness / 'tasks/native100_fixture.json', {'host_fields': list(host_config), 'host_source_sha256': native_sources})
    media.write(harness / 'tasks/native100_constraints_simple_regularization.json', host_config)
    files = {str(p.relative_to(snapshot)): media.sha(p) for p in snapshot.rglob('*') if p.is_file()}
    source = dict(snapshot_path=str(snapshot), files=files, digest=media.stable(files), origin_commit='a' * 40)
    case_id = 'atlas-original19-native-grid100'
    definition = dict(id=case_id, group='native', task='grid100',
                      original_host=dict(steps=7000, seed=1234, num_particles=20000, z_dim=2, batch_size=2048,
                                         evaluation_samples=20000, terminal_steps=media.TERMINAL_STEPS, holdout_samples=100000,
                                         laws=['noisy primary', 'clean diagnostic']),
                      observation_steps=media.STEPS, original_options=deepcopy(media.core.OPTIONS),
                      config_sha256=media.core.CONFIG_SHA, reference_sha256=media.core.REFERENCE_SHA,
                      original_terminal_gate=True, added_policy_hold_gate=False,
                      original_requirements=[list(v) for v in COVERAGE + ACCURACY])
    packet = dict(family='atlas', source=dict(commit='a' * 40, execution_digest=source['digest'],
                                            files_sha256={media.core.CONFIG: media.core.CONFIG_SHA, media.core.BASELINE: media.core.BASELINE_SHA}),
                  execution_source=source, case_definitions={case_id: definition},
                  snapshot_locations=dict(package=str(snapshot), adapter=str(adapter), harness=str(harness), native_root=str(native_root)),
                  recipe_overrides=dict(original_config_sha256=media.core.CONFIG_SHA, original_options=deepcopy(media.core.OPTIONS)))
    requested = dict(id=case_id, group='native', task='grid100', case_sha256=media.stable(definition), status='UNKNOWN')
    media.write(run / 'request.json', dict(packet=packet, row=requested, target=str(run)))
    for filename in ('final-state.pt', 'adapter-relocation.json', 'native-score-wrapper.py'):
        (run / filename).write_bytes(b'synthetic identity-only control; never executed')
    full_config = {**host_config, 'problem': 'grid100', 'device': 'cuda:0', 'host': media.HOST_MESSAGE}
    coverage_metrics = dict(n=20000, valid_n=20000, modes=100, hq=.98, precision=.98, mass_tv=.02,
                            min_hq_mode_mass=.008, max_mode_mass=.012, min_cov_eig_ratio=.8,
                            max_cov_eig_ratio=1.2, min_radial_median_ratio=.8, max_radial_median_ratio=1.2, passed=True)
    accuracy_metrics = dict(protocol='toy100-accuracy-v1', n=20000, valid_n=20000, all_finite=True,
                            center_rms_sigma=.1, abs_cov_trace_bias=.05, radial_ks=.02, mass_tv=.02,
                            frozen_pass=True, accuracy_pass=True, passed=True)
    observations = []
    for step in media.STEPS:
        metrics = {key: coverage_metrics[key] for key in ('modes', 'precision', 'mass_tv')}
        metrics.update({f'acc_{key}': accuracy_metrics[key] for key in ('center_rms_sigma', 'abs_cov_trace_bias', 'radial_ks')})
        if step == 0:
            metrics.update(modes=0, acc_center_rms_sigma=None, acc_abs_cov_trace_bias=None, acc_radial_ks=None)
        observations.append(dict(step=step, **metrics, **{'pass': step > 0, 'acc_passed': step > 0}))
    (run / 'metrics.jsonl').write_text(''.join(json.dumps(o) + '\n' for o in observations))
    result = dict(task='grid100', status='PASS', completed_steps=7000, stream_deviations=0, eval_output_noise=True,
                  header=dict(options=media.core.OPTIONS, device='cuda:0', cuda_visible_devices='1', package_root=str(snapshot),
                              overrides={**media.read(config), 'initialization': 'batch_feature_zero'}))
    media.write(run / 'result.json', result)
    centers = np.stack(np.meshgrid(np.arange(10) - 4.5, np.arange(10) - 4.5, indexing='ij'), -1).reshape(100, 2).astype(np.float32)
    def arrays(count, step):
        reference = centers[np.arange(count) % 100]
        samples = (reference + np.float32(.1 * (7000 - step) / 7000)).astype(np.float32)
        return dict(live=samples, ema=samples.copy(), target=reference.copy())
    for law in ('noisy', 'clean'):
        folder = run / f'native-{law}'; (folder / 'quality_checks').mkdir(parents=True)
        cov_m = deepcopy(coverage_metrics); acc_m = deepcopy(accuracy_metrics)
        passed = law == 'noisy'; status = 'PASS' if passed else 'FAIL'
        cov_m['passed'] = passed; acc_m.update(passed=passed, frozen_pass=passed, accuracy_pass=passed)
        hold_m = {**acc_m, 'n': 100000, 'valid_n': 100000}
        summary = dict(status='complete', problem='grid100', budget_steps=7000, completed_steps=7000,
                       config=full_config, evaluation=law, eval_steps=media.STEPS, snapshot_steps=media.STEPS,
                       final=dict(live=cov_m, ema=deepcopy(cov_m)), holdout=dict(live=hold_m, ema=deepcopy(hold_m)),
                       accuracy=dict(protocol='toy100-accuracy-v1', check_steps=media.TERMINAL_STEPS, sample_count=20000,
                                     holdout_samples=100000, holdout_seed_offsets=dict(target=1601, noise=1602, latent=1603)),
                       accuracy_check_steps=media.TERMINAL_STEPS)
        cov = dict(problem='grid100', status=status, passed=passed, budget_steps=7000, final_step=7000,
                   observations=34, required_stable_checks=5, stable_checks=5 if passed else 0,
                   audited_final_samples=True, final_metrics=cov_m)
        acc = dict(problem='grid100', status=status, passed=passed, coverage_status=status,
                   terminal_checks=[dict(step=step, passed=passed, metrics=deepcopy(acc_m)) for step in media.TERMINAL_STEPS],
                   final_metrics=acc_m, holdout_metrics=hold_m)
        media.write(folder / 'config.json', full_config); media.write(folder / 'summary.json', summary)
        media.write(folder / 'verdict.json', dict(coverage=cov, accuracy=acc, sources=native_sources))
        for step in media.TERMINAL_STEPS:
            np.savez_compressed(folder / f'quality_checks/step_{step:06d}.npz', **arrays(20000, step))
        np.savez_compressed(folder / 'final_samples.npz', **arrays(20000, 7000))
        np.savez_compressed(folder / 'holdout_samples.npz', **arrays(100000, 7000))
        events = []
        for point in observations:
            cov_point = {**cov_m, 'modes': point['modes']}
            acc_point = {**acc_m, **{key[4:]: point[key] for key in media.METRICS if key.startswith('acc_')}}
            events.extend(dict(event='eval', step=point['step'], model=model, metrics=deepcopy(cov_point), accuracy=deepcopy(acc_point))
                          for model in ('live', 'ema'))
        (folder / 'events.jsonl').write_text(''.join(json.dumps(o) + '\n' for o in events))
        if law == 'noisy':
            (folder / 'snapshots').mkdir()
            for step in media.STEPS:
                np.savez_compressed(folder / f'snapshots/step_{step:06d}.npz', **arrays(4096, step))
    images = [Image.new('RGB', (20, 20), color) for color in ('red', 'green')]
    images[0].save(run / 'goal-metrics.gif', save_all=True, append_images=images[1:], duration=100)
    original_media = {**media.identity(run / 'goal-metrics.gif'), 'metric_only': True}
    media.write(run / 'media-receipt.json', original_media)
    row = dict(**requested, original_gate='PASS', full_protocol_complete=True, completed_steps=7000,
               child_returncode=0, metric_observations=34, reported_original_status='PASS',
               result_path=str(run / 'result.json'), result_sha256=media.sha(run / 'result.json'),
               native_gates=dict(noisy=dict(coverage='PASS', accuracy='PASS'), clean=dict(coverage='FAIL', accuracy='FAIL')),
               media=original_media)
    row['status'] = 'PASS'
    row['artifacts'] = {str(p.relative_to(run)): dict(sha256=media.sha(p), bytes=p.stat().st_size)
                        for p in run.rglob('*') if p.is_file()}
    study = dict(rows=[row], case_definitions={case_id: definition}, source=packet['source'], execution_source=source)
    fixture = SimpleNamespace(run=run, snapshot=snapshot, row=row, packet=packet, definition=definition,
                              study=study, study_path=run.parents[1] / 'study.json', output=tmp_path / 'new-media')
    media.write(fixture.study_path, study)
    return fixture


def oracle(monkeypatch):
    monkeypatch.setattr(media, 'renderer_identity', lambda: dict(commit='b' * 40, files={}, synthetic_software_oracle=True))


def test_native_scope_keeps_original100_modes_full34_and_none(archive):
    data = media.load_run(archive.run)
    assert len(data['snapshots']) == len(data['observations']) == 34
    assert data['observations'][0]['acc_center_rms_sigma'] is None
    assert data['original_gate'] == 'PASS'
    assert data['thresholds'] == [list(v) for v in COVERAGE + ACCURACY]
    assert data['verdicts']['clean']['accuracy']['status'] == 'FAIL'
    assert media.DISPLAY_STEPS[0] == 0 and media.DISPLAY_STEPS[-1] == 7000
    assert media.format_metric(None) == 'unavailable'


@pytest.mark.parametrize('mutation', ['budget', 'missing_snapshot', 'count', 'nonfinite', 'target_change', 'terminal_prefix', 'missing_holdout'])
def test_rebound_partial_or_array_corruptions_are_rejected(archive, mutation):
    if mutation == 'budget':
        change_json(archive, 'native-noisy/summary.json', lambda x: x.update(completed_steps=6999))
    elif mutation == 'missing_snapshot':
        (archive.run / 'native-noisy/snapshots/step_000001.npz').unlink()
    elif mutation == 'missing_holdout':
        (archive.run / 'native-noisy/holdout_samples.npz').unlink()
    else:
        name = 'native-noisy/snapshots/step_000250.npz'
        def mutate(values):
            if mutation == 'count': values['live'] = values['live'][:4095]
            elif mutation == 'nonfinite': values['live'][0, 0] = np.nan
            else: values['target'][0, 0] += 1
        if mutation == 'terminal_prefix':
            name = 'native-noisy/snapshots/step_006000.npz'
            mutate = lambda values: values['live'].__setitem__((0, 0), 999)
        change_arrays(archive, name, mutate)
    with pytest.raises((ValueError, FileNotFoundError)):
        media.export(archive.run, archive.output)
    assert not archive.output.exists()


@pytest.mark.parametrize('mutation', ['law', 'stream', 'source', 'event_cadence', 'metric_endpoint', 'accuracy_partial'])
def test_source_law_and_original_records_are_bound_after_repin(archive, mutation):
    if mutation == 'law': change_json(archive, 'result.json', lambda x: x.update(eval_output_noise=False))
    elif mutation == 'stream': change_json(archive, 'result.json', lambda x: x.update(stream_deviations=1))
    elif mutation == 'source':
        (archive.snapshot / media.core.CONFIG).write_text('{"lr": 999}\n')
    elif mutation == 'event_cadence':
        name = 'native-noisy/events.jsonl'; p = archive.run / name
        p.write_text('\n'.join(p.read_text().splitlines()[2:]) + '\n'); repin(archive, name)
    elif mutation == 'metric_endpoint':
        name = 'metrics.jsonl'; p = archive.run / name
        rows = [json.loads(line) for line in p.read_text().splitlines()]; rows[-1]['precision'] = .5
        p.write_text(''.join(json.dumps(row) + '\n' for row in rows)); repin(archive, name)
    else: change_json(archive, 'native-noisy/verdict.json', lambda x: x['accuracy']['terminal_checks'].pop())
    with pytest.raises(ValueError):
        media.load_run(archive.run)


def test_valid_coverage_fail_accuracy_pass_preserves_joint_fail(archive):
    def fail_coverage(value):
        value['coverage'].update(status='FAIL', passed=False, stable_checks=0)
        value['accuracy']['coverage_status'] = 'FAIL'
    change_json(archive, 'native-noisy/verdict.json', fail_coverage)
    archive.row.update(status='FAIL', original_gate='FAIL')
    archive.row['native_gates']['noisy']['coverage'] = 'FAIL'
    media.write(archive.study_path, archive.study)
    data = media.load_run(archive.run)
    assert data['original_gate'] == 'FAIL'
    assert data['row']['reported_original_status'] == 'PASS'
    assert data['verdicts']['noisy']['accuracy']['status'] == 'PASS'


def test_original_early_none_stays_visible_with_new_media_only(archive, monkeypatch):
    oracle(monkeypatch)
    before = {str(p): p.read_bytes() for base in (archive.run.parents[1], archive.snapshot) for p in base.rglob('*') if p.is_file()}
    receipt = media.export(archive.run, archive.output)
    after = {str(p): p.read_bytes() for base in (archive.run.parents[1], archive.snapshot) for p in base.rglob('*') if p.is_file()}
    assert before == after
    assert receipt['original_gate'] == 'PASS' and receipt['original_gates']['clean']['accuracy'] == 'FAIL'
    assert receipt['displayed_observations'][0]['recorded_20k_metrics']['acc_center_rms_sigma'] is None
    assert receipt['training_updates'] == receipt['model_forwards'] == receipt['new_draws'] == 0
    assert receipt['rescoring'] is receipt['qualification_input'] is False
    assert len(receipt['original_observation_steps']) == 34 and len(receipt['displayed_steps']) == 9
    assert receipt['original_gate_samples']['independent_holdout'] == 100000
    assert receipt['original_gate_samples']['final_five_steps'] == [6000, 6250, 6500, 6750, 7000]
    assert receipt['gif']['frames'] == 9 and receipt['original_metric_gif_unchanged']
    assert media.read(archive.output / 'native-media-receipt.json') == receipt
    with Image.open(archive.output / 'native-goal.gif') as image: assert image.n_frames == 9


def test_missing_original_metric_media_remains_explicitly_unavailable(archive):
    archive.row['media'] = None
    media.write(archive.study_path, archive.study)
    data = media.load_run(archive.run)
    assert not data['original_metric_gif_available']
    assert data['original_gate'] == 'PASS'


def test_unknown_row_rejects_without_publication(archive):
    archive.row['status'] = 'UNKNOWN'; media.write(archive.study_path, archive.study)
    with pytest.raises(ValueError, match='complete original'):
        media.export(archive.run, archive.output)
    assert not archive.output.exists()


def test_camera_keeps_all_samples_but_zoom_depends_only_on_target(archive):
    data = media.load_run(archive.run)
    whole, zoom = media.camera(data['snapshots'])
    data['snapshots'][7000]['live'][0] = [-100, 100]
    changed, zoom2 = media.camera(data['snapshots'])
    assert zoom == zoom2 and whole != changed
    assert changed[0] < -100 and changed[3] > 100
    assert np.isclose(whole[1] - whole[0], whole[3] - whole[2])


def test_literal_threshold_reader_never_executes_source(tmp_path):
    path = tmp_path / 'bounds.py'
    path.write_text(f'raise RuntimeError("must never execute")\nNATIVE_COVERAGE_THRESHOLDS = {COVERAGE!r}\nNATIVE_ACCURACY_THRESHOLDS = {ACCURACY!r}\n')
    assert media.source_thresholds(path) == [list(v) for v in COVERAGE + ACCURACY]


def test_renderer_binds_both_current_files_to_committed_bytes(tmp_path, monkeypatch):
    for relative in (media.SELF, media.HELPER):
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(relative + '\n')
    monkeypatch.setattr(media, 'ROOT', tmp_path)
    def git(command, **kwargs):
        if command[1] == 'rev-parse': return 'f' * 40 + '\n'
        return (command[-1].split(':', 1)[1] + '\n').encode()
    monkeypatch.setattr(media.subprocess, 'check_output', git)
    assert set(media.renderer_identity()['files']) == {media.SELF, media.HELPER}
    (tmp_path / media.HELPER).write_text('# uncommitted helper change\n')
    with pytest.raises(ValueError, match='commit the exact'):
        media.renderer_identity()


def test_cli_help_does_not_import_models_or_require_cuda(tmp_path):
    result = subprocess.run([sys.executable, '-B', str(MODULE), '--help'], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': '', 'CUDA_VISIBLE_DEVICES': ''},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert '--run' in result.stdout and '--output' in result.stdout
