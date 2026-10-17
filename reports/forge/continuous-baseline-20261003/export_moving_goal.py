"""Illustrate completed original Atlas rotations using retained arrays only.

Run after the original moving row is complete and this exporter is committed:
  python reports/forge/continuous-baseline-20261003/export_moving_goal.py \
    --run /path/baseline-atlas19/moving/grid100 --output /new/media/grid100

No trainer, host sampler or scorer is imported. Four original float16 sample
clouds and stored analytical target centers are displayed; the independent 20k
period metrics remain their original recorded observations. Raw evidence and the
existing metric GIF stay untouched. This creates a separate media receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SELF = 'reports/forge/continuous-baseline-20261003/export_moving_goal.py'
PRODUCER_SHA = '8057a8d4298d251edabcaeaaf02ef5b83834bdb56c296d2681c70aec6f277ed1'
BASELINE = 'reports/forge/continuous-baseline-20261003/run_atlas_baseline.py'
BASELINE_SHA = '8dd658294dbbb2027e69a0e46097e2d2c0f5382078d407ade25e06c6c22ff1c1'
CONFIG = 'configs/100gaussians/atlas.json'
CONFIG_SHA = 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'
REFERENCE_SHA = '1e1b536bfd66e20a240436b417d429059d22ea55c2b6ecfff31e796b96c470fa'
OPTIONS = dict(eval_output_noise=True, save_final_state=True, strict_streams=True,
               diagnostics=True, evaluation_generate='indexed', serial_backward_argument=True,
               initialization='batch_feature_zero', image_prior_perturb=False, ring_frozen_control=False)
STEPS = [0, 500, 1000, 1500]
ANGLES = [0., 0., math.pi / 6, math.pi / 3]
CASES = ('grid100', 'rotated100', 'staggered100')
RULE = 'after each turn: modes >= 95 and hq >= .9 x pre-turn hq'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def renderer_identity():
    """Every published renderer identity must name committed exact file bytes."""
    path = ROOT / SELF
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    data = subprocess.check_output(['git', 'show', f'{commit}:{SELF}'], cwd=ROOT)
    if hashlib.sha256(data).hexdigest() != sha(path):
        raise ValueError('commit the exact exporter before rendering final media')
    return {**identity(path), 'commit': commit, 'git_path': SELF, 'python': platform.python_version()}


def _artifact(run, name, declaration):
    actual = identity(run / name)
    if (actual['sha256'], actual['bytes']) != (declaration['sha256'], declaration['bytes']):
        raise ValueError(f'retained artifact changed: {name}')
    return actual


def verify_source(source):
    """Verify the complete declared snapshot using bytes, without imports."""
    snapshot = Path(source['snapshot_path']).resolve()
    if not source['files'] or stable(source['files']) != source['digest']:
        raise ValueError('execution source digest differs')
    for relative, expected in source['files'].items():
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot) or sha(path) != expected:
            raise ValueError(f'original execution source changed: {relative}')
    return snapshot


def load_run(run):
    """Fail before export for partial, unknown, inconsistent or unbound evidence."""
    run = Path(run).resolve()
    request_path = run / 'request.json'
    request = read(request_path)
    packet, requested_row = request['packet'], request['row']
    case_id = requested_row['id']
    if requested_row['group'] != 'moving' or requested_row['task'] not in CASES:
        raise ValueError('only the three original moving cases are supported')
    if case_id != f"atlas-original19-moving-{requested_row['task']}":
        raise ValueError('original moving case identity differs')
    study = read(run.parents[1] / 'study.json')
    row = next(r for r in study['rows'] if r['id'] == case_id)
    definition = packet['case_definitions'][case_id]
    if (study['case_definitions'][case_id] != definition or row['case_sha256'] != stable(definition)
            or requested_row['case_sha256'] != stable(definition)
            or row['group'] != 'moving' or row['task'] != requested_row['task']
            or definition['id'] != case_id or definition['group'] != 'moving'
            or definition['task'] != requested_row['task']):
        raise ValueError('source-bound original case declaration differs')
    host = definition['original_host']
    if host != {
            'steps': 1500, 'seed': 1234, 'num_particles': 20000, 'z_dim': 2,
            'batch_size': 2048, 'period_steps': 500, 'rotate_degrees': 30,
            'turns': 2, 'captured_points': 4096, 'evaluation_samples': 20000}:
        raise ValueError('original moving resource/rotation/sampling contract differs')
    if definition['observation_steps'] != STEPS[1:] or definition['original_requirements'] != [
            ['modes', '>=', 95], ['hq', '>=', '0.9 * observed pre_turn_hq']]:
        raise ValueError('original moving metric contract differs')
    if (packet['family'] != 'atlas' or definition['config_sha256'] != CONFIG_SHA
            or definition['reference_sha256'] != REFERENCE_SHA
            or definition['original_options'] != OPTIONS
            or packet['recipe_overrides'] != {'original_config_sha256': CONFIG_SHA, 'original_options': OPTIONS}
            or definition['original_terminal_gate'] is not True
            or definition['added_policy_hold_gate'] is not False):
        raise ValueError('original Atlas config/options/gate scope differs')
    if (row.get('status') not in {'PASS', 'FAIL'} or row.get('full_protocol_complete') is not True
            or row.get('completed_steps') != 1500 or row.get('metric_observations') != 3
            or row.get('child_returncode') != 0):
        raise ValueError('complete original moving evidence is required; no convergence inferred')
    if study['source'] != packet['source'] or study['execution_source'] != packet['execution_source']:
        raise ValueError('original execution source binding differs')
    source = packet['execution_source']
    snapshot = verify_source(source)
    if (packet['source']['execution_digest'] != source['digest']
            or packet['source']['commit'] != source['origin_commit']
            or Path(packet['snapshot_locations']['package']).resolve() != snapshot
            or source['files'].get(CONFIG) != CONFIG_SHA
            or source['files'].get(BASELINE) != BASELINE_SHA
            or any(source['files'].get(name) != expected for name, expected in packet['source']['files_sha256'].items())):
        raise ValueError('executed Atlas package/config/driver manifest differs')
    rotate = Path(packet['snapshot_locations']['rotate']).resolve()
    if not rotate.is_relative_to(snapshot):
        raise ValueError('original producer is outside the immutable snapshot')
    relative = str(rotate.relative_to(snapshot))
    if source['files'].get(relative) != PRODUCER_SHA or sha(rotate) != PRODUCER_SHA:
        raise ValueError('original retained-array producer changed')
    # Bind the actual executed location-only/indexed-generation/checkpoint
    # wrapper as a raw artifact; do not import it or reconstruct a model.
    names = ['request.json', 'runner.py', 'frames.npz', 'frames.npz.verdict.json'] + [
        f'frames.npz.checkpoint-{step:06d}.pt' for step in STEPS[1:]]
    artifacts = {name: _artifact(run, name, row['artifacts'][name]) for name in names}
    result_path = run / 'frames.npz.verdict.json'
    if Path(row['result_path']).resolve() != result_path or row['result_sha256'] != sha(result_path):
        raise ValueError('original result path/hash differs')
    result = read(result_path)
    periods = result['periods']
    if (result['task'] != requested_row['task'] or result['rule'] != RULE or result['turns'] != 2
            or [r['period_end'] for r in periods] != STEPS[1:]
            or [r['target_deg'] for r in periods] != [0, 30, 60]
            or any(r.get('task') != requested_row['task'] for r in periods)):
        raise ValueError('original period timing/gate definition differs')
    for r in periods:
        if type(r['modes']) is not int or not 0 <= r['modes'] <= 100:
            raise ValueError('invalid recorded mode count')
        if type(r['hq']) not in (int, float) or not math.isfinite(r['hq']) or not 0 <= r['hq'] <= 1:
            raise ValueError('invalid recorded HQ fraction')
    if result['pre_turn_hq'] != periods[0]['hq']:
        raise ValueError('pre-turn HQ does not match the original observation')
    recorded_checks = [p['modes'] >= 95 and p['hq'] >= .9 * result['pre_turn_hq'] for p in periods[1:]]
    expected_status = 'PASS' if all(recorded_checks) else 'FAIL'
    if (result['status'] != expected_status or row['status'] != expected_status
            or row['original_gate'] != expected_status or result['passed_periods'] != sum(recorded_checks)):
        raise ValueError('recorded original gate/row is internally inconsistent')
    with np.load(run / 'frames.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != {'frames', 'steps', 'centers', 'angles', 'final'}:
            raise ValueError('unknown saved moving-array schema')
        frames, steps, centers, angles = [arrays[k].copy() for k in ('frames', 'steps', 'centers', 'angles')]
        if (frames.shape != (4, 4096, 2) or frames.dtype != np.float16
                or centers.shape != (100, 2) or centers.dtype != np.float32
                or steps.dtype.kind not in 'iu' or steps.tolist() != STEPS
                or angles.shape != (4,) or not np.array_equal(angles, np.asarray(ANGLES))
                or not np.isfinite(frames).all() or not np.isfinite(centers).all()):
            raise ValueError('saved full sample shape/cadence/rotation/finiteness differs')
        if arrays['final'].shape != () or arrays['final'].dtype.kind not in 'US':
            raise ValueError('saved terminal metric receipt differs')
        raw_final = arrays['final'].item()
        final = json.loads(raw_final.decode('utf8') if isinstance(raw_final, bytes) else raw_final)
    if (type(final.get('modes')) is not int or final['modes'] != periods[-1]['modes']
            or type(final.get('hq')) not in (int, float) or not math.isfinite(final['hq']) or not 0 <= final['hq'] <= 1
            or round(final['hq'], 4) != periods[-1]['hq']):
        raise ValueError('terminal full-precision metric disagrees with recorded 20k gate')
    media = row.get('media')
    if not media or Path(media['path']).resolve() != run / 'goal-metrics.gif':
        raise ValueError('immutable original metric GIF required')
    artifacts['goal-metrics.gif'] = _artifact(run, 'goal-metrics.gif', media)
    artifacts['media-receipt.json'] = identity(run / 'media-receipt.json')
    if read(run / 'media-receipt.json') != media:
        raise ValueError('original metric-media receipt differs')
    if media.get('actual_steps') != STEPS[1:] or media.get('frames') != 3 or media.get('metric_only') is not True:
        raise ValueError('original metric GIF cadence/scope differs')
    return {'run': run, 'row': row, 'definition': definition, 'source': packet['source'],
            'execution_source': source,
            'execution_source_digest': source['digest'], 'producer': identity(rotate),
            'source_snapshot': str(snapshot), 'result': result, 'artifacts': artifacts,
            'frames': frames, 'steps': steps, 'centers': centers, 'angles': angles}


def target_centers(centers, angle):
    c, s = np.float32(math.cos(float(angle))), np.float32(math.sin(float(angle)))
    return centers @ np.asarray([[c, -s], [s, c]], dtype=np.float32).T


def camera(centers, angles, frames):
    targets = np.stack([target_centers(centers, a) for a in angles])
    lo = np.minimum(targets.min(axis=(0, 1)), frames.min(axis=(0, 1)).astype(float)) - .25
    hi = np.maximum(targets.max(axis=(0, 1)), frames.max(axis=(0, 1)).astype(float)) + .25
    extent = max(hi - lo); midpoint = (lo + hi) / 2
    whole = [float(midpoint[0] - extent / 2), float(midpoint[0] + extent / 2),
             float(midpoint[1] - extent / 2), float(midpoint[1] + extent / 2)]
    # Stable tie order is important; this selects the same retained target modes
    # across frames, using only reference geometry rather than apparent success.
    modes = np.argsort(np.square(centers.astype(float)).sum(axis=1), kind='stable')[:4]
    central = targets[:, modes]
    bound = float(np.abs(central).max()) + .25
    return targets, whole, [-bound, bound, -bound, bound], modes.tolist()


def export(run, output):
    bundle = load_run(run)
    renderer = renderer_identity()
    output = Path(output).resolve(); run = bundle['run']
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError('supplemental media must be outside the original run/archive parent')
    if output.exists() and any(output.iterdir()):
        raise ValueError('new empty output directory required; no media overwrite')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image, __version__ as pillow_version
    centers, frames, angles = bundle['centers'], bundle['frames'], bundle['angles']
    targets, full_axes, zoom_axes, zoom_modes = camera(centers, angles, frames)
    full_status = bundle['result']['status']; baseline = bundle['result']['pre_turn_hq']
    panels = []; displayed = []
    for index, step in enumerate(STEPS):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.7))
        for ax, extent, title in zip(axes, [full_axes, zoom_axes], ['All 100 target modes', 'Fixed central-mode detail']):
            ax.scatter(frames[index, :, 0], frames[index, :, 1], s=3, alpha=.35,
                       color='#296ba3', linewidths=0, label='Retained noisy samples')
            ax.scatter(targets[index, :, 0], targets[index, :, 1], s=14, marker='+',
                       color='#bd5827', linewidths=.8, label='Analytical target centers')
            for x, y in targets[index]:
                ax.add_patch(Circle((float(x), float(y)), .09, fill=False, color='#bd5827', linewidth=.5, alpha=.65))
            ax.set(xlim=extent[:2], ylim=extent[2:], xlabel='x', ylabel='y', title=title)
            ax.set_aspect('equal', adjustable='box')
        axes[0].legend(loc='lower left', fontsize=8)
        degrees = [0, 0, 30, 60][index]
        fig.suptitle(f'{bundle["row"]["task"]}: retain coverage after target rotation\nActual update {step} / 1500; target rotation {degrees}°', fontsize=13)
        if step == 0:
            phase = 'Initial state: no period gate yet.'
            metrics = None
        else:
            metrics = bundle['result']['periods'][index - 1]
            if step == 500:
                phase = f'Pre-turn baseline: modes {metrics["modes"]}; HQ {metrics["hq"]:.4f}. Turn to 30° begins update 501.'
            else:
                passed = metrics['modes'] >= 95 and metrics['hq'] >= .9 * baseline
                phase = (f'After turn: period gate {"PASS" if passed else "FAIL"}; modes {metrics["modes"]} ≥ 95; '
                         f'HQ {metrics["hq"]:.4f} ≥ {(.9 * baseline):.4f} (90% of pre-turn HQ).')
                if step == 1000:
                    phase += ' Turn to 60° begins update 1001.'
        fig.text(.055, .085, f'Full original moving test: {full_status} (retrospective verdict)\n{phase}', fontsize=9)
        fig.text(.055, .026, 'Plot: 4096 retained float16 samples; gate: separate original 20k draws. Orange circles show the original 0.09 quality radius.\nFour saved states only; no intermediate response, interpolation, new draws, rescoring, or ordinary native-accuracy qualification.', fontsize=8)
        fig.subplots_adjust(left=.06, right=.98, top=.82, bottom=.22, wspace=.20)
        fig.canvas.draw();panels.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'));plt.close(fig)
        displayed.append({'step': step, 'target_rotation_degrees': degrees,
                          'original_20k_period_observation': metrics})
    # Write only new media; verify every consumed immutable input before and
    # after rendering. No changing study.json digest is treated as a raw pin.
    for artifact in bundle['artifacts'].values():
        if identity(artifact['path']) != artifact:
            raise ValueError('original input changed during rendering')
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'moving-goal.gif'
    panels[0].save(destination, save_all=True, append_images=panels[1:], duration=1200, loop=0)
    with Image.open(destination) as image:
        frame_count = image.n_frames
    if frame_count != 4:
        raise ValueError('supplemental GIF did not retain four distinct actual frames')
    for artifact in bundle['artifacts'].values():
        if identity(artifact['path']) != artifact:
            raise ValueError('original input changed during media write')
    verify_source(bundle['execution_source'])
    receipt = {'schema': 'original_atlas_moving_goal_media_v1', 'case_id': bundle['row']['id'],
               'original_gate': full_status, 'full_budget_complete': True,
               'posthoc_media_only': True, 'training_updates': 0, 'model_forwards': 0,
               'new_draws': 0, 'rescoring': False, 'interpolated_frames': 0,
               'source': bundle['source'], 'execution_source_digest': bundle['execution_source_digest'],
               'complete_execution_snapshot_verified': True,
               'original_producer': bundle['producer'], 'renderer_source': renderer,
               'raw_inputs': bundle['artifacts'], 'immutable_row_sha256': stable(bundle['row']),
               'source_bound_case_sha256': stable(bundle['definition']),
               'raw_inputs_unchanged': True, 'original_metric_gif_unchanged': True,
               'actual_steps': STEPS, 'target_angles_radians': ANGLES,
               'rotation_transition_updates': [501, 1001], 'displayed_observations': displayed,
               'camera': {'full_fixed_axes': full_axes, 'zoom_fixed_axes': zoom_axes,
                          'zoom_reference_mode_indices': zoom_modes,
                          'selection': 'Four saved reference centers closest to origin, stable tie order; camera fixed over all retained rotations.'},
               'reference': 'Saved float32 centers rotated by recorded angle; analytical centers and 0.09 circles, not sampled target clouds.',
               'display_samples': {'count_per_frame': 4096, 'stored_dtype': 'float16', 'captured_latent_seed': 77, 'captured_noise_seed': 78},
               'gate_samples': 20000, 'gif': {**identity(destination), 'frames': frame_count},
               'render_runtime': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'pillow': pillow_version},
               'qualification_input': False}
    write(output / 'moving-media-receipt.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = export(args.run, args.output)
    print(json.dumps({'case': result['case_id'], 'original_gate': result['original_gate'],
                      'gif': result['gif'], 'receipt': str(args.output / 'moving-media-receipt.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
