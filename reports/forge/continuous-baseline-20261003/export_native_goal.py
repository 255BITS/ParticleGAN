"""Posthoc native100 goal GIFs from completed original Atlas19 saved arrays.

  python reports/forge/continuous-baseline-20261003/export_native_goal.py \
    --run /path/baseline-atlas19/native/grid100 --output /new/media/grid100

No model, host sampler or scorer is imported. All 34 original noisy snapshots
are verified; nine unbiased actual states are displayed against the fixed saved
target cloud. Original coverage and accuracy verdicts, final-five 20k evidence
and independent 100k holdout stay separate from this media-only operation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SELF = 'reports/forge/continuous-baseline-20261003/export_native_goal.py'
HELPER = 'reports/forge/continuous-baseline-20261003/export_moving_goal.py'
_spec = importlib.util.spec_from_file_location('_native_media_utilities', ROOT / HELPER)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
sha, stable, read, identity, write = core.sha, core.stable, core.read, core.identity, core.write
SCREEN_SHA = 'c60d76627f1846e807ae4fa36075a52516ddf568b70d616ad232b4d628118da2'
STEPS = [0, 1, 10, 25, 50, 100] + list(range(250, 7001, 250))
TERMINAL_STEPS = STEPS[-5:]
DISPLAY_INDICES = np.linspace(0, len(STEPS) - 1, 9, dtype=int).tolist()
DISPLAY_STEPS = [STEPS[i] for i in DISPLAY_INDICES]
HOST_MESSAGE = 'frozen native100 (constraints_simple_regularization.json resources, canonical CUDA fixture); learner = candidate public GANTrainer, see job-header.json'
METRICS = ('modes', 'precision', 'mass_tv', 'acc_center_rms_sigma', 'acc_abs_cov_trace_bias', 'acc_radial_ks')


def renderer_identity():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    files = {}
    for relative in (SELF, HELPER):
        blob = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != sha(ROOT / relative):
            raise ValueError('commit the exact native exporter and media helper before rendering')
        files[relative] = identity(ROOT / relative)
    return {'commit': commit, 'files': files, 'python': platform.python_version()}


def source_thresholds(path):
    """Read literal original bounds without importing mathematical modules."""
    names = ('NATIVE_COVERAGE_THRESHOLDS', 'NATIVE_ACCURACY_THRESHOLDS')
    found = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = [list(v) for v in ast.literal_eval(node.value)]
    if set(found) != set(names):
        raise ValueError('original native literal thresholds missing')
    return found[names[0]] + found[names[1]]


def load_run(run):
    run = Path(run).resolve()
    request = read(run / 'request.json'); packet, requested = request['packet'], request['row']
    if requested['group'] != 'native' or requested['task'] not in core.CASES:
        raise ValueError('only the three original static native cases are supported')
    case_id = f"atlas-original19-native-{requested['task']}"
    if requested['id'] != case_id or Path(request['target']).resolve() != run:
        raise ValueError('original native case/location differs')
    study = read(run.parents[1] / 'study.json')
    row = next(r for r in study['rows'] if r['id'] == case_id)
    definition = packet['case_definitions'][case_id]
    if (definition != study['case_definitions'][case_id] or stable(definition) != row['case_sha256']
            or stable(definition) != requested['case_sha256']
            or (definition['id'], definition['group'], definition['task']) != (case_id, 'native', requested['task'])
            or (row['group'], row['task']) != ('native', requested['task'])):
        raise ValueError('source-bound original native declaration differs')
    if definition['original_host'] != dict(steps=7000, seed=1234, num_particles=20000, z_dim=2,
                                          batch_size=2048, evaluation_samples=20000,
                                          terminal_steps=TERMINAL_STEPS, holdout_samples=100000,
                                          laws=['noisy primary', 'clean diagnostic']):
        raise ValueError('original native resource/sampling contract differs')
    if (definition['observation_steps'] != STEPS or definition['original_options'] != core.OPTIONS
            or definition['config_sha256'] != core.CONFIG_SHA or definition['reference_sha256'] != core.REFERENCE_SHA
            or definition['original_terminal_gate'] is not True or definition['added_policy_hold_gate'] is not False
            or packet['family'] != 'atlas'
            or packet['recipe_overrides'] != {'original_config_sha256': core.CONFIG_SHA, 'original_options': core.OPTIONS}):
        raise ValueError('original native recipe/options/cadence/gate scope differs')
    if (row.get('status') not in {'PASS', 'FAIL'} or row.get('full_protocol_complete') is not True
            or row.get('completed_steps') != 7000 or row.get('metric_observations') != 34
            or row.get('child_returncode') != 0):
        raise ValueError('complete original 7000-update native evidence required')
    if study['source'] != packet['source'] or study['execution_source'] != packet['execution_source']:
        raise ValueError('original native execution source binding differs')
    source = packet['execution_source']; snapshot = core.verify_source(source)
    if (packet['source']['execution_digest'] != source['digest']
            or packet['source']['commit'] != source['origin_commit']
            or Path(packet['snapshot_locations']['package']).resolve() != snapshot
            or source['files'].get(core.CONFIG) != core.CONFIG_SHA
            or source['files'].get(core.BASELINE) != core.BASELINE_SHA
            or any(source['files'].get(name) != expected for name, expected in packet['source']['files_sha256'].items())):
        raise ValueError('executed Atlas native package/config/driver manifest differs')
    screen = Path(packet['snapshot_locations']['adapter']) / 'screen_current.py'
    if not screen.resolve().is_relative_to(snapshot) or sha(screen) != SCREEN_SHA:
        raise ValueError('original native observation/scoring adapter changed')
    thresholds = source_thresholds(screen)
    if definition['original_requirements'] != thresholds:
        raise ValueError('native bounds differ from original source literals')
    artifacts = {}
    def consume(name):
        artifacts[name] = core._artifact(run, name, row['artifacts'][name])
        return run / name
    for name in ('request.json', 'final-state.pt', 'adapter-relocation.json', 'native-score-wrapper.py'):
        consume(name)
    result_path = consume('result.json')
    if Path(row['result_path']).resolve() != result_path or row['result_sha256'] != sha(result_path):
        raise ValueError('original native result path/hash differs')
    result = read(result_path)
    if (result['task'] != requested['task'] or result['completed_steps'] != 7000
            or result['stream_deviations'] != 0 or result['header']['options'] != core.OPTIONS
            or result['eval_output_noise'] is not True):
        raise ValueError('original native full budget/stream/primary law differs')
    header = result['header']
    if (header['package_root'] != str(snapshot) or header['device'] != 'cuda:0'
            or header['cuda_visible_devices'] != '1'
            or header['overrides'] != {**read(snapshot / core.CONFIG), 'initialization': 'batch_feature_zero'}):
        raise ValueError('original native executed config/runtime location differs')
    observations = [json.loads(line) for line in consume('metrics.jsonl').read_text().splitlines() if line.strip()]
    if [r['step'] for r in observations] != STEPS:
        raise ValueError('original complete native metric cadence differs')
    for observation in observations:
        if type(observation.get('pass')) is not bool:
            raise ValueError('original instantaneous coverage flag missing')
        for key in METRICS:
            value = observation.get(key)
            if value is None and key.startswith('acc_'):
                continue  # Original early fidelity statistics can be unavailable.
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f'invalid recorded native metric: {key}')
    fixture_path = Path(packet['snapshot_locations']['harness']) / 'tasks/native100_fixture.json'
    config_path = fixture_path.with_name('native100_constraints_simple_regularization.json')
    fixture, frozen_config = read(fixture_path), read(config_path)
    host_config = {key: frozen_config[key] for key in fixture['host_fields']}
    expected_config = {**host_config, 'problem': requested['task'], 'device': 'cuda:0', 'host': HOST_MESSAGE}
    expected_accuracy = dict(protocol='toy100-accuracy-v1', check_steps=TERMINAL_STEPS,
                             sample_count=20000, holdout_samples=100000,
                             holdout_seed_offsets={'target': 1601, 'noise': 1602, 'latent': 1603})
    verdicts = {}; snapshots = {}
    def array_file(name, count):
        with np.load(consume(name), allow_pickle=False) as data:
            if set(data.files) != {'live', 'ema', 'target'}:
                raise ValueError('unknown native saved-array schema')
            arrays = {key: data[key].copy() for key in data.files}
        if any(a.shape != (count, 2) or a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('native saved-array count/type/finiteness differs')
        return arrays
    for law in ('noisy', 'clean'):
        base = f'native-{law}'
        summary = read(consume(f'{base}/summary.json')); verdict = read(consume(f'{base}/verdict.json'))
        if (read(consume(f'{base}/config.json')) != expected_config or summary['config'] != expected_config
                or summary['evaluation'] != law or summary['problem'] != requested['task']
                or summary['status'] != 'complete' or summary['budget_steps'] != 7000 or summary['completed_steps'] != 7000
                or summary['eval_steps'] != STEPS or summary['snapshot_steps'] != STEPS
                or summary['accuracy'] != expected_accuracy or summary['accuracy_check_steps'] != TERMINAL_STEPS):
            raise ValueError('original native law/resources/accuracy protocol differs')
        if verdict['sources'] != fixture['host_source_sha256']:
            raise ValueError('original native scorer source manifest differs')
        native_root = Path(packet['snapshot_locations']['native_root'])
        if not native_root.resolve().is_relative_to(snapshot):
            raise ValueError('native scorer package outside immutable source snapshot')
        if any(sha(native_root / name) != expected for name, expected in verdict['sources'].items()):
            raise ValueError('original native scorer bytes differ')
        cov, acc = verdict['coverage'], verdict['accuracy']
        if (cov['status'] not in {'PASS', 'FAIL'} or acc['status'] not in {'PASS', 'FAIL'}
                or cov['problem'] != requested['task'] or acc['problem'] != requested['task']
                or cov['budget_steps'] != 7000 or cov['final_step'] != 7000 or cov['observations'] != 34
                or cov['required_stable_checks'] != 5 or cov['audited_final_samples'] is not True
                or type(cov.get('stable_checks')) is not int or not 0 <= cov['stable_checks'] <= 33
                or (cov['status'] == 'PASS' and cov['stable_checks'] < 5)
                or cov.get('passed') is not (cov['status'] == 'PASS')
                or acc.get('passed') is not (acc['status'] == 'PASS')
                or [c['step'] for c in acc['terminal_checks']] != TERMINAL_STEPS
                or acc['coverage_status'] != cov['status']
                or cov['final_metrics'] != summary['final']['live']
                or acc['final_metrics'] != acc['terminal_checks'][-1]['metrics']
                or acc['holdout_metrics'] != summary['holdout']['live']):
            raise ValueError('original complete native verdict/evidence differs')
        checks = [c['passed'] for c in acc['terminal_checks']]
        if any(type(c) is not bool for c in checks):
            raise ValueError('original accuracy terminal flags invalid')
        accuracy_pass = all(checks) and acc['holdout_metrics']['frozen_pass'] and acc['holdout_metrics']['accuracy_pass']
        if acc['status'] != ('PASS' if accuracy_pass else 'FAIL'):
            raise ValueError('recorded native accuracy verdict inconsistent')
        quality = {step: array_file(f'{base}/quality_checks/step_{step:06d}.npz', 20000) for step in TERMINAL_STEPS}
        final = array_file(f'{base}/final_samples.npz', 20000)
        array_file(f'{base}/holdout_samples.npz', 100000)
        if any(not np.array_equal(final[key], quality[7000][key]) for key in final):
            raise ValueError('native final/terminal saved draws disagree')
        events = [json.loads(line) for line in consume(f'{base}/events.jsonl').read_text().splitlines() if line.strip()]
        if [(event.get('step'), event.get('model')) for event in events if event.get('event') == 'eval'] != [
                (step, model) for step in STEPS for model in ('live', 'ema')]:
            raise ValueError('original native live/EMA event cadence differs')
        if any(event.get('event') != 'eval' for event in events):
            raise ValueError('unknown native scientific event schema')
        if any(events[-2 + index]['metrics'] != summary['final'][model] for index, model in enumerate(('live', 'ema'))):
            raise ValueError('native recorded endpoint metrics differ')
        if law == 'noisy':
            for observation, event in zip(observations, events[::2]):
                for key in METRICS:
                    value = event['accuracy'].get(key[4:]) if key.startswith('acc_') else event['metrics'].get(key)
                    if value != observation.get(key):
                        raise ValueError('native displayed metrics differ from original noisy live event')
            snapshots = {step: array_file(f'{base}/snapshots/step_{step:06d}.npz', 4096) for step in STEPS}
            target = snapshots[0]['target']
            if any(not np.array_equal(data['target'], target) for data in snapshots.values()):
                raise ValueError('original fixed target cloud changed across snapshots')
            if any(not np.array_equal(snapshots[step][key], quality[step][key][:4096]) for step in TERMINAL_STEPS for key in final):
                raise ValueError('native displayed prefix does not match the original 20k terminal draws')
        verdicts[law] = verdict
    primary = verdicts['noisy']; joint = 'PASS' if all(primary[k]['status'] == 'PASS' for k in ('coverage', 'accuracy')) else 'FAIL'
    expected_gates = {law: {key: verdicts[law][key]['status'] for key in ('coverage', 'accuracy')} for law in verdicts}
    if (row['native_gates'] != expected_gates or row['original_gate'] != joint or row['status'] != joint
            or result['status'] != primary['accuracy']['status']
            or row['reported_original_status'] != result['status']):
        raise ValueError('original noisy joint coverage/accuracy gate differs')
    original_media = row.get('media')
    if original_media:
        if Path(original_media['path']).resolve() != run / 'goal-metrics.gif':
            raise ValueError('original metric GIF path differs')
        artifacts['goal-metrics.gif'] = core._artifact(run, 'goal-metrics.gif', original_media)
        artifacts['media-receipt.json'] = identity(run / 'media-receipt.json')
        if read(run / 'media-receipt.json') != original_media:
            raise ValueError('original metric GIF receipt differs')
    return dict(run=run, row=row, definition=definition, source=packet['source'], execution_source=source,
                producer=identity(screen), observations=observations, snapshots=snapshots, verdicts=verdicts,
                artifacts=artifacts, original_gate=joint, original_metric_gif_available=bool(original_media), thresholds=thresholds)


def camera(snapshots):
    reference = snapshots[0]['target']
    arrays = [reference] + [s['live'] for s in snapshots.values()]
    lo = np.stack([a.min(axis=0) for a in arrays]).min(axis=0).astype(float) - .15
    hi = np.stack([a.max(axis=0) for a in arrays]).max(axis=0).astype(float) + .15
    extent = float(max(hi - lo)); midpoint = (lo + hi) / 2
    whole = [float(midpoint[0] - extent / 2), float(midpoint[0] + extent / 2),
             float(midpoint[1] - extent / 2), float(midpoint[1] + extent / 2)]
    zoom = float(np.abs(reference).max()) * .2 + .15
    return whole, [-zoom, zoom, -zoom, zoom]


def format_metric(value):
    return 'unavailable' if value is None else f'{value:.4f}'


def export(run, output):
    bundle = load_run(run); renderer = renderer_identity()
    output = Path(output).resolve(); run = bundle['run']
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError('supplemental native media must be outside original archive')
    if output.exists() and any(output.iterdir()):
        raise ValueError('new empty native media directory required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, __version__ as pillow_version
    whole, zoom = camera(bundle['snapshots']); panels = []; displayed = []
    by_step = {r['step']: r for r in bundle['observations']}
    original = bundle['verdicts']['noisy']; clean = bundle['verdicts']['clean']
    target = bundle['snapshots'][0]['target']
    for step in DISPLAY_STEPS:
        observed = by_step[step]; samples = bundle['snapshots'][step]['live']
        fig, axes = plt.subplots(1, 2, figsize=(12, 7.1))
        for ax, bounds, title in zip(axes, (whole, zoom), ('All 100 target modes', 'Fixed central detail')):
            ax.scatter(target[:, 0], target[:, 1], s=4, alpha=.35, color='#c66a27', marker='+', linewidths=.5,
                       label='Retained target cloud')
            ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=.4, color='#296ba3', linewidths=0,
                       label='Retained noisy live samples')
            ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='x', ylabel='y', title=title)
            ax.set_aspect('equal', adjustable='box')
        axes[0].legend(loc='lower left', fontsize=8)
        fig.suptitle(f'{bundle["row"]["task"]}: recover all modes and Gaussian mass/shape\nActual update {step} / 7000; original noisy primary', fontsize=13)
        title = (f'Full original protocol: {bundle["original_gate"]} (retrospective); '
                 f'coverage {original["coverage"]["status"]}; accuracy {original["accuracy"]["status"]}.')
        coverage = 'PASS' if observed['pass'] else 'FAIL'
        fidelity = 'unavailable' if any(observed[k] is None for k in METRICS if k.startswith('acc_')) else ('PASS' if observed.get('acc_passed') else 'FAIL')
        line = (f'This 20k observation: coverage {coverage}; fidelity {fidelity}. '
                f'Modes {observed["modes"]} (require 100); precision {observed["precision"]:.4f} (≥ .97); mass TV {observed["mass_tv"]:.4f} (≤ .10).')
        shape = (f'Center RMS/σ {format_metric(observed["acc_center_rms_sigma"])} (≤ .20); '
                 f'|covariance bias| {format_metric(observed["acc_abs_cov_trace_bias"])} (≤ .10); '
                 f'radial KS {format_metric(observed["acc_radial_ks"])} (≤ .04).')
        fig.text(.055, .158, title + '\n' + line + '\n' + shape, fontsize=9)
        foot = (f'Plot: 4096 saved float32 points per cloud; no new draws. Full gate: all 9 coverage bounds and ≥ 5 terminal checks,\n'
                f'plus final five 20k fidelity clouds and independent 100k holdout. Clean diagnostic: coverage {clean["coverage"]["status"]}, '
                f'accuracy {clean["accuracy"]["status"]}; EMA diagnostic stays separate.\n'
                'Nine selected saved states from 34 actual observations; no interpolation/rescoring or ordinary Forge MoG/default credit.')
        fig.text(.055, .029, foot, fontsize=8)
        fig.subplots_adjust(left=.06, right=.98, top=.82, bottom=.30, wspace=.2)
        fig.canvas.draw(); panels.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')); plt.close(fig)
        displayed.append({'step': step, 'recorded_20k_metrics': {key: observed[key] for key in METRICS},
                          'recorded_instant_coverage': observed['pass'], 'recorded_instant_fidelity': observed.get('acc_passed')})
    for artifact in bundle['artifacts'].values():
        if identity(artifact['path']) != artifact:
            raise ValueError('original native input changed during rendering')
    output.mkdir(parents=True, exist_ok=True); destination = output / 'native-goal.gif'
    panels[0].save(destination, save_all=True, append_images=panels[1:], duration=900, loop=0)
    with Image.open(destination) as image:
        count = image.n_frames
    if count != 9:
        raise ValueError('native GIF did not retain nine distinct actual frames')
    for artifact in bundle['artifacts'].values():
        if identity(artifact['path']) != artifact:
            raise ValueError('original native input changed during media write')
    core.verify_source(bundle['execution_source'])
    receipt = dict(schema='original_atlas_native_goal_media_v1', case_id=bundle['row']['id'],
                   original_gate=bundle['original_gate'], original_gates=bundle['row']['native_gates'],
                   reported_original_status=bundle['row']['reported_original_status'], full_budget_complete=True,
                   posthoc_media_only=True, training_updates=0, model_forwards=0, new_draws=0,
                   rescoring=False, interpolated_frames=0, qualification_input=False, default_adoption=False,
                   source=bundle['source'], execution_source_digest=bundle['execution_source']['digest'],
                   complete_execution_snapshot_verified=True, original_producer=bundle['producer'],
                   renderer_source=renderer, raw_inputs=bundle['artifacts'], raw_inputs_unchanged=True,
                   original_metric_gif_available=bundle['original_metric_gif_available'],
                   original_metric_gif_unchanged=True if bundle['original_metric_gif_available'] else None,
                   source_bound_case_sha256=stable(bundle['definition']), immutable_row_sha256=stable(bundle['row']),
                   original_observation_steps=STEPS, displayed_steps=DISPLAY_STEPS,
                   selection='Nine inclusive equally spaced indices of the 34 retained observations; no interpolation.',
                   displayed_observations=displayed, original_thresholds=bundle['thresholds'],
                   primary_law='noisy live, original trainer.G/prior references; policy may apply served parameters',
                   clean_and_ema='Diagnostic evidence retained separately; never substituted for noisy primary.',
                   display_samples=dict(count_per_cloud=4096, stored_dtype='float32', target_seed=1635, latent_seed=1637, noise_seed=1636),
                   original_gate_samples=dict(each_observation=20000, final_five_steps=TERMINAL_STEPS,
                                              independent_holdout=100000, holdout_seeds=dict(target=2835, noise=2836, latent=2837)),
                   camera=dict(full_fixed_axes=whole, central_fixed_axes=zoom,
                               selection='Full camera contains all saved samples/reference; central half-width is 20% of saved reference extent plus .15, fixed across time.'),
                   gif={**identity(destination), 'frames': count},
                   render_runtime=dict(numpy=np.__version__, matplotlib=matplotlib.__version__, pillow=pillow_version))
    write(output / 'native-media-receipt.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv); receipt = export(args.run, args.output)
    print(json.dumps({'case': receipt['case_id'], 'original_gate': receipt['original_gate'], 'gif': receipt['gif']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
