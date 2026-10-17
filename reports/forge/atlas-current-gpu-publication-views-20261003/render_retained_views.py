"""Retained-array-only supplements; no checkpoints, scorers or new samples."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
SOURCE_COMMIT = '9563dea57bb150f2a0275bbe8d785bf76210fca3'
SOURCE_DIGEST = 'db5492df4aa5ef60ce9e7d5869b2b6be8d492f19191a2889967274555f8d0037'
TASKS = ('two_pole_policy_selected_cloud_v1', 'grid100_policy_selected_cloud_v1',
         'rotated100_policy_selected_cloud_v1', 'staggered100_policy_selected_cloud_v1')
ORIGINAL_DIR = Path('reports/forge/atlas-current-gpu-diagnostics-native-v2-20261003/gifs')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for payload in iter(lambda: stream.read(2**20), b''):
            h.update(payload)
    return h.hexdigest()


def pin(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('safe regular input required: ' + str(path))
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}


def records_hash(records):
    """Bind exact display projection, including array types/shapes/bytes."""
    h = hashlib.sha256()
    def visit(value):
        if isinstance(value, np.ndarray):
            h.update(b'array'); visit((str(value.dtype), value.shape)); h.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value): visit(key); visit(value[key])
        elif isinstance(value, (list, tuple)):
            h.update(type(value).__name__.encode())
            for child in value: visit(child)
        else:
            h.update(repr((type(value).__name__, value)).encode())
    visit(records)
    return h.hexdigest()


def mode0_reference(problem):
    """Exact declared geometry, not random samples or a fitted density oracle.

    problems.py defines mode0=(-4.5,-4.5), sigma=.03; rotated is25deg;
    staggered scales its first coordinate by.85 and subtracts.25 from its
    second coordinate in row0. Source bytes are separately pinned below.
    """
    center = np.array([-4.5, -4.5], dtype=np.float64)
    if problem == 'rotated100':
        theta = math.radians(25.)
        center = np.array([[math.cos(theta), -math.sin(theta)],
                           [math.sin(theta), math.cos(theta)]]) @ center
    elif problem == 'staggered100':
        center = np.array([center[0] * .85, center[1] - .25])
    elif problem != 'grid100':
        raise ValueError('unknown native geometry')
    angles = np.linspace(0., 2 * math.pi, 129, endpoint=True)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    return center, .03, np.concatenate((.03 * circle, .06 * circle))


def build_case(task_id, raw, media):
    """Project only the nine original media states; no numeric gate decisions."""
    evidence = raw['evidence']
    inputs = {Path(row['path']).name: Path(row['path']) for row in media['inputs']}
    steps = media['actual_steps']
    observations = []
    display = []
    selected = {row['completed_steps']: row for row in evidence['policy_observations']}
    if task_id == TASKS[0]:
        if steps != [4, 14, 24, 34, 44, 50, 60, 70, 80]:
            raise ValueError('original TwoPole media boundaries changed')
        metrics = {row['step']: row for row in evidence['observations']}
        for step in steps:
            with np.load(inputs[f'step_{step:06d}.npz'], allow_pickle=False) as cloud:
                data = {key: cloud[key].copy() for key in cloud.files}
            if set(data) != {'real', 'particles', 'critic_inputs', 'critic_gradient'}:
                raise ValueError('original TwoPole arrays changed')
            if (data['real'].shape != (12, 1) or data['particles'].shape != (12, 1)
                    or data['critic_inputs'].shape != (24, 1)
                    or data['critic_gradient'].shape != (24, 1)):
                raise ValueError('original TwoPole resource/gradient arrays changed')
            row = metrics[step]
            # The renderer treats one-column arrays as index-versus-value.
            # Explicit two-column display coordinates preserve every original
            # scalar as x and add only the declared display-only y=0 axis.
            real_display = np.column_stack((data['real'][:, 0], np.zeros(12)))
            particle_display = np.column_stack((data['particles'][:, 0], np.zeros(12)))
            observations.append({'step': step, 'passed': False,
                'metrics': {key: row[key] for key in ('mean_abs', 'grad_med')}, 'views': [
                    {'kind': 'scatter', 'title': 'Selected particles versus the two real poles',
                     'target': real_display, 'samples': particle_display, 'aspect': 'auto',
                     'xlim': [-1.1, 1.1], 'ylim': [-.08, .08], 'xlabel': 'particle value',
                     'ylabel': 'one-dimensional display axis',
                     'caption': 'Retained real/selected-table values. The vertical axis has no scientific meaning. Original mean_abs >= .30; no latent/output-noise draw.'},
                    {'kind': 'bar', 'title': 'Selected critic: absolute input gradients',
                     'target': np.full(data['critic_gradient'].size, 1., dtype=np.float64),
                     'samples': np.abs(data['critic_gradient'].reshape(-1)),
                     'xlabel': 'retained real12 then particle12 inputs', 'ylabel': '|critic input gradient|',
                     'caption': 'Retained selected-critic gradients, unchanged. Gray level 1 is the original median-gradient upper bound, not a per-bar scientific test.'}]})
            display.append({'step': step, 'input': str(inputs[f'step_{step:06d}.npz']),
                            'measurement': 'selected_table_and_critic_gradient',
                            'display_projection': 'original scalar values as x; display-only y=0; every original row retained',
                            'selected_policy_observation': deepcopy(selected[step]),
                            'original_metrics': deepcopy(row)})
        goal = 'TwoPole: particles must leave zero toward the two poles while critic gradients stay controlled. Original FAIL; display-only supplement, no ordinary credit.'
        horizon, name = 80, 'two_pole-1d-goal.gif'
    else:
        if steps != [0, 50, 750, 1750, 2750, 4000, 5000, 6000, 7000]:
            raise ValueError('original native media boundaries changed')
        problem = task_id.split('_policy_', 1)[0]
        events_path = inputs['events.jsonl']
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        primary = {row['step']: row for row in events if row['model'] == 'live'}
        center, sigma, circles = mode0_reference(problem)
        for step in steps:
            with np.load(inputs[f'step_{step:06d}.npz'], allow_pickle=False) as cloud:
                if set(cloud.files) != {'live', 'ema', 'target'}:
                    raise ValueError('original native arrays changed')
                samples, target = cloud['live'].copy(), cloud['target'].copy()
            if samples.shape != (4096, 2) or target.shape != (4096, 2):
                raise ValueError('original native display sample count changed')
            relative = samples.astype(np.float64) - center
            visible = np.isfinite(relative).all(1) & (np.abs(relative) <= 4 * sigma).all(1)
            row = primary[step]
            # Legacy `live` is the actual selected-policy primary in this
            # cohort. Forced EMA arrays never receive supplementary credit.
            saved = {**row['metrics'], **row['accuracy']}
            names = ('modes', 'min_cov_eig_ratio', 'min_radial_median_ratio', 'mass_tv',
                     'abs_cov_trace_bias', 'radial_ks')
            shown = {key: saved[key] for key in names if saved[key] is not None}
            state = 'Recorded covariance/radial diagnostics unavailable at this early state.' if saved['radial_ks'] is None else (
                f"Recorded abs_trace_bias={saved['abs_cov_trace_bias']:.4g} (<=.10); radial_KS={saved['radial_ks']:.4g} (<=.04).")
            observations.append({'step': step, 'passed': row['accuracy']['passed'], 'metrics': shown, 'views': [
                {'kind': 'scatter', 'title': 'All 100 locations: retained target and selected outputs',
                 'target': target, 'samples': samples, 'aspect': 'equal', 'xlabel': 'x', 'ylabel': 'y',
                 'caption': 'The original 4096 retained display draws; the full axes illustrate center geometry, which alone cannot prove within-mode Gaussian width.'},
                {'kind': 'scatter', 'title': 'Mode 0 width: analytic 1 sigma and 2 sigma contours',
                 'target': circles, 'samples': relative, 'aspect': 'equal',
                 'xlim': [-4 * sigma, 4 * sigma], 'ylim': [-4 * sigma, 4 * sigma],
                 'xlabel': 'x minus declared mode 0 center', 'ylabel': 'y minus declared mode 0 center',
                 'caption': f'Declared sigma=.03; gray circles are analytic contours, not samples. {int(visible.sum())}/4096 existing draws lie in this display viewport; outside points are clipped only by the axes. {state} No local metric is computed.'}]})
            display.append({'step': step, 'input': str(inputs[f'step_{step:06d}.npz']),
                            'display_selection': 'all4096 original coordinates translated by declaredmode0; axes clip to +/-4sigma viewport',
                            'visible_samples': int(visible.sum()), 'mode0_center': center.tolist(), 'sigma': sigma,
                            'original_event_model_label': 'live', 'actual_weights': 'state_selected',
                            'selected_policy_observation': deepcopy(selected[step]),
                            'metric_fields': list(shown), 'source_metric_step': row['step']})
        goal = f'{problem}: recover 100 Gaussian locations AND their within-mode width. Original FAIL; retained-array display supplement, no ordinary credit.'
        horizon, name = 7000, problem + '-mode0-goal.gif'
    return {'id': task_id, 'goal': goal, 'default_steps': horizon}, observations, display, name


def publish(raw_root, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '1' for k in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('display reproduction requires CUDA invisible and one CPU thread')
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a new output directory; original and prior supplements are immutable')
    output.mkdir(parents=True, exist_ok=True)
    consumed = {}
    def consume(path, attested=None):
        row = pin(path)
        if attested is not None and (row['sha256'] != attested['sha256']
                or row['bytes'] != attested.get('bytes', attested.get('size'))):
            raise ValueError('retained byte identity drift: ' + str(path))
        consumed[row['path']] = row
        return row
    study_pin = consume(raw_root / 'study.json')
    study = json.loads((raw_root / 'study.json').read_text())
    if (study['status'] != 'COMPLETE_DIAGNOSTIC' or study['qualification_input'] is not False
            or study['source']['origin_commit'] != SOURCE_COMMIT or study['source']['digest'] != SOURCE_DIGEST):
        raise ValueError('requires the exact completed native-v2 diagnostic source')
    snapshot = Path(study['source']['snapshot_path'])
    source_files = study['source']['files']
    for relative, wanted in source_files.items():
        path = snapshot / relative
        if not path.resolve().is_relative_to(snapshot.resolve()) or sha(path) != wanted:
            raise ValueError('pinned executable source changed: ' + relative)
    geometry_source = consume(snapshot / 'benchmarks/toy100/problems.py')
    sys.path.insert(0, str(REPO))
    from benchmarks.toy_audit.api_run import render_gif
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('display cannot initialize or use CUDA')
    torch.set_num_threads(1)
    renderer = {name: consume(REPO / name) for name in
                ('benchmarks/toy_audit/api_run.py', 'benchmarks/toy_audit/api_contract.py')}
    started = time.monotonic()
    emitted = []
    for task_id in TASKS:
        job = next(job for job in study['jobs'] if task_id in job['task_ids'])
        if job['status'] != 'COMPLETE' or job['outcome']['statuses'][task_id] != 'FAIL':
            raise ValueError('only exact completed diagnosticFAIL cases are supported')
        for key in ('raw', 'grading', 'resolved'):
            consume(job['outcome'][key]['path'], job['outcome'][key])
        consume(job['terminal']['path'], job['terminal'])
        raw = json.loads(Path(job['outcome']['raw']['path']).read_text())
        grading = json.loads(Path(job['outcome']['grading']['path']).read_text())
        if grading['grades'][task_id]['status'] != 'FAIL':
            raise ValueError('original grade drift')
        m = job['outcome']['media'][task_id]
        consume(m['gif']['path'], m['gif']); consume(m['receipt']['path'], m['receipt'])
        media = json.loads(Path(m['receipt']['path']).read_text())
        if (media['original_gate'] != 'FAIL' or media['qualification_input'] is not False
                or media['draws'] != 0 or media['optimizer_updates'] != 0
                or media['source_digest'] != SOURCE_DIGEST or media['gif'] != m['gif']
                or len(media['actual_steps']) != 9):
            raise ValueError('originalmedia contract changed')
        for source in media['inputs']: consume(source['path'], source)
        original_repo = REPO / ORIGINAL_DIR / (task_id + '.gif')
        consume(original_repo, m['gif'])
        case, observations, display, name = build_case(task_id, raw, media)
        before = records_hash(observations)
        target = output / name
        annotations = render_gif(case, observations, target, full_budget=True,
                                 requested_steps=case['default_steps'], final_verdict='FAIL')
        if before != records_hash(observations) or annotations['numeric_observations_changed'] is not False:
            raise ValueError('renderer mutated retained display projections')
        if annotations['default_verdict_displayed'] != 'FAIL':
            raise ValueError('display changed originalFAIL')
        with Image.open(target) as gif:
            if gif.n_frames != 9: raise ValueError('an original media state disappeared')
            dimensions = list(gif.size); gif.seek(8)
            poster = output / (Path(name).stem + '-poster.png'); gif.convert('RGB').save(poster)
        emitted.append({'task_id': task_id, 'original_verdict': 'FAIL', 'qualification_upgrade': False,
            'training_or_rescoring': False, 'original_raw': job['outcome']['raw'], 'original_grade': job['outcome']['grading'],
            'original_media': m, 'original_repo_gif': str(ORIGINAL_DIR / (task_id + '.gif')),
            'actual_steps': media['actual_steps'], 'frames': 9, 'views_projection_sha256': before,
            'display_projection': display, 'supplement_gif': {'file': name, 'sha256': sha(target),
                'bytes': target.stat().st_size, 'frames': 9, 'dimensions': dimensions},
            'poster': {'file': poster.name, 'sha256': sha(poster), 'bytes': poster.stat().st_size},
            'annotations': annotations, 'pixels_review': 'PENDING'})
    for row in consumed.values():
        if pin(row['path']) != row: raise ValueError('consumed input bytes changed during export')
    for relative, wanted in source_files.items():
        if sha(snapshot / relative) != wanted: raise ValueError('source changed during export')
    if torch.cuda.is_initialized(): raise ValueError('display unexpectedly initialized CUDA')
    receipt = {'schema': 'particlegan_atlas_retained_goal_supplements_v1', 'status': 'EXPORTED_PIXEL_REVIEW_PENDING',
        'script': pin(__file__), 'scientific_source': {'commit': SOURCE_COMMIT, 'digest': SOURCE_DIGEST,
            'snapshot': str(snapshot), 'executable_files_pre_post_verified': len(source_files)},
        'analytic_geometry_source': geometry_source, 'renderer_sources': renderer, 'original_study': study_pin,
        'display_cost_seconds': time.monotonic() - started, 'display_cost_scope': 'CPU rendering and post-import validation; source preparation/imports outside this clock; excluded from paid scientific acquisition and speed/convergence ranking',
        'scientific_paid_cost_unchanged': study['measured_paid_seconds'],
        'optimizer_updates': 0, 'model_constructions': 0, 'checkpoint_deserializations': 0,
        'sampler_calls': 0, 'new_draws': 0, 'official_scorer_calls': 0, 'GPU_calls': 0,
        'raw_files_unchanged': True, 'metrics_or_gates_changed': False, 'qualification_input': False,
        'records': emitted, 'consumed_inputs': list(consumed.values())}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = publish(args.raw.resolve(), args.output.resolve())
    print(json.dumps({'status': receipt['status'], 'records': len(receipt['records']),
        'display_cost_seconds': receipt['display_cost_seconds'], 'receipt_sha256': sha(args.output / 'receipt.json')}))


if __name__ == '__main__': main()
