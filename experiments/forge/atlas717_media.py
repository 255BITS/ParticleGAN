"""Render actual saved training observations; no model, sampler or rescoring.

The exporter consumes a certified request/result envelope and verifies retained
arrays against the adapter's descriptor. It adds zero training or sampling.
"""
import argparse
from io import BytesIO
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .contracts import atomic_json, file_hash, read_json, stable_hash


def _indices(count):
    if count < 1:
        raise ValueError('actual observations are required for training media')
    return sorted({round(i * (count - 1) / min(8, count - 1)) for i in range(min(9, count))}) if count > 1 else [0]


def _noisy_media_binding(task, evidence):
    """Validate the isolated prior substitution before illustrating saved reads.

    This invokes only the source-bound metadata/owner receipt guards. Original
    numeric grades are carried by the certified envelope, never recomputed.
    """
    prior = task.get('execution', {}).get('prior', {})
    if not (task.get('task_cohort') == 'atlas_noisy_particle025_tier1_717_v1'
            or str(task.get('id', '')).endswith('_noisy025717_v1')
            or prior.get('kind') == 'noisy_particle_cloud'
            or 'prior_substitution_parent' in task):
        return None
    from .atlas_noisy025_tier1 import validate
    from .atlas_noisy025_adapters import validate_evidence
    binding = validate(task)
    invalid = validate_evidence(task, evidence)
    if invalid is not None:
        raise ValueError('Noisy media requires the original complete owner evidence: ' + invalid['reason'])
    observations = evidence.get('observations', [])
    clocks = sorted({math.ceil(i * task['execution']['steps'] / 24) for i in range(1, 25)})
    if (len(observations) != 24
            or any(type(row.get('step')) is not int for row in observations)
            or [row['step'] for row in observations] != clocks):
        raise ValueError('Noisy media requires exactly the original24 scored clocks')
    parent = binding['parent_task_id']
    if parent in {'gaussian1d_acquisition', 'ring16_acquisition'}:
        descriptor = evidence.get('saved_observer_outputs', {})
        if (descriptor.get('path') != 'observed-samples.pt'
                or descriptor.get('kind') != 'scored_vector_samples_v1'
                or type(descriptor.get('observation_count')) is not int
                or descriptor['observation_count'] != 24
                or any(type(descriptor.get(key)) is not int or descriptor[key] != 0
                       for key in ('optimizer_updates_added', 'sampling_draws_added'))
                or type(descriptor.get('bytes')) is not int or descriptor['bytes'] <= 0
                or not isinstance(descriptor.get('sha256'), str)
                or len(descriptor['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in descriptor['sha256'])):
            raise ValueError('Noisy distribution media requires the original retained scored-output descriptor')
        if evidence.get('host', {}).get('definition') != task['execution']['host_definition']:
            raise ValueError('Noisy media target geometry differs from the fixed original host')
    if parent == 'two_pole' and not evidence.get('saved_particle_observations'):
        raise ValueError('Noisy two-pole media requires saved actual live coordinates')
    if parent == 'ae_gan_hold':
        _validate_noisy_ae_records(observations, evidence.get('saved_ae_observations'))
    return {**binding, 'public_prior_type': 'particlegan.noisy_particle_prior.NoisyParticlePrior',
            'sigma_units': 'raw_latent_coordinates',
            'latent_prior_sampled': parent != 'two_pole',
            'weights': task['evaluation']['scoring_weights'],
            'sampling_law': task['evaluation']['sampling_law'],
            'eval_output_noise': task['evaluation']['eval_output_noise']}


def _validate_noisy_scored_records(task, observations, records):
    """Join original retained tensors to their scored clocks without a draw."""
    dimensions = len(task['execution']['host_definition']['means'][0])
    if (not isinstance(records, list) or len(records) != 24
            or any(not isinstance(row, dict) or set(row) != {'step', 'samples'}
                   or type(row['step']) is not int
                   or not isinstance(row['samples'], torch.Tensor)
                   or tuple(row['samples'].shape) != (4096, dimensions)
                   or not row['samples'].is_floating_point() for row in records)
            or [row['step'] for row in records] != [row['step'] for row in observations]):
        raise ValueError('Noisy media requires the original24 scored4096 tensors and exact clocks')


def _validate_noisy_ae_records(observations, records):
    """Bind the original two decoder outputs and paired target from each read."""
    shapes = {'generated': (1024, 2), 'reconstructed': (1024, 2),
              'target': (1024, 2), 'prior': (12, 2), 'anchors': (2, 2)}
    if not isinstance(records, list) or len(records) != 24 or len(observations) != 24:
        raise ValueError('Noisy AE media requires all original24 saved reads')
    for row, measured in zip(records, observations):
        if (not isinstance(row, dict) or type(row.get('step')) is not int
                or type(measured.get('step')) is not int
                or row['step'] != measured['step']
                or row.get('metrics') != {k: v for k, v in measured.items() if k != 'step'}):
            raise ValueError('saved AE arrays differ from original scored clocks/metrics')
        for key, shape in shapes.items():
            values = np.asarray(row.get(key), dtype=float)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError('saved original finite AE ' + key + ' shape differs')


def render(request, task, row, local, output):
    from .atlas_noisy025_tier1 import validate_request_scope
    validate_request_scope(request)
    if task != request['tasks'].get(task.get('id')) or row.get('task_id') != task.get('id'):
        raise ValueError('Track A717 media task/result differs from the exact admitted request')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    local, output = Path(local), Path(output)
    evidence = row['evidence']
    observations = evidence.get('observations', [])
    noisy = _noisy_media_binding(task, evidence)
    frames, inputs = [], {}
    samples = None
    descriptor = evidence.get('saved_observer_outputs')
    if descriptor:
        path = (local / descriptor['path']).resolve()
        if not path.is_relative_to(local.resolve()) or file_hash(path) != descriptor['sha256'] or path.stat().st_size != descriptor['bytes']:
            raise ValueError('retained scored-output descriptor differs from saved bytes')
        samples = torch.load(path, map_location='cpu', weights_only=True)
        if [point['step'] for point in samples] != [point['step'] for point in observations]:
            raise ValueError('scored output and numerical observation schedules differ')
        if noisy and noisy['parent_task_id'] in {'gaussian1d_acquisition', 'ring16_acquisition'}:
            _validate_noisy_scored_records(task, observations, samples)
        inputs[str(path)] = file_hash(path)
    if task['adapter'] == 'clockfree_audit':
        from .artifacts import verify_artifacts
        from .clockfree import learning_state
        root = Path(evidence['artifact_root'])
        verify_artifacts(root, evidence['artifact_manifest'])
        proof_path = root / 'comparisons.pt'
        proof = torch.load(proof_path, map_location='cpu', weights_only=True)
        inputs[str(proof_path)] = file_hash(proof_path)
        # The saved digest comparisons are the declared numerical gate. Every
        # frame also shows actual learned G parameter movement in each branch.
        names = list(proof['trajectories'])
        initial = proof['initial']['trainer']['models']['G']
        parameter = next(key for key, value in initial.items() if value.is_floating_point() and value.numel())
        observations = [{'step': i + 1, **{name: float((proof['trajectories'][name][i]['trainer']['models']['G'][parameter] - initial[parameter]).abs().mean()) for name in names}} for i in range(task['execution']['probe_steps'])]
        thresholds = []
    else:
        thresholds = task['evaluation']['thresholds']
        names = [name for name, _, _ in thresholds]
    if not observations:
        raise ValueError('no saved actual training observations')
    indices = _indices(len(observations))
    x = [point['step'] for point in observations]
    for index in indices:
        figure, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
        top, bottom = axes
        if samples and 'samples' in samples[index]:
            values = samples[index]['samples'].numpy()
            spec = task['execution']['host_definition']
            if values.shape[1] == 1:
                sigma = float(np.sqrt(spec['covariances'][0][0][0])); mean = spec['means'][0][0]
                limits = (mean - 4 * sigma, mean + 4 * sigma)
                bins = np.linspace(*limits, 45)
                counts, _ = np.histogram(values[:, 0], bins=bins)
                top.stairs(counts / (len(values) * np.diff(bins)), bins, fill=True, alpha=.6, label='Saved scored samples')
                top.set_title(f'{1 - counts.sum() / len(values):.1%} of scored samples outside fixed target axes')
                grid = np.linspace(*limits, 200)
                top.plot(grid, np.exp(-.5 * ((grid - mean) / sigma)**2) / (sigma * np.sqrt(2*np.pi)), color='black', label='Declared target density')
                top.set_xlim(*limits); top.set_ylim(0, 2 / sigma)
            else:
                centers = np.asarray(spec['means'])
                top.scatter(values[:, 0], values[:, 1], s=2, alpha=.25, label='Saved scored samples')
                top.scatter(centers[:, 0], centers[:, 1], marker='+', color='black', label='Declared target centers')
                if noisy:
                    from matplotlib.patches import Ellipse
                    for i, (center, covariance) in enumerate(zip(centers, spec['covariances'])):
                        eigenvalues, vectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
                        order = np.argsort(eigenvalues)[::-1]
                        eigenvalues, vectors = eigenvalues[order], vectors[:, order]
                        angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
                        top.add_patch(Ellipse(center, *[float(6 * np.sqrt(v)) for v in eigenvalues],
                                             angle=angle, fill=False, color='black', linewidth=.6,
                                             label='Declared target 3-sigma contours' if i == 0 else None))
                extent = max(4., float(np.abs(centers).max()) + 1.)
                top.set_xlim(-extent, extent); top.set_ylim(-extent, extent); top.set_aspect('equal')
            top.legend(loc='upper right', fontsize=8)
        elif evidence.get('saved_particle_observations'):
            retained = evidence['saved_particle_observations']
            if ((task['id'] != 'two_pole' and not (noisy and noisy['parent_task_id'] == 'two_pole'))
                    or len(retained) != len(observations)
                    or [point['step'] for point in retained] != x
                    or any(point['metrics'] != measured
                           for point, measured in zip(retained, observations))):
                raise ValueError('saved live particle observations differ from scored clocks/metrics')
            particles = np.asarray(retained[index]['particles'], dtype=float)
            target = np.asarray(retained[index]['target'], dtype=float)
            if particles.shape != (12, 1) or target.shape != (12, 1):
                raise ValueError('ordinary two-pole media requires the original twelve live coordinates')
            all_particles = np.asarray([point['particles'] for point in retained], dtype=float)
            if not np.isfinite(all_particles).all():
                raise ValueError('saved live coordinates contain nonfinite values')
            extent = max(1.2, float(np.abs(all_particles).max()) * 1.1)
            top.axvline(0, color='grey', linestyle=':', label='Original zero start')
            top.axvline(-1, color='black', linestyle='--', label='Declared target poles')
            top.axvline(1, color='black', linestyle='--')
            top.scatter(particles[:, 0], np.arange(12), c=particles[:, 0],
                        cmap='coolwarm', vmin=-extent, vmax=extent, s=42,
                        label='Saved live particle positions')
            top.set_xlim(-extent, extent); top.set_ylim(-.7, 11.7)
            top.set_xlabel('Live coordinate'); top.set_ylabel('Particle row')
            top.set_title('Travel from zero + bounded critic gradient; both-pole balance is ungated', fontsize=10)
            top.legend(loc='upper right', fontsize=7)
        elif noisy and noisy['parent_task_id'] == 'ae_gan_hold':
            retained = evidence['saved_ae_observations']
            record = retained[index]
            target, generated, reconstructed = [np.asarray(record[key], dtype=float)
                for key in ('target', 'generated', 'reconstructed')]
            anchors = np.asarray(record['anchors'], dtype=float)
            top.scatter(target[:, 0], target[:, 1], s=3, alpha=.2, color='grey', label='Saved scored target')
            top.scatter(generated[:, 0], generated[:, 1], s=3, alpha=.3, label='Saved unconditional generation')
            top.scatter(reconstructed[:, 0], reconstructed[:, 1], s=3, alpha=.3, label='Saved paired reconstruction')
            top.scatter(anchors[:, 0], anchors[:, 1], marker='+', color='black', s=70, label='Original target anchors')
            extent = max(2., max(float(np.abs(np.asarray(saved[key], dtype=float)).max())
                for saved in retained for key in ('target', 'generated', 'reconstructed', 'anchors')) * 1.1)
            top.set_xlim(-extent, extent); top.set_ylim(-extent, extent); top.set_aspect('equal')
            top.set_title('Original live generation + reconstruction, with scheduled output noise', fontsize=10)
            top.legend(loc='upper right', fontsize=7)
        elif samples and 'views' in samples[index]:
            # Established public API word renderer consumes the already retained
            # generated and paired-reconstructed views, without querying models.
            from benchmarks.toy_audit.api_run import render_gif
            records = []
            for i in indices:
                record = samples[i]
                records.append({'step': observations[i]['step'], 'metrics': record['metrics'],
                                'views': record['views'], 'passed': record['passed'],
                                'failed_bounds': record['failed_bounds']})
            case = {'id': task['id'], 'goal': task.get('description', task['id']),
                    'default_steps': task['execution']['steps'], 'sampling': evidence['sampling_law']}
            output.parent.mkdir(parents=True, exist_ok=True)
            render_gif(case, records, output, full_budget=True, requested_steps=task['execution']['steps'], final_verdict=row['gate_status'])
            plt.close(figure)
            break
        else:
            for name in names:
                top.plot(x[:index+1], [point[name] for point in observations[:index+1]], label=name)
            all_values = [point[name] for point in observations for name in names]
            low, high = min(all_values), max(all_values)
            padding = max(.01, (high-low)*.05)
            top.set_ylim(low-padding, high+padding)
            top.set_xlim(0, max(x)); top.set_title('Actual training measurements' if thresholds else 'Actual G parameter movement from the shared saved start')
            top.legend(fontsize=7)
        if thresholds:
            for name, operator, bound in thresholds:
                values = [point[name] for point in observations]
                denominator = abs(bound) if bound else 1.
                bottom.plot(x[:index+1], [(v - bound) / denominator for v in values[:index+1]], label=f'{name} {operator} {bound}')
            all_margins = [(point[name]-bound)/(abs(bound) if bound else 1.) for point in observations for name, _, bound in thresholds]
            low, high = min(0., min(all_margins)), max(0., max(all_margins))
            padding = max(.01, (high-low)*.05)
            bottom.set_ylim(low-padding, high+padding)
            bottom.axhline(0, color='black', lw=1)
            bottom.set_title('Measured distance from numerical bounds; direction follows each listed inequality')
            bottom.legend(fontsize=7, ncol=2)
        else:
            comparisons = evidence['comparisons']
            labels = [point['condition'] for point in comparisons]
            bottom.bar(labels, [int(point['reference_sha256'] != point['changed_sha256']) for point in comparisons])
            bottom.set_ylim(0, 1.2); bottom.set_title('Full saved-state parity: 0 equal / 1 different')
        bottom.set_xlim(0, max(x)) if thresholds else None
        caption = f"{task['id']} · update {observations[index]['step']} · recorded {row['gate_status']}\n{evidence.get('scoring_weights', 'live')} / {evidence['sampling_law']}"
        if noisy:
            kernel = f"fixed latent sigma {noisy['sigma']:g} in raw units" if noisy['latent_prior_sampled'] else 'sigma0 direct coordinates; no prior draw'
            caption += '\nTrack A717 · NoisyParticlePrior · ' + kernel + '; declared .025 kernel; original observer retained'
        if task.get('id') in {'two_pole', 'unused_token_hold'}:
            caption += '\nTrack A717 · unchanged direct Parameter control; no latent kernel draw'
        figure.suptitle(caption, fontsize=9 if noisy else 10)
        buffer = BytesIO(); figure.savefig(buffer, format='png', dpi=100); plt.close(figure)
        buffer.seek(0); frames.append(Image.open(buffer).convert('RGB'))
    if frames:
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=400, loop=0)
    receipt = {'schema_version': 1, 'task_id': task['id'], 'recorded_grade': row['gate_status'],
               'kind': 'actual_training_saved_observations_gif', 'observation_count': len(observations),
               'selected_observation_indices': indices, 'source_inputs': inputs,
               'observations_sha256': stable_hash(observations), 'gif_sha256': file_hash(output),
               'optimizer_updates_added': 0, 'sampling_draws_added': 0, 'qualification_input': False,
               'renderer_sha256': file_hash(Path(__file__))}
    if evidence.get('saved_particle_observations'):
        receipt['saved_particle_observations_sha256'] = stable_hash(evidence['saved_particle_observations'])
        receipt['illustrated_goal'] = 'Live coordinate travel >=0.30 and critic gradient median <=1.0 at the final five observations; no pole-balance gate.'
    if noisy and noisy['parent_task_id'] == 'ae_gan_hold':
        receipt['saved_ae_observations_sha256'] = stable_hash(evidence['saved_ae_observations'])
        receipt['illustrated_goal'] = 'Original live unconditional hold <=0.35 and paired reconstruction MSE <=0.05 at the final five observations; both saved decoder outputs shown.'
    if noisy:
        receipt['prior_substitution'] = noisy
        receipt['target_geometry_sha256'] = stable_hash(task['execution']['host_definition'])
        receipt['illustrated_distribution'] = ('original_scored4096_samples_and_declared_target_geometry'
            if noisy['parent_task_id'] in {'gaussian1d_acquisition', 'ring16_acquisition'}
            else 'original_saved_live_coordinates' if noisy['parent_task_id'] == 'two_pole'
            else 'original_saved_unconditional_generation_paired_reconstruction_and_target' if noisy['parent_task_id'] == 'ae_gan_hold'
            else 'original_saved_numeric_observations')
        receipt['extra_noise_assumed'] = False
    atomic_json(output.with_suffix('.json'), receipt)
    return receipt


def export_attempt(directory, output):
    directory = Path(directory)
    envelope, result, certificate = [read_json(directory / (name + '.json')) for name in ('request', 'result', 'evidence')]
    request = envelope.get('request', envelope)
    from .atlas_noisy025_tier1 import validate_request_scope
    validate_request_scope(request)
    if certificate['result_hash'] != stable_hash(result) or certificate['source'] != request['source']:
        raise ValueError('original source/result certificate differs')
    if result['candidate_revision'] != request['candidate_revision']:
        raise ValueError('candidate revision differs')
    local = Path(certificate['local_artifact_root'])
    if read_json(local / 'result.json') != result:
        raise ValueError('local result differs from certified envelope')
    return [render(request, request['tasks'][row['task_id']], row, local,
                   Path(output) / (row['task_id'] + '.gif')) for row in result['task_results']
            if row['gate_status'] in {'PASS', 'FAIL', 'BLOCKED'} and row.get('evidence')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_attempt(args.attempt, args.output)

if __name__ == '__main__':
    main()
