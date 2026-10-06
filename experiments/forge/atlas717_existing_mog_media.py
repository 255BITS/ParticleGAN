"""Render actual saved training observations; no model, sampler or rescoring.

The exporter consumes a certified request/result envelope and verifies retained
arrays against the adapter's descriptor. It adds zero training or sampling. Atlas760 corrected MoG keeps the original task priors;
this module does not substitute NoisyParticlePrior or transfer Track A credit.
"""
import argparse
import importlib
from io import BytesIO
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .contracts import atomic_json, file_hash, read_json, stable_hash


EXPECTED_VIEW = {'assignments': [{'importance': 'required', 'order': 0, 'qualification_tier': 1, 'task': 'gaussian1d_acquisition'}, {'importance': 'required', 'order': 1, 'qualification_tier': 1, 'task': 'two_pole'}, {'importance': 'required', 'order': 2, 'qualification_tier': 1, 'task': 'unused_token_hold'}, {'importance': 'required', 'order': 3, 'qualification_tier': 1, 'task': 'ae_gan_hold'}, {'importance': 'required', 'order': 4, 'qualification_tier': 1, 'task': 'ring16_acquisition'}, {'importance': 'required', 'order': 5, 'qualification_tier': 1, 'task': 'five_word_joint_acquisition'}], 'calibration': {'adoption_blocker': 'Isolated Atlas760 corrected MoG reaction-kernel validation; no baseline grade transfer or default qualification.', 'status': 'provisional'}, 'eligibility': {'claim_contract': {'experimental_track': 'atlas760_existing_mog_kernel'}}, 'evidence_scope': 'original_task_existing_mog_compatibility', 'goal': 'discriminator_stability', 'id': 'atlas_existing_mog_kernel760_v1', 'ranking': {'compare_compatible_cohorts': True, 'cost_separate': True, 'policy': 'qualified_tier_only_with_raw_metrics'}, 'reporting': {'family_totals': False}, 'revision': 1, 'schema_version': 1}
EXPECTED_CLAIM_CONTRACT = {'experimental_track': 'atlas760_existing_mog_kernel', 'sampling_law': 'task_declared', 'schedule': 'schedule_free', 'scoring_weights': 'live'}
OWNER_PINS = {'atlas_existing_mog': 'c04db8875742ca5658919be94d76dae294fa0d1008208a3732ef96c68cc63479', 'atlas_existing_mog_ae': '2b0f4df2579731286d7e93d4505e304e3f928c6033bf1e63826e4abc6ff68fc5', 'atlas_two_pole': '4159eaa0d8d370615374baa460780e12a100640f22c352488ee6c5ed3ace65f9'}

def _indices(count):
    if count < 1:
        raise ValueError('actual observations are required for training media')
    return sorted({round(i * (count - 1) / min(8, count - 1)) for i in range(min(9, count))}) if count > 1 else [0]


def _validate_request_scope(request):
    """Keep this exporter confined to one separately certified Atlas760 corrected MoG request."""
    from .atlas_existing_mog import PARENTS, supports_candidate, validate
    for name, expected in OWNER_PINS.items():
        module = importlib.import_module('.' + name, __package__)
        if not getattr(module, '__file__', None) or file_hash(Path(module.__file__)) != expected:
            raise ValueError('Atlas760 corrected MoG media owner Source differs: ' + name)
    if (not isinstance(request, dict) or not supports_candidate(request.get('candidate', {}))
            or request['candidate'].get('claim_contract') != EXPECTED_CLAIM_CONTRACT
            or request.get('view') != EXPECTED_VIEW or request.get('through_tier') != 1
            or not isinstance(request.get('tasks'), dict)
            or set(request['tasks']) != set(PARENTS)):
        raise ValueError('Atlas760 corrected MoG media requires its exact candidate, view and six original tasks')
    for name, task in request['tasks'].items():
        if not isinstance(task, dict) or task.get('id') != name:
            raise ValueError('Atlas760 corrected MoG task-map keys differ from their original identities')
        validate(task)


def _existing_mog_media_binding(task, evidence):
    """Bind original scored outputs to the actual existing-prior owner receipt."""
    from .atlas_existing_mog import CANDIDATE_ID, COHORT, SUPPORTED, validate, validate_evidence
    binding = validate(task)
    parent = binding['task_id']
    if parent not in SUPPORTED:
        raise ValueError('a blocked owner has no certified scientific goal media')
    invalid = validate_evidence(task, evidence)
    if invalid is not None:
        raise ValueError('Atlas760 corrected MoG media requires complete original owner evidence: ' + invalid['reason'])
    observations = evidence.get('observations', [])
    clocks = sorted({math.ceil(i * task['execution']['steps'] / 24) for i in range(1, 25)})
    if (not isinstance(observations, list) or len(observations) != 24
            or any(not isinstance(row, dict) or type(row.get('step')) is not int for row in observations)
            or [row['step'] for row in observations] != clocks):
        raise ValueError('Atlas760 corrected MoG media requires exactly the original24 scored clocks')
    sampled = parent in {'gaussian1d_acquisition', 'ring16_acquisition', 'ae_gan_hold'}
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
            raise ValueError('Atlas760 corrected MoG media requires the original retained scored-output descriptor')
        if evidence.get('host', {}).get('definition') != task['execution']['host_definition']:
            raise ValueError('Atlas760 corrected MoG target geometry differs from the fixed original host')
    if parent == 'two_pole' and not evidence.get('saved_particle_observations'):
        raise ValueError('Atlas760 corrected MoG direct control requires saved actual live coordinates')
    if parent == 'ae_gan_hold':
        _validate_ae_records(observations, evidence.get('saved_ae_observations'))
    if parent == 'two_pole':
        _validate_particle_records(observations, evidence.get('saved_particle_observations'))
    return {**binding, 'parent_task_id': parent, 'candidate_id': CANDIDATE_ID,
            'cohort': COHORT, 'public_prior_type': 'particlegan.particle_prior.MoGParticlePrior' if sampled else None,
            'sigma': binding['prior']['sigma'], 'sigma_units': 'raw_latent_coordinates',
            'latent_prior_sampled': sampled,
            'weights': task['evaluation']['scoring_weights'],
            'sampling_law': task['evaluation']['sampling_law'],
            'eval_output_noise': task['evaluation']['eval_output_noise'],
            'prior_substituted': False, 'cross_track_credit': False}


def _validate_scored_records(task, observations, records):
    """Join original retained tensors to their scored clocks without a draw."""
    dimensions = len(task['execution']['host_definition']['means'][0])
    if (not isinstance(records, list) or len(records) != 24
            or any(not isinstance(row, dict) or set(row) != {'step', 'samples'}
                   or type(row['step']) is not int
                   or not isinstance(row['samples'], torch.Tensor)
                   or tuple(row['samples'].shape) != (4096, dimensions)
                   or not row['samples'].is_floating_point()
                   or not bool(torch.isfinite(row['samples']).all()) for row in records)
            or [row['step'] for row in records] != [row['step'] for row in observations]):
        raise ValueError('Atlas760 corrected MoG media requires the original24 scored4096 tensors and exact clocks')


def _validate_ae_records(observations, records):
    """Bind the original two decoder outputs and paired target from each read."""
    shapes = {'generated': (1024, 2), 'reconstructed': (1024, 2),
              'target': (1024, 2), 'prior': (12, 2), 'anchors': (2, 2)}
    if not isinstance(records, list) or len(records) != 24 or len(observations) != 24:
        raise ValueError('Atlas760 corrected MoG AE media requires all original24 saved reads')
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


def _validate_particle_records(observations, records):
    """Join all original direct coordinates, targets and gradients to scored reads."""
    if not isinstance(records, list) or len(records) != 24 or len(observations) != 24:
        raise ValueError('Atlas760 corrected MoG direct media requires all original24 saved reads')
    for row, measured in zip(records, observations):
        if (not isinstance(row, dict) or type(row.get('step')) is not int
                or row['step'] != measured['step'] or row.get('metrics') != measured):
            raise ValueError('saved direct arrays differ from original scored clocks/metrics')
        for key in ('particles', 'target', 'critic_gradient'):
            values = np.asarray(row.get(key), dtype=float)
            shape = (24, 1) if key == 'critic_gradient' else (12, 1)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError('saved original finite direct ' + key + ' shape differs')


def render(request, task, row, local, output):
    _validate_request_scope(request)
    if task != request['tasks'].get(task.get('id')) or row.get('task_id') != task.get('id'):
        raise ValueError('Atlas760 corrected MoG media task/result differs from the exact admitted request')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    local, output = Path(local), Path(output)
    if row.get('gate_status') not in {'PASS', 'FAIL'}:
        raise ValueError('only certified numerical outcomes have scientific goal media')
    evidence = row['evidence']
    observations = evidence.get('observations', [])
    existing = _existing_mog_media_binding(task, evidence)
    frames, inputs = [], {}
    samples = None
    descriptor = evidence.get('saved_observer_outputs')
    if descriptor:
        path = (local / descriptor['path']).resolve()
        if path != local.resolve() / descriptor['path'] or path.is_symlink() or file_hash(path) != descriptor['sha256'] or path.stat().st_size != descriptor['bytes']:
            raise ValueError('retained scored-output descriptor differs from saved bytes')
        samples = torch.load(path, map_location='cpu', weights_only=True)
        if [point['step'] for point in samples] != [point['step'] for point in observations]:
            raise ValueError('scored output and numerical observation schedules differ')
        if existing and existing['parent_task_id'] in {'gaussian1d_acquisition', 'ring16_acquisition'}:
            _validate_scored_records(task, observations, samples)
        inputs[descriptor['path']] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    thresholds = task['evaluation']['thresholds']
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
                if existing:
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
            if ((task['id'] != 'two_pole' and not (existing and existing['parent_task_id'] == 'two_pole'))
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
        elif existing and existing['parent_task_id'] == 'ae_gan_hold':
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
        else:
            raise ValueError('Atlas760 corrected MoG media requires original scored distributions or direct coordinates')
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
        bottom.set_xlim(0, max(x)) if thresholds else None
        caption = f"{task['id']} · update {observations[index]['step']} · recorded {row['gate_status']}\n{evidence.get('scoring_weights', 'live')} / {evidence['sampling_law']}"
        if existing['latent_prior_sampled']:
            kernel = f"fixed latent sigma {existing['sigma']:g} in raw units" if existing['latent_prior_sampled'] else 'sigma0 direct coordinates; no prior draw'
            caption += '\nAtlas760 corrected MoG · original MoGParticlePrior · ' + kernel + '; original observer retained'
        if not existing['latent_prior_sampled']:
            caption += '\nAtlas760 corrected MoG · unchanged direct Parameter control; no latent kernel draw'
        figure.suptitle(caption, fontsize=9 if existing else 10)
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
    if existing and existing['parent_task_id'] == 'ae_gan_hold':
        receipt['saved_ae_observations_sha256'] = stable_hash(evidence['saved_ae_observations'])
        receipt['illustrated_goal'] = 'Original live unconditional hold <=0.35 and paired reconstruction MSE <=0.05 at the final five observations; both saved decoder outputs shown.'
    if existing:
        receipt['existing_prior_contract'] = existing
        receipt['candidate_revision'] = request['candidate_revision']
        receipt['source'] = {key: request['source'][key] for key in ('commit', 'origin_commit', 'digest')
                             if key in request['source']}
        receipt['protocol_sha256'] = stable_hash(request['protocol'])
        receipt['request_sha256'] = stable_hash(request)
        if existing['parent_task_id'] in {'gaussian1d_acquisition', 'ring16_acquisition'}:
            receipt['target_geometry_sha256'] = stable_hash(task['execution']['host_definition'])
        receipt['illustrated_distribution'] = ('original_scored4096_samples_and_declared_target_geometry'
            if existing['parent_task_id'] in {'gaussian1d_acquisition', 'ring16_acquisition'}
            else 'original_saved_live_coordinates' if existing['parent_task_id'] == 'two_pole'
            else 'original_saved_unconditional_generation_paired_reconstruction_and_target' if existing['parent_task_id'] == 'ae_gan_hold'
            else 'original_saved_numeric_observations')
        receipt['prior_substituted'] = False
        receipt['cross_track_credit'] = False
        receipt['extra_noise_assumed'] = False
    atomic_json(output.with_suffix('.json'), receipt)
    return receipt


def export_attempt(directory, output):
    """Use the maintained collected certificate; never evaluate or launch a case."""
    directory = Path(directory)
    envelope, result, certificate = [read_json(directory / (name + '.json')) for name in ('request', 'result', 'evidence')]
    request, job, worker = envelope['request'], envelope['job'], envelope['worker']
    _validate_request_scope(request)
    if (certificate['result_hash'] != stable_hash(result) or certificate['source'] != request['source']
            or certificate.get('runtime') != request.get('runtime')
            or result['candidate_revision'] != request['candidate_revision']
            or result['attempt_id'] != worker['attempt']):
        raise ValueError('original source/request/result certificate differs')
    raw = result['raw']
    if (raw.get('attempt_status') != 'completed' or raw.get('token') != worker['token']
            or raw.get('grading', {}).get('raw_hash') != stable_hash(raw.get('result', {}))
            or raw.get('grading', {}).get('source_digest') != request['source']['digest']):
        raise ValueError('a matching completed terminal and frozen grader certificate are required')
    rows = result['task_results']
    expected_tasks = job.get('task_ids', [job['task_id']])
    if ([row['task_id'] for row in rows] != expected_tasks
            or len(set(expected_tasks)) != len(expected_tasks)
            or any(name not in request['tasks'] for name in expected_tasks)):
        raise ValueError('collected task rows differ from the original admitted job')
    for row in rows:
        if row.get('gate_status') in {'PASS', 'FAIL'}:
            grade = raw['grading']['grades'].get(row['task_id'], {})
            if (row.get('raw_status') != 'completed'
                    or row['gate_status'] != grade.get('gate_status', grade.get('status', 'INVALID'))):
                raise ValueError('scientific media status differs from the original frozen grade')
    local = Path(certificate['local_artifact_root'])
    if (local.resolve() != local or local != Path(worker['directory'])
            or read_json(local / 'result.json') != result):
        raise ValueError('local original artifact root/result differs from the certified attempt')
    return [render(request, request['tasks'][row['task_id']], row, local,
                   Path(output) / (row['task_id'] + '.gif')) for row in rows
            if row['gate_status'] in {'PASS', 'FAIL'} and row.get('evidence')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_attempt(args.attempt, args.output)

if __name__ == '__main__':
    main()
