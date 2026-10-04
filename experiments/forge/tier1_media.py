"""Render actual saved training observations; no model, sampler or rescoring.

The exporter consumes a certified request/result envelope and verifies retained
arrays against the adapter's descriptor. It adds zero training or sampling.
"""
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .contracts import atomic_json, file_hash, read_json, stable_hash


def _indices(count):
    if count < 1:
        raise ValueError('actual observations are required for training media')
    return sorted({round(i * (count - 1) / min(8, count - 1)) for i in range(min(9, count))}) if count > 1 else [0]


def render(task, row, local, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    local, output = Path(local), Path(output)
    evidence = row['evidence']
    observations = evidence.get('observations', [])
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
                extent = max(4., float(np.abs(centers).max()) + 1.)
                top.set_xlim(-extent, extent); top.set_ylim(-extent, extent); top.set_aspect('equal')
            top.legend(loc='upper right', fontsize=8)
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
        figure.suptitle(f"{task['id']} · update {observations[index]['step']} · recorded {row['gate_status']}\n{evidence.get('scoring_weights', 'live')} / {evidence['sampling_law']}", fontsize=10)
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
    atomic_json(output.with_suffix('.json'), receipt)
    return receipt


def export_attempt(directory, output):
    directory = Path(directory)
    envelope, result, certificate = [read_json(directory / (name + '.json')) for name in ('request', 'result', 'evidence')]
    request = envelope.get('request', envelope)
    if certificate['result_hash'] != stable_hash(result) or certificate['source'] != request['source']:
        raise ValueError('original source/result certificate differs')
    if result['candidate_revision'] != request['candidate_revision']:
        raise ValueError('candidate revision differs')
    local = Path(certificate['local_artifact_root'])
    if read_json(local / 'result.json') != result:
        raise ValueError('local result differs from certified envelope')
    return [render(request['tasks'][row['task_id']], row, local,
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
