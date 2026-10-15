"""Export existing certified training observations, without model evaluation."""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path('/ml2/hypergan/particlegan-selective-history-20261015')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from experiments.forge import tier1_media
import torch

ORIGINAL_SCORED_OUTPUTS = tier1_media._scored_outputs
ALIGNMENTS = []


def same(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def aligned_scored_outputs(task, evidence, local):
    try:
        return ORIGINAL_SCORED_OUTPUTS(task, evidence, local)
    except ValueError as error:
        if str(error) != 'scored output and numerical observation schedules differ':
            raise
    # The original reader has already verified the source hash, byte count and
    # payload types. Some API recorders also retain initial/visual-only frames.
    # Select only exact scored steps and exact metric matches, without rescoring.
    path = Path(local) / evidence['saved_observer_outputs']['path']
    samples = torch.load(path, map_location='cpu', weights_only=True)
    indices = []
    for observed in evidence['observations']:
        candidates = [i for i, sample in enumerate(samples)
                      if sample.get('step') == observed['step']
                      and all(sample.get('metrics', {}).get(key) == value
                              for key, value in observed.items() if key != 'step')]
        if not candidates or any(not same(samples[candidates[0]], samples[i]) for i in candidates[1:]):
            raise ValueError('missing or ambiguous exact scored media frame')
        indices.append(candidates[0])
    ALIGNMENTS.append({'task_id': task['id'], 'source_path': str(path),
                       'source_frame_count': len(samples),
                       'selected_source_indices': indices,
                       'method': 'exact scored step and all authoritative metric fields; identical duplicates only'})
    return [samples[i] for i in indices], {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}


tier1_media._scored_outputs = aligned_scored_outputs


def main():
    output = ROOT / 'media'
    index = []
    campaigns = sorted((ROOT / 'queue').iterdir())
    for campaign in campaigns:
        if not campaign.is_dir():
            continue
        for directory in sorted(campaign.iterdir()):
            if not directory.is_dir() or not (directory / 'terminal.json').exists():
                continue
            certified = REPO / 'reports/forge/attempts' / directory.name
            if not (certified / 'evidence.json').exists():
                continue
            envelope = json.loads((certified / 'request.json').read_text())
            request = envelope.get('request', envelope)
            candidate = request['candidate']['id']
            destination = output / candidate / directory.name
            ALIGNMENTS.clear()
            receipts = tier1_media.export_attempt(certified, destination)
            for receipt in receipts:
                receipt['postprocessor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                receipt['scored_frame_alignment'] = list(ALIGNMENTS)
                (destination / (receipt['task_id'] + '.json')).write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + '\n')
            index.append({'attempt_id': directory.name, 'candidate': candidate,
                          'campaign': campaign.name, 'certified_directory': str(certified),
                          'output_directory': str(destination), 'media': receipts})
            print(candidate, directory.name, len(receipts), flush=True)
    output.mkdir(exist_ok=True)
    (output / 'index.json').write_text(json.dumps({
        'kind': 'certified_actual_training_media',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'optimizer_updates_added': 0, 'sampling_draws_added': 0,
        'attempts': index}, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
