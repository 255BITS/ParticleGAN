"""Package completed training evidence; preserve originals and avoid tensor copies."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/ml2/hypergan/particlegan-selective-history-20261015')
OUT = REPO / 'reports/forge/selective-history'


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert sha(source) == sha(destination)


def main():
    stages = {'initial': ROOT, 'repairs': ROOT / 'repairs',
              'final-critic': ROOT / 'final-critic-screen',
              'ordinary-tier1': ROOT / 'ordinary-tier1'}
    completions = []
    for name, stage in stages.items():
        completion = json.loads((stage / 'completion.json').read_text())
        assert completion['status'] == 'COMPLETE', name
        completions.append(completion)
        for filename in ('completion.json', 'independent-result-audit.json',
                         'independent-admission-audit.json', 'INDEPENDENT-AUDIT.md',
                         'launch-manifest.json', 'selection.json',
                         'word-execution-repair-plan.json', 'word-execution-repair-transition.json',
                         'word-execution-repair-disposition.json',
                         'resource-observations.jsonl', 'word-progress-observations.jsonl'):
            source = stage / filename
            if source.is_file():
                copy(source, OUT / 'training-evidence' / name / filename)
    for filename in ('authorization.json', 'planned-roster.json', 'coordinator-transition.json',
                     'historical-repair-counterevidence.json', 'optimizer-state-analysis.json',
                     'source-and-frozen-file-verification.json', 'executed-baseline-recipe.json',
                     'independent-media-audit.json'):
        copy(ROOT / filename, OUT / 'training-evidence' / filename)
    copy(ROOT / 'TRAINING-REPOSITORY-DRAFT.md', OUT / 'TRAINING.md')
    copy(ROOT / 'training-summary.json', OUT / 'training-summary.json')
    figure_sources = []
    for filename in ('training-curves.png', 'training-curves.svg',
                     'repair-training-curves.png', 'repair-training-curves.svg'):
        source, target = ROOT / filename, OUT / 'training-media' / filename
        copy(source, target)
        if target.suffix == '.svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
        figure_sources.append({'file': filename, 'original_sha256': sha(source),
                               'packaged_sha256': sha(target),
                               'normalization': 'trailing line whitespace removed' if target.suffix == '.svg' else 'none'})
    (OUT / 'training-media/figure-sources.json').write_text(json.dumps(figure_sources, indent=2) + '\n')
    for filename in ('analyze_curves.py', 'analyze_optimizer_states.py', 'plot_curves.py',
                     'plot_repair_curves.py', 'render_media.py', 'launch.py',
                     'resume_without_compile.py', 'admit_and_run_repairs.py',
                     'run_registered_campaign.py', 'finalize_execution.py',
                     'finalize_ordinary.py', 'retry_word_after_contention.py',
                     'independent_result_audit.py', 'audit_media.py', 'summarize_training.py', 'package_readout.py'):
        copy(ROOT / filename, OUT / 'training-repro' / filename)
    for name, stage in stages.items():
        for filename in ('independent_result_audit.py', 'independent_audit.py'):
            if (stage / filename).is_file():
                copy(stage / filename, OUT / 'training-repro' / name / filename)
    source_index = json.loads((ROOT / 'media/index.json').read_text())
    index = {key: value for key, value in source_index.items() if key != 'attempts'}
    index['original_index_sha256'] = sha(ROOT / 'media/index.json')
    index['attempts'] = []
    checkpoints = []
    for row in source_index['attempts']:
        item = dict(row)
        source_dir = Path(row['output_directory'])
        destination = OUT / 'training-media' / row['candidate'] / row['attempt_id']
        item['output_directory'] = destination.relative_to(OUT).as_posix()
        for media in row['media']:
            task = media['task_id']
            for suffix in ('.gif', '.json'):
                copy(source_dir / (task + suffix), destination / (task + suffix))
            assert sha(destination / (task + '.gif')) == media['gif_sha256']
        index['attempts'].append(item)
    expected_attempts = sum(item['scientific_attempts'] for item in completions)
    expected = {row['attempt_id']: (completion, row) for completion in completions
                for row in completion['jobs']}
    assert len(expected) == expected_attempts, (len(expected), expected_attempts)
    exported = {row['attempt_id'] for row in index['attempts'] if row['media']}
    assert exported <= set(expected)
    index['attempts_without_media'] = [
        {'attempt_id': aid, 'reason': 'No exported certified observation media; see original result.'}
        for aid in sorted(set(expected) - exported)]
    for aid, (completion, row) in expected.items():
        attempt = ROOT / 'queue' / completion['campaign'] / aid
        checkpoints.append({'candidate': row['candidate'], 'attempt_id': aid,
                            'campaign': completion['campaign'], 'external_directory': str(attempt),
                            'files': [{'path': str(path), 'bytes': path.stat().st_size,
                                       'sha256': sha(path)} for path in sorted(attempt.rglob('*.pt'))]})
    (OUT / 'training-media/index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    (OUT / 'training-evidence/checkpoints.json').write_text(json.dumps({
        'kind': 'external_original_tensor_artifacts', 'copied_into_git': False,
        'note': 'Original saved states and observations remain at these local paths. '
                'Checkpoint suitability is governed by each task and its saved payload; '
                'this index does not assert arbitrary continuation or checkpoint portability.',
        'attempts': checkpoints}, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'attempts': len(index['attempts']),
                      'gifs': sum(len(row['media']) for row in index['attempts']),
                      'output': str(OUT)}))


if __name__ == '__main__':
    main()
