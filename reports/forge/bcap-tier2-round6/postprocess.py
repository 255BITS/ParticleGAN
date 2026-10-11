"""Conclude and archive saved round-six evidence; never train or sample."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.knowledge import readout

ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-tier2-round6-20261011')


def preserve(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert file_hash(source) == file_hash(target), f'Archive collision: {target}'
    else:
        shutil.copy2(source, target)


def conclude(archive, results):
    records = []
    for role, counts in results['arm_counts'].items():
        candidate = f'bcap-tier2-round6-{role.replace("_", "-")}-v1'
        existing = [p for p in (ROOT / 'reports/forge/records').glob('readout-*.json')
                    if read_json(p).get('candidate_id') == candidate]
        if existing:
            assert len(existing) == 1 and read_json(existing[0]).get('summary_curation')
            records.append(str(existing[0].relative_to(ROOT)))
            continue
        tier1 = counts['1'].get('PASS', 0)
        tier2 = counts['2'].get('PASS', 0)
        complete = sum(counts['2'].get(s, 0) for s in ('PASS', 'FAIL')) == 21
        conclusion = f'{tier1}/6 required Tier1 passes; '
        conclusion += f'{tier2}/21 Tier2 passes.' if complete else 'all21 Tier2 cells blocked and unmeasured.'
        if role == 'control':
            comparison = 'Fresh reference reproduces9/21; no challenger satisfies the registered global win rule.'
        elif role == 'zero_damping':
            comparison = 'Fresh control9/21 versus hybrid8/21; no added pass and own-prefix word hold regresses. Word prefixes834vs1667 leave equal complete-stream consumption unverified.'
        else:
            comparison = 'Fails required Tier1; Tier2 is unmeasured, not a numerical regression. Fresh control passes6/6 and9/21.'
        record = readout(ROOT, candidate, conclusion, comparison,
                         'Retain current BCAP control; stop this fixed revision. Preserve local improvements and unavailable forecasts without promotion or unchanged reruns.',
                         study_id=candidate + '-study')
        path = ROOT / 'reports/forge/records' / (record['record_id'] + '.json')
        original = archive / 'original-readouts' / path.name
        preserve(path, original)
        record['original_source'] = record['source']
        record['source'] = {'path': 'reports/forge/bcap-tier2-round6/results.json'}
        record['task_results'] = [dict(r, _attempt_id=r['attempt_id'])
                                  for r in results['task_results'] if r['role'] == role]
        record['summary_curation'] = 'Compact numerical results retain original gates, costs, receipt hashes and source identity; bulk traces are archived.'
        record['original_readout'] = {'path': str(original), 'sha256': file_hash(original)}
        atomic_json(path, record)
        records.append(str(path.relative_to(ROOT)))
        print(json.dumps({'event': 'concluded', 'role': role, 'record': records[-1]}), flush=True)
    atomic_json(OUT / 'readouts.json', {'records': records})


def archive_receipt(archive, results):
    for attempt in results['paid_attempt_history']:
        aid = attempt['attempt_id']
        source_root = ROOT / 'reports/forge/attempts' / aid
        assert all((source_root / name).is_file() for name in ('request.json', 'result.json', 'evidence.json'))
        for source in source_root.rglob('*'):
            if source.is_file():
                preserve(source, archive / 'attempt-envelopes' / aid / source.relative_to(source_root))
    members = {}
    for folder in ('queue', 'attempt-envelopes', 'original-readouts'):
        for path in sorted((archive / folder).rglob('*')):
            if path.is_file() and path.name not in {'queue.lock', 'state.lock'}:
                members[str(path.relative_to(archive))] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    for name in ('executed-terminal-queue-state.json', 'executed-progress.json',
                 'logs/driver.log', 'logs/status.log', 'logs/publish.log', 'logs/readout.log'):
        path = archive / name
        assert path.is_file()
        members[name] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    manifest = archive / 'original-files.json'
    atomic_json(manifest, {'files': members})
    atomic_json(OUT / 'archive.json', {
        'schema_version': 1, 'root': str(archive), 'source_commit': results['source_commit'],
        'source_digest': results['source_digest'], 'original_file_manifest': {
            'path': str(manifest), 'sha256': file_hash(manifest), 'files': len(members),
            'bytes': sum(v['bytes'] for v in members.values())},
        'paid_attempts': len(results['paid_attempt_history']),
        'executed_terminal_queue_state_sha256': file_hash(archive / 'executed-terminal-queue-state.json'),
        'restoration': 'Verify original-files.json and its members; restore attempt-envelopes under reports/forge/attempts and external queue artifacts at their recorded absolute root. Use executed-terminal-queue-state.json as queue/queue/state.json for the pre-readout publication. Check out the executed source commit for run.py publish; no training rerun is required.',
        'bulk_in_git': False})
    print(json.dumps({'event': 'archived', 'files': len(members)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('conclude', 'archive'))
    parser.add_argument('--artifacts', type=Path, default=ARCHIVE)
    options = parser.parse_args()
    progress, results = read_json(options.artifacts / 'progress.json'), read_json(OUT / 'results.json')
    assert progress['phase'] == 'complete' and results['source_commit'] == progress['source_commit']
    assert results['source_digest'] == progress['source_digest']
    assert not options.artifacts.resolve().is_relative_to(ROOT)
    globals()[options.action if options.action == 'conclude' else 'archive_receipt'](options.artifacts, results)


if __name__ == '__main__':
    main()
