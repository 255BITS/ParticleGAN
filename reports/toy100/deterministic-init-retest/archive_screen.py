"""Verify and archive a completed fixed-init screen; never import the learner."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summarize(rows):
    good = lambda row: row['modes'] == 8 and .9 <= row['hq'] <= 1
    first = next((row['step'] for row in rows if good(row)), None)
    after = [row for row in rows if first is not None and row['step'] >= first]
    suffix = []
    for row in reversed(rows):
        if not good(row):
            break
        suffix.append(row)
    return dict(observations=len(rows), passing=sum(map(good, rows)),
                first_arrival=first, observations_since_arrival=len(after),
                passing_since_arrival=sum(map(good, after)),
                departures=[row['step'] for row in after if not good(row)],
                minimum_hq_since_arrival=min((row['hq'] for row in after), default=None),
                minimum_modes_since_arrival=min((row['modes'] for row in after), default=None),
                final_suffix=len(suffix), final_suffix_start=suffix[-1]['step'] if suffix else None,
                final_modes=rows[-1]['modes'] if rows else None,
                final_hq=rows[-1]['hq'] if rows else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--old-comparison', default='NOT_MEASURED_ON_THIS_EXACT_SCREEN')
    parser.add_argument('--eligibility', default='UNQUALIFIED_RETEST_IN_PROGRESS')
    args = parser.parse_args()
    assert args.id and '/' not in args.id and args.id not in ('.', '..')
    source = args.source.resolve()
    hashes = json.loads((source / 'artifact-sha256.json').read_text())
    for name, expected in hashes.items():
        assert Path(name).name == name and sha((source / name).read_bytes()) == expected, name
    result = json.loads((source / 'result.json').read_text())
    declaration = json.loads((source / 'declaration.json').read_text())
    protocol = json.loads((source / 'protocol.json').read_text())
    assert protocol['initializer_commit'] == 'c720645ecae6b648e9fc6034e9d6b48ccff06ed3'
    assert declaration['recipe_overrides']['initialization'] == 'batch_feature_zero'
    with zipfile.ZipFile(source / 'source.zip') as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for name, expected in declaration['package_sha256'].items():
            assert sha(archive.read(name)) == expected, name
        for name, expected in protocol['source_sha256'].items():
            assert sha(archive.read(name)) == expected, name
        assert sha(archive.read('candidate-declaration.json')) == sha((source / 'declaration.json').read_bytes())
    obs_path = source / 'metrics.jsonl'
    rows = [json.loads(line) for line in obs_path.read_text().splitlines()] if obs_path.exists() else []
    if result['status'] in ('PASS', 'FAIL'):
        assert [row['step'] for row in rows] == list(range(50, 1201, 50))
        assert result['metrics']['updates'] == result['metrics']['matched_batch_receipts'] == 1200
    summary = summarize(rows)
    if result['status'] == 'PASS':
        assert summary['final_suffix'] >= 5
    elif result['status'] == 'FAIL':
        assert summary['final_suffix'] < 5
    table_path = HERE / 'screen-results.json'
    table = json.loads(table_path.read_text()) if table_path.exists() else dict(
        schema=1, initializer_commit=protocol['initializer_commit'], results=[])
    assert args.id not in {row['id'] for row in table['results']}, 'already archived'
    target = HERE / 'evidence' / args.id
    target.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    retained_external = {}
    for original in sorted(source.iterdir()):
        if not original.is_file():
            continue
        if original.suffix == '.pt':
            retained_external[original.name] = dict(path=str(original), sha256=sha(original.read_bytes()),
                                                   bytes=original.stat().st_size)
            continue
        raw = original.read_bytes()
        destination = target / (original.name + '.gz' if original.suffix == '.jsonl' else original.name)
        destination.write_bytes(gzip.compress(raw, mtime=0) if original.suffix == '.jsonl' else raw)
        artifacts[str(destination.relative_to(HERE))] = dict(sha256=sha(destination.read_bytes()),
            original_sha256=sha(raw), bytes=destination.stat().st_size)
    entry = dict(id=args.id, candidate=result['candidate'], status=result['status'],
        seconds=result['seconds'], source_directory=str(source), summary=summary,
        old_comparison=args.old_comparison, continuous_eligibility=args.eligibility,
        recipe=declaration['recipe_overrides'], complete=result['status'] in ('PASS', 'FAIL'),
        artifacts=artifacts, raw_checkpoints=retained_external, old_quality_inherited=False)
    write(target / 'archive-manifest.json', entry)
    entry['archive_manifest'] = str((target / 'archive-manifest.json').relative_to(HERE))
    table['results'].append(entry)
    table['recorded_utc'] = datetime.now(timezone.utc).isoformat()
    write(table_path, table)
    lines = ['# New initialization: quick coverage screen', '',
        'All runs use develop’s deterministic network and prior initialization, 1,200 public API updates, and the same 24 observations. A pass requires all eight modes and at least 90% high-quality samples for the final five observations. This screen does not select a release default.', '',
        '| Configuration | New result | Passing observations | First arrival | Final passing streak | Final modes / quality | Old result on this screen |',
        '|---|---|---:|---:|---:|---|---|']
    for row in table['results']:
        s = row['summary']
        quality = f"{s['final_hq']:.1%}" if s['final_hq'] is not None else '—'
        lines.append(f"| [{row['candidate']}]({row['archive_manifest']}) | {row['status']} | {s['passing']}/{s['observations']} | {s['first_arrival'] or '—'} | {s['final_suffix']} | {s['final_modes'] or '—'} / {quality} | {row['old_comparison']} |")
    lines += ['', 'Arrival means the first observation meeting both quality and coverage. All later departures and the complete observations are preserved in each linked record. Scheduled controls retain their declared horizon and noise settings; their quality results do not establish autonomous indefinite operation.', '']
    (HERE / 'leaderboard.md').write_text('\n'.join(lines))
    print(json.dumps(dict(id=args.id, status=result['status'], summary=summary)))


if __name__ == '__main__':
    main()
