"""Preserve one completed declared API evaluation and derive arrival/stability.

Example: python archive_run.py --id api-c3-single --source /path/to/C3-single \
    --protocol single_shift
No training, imports from a learner, or changes to original evidence.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def good(point):
    return point['modes'] == 8 and .9 <= point['hq'] <= 1


def segment(rows, start, end):
    points = [p for p in rows if start < p['step'] <= end]
    first = next((p['step'] for p in points if good(p)), None)
    after = [p for p in points if first is not None and p['step'] >= first]
    suffix = []
    for point in reversed(points):
        if not good(point):
            break
        suffix.append(point)
    return dict(start=start, end=end, first_arrival=first,
                arrival_delay=None if first is None else first - start,
                passing_since_arrival=sum(map(good, after)),
                observations_since_arrival=len(after),
                all_departures=[p['step'] for p in after if not good(p)],
                minimum_hq_since_arrival=min((p['hq'] for p in after), default=None),
                minimum_modes_since_arrival=min((p['modes'] for p in after), default=None),
                final_suffix_start=suffix[-1]['step'] if suffix else None,
                final_suffix_observations=len(suffix))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--protocol', required=True,
                        choices=['single_shift', 'stationary', 'delayed_repeated', 'long_continuation'])
    parser.add_argument('--scope', default='PUBLIC_API')
    args = parser.parse_args()
    source = args.source.resolve()
    receipt_path = HERE / 'first-results.json'
    receipt = json.loads(receipt_path.read_text())
    if not args.id or '/' in args.id or args.id in {'.', '..'}:
        parser.error('id must be a simple directory name')
    if any(r['id'] == args.id for r in receipt['results']):
        parser.error('already archived; preserve the existing receipt')
    protocols = HERE.parent / 'continuous-eligibility/launch/evaluation-protocols.json'
    protocol = json.loads(protocols.read_text())['protocols'][args.protocol]
    end = protocol['total_updates']
    rows = [json.loads(line) for line in (source / 'metrics.jsonl').read_text().splitlines()]
    assert [p['step'] for p in rows] == list(range(10, end + 1, 10)), 'incomplete/duplicate observations'
    declaration = json.loads((source / 'declaration.json').read_text())
    with zipfile.ZipFile(source / 'source.zip') as archive:
        for name, expected in declaration['source_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    destination = HERE / 'evidence' / args.id
    destination.mkdir(exist_ok=False)
    artifacts = {}
    for name in ['source.zip', 'declaration.json', 'initial.json', 'result.json',
                 'summary.json', 'assessment.json', 'metrics.jsonl',
                 'learning-rates.jsonl', 'state-hashes.jsonl']:
        original = source / name
        if not original.exists():
            continue
        target = destination / (name + '.gz' if name.endswith('.jsonl') else name)
        if name.endswith('.jsonl'):
            target.write_bytes(gzip.compress(original.read_bytes(), mtime=0))
        else:
            shutil.copyfile(original, target)
        artifacts[str(target.relative_to(HERE))] = dict(
            sha256=sha(target), original_sha256=sha(original), original=str(original),
            bytes=target.stat().st_size)
    bounds = [0] + [change['after_update'] for change in protocol['target_changes']] + [end]
    prehold = [p for p in rows if 1210 <= p['step'] <= 2400]
    entry = dict(id=args.id, scope=args.scope, source_directory=str(source),
                 observed_through=end, complete_declared_evaluation=True,
                 protocol=args.protocol, protocol_sha256=sha(protocols),
                 segments=[segment(rows, a, b) for a, b in zip(bounds, bounds[1:])],
                 historical_prehold_window=dict(passing=sum(map(good, prehold)),
                     observations=len(prehold),
                     note='Not an acquisition deadline; use retention since actual arrival.'),
                 artifacts=artifacts)
    receipt['results'].append(entry)
    receipt['recorded_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = receipt_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(receipt_path)
    print(json.dumps({args.id: [{k: v for k, v in s.items() if k != 'all_departures'}
                              for s in entry['segments']]}))


if __name__ == '__main__':
    main()
