"""Compare complete replay evidence; exit nonzero on any matched-run difference."""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path


def read(path):
    if not path.exists():
        path = path.with_suffix(path.suffix + '.gz')
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(data)


def lines(path):
    if not path.exists():
        path = path.with_suffix(path.suffix + '.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        yield from (json.loads(line) for line in stream)


def difference(left, right, path=''):
    if left == right:
        return None
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return next(d for key in left if (d := difference(left[key], right[key], path + '/' + key)))
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return next(d for i, (a, b) in enumerate(zip(left, right))
                    if (d := difference(a, b, path + '/' + str(i))))
    return dict(path=path, research=left, public=right)


def compare_stream(left, right, name):
    first = None
    mismatches = count = 0
    hashes = [hashlib.sha256(), hashlib.sha256()]
    for count, (a, b) in enumerate(itertools.zip_longest(lines(left / name), lines(right / name)), 1):
        for digest, row in zip(hashes, (a, b)):
            digest.update(json.dumps(row, sort_keys=True).encode())
        if a != b:
            mismatches += 1
            if first is None:
                first = dict(row=count, step=(a or b).get('step'), role=(a or b).get('role'),
                             difference=difference(a, b))
    return dict(rows=count, differing_rows=mismatches, first_difference=first,
                research_sha256=hashes[0].hexdigest(), public_sha256=hashes[1].hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('research', type=Path)
    parser.add_argument('public', type=Path)
    parser.add_argument('--canonical', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    research, public = (read(folder / 'result.json') for folder in (args.research, args.public))
    keys = ('randomness', 'diagnostic', 'continued_hold', 'shift_recovery', 'final', 'ema',
            'optimizer_final', 'rate_ranges')
    checks = {key: research[key] == public[key] for key in keys}
    checks['initial_parameters'] = research['proof']['initial_optimizers'] == public['proof']['initial_optimizers']
    checks['initial_host'] = read(args.research / 'initial-host.json') == read(args.public / 'initial-host.json')
    checks['completed'] = all(x.get('status') != 'ERROR' and x.get('steps') == 3600 for x in (research, public))
    streams = {name: compare_stream(args.research, args.public, name)
               for name in ('updates.jsonl', 'randomness.jsonl')}
    canonical_checks = None
    if args.canonical:
        canonical = read(args.canonical)
        canonical_checks = {key: canonical[key] == research[key] for key in keys}
        canonical_checks['initial_parameters'] = canonical['proof']['initial_optimizers'] == research['proof']['initial_optimizers']
    passed = all(checks.values()) and all(x['differing_rows'] == 0 for x in streams.values())
    passed &= canonical_checks is None or all(canonical_checks.values())
    result = dict(status='EXACT_MATCH' if passed else 'DIFFERENCE', checks=checks,
                  canonical_checks=canonical_checks, streams=streams,
                  scores={name: dict(hold=x['continued_hold']['passing_checks'],
                                     recovery=x['shift_recovery']['deadline_window']['passing_checks'],
                                     final=x['final']) for name, x in [('research', research), ('public', public)]})
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
