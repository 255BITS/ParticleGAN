#!/usr/bin/env python3
"""Archive completed search evidence; this does not grade or promote candidates."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def process_live(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def collect(batch, output, lanes=None):
    plan = json.loads((batch / 'batch.json').read_text())
    if lanes:
        unknown = set(lanes) - {row['lane'] for row in plan}
        if unknown:
            raise ValueError(f'Unknown lanes: {sorted(unknown)}')
        plan = [row for row in plan if row['lane'] in lanes]
    # Validate all requested attempts before writing the archive.
    runs = []
    for row in plan:
        statuses = sorted(Path(row['directory']).glob('*/status.txt'))
        if len(statuses) != 1:
            raise ValueError(f"Expected exactly one attempt for {row['lane']}")
        status = statuses[0].read_text().strip()
        if process_live(row['pid']) or status == 'running':
            raise ValueError(f"Attempt is still live: {row['lane']} (pid {row['pid']})")
        runs.append((row, statuses[0].parent.resolve(), status))
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / 'evidence'
    evidence.mkdir(exist_ok=True)
    records, attempts = [], []
    for row, run, status in runs:
        dest = output / 'attempts' / row['lane']
        dest.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in ('result.md', 'final.md', 'tests.jsonl', 'run.txt', 'status.txt', 'exit-code.txt'):
            source = run / name
            if source.exists():
                shutil.copy2(source, dest / name)
                files[name] = sha256(source.read_bytes())
        attempts.append(dict(lane=row['lane'], engine=row['engine'], run=str(run),
                             status=status, files_sha256=files))
        ledger = run / 'tests.jsonl'
        for index, line in enumerate(ledger.read_text().splitlines() if ledger.exists() else []):
            if not line.strip():
                continue
            record = dict(json.loads(line), lane=row['lane'], engine=row['engine'], ledger_row=index + 1)
            artifact = record.get('artifact')
            if isinstance(artifact, str):
                source = Path(artifact)
                if not source.is_absolute():
                    # Ledgers use both attempt-relative and checkout-relative paths.
                    candidates = [run / source, run / 'repo' / source]
                    source = next((p for p in candidates if p.exists()), candidates[0])
                source = source.resolve()
                if source.is_dir() and (source / 'result.json').is_file():
                    source = (source / 'result.json').resolve()
                record['resolved_artifact'] = str(source)
                if not source.is_relative_to(run):
                    record['snapshot_note'] = 'External reference; not copied as this attempt evidence'
                elif source.is_file() and source.suffix == '.json':
                    raw = source.read_bytes()
                    json.loads(raw)  # Do not archive a half-written or malformed JSON result.
                    digest = sha256(raw)
                    target = evidence / (digest + '.json.gz')
                    if not target.exists():
                        target.write_bytes(gzip.compress(raw, mtime=0))
                    assert sha256(gzip.decompress(target.read_bytes())) == digest
                    record.update(snapshot=str(target.relative_to(output)), artifact_sha256=digest)
                elif not source.is_file():
                    record['snapshot_note'] = 'Referenced artifact has no result file; no snapshot claimed'
            records.append(record)
    result = dict(batch=str(batch), observed_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Completed attempt archive; ledger PASS is not formulation qualification',
                  attempts=attempts, ledger_status_counts=dict(Counter(x.get('status') for x in records)),
                  records=records)
    (output / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lane', action='append', help='Archive only named completed lanes')
    args = parser.parse_args()
    result = collect(args.batch.resolve(), args.output.resolve(), args.lane)
    print(json.dumps(dict(attempts=len(result['attempts']), records=len(result['records']),
                          ledger_status_counts=result['ledger_status_counts']), indent=2))


if __name__ == '__main__':
    main()
