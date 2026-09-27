"""Check archived API/research bytes and lossless compression without training."""
import collections
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ledgers = {}
    checked = {}
    for name in ('screen-results.json', 'research-results.json'):
        path = ROOT / name
        data = json.loads(path.read_text())
        rows = data['results']
        assert len({row['id'] for row in rows}) == len(rows), name
        for row in rows:
            for relative, receipt in row['artifacts'].items():
                artifact = ROOT / relative
                assert artifact.resolve().is_relative_to(ROOT), relative
                raw = artifact.read_bytes()
                assert digest(raw) == receipt['sha256'], relative
                assert len(raw) == receipt['bytes'], relative
                if receipt['original_sha256'] != receipt['sha256']:
                    assert artifact.suffix == '.gz', relative
                    assert digest(gzip.decompress(raw)) == receipt['original_sha256'], relative
                checked[relative] = receipt['sha256']
            if row.get('audit'):
                assert digest(Path(row['audit']).read_bytes()) == row['audit_sha256'], row['id']
            if row.get('archive_manifest'):
                archived = json.loads((ROOT / row['archive_manifest']).read_text())
                for field in ('id', 'candidate', 'status', 'summary', 'artifacts'):
                    assert archived[field] == row[field], (row['id'], field)
        ledgers[name] = dict(sha256=digest(path.read_bytes()), rows=len(rows),
                            status_counts=dict(collections.Counter(row['status'] for row in rows)))
    result = dict(status='PASS', recorded_utc=datetime.now(timezone.utc).isoformat(),
                  ledgers=ledgers, unique_archived_files=len(checked),
                  checks=['Unique IDs within each ledger', 'Archived byte lengths and SHA256',
                          'Lossless gzip original SHA256', 'Research audit hashes',
                          'Archive manifest agrees with ledger'],
                  limits=['Does not rerun training or replace the independent runtime audits',
                          'External raw checkpoint bytes were checked by their runtime audits; this check covers committed archives',
                          'Follow-up image/vector/ring evidence has its own independent audit'])
    (ROOT / 'archive-integrity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
