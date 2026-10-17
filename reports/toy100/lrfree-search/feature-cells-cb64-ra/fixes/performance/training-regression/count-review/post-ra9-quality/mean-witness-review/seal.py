"""Post-exit seal of the source-only review, including retained private failure."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    target = HERE / 'FROZEN.json'
    assert not target.exists()
    receipt = json.loads((HERE / 'receipt.json').read_text())
    bundle = json.loads((HERE / 'BUNDLE-RECEIPT.json').read_text())
    assert receipt['status'] == bundle['status'] == 'PASS'
    assert digest(HERE / 'receipt.json') == bundle['mathematical_receipt_sha256']
    for path, expected in bundle['input_sha256'].items():
        assert digest(path) == expected, path
    frozen = dict(
        status='PASS', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        post_exit=True, first_private_failure_preserved=True,
        receipt_sha256=digest(HERE / 'receipt.json'),
        authoritative_bundle_receipt_sha256=digest(HERE / 'BUNDLE-RECEIPT.json'),
        report_sha256=digest(HERE / 'REPORT.md'),
        source_preseal_sha256=bundle['source_preseal_sha256'],
        input_sha256=bundle['input_sha256'],
        local_sha256={str(p): digest(p) for p in sorted(HERE.iterdir()) if p.is_file()})
    target.write_text(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=frozen['status'], FROZEN_sha256=digest(target),
                         receipt_sha256=frozen['receipt_sha256'],
                         authoritative_bundle_receipt_sha256=frozen['authoritative_bundle_receipt_sha256']), sort_keys=True))


if __name__ == '__main__':
    main()
