"""Verify retained replay evidence without PyTorch or training."""
import hashlib
import json
from pathlib import Path

from compare import difference, lines, read


root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
for entry in manifest['files']:
    data = (root / entry['path']).read_bytes()
    assert len(data) == entry['bytes'], entry['path']
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']

research = root / 'results/ka2-research'
old = root / 'results/ka2-public-before-fix'
before = json.loads((root / 'before-fix-comparison.json').read_text())
left, right = list(lines(research / 'updates.jsonl')), list(lines(old / 'updates.jsonl'))
first = {}
for key in ('controller', 'before', 'after', 'loss_sha256'):
    for a, b in zip(left, right):
        if a[key] != b[key]:
            first[key] = dict(step=a['step'], role=a['role'], difference=difference(a[key], b[key]))
            break
assert first == before['first_numeric_differences']


def training_draws(folder):
    return [{k: v for k, v in row.items() if k != 'call'}
            for row in lines(folder / 'randomness.jsonl') if row['shape'][0] == 128]


a, b = training_draws(research), training_draws(old)
assert a[:len(b)] == b and len(b) == before['training_draws'] == 10318
assert json.loads((research / 'initial-host.json').read_text()) == json.loads((old / 'initial-host.json').read_text())
for name in ('k3p-research', 'k3p-public', 'ka2-research', 'ka2-public', 'ka2-public-before-fix'):
    result = read(root / 'results' / name / 'result.json')
    worker = 'worker-k3p.py' if name.startswith('k3p-') else 'worker.py'
    assert result['worker_sha256'] == hashlib.sha256((root / worker).read_bytes()).hexdigest()
print(f"PASS: {len(manifest['files'])} artifact hashes, worker identities and first pre-fix divergence at update 831.")
