"""Reproduce compact PR399 receipts from immutable original Git objects.

Run from any directory after fetching the original PR head:
    git fetch origin e71cd6ea13f841092ed346b3217ecee0c4d53d29
    python reports/forge/reduce_pr399_evidence.py

This reduces publications only; it does not train, evaluate, or alter gates.
"""
import json, hashlib, subprocess, copy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = 'e71cd6ea13f841092ed346b3217ecee0c4d53d29'
BASE = 'reports/forge/test-performance-evidence/'
ORIGINAL_SHA256 = {'successor-nonatlas-21': '087d367dd29d8b13b285d6df7d0b55a214692533a4c4dfa5e2a3a3b890984562', 'atlas-seed0-diagnostics-3': '0fde5927e4225eaba30aa25ff034f5a49166203f7a544b2528373ed267c9237a', 'atlas-dynamics-6': '4e4ccfdf6749c9e596e84d72b0673c267062f14b22cf5bf62493bbf8c3686a0c', 'pr399-paired-tier1-12': 'e4d409d36acc5dc6b989c4f1007e5d67d8268a5db9117ec93a6839dcfb439f82'}

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
index = json.loads((ROOT / (BASE + 'index.json')).read_text())
pins = []
for name in ['successor-nonatlas-21', 'atlas-seed0-diagnostics-3', 'pr399-paired-tier1-12', 'atlas-dynamics-6']:
    original_path = BASE + name + '.json'
    raw = subprocess.check_output(['git', 'show', ARCHIVE + ':' + original_path], cwd=ROOT)
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA256[name]:
        raise ValueError('original archive bytes changed: ' + name)
    d = json.loads(raw)
    pin = {'commit': ARCHIVE, 'path': original_path, 'git_blob': subprocess.check_output(['git', 'rev-parse', ARCHIVE + ':' + original_path], cwd=ROOT).decode().strip(), 'sha256': hashlib.sha256(raw).hexdigest(), 'url': 'https://github.com/255BITS/ParticleGAN/blob/' + ARCHIVE + '/' + original_path}
    d['archived_publication'] = pin
    d['reduction'] = 'Only repeated task field-ownership prose and per-observation dynamics curves are archived by hash; all recipe, execution, gate, RNG, final-grade, sample/state and cost identities remain. Original verdicts are unchanged; no qualification reuse.'
    conditions = {}
    mapping = {}
    for (old, co) in d['condition_contracts'].items():
        assert digest(co) == old
        co = copy.deepcopy(co)
        co['archived_conditions_sha256'] = old
        removed = co['task'].pop('field_ownership', None)
        if removed is not None:
            co['task']['archived_field_ownership_sha256'] = digest(removed)
        new = digest(co)
        conditions[new] = co
        mapping[old] = new
    d['condition_contracts'] = conditions
    for c in d['cells']:
        c['archived_cell_sha256'] = digest(c)
        c['archived_conditions_sha256'] = c['conditions_sha256']
        c['conditions_sha256'] = mapping[c['conditions_sha256']]
        diag = c['result'].get('saved_sample_diagnostic')
        if isinstance(diag, dict):
            for key in ['observations', 'curve', 'rows']:
                values = diag.pop(key, None)
                if values is not None:
                    diag['archived_' + key + '_sha256'] = digest(values)
                    diag['archived_' + key + '_count'] = len(values)
        c['result']['archived_full_cell_url'] = pin['url']
    target = BASE + 'pr399-retained-' + name + '.json'
    data = (json.dumps(d, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    (ROOT / target).write_bytes(data)
    newpin = {'path': target, 'sha256': hashlib.sha256(data).hexdigest()}
    pins.append(newpin)
    index['publications'] = [p for p in index['publications'] if p['path'] != target] + [newpin]
    print(target, len(data), newpin['sha256'])
(ROOT / (BASE + 'index.json')).write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
