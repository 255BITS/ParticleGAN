"""Compose the reviewed finite-fit resolution rule and its one config change."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'pkg-CB64-RA8'
OWNER = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra8-quality/resolution'
PACKAGE = ROOT / 'pkg-CB64-RA9'
AREA = ROOT / 'quality/ra9'
CONFIG = ROOT / 'configs/overrides-CB64-RA9.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not PACKAGE.exists() and not CONFIG.exists()
    assert not (AREA / 'COMPOSITION.json').exists() and not (AREA / 'READY.json').exists()
    if AREA.exists():
        assert all(p.relative_to(AREA).parts[0] in ('integration-contract', 'independent-source')
            for p in AREA.rglob('*') if p.is_file()), 'Unexpected pre-composition artifacts'
    owner = json.loads((OWNER / 'READY.json').read_text())
    for name, expected in owner['numerical_source_sha256'].items():
        assert sha(Path(name)) == expected, name
    proposal = Path(owner['package_root'])
    sources = {str(p.relative_to(proposal / 'particlegan')): sha(p)
        for p in sorted((proposal / 'particlegan').rglob('*.py'))}
    assert sources == owner['package_source_sha256'] and len(sources) == 29
    baseline = {str(p.relative_to(BASE / 'particlegan')): sha(p)
        for p in sorted((BASE / 'particlegan').rglob('*.py'))}
    assert sources.keys() == baseline.keys()
    assert [name for name in sources if sources[name] != baseline[name]] == ['feature_cells.py']
    assert owner['backend_schema'] == 8 and owner['trainer_schema'] == 5
    digest = hashlib.sha256()
    for name in sorted(sources):
        data = (proposal / 'particlegan' / name).read_bytes()
        digest.update(name.encode() + b'\0' + data + b'\0')
    assert digest.hexdigest() == owner['package_sha256']
    original_config = ROOT / 'configs/overrides-CB64-RA8.json'
    original = original_config.read_text()
    assert original.count('"birth_death_cells": 64') == 1
    changed = original.replace('"birth_death_cells": 64', '"birth_death_cells": 128')
    expected_config = json.loads(original)
    expected_config['birth_death_cells'] = 128
    assert json.loads(changed) == expected_config
    assert json.loads(changed) == json.loads((OWNER / 'config.json').read_text())
    proof_paths = [Path(__file__), ROOT / 'quality/RA9-PLAN.md',
        OWNER / 'READY.json', OWNER / 'FROZEN.json', OWNER / 'SOURCE-FROZEN.json']
    proof_hashes = {str(p): sha(p) for p in proof_paths}
    base_ready_hash = sha(ROOT / 'quality/ra8/READY.json')
    (PACKAGE / 'particlegan').mkdir(parents=True)
    for name in sorted(sources):
        target = PACKAGE / 'particlegan' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((proposal / 'particlegan' / name).read_bytes())
    CONFIG.write_text(changed)
    AREA.mkdir(exist_ok=True)
    value = dict(status='COMPOSED_FINITE_FIT_RESOLUTION_CPU_REVIEW_PENDING',
        base_package=str(BASE), base_ready_sha256=base_ready_hash,
        package_root=str(PACKAGE), package_sha256=digest.hexdigest(),
        source_sha256=sources, config_sha256=sha(CONFIG),
        backend_schema=8, trainer_schema=5, composed_from=proof_hashes,
        config_changes={'birth_death_cells': {'before': 64, 'after': 128}},
        hypothesis='Resolve more learned grid supports while retaining the smaller toy chart through an even-real-fit sample cap per effective metric rank.',
        scope='Finite-fit resolution regularization on average; neither a per-cell occupancy guarantee nor a quality certificate. Both original final toy and full grid gates are required.',
        unchanged=['RA8 optimizer bases, adaptive update laws and learned noise formula/floor',
            'serving geometry, expiry and population stationarity law',
            'copy/birth equations, quotas and parent reservations',
            'count family formula 3K+2, using actual fitted K',
            'all original fixtures, seeds, budgets, scorers and quality gates',
            'default package'], quality_verdict=None)
    (AREA / 'COMPOSITION.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status=value['status'], package_sha256=digest.hexdigest(),
        config_sha256=sha(CONFIG), composition_sha256=sha(AREA / 'COMPOSITION.json'))))


if __name__ == '__main__':
    main()
