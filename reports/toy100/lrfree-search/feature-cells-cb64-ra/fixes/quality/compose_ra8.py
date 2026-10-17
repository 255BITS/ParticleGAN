"""Compose a reviewed paired-geometry serving rule with RA7's exact config."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'pkg-CB64-RA7'
OWNER = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra7-quality/paired-average'
PACKAGE = ROOT / 'pkg-CB64-RA8'
AREA = ROOT / 'quality/ra8'
CONFIG = ROOT / 'configs/overrides-CB64-RA8.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not PACKAGE.exists() and not CONFIG.exists()
    assert not (AREA / 'COMPOSITION.json').exists() and not (AREA / 'READY.json').exists()
    if AREA.exists():
        assert all(str(p.relative_to(AREA)).startswith('integration-contract/')
            for p in AREA.rglob('*') if p.is_file()), 'Unexpected pre-composition artifacts'
    owner = json.loads((OWNER / 'READY.json').read_text())
    for name, expected in owner['numerical_source_sha256'].items():
        assert sha(Path(name)) == expected, name
    proposal = Path(owner['package_root'])
    sources = {str(p.relative_to(proposal / 'particlegan')): sha(p)
        for p in sorted((proposal / 'particlegan').rglob('*.py'))}
    assert sources == owner['package_source_sha256']
    digest = hashlib.sha256()
    for name in sorted(sources):
        data = (proposal / 'particlegan' / name).read_bytes()
        digest.update(name.encode() + b'\0' + data + b'\0')
    assert digest.hexdigest() == owner['package_sha256']
    (PACKAGE / 'particlegan').mkdir(parents=True)
    for name in sorted(sources):
        target = PACKAGE / 'particlegan' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((proposal / 'particlegan' / name).read_bytes())
    original_config = ROOT / 'configs/overrides-CB64-RA7.json'
    CONFIG.write_bytes(original_config.read_bytes())
    assert sha(CONFIG) == sha(original_config)
    AREA.mkdir(exist_ok=True)
    proof_paths = [Path(__file__), ROOT / 'quality/RA8-PLAN.md',
        OWNER / 'READY.json', OWNER / 'FROZEN.json']
    value = dict(status='COMPOSED_PAIRED_GEOMETRY_AVERAGE_CPU_REVIEW_PENDING',
        base_package=str(BASE), base_ready_sha256=sha(ROOT / 'quality/ra7/READY.json'),
        package_root=str(PACKAGE), package_sha256=digest.hexdigest(),
        source_sha256=sources, config_sha256=sha(CONFIG),
        backend_schema=owner['backend_schema'], trainer_schema=owner['trainer_schema'],
        composed_from={str(p): sha(p) for p in proof_paths}, config_changes={},
        hypothesis='Permit averaging only while current EMA anchors remain in learned support and agree with corresponding live rows on real-only support regions.',
        scope='Empirical geometry for averaging; not a distribution-equivalence, stationarity, or quality certificate. All original emitted quality gates remain required.',
        unchanged=['RA7 config and adaptive optimizer laws',
            'live training trajectory, copy/birth actions and random draws',
            'population stationarity requirement and action count family',
            'learnable noise formula, floor and generation API',
            'original toy/grid fixtures, seeds, budgets and gates',
            'reference fallback for small populations and default package'],
        quality_verdict=None)
    (AREA / 'COMPOSITION.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status=value['status'], package_sha256=digest.hexdigest(),
        config_sha256=sha(CONFIG), composition_sha256=sha(AREA / 'COMPOSITION.json'))))


if __name__ == '__main__':
    main()
