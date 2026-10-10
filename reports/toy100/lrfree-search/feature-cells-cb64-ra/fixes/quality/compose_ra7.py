"""Compose the reviewed exact count reduction and a constant rate config."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'pkg-CB64-RA6'
OWNER = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra4-quality/anchor-profile'
PACKAGE = ROOT / 'pkg-CB64-RA7'
AREA = ROOT / 'quality/ra7'
CONFIG = ROOT / 'configs/overrides-CB64-RA7.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not PACKAGE.exists() and not AREA.exists() and not CONFIG.exists()
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
    before = json.loads((ROOT / 'configs/overrides-CB64-RA6.json').read_text())
    assert (before['lr'], before['prior_lr_mult'], before['d_lr_mult']) == (.00425, 2., 1.)
    after = dict(before, lr=.0010625, prior_lr_mult=8., d_lr_mult=4.)
    assert after['lr'] * after['prior_lr_mult'] == before['lr'] * before['prior_lr_mult']
    assert after['lr'] * after['d_lr_mult'] == before['lr'] * before['d_lr_mult']
    (PACKAGE / 'particlegan').mkdir(parents=True)
    for name in sorted(sources):
        target = PACKAGE / 'particlegan' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((proposal / 'particlegan' / name).read_bytes())
    CONFIG.write_text(json.dumps(after, indent=2) + '\n')
    AREA.mkdir()
    proof_paths = [Path(__file__), OWNER / 'READY.json', OWNER / 'FROZEN.json',
        OWNER / 'GROUP-COUNT.patch', OWNER / 'cpu-group-contract.json',
        ROOT / 'integration/review/group-count-gpu/result.json',
        ROOT / 'integration/review/group-count-gpu/PHASE-RESULT.json',
        ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra4-quality/group-gpu-review/receipt.json',
        ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra4-quality/group-gpu-review/FROZEN.json']
    for name in ('result.json', 'PHASE-RESULT.json'):
        assert json.loads((ROOT / 'integration/review/group-count-gpu' / name).read_text())['status'] == 'PASS'
    value = dict(status='COMPOSED_EXACT_COUNTS_AND_QUARTER_G_SIGMA_RATES_CPU_REVIEW_PENDING',
        base_package=str(BASE), base_ready_sha256=sha(ROOT / 'quality/ra6/READY.json'),
        package_root=str(PACKAGE), package_sha256=digest.hexdigest(),
        source_sha256=sources, config_sha256=sha(CONFIG), backend_schema=6, trainer_schema=5,
        composed_from={str(p): sha(p) for p in proof_paths},
        config_changes={key: dict(before=before[key], after=after[key])
            for key in ('lr', 'prior_lr_mult', 'd_lr_mult')},
        hypothesis='Lower constant generator and learned-sigma rates reduce model/table coadaptation while preserving prior and critic base rates.',
        scope='Same RA6 mechanisms plus exact integer count reduction; quarter G and sigma base rates, exact prior/D base rates. Intrinsic clocks remain relative to each group base.',
        unchanged=['noisy live serving and learnable noise formula/floor',
            'original toy/grid fixtures, seeds, horizons and quality gates',
            'birth/copy budgets, count family, solver and population law',
            'backend/trainer schemas and default package'],
        quality_verdict=None)
    (AREA / 'COMPOSITION.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status=value['status'], package_sha256=digest.hexdigest(),
        config_sha256=sha(CONFIG), composition_sha256=sha(AREA / 'COMPOSITION.json'))))


if __name__ == '__main__':
    main()
