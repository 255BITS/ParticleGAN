"""Compose the reviewed bounded mean phase with RA9's exact configuration."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'pkg-CB64-RA9'
OWNER = ROOT / 'integration/review/training-regression/post-ra9-quality/mean-category-production'
PACKAGE = ROOT / 'pkg-CB64-RA10'
AREA = ROOT / 'quality/ra10'
CONFIG = ROOT / 'configs/overrides-CB64-RA10.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_maps(value):
    if isinstance(value, dict):
        for name, item in value.items():
            if isinstance(name, str) and name.startswith('/') and isinstance(item, str) and len(item) == 64:
                assert sha(name) == item, name
            verify_maps(item)
    elif isinstance(value, list):
        for item in value:
            verify_maps(item)


def sources(package):
    return {str(path.relative_to(package / 'particlegan')): sha(path)
            for path in sorted((package / 'particlegan').rglob('*.py'))}


def main():
    assert not PACKAGE.exists() and not CONFIG.exists()
    assert not (AREA / 'COMPOSITION.json').exists() and not (AREA / 'READY.json').exists()
    if AREA.exists():
        assert all(path.relative_to(AREA).parts[0] in ('integration-contract', 'independent-source')
                   for path in AREA.rglob('*') if path.is_file())
    selection_path = ROOT / 'quality/results/RA10-selection.json'
    selection = json.loads(selection_path.read_text())
    assert selection['status'] == 'PROSPECTIVE_MEAN_COPY_CANDIDATE_SELECTED'
    verify_maps(selection)
    owner = json.loads((OWNER / 'READY.json').read_text())
    assert owner['status'] == 'FROZEN_CPU_QUALIFIED'
    assert owner['backend_schema'] == 9 and owner['trainer_schema'] == 5
    verify_maps(owner)
    proposal = Path(owner['package_root'])
    proposal_sources, baseline = sources(proposal), sources(BASE)
    assert proposal_sources == owner['package_source_sha256']
    assert len(baseline) == 29 and len(proposal_sources) == 30
    assert set(proposal_sources) - set(baseline) == {'mean_transport.py'}
    assert not set(baseline) - set(proposal_sources)
    changed = [name for name in baseline if proposal_sources[name] != baseline[name]]
    assert changed == ['birth_phase.py', 'feature_cells.py'], changed
    original_config = ROOT / 'configs/overrides-CB64-RA9.json'
    assert Path(owner['config_path']).read_bytes() == original_config.read_bytes()
    assert owner['config_sha256'] == sha(original_config)
    assert sha(original_config) == 'b3656ea7494413106484c556e53877900dd5ccebf2597b3f80b7d0e7d5ba4437'
    digest = hashlib.sha256()
    for name in sorted(proposal_sources):
        data = (proposal / 'particlegan' / name).read_bytes()
        digest.update(name.encode() + b'\0' + data + b'\0')
    assert digest.hexdigest() == owner['package_sha256']
    proofs = [Path(__file__), ROOT / 'quality/RA10-PLAN.md', selection_path,
              OWNER / 'READY.json', OWNER / 'FROZEN.json', OWNER / 'SOURCE-FROZEN.json']
    for path in proofs[2:]:
        verify_maps(json.loads(path.read_text()))
    proof_hashes = {str(path): sha(path) for path in proofs}
    base_ready_hash = sha(ROOT / 'quality/ra9/READY.json')
    (PACKAGE / 'particlegan').mkdir(parents=True)
    for name in sorted(proposal_sources):
        target = PACKAGE / 'particlegan' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((proposal / 'particlegan' / name).read_bytes())
    CONFIG.write_bytes(original_config.read_bytes())
    AREA.mkdir(exist_ok=True)
    value = dict(status='COMPOSED_BOUNDED_MEAN_COPY_CPU_REVIEW_PENDING',
        base_package=str(BASE), base_ready_sha256=base_ready_hash,
        package_root=str(PACKAGE), package_sha256=digest.hexdigest(),
        source_sha256=proposal_sources, config_sha256=sha(CONFIG),
        backend_schema=9, trainer_schema=5, composed_from=proof_hashes,
        config_changes={}, changed_original_modules=changed, added_modules=['mean_transport.py'],
        hypothesis='Repair persistent conditional mean discrepancy using a fixed bounded witness and actual category-preserving paired copy progress within the original ordinary budget.',
        scope='Empirical negative-evidence trigger and feature mean objective; no population, per-group or emitted quality certificate. Both unchanged final CUDA toy and full canonical grid gates remain required.',
        unchanged=['exact RA9 configuration bytes, finite-fit cell cap and rank/pool bounds',
            'all optimizer bases and adaptive update laws, learned noise formula/floor',
            'population participation and reset law, empirical serving geometry and expiry',
            'existing birth/copy equations, legacy quota and reservation policies',
            'all other27 original Python modules, including training.py',
            'all original fixtures, seeds, budgets, scorers and quality gates',
            'main/default package'], quality_verdict=None, default_package_promoted=False)
    (AREA / 'COMPOSITION.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status=value['status'], package_sha256=digest.hexdigest(),
        config_sha256=sha(CONFIG), composition_sha256=sha(AREA / 'COMPOSITION.json'))))


if __name__ == '__main__':
    main()
