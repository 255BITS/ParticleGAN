"""Freeze reviewed candidate source and CPU receipts before CUDA quality."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'quality/ra5'
PACKAGE = ROOT/'pkg-CB64-RA5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_review_inputs(value):
    if isinstance(value, dict):
        for name, item in value.items():
            if isinstance(name, str) and name.startswith('/') and isinstance(item, str) and len(item) == 64:
                assert sha(name) == item, f'Reviewed input changed: {name}'
            verify_review_inputs(item)
    elif isinstance(value, list):
        for item in value:
            verify_review_inputs(item)


def main():
    ready = HERE/'READY.json'
    assert not ready.exists()
    reviews = (HERE/'independent-source/receipt.json',HERE/'integration-contract/receipt.json')
    for path in reviews:
        review = json.loads(path.read_text())
        assert review['status'] == 'PASS', path
        verify_review_inputs(review)
    sources = {str(p.relative_to(PACKAGE/'particlegan')):sha(p) for p in sorted(PACKAGE.rglob('*.py'))}
    composition = json.loads((HERE/'COMPOSITION.json').read_text())
    assert sources == composition['source_sha256'], 'Composed source changed after CPU review'
    assert sha(ROOT/'configs/overrides-CB64-RA5.json') == composition['config_sha256'], 'Config changed after CPU review'
    for name, expected in composition['composed_from'].items():
        assert sha(name) == expected, f'Composition input changed: {name}'
    digest = hashlib.sha256()
    for p in sorted((PACKAGE/'particlegan').rglob('*.py')):
        digest.update(str(p.relative_to(PACKAGE/'particlegan')).encode()+b'\0'+p.read_bytes()+b'\0')
    numerical = {str(p):sha(p) for p in sorted(PACKAGE.rglob('*.py'))}
    numerical.update({str(p):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()})
    inputs = [ROOT/'quality/compose_ra5.py',ROOT/'quality/prepare_lane.py',ROOT/'quality/nested_slot.py',
        Path(__file__),ROOT/'quality/PROTOCOL.md',ROOT/'configs/overrides-CB64-RA5.json',
        *(Path(p) for p in composition['composed_from'])]
    numerical.update({str(p):sha(p) for p in inputs})
    value = dict(status='CPU_VALID_GPU_PENDING',frozen_utc=datetime.now(timezone.utc).isoformat(),
        package_root=str(PACKAGE),package_sha256=digest.hexdigest(),package_source_sha256=sources,
        config_path=str(ROOT/'configs/overrides-CB64-RA5.json'),
        config_sha256=sha(ROOT/'configs/overrides-CB64-RA5.json'),
        numerical_source_sha256=numerical,cpu_reviews={str(p):sha(p) for p in reviews},
        backend_schema=6,trainer_schema=5,
        mechanisms=['separate live/EMA copy geometry from one shared noise draw',
            'bounded paired even-real-anchor latent births within existing certificates and action budget',
            'population continuity validation and one descent undo for replaced rows'],
        matched_counter_scope='ordinary copied matches; novel_birth_moves counts novel replacements separately',
        quality_acceptance='both unchanged final toy and full canonical grid gates required',
        quality_verdict=None,default_package_promoted=False)
    ready.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],package_sha256=value['package_sha256'],ready_sha256=sha(ready))))


if __name__ == '__main__':
    main()
