"""Bind composed public bytes to the closed private source/math and CPU proof."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT / 'integration/review/training-regression/post-ra10-quality/linear-output-production'
MATH = ROOT / 'integration/review/training-regression/post-ra10-quality/linear-output-production-math-review'
PACKAGE = ROOT / 'pkg-CB64-RA11'
CONFIG = ROOT / 'configs/overrides-CB64-RA11.json'
COMPOSITION = ROOT / 'quality/ra11/COMPOSITION.json'
EXPECTED_MATH = 'bc8427d7105eb3dc2579cad82c500e72be372e103624c1091f9a7b3a6777fb52'
EXPECTED_MATH_FREEZE = '1771786ce961510a3005cccf65905452c3201602d4d8ed709636ce6c2b0696e1'
EXPECTED_SOURCE_FREEZE = '1294e397fcb1a59bfb8629c373a12eb866384f0ebcf1344276234bd97b0291f3'
protected = {}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def guard(path, digest=None):
    path = str(Path(path).resolve())
    actual = sha(path)
    if digest is not None and actual != digest:
        raise ValueError('raw binding mismatch: ' + path)
    if path in protected and protected[path] != actual:
        raise ValueError('conflicting binding: ' + path)
    protected[path] = actual
    return actual

def verify_maps(value):
    if isinstance(value, dict):
        for path, item in value.items():
            if isinstance(path, str) and path.startswith('/') and isinstance(item, str) and len(item) == 64:
                guard(path, item)
            verify_maps(item)
    elif isinstance(value, list):
        for item in value:
            verify_maps(item)

def proof(path, digest=None):
    guard(path, digest)
    value = json.loads(Path(path).read_text())
    verify_maps(value)
    return value

def sources(package):
    return {str(p.relative_to(package / 'particlegan')): guard(p)
            for p in sorted((package / 'particlegan').rglob('*.py'))}

def main():
    if (HERE / 'receipt.json').exists():
        raise RuntimeError('closed public review exists; refusing overwrite')
    if not COMPOSITION.exists():
        raise RuntimeError('wait for root composition before invoking this binding gate')
    math = proof(MATH / 'receipt.json', EXPECTED_MATH)
    proof(MATH / 'FROZEN.json', EXPECTED_MATH_FREEZE)
    assert math['status'] == 'PASS' and math['owner_source_freeze_sha256'] == EXPECTED_SOURCE_FREEZE
    source = proof(OWNER / 'SOURCE-FROZEN.json', EXPECTED_SOURCE_FREEZE)
    ready = proof(OWNER / 'READY.json')
    proof(OWNER / 'FROZEN.json')
    composition = proof(COMPOSITION)
    assert ready['status'] == 'FROZEN_CPU_QUALIFIED'
    assert ready['backend_schema'] == composition['backend_schema'] == 10
    assert ready['trainer_schema'] == composition['trainer_schema'] == 5
    cpu_path = Path(ready.get('authoritative_CPU_receipt_path', OWNER / 'CPU-RECEIPT.json'))
    cpu = proof(cpu_path, ready.get('CPU_receipt_sha256'))
    assert cpu['status'] == 'PASS' and cpu.get('post_exit', True)
    assert cpu.get('backend_schema', 10) == 10 and cpu.get('trainer_schema', 5) == 5
    assert protected[str(OWNER / 'SOURCE-FROZEN.json')] == math['owner_source_freeze_sha256']
    proposal = Path(ready['package_root'])
    public_sources, private_sources = sources(PACKAGE), sources(proposal)
    baseline = sources(ROOT / 'pkg-CB64-RA10')
    assert len(public_sources) == 31 and len(baseline) == 30
    assert public_sources == private_sources == ready['package_source_sha256'] == composition['source_sha256']
    changed = sorted(name for name in baseline if public_sources[name] != baseline[name])
    added = sorted(set(public_sources) - set(baseline))
    assert changed == ['feature_cells.py', 'mean_transport.py'] and added == ['output_moments.py']
    assert not set(baseline) - set(public_sources)
    for name, digest in public_sources.items():
        assert math['package_source_sha256'][name] == digest
    digest = hashlib.sha256()
    for name in sorted(public_sources):
        digest.update(name.encode() + b'\0' + (PACKAGE / 'particlegan' / name).read_bytes() + b'\0')
    package_digest = digest.hexdigest()
    assert package_digest == ready['package_sha256'] == source['package_sha256'] == composition['package_sha256']
    config_digest = guard(CONFIG)
    assert config_digest == ready['config_sha256'] == source['config_sha256'] == composition['config_sha256']
    for path in (Path(ready['config_path']), ROOT / 'configs/overrides-CB64-RA9.json', ROOT / 'configs/overrides-CB64-RA10.json'):
        assert guard(path) == config_digest and path.read_bytes() == CONFIG.read_bytes()
    for path in (ROOT / 'quality/RA11-PLAN.md', ROOT / 'quality/results/RA11-selection.json', ROOT / 'quality/results/RA11-selection-provenance.json'):
        if path.suffix == '.json':
            proof(path)
        else:
            guard(path)
    guard(__file__)
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError('binding changed during gate: ' + path)
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(),
        scope='PUBLIC_BYTE_BINDING_TO_CLOSED_PRIVATE_MATH_AND_MECHANICS_PROOF_ONLY',
        backend_schema=10, trainer_schema=5, package_sha256=package_digest,
        package_source_sha256=public_sources, config_sha256=config_digest,
        unchanged_original_modules=28, changed_original_modules=changed, added_modules=added,
        authoritative_private_math_receipt_sha256=EXPECTED_MATH,
        selected_core_AST_preserved_by_exact_source_binding=True,
        private_CPU_receipt_path=str(cpu_path), private_CPU_receipt_sha256=sha(cpu_path),
        protected_sha256=protected, reviewer_Torch_PT_array_imports=0,
        reviewer_numerical_execution=False, quality_verdict=None)
    (HERE / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', package_sha256=package_digest,
        protected_files=len(protected), receipt_sha256=sha(HERE / 'receipt.json'))))

if __name__ == '__main__':
    main()
