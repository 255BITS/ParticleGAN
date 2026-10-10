"""Freeze an independently reviewed candidate and its focused CPU contracts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_files(value):
    if isinstance(value, dict):
        for name,item in value.items():
            if isinstance(name,str) and name.startswith('/') and isinstance(item,str) and len(item)==64:
                assert sha(name)==item, f'Reviewed input changed: {name}'
            verify_files(item)
    elif isinstance(value,list):
        for item in value:
            verify_files(item)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',required=True)
    parser.add_argument('--area',type=Path,required=True)
    args=parser.parse_args()
    assert re.fullmatch('CB64-[A-Z0-9_-]+',args.variant)
    area=args.area.resolve()
    assert area.is_relative_to(ROOT/'quality')
    ready=area/'READY.json'
    assert not ready.exists()
    package=ROOT/('pkg-'+args.variant)
    config=ROOT/'configs'/('overrides-'+args.variant+'.json')
    reviews=[area/'independent-source/receipt.json',area/'integration-contract/receipt.json']
    for path in reviews:
        review=json.loads(path.read_text())
        assert review['status']=='PASS',path
        verify_files(review)
    composition=json.loads((area/'COMPOSITION.json').read_text())
    sources={str(p.relative_to(package/'particlegan')):sha(p) for p in sorted(package.rglob('*.py'))}
    assert sources==composition['source_sha256']
    assert sha(config)==composition['config_sha256']
    inputs=[Path(__file__),ROOT/'quality/prepare_lane.py',ROOT/'quality/nested_slot.py',ROOT/'quality/PROTOCOL.md',config]
    if 'base_ready_sha256' in composition:
        base=Path(composition['base_package'])
        base_area=ROOT/'quality'/base.name.removeprefix('pkg-CB64-').lower()
        base_ready=base_area/'READY.json'
        assert sha(base_ready)==composition['base_ready_sha256']
        inputs.append(base_ready)
        verify_files(json.loads(base_ready.read_text()))
    for name,expected in composition.get('composed_from',{}).items():
        assert sha(name)==expected
        inputs.append(Path(name))
    digest=hashlib.sha256()
    for p in sorted((package/'particlegan').rglob('*.py')):
        digest.update(str(p.relative_to(package/'particlegan')).encode()+b'\0'+p.read_bytes()+b'\0')
    numerical={str(p):sha(p) for p in sorted(package.rglob('*.py'))}
    numerical.update({str(p):sha(p) for p in sorted(area.rglob('*')) if p.is_file()})
    numerical.update({str(p):sha(p) for p in inputs})
    value=dict(status='CPU_VALID_GPU_PENDING',variant=args.variant,
        frozen_utc=datetime.now(timezone.utc).isoformat(),package_root=str(package),
        package_sha256=digest.hexdigest(),package_source_sha256=sources,
        config_path=str(config),config_sha256=sha(config),numerical_source_sha256=numerical,
        cpu_reviews={str(p):sha(p) for p in reviews},
        backend_schema=composition['backend_schema'],trainer_schema=composition['trainer_schema'],
        scope=composition['status'],quality_verdict=None,
        quality_acceptance='both unchanged final toy and full canonical grid gates required',default_package_promoted=False)
    ready.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],package_sha256=value['package_sha256'],ready_sha256=sha(ready))))


if __name__=='__main__':
    main()
