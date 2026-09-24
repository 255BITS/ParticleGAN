"""Deterministically archive the source-frozen two-state GAN noise assay."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'reports/toy100/continuous-evidence/common-instance-noise-round10'
COLD=ROOT/'artifacts/continuous-learning/round10/common-instance-noise-cold-endpoint-v1'
CONT=ROOT/'artifacts/continuous-learning/round10/common-instance-noise-short-continuation-v1'
ROOT_COLD=Path('/ml2/hypergan/ParticleGAN-continuous-learning/reports/toy100/continuous-evidence/pr88-cold-independent-audit/mode_hold-pr84-state.pt.gz')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compress(data):
    target=io.BytesIO()
    with gzip.GzipFile(fileobj=target,mode='wb',filename='',mtime=0,compresslevel=9) as stream:
        stream.write(data)
    return target.getvalue()


def main():
    if EVIDENCE.exists():
        raise FileExistsError(EVIDENCE)
    specs=[
        ('cold-endpoint/declaration.json',COLD/'declaration.json',False),
        ('cold-endpoint/result.json',COLD/'result.json',False),
        ('cold-endpoint/summary.json',COLD/'summary.json',False),
        ('cold-endpoint/geometry-analysis.json',COLD/'geometry-analysis.json',False),
        ('continuation/declaration.json',CONT/'declaration.json',False),
        ('continuation/warm1325-result.json',CONT/'warm1325-result.json',False),
        ('continuation/cold1200-result.json',CONT/'cold1200-result.json',False),
        ('continuation/summary.json',CONT/'summary.json',False),
        ('states/pr84-cold1200.pt',ROOT_COLD,True),
    ]
    for filename in ('common_instance_noise_falsifier.py',
                     'common_instance_noise_cold_endpoint.py',
                     'common_instance_noise_short_continuation.py',
                     'common_instance_noise_geometry.py',
                     'archive_common_instance_noise_round10.py'):
        specs.append(('source/'+filename,ROOT/'reports/toy100'/filename,False))
    manifest={
        'scope':'frozen two-state GAN common-instance-noise copied-critic assay and 8-step local alternating continuation',
        'source_commit':'same commit as this manifest; per-file hashes below are authoritative',
        'warm_first_assay':'../common-instance-noise-2state/v1-state-selection-failure (1325 valid; 1539 invalid for nonlocal acquisition)',
        'input_cold_origin':'PR88 independent audit PR84 control, exact post-final-evaluation saved state',
        'noisy_width':1.1279860476026753,
        'files':[],
    }
    EVIDENCE.mkdir(parents=True)
    for name,path,is_gzip in specs:
        source=path.read_bytes()
        raw=gzip.decompress(source) if is_gzip else source
        packed=compress(raw)
        destination=EVIDENCE/(name+'.gz')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(packed)
        manifest['files'].append({'path':str(destination.relative_to(EVIDENCE)),
            'raw_path':name,'raw_sha256':sha(raw),'gzip_sha256':sha(packed),
            'raw_bytes':len(raw),'gzip_bytes':len(packed)})
    (EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':len(specs),'manifest':str(EVIDENCE/'manifest.json')}))


if __name__=='__main__':
    main()
