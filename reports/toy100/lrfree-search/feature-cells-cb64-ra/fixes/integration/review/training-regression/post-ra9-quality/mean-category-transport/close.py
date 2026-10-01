"""Post-exit stdlib evidence seal; no new measurements or source edits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def main():
    if (HERE/'FROZEN.json').exists() or (HERE/'receipt.json').exists():
        raise SystemExit('closed evidence already exists')
    pre=read(HERE/'SOURCE-FROZEN.json')
    for path,digest in pre['source_and_input_sha256'].items():
        assert sha(path)==digest,path
    math_area=ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-witness-review'
    preview_area=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/mean-preview-review'
    review_paths=[math_area/'BUNDLE-RECEIPT.json',math_area/'FROZEN.json',
                  preview_area/'receipt.json',preview_area/'FROZEN.json']
    assert read(review_paths[0])['status']=='PASS' and read(review_paths[2])['status']=='PASS'
    for review in (read(review_paths[1]),read(review_paths[3])):
        for key,mapping in review.items():
            if isinstance(mapping,dict) and key.endswith('sha256'):
                for path,digest in mapping.items():
                    assert sha(path)==digest,path
    result=read(HERE/'attempt1/result.json')
    assert result['status']=='PASS_FIXED_PROTOTYPE'
    assert not (HERE/'attempt1/failure.json').exists()
    outputs=[HERE/'attempt1/result.json',HERE/'attempt1/grid.json',HERE/'attempt1/toy.json',HERE/'attempt1.log']
    receipt=dict(status='PASS',scope='single fixed CPU scratch mean-copy prototype; no production/quality qualification',
        post_exit=True,exit_code=0,numerical_attempts=1,failed_numerical_attempts=0,
        source_preseal_sha256=sha(HERE/'SOURCE-FROZEN.json'),
        source_and_input_sha256=pre['source_and_input_sha256'],
        independent_review_sha256={str(p):sha(p) for p in review_paths},
        numerical_output_sha256={str(p):sha(p) for p in outputs},
        created_UTC=datetime.now(timezone.utc).isoformat())
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    local=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='FROZEN.json')
    frozen=dict(status='PASS',post_exit=True,created_UTC=datetime.now(timezone.utc).isoformat(),
        local_source_and_evidence_sha256={str(p):sha(p) for p in local},
        protected_source_and_input_sha256=pre['source_and_input_sha256'],
        independent_review_sha256=receipt['independent_review_sha256'],
        receipt_sha256=sha(HERE/'receipt.json'))
    (HERE/'FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',receipt_sha256=sha(HERE/'receipt.json'),
                         FROZEN_sha256=sha(HERE/'FROZEN.json'),local_files=len(local))))


if __name__=='__main__':
    main()
