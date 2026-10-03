"""Stdlib-only corrected fixture source/input preseal; no numerical imports."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    target=HERE/'SOURCE-FROZEN-v2.json'
    assert not target.exists()
    original=json.loads((HERE/'SOURCE-FROZEN.json').read_text())
    files=dict(original['source_and_input_sha256'])
    for p,d in files.items():assert sha(p)==d,p
    paths=[HERE/'SOURCE-FROZEN.json',HERE/'run_mechanics_v2.py',HERE/'seal_mechanics_v2.py',HERE/'PROTOCOL-v2.md',
        HERE/'FAILED-V1-REPORT.md',HERE/'FAILED-V1-FROZEN.json',HERE/'source-scope.json',
        ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-api-review/source-review.json',
        ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-api-review/SOURCE-REVIEW-FINAL-FROZEN.json',
        ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/mean-production-controls/PREEXEC-RECEIPT.json',
        ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/mean-production-controls/PREEXEC-FROZEN.json']
    failure=json.loads((HERE/'FAILED-V1-FROZEN.json').read_text())
    for p,d in failure['local_evidence_sha256'].items():assert sha(p)==d,p;files[p]=d
    for p in paths:files[str(p)]=sha(p)
    ast.parse((HERE/'run_mechanics_v2.py').read_text())
    result=dict(status='PRE_NUMERICAL_CORRECTED_FIXTURE_FROZEN',created_UTC=datetime.now(timezone.utc).isoformat(),
        original_source_preseal_sha256=sha(HERE/'SOURCE-FROZEN.json'),production_source_unchanged=True,
        backend_schema=9,trainer_schema=5,fixture_version=2,numerical_imports=0,PT_interpretations=0,
        source_and_input_sha256=dict(sorted(files.items())),
        CPU_command=['/tmp/pr38-default-env/bin/python',str(HERE/'run_mechanics_v2.py'),'--device','cpu','--package-root',str(HERE/'pkg-MEAN'),'--output',str(HERE/'cpu-contract-attempt2/result.json')],
        GPU_command=['/tmp/pr38-default-env/bin/python',str(HERE/'run_mechanics_v2.py'),'--device','cuda','--package-root',str(HERE/'pkg-MEAN'),'--output',str(ROOT/'integration/review/ra10-mechanics-gpu/result.json')],
        changes=['restore/assert exact raw model inputs after native constructor','preserve temporary hook instance ownership'],
        quality_verdict=None)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(target),sha256=sha(target),files=len(files))))

if __name__=='__main__':main()
