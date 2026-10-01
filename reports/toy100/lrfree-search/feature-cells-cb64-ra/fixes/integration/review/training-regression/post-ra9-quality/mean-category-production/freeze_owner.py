"""Seal owner CPU mechanics and immutable source/report maps; stdlib only."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (HERE/'READY.json').exists() and not (HERE/'FROZEN.json').exists()
    cpu=json.loads((HERE/'CPU-RECEIPT.json').read_text());assert cpu['status']=='PASS' and cpu['post_exit']
    source=json.loads((HERE/'SOURCE-FROZEN-v2.json').read_text())
    files=dict(source['source_and_input_sha256'])
    def collect(value):
        if isinstance(value,dict):
            for name,item in value.items():
                if isinstance(name,str) and name.startswith('/') and isinstance(item,str) and len(item)==64:
                    assert sha(name)==item,name
                    assert name not in files or files[name]==item,name
                    files[name]=item
                collect(item)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(cpu)
    reviews=[
        ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-api-review/source-review.json',
        ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-api-review/SOURCE-REVIEW-FINAL-FROZEN.json',
        ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-api-review/fixture-v2-source-review.json',
        ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-api-review/FIXTURE-V2-SOURCE-REVIEW-FROZEN.json',
        ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/mean-production-controls/PREEXEC-RECEIPT.json',
        ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/mean-production-controls/PREEXEC-FROZEN.json',
        ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/mean-production-controls/PREEXEC-V2-RECEIPT.json',
        ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/mean-production-controls/PREEXEC-V2-FROZEN.json']
    for p in reviews:
        collect(json.loads(p.read_text()));files[str(p)]=sha(p)
    for p,d in files.items():assert sha(p)==d,p
    package=HERE/'pkg-MEAN'
    modules={p.name:sha(p) for p in sorted((package/'particlegan').glob('*.py'))}
    assert len(modules)==30 and modules==json.loads((HERE/'SOURCE-FROZEN.json').read_text())['package_source_sha256']
    h=hashlib.sha256()
    for name in sorted(modules):h.update(name.encode()+b'\0'+(package/'particlegan'/name).read_bytes()+b'\0')
    for p in HERE.rglob('*'):
        if p.is_file():files[str(p)]=sha(p)
    ready=dict(status='FROZEN_CPU_QUALIFIED',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        package_root=str(package),package_sha256=h.hexdigest(),package_source_sha256=modules,
        numerical_source_sha256=dict(sorted(files.items())),
        config_path=str(HERE/'config.json'),config_sha256=sha(HERE/'config.json'),
        backend_schema=9,trainer_schema=5,owner_frozen_path=str(HERE/'FROZEN.json'),
        source_preseal_path=str(HERE/'SOURCE-FROZEN.json'),source_preseal_sha256=sha(HERE/'SOURCE-FROZEN.json'),
        authoritative_fixture_preseal_path=str(HERE/'SOURCE-FROZEN-v2.json'),
        authoritative_fixture_preseal_sha256=sha(HERE/'SOURCE-FROZEN-v2.json'),
        authoritative_CPU_receipt_path=str(HERE/'CPU-RECEIPT.json'),CPU_receipt_sha256=sha(HERE/'CPU-RECEIPT.json'),
        independent_source_review_sha256={str(p):sha(p) for p in reviews},
        authoritative_fixture_directory=str(HERE/'cpu-contract-attempt2'),
        failed_fixture_retained=str(HERE/'FAILED-V1-FROZEN.json'),
        cpu_command=source['CPU_command'],gpu_command=source['GPU_command'],
        source_changes=['feature_cells.py','birth_phase.py','new mean_transport.py'],unchanged_original_modules=27,
        config_changes={},quality_verdict=None,default_promoted=False,
        pending_gates=['independent packet/observer and cold-load/API controls on sealed fixtures',
            'root composed source/API review and serialized CUDA mechanics','unchanged final toy25/full canonical grid100 and original required replay/portability gates'],
        scope='Two fixed raw-input CPU reaction mechanics plus pre-execution source reviews; no emitted quality or general statistical certificate.')
    path=HERE/'READY.json';path.write_text(json.dumps(ready,indent=2)+'\n')
    files[str(path)]=sha(path)
    frozen=dict(status='FROZEN_CPU_QUALIFIED',post_exit=True,
        package_sha256=h.hexdigest(),package_source_sha256=modules,owner_ready_sha256=sha(path),
        local_and_protected_source_evidence_sha256=dict(sorted(files.items())),
        authoritative_fixture_version=2,failed_fixture_version1_preserved=True,quality_verdict=None)
    p=HERE/'FROZEN.json';p.write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(status=ready['status'],package_sha256=h.hexdigest(),ready_sha256=sha(path),frozen_sha256=sha(p),files=len(files))))

if __name__=='__main__':main()
