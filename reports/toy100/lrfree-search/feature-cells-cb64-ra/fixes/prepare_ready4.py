"""Freeze the merged candidate and all sources used by its GPU diagnostics."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
HERE=ROOT/'integration/iteration-4'
PACKAGE=ROOT/'pkg-CB64-RA4'
COUNT=ROOT/'integration/review/training-regression/global-count'
PLAN=ROOT/'performance/sampler-regression/cpu-plan-review/plan-batching'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target=HERE/'READY.json'
    assert not target.exists(), 'Do not overwrite frozen candidate readiness'
    numerical={}

    def add(path,expected=None):
        path=path.resolve()
        actual=sha(path)
        assert expected is None or actual==expected,str(path)
        numerical[str(path)]=actual

    count=json.loads((COUNT/'READY.json').read_text())
    for key in ('numerical_source_sha256','local_source_sha256','evidence_sha256','external_evidence_sha256'):
        for name,expected in count[key].items():
            path=Path(name)
            add(path if path.is_absolute() else COUNT/path,expected)
    plan=json.loads((PLAN/'READY.json').read_text())
    for name,expected in plan['package_source_sha256'].items():
        add(Path(plan['package_root'])/name,expected)
    for name,expected in plan['artifact_sha256'].items():
        add(Path(name),expected)
    for path in sorted((PACKAGE/'particlegan').rglob('*.py')):
        compile(path.read_text(),str(path),'exec')
        add(path)
    for path in (COUNT/'READY.json',PLAN/'READY.json',ROOT/'compose4.py',Path(__file__),
                 ROOT/'prepare_validation4.py',HERE/'run_gpu.py',HERE/'COMPOSITION.json',
                 ROOT/'configs/overrides-CB64-RA4.json',
                 ROOT/'performance/sampler-regression/cpu-plan-review/AXIS-ID-READY.json',
                 ROOT/'integration/axis-gpu/READY.json',ROOT/'integration/axis-gpu/result.json'):
        add(path)
    cpu={}
    for name in ('cpu-lineage.json','cpu-axis.json','cpu-count/result.json'):
        path=HERE/name
        data=json.loads(path.read_text())
        assert data['status']=='PASS' and not data['cuda_initialized']
        add(path)
        cpu[name]=dict(status='PASS',sha256=sha(path))
    for name in ('geometry/training-regression/test_lineage.py',
                 'geometry/training-regression/lineage_checks.py',
                 'performance/sampler-regression/cpu-plan-review/test_axis_id.py',
                 'performance/training-regression/count-review/FINAL-GLOBAL-PLAN-FROZEN.json'):
        add(ROOT/name)
    package_sources={}
    digest=hashlib.sha256()
    for path in sorted((PACKAGE/'particlegan').rglob('*.py')):
        relative=str(path.relative_to(PACKAGE/'particlegan'))
        package_sources[relative]=sha(path)
        digest.update(relative.encode()+b'\0'+path.read_bytes()+b'\0')
    ready=dict(status='FROZEN_MERGED_CPU_VALIDATED_GPU_PENDING',
        frozen_utc=datetime.now(timezone.utc).isoformat(),package_root=str(PACKAGE),
        package_sha256=digest.hexdigest(),package_source_sha256=package_sources,
        config_sha256=sha(ROOT/'configs/overrides-CB64-RA4.json'),
        backend_schema=4,trainer_schema=4,cpu=cpu,
        numerical_source_sha256=numerical,seed_policy='Existing seeds only; no seed experiments',
        geometry_policy='AXIS lineage classes and training/API bytes exact',
        count_policy='Fixed even-fit K+2K+2 overlap family; common Q/(3K+2); shared action budget/ledgers',
        performance_policy='Four proven AST splices; original MST and count law unchanged',
        gpu_policy='Physical GPU0 only, serial, deterministic, TF32 disabled, 20% allocator limit',
        quality_policy='Original learned and canonical data/init/scorers/gates/budgets unchanged')
    target.write_text(json.dumps(ready,indent=2)+'\n')
    print(json.dumps(dict(status=ready['status'],package_sha256=ready['package_sha256'],
        numerical_files=len(numerical),ready_sha256=sha(target))))


if __name__=='__main__':
    main()
