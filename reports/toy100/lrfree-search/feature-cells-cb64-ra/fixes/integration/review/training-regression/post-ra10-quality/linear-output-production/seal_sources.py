"""Stdlib-only pre-numerical source/helper/raw-input seal; create once."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
PYTHON='/tmp/pr38-default-env/bin/python'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    target=HERE/'SOURCE-FROZEN.json'
    assert not target.exists()
    scratch=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra10-quality/linear-output-mean-prototype'
    prior=scratch/'attempt1/FINAL-PROVENANCE-FROZEN.json'
    files=dict(json.loads(prior.read_text())['files'])
    files[str(prior)]=sha(prior)
    package=HERE/'pkg-OUTPUT-MEAN'
    modules={str(p.relative_to(package/'particlegan')):sha(p)
        for p in sorted((package/'particlegan').rglob('*.py'))}
    base=ROOT/'pkg-CB64-RA10'
    base_modules={str(p.relative_to(base/'particlegan')):sha(p)
        for p in sorted((base/'particlegan').rglob('*.py'))}
    assert len(modules)==31 and len(base_modules)==30
    assert set(modules)-set(base_modules)=={'output_moments.py'}
    changed=[name for name in base_modules if modules[name]!=base_modules[name]]
    assert changed==['feature_cells.py','mean_transport.py'],changed
    config=HERE/'overrides.json'
    assert config.read_bytes()==(ROOT/'configs/overrides-CB64-RA10.json').read_bytes()
    digest=hashlib.sha256()
    for name in sorted(modules):
        path=package/'particlegan'/name
        compile(path.read_text(),str(path),'exec')
        digest.update(name.encode()+b'\0'+path.read_bytes()+b'\0')
        files[str(path)]=modules[name]
    paths=[HERE/name for name in ('DESIGN.md','PROTOCOL.md','overrides.json','run_mechanics.py','raw_moments.py','seal_sources.py')]
    paths += [ROOT/'quality/results/RA11-selection.json',ROOT/'quality/results/RA11-selection-provenance.json',ROOT/'quality/RA11-PLAN.md',
        ROOT/'integration/review/training-regression/post-ra10-quality/linear-output-result-review/receipt.json',
        ROOT/'integration/review/training-regression/post-ra10-quality/linear-output-result-review/FROZEN.json']
    for path in paths:
        files[str(path)]=sha(path)
        if path.suffix=='.py':compile(path.read_text(),str(path),'exec')
    for path,expected in files.items():
        assert sha(path)==expected,path
    cpu_command=[PYTHON,'-B',str(HERE/'run_mechanics.py'),'--device','cpu',
        '--package-root',str(package),'--output',str(HERE/'cpu-contract-attempt1/result.json')]
    gpu_command=[PYTHON,'-B',str(HERE/'run_mechanics.py'),'--device','cuda',
        '--package-root',str(ROOT/'pkg-CB64-RA11'),'--output',str(ROOT/'integration/review/ra11-mechanics-gpu/result.json')]
    value=dict(status='FROZEN_SOURCE_HELPERS_INPUTS_NUMERICAL_PENDING',utc=datetime.now(timezone.utc).isoformat(),
        backend_schema=10,trainer_schema=5,mean_schema=2,package_root=str(package),
        package_source_sha256=modules,package_sha256=digest.hexdigest(),config_path=str(config),config_sha256=sha(config),
        source_and_input_sha256=dict(sorted(files.items())),changed_original_modules=changed,added_modules=['output_moments.py'],
        unchanged_original_modules=28,cpu_command=cpu_command,gpu_command=gpu_command,
        CPU_invocations_allowed=1,per_case_reactions=1,fixture_seed=314159,new_seed_experiments=0,
        PT_interpretations=0,Torch_imported=False,model_forwards=0,new_quality_emissions=0,quality_verdict=None,
        before_CPU_required=['independent math/source PASS','independent state/API source PASS','root GO for this exact source seal'],
        before_CUDA_required=['CPU/source/API qualification','root composed candidate freeze','physical serial-slot source/provenance guards'],
        checkpoint_scope='genuine freshly constructed backend10 initial/reacted states from qualified raw RA9 model/table tensors; no prior backend load/relabel')
    with target.open('x') as f:
        f.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],guarded_files=len(files),source_freeze_sha256=sha(target),package_sha256=value['package_sha256'])))


if __name__=='__main__':main()
