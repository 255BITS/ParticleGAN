"""Stdlib-only production preseal; no Torch or numerical/PT interpretation."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
BASE = ROOT / 'pkg-CB64-RA9/particlegan'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def main():
    target=HERE/'SOURCE-FROZEN.json'
    if target.exists(): raise SystemExit('preserve preseal; archive a failed preparation before a new one')
    base={p.name:sha(p) for p in sorted(BASE.glob('*.py'))}
    proposal={p.name:sha(p) for p in sorted((HERE/'pkg-MEAN/particlegan').glob('*.py'))}
    assert len(base)==29 and len(proposal)==30
    assert set(proposal)-set(base)=={'mean_transport.py'} and not set(base)-set(proposal)
    assert [k for k in sorted(base) if base[k]!=proposal[k]]==['birth_phase.py','feature_cells.py']
    assert (HERE/'config.json').read_bytes()==(ROOT/'configs/overrides-CB64-RA9.json').read_bytes()
    for path in [*HERE.glob('*.py'),*(HERE/'pkg-MEAN/particlegan').glob('*.py')]: ast.parse(path.read_text())
    paths=set([*BASE.glob('*.py'),*(HERE/'pkg-MEAN/particlegan').glob('*.py'),*HERE.glob('*.py'),
        HERE/'config.json',HERE/'DESIGN.md',HERE/'PROTOCOL.md',
        ROOT/'quality/RA10-PLAN.md',ROOT/'quality/results/RA10-selection.json',
        ROOT/'quality/compose_ra10.py',ROOT/'quality/run_ra10_mechanics.py',ROOT/'quality/freeze_ra10_mechanics.py',
        ROOT/'quality/ra9/READY.json',ROOT/'quality/ra9/COMPOSITION.json',ROOT/'configs/overrides-CB64-RA9.json',
        ROOT/'validation-cb64-ra9/source-freeze.json',
        ROOT/'validation-cb64-ra9/screens/runs/grid100/final-state.pt',
        ROOT/'validation-cb64-ra9/learned/training/toy/CB64-RA9/checkpoint-2000.pt',
        ROOT/'quality/ra8/integration-contract/check_api.py',
        Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100/toy_models.py'),
        ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/mean-production-inventory/DESIGN.md',
        ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-checkpoint-design/DESIGN.md',
        ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-checkpoint-design/FROZEN.json'])
    def collect(value):
        if isinstance(value,dict):
            for name,item in value.items():
                if isinstance(name,str) and name.startswith('/') and isinstance(item,str) and len(item)==64:
                    assert sha(name)==item,name;paths.add(Path(name))
                collect(item)
        elif isinstance(value,list):
            for child in value:collect(child)
    selection=json.loads((ROOT/'quality/results/RA10-selection.json').read_text())
    assert selection['status']=='PROSPECTIVE_MEAN_COPY_CANDIDATE_SELECTED'
    collect(selection)
    proto=ROOT/'integration/review/training-regression/post-ra9-quality/mean-category-transport'
    for name in ('SOURCE-FROZEN.json','FROZEN.json','receipt.json'):
        paths.add(proto/name);collect(json.loads((proto/name).read_text()))
    assert all(p.is_file() for p in paths)
    result=dict(status='PRE_NUMERICAL_SOURCE_INPUT_FROZEN',created_UTC=datetime.now(timezone.utc).isoformat(),
        backend_schema=9,trainer_schema=5,numerical_imports=0,PT_interpretations=0,
        source_and_input_sha256={str(p):sha(p) for p in sorted(paths)},package_source_sha256=proposal,
        changed_original_modules=['birth_phase.py','feature_cells.py'],added_modules=['mean_transport.py'],
        config_sha256=sha(HERE/'config.json'),
        intended_mechanics_runs=dict(owner_CPU=1,root_serial_CUDA=1),
        CPU_command=['/tmp/pr38-default-env/bin/python',str(HERE/'run_mechanics.py'),'--device','cpu',
            '--package-root',str(HERE/'pkg-MEAN'),'--output',str(HERE/'cpu-contract-attempt1/result.json')],
        GPU_command=['/tmp/pr38-default-env/bin/python',str(HERE/'run_mechanics.py'),'--device','cuda',
            '--package-root',str(HERE/'pkg-MEAN'),'--output',str(ROOT/'integration/review/ra10-mechanics-gpu/result.json')],
        quality_verdict=None)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],path=str(target),sha256=sha(target),files=len(paths))))


if __name__=='__main__':main()
