"""Stdlib-only binding of closed genuine10 mechanics and composed sources."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-api-review'
OWNER=ROOT/'integration/review/training-regression/post-ra10-quality/linear-output-production'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def add_maps(value,mapping):
    # The owner may bind both inputs and already-closed local outputs.
    if not isinstance(value,dict):return
    for key,item in value.items():
        if isinstance(item,dict):
            if item and all(isinstance(p,str) and p.startswith('/') and isinstance(d,str) and len(d)==64
                            for p,d in item.items()):
                for p,d in item.items():
                    assert p not in mapping or mapping[p]==d,p
                    assert sha(p)==d,p
                    mapping[p]=d
            else:add_maps(item,mapping)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-root',type=Path,required=True)
    p.add_argument('--composition',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--fixtures-root',type=Path,required=True)
    p.add_argument('--mechanics-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert not args.output.exists(),'retain prior bindings'
    mapping={}
    files=[OWNER/'READY.json',OWNER/'FROZEN.json',OWNER/'SOURCE-FROZEN.json',
        args.composition,args.config,args.mechanics_receipt,
        HERE/'check_affected_api.py',Path(__file__).resolve(),HERE/'seal_result.py',
        HERE/'PLAN-FROZEN.json',HERE/'HELPER-FROZEN.json',
        HERE/'preexecution-source/receipt.json',HERE/'preexecution-source/FROZEN.json',
        OLD/'check_api.py',OLD/'check_api_skeleton.py',OLD/'check_continuation.py',
        ROOT/'quality/ra10/integration-contract/receipt.json',ROOT/'quality/ra10/integration-contract/FROZEN.json',
        ROOT/'quality/ra8/integration-contract/check_api.py',
        Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100/toy_models.py')]
    for path in files:
        assert path.is_file(),path
        mapping[str(path.resolve())]=sha(path)
        if path.suffix=='.json':add_maps(json.loads(path.read_text()),mapping)
    composition=json.loads(args.composition.read_text())
    ready=json.loads((OWNER/'READY.json').read_text())
    assert ready['package_sha256']==composition['package_sha256']
    actual={str(path.relative_to(args.package_root/'particlegan')):sha(path)
            for path in sorted((args.package_root/'particlegan').rglob('*.py'))}
    assert len(actual)==31 and actual==composition['source_sha256']==ready['package_source_sha256']
    for name,digest in actual.items():mapping[str((args.package_root/'particlegan'/name).resolve())]=digest
    assert args.config.read_bytes()==(ROOT/'configs/overrides-CB64-RA9.json').read_bytes()
    mapping[str(ROOT/'configs/overrides-CB64-RA9.json')]=sha(ROOT/'configs/overrides-CB64-RA9.json')
    fixture_names={'before':'before.pt','after':'after.pt','hook':'hook-state.pt',
                   'trace':'trace.json','provenance':'provenance.json'}
    fixtures={}
    for case in ('grid','toy'):
        directory=args.fixtures_root/case
        fixtures[case]={key:str((directory/name).resolve()) for key,name in fixture_names.items()}
        for path in fixtures[case].values():mapping[path]=sha(path)
        provenance=json.loads((directory/'provenance.json').read_text())
        add_maps(provenance,mapping)
        assert provenance['case']==case and provenance['device']=='cpu'
        assert provenance['construction']=='fresh native backend10 constructor and explicitly copied qualified raw model/table inputs; genuine initial/reacted checkpoints; no old backend state load/relabel'
        for name,digest in provenance['output_sha256'].items():
            path=str((directory/name).resolve());assert sha(path)==digest
            assert path not in mapping or mapping[path]==digest
            mapping[path]=digest
    legacy=ROOT/'integration/review/training-regression/post-ra9-quality/mean-category-production/cpu-contract-attempt2/grid/after.pt'
    assert sha(legacy)=='ca6423941e75e70f5005e86cbe78aba3bf42a02dbbd08e20c19ba6a0e29b8791'
    mapping[str(legacy)]=sha(legacy)
    for name in ('check_affected_api.py','bind_inputs.py','seal_result.py'):
        ast.parse((HERE/name).read_text())  # No imports or helper execution.
    for path,digest in mapping.items():assert sha(path)==digest,path
    value=dict(status='PRE_EXECUTION_AFFECTED_API_FROZEN',utc=datetime.now(timezone.utc).isoformat(),
        backend_schema=10,mean_schema=2,trainer_schema=5,package_root=str(args.package_root.resolve()),
        package_sha256=composition['package_sha256'],config_path=str(args.config.resolve()),
        config_sha256=sha(args.config),owner_ready=str(OWNER/'READY.json'),root_composition=str(args.composition.resolve()),
        mechanics_receipt=str(args.mechanics_receipt.resolve()),fixtures=fixtures,old9_grid_state=str(legacy),
        genuine_construction=provenance['construction'],protected_sha256=mapping,
        API_sample_calls=3,API_rows_per_sample=17,CPU_updates_total=2,Torch_imported=False,
        PT_objects_loaded=0,numerical_execution=False,before_execution='explicit root GO for these exact bytes')
    args.output.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],sha256=sha(args.output),protected=len(mapping))))

if __name__=='__main__':main()
