"""Verify focused receipts and freeze the owner proposal with standard maps."""
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
BASE=ROOT/'pkg-CB64-RA7/particlegan'
PACKAGE=HERE/'pkg-PAIR-AVERAGE'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (HERE/'READY.json').exists() and not (HERE/'FROZEN.json').exists()
base_map={str(p.relative_to(BASE)):sha(p) for p in sorted(BASE.rglob('*.py'))}
assert base_map==json.loads((ROOT/'quality/ra7/READY.json').read_text())['package_source_sha256']
source_map={str(p.relative_to(PACKAGE/'particlegan')):sha(p) for p in sorted((PACKAGE/'particlegan').rglob('*.py'))}
assert base_map.keys()==source_map.keys()
assert [n for n in source_map if source_map[n]!=base_map[n]]==['feature_cells.py','training.py']
allowed=dict(feature_cells={'__init__','maybe_apply','diagnostics','state_dict','check_state','load_state_dict'},
    training={'_serve_settled','_load_state_dict'})
source_scope={}
for filename,target in (('feature_cells.py','FeatureCellBirthDeath'),('training.py','GANTrainer')):
    old=ast.parse((BASE/filename).read_text());new=ast.parse((PACKAGE/'particlegan'/filename).read_text())
    old_class=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name==target)
    new_class=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name==target)
    dump=lambda n:ast.dump(n,include_attributes=False)
    old_methods={n.name:n for n in old_class.body if isinstance(n,ast.FunctionDef)}
    new_methods={n.name:n for n in new_class.body if isinstance(n,ast.FunctionDef)}
    changed=[n for n in old_methods if dump(old_methods[n])!=dump(new_methods[n])]
    assert set(changed)==allowed[filename.removesuffix('.py')]
    for oldnode in old.body:
        if isinstance(oldnode,ast.ClassDef) and oldnode.name==target:continue
        identity=(type(oldnode),getattr(oldnode,'name',None))
        candidates=[n for n in new.body if (type(n),getattr(n,'name',None))==identity]
        assert any(dump(oldnode)==dump(n) for n in candidates)
    source_scope[filename]=dict(changed_existing_methods=changed,
        unchanged_methods=[n for n in old_methods if n not in changed],
        added_methods=[n for n in new_methods if n not in old_methods])
patch=[]
for n in ('feature_cells.py','training.py'):
    patch.extend(difflib.unified_diff((BASE/n).read_text().splitlines(keepends=True),
        (PACKAGE/'particlegan'/n).read_text().splitlines(keepends=True),
        fromfile='a/particlegan/'+n,tofile='b/particlegan/'+n))
(HERE/'PAIR-AVERAGE.patch').write_text(''.join(patch))
(HERE/'source-scope.json').write_text(json.dumps(dict(status='PASS',changed_files=list(source_scope),
    scope=source_scope,entire_FeatureCellSnapshot_AST_unchanged=True,
    original_step_average_noise_sample_optimizer_population_law_AST_unchanged=True,
    source_sha256=source_map,base_source_sha256=base_map),indent=2)+'\n')
receipt_names=('gate-receipt.json','neutrality-receipt.json','reaction-RA7.json','reaction-proposal.json')
receipts={n:json.loads((HERE/n).read_text()) for n in receipt_names}
for n,r in receipts.items():
    assert r['status']=='PASS'
    for k,v in r['source_sha256'].items():assert sha(Path(k))==v,(n,k)
    for k,v in r['input_sha256'].items():assert sha(Path(k))==v,(n,k)
for a,b in zip(receipts['reaction-RA7.json']['records'],receipts['reaction-proposal.json']['records']):
    assert a['step']==b['step']
    for field in ('plan_sha256','numerical_state_sha256','original_semantic_event_sha256'):assert a[field]==b[field]
local=[HERE/n for n in ('gate_contract.py','neutrality_contract.py','reaction_contract.py','contract_utils.py','freeze_owner.py','PROTOCOL.md')]
numerical={str(PACKAGE/'particlegan'/n):v for n,v in source_map.items()}
numerical.update({str(p):sha(p) for p in local})
input_map={str(ROOT/'quality/ra7/READY.json'):sha(ROOT/'quality/ra7/READY.json'),
    str(ROOT/'configs/overrides-CB64-RA7.json'):sha(ROOT/'configs/overrides-CB64-RA7.json')}
for r in receipts.values():input_map.update(r['input_sha256'])
combined=dict(status='PASS',receipt_sha256={str(HERE/n):sha(HERE/n) for n in receipt_names},
    fixed_geometry_checks=len(receipts['gate-receipt.json']['checks']),
    observational_neutrality_checks=len(receipts['neutrality-receipt.json']['checks']),
    matched_reactions=len(receipts['reaction-proposal.json']['records']),
    matched_plan_actions_numerical_state_and_RNG_exact=True,
    source_sha256=numerical,input_sha256=input_map,cpu_only=True,cuda_initialized=False,
    new_training_steps=0,new_optimizer_steps=0,new_emissions=0,new_seed_experiments=0,quality_verdict=None,
    trainer_API_and_prospective_quality_root_owned=True)
(HERE/'receipt.json').write_text(json.dumps(combined,indent=2)+'\n')
digest=hashlib.sha256()
for n in sorted(source_map):digest.update(n.encode()+b'\0'+(PACKAGE/'particlegan'/n).read_bytes()+b'\0')
ready=dict(status='CPU_QUALIFIED_PRIVATE_PAIRED_AVERAGE_GPU_PENDING',frozen_utc=datetime.now(timezone.utc).isoformat(),
    package_root=str(PACKAGE),package_sha256=digest.hexdigest(),package_source_sha256=source_map,
    numerical_source_sha256=numerical,local_source_sha256={str(p):sha(p) for p in local},
    backend_schema=7,trainer_schema=5,base_ready_sha256=sha(ROOT/'quality/ra7/READY.json'),
    config_path=str(ROOT/'configs/overrides-CB64-RA7.json'),config_sha256=sha(ROOT/'configs/overrides-CB64-RA7.json'),
    config_changes={},owner_receipt_path=str(HERE/'receipt.json'),owner_receipt_sha256=sha(HERE/'receipt.json'),
    owner_frozen=str(HERE/'FROZEN.json'),patch_path=str(HERE/'PAIR-AVERAGE.patch'),
    gate='finite/chart/nonduplicate and sum(EMA_p>Q & EMA_inside & same_real_group)>=N-floor(Q*N)',
    expiry='rows_since_eval<N; empirical bounded-stale view, no independent step TTL',
    scope='geometry anti-blur serving only; not distribution equivalence, stationarity or quality',
    quality_verdict=None,default_package_promoted=False,root_trainer_API_GPU_and_quality_pending=True)
(HERE/'READY.json').write_text(json.dumps(ready,indent=2)+'\n')
files=[*sorted((PACKAGE/'particlegan').rglob('*.py')),*local,
    *[HERE/n for n in ('PAIR-AVERAGE.patch','source-scope.json','receipt.json','READY.json','REPORT.md',
        *receipt_names,'gate-attempt1.log','neutrality-attempt1.log','reaction-RA7-attempt1.log','reaction-proposal-attempt1.log')]]
frozen=dict(status='FROZEN_CPU_QUALIFIED_PRIVATE_PAIRED_AVERAGE',files={str(p):sha(p) for p in files},
    numerical_source_sha256=numerical,input_sha256=input_map,package_sha256=digest.hexdigest(),
    quality_verdict=None,default_package_promoted=False)
(HERE/'FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
print(json.dumps(dict(status=ready['status'],package_sha256=digest.hexdigest(),
    ready_sha256=sha(HERE/'READY.json'),frozen_sha256=sha(HERE/'FROZEN.json'),receipt_sha256=sha(HERE/'receipt.json'))))
