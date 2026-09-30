"""Freeze the distinct CPU-qualified birth helper/API proposal for root composition."""
import ast
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BASE=ROOT/'pkg-CB64-RA4';PACKAGE=HERE/'pkg-ANCHOR-CONTRACT'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (HERE/'READY.json').exists()
assert sha(HERE/'anchor_birth.py')=='aa50983d8b61304acf4e1138f16fd4784798983df91e931863bcacd3b511f76f'
assert sha(HERE/'birth_phase.py')=='aa794152ad55a44673adf3bac4284f6ab4bc3b559fac8cb4e7169fe7fafd633d'
for name in ('anchor_birth.py','birth_phase.py'):
    assert sha(HERE/name)==sha(PACKAGE/'particlegan'/name)
contract=json.loads((HERE/'cpu-contract-final-edges/result.json').read_text())
saved=json.loads((HERE/'saved-production-contract.json').read_text())
response=json.loads((HERE/'saved-proposal-response.json').read_text())
assert contract['status']==saved['status']=='PASS' and not contract['cuda_initialized'] and not saved['cuda_initialized']
assert len(contract['records'])==15 and len(contract['edge_records'])==9
assert all(r['birth']==4 for r in saved['records']) and saved['global_rng_unchanged']
assert response['status']=='COMPLETE_FIXED_POLICY_CPU_RESPONSE' and response['global_rng_unchanged']
base_map={str(p.relative_to(BASE)):sha(p) for p in sorted(BASE.rglob('*.py'))}
package_map={str(p.relative_to(PACKAGE)):sha(p) for p in sorted(PACKAGE.rglob('*.py'))}
changed=[name for name in package_map if package_map[name]!=base_map.get(name)]
assert changed==['particlegan/anchor_birth.py','particlegan/birth_phase.py','particlegan/feature_cells.py']
original=ast.parse((BASE/'particlegan/feature_cells.py').read_text())
candidate=ast.parse((PACKAGE/'particlegan/feature_cells.py').read_text())
def snapshot(tree):return next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='FeatureCellSnapshot')
original_class=snapshot(original);candidate_class=snapshot(candidate)
old=next(n for n in original_class.body if isinstance(n,ast.FunctionDef) and n.name=='select_parents')
new=next(n for n in candidate_class.body if isinstance(n,ast.FunctionDef) and n.name=='select_parents')
candidate_class.body[candidate_class.body.index(new)]=old
assert ast.dump(candidate,include_attributes=False)==ast.dump(original,include_attributes=False)
parts=[]
for name in changed:
    oldtext=(BASE/name).read_text() if (BASE/name).exists() else ''
    newtext=(PACKAGE/name).read_text()
    parts.append('diff --git a/'+name+' b/'+name+'\n')
    if not oldtext:parts.append('new file mode 100644\n')
    parts.extend(difflib.unified_diff(oldtext.splitlines(keepends=True),newtext.splitlines(keepends=True),
        fromfile='a/'+name if oldtext else '/dev/null',tofile='b/'+name))
patch=HERE/'ANCHOR-BIRTH.patch';patch.write_text(''.join(parts))
dry=subprocess.run(['patch','--dry-run','-p1','-i',str(patch)],cwd=BASE,text=True,capture_output=True)
assert dry.returncode==0,(dry.stdout,dry.stderr)
source_contract=dict(status='PASS',changed_package_files=changed,all_other_module_asts_exact=True,
    only_modified_existing_method='FeatureCellSnapshot.select_parents',
    isolation_extension=['supported_counts override','reserved_rows excluding children and parents'],
    patch_dry_run=0,patch_sha256=sha(patch),base_source_sha256=base_map,package_source_sha256=package_map)
(HERE/'source-contract.json').write_text(json.dumps(source_contract,indent=2)+'\n')
local_names=('PROTOCOL.md','REPORT.md','INTEGRATION.md','ANCHOR-BIRTH.patch','anchor_birth.py','birth_phase.py',
    'measure_saved_proposals.py','measure_saved_utils.py','check_saved_production.py','birth_contract_check.py',
    'birth_contract_cases.py','birth_edge_contracts.py','prepare_contract_package.py','freeze_ready.py')
local_map={name:sha(HERE/name) for name in local_names}
numerical_map={str(p.relative_to(HERE)):sha(p) for p in sorted(PACKAGE.rglob('*.py'))}
numerical_map.update({name:local_map[name] for name in local_names if name.endswith('.py')})
evidence_names=('saved-proposal-response.json','saved-proposal-response.log','saved-production-contract.json','saved-production-contract.log',
    'cpu-contract-attempt1/result.json','cpu-contract-attempt1.log','cpu-contract-final/result.json','cpu-contract-final.log',
    'cpu-contract-final-edges/result.json','cpu-contract-final-edges.log','prepare-contract-package.json','prepare-contract-package-final.json',
    'source-contract.json','source-attempt1/anchor_birth.py','source-attempt1/birth_phase.py')
external=[ROOT/'pkg-CB64-RA4/particlegan/feature_cells.py',
    HERE.parent/'global-count/inputs.pt',HERE.parent/'global-count/exhaustion-input.pt',HERE.parent/'global-count/contract_cases.py',
    HERE.parent/'post-ra4-mode-diagnosis/REPORT.md',HERE.parent/'post-ra4-mode-diagnosis/receipt.json']
external+=[Path(p) for p in saved['source_sha256'] if 'checkpoint-' in p]
package_digest=hashlib.sha256(json.dumps(package_map,sort_keys=True,separators=(',',':')).encode()).hexdigest()
ready=dict(status='FROZEN_CPU_BIRTH_HELPERS_AND_API_QUALIFIED_ROOT_INTEGRATION_AND_INDEPENDENT_REVIEW_PENDING',
    created_at=datetime.now(timezone.utc).isoformat(),package_root=str(PACKAGE),base_package=str(BASE),
    changed_package_files=changed,package_source_sha256=package_map,full_package_source_sha256=package_digest,
    local_source_sha256=local_map,numerical_source_sha256=numerical_map,
    evidence_sha256={name:sha(HERE/name) for name in evidence_names},
    external_evidence_sha256={str(p):sha(p) for p in external},
    policy='new_latent_even_real_anchor_shared_global_certificates',
    q=.05,multiplicity='unchanged3K+2',cutoff='unchangedQ/(3K+2)',
    source_seed_is_supported_copy_parent=False,new_destination_inside=True,paired_live_ema_acceptance_required=True,
    ordinary_budget='floor(.05*N) shared by mass/local/novel/global',
    phase_order=['mass copies','local support copies','novel real-anchor births','global support copies','legacy isolation'],
    work_bound=dict(cells=4,linearizations_per_model=4,models=2),
    novel_commit=dict(live_latent='new proposal',ema_latent='separate accepted proposal',
        optimizer_moments='zero child row only',latent_history='zero child row only',shared_adam_step='retained',
        lineage='invalidate old child edges, no seed link',row_evidence='all moved rows sent to existing trainer reset/rebase'),
    new_modules=['particlegan.anchor_birth','particlegan.birth_phase'],
    new_top_level_functions={name:[n.name for n in ast.parse((HERE/name).read_text()).body if isinstance(n,ast.FunctionDef)]
        for name in ('anchor_birth.py','birth_phase.py')},
    root_integration_instructions=str(HERE/'INTEGRATION.md'),
    original_existing_asts_except_isolation_api_exact=True,no_semantic_timing=True,
    cpu_mechanical_variants=15,cpu_rejection_null_reservation_edges=9,saved_production_states=2,saved_paired_births=8,
    independent_review_status='PENDING, separately assigned to state_review',
    cpu_only=True,cuda_initialized=False,new_training_steps=0,new_seeds=0,quality_verdict=None,
    root_gpu_command=['/tmp/pr38-default-env/bin/python','-u','-B',str(HERE/'birth_contract_check.py'),
        '--package-root','ROOT_COMPOSED_PACKAGE','--output','NEW_ROOT_OUTPUT','--device','cuda:0'],
    scope='Separate production-capable birth helpers/API with fixed-input CPU qualification; root reaction integration/checkpoint/quality pending',
    strict_quality_target=dict(learned_cuda_toy_precision_at_least=.90,toy_modes=25,toy_tv_at_most=.10,
        canonical_grid100='all frozen stability/holdout gates'),
    limits=['fixed CPU geometry is not historicalGPU replay','no learned quality or high-dimensional support-law qualification',
        'unreachable anchors can reject within work bound','adaptive critic/FIFO dependence unchanged'])
(HERE/'READY.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(dict(status=ready['status'],ready_sha256=sha(HERE/'READY.json'),
    full_package_source_sha256=package_digest,source_files=len(package_map),patch_sha256=sha(patch)),indent=2))
