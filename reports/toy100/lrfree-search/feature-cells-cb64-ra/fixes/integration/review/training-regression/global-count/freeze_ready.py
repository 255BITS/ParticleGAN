"""Freeze this final private count proposal after all required CPU review gates."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PACKAGE=HERE/'pkg-global-count'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
review=ROOT/'performance/training-regression/count-review'
plan=ROOT/'performance/sampler-regression/cpu-plan-review/plan-batching'
assert HERE.parent.name=='training-regression' and ROOT.name=='feature-cells-fixes-20260929'
feature=PACKAGE/'particlegan/feature_cells.py'
assert sha(feature)=='1d3c90aa3b72d086df0defcc8c356e48778310f005473e8c5c271afa44f89d76'
assert sha(HERE/'inputs.pt')=='3e8deeb2fe534d28cf3731a4694c0543e280d2de270e0d9644731d9ec40057f2'
checks={name:json.loads((HERE/name).read_text()) for name in ('cpu-contract/result.json','statistical-check.json','source-contract.json','response-diagnosis.json')}
assert all(v['status']=='PASS' for v in checks.values())
assert len(checks['cpu-contract/result.json']['records'])==14
assert not any(v.get('cuda_initialized') for v in checks.values())
for name in ('final-global-review.json','final-plan-review.json'):
    record=json.loads((review/name).read_text());assert record['status']=='PASS' and not record['cuda_initialized']
    assert all(sha(Path(path))==digest for path,digest in record['source_sha256'].items())
package_map={str(p.relative_to(PACKAGE)):sha(p) for p in sorted(PACKAGE.rglob('*.py'))}
base=HERE.parent/'joint-count'
base_ready=json.loads((base/'READY.json').read_text())
assert sha(base/'READY.json')=='7e23b8828a19ceb83973e52c0fceb08725a058d237756e1e5790b9bd6ae82f3c'
assert {str(p.relative_to(base/'pkg-joint-count')):sha(p) for p in (base/'pkg-joint-count').rglob('*.py')}==base_ready['package_source_sha256']
changed=[name for name in package_map if package_map[name]!=base_ready['package_source_sha256'][name]]
assert changed==['particlegan/feature_cells.py']
local_names=('PROTOCOL.md','REPORT.md','GLOBAL-COUNT.patch','prepare_inputs.py','prepare_exhaustion.py',
    'contract_check.py','contract_cases.py','statistical_check.py','diagnose_response.py','freeze_ready.py','V4-FEATURE-CELLS.py')
local_map={name:sha(HERE/name) for name in local_names}
numerical_map={str(p.relative_to(HERE)):sha(p) for p in sorted(PACKAGE.rglob('*.py'))}
numerical_map.update({name:local_map[name] for name in ('prepare_inputs.py','prepare_exhaustion.py','contract_check.py',
    'contract_cases.py','statistical_check.py','diagnose_response.py','V4-FEATURE-CELLS.py')})
evidence_names=('inputs.pt','exhaustion-input.pt','prepare-inputs.json','prepare-inputs.log','prepare-exhaustion.json','prepare-exhaustion.log',
    'cpu-contract/result.json','cpu-contract.log','cpu-contract-attempt1/result.json','cpu-contract-attempt1.log',
    'cpu-contract-attempt2/result.json','cpu-contract-attempt2.log','statistical-check.json','statistical-check.log',
    'source-contract.json','response-diagnosis.json','response-diagnosis.log')
evidence={name:sha(HERE/name) for name in evidence_names}
external_paths=[review/name for name in ('final-global-review.json','final-plan-review.json','audit_global_final.py','final-global-plan-attempt2.log')]
external_paths += [plan/name for name in ('READY.json','final-splices.json','final-cpu-02.json','plan_pair_final.py','test_final_plan_cpu.py')]
assert all(p.exists() for p in external_paths)
external={str(p):sha(p) for p in external_paths}
source_receipt=checks['source-contract.json']
ready=dict(status='FROZEN_FINAL_3K_PLUS_2_CPU_AND_INDEPENDENT_AUDIT_PASS_ROOT_GPU_PENDING',
    created_at=datetime.now(timezone.utc).isoformat(),package_root=str(PACKAGE),base_package=str(base/'pkg-joint-count'),
    base_ready_sha256=sha(base/'READY.json'),base_feature_sha256=source_receipt['base_feature_sha256'],
    frozen_v4_ready_sha256=sha(HERE.parent/'READY.json'),frozen_2K_ready_sha256=sha(HERE.parent/'support-count/READY.json'),
    changed_package_files=changed,package_source_sha256=package_map,numerical_source_sha256=numerical_map,
    local_source_sha256=local_map,evidence_sha256=evidence,external_evidence_sha256=external,
    source_map_paths=dict(package='relative to package_root',numerical='relative to proposal root',local='relative to proposal root'),
    mass_policy='joint_mass_local_global_common_3K_plus_2_unique_parents_v1',
    count_family='original_K_plus_support_2K_plus_global_2_common_Q_over_3K_plus_2',
    q=.05,count_family_sizes='K+2K+2',multiplicity='3K+2 including empty hypotheses',cutoff='Q/(3K+2)',
    count_partition='unchanged even-fit .95 score order statistic, ties inside',
    even_only_boundary=True,original_support_law_unchanged=True,
    original_mass_body_exact=True,local_support_body_exact=True,isolation_body_exact=True,
    new_top_level_functions=[],new_snapshot_methods=['_ordinary_global_transport'],
    root_composition_diagnostic_keywords=source_receipt['root_composition_diagnostic_keywords'],
    ordinary_phase_order=['v4 mass','local support residual','global support residual'],ordinary_budget='floor(Q*N) shared',
    shared_reservations=['children','parents','supported cell/group ledger','gross refined certificates','gross global certificates from both previous phases'],
    global_death_policy='flagged outside only',global_parent_policy='unflagged inside p>Q unique parents',
    opposite_actions_replenish_certificates=False,
    isolation_budget='unchanged small flagged remainder; may add actions beyond ordinary budget as in v4',
    cpu_action_cases=14,statistical_contracts=5,independent_audit_status='PASS',performance_cpu_exact_proof='PASS, separately frozen helper/method splices',
    quality_gates_unchanged=True,seed_experiments=False,new_seeds=0,optimizer_updates=0,cpu_only=True,
    saved_response=[{k:r[k] for k in ('name','mass_moves','local_moves','global_moves','ordinary_moves','remaining_ordinary_budget',
        'global_births_without_local_discovery','remaining_physical_birth_capacity','remaining_flagged_rows')}
        for r in checks['response-diagnosis.json']['rows']],
    gpu_command=['/tmp/pr38-default-env/bin/python','-u','-B',str(HERE/'contract_check.py'),
        '--package-root',str(PACKAGE),'--output',str(HERE/'gpu-contract'),'--device','cuda:0'],
    scope='Fixed conditional family and action accounting qualification; learned quality and high-dimensional support law unqualified',
    integration_status='Final separate proposed count law, root selects one before quality; earlier artifacts remain immutable',
    known_remaining=['root-only composed CUDA contract','canonical matched toy/MNIST/replay','5% ordinary action cap',
        'high-dimensional support-law qualification','existing rare support false positives','adaptive-head/FIFO dependence'])
output=HERE/'READY.json';assert not output.exists()
output.write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(dict(status=ready['status'],ready_sha256=sha(output),feature_sha256=sha(feature),
    package_files=len(package_map),local_files=len(local_map),numerical_files=len(numerical_map),
    evidence_files=len(evidence),external_files=len(external)),indent=2))
