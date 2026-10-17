"""Freeze private exact planner sources, AST splice hashes, tests and command."""
import ast
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import plan_pair_common as common
import plan_pair_final as final
from make_plan_proposal import ast_digest,node_source

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'integration/review/training-regression/global-count/pkg-global-count'
PACKAGE=HERE/'pkg-PLAN-FINAL'
INPUT=BASE.parent/'inputs.pt'
CONTRACT=BASE.parent/'contract_cases.py'


def main():
    ready=HERE/'READY.json';patch=HERE/'PLAN.patch'
    if ready.exists() or patch.exists():raise RuntimeError('Frozen artifacts already exist')
    receipt=json.loads((HERE/'final-cpu-02.json').read_text())
    assert receipt['status']=='PASS' and receipt['exhaustive_integer_cases']==6144
    assert len(receipt['cases'])==14 and receipt['quota_checks']['calls']==100
    maps=[common.sources(p) for p in (BASE,PACKAGE)]
    assert receipt['base_source_sha256']==maps[0] and receipt['source_sha256']==maps[1]
    assert common.sha(INPUT)==receipt['input_sha256'] and common.sha(CONTRACT)==receipt['contract_sha256']
    splices=json.loads((HERE/'final-splices.json').read_text())
    base=(BASE/'particlegan/feature_cells.py').read_text()
    source=(PACKAGE/'particlegan/feature_cells.py').read_text()
    assert common.sha(BASE/'particlegan/feature_cells.py')==splices['base_source_sha256']
    assert common.sha(PACKAGE/'particlegan/feature_cells.py')==splices['proposal_source_sha256']
    law=final.law_checks(BASE,PACKAGE)
    assert set(law['unchanged_count_class_except'])==final.PERFORMANCE_METHODS
    patch.write_text(''.join(difflib.unified_diff(base.splitlines(keepends=True),source.splitlines(keepends=True),
        fromfile='a/particlegan/feature_cells.py',tofile='b/particlegan/feature_cells.py')))
    subprocess.run(['git','apply','--check',str(patch)],cwd=BASE,check=True)
    helper=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_group_integer_allocate')
    assert ast_digest(helper)==splices['ast_splices'][0]['proposal_ast_sha256']
    helper_sha=hashlib.sha256(node_source(source,helper).encode()).hexdigest()
    artifacts=['PLAN.patch','PLAN-PROTOCOL.md','PLAN-REPORT.md','make_plan_proposal.py',
        'freeze_plan_receipt.py','test_final_plan_cpu.py','plan_pair_final.py','plan_pair_common.py',
        'profile_plan_pair_gpu.py','mst_pair_checks.py','final-splices.json',
        'final-cpu-01.json','final-cpu-01.log','final-cpu-02.json','final-cpu-02.log','freeze-attempt1.json']
    command=['/tmp/pr38-default-env/bin/python',str(HERE/'profile_plan_pair_gpu.py'),
        '--reference-package-root',str(BASE),'--package-root','COMBINED_PACKAGE',
        '--input',str(INPUT),'--contract-root',str(BASE.parent),'--output','FRESH_GPU_JSON']
    result=dict(status='CPU_VALIDATED_ROOT_GPU_PENDING',created_at=datetime.now(timezone.utc).isoformat(),
        package_root=str(PACKAGE),package_sha256=hashlib.sha256(json.dumps(maps[1],sort_keys=True).encode()).hexdigest(),
        package_digest_rule='sha256(json.dumps(package_source_sha256,sort_keys=True).encode())',
        package_source_sha256=maps[1],base_package_root=str(BASE),base_source_sha256=maps[0],
        numerical_source=str(PACKAGE/'particlegan/feature_cells.py'),numerical_source_sha256=common.sha(PACKAGE/'particlegan/feature_cells.py'),
        helper_source_sha256={'_group_integer_allocate':helper_sha},ast_splices=splices['ast_splices'],
        original_method_ast_sha256=law['original_method_ast_sha256'],proposal_method_ast_sha256=law['proposal_method_ast_sha256'],
        patch_path=str(patch),patch_sha256=common.sha(patch),patch_applies_to_base=True,
        artifact_sha256={str(HERE/name):common.sha(HERE/name) for name in artifacts},
        fixed_inputs={str(INPUT):common.sha(INPUT),str(CONTRACT):common.sha(CONTRACT)},
        cpu_tests=dict(exhaustive_integer_cases=6144,complete_planner_variants=14,captured_grouped_quota_calls=100),
        cpu_threads=1,cuda_initialized=False,new_seeds=0,quality_updates=0,
        law='exact final 3K+2 math/actions/RNG/accounting/state; bounded allocation/metadata optimization',
        statistics_gates_certificates_ledgers_unchanged=True,mst_unchanged=True,schema_changes=False,
        independent_prototype_review=str(ROOT/'performance/training-regression/count-review/PLAN-FROZEN.json'),
        independent_final_review_pending=True,gpu_command=command,
        gpu_runner=dict(path=str(HERE/'profile_plan_pair_gpu.py'),executed=False,
            owner='root physical GPU0 serial queue',same_final_count_law_required=True,optional_mst_included=False))
    ready.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(ready_sha256=common.sha(ready),patch_sha256=common.sha(patch),
        package_sha256=result['package_sha256'],numerical_source_sha256=result['numerical_source_sha256'])),flush=True)


if __name__=='__main__':main()
