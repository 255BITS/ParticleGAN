"""Read-only saved GPU profiler artifact audit; no Torch or CUDA import."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
AREA=Path(__file__).resolve().parent
PHASE=ROOT/'quality/group-profile-phase-v2/READY.json'
OWNER=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra4-quality/anchor-profile/READY.json'
OUTPUT=ROOT/'integration/review/group-count-gpu'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=json.loads(PHASE.read_text());owner=json.loads(OWNER.read_text())
result=json.loads((OUTPUT/'result.json').read_text())
launch=json.loads((OUTPUT/'LAUNCH.json').read_text())
phase_result=json.loads((OUTPUT/'PHASE-RESULT.json').read_text())
assert sha(PHASE)=='ed7da8badaf3372f72b22ca5935edc003d6b50716715fd5f401e5bc958c2de26'
assert len(frozen['source_sha256'])==88
assert all(sha(Path(p))==h for p,h in frozen['source_sha256'].items())
assert len(result['source_sha256'])==61
assert all(sha(Path(p))==h for p,h in result['source_sha256'].items())
assert set(result['source_sha256'])<=set(frozen['source_sha256'])
assert launch['command']==owner['gpu_command']
assert launch['phase_ready_sha256']==phase_result['phase_ready_sha256']==sha(PHASE)
assert phase_result['result_sha256']==sha(OUTPUT/'result.json')
assert launch['gpu_uuid']==phase_result['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
assert launch['source_integrity']==phase_result['source_integrity']=='VALID'
assert launch['numerical_parallelism']==phase_result['numerical_parallelism']==1
assert launch['started_utc']==phase_result['started_utc']<result['utc']<phase_result['finished_utc']
assert phase_result['original_supervisor_still_parked']
assert phase_result['status']==result['status']=='PASS' and phase_result['returncode']==0
assert result['device']=='cuda:0'
assert result['optimizer_updates']==result['new_seeds']==0
assert result['quality_verdict'] is phase_result['quality_verdict'] is None
assert result['inputs_sha256']==owner['helper_source_sha256'][str(OWNER.parent/'inputs.pt')]
assert [r['step'] for r in result['records']]==[1250,2000]
records=[]
for r in result['records']:
    old,new=r['baseline'],r['candidate']
    assert r['complete_plan_bit_exact']
    for name in ['moves','attempted_cells','linearizations','callback_calls','complete_output_sha256','snapshot_work']:
        assert old[name]==new[name],(r['step'],name)
    assert old['moves']==old['attempted_cells']==4
    assert old['RNG_unchanged'] is new['RNG_unchanged'] is True
    assert old['parameter_gradients_untouched'] is new['parameter_gradients_untouched'] is True
    assert old['operators'].keys()==new['operators'].keys()
    for op in old['operators']:
        delta=old['operators'][op]['count']-new['operators'][op]['count']
        assert delta==(300 if op=='aten::nonzero' else 0),(r['step'],op)
    for side in [old,new]:
        assert len(side['wall_ms'])==3
        assert statistics.median(side['wall_ms'])==side['median_wall_ms']
    records.append(dict(step=r['step'],full_plan_sha256=old['complete_output_sha256'],RNG_unchanged=True,snapshot_work_exact=True,nonzero_counts=[old['operators']['aten::nonzero']['count'],new['operators']['aten::nonzero']['count']],scalar_reads=new['operators']['aten::_local_scalar_dense']['count'],svd_calls=new['operators']['aten::_linalg_svd']['count'],callbacks=new['callback_calls'],median_wall_ms=[old['median_wall_ms'],new['median_wall_ms']],descriptive_reduction_pct=100*(1-new['median_wall_ms']/old['median_wall_ms']),candidate_scalar_cpu_total_ms=new['operators']['aten::_local_scalar_dense']['cpu_total_ms'],candidate_svd_cpu_total_ms=new['operators']['aten::_linalg_svd']['cpu_total_ms']))
receipt=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),scope='independent read-only provenance and metadata audit of the existing paired CUDA probe',source_sha256={str(p):sha(p) for p in [PHASE,OWNER,*sorted(OUTPUT.iterdir())]},phase_guard_count=88,recorded_input_guard_count=61,verified_phase_inputs_sha256=frozen['source_sha256'],records=records,numerical_reruns=0,cuda_imports=0,quality_verdict=None,limits=['Full plan equality was executed by the guarded frozen runner; this review verifies its provenance, assertions and byte hashes without recalculating GPU plans.','GPU probe covers birth planning only on two saved RA4 clean-table fixtures; CPU owner proof separately covers complete47-copy+4-birth actuation.','Three warm repetitions per side, reference then candidate order, one profile per side; no timing confidence interval or early32-SVD worst-case attribution.','Nested profiler CPU/device totals overlap and must not be added. Device sharing, clock changes and host scheduling are outside this fixed artifact proof.','No learned training, quality screen or historical RA6 action was rerun. RA6 learned toy quality remains FAIL.'],next_exact_performance_proposal='Bundle solver progress/acceptance scalars into one small detached CPU packet per evaluated iterate, retaining original source-dtype GPU comparisons, algebra, early exits and full trace. Reuse last progress scalars at return. This targets399/368 remaining scalar reads without changing SVD dimensions or solver precision; it requires a separately frozen fixed-input proof before use.')
(AREA/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status='PASS',records=records,receipt=str(AREA/'receipt.json')),indent=2))
