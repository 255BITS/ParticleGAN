"""Authoritative post-exit seal of the one selected fixed saved diagnostic."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not (HERE/'FROZEN.json').exists()
assert sha(HERE/'PREPARATION-FROZEN.json')=='c97b1e916b9e44156a687cd6a51305e257684417140c0b5a1f5aa486ed58347c'
assert sha(HERE/'probe.py')=='f4b0404ea39a2713585a2337d539808772f1dfd9e4d8ae160a6144f3fddad8ec'
pre=read(HERE/'PREPARATION-FROZEN.json')
for path,digest in pre['source_and_input_sha256'].items():assert sha(path)==digest,path
r=read(HERE/'result.json')
assert r['status']=='VALID_FIXED_SAVED_DIAGNOSTIC' and r['saved_step']==7000 and r['charts']==1
assert r['numerical_state_and_input_bytes_unchanged'] and r['global_Torch_NumPy_trainer_rng_unchanged']
assert r['state_before_sha256']==r['state_after_sha256']
assert r['model_constructions']==r['training_updates']==r['optimizer_steps']==r['new_emitted_samples']==r['new_seeds']==r['actions']==r['counterfactual_quality_scores']==0
assert not r['cuda_initialized'] and r['quality_verdict'] is None and r['statistical_certificate'] is None
assert r['original_result']['status']=='FAIL'
log=(HERE/'attempt1.log').read_text().strip().splitlines()
assert len(log)==1
last=json.loads(log[-1]);assert last['status']==r['status'] and last['result_sha256']==sha(HERE/'result.json')
receipt=dict(status=r['status'],evidence_status='VALID',quality_verdict=None,original_quality='RA9 full grid FAIL preserved',
    scope='One frozen final-state128/r8 real-only local-moment feasibility diagnostic, no actions or scoring',
    pre_numerical_freeze_sha256=sha(HERE/'PREPARATION-FROZEN.json'),result_sha256=sha(HERE/'result.json'),
    helper_sha256=sha(HERE/'probe.py'),source_and_input_files=len(pre['source_and_input_sha256']),
    chart_groups=r['chart']['groups'],backend_schema=8,trainer_schema=5,process_exited=True,
    one_run=True,failed_numerical_attempts=0,training_updates=0,new_emissions=0,new_seeds=0,cuda=False,
    report_sha256=sha(HERE/'REPORT.md'),A_EMA=r['anchor_reference_agreement']['EMA']['A_even_within_variance_standardized'],
    positive_EMA_groups=r['anchor_reference_agreement']['EMA']['positive_dot_groups'],
    limits=['Shared trained-D/FIFO dependence','Aggregate is not per-group authority','Covariance subtraction is not a calibrated target'])
with (HERE/'receipt.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
files={str(p):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
frozen=dict(status=r['status'],post_exit=True,frozen_utc=datetime.now(timezone.utc).isoformat(),
    receipt_sha256=sha(HERE/'receipt.json'),files=files,source_and_input_sha256=pre['source_and_input_sha256'],quality_verdict=None)
with (HERE/'FROZEN.json').open('x') as f:f.write(json.dumps(frozen,indent=2)+'\n')
for path,digest in {**files,**pre['source_and_input_sha256']}.items():assert sha(path)==digest,path
print(json.dumps(dict(status=r['status'],receipt_sha256=sha(HERE/'receipt.json'),
    freeze_sha256=sha(HERE/'FROZEN.json'),result_sha256=sha(HERE/'result.json'))))
