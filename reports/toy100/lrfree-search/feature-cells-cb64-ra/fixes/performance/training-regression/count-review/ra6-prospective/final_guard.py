"""Final read-only guard of completed RA6 toy artifacts and prior CPU receipts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
ACCEPTED = HERE / 'accepted-attempt2'
VALIDATION = ROOT / 'validation-cb64-ra6'
RUN = VALIDATION / 'learned/training/toy/CB64-RA6'
AUDIT = ROOT / 'integration/review/ra6-final-learned-artifact-audit'
STEPS = [0,100,250,500,750,1000,1250,1500,1750,2000]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
write = lambda p,v: Path(p).write_text(json.dumps(v,indent=2,allow_nan=True)+'\n')
assert not (HERE/'FINAL-RECEIPT.json').exists(), 'final receipt already exists'
checked = {}

def check_map(values):
    for name, expected in values.items():
        p = Path(name)
        assert sha(p)==expected, str(p)
        checked[str(p)] = expected

def check_freeze(p):
    frozen=read(p)
    for key in ('files','artifacts','reviewed_hashes'):
        if key in frozen: check_map(frozen[key])
    checked[str(p)] = sha(p)

check_freeze(HERE/'INITIAL-FROZEN.json')
check_freeze(HERE/'SOURCE-FROZEN.json')
check_freeze(ACCEPTED/'SOURCE-FROZEN.json')
source=read(ACCEPTED/'SOURCE-RECEIPT.json')
assert source['status']=='VALID' and source['package_sha256']=='6bb967c405c486d2a5cbd5dcc3bce7d1bd2cb93b3f9c7e916229f1327a8e5fc5'
check_map(source['reviewed_hashes'])
freeze=read(VALIDATION/'source-freeze.json')
check_map({str(VALIDATION/k):v for k,v in freeze['local_sources'].items()})
check_map(freeze['external_sources'])
summary=read(ACCEPTED/'summary.json')
assert summary['status']=='VALID' and summary['sealed_steps']==STEPS
assert summary['completed_checkpoint_audits']==10 and summary['training_updates']==0
assert summary['cpu_only'] and not summary['cuda_initialized']
endpoints=[]
for step in STEPS:
    check_freeze(ACCEPTED/f'checkpoint-{step:04d}-FROZEN.json')
    row=read(ACCEPTED/f'checkpoint-{step:04d}.json')
    assert row['status']=='VALID' and row['step']==step
    assert row['optimizer_updates']==0 and row['new_seeds']==0 and not row['cuda_initialized']
    assert row['checks']['semantic_timing_fields'] in ([],['trainer.birth_death.last.eval_seconds'])
    assert row['checks']['old_law_atomic_rejections']==['old-law','wrong-level','mask-dtype']
    endpoints.append(dict(step=step,receipt_sha256=sha(ACCEPTED/f'checkpoint-{step:04d}.json'),
        checkpoint_sha256=row['checkpoint_sha256'],population=row['population'],serving=row['serving']))
original=read(AUDIT/'summary.json')
assert original['source_integrity']['status']=='VALID' and not original['cuda_initialized'] and original['cpu_only']
toy=original['records']['training-toy']
assert toy['primary_status']=='COMPLETE' and toy['evidence_status']=='VALID'
gate=toy['toy_quality_gate']
assert gate==dict(status='FAIL',checks=dict(precision=False,coverage=False,mass_tv=False),
    thresholds=dict(precision_min=.9,coverage=25,mass_tv_max=.1))
assert all(original['records'][k]['primary_status']=='PENDING' for k in ('training-mnist','replay-toy','replay-mnist'))
result=read(RUN/'result.json')
assert result['status']=='COMPLETE' and result['steps']==2000 and result['variant']=='CB64-RA6'
assert result['final']==toy['final']
check_map({str(RUN/k):v for k,v in result['checkpoint_sha256'].items()})
events=[]
for line in (VALIDATION/'run.log').read_text().splitlines():
    try:events.append(json.loads(line))
    except json.JSONDecodeError:pass
done=[e for e in events if e.get('event')=='job_complete' and e.get('name')=='learned-toy-CB64-RA6']
assert len(done)==1 and done[0]['returncode']==0 and done[0]['result_sha256']==sha(RUN/'result.json')
assert len([e for e in events if e.get('event')=='job_start'])==1
for p in [HERE/'audit_checkpoints_v2.py',HERE/'WATCHER.json',HERE/'watcher.log',ACCEPTED/'summary.json',
        AUDIT/'summary.json',AUDIT/'REPORT.md',AUDIT/'AUDITOR-IDENTITY.json',ROOT/'integration/review/audit_learned.py',
        RUN/'result.json',RUN/'metrics.jsonl',RUN/'config.json',VALIDATION/'run.log',VALIDATION/'logs/learned-toy-CB64-RA6.log',
        HERE/'final_guard.py']:
    checked[str(p)]=sha(p)
receipt=dict(status='VALID',created_at=datetime.now(timezone.utc).isoformat(),candidate='CB64-RA6',
    evidence_status='VALID',quality_status='FAIL',quality_gate=gate,final_metrics=toy['final']['metrics'],
    training_seconds=result['training_seconds'],sealed_checkpoints=STEPS,original_artifact_auditor=str(AUDIT),
    preserved_initial_and_failed_checker_receipts=True,source_count=len(source['reviewed_hashes']),
    checks=dict(final_source_maps_exact=True,all_ten_endpoint_freezes_exact=True,
        original_typed_fingerprints_and_init_cursor_audit='VALID',named_index_api_declared_before_execution=True,
        trainer_schema=5,backend_schema=6,population_law_and_atomic_old_state_rejection=True,
        graph_birth_copy_count_ledgers='VALID',semantic_timing_only_existing_last_eval_seconds=True,
        quality_gates_unchanged=True),
    population_trace=endpoints,
    pending=dict(mnist='NOT RUN',replay='NOT RUN',grid100='NOT RUN; root stopped this candidate after toy FAIL'),
    target='Both strict learned toy and original full Grid100 must pass; this candidate does not meet the target.',
    cuda_initialized=False,model_optimizer_updates=0,new_seeds=0,verified_hashes=checked)
write(HERE/'FINAL-RECEIPT.json',receipt)
final=endpoints[-1]['population']
report=f'''# RA6 completed toy artifact audit

Evidence VALID; strict toy quality FAIL: emitted precision {receipt['final_metrics']['precision']:.9f}, coverage {receipt['final_metrics']['coverage']}/25, TV {receipt['final_metrics']['mass_tv']:.9f}. Thresholds remain precision ≥.90, all 25 modes, TV ≤.10. Training completed 2,000 updates in {result['training_seconds']:.2f}s. Grid100, MNIST and replay were not run; the required toy-and-grid target is unmet.

All ten sealed checkpoint receipts and their artifacts are unchanged. The original frozen learned artifact auditor independently verifies their typed GPU fingerprints using CPU storage, initialization, sources, data cursors, configuration and runtime. Its new output is `{AUDIT}`. {len(source['reviewed_hashes'])} frozen source/input files match the prospective source guard. The earlier tuple/list checker failure and its frozen receipts remain preserved; accepted-attempt2 uses value-identical JSON recipe comparison.

Population certificates were never accepted: two negative direction decisions were rejected for population coverage, with zero expiries. The last rejection at904 tested b32 with525/973 participants; it advanced b to64. Final table s=1, b=64, last_decisive=+1, active=false, {final['rebases']} rebases; current finite-pair participants are204 at b and53 at2b. Serving remains FAST under the unchanged predicate; EMA update weight is1/256. These are table participation diagnostics, not per-row stationarity or G/D convergence claims.

Strict population-law checkpoint rejection, backend6/trainer5, bounded symmetric lineage graph, copy/novel reset hook and shared count/birth budgets are valid at every sealed endpoint. Semantic state contains no timings beyond the inherited birth_death.last.eval_seconds. No CUDA context, model/optimizer update, new seed, trajectory or evaluator change was made by this audit.
'''
(HERE/'FINAL-REVIEW.md').write_text(report)
files=dict(checked)
for p in [HERE/'FINAL-RECEIPT.json',HERE/'FINAL-REVIEW.md']:
    files[str(p)]=sha(p)
write(HERE/'FINAL-FROZEN.json',dict(status='VALID',quality_status='FAIL',files=files))
print(json.dumps(dict(status='VALID',quality_status='FAIL',source_files=len(source['reviewed_hashes']),
    receipt=str(HERE/'FINAL-RECEIPT.json'),receipt_sha256=sha(HERE/'FINAL-RECEIPT.json'),
    freeze_sha256=sha(HERE/'FINAL-FROZEN.json'))))
