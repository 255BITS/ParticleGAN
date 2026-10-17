"""Freeze the completed RA7 toy saved-state audit; CPU loads, no training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import torch
torch.set_num_threads(1)

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
OUT=HERE/'accepted-attempt2'
LANE=ROOT/'validation-cb64-ra7'
RUN=LANE/'learned/training/toy/CB64-RA7'
ORIGINAL=ROOT/'integration/review/ra7-final-learned-artifact-audit'
STEPS=[0,100,250,500,750,1000,1250,1500,1750,2000]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,allow_nan=True)+'\n')
assert not (HERE/'FINAL-RECEIPT.json').exists()
checked={}

def check_map(mapping):
    for name,expected in mapping.items():
        assert sha(name)==expected,name
        checked[name]=expected

def check_freeze(path):
    record=read(path)
    for key in ('files','artifacts','reviewed_hashes'):
        if key in record:check_map(record[key])
    checked[str(path)]=sha(path)

check_freeze(HERE/'CHECKER-FROZEN.json')
check_freeze(HERE/'INITIAL-FROZEN.json')
check_freeze(OUT/'SOURCE-FROZEN.json')
source=read(OUT/'SOURCE-RECEIPT.json');assert source['status']=='VALID'
assert source['package_sha256']=='671404988209f615aefd1aec32ab1ef0a51807a9b07514c9bb35a662a6154c7c'
summary=read(OUT/'summary.json')
assert summary['status']=='VALID' and summary['sealed_steps']==STEPS and summary['completed_checkpoint_audits']==10
assert summary['training_updates']==0 and not summary['cuda_initialized'] and summary['cpu_only']
endpoints=[]
for step in STEPS:
    check_freeze(OUT/f'checkpoint-{step:04d}-FROZEN.json')
    row=read(OUT/f'checkpoint-{step:04d}.json')
    assert row['status']=='VALID' and row['optimizer_updates']==row['new_seeds']==0
    assert not row['cuda_initialized'] and row['quality_verdict'] is None
    assert row['base_learning_rates']==[[.0010625,.0085,.0010625],[.00425]]
    assert row['checks']['old_law_atomic_rejections']==['old-law','wrong-level','mask-dtype']
    assert row['checks']['semantic_timing_fields'] in ([],['trainer.birth_death.last.eval_seconds'])
    endpoints.append(dict(step=step,receipt_sha256=sha(OUT/f'checkpoint-{step:04d}.json'),
        checkpoint_sha256=row['checkpoint_sha256'],population=row['population'],serving=row['serving'],
        base_learning_rates=row['base_learning_rates'],applied_learning_rates=row['applied_learning_rates'],
        last_phase=row['last_phase']))
watcher=read(HERE/'WATCHER.json');proc=Path(f"/proc/{watcher['pid']}/stat")
if proc.exists():
    words=proc.read_text().split()
    assert int(words[21])!=watcher['startticks'] or words[2]=='Z','watcher still running'
frozen=read(LANE/'source-freeze.json')
check_map({str(LANE/name):expected for name,expected in frozen['local_sources'].items()})
check_map(frozen['external_sources'])
events=[]
for line in (LANE/'run.log').read_text().splitlines():
    try:events.append(json.loads(line))
    except json.JSONDecodeError:pass
done=[e for e in events if e.get('event')=='job_complete' and e.get('name')=='learned-toy-CB64-RA7']
assert len(done)==1 and done[0]['returncode']==0 and done[0]['result_sha256']==sha(RUN/'result.json')
result=read(RUN/'result.json');assert result['status']=='COMPLETE' and result['steps']==2000
check_map({str(RUN/name):expected for name,expected in result['checkpoint_sha256'].items()})
original=read(ORIGINAL/'summary.json');toy=original['records']['training-toy']
assert toy['evidence_status']=='VALID' and toy['primary_status']=='COMPLETE'
assert original['source_integrity']['status']=='VALID' and original['cpu_only'] and not original['cuda_initialized']
assert toy['final']==result['final']
gate=toy['toy_quality_gate']
assert gate['thresholds']==dict(precision_min=.9,coverage=25,mass_tv_max=.1)

# Inspect the latest committed novel births. These moments were overwritten
# after update2000, so their reset is directly observable before another update.
cpu_rng=torch.get_rng_state().clone()
path=RUN/'checkpoint-2000.pt';before=sha(path)
state=torch.load(path,map_location='cpu',weights_only=False)['trainer']
assert sha(path)==before==endpoints[-1]['checkpoint_sha256']
bd=state['birth_death'];assert bd['last']['step']==2000
children=bd['last']['novel_birth']['child_rows']
prior_group=state['optimizers'][0]['param_groups'][1]
assert len(prior_group['params'])==1
moments=state['optimizers'][0]['state'][prior_group['params'][0]]
for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
    assert bool((moments[key][children]==0).all())
assert float(moments['step'])==2000
assert bool((state['optimizers'][0]['regularizer']['latent']['history'][children]==0).all())
for key in ('M','Qs','W','S','flag'):
    assert bool((state['row_evidence'][key][children]==0).all())
assert bool((bd['lineage_neighbors'][children]==-1).all())
assert torch.equal(cpu_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
for p in [HERE/'final_guard.py',OUT/'summary.json',HERE/'watcher.log',RUN/'config.json',RUN/'metrics.jsonl',
        RUN/'result.json',LANE/'run.log',LANE/'logs/learned-toy-CB64-RA7.log',
        ROOT/'integration/review/audit_learned.py',ORIGINAL/'summary.json',ORIGINAL/'REPORT.md',ORIGINAL/'AUDITOR-IDENTITY.json']:
    checked[str(p)]=sha(p)
receipt=dict(status='VALID',utc=datetime.now(timezone.utc).isoformat(),candidate='CB64-RA7',
    scope='completed original toy phase; saved CPU state/API/population/ledger audit',
    evidence_status='VALID',original_saved_metric_gate=gate,no_new_quality_emission=True,
    final_metrics=toy['final']['metrics'],training_seconds=result['training_seconds'],
    source_count=len(source['reviewed_hashes']),sealed_steps=STEPS,endpoints=endpoints,
    checks=dict(source_and_initialization_cursor_exact=True,original_gpu_typed_fingerprints_cpu_audit='VALID',
        named_index_API_declared_before_run=True,all_ten_schema_population_rejections_graph_ledgers_JSON_pass=True,
        genuine_quarter_G_and_sigma_base_rates=True,prior_D_base_rates_exact=True,
        final_novel_moments_history_evidence_graph_zero=True,shared_optimizer_step_retained=2000,
        no_semantic_timings_except_inherited_last_eval_seconds=True,private_attempt_failure_preserved=True),
    final_novel_children=children,quality_target='Both strict toy and original full Grid100 must pass; target remains unmet.',
    other_phases='Grid100, MNIST and replay absent from this one-job phase; no qualification inferred.',
    model_optimizer_updates=0,new_seeds=0,cuda_initialized=False,global_cpu_rng_unchanged=True,verified_hashes=checked)
write(HERE/'FINAL-RECEIPT.json',receipt)
last=endpoints[-1]
report=f'''# RA7 completed toy saved-state audit

Evidence VALID across all ten sealed checkpoints. The original frozen artifact auditor independently reproduced typed GPU fingerprints from CPU storage and verified sources, initialization, data/stream cursors, recipe and runtime. Its unchanged saved-metric toy gate is {gate['status']}: {gate['checks']}. No samples, gradients, optimizer updates, replay, seed or CUDA context were produced by this audit. The toy-and-full-Grid100 target remains unmet.

All {len(source['reviewed_hashes'])} source/input maps and all endpoint freezes remain exact. Saved G and learned-sigma base rates are .0010625, prior .0085, critic .00425 at every endpoint; current applied rates are retained separately. Backend6/trainer5, strict population-law atomic rejection, graph/copy/count budgets, paired births and JSON semantic state pass. Timings remain limited to inherited birth_death.last.eval_seconds.

Final table s1, b64, no accepted population certificate and no expiry. The negative direction result at1464 used the b32/2b64 window; only2b was negative, with323/973 participants, so the next scale became64. At2000 potential participation is281 at b and75 at2b. The clock completes8 blocks plus tau24 since1464, exactly536 updates. Serving remainsFAST, EMA update weight1/256. Per-row gradient evidence remains immature in128 dimensions.

Latest reaction2000 has48 copies plus3 novel births,51 ordinary moves and0 isolation. The three novel rows {children} have zero optimizer moments, latent history, gradient evidence and lineage links, while the shared optimizer step stays2000. The initial source-construction indentation failure is preserved in failed-attempt1; its code-generation guard did not touch numerical or frozen production sources.
'''
(HERE/'FINAL-REVIEW.md').write_text(report)
for n in ('FINAL-RECEIPT.json','FINAL-REVIEW.md'):checked[str(HERE/n)]=sha(HERE/n)
write(HERE/'FINAL-FROZEN.json',dict(status='VALID',original_saved_metric_gate=gate['status'],files=checked))
print(json.dumps(dict(status='VALID',original_saved_metric_gate=gate['status'],
    receipt_sha256=sha(HERE/'FINAL-RECEIPT.json'),freeze_sha256=sha(HERE/'FINAL-FROZEN.json'),
    final_population=last['population'],final_novel_children=children)))
