"""Seal completed RA8 toy evidence after all private audit processes exit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
OUT=HERE/'accepted-attempt1'
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
LANE=ROOT/'validation-cb64-ra8'
RUN=LANE/'learned/training/toy/CB64-RA8'
ORIGINAL=ROOT/'integration/review/ra8-final-learned-artifact-audit'
STEPS=[0,100,250,500,750,1000,1250,1500,1750,2000]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,allow_nan=True)+'\n')
assert not (HERE/'FINAL-FROZEN.json').exists()
checked={}
def check_map(values):
    for name,expected in values.items():
        assert sha(name)==expected,name
        checked[name]=expected
def check_freeze(path):
    record=read(path)
    for key in ('files','artifacts','reviewed_hashes','original_checker'):
        if key in record:check_map(record[key])
    checked[str(path)]=sha(path)
for path in (HERE/'CHECKER-FROZEN.json',HERE/'INITIAL-FROZEN.json',OUT/'SOURCE-FROZEN.json'):
    check_freeze(path)
summary=read(OUT/'summary.json')
assert summary['status']=='VALID' and summary['sealed_steps']==STEPS and summary['completed_checkpoint_audits']==10
assert summary['cpu_only'] and not summary['cuda_initialized'] and summary['training_updates']==0
watcher=read(HERE/'WATCHER.json');stat=Path(f"/proc/{watcher['pid']}/stat")
if stat.exists():
    words=stat.read_text().split()
    assert int(words[21])!=watcher['startticks'] or words[2]=='Z','private watcher still writing'
source=read(OUT/'SOURCE-RECEIPT.json')
assert source['status']=='VALID' and source['package_sha256']=='da29f10340ddd0cf4c8452e234496579699bd8ea7e1154da22ffeac18796065c'
endpoints=[]
for step in STEPS:
    check_freeze(OUT/f'checkpoint-{step:04d}-FROZEN.json')
    row=read(OUT/f'checkpoint-{step:04d}.json')
    assert row['status']=='VALID' and row['quality_verdict'] is None
    assert row['optimizer_updates']==row['new_seeds']==0 and not row['cuda_initialized']
    assert row['base_learning_rates']==[[.0010625,.0085,.0010625],[.00425]]
    assert row['checks']['paired_average_atomic_rejections']==['old-backend6','wrong-geometry-policy','boolean-step','future-step']
    endpoints.append(dict(step=step,checkpoint_sha256=row['checkpoint_sha256'],population=row['population'],
        paired_average=row['paired_average'],serving=row['serving'],last_phase=row['last_phase'],
        base_learning_rates=row['base_learning_rates'],applied_learning_rates=row['applied_learning_rates']))
result=read(RUN/'result.json')
assert result['status']=='COMPLETE' and result['steps']==2000
check_map({str(RUN/name):expected for name,expected in result['checkpoint_sha256'].items()})
original=read(ORIGINAL/'summary.json');toy=original['records']['training-toy']
assert original['cpu_only'] and not original['cuda_initialized'] and original['source_integrity']['status']=='VALID'
assert toy['evidence_status']=='VALID' and toy['primary_status']=='COMPLETE' and toy['final']==result['final']
gate=toy['toy_quality_gate']
assert gate['status']=='PASS' and all(gate['checks'].values())
assert gate['thresholds']==dict(precision_min=.9,coverage=25,mass_tv_max=.1)
# One saved final-state reset inspection; no model construction or gradient.
rng=torch.get_rng_state().clone()
path=RUN/'checkpoint-2000.pt';before=sha(path)
state=torch.load(path,map_location='cpu',weights_only=False)['trainer']
assert sha(path)==before==endpoints[-1]['checkpoint_sha256']
bd=state['birth_death'];stamp=bd['paired_average']
assert bd['backend_schema']==7 and state['schema']==5
assert stamp['step']==2000 and stamp['snapshot']==250 and bd['rows_since_eval']==0
assert stamp['coherent_rows']==977 and stamp['required']==973 and stamp['eligible']
children=bd['last']['novel_birth']['child_rows']
group=state['optimizers'][0]['param_groups'][1]
moments=state['optimizers'][0]['state'][group['params'][0]]
for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
    assert bool((moments[key][children]==0).all())
assert float(moments['step'])==2000
assert bool((state['optimizers'][0]['regularizer']['latent']['history'][children]==0).all())
for key in ('M','Qs','W','S','flag'):
    assert bool((state['row_evidence'][key][children]==0).all())
assert bool((bd['lineage_neighbors'][children]==-1).all())
assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
for p in (Path(__file__),OUT/'summary.json',HERE/'watcher.log',HERE/'original-artifact-attempt1.log',
          RUN/'config.json',RUN/'metrics.jsonl',RUN/'result.json',LANE/'logs/learned-toy-CB64-RA8.log',
          ROOT/'integration/review/audit_learned.py',ORIGINAL/'summary.json',ORIGINAL/'REPORT.md',ORIGINAL/'AUDITOR-IDENTITY.json'):
    checked[str(p)]=sha(p)
receipt=dict(status='VALID',scope='Completed original RA8 toy saved-checkpoint evidence; independent CPU audit',
    evidence_status='VALID',source_guards=len(source['reviewed_hashes']),sealed_steps=STEPS,
    endpoints=endpoints,original_saved_metric_gate=gate,final_metrics=toy['final']['metrics'],
    backend_schema=7,trainer_schema=5,named_index_API_frozen=True,
    typed_GPU_initialization_fingerprints_and_input_stream_cursors_original_auditor='VALID',
    population_LR_law_unchanged=True,geometry_serving_independent_from_stationarity=True,
    geometry_epoch_and_FIFO_history_exact=True,paired_geometry_new_fields_scalar_no_timings=True,
    final_newborn_rows=children,final_newborn_moments_history_evidence_graph_reset=True,
    no_private_attempt_failed_in_this_lane=True,watcher_exited=True,
    overall_quality='Toy PASS; original full Grid100 still required and running. Candidate unqualified.',
    model_forwards=0,gradients=0,optimizer_updates=0,new_emissions=0,new_seeds=0,cuda_initialized=False,
    global_cpu_rng_unchanged=True,verified_hashes=checked)
write(HERE/'FINAL-RECEIPT.json',receipt)
report='''# RA8 completed toy evidence audit

Evidence VALID at all ten original sealed checkpoints. All 165 frozen source/input guards remain exact. The unchanged original artifact auditor validates typed GPU initialization fingerprints from CPU storage, inputs, stream/data cursors, configuration and the saved result. Its strict saved-metric toy gate is PASS: precision .96533203125, 25 modes, mass TV .0521142578125. No quality samples, model forwards, gradients, optimizer updates, seeds or CUDA context were produced by this audit.

Backend7/trainer5 are intentional. Typed geometry state, same reaction/snapshot chart identity, exact FIFO age/history, scalar semantic fields and old6/policy/boolean-step/future-step rejection pass at every endpoint. Population, row resets, bounded symmetric lineage, copy/novel budgets, count multiplicity, certificate accounting, JSON and stored RNG placement also pass. The final newborns have zero own optimizer/history/evidence/lineage entries with the shared step2000 retained.

Every earlier recorded checkpoint geometry lease vetoes averaging; at1750 joint963/973 and age768 real rows. At2000 joint977/973, snapshot250, age0 permits the paired average. Table stationarity remains inactive with s1/b64; no 973-row optimizer requirement was relaxed. This is an empirical anti-blur lease, not a stationarity, distribution-equivalence or every-update emitted support guarantee. Intermediate reaction decisions between saved milestones are not reconstructed.

The original full Grid100 remains required and is running under the same frozen package. This toy evidence does not qualify the combined target. All private audit logs were closed before this final seal; the evolving global validation journal is deliberately not pinned in this toy-only receipt. Other learned phases/replay are not claimed here.
'''
(HERE/'FINAL-REVIEW.md').write_text(report)
files={str(p):sha(p) for p in (HERE/'FINAL-RECEIPT.json',HERE/'FINAL-REVIEW.md',Path(__file__))}
write(HERE/'FINAL-FROZEN.json',dict(status='VALID',files=files,reviewed_hashes=checked))
for name,expected in {**files,**checked}.items():assert sha(name)==expected,name
print(json.dumps(dict(status='VALID',toy_quality_gate='PASS',receipt_sha256=sha(HERE/'FINAL-RECEIPT.json'),
    freeze_sha256=sha(HERE/'FINAL-FROZEN.json'),final_joint=977,required=973,full_Grid100_pending=True)))
