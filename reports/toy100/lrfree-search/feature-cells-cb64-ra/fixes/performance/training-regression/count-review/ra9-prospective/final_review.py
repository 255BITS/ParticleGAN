"""Close the original RA9 toy saved-artifact audit; no model forwards or updates."""
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
LANE=ROOT/'validation-cb64-ra9'
RUN=LANE/'learned/training/toy/CB64-RA9'
ORIGINAL=ROOT/'integration/review/ra9-final-learned-artifact-audit'
STEPS=[0,100,250,500,750,1000,1250,1500,1750,2000]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,allow_nan=True)+'\n')
assert not (HERE/'FINAL-RECEIPT.json').exists()
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
for path in (HERE/'CHECKER-FROZEN.json',OUT/'SOURCE-FROZEN.json'):
    check_freeze(path)
summary=read(OUT/'summary.json')
assert summary['status']=='VALID' and summary['sealed_steps']==STEPS and summary['completed_checkpoint_audits']==10
assert summary['cpu_only'] and not summary['cuda_initialized'] and summary['training_updates']==0
watcher=read(HERE/'WATCHER.json');stat=Path(f"/proc/{watcher['pid']}/stat")
if stat.exists():
    fields=stat.read_text().rsplit(')',1)[1].split()
    assert int(fields[19])!=watcher['startticks'] or fields[0]=='Z','private watcher still writing'
source=read(OUT/'SOURCE-RECEIPT.json')
assert source['status']=='VALID' and source['backend_schema']==8 and source['trainer_schema']==5
assert source['package_sha256']=='2d00c77ae0e7545ac253ff81aa729158d69016be82edc117e5597bdd664b86ce'
endpoints=[]
for step in STEPS:
    check_freeze(OUT/f'checkpoint-{step:04d}-FROZEN.json')
    row=read(OUT/f'checkpoint-{step:04d}.json')
    assert row['status']=='VALID' and row['quality_verdict'] is None
    assert row['optimizer_updates']==row['new_seeds']==0 and not row['cuda_initialized']
    assert row['base_learning_rates']==[[.0010625,.0085,.0010625],[.00425]]
    assert row['checks']['paired_average_atomic_rejections']==['old-backend7','wrong-geometry-policy','boolean-step','future-step']
    endpoints.append(dict(step=step,checkpoint_sha256=row['checkpoint_sha256'],population=row['population'],
        paired_average=row['paired_average'],serving=row['serving'],last_phase=row['last_phase'],
        base_learning_rates=row['base_learning_rates'],applied_learning_rates=row['applied_learning_rates']))
result=read(RUN/'result.json')
assert result['status']=='COMPLETE' and result['steps']==2000
check_map({str(RUN/name):expected for name,expected in result['checkpoint_sha256'].items()})
original=read(ORIGINAL/'summary.json');toy=original['records']['training-toy']
assert original['cpu_only'] and not original['cuda_initialized'] and original['source_integrity']['status']=='VALID'
assert toy['evidence_status']=='VALID' and toy['primary_status']=='COMPLETE' and toy['final']==result['final']
check_map({str(LANE/name):digest for name,digest in toy['artifact_hashes'].items()})
original_gate=toy['toy_quality_gate']
assert original_gate['thresholds']==dict(precision_min=.9,coverage=25,mass_tv_max=.1)
metrics=result['final']['metrics']
quality_checks=dict(precision=metrics['precision']>=.9,coverage=metrics['coverage']==25,
    mass_tv=metrics['mass_tv']<=.1,supported_mass_length=len(metrics['supported_mass'])==25,
    min_mass=min(metrics['supported_mass'])>=.01)
assert original_gate['checks']=={k:quality_checks[k] for k in ('precision','coverage','mass_tv')}
assert original_gate['status']==('PASS' if all(original_gate['checks'].values()) else 'FAIL')
full_gate=dict(status='PASS' if all(quality_checks.values()) else 'FAIL',checks=quality_checks,
    thresholds=dict(precision_min=.9,coverage=25,mass_tv_max=.1,supported_mass_length=25,min_mass_min=.01))
# Inspect the existing final newborn state, without restoring a network or RNG.
rng=torch.get_rng_state().clone()
path=RUN/'checkpoint-2000.pt';before=sha(path)
state=torch.load(path,map_location='cpu',weights_only=False)['trainer']
assert sha(path)==before==endpoints[-1]['checkpoint_sha256']
bd=state['birth_death'];stamp=bd['paired_average']
assert bd['backend_schema']==8 and state['schema']==5 and bd['settings']['cells']==128
assert stamp['step']==2000 and stamp['snapshot']==250 and bd['rows_since_eval']==0
assert stamp==endpoints[-1]['paired_average']['stamp']
assert stamp['cells']==min(128,max(1,512//max(1,stamp['rank'])))==64
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
for p in (Path(__file__),OUT/'summary.json',HERE/'watcher.log',HERE/'WATCHER.json',HERE/'start_watcher.py',
          HERE/'initial-attempt1.log',HERE/'prepare-attempt1.log',HERE/'original-artifact-attempt1.log',
          RUN/'config.json',RUN/'metrics.jsonl',RUN/'result.json',LANE/'logs/learned-toy-CB64-RA9.log',
          ROOT/'integration/review/audit_learned.py',ORIGINAL/'summary.json',ORIGINAL/'REPORT.md',ORIGINAL/'AUDITOR-IDENTITY.json'):
    checked[str(p)]=sha(p)
receipt=dict(status='VALID',evidence_status='VALID',scope='All ten original RA9 toy saved-artifact endpoints, CPU-only independent review',
    source_guards=len(source['reviewed_hashes']),sealed_steps=STEPS,endpoints=endpoints,
    original_artifact_auditor_unchanged=True,original_auditor_toy_subgate=original_gate,
    unchanged_root_full_toy_gate=full_gate,final_metrics=metrics,backend_schema=8,trainer_schema=5,
    requested_cells=128,actual_cells=64,count_multiplicity=194,
    named_index_API_frozen=True,matched_inputs_initialization_runtime_and_stream_cursors=True,
    population_LR_law_unchanged=True,geometry_serving_independent_from_stationarity=True,
    geometry_epoch_and_FIFO_history_exact=True,typed_resolution_count_partition_exact=True,
    scalar_geometry_fields_without_timings=True,final_newborn_rows=children,
    final_newborn_moments_history_evidence_graph_reset=True,watcher_exited=True,
    overall_quality='Toy '+full_gate['status']+'; full original Grid100 remains separately required. Candidate unqualified by this toy-only audit.',
    model_forwards=0,gradients=0,optimizer_updates=0,new_emissions=0,new_seeds=0,cuda_initialized=False,
    global_cpu_rng_unchanged=True,verified_hashes=checked)
write(HERE/'FINAL-RECEIPT.json',receipt)
report=f'''# RA9 completed toy evidence audit

Evidence VALID at all ten original sealed checkpoints. All {len(source['reviewed_hashes'])} source/input guards remain exact. The unchanged original completed-artifact auditor confirms saved source/runtime/input/initialization/cursor and artifact hashes. The full unchanged root toy gate, including minimum supported mode mass .01, is {full_gate['status']}: P {metrics['precision']}, modes {metrics['coverage']}, TV {metrics['mass_tv']}, minimum mass {min(metrics['supported_mass'])}.

Backend8/trainer5, requested128/actual64, typed even-fit partition, actual3K+2 family, reaction snapshot and FIFO expiry are consistent. Population masks and old-law atomic rejection, row reset ledger, bounded symmetric graph, copy/novel budget and gross certificates, scalar JSON metadata and CPU uint8 RNG placement pass. The final newborn rows have zero own optimizer/history/evidence/graph state. No models, forwards, draws, updates, new seeds or CUDA were used.

The paired average remains an empirical anti-blur FIFO lease. It is independent of table population stationarity, with no distribution equivalence, every-update support or quality guarantee. Intermediate unsaved reaction decisions are not reconstructed. Only completed toy evidence is sealed here; the full original Grid100 and any later phases require their own valid evidence and unchanged gates. Private processes/logs are closed before the authoritative final seal. The evolving lane journal is not pinned.
'''
with (HERE/'FINAL-REVIEW.md').open('x') as f:f.write(report)
for name,expected in checked.items():assert sha(name)==expected,name
print(json.dumps(dict(status='VALID',toy_quality_gate=full_gate['status'],receipt_sha256=sha(HERE/'FINAL-RECEIPT.json'),
    final_joint=stamp['coherent_rows'],required=stamp['required'],full_Grid100_required=True)))
