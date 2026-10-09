"""Verify frozen scientific source, initialization, prior, data and own states."""
from pathlib import Path
import sys,json,hashlib,time
import torch
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,read_json,file_hash
from experiments.forge.state import state_digest
from experiments.forge.rng import NamedStreams
from benchmarks.toy100.problems import sample_real
from benchmarks.transfer_suite.vector_tasks import sample_target
OUT=Path(__file__).parent;QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/mass_allocation/queue')

def main():
 began=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 state=read_json(QUEUE/'queue/state.json');proof=read_json(OUT/'provenance.json')['proofs'];pinned=None;source_checks=0
 for r in state['submissions'].values():
  req=r['request'];source=req['source'];snapshot=QUEUE/'snapshots'/source['digest']
  if pinned is None:pinned=source
  else:assert source==pinned
  for name,digest in source['files'].items():assert file_hash(snapshot/name)==digest,(name,'snapshot');source_checks+=1
 # Implementation and declarations must retain their measured bytes. Reporting-only
 # additions are not attributed to the frozen scientific source.
 for name,digest in pinned['files'].items():
  if name.startswith(('particlegan/','experiments/forge/','benchmarks/','lib/','configs/forge/ideas/mass-allocation','configs/forge/studies/mass-allocation','configs/forge/views/mass-allocation','tests/test_balanced_assignment')):assert file_hash(ROOT/name)==digest,(name,'working-science')
 for name,digest in read_json(OUT/'preservation.json')['files'].items():assert file_hash(ROOT/name)==digest,(name,'original-publication/task')
 saved={};checks=[]
 for p in proof:
  d=p['provenance_checkpoint'];path=Path(d['artifact_root'])/d['path'];assert file_hash(path)==d['sha256'];s=torch.load(path,map_location='cpu',weights_only=False);assert state_digest(s)==d['state_sha256'];saved[(p['role'],p['task_id'])]=s
 for tid in ['vector_unequal_mass','vector_two_broad','vector_unequal_width','grid100']:
  pair=[p for p in proof if p['task_id']==tid];assert len(pair)==2;req=state['submissions'][next(iter(read_json('/tmp/bcap-physics-round5-20261009/mass_allocation/progress.json')['requests'].values()))]['request'];task=req['tasks'][tid];samples=[saved[(p['role'],tid)] for p in pair];device=samples[0]['trainer']['device'] if tid=='grid100' else 'cpu';stream=NamedStreams(0,device=device).generator('data',component='target',purpose='training');digest=hashlib.sha256();n=task['execution']['resources']['batch_size'] if tid=='grid100' else task['execution']['host_definition']['batch']
  for step in range(task['execution']['steps']):
   batch=sample_real(tid,n,device=device,generator=stream) if tid=='grid100' else sample_target(task['execution']['host_definition'],n,stream,step);digest.update(batch.detach().cpu().numpy().tobytes())
  key=json.dumps(['data','target','training',device],separators=(',',':'));assert all(torch.equal(stream.get_state().cpu(),s['streams']['states'][key]) for s in samples)
  initial=[s['initialization'] for s in samples];assert initial[0]==initial[1]
  checks.append(dict(task_id=tid,reconstructed_actual_batch_sequence_sha256=digest.hexdigest(),steps=task['execution']['steps'],batch_size=n,both_consumed_target_streams_match=True,matched_public_initializer=True,prior=[s['prior'] for s in samples]))
 # Every Gaussian continuation uses its own exact1000-update prefix. Retained
 # restored-state checkpoints certify this more strongly than matching settings.
 continuations=[]
 for role in ['control','candidate']:
  rid=read_json('/tmp/bcap-physics-round5-20261009/mass_allocation/progress.json')['requests'][role]
  smoke=next(j for j in state['jobs'].values() if rid in j['subscribers'] and j['definition']['task_id']=='gaussian1d_smoke')
  hold=next(j for j in state['jobs'].values() if rid in j['subscribers'] and j['definition']['task_id']=='gaussian1d_stability')
  if not hold.get('result'):
   assert smoke['result']['task_results'][0]['gate_status']!='PASS';continuations.append(dict(role=role,status='BLOCKED',own_failed_smoke=True));continue
  e=hold['result']['task_results'][0]['evidence'];continuity=e['continuity'];continuations.append(dict(role=role,status='executed',continuity=continuity))
  assert continuity['prefix_steps']==1000 and continuity['restored_exactly'] is True and continuity['history_reset'] is False
  assert continuity['frozen_completed_steps']==4000 and continuity['matched_frozen_draws'] is True
  parent=smoke['result']['task_results'][0]['evidence'];assert smoke['result']['task_results'][0]['gate_status']=='PASS'
  assert continuity['parent_checkpoint_sha256']==parent['checkpoint']['sha256'] and continuity['parent_state_sha256']==parent['checkpoint']['state_sha256']
  assert continuity['parent_compatibility_key']==next(k for k,j in state['jobs'].items() if j is smoke)
  assert continuity['parent_candidate_revision']==smoke['result']['candidate_revision']
 r=dict(schema_version=1,qualification_input=False,source_commit=pinned['origin_commit'],source_digest=pinned['digest'],frozen_file_checks=source_checks,scientific_files_unchanged=True,original_task_and_publication_hashes_unchanged=True,data_replays=checks,own_gaussian_continuations=continuations,optimizer_updates_added=0,public_sampling_draws_added=0,trained_state_mutations=0,analysis_seconds=time.monotonic()-began,scope='Read-only frozen-source/receipt/state and deterministic target replay; original per-update target tensors were not retained. Named streams isolate evaluation from all training draws.')
 atomic_json(OUT/'validation.json',r);print(json.dumps(dict(source_checks=source_checks,data_replays=len(checks),seconds=r['analysis_seconds'])))
if __name__=='__main__':main()
