"""Read-only saved-center and conservative output-space census; no training."""
from pathlib import Path
import sys,json,time,math
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,read_json,file_hash
from experiments.forge.state import state_digest
from experiments.forge.rng import NamedStreams
from experiments.forge.tier1_media import _scored_outputs
from lib.toy_models import SimpleMLPGenerator
from benchmarks.toy100.problems import evaluation_geometry,sample_real
from benchmarks.transfer_suite.vector_tasks import sample_target
OUT=Path(__file__).parent
ARCHIVE=Path('/mnt/ml7tb/ParticleGAN-forge')
CASES=[('native_overshoot','native-overshoot-round4-control-v1','grid100'),
 ('projection_transport','projection_transport-round4-transport-v1','vector_unequal_mass'),
 ('projection_transport','projection_transport-round4-transport-v1','vector_two_broad'),
 ('kinetic_transport','kinetic_transport_local_v2','vector_unequal_width')]

def main():
 started=time.monotonic();torch.set_num_threads(1);rng=torch.get_rng_state().clone();rows=[]
 for track,candidate,tid in CASES:
  q=(ARCHIVE/f'bcap-physics-round4-20261009/{track}/queue' if track!='kinetic_transport' else ARCHIVE/'bcap-physics-round2-20261009/kinetic_transport/queue')
  state=read_json(q/'queue/state.json')
  job=next(j for j in state['jobs'].values() if j.get('result') and j['definition']['task_id']==tid and any(state['submissions'][r]['request']['candidate']['id']==candidate for r in j['subscribers']))
  rid=next(r for r in job['subscribers'] if state['submissions'][r]['request']['candidate']['id']==candidate);req=state['submissions'][rid]['request'];task=req['tasks'][tid];row=job['result']['task_results'][0];e=row['evidence'];desc=e['provenance_checkpoint'];p=Path(desc['artifact_root'])/desc['path']
  assert file_hash(p)==desc['sha256'];saved=torch.load(p,map_location='cpu',weights_only=False);assert state_digest(saved)==desc['state_sha256']
  for name in ['lib/toy_models.py','experiments/forge/rng.py','benchmarks/toy100/problems.py','benchmarks/transfer_suite/vector_tasks.py']:
   assert file_hash(ROOT/name)==req['source']['files'][name],name
  ms=saved['trainer']['models'];gs=ms['G'];layers=sum(k.endswith('.weight') for k in gs)-1
  with torch.device('meta'):g=SimpleMLPGenerator(gs['net.0.weight'].shape[1],gs['net.0.weight'].shape[0],layers,gs[f'net.{2*layers}.weight'].shape[0])
  g.to_empty(device='cpu');g.load_state_dict(gs);g.double().requires_grad_(False).eval();z=ms['prior']['z'].double();centers=g(z).detach()
  if tid=='grid100':means,sd=evaluation_geometry(tid,dtype=torch.float64);cov=torch.eye(2,dtype=torch.float64).repeat(len(means),1,1)*sd**2;masses=torch.ones(len(means),dtype=torch.float64)/len(means)
  else:
   spec=task['execution']['host_definition'];means=torch.tensor(spec['means'],dtype=torch.float64);cov=torch.tensor(spec['covariances'],dtype=torch.float64);masses=torch.tensor(spec['masses'],dtype=torch.float64)
  def assignment(x):return torch.cdist(x,means).argmin(1)
  def census(x):
   a=assignment(x);counts=torch.bincount(a,minlength=len(means));radius=torch.sqrt(torch.einsum('ni,nij,nj->n',x-means[a],torch.linalg.inv(cov)[a],x-means[a]));hq=torch.bincount(a[radius<=3],minlength=len(means))/len(x)
   cc=[]
   for k in range(len(means)):
    vals=x[a==k];delta=vals-vals.mean(0) if len(vals) else vals
    cc.append(float((delta.T@delta/max(1,len(vals))).trace()/cov[k].trace()))
   return dict(counts=counts.tolist(),mass_tv=float((counts/len(x)-masses).abs().sum()/2),genuine_3sigma_mass=hq.tolist(),modes_with_genuine_mass_ge_half_target=int((hq>=masses/2).sum()),precision_3sigma=float((radius<=3).double().mean()),center_covariance_trace_over_target=cc)
  # Retained served observations, not fresh public samples.
  if tid=='grid100':
   path=Path(e['artifact_root'])/tid/'final_samples.npz';assert file_hash(path)==e['artifact_manifest']['files'][f'{tid}/final_samples.npz']['sha256']
   x=torch.from_numpy(np.load(path)['live'])
  else:
   local=Path(desc['artifact_root']).parent;records,_=_scored_outputs(task,e,local);x=records[-1]['samples']
  x=x.double().flatten(1)
  # Exact frozen target replay with ephemeral stream. Never alter trained state.
  device=saved['trainer']['device'] if tid=='grid100' else 'cpu';stream=NamedStreams(0,device=device).generator('data',component='target',purpose='training');n=task['execution']['resources']['batch_size'] if tid=='grid100' else spec['batch']
  for step in range(task['execution']['steps']):y=sample_real(tid,n,device=device,generator=stream) if tid=='grid100' else sample_target(spec,n,stream,step)
  key=json.dumps(['data','target','training',device],separators=(',',':'));assert torch.equal(stream.get_state().cpu(),saved['streams']['states'][key])
  y=y.cpu().double()[:128];xx=x[:128];cost=torch.cdist(xx,y).square();_,columns=linear_sum_assignment(cost.numpy());matched=y[columns];a,b=assignment(xx),assignment(matched)
  flux=torch.bincount(a*len(means)+b,minlength=len(means)**2).reshape(len(means),len(means));cross=float((a!=b).double().mean())
  projected=[]
  for angle in torch.arange(32,dtype=torch.float64)*math.pi/32:
   v=torch.stack([angle.cos(),angle.sin()]);ix=(xx@v).argsort();iy=(y@v).argsort();projected.append(float((a[ix]!=assignment(y)[iy]).double().mean()))
  rows.append(dict(candidate_id=candidate,task_id=tid,source_commit=req['source']['origin_commit'],source_digest=req['source']['digest'],attempt_id=job['result']['attempt_id'],checkpoint=dict(path=str(p),sha256=desc['sha256'],state_sha256=desc['state_sha256']),center_population=census(centers),served_population=census(x),authoritative_metrics=row['metrics'],data_replay_matches_consumed_stream=True,output_probe=dict(rows=128,balanced_assignment_cross_component_fraction=cross,sliced_mean_cross_component_fraction=sum(projected)/32,conservative_assignment_flux=flux.tolist() if len(means)<5 else None,target_label_counts=torch.bincount(b,minlength=len(means)).tolist(),scope='128 saved served rows matched to replayed final target batch prefix; not original latent rows or a G/prior proposal')))
  assert state_digest(saved)==desc['state_sha256']
 probes=Path('/home/martyn/dev/ParticleGAN-bcap-r4-role_motion/reports/forge/bcap-physics/role_motion/round4/saved-role-probes.json');prior=read_json(probes)
 receipt=dict(schema_version=1,qualification_input=False,reservation_seconds=120,analysis_seconds=time.monotonic()-started,optimizer_updates_added=0,public_sampling_draws_added=0,trained_state_mutations=0,rows=rows,finite_role_motion_prior=dict(path=str(probes),sha256=file_hash(probes),scope=prior['scope'],source_bound_archived_probe_only=True),limitations='Endpoint center assignments and diagnostic three-sigma mass only; these do not replace authoritative four-sigma/full gates or attribute training history. Conservative flux is an output-space coupling of saved samples, not measured training flux. Target means/covariances/labels are diagnostic only.')
 assert torch.equal(rng,torch.get_rng_state());receipt['caller_rng_unchanged']=True;atomic_json(OUT/'prior-diagnostics.json',receipt)
 print(json.dumps(dict(seconds=receipt['analysis_seconds'],rows=[dict(task=r['task_id'],centerTV=r['center_population']['mass_tv'],servedTV=r['served_population']['mass_tv'],qualityModes=r['served_population']['modes_with_genuine_mass_ge_half_target'],slicedCross=r['output_probe']['sliced_mean_cross_component_fraction'],balancedCross=r['output_probe']['balanced_assignment_cross_component_fraction']) for r in rows])))
if __name__=='__main__':main()
