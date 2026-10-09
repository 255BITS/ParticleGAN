"""Matched trained endpoint center populations; no new samples or updates."""
from pathlib import Path
import sys,time,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import read_json,atomic_json,file_hash
from experiments.forge.state import state_digest
from experiments.forge.tier1_media import _scored_outputs
from lib.toy_models import SimpleMLPGenerator
from benchmarks.toy100.problems import evaluation_geometry
OUT=Path(__file__).parent

def main():
 start=time.monotonic();torch.set_num_threads(1);rng=torch.get_rng_state().clone();rows=[]
 for p in read_json(OUT/'provenance.json')['proofs']:
  tid=p['task_id']
  if tid.startswith('gaussian'):continue
  d=p['provenance_checkpoint'];path=Path(d['artifact_root'])/d['path'];assert file_hash(path)==d['sha256'];saved=torch.load(path,map_location='cpu',weights_only=False);assert state_digest(saved)==d['state_sha256'];models=saved['trainer']['models'];gs=models['G'];layers=sum(k.endswith('.weight') for k in gs)-1
  with torch.device('meta'):g=SimpleMLPGenerator(gs['net.0.weight'].shape[1],gs['net.0.weight'].shape[0],layers,gs[f'net.{2*layers}.weight'].shape[0])
  g.to_empty(device='cpu');g.load_state_dict(gs);g.double().requires_grad_(False).eval();center=g(models['prior']['z'].double()).detach()
  if tid=='grid100':means,sd=evaluation_geometry(tid,dtype=torch.float64);cov=torch.eye(2,dtype=torch.float64).repeat(100,1,1)*sd**2;masses=torch.full((100,),.01,dtype=torch.float64)
  else:
   spec=read_json(ROOT/f'configs/forge/tasks/{tid}.json')['execution']['host_definition'];means=torch.tensor(spec['means'],dtype=torch.float64);cov=torch.tensor(spec['covariances'],dtype=torch.float64);masses=torch.tensor(spec['masses'],dtype=torch.float64)
  assignments=torch.cdist(center,means).argmin(1);count=torch.bincount(assignments,minlength=len(means));pop=[]
  for k in range(len(means)):
   x=center[assignments==k];delta=x-x.mean(0) if len(x) else x;c=delta.T@delta/max(1,len(x))
   pop.append(dict(component=k,locations=len(x),center_trace_over_target_trace=float(c.trace()/cov[k].trace()),center_covariance_relative_error=float((c-cov[k]).norm()/cov[k].norm()),center_mean_error_sigma=float(torch.sqrt((x.mean(0)-means[k])@torch.linalg.solve(cov[k],x.mean(0)-means[k]))) if len(x) else None))
  rows.append(dict(role=p['role'],task_id=tid,checkpoint_sha256=d['sha256'],checkpoint_state_sha256=d['state_sha256'],prior=saved['prior'],center_counts=count.tolist(),center_population_mass_tv=float((count/len(center)-masses).abs().sum()/2),center_population=pop,all_actual_metrics_retained_in='results.json',scope='Exact G(z_location) population, diagnostic nearest-center fixed assignment and uncensored between-location moments; no within-kernel linearization and no claim of history attribution.'))
 assert torch.equal(rng,torch.get_rng_state());atomic_json(OUT/'center-populations.json',dict(schema_version=1,qualification_input=False,scope='Matched trained endpoints only; target means/covariance/labels are diagnostics, never training supervision. Served samples and sustained full gates are authoritative.',rows=rows,optimizer_updates_added=0,sampling_draws_added=0,caller_rng_unchanged=True,analysis_seconds=time.monotonic()-start));print(json.dumps(dict(rows=len(rows),seconds=time.monotonic()-start)))
if __name__=='__main__':main()
