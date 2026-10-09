"""Certify matched saved evidence and publish media without training/sampling."""
from pathlib import Path
from copy import deepcopy
import importlib.util,json,sys
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,read_json,file_hash,stable_hash
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.tier1_media import render,_scored_outputs
from reports.forge.regenerate_technique_inventory import project_receipt
OUT=Path(__file__).parent;QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/mass_allocation/queue')
PROGRESS=Path('/tmp/bcap-physics-round5-20261009/mass_allocation/progress.json')

def full_native_moments(samples):
 from benchmarks.toy100.problems import evaluation_geometry
 x=np.asarray(samples,dtype=np.float64);centers,sigma=evaluation_geometry('grid100',dtype=torch.float64);centers=centers.numpy();ids=((x[:,None]-centers[None])**2).sum(2).argmin(1);res=(x-centers[ids])/sigma;count=np.bincount(ids,minlength=100);quality=np.linalg.norm(res,axis=1)<=3;qmass=np.bincount(ids[quality],minlength=100)/len(x);covs=[];means=[];ks=[]
 for i in range(100):
  local=res[ids==i]
  if len(local)<2:continue
  covs.append(np.cov(local,rowvar=False,bias=True));means.append(local.mean(0));radius=np.sort(np.linalg.norm(local,axis=1));cdf=1-np.exp(-radius**2/2);ks.append(max(np.max(np.arange(1,len(local)+1)/len(local)-cdf),np.max(cdf-np.arange(len(local))/len(local))))
 covs=np.stack(covs);eig=np.linalg.eigvalsh(covs)
 return dict(scope='Full uncensored nearest-cell moments; oracle geometry diagnostic only.',sample_count=len(x),occupied_cells=int((count>0).sum()),cells_with_covariance=len(covs),modes_meeting_quality_mass_0_005=int((qmass>=.005).sum()),minimum_genuine_quality_mass=float(qmass.min()),maximum_genuine_quality_mass=float(qmass.max()),full_covariance_frobenius_rms=float(np.sqrt(((covs-np.eye(2))**2).sum((1,2)).mean())),full_covariance_min_eigen_ratio=float(eig[:,0].min()),full_covariance_max_eigen_ratio=float(eig[:,1].max()),full_covariance_mean_trace_ratio=float(np.trace(covs,axis1=1,axis2=2).mean()/2),full_center_rms_sigma=float(np.sqrt(np.square(means).sum(1).mean())),full_radial_ks_mean=float(np.mean(ks)),full_radial_ks_maximum=float(np.max(ks)),spill_fraction=float(1-quality.mean()))

def main():
 torch.set_num_threads(1);state=read_json(QUEUE/'queue/state.json');requests=read_json(PROGRESS)['requests'];rows=[];receipts=[];media=[];proof=[];superseded=[]
 assert all(state['submissions'][r]['status'] not in {'queued','running','paused'} for r in requests.values())
 spec=importlib.util.spec_from_file_location('saved_media',ROOT/'reports/forge/gaussian-smoke-inventory/export_media.py');saved_media=importlib.util.module_from_spec(spec);spec.loader.exec_module(saved_media)
 for role,rid in requests.items():
  request=state['submissions'][rid]['request']
  for job in state['jobs'].values():
   if rid not in job['subscribers']:continue
   if not job.get('result'):
    rows.append(dict(role=role,task_id=job['definition']['task_id'],gate_status='BLOCKED',metrics={},reason=job.get('reason','own prerequisite did not pass')));continue
   for old in job['attempts'][:-1]:
    oldaid=old['attempt_id'];olddurable=ROOT/'reports/forge/attempts'/oldaid;oldresult=read_json(olddurable/'result.json');oldcert=read_json(olddurable/'evidence.json');assert oldcert['result_hash']==stable_hash(oldresult);receipts.append(project_receipt(ROOT,oldaid))
    superseded.append(dict(role=role,task_id=job['definition']['task_id'],attempt_id=oldaid,attempt_path=old['path'],grade=oldresult['task_results'][0]['gate_status'],raw_attempt_status=oldresult['raw']['attempt_status'],paid_seconds=next(c['seconds'] for c in state['charges'] if c['attempt_id']==oldaid),original_certificates={n:file_hash(olddurable/f'{n}.json') for n in ['request','result','evidence']},replacement_attempt_id=job['result']['attempt_id']))
   aid=job['result']['attempt_id'];durable=ROOT/'reports/forge/attempts'/aid;compact=project_receipt(ROOT,aid);receipts.append(compact)
   envelope,result,cert=[read_json(durable/f'{n}.json') for n in ('request','result','evidence')];assert cert['result_hash']==stable_hash(result);assert cert['source']==request['source'];local=Path(cert['local_artifact_root']);assert read_json(local/'result.json')==result
   for row in result['task_results']:
    task=request['tasks'][row['task_id']];e=row['evidence'];item=dict(role=role,task_id=row['task_id'],attempt_id=aid,gate_status=row['gate_status'],metrics=row.get('metrics',{}),evaluator_summary=next(x['evaluator_summary'] for x in compact['task_results'] if x['task_id']==row['task_id']),paid_seconds=next(x['seconds'] for x in state['charges'] if x['attempt_id']==aid),cost=row.get('cost',{}),sampling={k:e.get(k) for k in ('sampling_law','eval_output_noise','scoring_weights')})
    d=e.get('provenance_checkpoint');saved=None
    if d:
     path=Path(d['artifact_root'])/d['path'];verify_artifacts(Path(d['artifact_root']),d['artifact_manifest']);assert file_hash(path)==d['sha256'];saved=torch.load(path,map_location='cpu',weights_only=False)
    gif=OUT/'media'/f'{role}-{row["task_id"]}.gif';gif.parent.mkdir(parents=True,exist_ok=True)
    if task['adapter']=='native100':
     root=Path(e['artifact_root']);verify_artifacts(root,e['artifact_manifest']);path=root/task['id']/'holdout_samples.npz';item['holdout_full_moments']=full_native_moments(np.load(path)['live']);item['holdout_archive_sha256']=file_hash(path);summary=read_json(root/task['id']/'summary.json');item['holdout']=summary.get('holdout');item['accuracy']=summary.get('accuracy');artifact=saved_media.render_native(task,row,gif);inputs=artifact['source_inputs']
    else:
     samples,inputs=_scored_outputs(task,e,local);scalar=row['task_id'].startswith('gaussian')
     if samples:
      if scalar:from benchmarks.toy_audit.gaussian1d_quality import score_samples
      else:from benchmarks.transfer_suite.vector_tasks import score_samples
      law={**task['execution']['host_definition'],'thresholds':task['evaluation']['thresholds']}
      for record,observation in zip(samples,e['observations']):
       if scalar and row['task_id']=='gaussian1d_stability' and record['step']>4000:law['means']=[[3.]]
       assert score_samples(record['samples'],law,record['step'])=={k:v for k,v in observation.items() if k!='step'}
      item['metric_sets_reproduced']=len(samples)
     # Structured scorer metadata remains in the full receipt; display scalars.
     from unittest.mock import patch
     from benchmarks.toy_audit import api_run
     original=api_run.render_gif
     def display(case,records,*args,**kw):return original(case,[{**r,'metrics':{k:v for k,v in r['metrics'].items() if type(v) in (int,float)}} for r in records],*args,**kw)
     with patch.object(api_run,'render_gif',display):artifact=render(task,row,local,gif)
    media.append({**artifact,'role':role,'gif':str(gif.relative_to(OUT))})
    applied=row.get('applied',{})
    if not applied:
     raw=read_json(local/'raw-result.json');applied=raw.get('applied',raw)
    proof.append(dict(role=role,task_id=row['task_id'],attempt_id=aid,source_commit=request['source']['origin_commit'],source_digest=request['source']['digest'],runtime=request['runtime'],protocol=request['protocol'],recipe=row.get('recipe',applied.get('recipe')),initialization=row.get('initialization',applied.get('initialization')),data_sha256=e.get('data_sha256'),named_stream_state_sha256=(d or {}).get('named_stream_state_sha256'),named_stream_hashes={k:stable_hash(v.tolist()) for k,v in (saved or {}).get('streams',{}).get('states',{}).items()},original_certificates={n:file_hash(durable/f'{n}.json') for n in ('request','result','evidence')},retained_inputs=inputs,provenance_checkpoint=d))
    rows.append(item)
 checks=[]
 for tid in sorted({p['task_id'] for p in proof}):
  pair=[p for p in proof if p['task_id']==tid]
  if len(pair)!=2:continue
  for field in ('source_digest','runtime','protocol','initialization'):assert pair[0][field]==pair[1][field],(tid,field)
  recipes=[{k:v for k,v in p['recipe'].items() if k!='kinetic_transport_mode'} for p in pair];assert recipes[0]==recipes[1],tid
  streams=[{k:v for k,v in p['named_stream_hashes'].items() if json.loads(k)[0]!='eval'} for p in pair];assert streams[0]==streams[1],tid
  if pair[0]['data_sha256'] and pair[1]['data_sha256']:assert pair[0]['data_sha256']==pair[1]['data_sha256']
  checks.append(dict(task_id=tid,source_runtime_protocol_equal=True,initialization_equal=True,consumed_named_training_streams_equal=True,data_digest_equal=True if pair[0]['data_sha256'] else None,recipe_delta_only='kinetic_transport_mode',own_smoke_restoration=tid=='gaussian1d_stability'))
 campaign=state['campaigns']['mass-allocation-round5-v1'];assert campaign['reserved_seconds']==0
 request=state['submissions'][requests['candidate']]['request'];summary=dict(schema_version=1,qualification_input=False,scope='research_diagnostic',requests=requests,arms={r:dict(candidate_id=state['submissions'][rid]['request']['candidate']['id'],revision=state['submissions'][rid]['request']['candidate_revision']) for r,rid in requests.items()},source_commit=request['source']['origin_commit'],source_digest=request['source']['digest'],reservation_ceiling=19440,track_ceiling=21600,planned_full_reservations=19440,executed_full_reservations=sum(j['definition']['budget_seconds']*len(j['attempts']) for j in state['jobs'].values()),campaign_accounting=campaign,paid_seconds=campaign['spent_seconds'],scientific_attempts=sum(len(j['attempts']) for j in state['jobs'].values()),scientific_retries=sum(max(0,len(j['attempts'])-1) for j in state['jobs'].values()),outcomes={role:{status:sum(r['role']==role and r['gate_status']==status for r in rows) for status in ('PASS','FAIL','BLOCKED','INCOMPLETE','INVALID')} for role in requests},task_results=rows,superseded_execution_attempts=superseded,optimizer_updates_added_by_publication=0,sampling_draws_added_by_publication=0)
 atomic_json(OUT/'results.json',summary);atomic_json(OUT/'receipts.json',receipts);atomic_json(OUT/'provenance.json',dict(schema_version=1,qualification_input=False,proofs=proof,matched_checks=checks));atomic_json(OUT/'media/index.json',dict(schema_version=1,qualification_input=False,media=media,optimizer_updates_added=0,sampling_draws_added=0));print(json.dumps(dict(outcomes=summary['outcomes'],media=len(media),paid_seconds=summary['paid_seconds'])),flush=True)
if __name__=='__main__':main()
