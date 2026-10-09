"""Publish source-certified numerical results and actual saved-state GIFs only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,file_hash,read_json,stable_hash
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.tier1_media import render,_scored_outputs
from reports.forge.regenerate_technique_inventory import project_receipt

OUT=Path(__file__).resolve().parent
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/force_distortion/queue')
BRIEF=Path('/tmp/bcap-physics-round5-20261009/force_distortion')

def metadata(value,key):
    if isinstance(value,dict):return ([value[key]['stats']] if key in value else [])+[x for v in value.values() for x in metadata(v,key)]
    if isinstance(value,(tuple,list)):return [x for v in value for x in metadata(v,key)]
    return []

def scalar_render(task,row,local,gif):
    from unittest.mock import patch
    from benchmarks.toy_audit import api_run
    original=api_run.render_gif
    def display(case,records,*args,**kwargs):
        records=[{**r,'metrics':{k:v for k,v in r['metrics'].items() if type(v) in (int,float)}} for r in records]
        return original(case,records,*args,**kwargs)
    with patch.object(api_run,'render_gif',display):return render(task,row,local,gif)

def full_moments(points):
    from benchmarks.toy100.problems import evaluation_geometry
    centers,sigma=evaluation_geometry('grid100',dtype=torch.float64);centers=centers.numpy()
    points=np.asarray(points,dtype=np.float64)
    labels=np.square(points[:,None]-centers[None]).sum(2).argmin(1)
    residual=(points-centers[labels])/sigma;quality=np.linalg.norm(residual,axis=1)<=3
    count=np.bincount(labels,minlength=100);quality_mass=np.bincount(labels[quality],minlength=100)/len(points)
    covariance=np.stack([np.cov(residual[labels==k],rowvar=False,bias=True) if count[k]>=2 else np.zeros((2,2)) for k in range(100)])
    return dict(scope='Uncensored nearest-cell output moments; target geometry is diagnostic only.',
        samples=len(points),occupied_cells=int((count>0).sum()),missing_cells=int((count==0).sum()),
        genuine_modes_at_0_005=int((quality_mass>=.005).sum()),minimum_genuine_mass=float(quality_mass.min()),
        spill_fraction=float(1-quality.mean()),
        full_covariance_frobenius_rms=float(np.sqrt(np.square(covariance-np.eye(2)).sum((1,2)).mean())))

def main():
    torch.set_num_threads(1);state=read_json(QUEUE/'queue/state.json');requests=read_json(BRIEF/'progress.json')['requests']
    assert all(state['submissions'][rid]['status'] not in {'queued','running','paused'} for rid in requests.values())
    spec=importlib.util.spec_from_file_location('native_saved_media',ROOT/'reports/forge/gaussian-smoke-inventory/export_media.py')
    native_media=importlib.util.module_from_spec(spec);spec.loader.exec_module(native_media)
    rows=[];proofs=[];receipts=[];media=[];history=[]
    for role,rid in requests.items():
        request=state['submissions'][rid]['request']
        for job in state['jobs'].values():
            if rid not in job['subscribers']:continue
            for previous in job['attempts'][:-1]:
                aid=previous['attempt_id'];durable=ROOT/'reports/forge/attempts'/aid
                result=read_json(durable/'result.json');cert=read_json(durable/'evidence.json')
                assert cert['result_hash']==stable_hash(result)
                original=result['task_results'][0]
                history.append(dict(role=role,task_id=job['definition']['task_id'],attempt_id=aid,
                    source_digest=request['source']['digest'],gate_status=original['gate_status'],
                    raw_status=original.get('raw_status'),reason=original.get('reason'),cost=original.get('cost'),
                    paid_seconds=result['raw'].get('elapsed_seconds'),artifact_root=cert['local_artifact_root'],
                    original_certificates={n:dict(path=str((durable/f'{n}.json').relative_to(ROOT)),sha256=file_hash(durable/f'{n}.json')) for n in ('request','result','evidence')}))
            if not job.get('result'):
                rows.append(dict(role=role,task_id=job['definition']['task_id'],gate_status='BLOCKED',metrics={},
                    reason=job.get('reason') or 'own passing prerequisite is unavailable'))
                continue
            aid=job['result']['attempt_id'];durable=ROOT/'reports/forge/attempts'/aid
            envelope,result,cert=[read_json(durable/f'{n}.json') for n in ('request','result','evidence')]
            assert cert['result_hash']==stable_hash(result) and cert['source']==request['source']
            local=Path(cert['local_artifact_root']);assert read_json(local/'result.json')==result
            compact=project_receipt(ROOT,aid);receipts.append(compact)
            for row in result['task_results']:
                task=request['tasks'][row['task_id']];evidence=row.get('evidence',{});saved=None
                item=dict(role=role,task_id=row['task_id'],attempt_id=aid,gate_status=row['gate_status'],metrics=row.get('metrics',{}),
                    reason=row.get('reason'),evaluator_summary=next(r['evaluator_summary'] for r in compact['task_results'] if r['task_id']==row['task_id']),
                    paid_seconds=result.get('raw',{}).get('elapsed_seconds',result.get('raw',{}).get('seconds')),cost=row.get('cost',{}))
                descriptor=evidence.get('provenance_checkpoint') or evidence.get('checkpoint')
                if descriptor:
                    root=Path(descriptor.get('artifact_root',evidence.get('artifact_root',str(local))))
                    manifest=descriptor.get('artifact_manifest') or evidence.get('artifact_manifest')
                    if manifest:verify_artifacts(root,manifest)
                    checkpoint=root/descriptor['path'];assert file_hash(checkpoint)==descriptor['sha256']
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    for key in ('constraint_geometry','direction_blend','sample_force'):item[key]=metadata(saved,key)
                gif=OUT/'media'/f'{role}-{row["task_id"]}.gif';gif.parent.mkdir(parents=True,exist_ok=True)
                inputs={}
                if row['gate_status'] in {'PASS','FAIL'}:
                    if task['adapter']=='native100':
                        root=Path(evidence['artifact_root']);verify_artifacts(root,evidence['artifact_manifest'])
                        holdout_path=root/task['id']/'holdout_samples.npz'
                        item['holdout_full_moments']=full_moments(np.load(holdout_path)['live'])
                        summary=read_json(root/task['id']/'summary.json');item['holdout']=summary.get('holdout');item['accuracy']=summary.get('accuracy')
                        artifact=native_media.render_native(task,row,gif);inputs=artifact['source_inputs']
                    else:
                        samples,inputs=_scored_outputs(task,evidence,local);artifact=scalar_render(task,row,local,gif)
                    media.append(dict(**artifact,role=role,gif=str(gif.relative_to(OUT))))
                applied=row.get('applied',{})
                if not applied and (local/'raw-result.json').exists():
                    raw=read_json(local/'raw-result.json');applied=raw.get('applied',raw)
                proofs.append(dict(role=role,task_id=row['task_id'],attempt_id=aid,source_commit=request['source']['origin_commit'],
                    source_digest=request['source']['digest'],runtime=request['runtime'],protocol=request['protocol'],
                    recipe=applied.get('recipe',row.get('recipe')),initialization=applied.get('initialization',row.get('initialization')),
                    data_sha256=evidence.get('data_sha256'),
                    named_stream_hashes={k:stable_hash(v.tolist()) for k,v in (saved or {}).get('streams',{}).get('states',{}).items()},
                    original_certificates={n:dict(path=str((durable/f'{n}.json').relative_to(ROOT)),sha256=file_hash(durable/f'{n}.json')) for n in ('request','result','evidence')},
                    retained_inputs=inputs,provenance_checkpoint=descriptor,artifact_root=str(local)))
                rows.append(item)
    comparisons=[]
    for taskid in sorted({p['task_id'] for p in proofs}):
        pair=[p for p in proofs if p['task_id']==taskid]
        if len(pair)!=2:continue
        for key in ('source_digest','runtime','protocol'):assert pair[0][key]==pair[1][key]
        if all(p['initialization'] for p in pair):assert pair[0]['initialization']==pair[1]['initialization'],taskid
        if all(p['data_sha256'] for p in pair):assert pair[0]['data_sha256']==pair[1]['data_sha256'],taskid
        recipes=[deepcopy(p['recipe']) for p in pair]
        if all(recipes):
            for recipe in recipes:recipe.pop('constraint_geometry_mode',None)
            assert recipes[0]==recipes[1],taskid
        streams=[{k:v for k,v in p['named_stream_hashes'].items() if json.loads(k)[0]!='eval'} for p in pair]
        if all(streams):assert streams[0]==streams[1],taskid
        comparisons.append(dict(task_id=taskid,source_runtime_protocol_equal=True,
            initialization_equal=True if all(p['initialization'] for p in pair) else None,
            actual_batches_equal=True if all(p['data_sha256'] for p in pair) else None,
            named_training_streams_equal=True if all(streams) else None,recipe_delta_only='constraint_geometry_mode' if all(recipes) else None))
    request=state['submissions'][requests['candidate']]['request']
    summary=dict(schema_version=1,qualification_input=False,scope='research_diagnostic',requests=requests,
        arm_ids={role:state['submissions'][rid]['request']['candidate']['id'] for role,rid in requests.items()},
        source_digest=request['source']['digest'],source_commit=request['source']['origin_commit'],
        arm_revisions={role:state['submissions'][rid]['request']['candidate_revision'] for role,rid in requests.items()},
        initial_main_reservation_ceiling=19440,total_track_ceiling=21600,
        initial_ancillary_allowance=2160,ancillary_allowance_after_retries=1920,
        campaign_accounting=state['campaigns']['force-distortion-round5-v1'],
        execution_retry_history=history,
        full_main_reservations_including_retries=19440+sum(state['submissions'][requests[h['role']]]['request']['tasks'][h['task_id']]['resources']['timeout_seconds'] for h in history),
        outcomes={role:{status:sum(r['role']==role and r['gate_status']==status for r in rows) for status in ('PASS','FAIL','BLOCKED','INCOMPLETE','INVALID')} for role in requests},
        task_results=rows,optimizer_updates_added_by_publication=0,sampling_draws_added_by_publication=0)
    atomic_json(OUT/'results.json',summary);atomic_json(OUT/'receipts.json',receipts)
    atomic_json(OUT/'provenance.json',dict(schema_version=1,qualification_input=False,proofs=proofs,matched_checks=comparisons))
    atomic_json(OUT/'media/index.json',dict(schema_version=1,qualification_input=False,media=media,optimizer_updates_added=0,sampling_draws_added=0))
    print(json.dumps(dict(outcomes=summary['outcomes'],media=len(media),paid_seconds=summary['campaign_accounting']['spent_seconds'])),flush=True)

if __name__=='__main__':main()
