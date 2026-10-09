"""Certify and summarize all frozen arms from saved artifacts; no training."""
from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,file_hash,read_json,stable_hash
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.tier1_media import render,_scored_outputs
from reports.forge.regenerate_technique_inventory import project_receipt
from benchmarks.toy100.problems import evaluation_geometry

OUT=Path(__file__).resolve().parent
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/native_overshoot/queue')
STATUS=Path('/tmp/bcap-physics-round4-20261009/native_overshoot/progress.json')


def full_native_moments(samples):
    points=np.asarray(samples,dtype=np.float64)
    centers,sigma=evaluation_geometry('grid100',dtype=torch.float64)
    centers=centers.numpy()
    distance=((points[:,None]-centers[None])**2).sum(2)
    ids=distance.argmin(1)
    count=np.bincount(ids,minlength=100)
    residual=(points-centers[ids])/sigma
    quality=np.linalg.norm(residual,axis=1)<=3
    quality_mass=np.bincount(ids[quality],minlength=100)/len(points)
    means=[];covariances=[];eigenvalues=[];radial_ks=[]
    for i in range(100):
        local=residual[ids==i]
        if len(local)<2:continue
        means.append(local.mean(0))
        cov=np.cov(local,rowvar=False,bias=True);covariances.append(cov)
        eigenvalues.append(np.linalg.eigvalsh(cov))
        radius=np.sort(np.linalg.norm(local,axis=1))
        cdf=1-np.exp(-radius**2/2)
        radial_ks.append(max(np.max(np.arange(1,len(local)+1)/len(local)-cdf),
                             np.max(cdf-np.arange(len(local))/len(local))))
    covariance=np.stack(covariances);eig=np.stack(eigenvalues)
    return dict(scope='Full nearest-cell conditional residuals including spill; evaluation labels only, not latent/component identities or training oracles.',
        samples=len(points),occupied_cells=int((count>0).sum()),cells_with_covariance=len(covariances),
        minimum_genuine_quality_mass=float(quality_mass.min()),maximum_genuine_quality_mass=float(quality_mass.max()),
        modes_meeting_quality_mass_0_005=int((quality_mass>=.005).sum()),
        full_covariance_minimum_eigen_ratio=float(eig[:,0].min()),full_covariance_maximum_eigen_ratio=float(eig[:,1].max()),
        full_covariance_frobenius_rms=float(np.sqrt(((covariance-np.eye(2))**2).sum((1,2)).mean())),
        full_covariance_mean_trace_ratio=float(np.trace(covariance,axis1=1,axis2=2).mean()/2),
        full_center_rms_sigma=float(np.sqrt(np.square(means).sum(1).mean())),
        full_radial_ks_mean=float(np.mean(radial_ks)),full_radial_ks_maximum=float(np.max(radial_ks)),
        spill_fraction=float(1-quality.mean()))


def main():
    torch.set_num_threads(1)
    state=read_json(QUEUE/'queue/state.json')
    requests=read_json(STATUS)['requests']
    assert all(state['submissions'][rid]['status'] not in {'queued','running','paused'} for rid in requests.values())
    spec=importlib.util.spec_from_file_location('saved_media',ROOT/'reports/forge/gaussian-smoke-inventory/export_media.py')
    saved_media=importlib.util.module_from_spec(spec);spec.loader.exec_module(saved_media)
    rows=[];receipts=[];proof=[];media=[];trace_audit=[]
    for role,rid in requests.items():
        request=state['submissions'][rid]['request']
        for job in state['jobs'].values():
            if rid not in job['subscribers']:continue
            if not job.get('result'):
                taskid=job['definition']['task_id']
                blockers=request['tasks'][taskid].get('preflight_blockers',[])
                rows.append(dict(role=role,task_id=taskid,gate_status='BLOCKED',metrics={},reason=job.get('reason') or '; '.join(blockers) or 'own prerequisite did not pass'))
                continue
            aid=job['result']['attempt_id'];durable=ROOT/'reports/forge/attempts'/aid
            envelope,result,cert=[read_json(durable/f'{n}.json') for n in ('request','result','evidence')]
            assert cert['result_hash']==stable_hash(result) and cert['source']==request['source']
            local=Path(cert['local_artifact_root'])
            assert read_json(local/'result.json')==result
            compact=project_receipt(ROOT,aid);receipts.append(compact)
            for row in result['task_results']:
                task=request['tasks'][row['task_id']];evidence=row['evidence']
                item=dict(role=role,task_id=row['task_id'],attempt_id=aid,gate_status=row['gate_status'],metrics=row.get('metrics',{}),
                    evaluator_summary=next(r['evaluator_summary'] for r in compact['task_results'] if r['task_id']==row['task_id']),
                    cost=row.get('cost',{}),paid_seconds=result.get('raw',{}).get('seconds'),
                    sampling={k:evidence.get(k) for k in ('sampling_law','eval_output_noise','scoring_weights')})
                descriptor=evidence.get('provenance_checkpoint')
                saved=None
                if descriptor:
                    root=Path(descriptor['artifact_root']);verify_artifacts(root,descriptor['artifact_manifest'])
                    assert file_hash(root/descriptor['path'])==descriptor['sha256']
                    saved=torch.load(root/descriptor['path'],map_location='cpu',weights_only=True)
                    item['finite_step']=saved.get('trainer',{}).get('finite_step')
                if item['finite_step']:
                    # Independent arithmetic audit of every recorded proposal.
                    logfile=local/'run.log'
                    events=[]
                    for line in logfile.read_text().splitlines():
                        try:entry=json.loads(line)
                        except json.JSONDecodeError:continue
                        if entry.get('event')=='finite_step':events.append(entry)
                    violations=[e['step'] for e in events if e['accepted'] and not
                        (e['checked_loss']<=e['before_loss']+.1*e['scale']*e['slope'] and
                         (e['checked_loss']<e['before_loss'] or e['slope']==0))]
                    assert not violations
                    stats=item['finite_step']['stats']
                    prefix_stats={key:0 for key in ('proposals','accepted')}
                    if task['evaluation']['kind']=='gaussian_stability':
                        prefix=torch.load(Path(evidence['artifact_root'])/'initial-state.pt',map_location='cpu',weights_only=True)
                        prefix_stats=prefix['trainer']['finite_step']['stats']
                    assert len(events)==stats['proposals']-prefix_stats['proposals'],(len(events),stats)
                    assert sum(e['accepted'] for e in events)==stats['accepted']-prefix_stats['accepted']
                    item['finite_trace_audit']=dict(proposals=len(events),violations=violations,raw_log=str(logfile),raw_log_sha256=file_hash(logfile))
                    item['finite_trace_audit']['prefix_proposals_restored']=prefix_stats['proposals']
                    scales=np.array([e['scale'] for e in events])
                    item['finite_trace_audit']['scales']=dict(mean=float(scales.mean()),median=float(np.median(scales)),minimum=float(scales.min()),
                        full_scale=int((scales==1).sum()),zero_scale=int((scales==0).sum()))
                    item['finite_trace_audit']['downhill_shrunk']=sum(e['slope']<0 and e['accepted'] and e['scale']<1 for e in events)
                    item['finite_trace_audit']['full_step_loss_decreases_but_fails_armijo']=sum(e['slope']<0 and e['full_loss']<e['before_loss']
                         and e['full_loss']>e['before_loss']+.1*e['slope'] for e in events)
                    trace_audit.append(dict(role=role,task_id=row['task_id'],**item['finite_trace_audit']))
                gif=OUT/'media'/f'{role}-{row["task_id"]}.gif';gif.parent.mkdir(parents=True,exist_ok=True)
                if task['adapter']=='native100':
                    artifact_root=Path(evidence['artifact_root']);verify_artifacts(artifact_root,evidence['artifact_manifest'])
                    holdout_path=artifact_root/task['id']/'holdout_samples.npz'
                    holdout=np.load(holdout_path)
                    item['holdout_full_moments']=full_native_moments(holdout['live'])
                    item['holdout_archive_sha256']=file_hash(holdout_path)
                    summary=read_json(artifact_root/task['id']/'summary.json')
                    item['holdout']=summary.get('holdout')
                    item['accuracy']=summary.get('accuracy')
                    artifact=saved_media.render_native(task,row,gif)
                    inputs=artifact['source_inputs']
                else:
                    samples,inputs=_scored_outputs(task,evidence,local)
                    artifact=render(task,row,local,gif)
                media.append(dict(**artifact,role=role,gif=str(gif.relative_to(OUT))))
                applied=row.get('applied',{})
                if not applied:
                    raw=read_json(local/'raw-result.json');applied=raw.get('applied',raw)
                proof.append(dict(role=role,task_id=row['task_id'],attempt_id=aid,
                    source_commit=request['source']['origin_commit'],source_digest=request['source']['digest'],runtime=request['runtime'],
                    protocol=request['protocol'],recipe=applied.get('recipe',row.get('recipe')),
                    initialization=applied.get('initialization',row.get('initialization')),
                    data_sha256=evidence.get('data_sha256'),
                    named_stream_state_sha256=(descriptor or {}).get('named_stream_state_sha256'),
                    named_stream_hashes={k:stable_hash(v.tolist()) for k,v in (saved or {}).get('streams',{}).get('states',{}).items()},
                    original_certificates={n:file_hash(durable/f'{n}.json') for n in ('request','result','evidence')},
                    retained_inputs=inputs,provenance_checkpoint=descriptor))
                rows.append(item)
    comparisons=[]
    for taskid in sorted({r['task_id'] for r in proof}):
        pair=[r for r in proof if r['task_id']==taskid]
        if len(pair)!=2:continue
        for field in ('source_digest','runtime','protocol'):assert pair[0][field]==pair[1][field]
        if pair[0]['initialization'] is not None and pair[1]['initialization'] is not None:
            assert pair[0]['initialization']==pair[1]['initialization'],taskid
        if pair[0]['data_sha256'] and pair[1]['data_sha256']:
            assert pair[0]['data_sha256']==pair[1]['data_sha256'],taskid
        recipes=[deepcopy(p['recipe']) for p in pair]
        if all(recipes):
            for recipe in recipes:recipe.pop('finite_step_mode',None)
            assert recipes[0]==recipes[1],taskid
        streams=[{k:v for k,v in p['named_stream_hashes'].items() if json.loads(k)[0]!='eval'} for p in pair]
        if all(streams):assert streams[0]==streams[1],taskid
        comparisons.append(dict(task_id=taskid,source_runtime_protocol_equal=True,
            initialization_equal=None if pair[0]['initialization'] is None else pair[0]['initialization']==pair[1]['initialization'],
            batch_sequence_equal=None if not all(p['data_sha256'] for p in pair) else pair[0]['data_sha256']==pair[1]['data_sha256'],
            named_training_streams_equal=True if all(streams) else None,
            recipe_delta_only='finite_step_mode' if all(recipes) else None,
            continuation_note='Own smoke parent state, same declared prefix and schedule; model weights differ by trained arm.' if taskid=='gaussian1d_stability' else None))
    request=state['submissions'][requests['candidate']]['request']
    summary=dict(schema_version=1,qualification_input=False,scope='research_diagnostic',requests=requests,
        candidate=request['candidate']['id'],control=state['submissions'][requests['control']]['request']['candidate']['id'],
        source_digest=request['source']['digest'],source_commit=request['source']['origin_commit'],
        candidate_revision=request['candidate_revision'],control_revision=state['submissions'][requests['control']]['request']['candidate_revision'],
        reservation_ceiling=12840,campaign_accounting=state['campaigns']['native-overshoot-round4-v1'],
        outcomes={role:{status:sum(r['role']==role and r['gate_status']==status for r in rows) for status in ('PASS','FAIL','BLOCKED','INCOMPLETE','INVALID')} for role in requests},
        task_results=rows,optimizer_updates_added_by_publication=0,sampling_draws_added_by_publication=0)
    atomic_json(OUT/'results.json',summary)
    atomic_json(OUT/'receipts.json',receipts)
    atomic_json(OUT/'provenance.json',dict(schema_version=1,qualification_input=False,proofs=proof,matched_checks=comparisons,trace_audits=trace_audit))
    atomic_json(OUT/'media/index.json',dict(schema_version=1,qualification_input=False,media=media,optimizer_updates_added=0,sampling_draws_added=0))
    print(json.dumps(dict(outcomes=summary['outcomes'],media=len(media),paid_seconds=summary['campaign_accounting']['spent_seconds'])),flush=True)

if __name__=='__main__':main()
