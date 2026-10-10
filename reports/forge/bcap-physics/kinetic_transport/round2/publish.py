"""Publish only this completed campaign's saved evidence; never train or sample."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge import tier1_media

CAMPAIGN = 'kinetic_transport_round2'
WINNER = 'kinetic_transport_sliced_v1'
OUT = ROOT / 'reports/forge/bcap-physics/kinetic_transport/round2'


def vector_state_diagnostic(request, task, evidence):
    """Deterministic saved-center census and data replay in a separate CPU cohort."""
    from experiments.forge.rng import NamedStreams
    from experiments.forge.api import task_formulation_context
    from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
    from benchmarks.transfer_suite.vector_tasks import sample_target
    descriptor=evidence['provenance_checkpoint']
    path=Path(descriptor['artifact_root'])/descriptor['path']
    assert file_hash(path)==descriptor['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    streams=NamedStreams(request['protocol']['seed'])
    data=streams.generator('data',component='target',purpose='training',device='cpu')
    spec=resolve_vector_spec(task);digest=hashlib.sha256();rare_absent=0;rare_counts=[]
    means=torch.tensor(spec['means']);rare=min(range(len(spec['masses'])),key=lambda i:spec['masses'][i])
    for step in range(task['execution']['steps']):
        batch=sample_target(spec,spec['batch'],data,step)
        digest.update(batch.numpy().tobytes())
        count=int((torch.cdist(batch,means).argmin(1)==rare).sum())
        rare_counts.append(count);rare_absent+=count==0
    key=next(key for key,binding in saved['streams']['manifest']['bindings'].items()
             if binding['family']=='data' and binding['component']=='target' and binding['purpose']=='training')
    assert torch.equal(data.get_state(),saved['streams']['states'][key])
    context=task_formulation_context(request['candidate'],task,request['protocol'],device='cpu',root=ROOT)
    generator,_=build_vector_models(context,spec)
    generator.load_state_dict(saved['trainer']['models']['G']);generator.eval()
    locations=saved['trainer']['models']['prior']['z']
    with torch.no_grad():
        points=generator(locations)
        means=points.new_tensor(spec['means'])
        assigned=torch.cdist(points,means).argmin(1)
        counts=torch.bincount(assigned,minlength=len(means))
    return {'task_id':task['id'],'scope':'deterministic_centers_not_served_samples','device':'cpu',
            'optimizer_updates_added':0,'generated_sampling_draws_added':0,
            'target_training_batches_reconstructed':task['execution']['steps'],
            'replayed_data_sha256':digest.hexdigest(),'checkpoint_data_stream_equal':True,
            'posthoc_rarest_nearest_component_index':rare,
            'target_batches_missing_rarest_component':rare_absent,
            'target_batch_rarest_count_mean':sum(rare_counts)/len(rare_counts),
            'center_counts':counts.tolist(),'center_mass':(counts/len(locations)).tolist(),
            'target_mass':spec['masses'],'input_checkpoint_sha256':descriptor['sha256']}


def public_bounds(task, metrics):
    return [[key, op, bound, metrics.get(key)] for key, op, bound in task['evaluation']['thresholds']
            if metrics.get(key) is None or not math.isfinite(metrics[key]) or not
            (metrics[key] >= bound if op == '>=' else metrics[key] <= bound if op == '<=' else metrics[key] == bound)]


def summary(task, row):
    evidence = row.get('evidence', {})
    curve = evidence.get('observations', [])
    passing = [not public_bounds(task, p) for p in curve]
    suffix = 0
    for passed in reversed(passing):
        if not passed: break
        suffix += 1
    final = evidence.get('live', curve[-1] if curve else {})
    failure_counts=Counter(key for point in curve for key,_,_,_ in public_bounds(task,point))
    last_window=[{'step':point['step'],'failed_bounds':public_bounds(task,point)} for point in curve[-5:]]
    guards={k:v for k,v in evidence.get('guards',{}).items() if k!='mechanism_audit'}
    if evidence.get('guards',{}).get('mechanism_audit'):
        guards['mechanism_audit_sha256']=stable_hash(evidence['guards']['mechanism_audit'])
    evaluator={k:v for k,v in row.get('evaluator_result',{}).items() if k not in ('metrics','confirmed_steps')}
    return dict(task_id=task['id'], status=row['gate_status'], final=final,
                failed_endpoint_bounds=public_bounds(task, final) if final else [],
                observations=len(curve), passing_checks=sum(passing), terminal_passing_suffix=suffix,
                failed_bound_counts=dict(failure_counts),last_five_gate_failures=last_window,
                sustained_gate=evaluator, reason=row.get('reason'), guards=guards,
                continuity=evidence.get('continuity'), data_sha256=evidence.get('data_sha256'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/kinetic_transport/queue'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    state=read_json(args.queue/'queue/state.json')
    requests={key:v for key,v in state['submissions'].items() if v['request']['campaign_id']==CAMPAIGN}
    assert len(requests)==2
    assert all(v['status'] not in ('queued','running','paused') for v in requests.values()), 'finish all jobs before publication'
    assert state['campaigns'][CAMPAIGN]['reserved_seconds']==0
    arms={}; provenance=[]; media=[]; conditions={}; data_audits=[]; full_reservations=0
    for rid,submission in requests.items():
        request=submission['request'];arm='control' if request['candidate']['id']==WINNER else 'candidate'
        rows=[]
        for job in state['jobs'].values():
            if rid not in job.get('subscribers',[]): continue
            result=job.get('result');task_id=job['definition']['task_id'];task=request['tasks'][task_id]
            if result is None:
                reasons=task.get('preflight_blockers',[])
                rows.append(dict(task_id=task_id,status='BLOCKED',reason=reasons or 'own prerequisite has no passing checkpoint',observations=0))
                continue
            attempt=result['attempt_id'];directory=Path(job['attempts'][-1]['path'])
            full_reservations+=task['resources']['timeout_seconds']
            envelope=read_json(directory/'request.json');saved=read_json(directory/'result.json')
            assert saved==result and saved['candidate_revision']==request['candidate_revision']
            assert envelope['request']['source']['digest']==request['source']['digest']
            row=next(r for r in result['task_results'] if r['task_id']==task_id)
            item=summary(task,row);item.update(attempt_id=attempt,wall_seconds=row['cost']['wall_seconds']);rows.append(item)
            raw=read_json(directory/'raw-result.json')
            evidence=row.get('evidence',{})
            applied=raw.get('applied',raw)
            if task_id.startswith('vector_') and evidence.get('provenance_checkpoint'):
                diagnostic=vector_state_diagnostic(request,task,evidence)
                data_audits.append({'arm':arm,**diagnostic})
            conditions.setdefault(task_id,{})[arm]={'initialization':applied.get('initialization'),
                'prior':applied.get('prior'),'recipe':applied.get('recipe'),'data_sha256':evidence.get('data_sha256'),
                'named_stream_state_sha256':evidence.get('provenance_checkpoint',{}).get('named_stream_state_sha256')}
            provenance.append({'arm':arm,'task_id':task_id,'attempt_id':attempt,'artifact_root':str(directory),
                'inputs':{name:file_hash(directory/name) for name in ('request.json','result.json','raw-result.json','terminal.json')},
                'result_hash':stable_hash(result),'source_digest':request['source']['digest'],'candidate_revision':request['candidate_revision'],
                'task_science_sha256':stable_hash(job['definition']['science']),
                'task_execution_fingerprint':job['definition']['science']['execution'],
                'task_evaluation_fingerprint':job['definition']['science']['evaluation'],
                'compute':job['definition']['science']['compute'],
                'protocol_sha256':stable_hash(job['definition']['science']['protocol']),
                'rng_binding_sha256':stable_hash(job['definition']['science']['rng']),
                'provenance_checkpoint':{k:v for k,v in evidence.get('provenance_checkpoint',{}).items()
                    if k in ('artifact_root','path','sha256','state_sha256','completed_steps','bytes','prerequisite_eligible')}})
            if evidence.get('saved_observer_outputs'):
                records,_=tier1_media._scored_outputs(task,evidence,directory)
                scalar=task_id.startswith('gaussian')
                if scalar:
                    from benchmarks.toy_audit.gaussian1d_quality import score_samples
                else:
                    from benchmarks.transfer_suite.vector_tasks import score_samples
                observations=evidence['observations']
                spec={**task['execution']['host_definition'],'thresholds':task['evaluation']['thresholds']}
                for record,observed in zip(records,observations):
                    if scalar and task_id=='gaussian1d_stability' and record['step']>4000:spec['means']=[[3.]]
                    rescored=score_samples(record['samples'],spec,record['step'])
                    assert rescored=={k:v for k,v in observed.items() if k!='step'},(task_id,record['step'])
                receipt=tier1_media.render(task,row,directory,OUT/f'{arm}-{task_id}.gif')
                media.append({'arm':arm,**receipt,'metric_sets_reproduced':len(records)})
            elif evidence.get('observations'):
                receipt=tier1_media.render(task,row,directory,OUT/f'{arm}-{task_id}.gif')
                media.append({'arm':arm,**receipt,'metric_sets_reproduced':None,
                              'scope':'saved_actual_training_measurements'})
        arms[arm]={'candidate_id':request['candidate']['id'],'revision':request['candidate_revision'],
            'source_digest':request['source']['digest'],'source_origin_commit':request['source'].get('origin_commit'),
            'runtime':request['runtime'],'request_id':rid,'submission_status':submission['status'],
            'outcomes':dict(Counter(r['status'] for r in rows)),'tasks':sorted(rows,key=lambda r:r['task_id'])}
    matched=[]
    for task_id,values in conditions.items():
        if set(values)!={'candidate','control'}:continue
        a,b=values['candidate'],values['control']
        equality={key:a[key]==b[key] for key in ('initialization','prior','data_sha256','named_stream_state_sha256')}
        assert all(equality.values()),(task_id,equality)
        # Preserve archived omission of default additions, compare effective values.
        defaults={'kinetic_transport_weight':0.,'kinetic_transport_projections':32,'kinetic_transport_local_weight':0.}
        value=lambda recipe,key:recipe.get(key,defaults.get(key))
        delta={key:[value(b['recipe'],key),value(a['recipe'],key)] for key in a['recipe'].keys()|b['recipe'].keys()
               if value(a['recipe'],key)!=value(b['recipe'],key)}
        assert delta=={'kinetic_transport_local_weight':[0.,1.]},(task_id,delta)
        matched.append({'task_id':task_id,'equal':equality,'declared_and_consumed_recipe_delta':delta,
            'binding_hashes':{arm:{key:stable_hash(value) for key,value in v.items()} for arm,v in values.items()},
            'resolved_task_recipe_resources':{k:a['recipe'].get(k) for k in ('z_dim','num_particles','batch_size','total_steps','prior_kind','standardize')},
            'missing_adapter_data_digest':a['data_sha256'] is None,
            'data_digest_reconstruction':'saved-state-diagnostics.json' if task_id.startswith('vector_') else None})
    for task_id in {r['task_id'] for r in data_audits}:
        hashes={r['replayed_data_sha256'] for r in data_audits if r['task_id']==task_id}
        assert len(hashes)==1
    cost=state['campaigns'][CAMPAIGN]
    cost['full_reservations_for_executed_jobs_seconds']=full_reservations
    readout_records={}
    for arm,entry in arms.items():
        request=requests[entry['request_id']]['request']
        identity={'candidate':entry['candidate_id'],'revision':entry['revision'],
                  'study_id':request['study']['id'],'study_sha256':stable_hash(request['study'])}
        record_id='readout-'+stable_hash(identity)[:24]
        record_path=ROOT/'reports/forge/records'/f'{record_id}.json'
        if record_path.is_file():
            record=read_json(record_path)
            assert record['qualification_input'] is record['qualification_reuse'] is False
            assert {r['task_id']:r['gate_status'] for r in record['task_results']}=={
                r['task_id']:r['status'] for r in entry['tasks'] if r.get('attempt_id')}
            record.setdefault('original_source',record['source'])
            record['source']={'path':'reports/forge/bcap-physics/kinetic_transport/round2/results.json'}
            record['summary_curation']='Compact committed report navigation; original receipt identities and numerical verdicts unchanged.'
            atomic_json(record_path,record)
            readout_records[arm]={'record_id':record_id,'path':str(record_path.relative_to(ROOT)),'sha256':file_hash(record_path)}
    atomic_json(OUT/'results.json',{'schema_version':1,'campaign':CAMPAIGN,'qualification_input':False,
        'scope':'research_diagnostic','protocol_seed':0,'arms':arms,'cost':cost,
        'readout_records':readout_records,
        'scientific_retries':sum(max(0,len(j['attempts'])-1) for j in state['jobs'].values() if set(requests)&set(j.get('subscribers',[])))})
    atomic_json(OUT/'provenance.json',{'schema_version':1,'campaign':CAMPAIGN,'public_package':__import__('particlegan').__file__,
        'paid_worker_seconds':cost['spent_seconds'],'maximum_declared_reservation_seconds':12840,'remaining_reserved_seconds':cost['reserved_seconds'],
        'full_reservations_for_executed_jobs_seconds':full_reservations,
        'full_resolved_recipe_example':conditions['vector_unequal_mass'],
        'attempts':provenance,'matched_conditions':matched,'bulk_artifacts':'Shared local queue; original bytes retained, not committed.'})
    atomic_json(OUT/'media.json',{'schema_version':1,'optimizer_updates_added':0,'sampling_draws_added':0,'media':media})
    atomic_json(OUT/'saved-state-diagnostics.json',{'schema_version':1,'qualification_input':False,
        'optimizer_updates_added':0,'generated_sampling_draws_added':0,'diagnostics':data_audits})
    print({'stage':'published','outcomes':{a:v['outcomes'] for a,v in arms.items()},'cost':cost,'media':len(media)},flush=True)

if __name__=='__main__':main()
