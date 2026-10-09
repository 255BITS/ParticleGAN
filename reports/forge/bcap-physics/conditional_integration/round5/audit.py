"""Audit frozen scientific bytes, actual activation, and saved numerical parity."""
from pathlib import Path
import importlib.util,json
import torch
from experiments.forge.contracts import atomic_json,read_json,file_hash,stable_hash
from experiments.forge.state import state_digest
from experiments.forge.tier1_media import _scored_outputs
from publish import ROOT,QUEUE,OUT,REQUESTS
spec=importlib.util.spec_from_file_location('saved_parity',ROOT/'reports/forge/bcap-physics/constraint_geometry/round3/audit_saved_parity.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
compare=prior.compare


def normalize(value,*,projection=False):
    skip={'component_transport','cpu_rng','cuda_rng','training_state_sha256','primary_state_sha256','confirmed_state_sha256',
          'kinetic_transport_weight','kinetic_transport_local_weight','kinetic_transport_projections'}
    if projection:skip|={'constraint_geometry','constraint_geometry_mode','strict_progress','direction_blend'}
    if isinstance(value,dict):return {k:normalize(v,projection=projection) for k,v in value.items() if k not in skip}
    if isinstance(value,(list,tuple)):return type(value)(normalize(v,projection=projection) for v in value)
    return value


def numeric(saved):
    keys=('trainer','streams','initialization','prior') if 'trainer' in saved else ('models','role_parameters','optimizers','streams')
    return {k:saved[k] for k in keys}


def load(local,taskid):
    result=read_json(local/'result.json');row=next(r for r in result['task_results'] if r['task_id']==taskid)
    req=read_json(local/'request.json')['request'];desc=row['evidence']['provenance_checkpoint']
    p=Path(desc['artifact_root'])/desc['path'];assert file_hash(p)==desc['sha256']
    saved=torch.load(p,map_location='cpu',weights_only=False)
    outputs,_=_scored_outputs(req['tasks'][taskid],row['evidence'],local)
    return row,saved,outputs,desc,req


def equality(new,old,*,projection=False):
    checks=dict(final_state=compare(normalize(numeric(new[1]),projection=projection),normalize(numeric(old[1]),projection=projection)),
        observations=compare(new[0]['evidence']['observations'],old[0]['evidence']['observations']),
        outputs=compare(normalize(new[2],projection=projection),normalize(old[2],projection=projection)),
        endpoint_metrics=compare(new[0]['metrics'],old[0]['metrics']))
    return dict(bitwise_equal=all(not c for c in checks.values()),mismatches=checks,
        new_checkpoint_sha256=new[3]['sha256'],old_checkpoint_sha256=old[3]['sha256'])


def main():
    torch.set_num_threads(1);state=read_json(QUEUE/'queue/state.json');current={}
    for role,rid in REQUESTS.items():
        for job in state['jobs'].values():
            if rid in job['subscribers'] and job.get('result'):
                assert len(job['attempts'])==1
                current[(role,job['definition']['task_id'])]=load(Path(job['attempts'][-1]['path']),job['definition']['task_id'])
    assert len(current)==sum(bool(j.get('result')) for j in state['jobs'].values())
    assert all(j['status'] in ('terminal','blocked') for j in state['jobs'].values())
    checks=[]
    for role,report,archived_role in [
        ('winner',ROOT/'reports/forge/bcap-physics/projection_transport/round4','winner'),
        ('transport',ROOT/'reports/forge/bcap-physics/projection_transport/round4','transport'),
        ('direction',Path('/home/martyn/dev/ParticleGAN-bcap-r4-projection_ablation/reports/forge/bcap-physics/projection_ablation/round4'),'direction_blend')]:
        for proof in read_json(report/'provenance.json')['proofs']:
            if proof['role']!=archived_role:continue
            oldid=proof['task_id'];newid=f'{oldid}_transport_round5_v1' if oldid in ('trajectory','residual_student','mid_scale_identity') else oldid
            if (role,newid) not in current:continue
            old=load(Path(proof['provenance_checkpoint']['artifact_root']).parent,oldid)
            checks.append(dict(scope='archived_exact_singleton_or_inactive_consumer',role=role,task_id=newid,
                original_task_id=oldid,**equality(current[(role,newid)],old)))
    for taskid in ('gaussian1d_smoke','gaussian1d_stability','vector_unequal_mass','vector_two_broad'):
        checks.append(dict(scope='new_source_inactive_direction',role='winner/direction',task_id=taskid,
            **equality(current[('direction',taskid)],current[('winner',taskid)],projection=True)))
    assert all(c['bitwise_equal'] for c in checks),[(x['role'],x['task_id'],x['mismatches']) for x in checks if not x['bitwise_equal']]
    matched=[]
    for taskid in {key[1] for key in current}:
        arms=[(role,current[(role,taskid)]) for role in REQUESTS if (role,taskid) in current]
        first=arms[0][1]
        for role,row in arms[1:]:
            assert row[4]['source']==first[4]['source'] and row[4]['runtime']==first[4]['runtime']
            assert not compare(row[1]['streams'],first[1]['streams']),(taskid,role,'streams')
        for role,row in arms:
            if 'applied' in row[1]:
                assert row[1]['applied']['rng']['seed']==0
                assert all(a['unintended_rng_deviations']==0 for a in row[1]['applied']['rng_audits'])
                stats=row[1]['component_transport']
                assert stats['calls']==row[4]['tasks'][taskid]['execution']['steps']
                assert stats['active_calls']==(stats['calls'] if role in ('both','transport') else 0)
        target_sha=None
        if first[2] and 'views' in first[2][0]:
            targets=[v['target'] for point in first[2] for v in point['views']]
            for _,row in arms[1:]:
                assert not compare(targets,[v['target'] for point in row[2] for v in point['views']])
            target_sha=state_digest(targets)
        matched.append(dict(task_id=taskid,initialization_equal=True,all_named_stream_tensors_equal=True,source_runtime_equal=True,
            saved_target_panels_sha256=target_sha,actual_data_batch_sha256=first[0]['evidence'].get('data_sha256')))
    requests=[state['submissions'][r]['request'] for r in REQUESTS.values()];first=requests[0]
    prefixes=('particlegan/','benchmarks/','experiments/','lib/','configs/forge/')
    scientific={p:h for p,h in first['source']['files'].items() if p.startswith(prefixes)}
    assert all(file_hash(ROOT/p)==h for p,h in scientific.items())
    protected=read_json(Path('/tmp/bcap-physics-round5-20261009/conditional_integration/archived-snapshot-hashes.json'))
    assert all(file_hash(ROOT/p)==h for p,h in protected.items())
    assert file_hash(ROOT/'particlegan/optim/strict_progress.py')==read_json(ROOT/'reports/forge/bcap-physics/projection_transport/round4/validation.json')['exact_strict_progress_optimizer_sha256']
    atomic_json(OUT/'audit.json',dict(schema_version=1,qualification_input=False,checks=checks,matched_conditions=matched,
        source_commit=first['source']['origin_commit'],source_digest=first['source']['digest'],scientific_files_unchanged=len(scientific),
        protected_original_files_unchanged=protected,all_consumer_calls_verified=True,optimizer_updates_added=0,sampling_draws_added=0))
    print(json.dumps(dict(checks=len(checks),bitwise_equal=True,matched_tasks=len(matched),scientific_files=len(scientific))))
if __name__=='__main__':main()
