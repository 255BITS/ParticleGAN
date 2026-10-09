"""Read-only parity of saved singletons and inactive projection, without sampling."""
from pathlib import Path
import importlib.util
import torch
from experiments.forge.contracts import atomic_json, read_json, file_hash
from experiments.forge.tier1_media import _scored_outputs
from publish import ROOT, QUEUE, OUT, REQUESTS

# Shared tensor comparison from the already-published predecessor audit.
spec=importlib.util.spec_from_file_location('saved_parity',ROOT/'reports/forge/bcap-physics/constraint_geometry/round3/audit_saved_parity.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
compare=prior.compare

DEFAULT_FIELDS={'kinetic_transport_weight','kinetic_transport_local_weight','kinetic_transport_projections'}

def normalize(value, *, remove_projection=False):
    skip=DEFAULT_FIELDS|{'cpu_rng','cuda_rng','training_state_sha256','primary_state_sha256','confirmed_state_sha256'}
    if remove_projection:skip|={'constraint_geometry','constraint_geometry_mode','strict_progress'}
    if isinstance(value,dict):return {k:normalize(v,remove_projection=remove_projection) for k,v in value.items() if k not in skip}
    if isinstance(value,(list,tuple)):return type(value)(normalize(v,remove_projection=remove_projection) for v in value)
    return value


def state_cohort(saved):
    # Recipe/optimizer class labels are provenance, not trained numerical state.
    keys=('trainer','streams','initialization','prior') if 'trainer' in saved else ('models','role_parameters','optimizers','streams')
    return {k:saved[k] for k in keys}


def load(directory,task):
    result=read_json(directory/'result.json')
    row=next(r for r in result['task_results'] if r['task_id']==task['id'])
    descriptor=row['evidence']['provenance_checkpoint'];path=Path(descriptor['artifact_root'])/descriptor['path']
    assert file_hash(path)==descriptor['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    arrays,_=_scored_outputs(task,row['evidence'],directory)
    return {**row,'attempt_id':result['attempt_id']},saved,arrays,descriptor


def main():
    torch.set_num_threads(1)
    state=read_json(QUEUE/'queue/state.json');checks=[]
    current={}
    for role,rid in REQUESTS.items():
        request=state['submissions'][rid]['request']
        for job in state['jobs'].values():
            if rid not in job['subscribers'] or not job.get('result'):continue
            task=request['tasks'][job['definition']['task_id']]
            current[(role,task['id'])]=load(Path(job['attempts'][-1]['path']),task)
    projection_report=ROOT/'reports/forge/bcap-physics/constraint_geometry/round3'
    transport_report=Path('/home/martyn/dev/ParticleGAN-bcap-physics-kinetic_transport/reports/forge/bcap-physics/kinetic_transport/round2')
    for role,report in [('projection',projection_report),('transport',transport_report)]:
        provenance=read_json(report/'provenance.json')
        for proof in provenance.get('proofs',provenance.get('attempts',[])):
            if proof.get('role',proof.get('arm'))!='candidate':continue
            task_id=proof['task_id']
            if (role,task_id) not in current:continue
            request=state['submissions'][REQUESTS[role]]['request'];task=request['tasks'][task_id]
            if 'artifact_root' in proof:directory=Path(proof['artifact_root'])
            else:
                directory=Path(proof['provenance_checkpoint']['artifact_root']).parent
            # Original projection receipts live in this checkout; local original evidence stays archived.
            old=load(directory,task);new=current[(role,task_id)]
            diffs=dict(final_state=compare(normalize(state_cohort(new[1])),normalize(state_cohort(old[1]))),
                       observations=compare(new[0]['evidence']['observations'],old[0]['evidence']['observations']),
                       endpoint_metrics=compare(new[0]['metrics'],old[0]['metrics']),
                       numerical_outputs=compare(normalize(new[2]),normalize(old[2])))
            checks.append(dict(scope='archived_singleton_reproduction_no_qualification',role=role,task_id=task_id,
                               archived_attempt_id=proof['attempt_id'],new_attempt_id=new[0].get('attempt_id'),
                               archived_result_sha256=file_hash(directory/'result.json'),
                               archived_checkpoint_sha256=old[3]['sha256'],new_checkpoint_sha256=new[3]['sha256'],
                               mismatches=diffs,bitwise_equal=all(not d for d in diffs.values())))
    for task_id in ('two_pole','gaussian1d_smoke','gaussian1d_stability','vector_unequal_mass','vector_two_broad'):
        base=current[('winner',task_id)];guarded=current[('projection',task_id)]
        diffs=dict(final_state=compare(normalize(state_cohort(base[1]),remove_projection=True),normalize(state_cohort(guarded[1]),remove_projection=True)),
                   observations=compare(base[0]['evidence']['observations'],guarded[0]['evidence']['observations']),
                   numerical_outputs=compare(normalize(base[2],remove_projection=True),normalize(guarded[2],remove_projection=True)))
        checks.append(dict(scope='new_source_inactive_projection_parity',role='winner/projection',task_id=task_id,
                           mismatches=diffs,bitwise_equal=all(not d for d in diffs.values())))
    atomic_json(OUT/'saved-parity.json',dict(schema_version=1,qualification_input=False,checks=checks,
               excluded_provenance_only_component_fields=['applied.public_optimizers','applied.optimizer_group_bindings.optimizer'],excluded_unused_ambient_state=['cpu_rng','cuda_rng'],optimizer_updates_added=0,sampling_draws_added=0))
    print([(x['role'],x['task_id'],x['bitwise_equal']) for x in checks])
    assert all(x['bitwise_equal'] for x in checks),'inspect saved mismatch paths; never rerun unchanged science for parity'

if __name__=='__main__':main()
