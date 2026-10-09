"""Read saved evidence, declare the four-arm question, and inspect full costs."""
from pathlib import Path
from copy import deepcopy
import json, subprocess, time
import torch
from experiments.forge.contracts import atomic_json, read_json, file_hash
from experiments.forge.planning import resolve_idea
from experiments.forge.tier1_media import _scored_outputs
from particlegan.kinetic_transport import kinetic_transport_loss, kinetic_transport_local_loss
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
BRIEF=Path('/tmp/bcap-physics-round5-20261009/conditional_integration')
ARCHIVE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/conditional_integration')
ROLES=('winner','direction','transport','both')
PREFIX='conditional-integration-round5'

def main():
    start=time.monotonic();torch.set_num_threads(1)
    inspected=[]
    for family, location, roles in [
        ('direction',Path('/home/martyn/dev/ParticleGAN-bcap-r4-projection_ablation/reports/forge/bcap-physics/projection_ablation/round4'),{'direction_blend'}),
        ('composition',ROOT/'reports/forge/bcap-physics/projection_transport/round4',{'winner','transport','both'})]:
        publication=read_json(location/'provenance.json')
        for proof in publication['proofs']:
            taskid=proof['task_id']
            if proof['role'] not in roles or taskid not in ('trajectory','residual_student','mid_scale_identity','vector_unequal_mass','vector_two_broad','gaussian1d_stability'):continue
            desc=proof['provenance_checkpoint'];path=Path(desc['artifact_root'])/desc['path']
            assert file_hash(path)==desc['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=False)
            local=Path(desc['artifact_root']).parent
            result=read_json(local/'result.json')
            row=next(x for x in result['task_results'] if x['task_id']==taskid)
            request=read_json(local/'request.json')['request'];task=request['tasks'][taskid]
            outputs,_=_scored_outputs(task,row['evidence'],local)
            item=dict(family=family,role=proof['role'],task_id=taskid,
                original_source={k:request['source'][k] for k in ('origin_commit','digest')},checkpoint=str(path),checkpoint_sha256=desc['sha256'],
                completed_steps=desc['completed_steps'],gate_status=row['gate_status'],metrics=row['metrics'],
                result_sha256=file_hash(local/'result.json'))
            if outputs and 'views' in outputs[-1]:
                panel=outputs[-1]['views'][0];fake=panel['samples'].float().flatten(1);real=panel['target'].float().flatten(1)
                item['saved_output_marginal_probe']=dict(sliced_w2=float(kinetic_transport_loss(fake,real)),
                    local_v2=float(kinetic_transport_local_loss(fake,real)),
                    permuted_oracle_sliced_w2=float(kinetic_transport_loss(real.roll(1,0),real)),
                    permuted_oracle_local_v2=float(kinetic_transport_local_loss(real.roll(1,0),real)),
                    note='No draw, update, identity label or new paired training signal. Marginal matching admits row permutations.')
            inspected.append(item)
    assert len(inspected)>=12
    atomic_json(OUT/'prior-evidence.json',dict(schema_version=1,qualification_input=False,
        scope='source-pinned-saved-motivation',evidence=inspected,elapsed_seconds=time.monotonic()-start,
        diagnostic_allowance_seconds=120,optimizer_updates_added=0,sampling_draws_added=0))
    old=read_json(ROOT/'reports/forge/bcap-physics/projection_transport/round4/validation.json')
    snapshots=old['protected_original_files']
    if snapshots is None:
        snapshots=read_json(Path('/tmp/bcap-physics-round4-20261009/projection_transport/archived-snapshot-hashes.json'))
    assert all(file_hash(ROOT/path)==digest for path,digest in snapshots.items())
    atomic_json(BRIEF/'archived-snapshot-hashes.json',snapshots)
    tasks=['gaussian1d_smoke','gaussian1d_stability']
    variants=[]
    for host in ('trajectory','residual_student','mid_scale_identity'):
        original=read_json(ROOT/f'configs/forge/tasks/{host}.json');variant=deepcopy(original)
        variant['id']=f'{host}_transport_round5_v1';variant['task_cohort']='conditional_transport_round5_v1'
        variant['retained_question_ids']=[host]
        variant['description']=f'Explicit output-marginal transport-consumer variant of {host}; original objectives, conditions and full gates retained.'
        variant['execution']['transport_consumer']='output_marginal_v1'
        variant['execution']['transport_contract']=dict(sample_space='output_panel',
            conditioning='slow_arc' if host!='mid_scale_identity' else 'four_original_training_scales',
            conditioning_in_distance=False,target_law='original_training_target_panel',
            gradients='generated_outputs_to_original_network_and_learned_prior_locations',
            paired_correctness='original_objectives_and_original_full_identity_gates',
            new_paired_supervision=False)
        for path in ['particlegan/conditional_transport.py','particlegan/kinetic_transport.py','experiments/forge/behavior_adapters.py',
                'benchmarks/locked_shared/trajectory.py' if host=='trajectory' else f'benchmarks/locked_shared/hosts/{host}.py']:
            variant['evaluation']['sources'][path]=file_hash(ROOT/path)
        atomic_json(ROOT/f'configs/forge/tasks/{variant["id"]}.json',variant)
        tasks.append(variant['id']);variants.append(dict(id=variant['id'],original=host,original_sha256=file_hash(ROOT/f'configs/forge/tasks/{host}.json')))
    tasks+=['vector_unequal_mass','vector_two_broad']
    view=read_json(ROOT/'configs/forge/views/projection_transport-round4-diagnostic-v1.json')
    view.update(id=f'{PREFIX}-diagnostic-v1',policy_change_reason='Authorized explicit conditional consumers and four matched winner/direction/local-v2/BOTH arms; original gates, no ordinary qualification.')
    view['assignments']=[dict(task=t,order=i,qualification_tier=1,importance='diagnostic') for i,t in enumerate(tasks)]
    atomic_json(ROOT/f'configs/forge/views/{view["id"]}.json',view)
    plans={}
    for role in ROLES:
        card=read_json(ROOT/'configs/forge/ideas/projection_transport-round4-winner-v1.json')
        card.update(id=f'{PREFIX}-{role}-v1',guide=str(OUT.relative_to(ROOT)/'README.md'),
            parent='projection_transport-round4-winner-v1',mechanism_rationale=f'Four-arm explicit conditional-consumer diagnostic: {role}; exact direction and local-v2, original protected losses.',
            changed_factors=[f'One global {role} recipe on identical explicit conditional variants and unchanged scalar/vector hosts.'])
        card['recipe_overrides']['constraint_geometry_mode']='direction_blend' if role in ('direction','both') else 'none'
        for k in ('kinetic_transport_weight','kinetic_transport_local_weight'):
            card['recipe_overrides'][k]=1.0 if role in ('transport','both') else 0.0
        atomic_json(ROOT/f'configs/forge/ideas/{card["id"]}.json',card)
        study=read_json(ROOT/'configs/forge/studies/projection_transport-round4-both-study-v1.json')
        study.update(id=f'{PREFIX}-{role}-study-v1',candidate=card['id'],
            hypothesis='Direction-only blend plus exact local-v2 may retain both conditional identity repairs and rare/broad density repairs in one global recipe with explicit output-marginal consumers.',
            competing_explanation='Marginal transport is permutation invariant and may compete with existing identity signals; first-order common descent cannot guarantee finite loss descent, density or retention.')
        study['control']['candidate_id']=f'{PREFIX}-{"direction" if role=="winner" else "winner"}-v1'
        study['scope']['view']=view['id']
        study['campaign']=dict(id=f'{PREFIX}-v1',budget_seconds=42000,candidate_budget_seconds=10500)
        study['prior_evidence']=[dict(path=str((OUT/'prior-evidence.json').relative_to(ROOT)),selector=[],identity={'scope':'source-pinned-saved-motivation'},use='motivation_only')]
        atomic_json(ROOT/f'configs/forge/studies/{study["id"]}.json',study)
    for role in ROLES:
        request=resolve_idea(ROOT,f'{PREFIX}-{role}-v1',study=f'{PREFIX}-{role}-study-v1',queue_root=ARCHIVE/'queue',freeze_source=False)
        plans[role]=dict(candidate=request['candidate']['id'],study_review=request['study_review']['status'],jobs=[{k:j[k] for k in ('task_id','task_ids','budget_seconds')} for j in request['jobs']],blockers=request.get('preflight_blockers'))
    reservation=sum(read_json(ROOT/f'configs/forge/tasks/{t}.json')['resources']['timeout_seconds'] for t in tasks)*4
    assert reservation==38880 and reservation+1200<=43200
    atomic_json(OUT/'preregistration.json',dict(schema_version=1,scope='research_diagnostic',qualification_input=False,
        arms=ROLES,tasks=tasks,variants=variants,planned_full_reservations=reservation,campaign_ceiling=42000,
        track_ceiling=43200,ancillary_allowance_seconds=1200,saved_probe_allowance_seconds=120,
        prediction='BOTH must PASS all three conditional gates, unequal mass and broad to retain the measured union; Gaussian full retention is required for any global repair.',
        falsifier='Any conditional/rare/broad full-gate FAIL falsifies retained union; Gaussian retention FAIL blocks global repair regardless of endpoint improvement.',
        two_pole='Excluded; original transport-bearing two-pole contract remains BLOCKED; no implicit consumer.',
        original_unsupported_contracts_preserved=True,plans=plans))
    atomic_json(BRIEF/'progress.json',dict(phase='preregistered',hypothesis='Explicit marginal consumer plus exact direction blend/local-v2 union',
        armIDs=[f'{PREFIX}-{r}-v1' for r in ROLES],studies=[f'{PREFIX}-{r}-study-v1' for r in ROLES],campaign=f'{PREFIX}-v1',
        report=str(OUT/'README.md'),log=str(ARCHIVE/'logs/driver.log')))
    print(json.dumps(dict(saved_states=len(inspected),tasks=tasks,reservation=reservation)))
if __name__=='__main__':main()
