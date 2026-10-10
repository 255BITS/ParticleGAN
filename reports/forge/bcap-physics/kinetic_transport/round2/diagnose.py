"""Read-only output-field decomposition of certified round-one saved samples."""
from pathlib import Path
import sys, time
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
import torch
from experiments.forge.contracts import read_json,atomic_json,file_hash
from experiments.forge.api import task_formulation_context
from experiments.forge.vectorprofiles import resolve_vector_spec,build_vector_models
from experiments.forge.rng import NamedStreams
from benchmarks.transfer_suite.vector_tasks import sample_target
from particlegan.kinetic_transport import kinetic_transport_loss, kinetic_transport_local_loss


def main():
    start=time.monotonic(); torch.set_num_threads(1)
    provenance=read_json(ROOT/'reports/forge/bcap-physics/kinetic_transport/provenance.json')
    rows=[]
    for source in provenance['attempts']:
        if source['arm']!='candidate' or not source['task_id'].startswith('vector_'):continue
        local=Path(source['artifact_root'])
        for name,sha in source['inputs'].items(): assert file_hash(local/name)==sha
        request=read_json(local/'request.json')['request'];task=request['tasks'][source['task_id']]
        descriptor=source['provenance_checkpoint']; checkpoint=Path(descriptor['artifact_root'])/descriptor['path']
        assert file_hash(checkpoint)==descriptor['sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        result=read_json(local/'result.json')
        evidence=next(r['evidence'] for r in result['task_results'] if r['task_id']==task['id'])
        observed=local/evidence['saved_observer_outputs']['path']
        assert file_hash(observed)==evidence['saved_observer_outputs']['sha256']
        outputs=torch.load(observed,map_location='cpu',weights_only=True)[-1]['samples']
        spec=resolve_vector_spec(task)
        streams=NamedStreams(request['protocol']['seed'])
        data=streams.generator('data',component='target',purpose='training',device='cpu')
        for step in range(task['execution']['steps']):real=sample_target(spec,spec['batch'],data,step)
        key=next(k for k,b in saved['streams']['manifest']['bindings'].items() if b['family']=='data' and b['component']=='target' and b['purpose']=='training')
        assert torch.equal(data.get_state(),saved['streams']['states'][key])
        context=task_formulation_context(request['candidate'],task,request['protocol'],device='cpu',root=ROOT)
        _,critic=build_vector_models(context,spec)
        critic.load_state_dict(saved['trainer']['models']['D']);critic.eval();critic.requires_grad_(False)
        means=outputs.new_tensor(spec['means']);cov=outputs.new_tensor(spec['covariances'])
        transports=[];adversarials=[];locals_=[];mismatch=[]
        angles=torch.arange(32)*(torch.pi/32);frame=torch.stack((angles.cos(),angles.sin()))
        real_label=torch.cdist(real,means).argmin(1)
        for block in outputs.split(spec['batch']):
            fake=block.clone().requires_grad_(True)
            transport=kinetic_transport_loss(fake,real)
            local_loss=kinetic_transport_local_loss(fake,real)
            locals_.append(-torch.autograd.grad(local_loss,fake)[0].detach())
            adv=context.recipe.make_loss().g_loss(critic(fake),critic(real))
            transports.append(-torch.autograd.grad(transport,fake)[0].detach())
            adversarials.append(-torch.autograd.grad(adv,fake)[0].detach())
            assigned=torch.cdist(fake.detach(),means).argmin(1)
            ix=(fake.detach()@frame).argsort(0);iy=(real@frame).argsort(0)
            mismatch.append((assigned[ix]!=real_label[iy]).float().mean().item())
        ft=torch.cat(transports);fa=torch.cat(adversarials);fl=torch.cat(locals_)
        assigned=torch.cdist(outputs,means).argmin(1);delta=outputs-means[assigned]
        radius=torch.einsum('ni,nij,nj->n',delta,torch.linalg.inv(cov)[assigned],delta).sqrt()
        cohorts=[]
        for mode in range(len(means)):
            for scope,mask in [('core',radius<=3),('spill',radius>3)]:
                selected=(assigned==mode)&mask;n=int(selected.sum())
                if not n:continue
                radial=delta[selected]/delta[selected].norm(dim=1,keepdim=True).clamp_min(1e-12)
                item={'component':mode,'scope':scope,'count':n}
                for name,force in [('transport',ft),('adversarial',fa),('sum',ft+fa),('local',fl),('candidate_sum',ft+fa+fl)]:
                    f=force[selected];dot=(f*radial).sum(1)
                    item[name]={'mean_force_norm':f.norm(dim=1).mean().item(),'mean_outward_radial_force':dot.mean().item(),'inward_fraction':(dot<0).float().mean().item()}
                cosine=torch.nn.functional.cosine_similarity(ft[selected],fa[selected],dim=1)
                item['transport_adversarial_cosine_mean']=cosine.mean().item();cohorts.append(item)
        rows.append({'task_id':task['id'],'attempt_id':source['attempt_id'],'source_digest':source['source_digest'],
                     'checkpoint_sha256':file_hash(checkpoint),'observed_samples_sha256':file_hash(observed),
                     'probe_count':len(outputs),'real_batch':'exact final consumed target batch replayed from seed0 named data stream',
                     'mean_projected_pair_component_mismatch_fraction':sum(mismatch)/len(mismatch),'cohorts':cohorts})
    receipt={'schema_version':1,'scope':'posthoc_saved_output_field_diagnostic_not_training_phase_reconstruction',
             'numerical_contract':'Signs/norms of negative loss gradients in output space, evaluated with final saved critic and fixed saved served-output chunks. Target component labels used only for posthoc decomposition.',
             'optimizer_updates_added':0,'generated_sampling_draws_added':0,'earlier_baseline_probe_cpu_seconds':0.38017492298968136,'cpu_seconds':time.monotonic()-start,'local_module_sha256':file_hash(ROOT/'particlegan/kinetic_transport.py'),'diagnostics':rows}
    atomic_json(Path(__file__).with_name('prior-field-diagnostics.json'),receipt)
    print({'seconds':receipt['cpu_seconds'],'tasks':[(r['task_id'],r['mean_projected_pair_component_mismatch_fraction']) for r in rows]},flush=True)

if __name__=='__main__':main()
