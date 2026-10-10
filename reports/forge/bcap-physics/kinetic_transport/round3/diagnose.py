"""Frozen local-v2 endpoint: output fields, parameter pullbacks and finite proposals."""
from pathlib import Path
from copy import deepcopy
import sys,time
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
import torch
from experiments.forge.contracts import read_json,atomic_json,file_hash
from experiments.forge.api import task_formulation_context
from experiments.forge.vectorprofiles import resolve_vector_spec,build_vector_models
from experiments.forge.rng import NamedStreams


def dot(xs,ys):return sum((x*y).sum() for x,y in zip(xs,ys))
def cosine(xs,ys):return float(dot(xs,ys)/(dot(xs,xs)*dot(ys,ys)).sqrt().clamp_min(1e-30))


def main():
    started=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    prior=read_json(ROOT/'reports/forge/bcap-physics/kinetic_transport/round2/provenance.json');rows=[]
    for origin in prior['attempts']:
        if origin['arm']!='candidate' or origin['task_id']=='gaussian1d_smoke':continue
        local=Path(origin['artifact_root']);request=read_json(local/'request.json')['request'];task=request['tasks'][origin['task_id']]
        for name,sha in origin['inputs'].items():assert file_hash(local/name)==sha
        d=origin['provenance_checkpoint'];checkpoint=Path(d['artifact_root'])/d['path'];assert file_hash(checkpoint)==d['sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        spec=resolve_vector_spec(task);scalar=task['id'].startswith('gaussian')
        if scalar:
            from benchmarks.toy_audit.gaussian1d_quality import sample_target
        else:
            from benchmarks.transfer_suite.vector_tasks import sample_target
        data=NamedStreams(request['protocol']['seed']).generator('data',component='target',purpose='training',device='cpu')
        for step in range(1,saved['trainer']['completed_steps']+1):
            if scalar and step==4001:spec['means']=[[3.]]
            real=sample_target(spec,spec['batch'],data,step-1)
        key=next(k for k,b in saved['streams']['manifest']['bindings'].items() if b['family']=='data' and b['component']=='target' and b['purpose']=='training')
        assert torch.equal(data.get_state(),saved['streams']['states'][key])
        context=task_formulation_context(request['candidate'],task,request['protocol'],device='cuda:0',root=ROOT)
        g,critic=build_vector_models(context,resolve_vector_spec(task));trainer=context.build_trainer(g,critic,max_steps=saved['trainer']['completed_steps'])
        trainer.load_state_dict(saved['trainer']);trainer.G.train();trainer.D.eval();trainer.D.requires_grad_(False)
        real=real.to(trainer.device)
        # Additional explicitly scoped probe draws from cloned final streams.
        latent_rng=torch.Generator(device=trainer.device);latent_rng.set_state(saved['trainer']['streams']['latent_generator'])
        noise_rng=torch.Generator(device=trainer.device);noise_rng.set_state(saved['trainer']['streams']['prior_noise_generator'])
        latent,indices=trainer.prior.sample(len(real),generator=latent_rng,noise_generator=noise_rng)
        jitter=latent.detach()-trainer.prior.z[indices].detach()
        fake=trainer.G(latent)
        def losses(x):
            return {'adversarial':trainer.loss.g_loss(trainer.D(x),trainer.D(real)),
                    'sliced':trainer.recipe.kinetic_transport_loss(x,real),
                    'local':trainer.recipe.kinetic_transport_local_loss(x,real)}
        ls=losses(fake);params=[p for group in trainer.opt_g.param_groups for p in group['params']]
        output={name:torch.autograd.grad(value,fake,retain_graph=True)[0] for name,value in ls.items()}
        pullbacks={name:torch.autograd.grad(value,params,retain_graph=True,allow_unused=True) for name,value in ls.items()}
        pullbacks={name:[torch.zeros_like(p) if v is None else v for p,v in zip(params,grad)] for name,grad in pullbacks.items()}
        pairs=[]
        for a,b in [('adversarial','local'),('adversarial','sliced'),('sliced','local')]:
            pairs.append({'terms':[a,b],'output_gradient_cosine':cosine([output[a]],[output[b]]),'joint_parameter_gradient_cosine':cosine(pullbacks[a],pullbacks[b])})
        total=sum(ls.values());before=[p.detach().clone() for p in params]
        trainer.opt_g.zero_grad();total.backward();gradient=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
        trainer.opt_g.set_sampled_rows(trainer.prior.z,indices);trainer.opt_g.step()
        delta=[p.detach()-b for p,b in zip(params,before)];slope=float(dot(gradient,delta))
        trials=[];full_fake=None
        with torch.no_grad():
            for exponent in range(8):
                alpha=2.**(-exponent)
                if exponent:
                    for p,b,u in zip(params,before,delta):p.copy_(b+alpha*u)
                trial_fake=trainer.G(trainer.prior.z[indices]+jitter)
                values={name:float(value) for name,value in losses(trial_fake).items()}
                measured=sum(values.values());threshold=float(total)+1e-4*alpha*min(slope,0.)
                trials.append({'scale':alpha,'losses':values,'composite':measured,'armijo_bound':threshold,'accepted':measured<=threshold})
                if exponent==0:full_fake=trial_fake.detach().clone()
        output_slope=float(sum((field*(full_fake-fake.detach())).sum() for field in output.values()))
        rows.append({'task_id':task['id'],'attempt_id':origin['attempt_id'],'checkpoint_sha256':file_hash(checkpoint),'original_source_digest':origin['source_digest'],
                     'batch':'exact final consumed real batch replayed and final data stream verified; generated probe reuses one cloned final-stream 128-row/jitter draw at the frozen endpoint',
                     'generated_probe_draws':len(real),'archived_streams_mutated':False,'parameter_tensor_count':len(params),'gradient_cosines':pairs,
                     'losses_before':{k:float(v.detach()) for k,v in ls.items()},'composite_before':float(total.detach()),
                     'parameter_directional_derivative':slope,'output_gradient_dot_actual_full_output_motion':output_slope,
                     'full_proposal_composite_change':trials[0]['composite']-float(total.detach()),'trials':trials})
    torch.cuda.synchronize();receipt={'schema_version':1,'scope':'frozen_endpoint_proposal_diagnostic_not_actual_last_training_step','saved_training_updates_added':0,'optimizer_proposals_constructed':len(rows),'generated_probe_draws':sum(r['generated_probe_draws'] for r in rows),'wall_seconds':time.monotonic()-started,'device':'physical_gpu1','diagnostics':rows}
    atomic_json(Path(__file__).with_name('prior-response-diagnostics.json'),receipt)
    print({'seconds':receipt['wall_seconds'],'rows':[(r['task_id'],r['parameter_directional_derivative'],r['full_proposal_composite_change']) for r in rows]},flush=True)

if __name__=='__main__':main()
