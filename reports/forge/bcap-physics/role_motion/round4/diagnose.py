"""Five disposable one-update probes from byte-exact archived sources.

This is a next-update diagnostic, not continuation evidence. Restore every
model, optimizer and stream after role swaps; component labels are posthoc only.
Full allowance: five120-second probes, within the18000-second track ceiling.
"""
from pathlib import Path
import argparse, json, sys, time

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())['request']
    source=Path(request['source']['snapshot_path'])
    sys.path.insert(0,str(source))
    import torch
    from torch.func import functional_call
    from experiments.forge.api import task_formulation_context
    from experiments.forge.adapters import _models
    from experiments.forge.vectorprofiles import build_vector_models,resolve_vector_spec
    from experiments.forge.contracts import atomic_json,file_hash
    from experiments.forge.state import state_digest
    from benchmarks.transfer_suite.vector_tasks import sample_target
    from benchmarks.toy100.problems import sample_real,evaluation_geometry
    start=time.monotonic();torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for name,sha in request['source']['files'].items(): assert file_hash(source/name)==sha
    assert file_hash(args.checkpoint)==args.sha256
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    before=state_digest(saved)
    task=request['tasks'][args.task];device='cuda:0'
    context=task_formulation_context(request['candidate'],task,request['protocol'],device=device,root=source)
    if args.task=='grid100':
        g,d=_models(context,task['execution']['model'])
    else:
        spec=resolve_vector_spec(task,root=source);g,d=build_vector_models(context,spec)
    trainer=context.build_trainer(g,d,max_steps=saved['trainer'].get('max_steps',context.recipe.total_steps))
    # Register retained evaluation/data streams before validating the registry.
    for b in saved['streams']['manifest']['bindings'].values():
        context.streams.generator(b['family'],component=b['component'],purpose=b['purpose'],device=b['device'])
    context.load_state_dict(saved)
    original=context.state_dict();original_digest=state_digest(original)
    binding=next(b for b in saved['streams']['manifest']['bindings'].values()
                 if b['family']=='data' and b['component']=='target' and b['purpose']=='training')
    data=context.streams.generator('data',component='target',purpose='training',device=binding['device'])
    real=(sample_real(args.task,context.recipe.batch_size,device=device,generator=data)
          if args.task=='grid100' else sample_target(spec,spec['batch'],data,trainer.completed_steps+1).to(device))
    cap=trainer.max_steps;trainer.extend_execution(trainer.completed_steps+1)
    sample=trainer._sample_training_prior;captured={}
    def capture(n):
        latent,indices=sample(n)
        captured['latent']=latent.detach().clone();captured['indices']=indices.detach().clone()
        return latent,indices
    trainer._sample_training_prior=capture
    step=trainer.opt_g.step
    def applied(*a,**kw):
        captured['G0']={k:p.detach().clone() for k,p in g.named_parameters()}
        captured['z0']=trainer.prior.z.detach().clone()
        result=step(*a,**kw)
        captured['G1']={k:p.detach().clone() for k,p in g.named_parameters()}
        captured['z1']=trainer.prior.z.detach().clone()
        return result
    trainer.opt_g.step=applied
    trainer.step(real)
    ix=captured['indices'];z=captured['latent'];c=captured['z0'][ix]
    dz=captured['z1'][ix]-c;jitter=z-c
    def forward(which,latent):return functional_call(g,captured[which],(latent,))
    def rms(x):return float(x.square().flatten(1).sum(1).mean().sqrt())
    def stats(x):
        x=x.detach().flatten().double()
        return dict(mean=float(x.mean()),median=float(x.median()),p90=float(x.quantile(.9)),max=float(x.max()))
    with torch.no_grad():
        y0=forward('G0',z);yg=forward('G1',z);yp=forward('G0',z+dz);yj=forward('G1',z+dz)
        dcg=forward('G1',c)-forward('G0',c)
        dcp=forward('G0',c+dz)-forward('G0',c)
        dcj=forward('G1',c+dz)-forward('G0',c)
        motions={'network':yg-y0,'prior':yp-y0,'joint':yj-y0}
        centers={'network':dcg,'prior':dcp,'joint':dcj}
        cross=motions['joint']-motions['network']-motions['prior']
        result=dict(schema_version=1,scope='one_next_update_disposable_copy_of_frozen_endpoint',
            qualification_input=False,task_id=args.task,candidate_id=request['candidate']['id'],
            source_digest=request['source']['digest'],source_commit=request['source']['origin_commit'],
            checkpoint=str(args.checkpoint),checkpoint_sha256=args.sha256,
            trained_prefix_steps=saved['trainer']['completed_steps'],preview_updates=1,
            diagnostic_new_target_batch=True,extra_evaluation_rng_draws=0,
            sampled_rows=len(ix),prior_learnable=trainer.prior.z.requires_grad,
            role_motion={k:dict(output_rms=rms(v),norms=stats(v.norm(dim=1)),
                center_rms=rms(centers[k]),jitter_deformation_rms=rms(v-centers[k])) for k,v in motions.items()},
            initial_jitter_output_rms=rms(y0-forward('G0',c)),
            nonlinear_interaction_rms=rms(cross),role_output_cosine=float(torch.nn.functional.cosine_similarity(
                motions['network'].flatten()[None],motions['prior'].flatten()[None])),
            prior_latent_drift_rms=rms(dz),cohorts=[])
        if args.task=='grid100':
            means,sigma=evaluation_geometry(args.task,device=device)
            cov=torch.eye(2,device=device).repeat(100,1,1)*sigma**2
        else:
            means=real.new_tensor(spec['means']);cov=real.new_tensor(spec['covariances'])
        labels=torch.cdist(y0,means).argmin(1);delta=y0-means[labels]
        radii=torch.einsum('ni,nij,nj->n',delta,torch.linalg.inv(cov)[labels],delta).sqrt()
        for k in range(len(means)):
            for region,mask in [('core',radii<=3),('spill',radii>3)]:
                mask=mask&(labels==k)
                if not mask.any():continue
                cohort=dict(component=k,region=region,rows=int(mask.sum()),role_motion={})
                radial=delta[mask]/delta[mask].norm(dim=1,keepdim=True).clamp_min(1e-12)
                for role,motion in motions.items():
                    cohort['role_motion'][role]=dict(output_rms=rms(motion[mask]),
                        mean_outward_displacement=float((motion[mask]*radial).sum(1).mean()),
                        jitter_deformation_rms=rms((motion-centers[role])[mask]))
                result['cohorts'].append(cohort)
    trainer.max_steps=cap
    context.load_state_dict(original)
    result.update(restored_full_context_exact=state_digest(context.state_dict())==original_digest,
                  input_checkpoint_unchanged=state_digest(saved)==before and file_hash(args.checkpoint)==args.sha256,
                  paid_probe_seconds=time.monotonic()-start,full_reserved_seconds=120)
    assert result['restored_full_context_exact'] and result['input_checkpoint_unchanged']
    atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('task_id','candidate_id','role_motion','initial_jitter_output_rms','nonlinear_interaction_rms','role_output_cosine','paid_probe_seconds')}),flush=True)

if __name__=='__main__':main()
