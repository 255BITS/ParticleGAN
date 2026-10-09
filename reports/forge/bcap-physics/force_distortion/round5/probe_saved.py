"""Bounded reversible force traces from the exact winner's saved source.

Native/vector: one discarded public next step, including its actual D update.
Conditional: fixed saved critic, exact existing loss, actual saved optimizers.
Neither cohort attributes the training history. No new qualified evidence.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time

ORIGINAL = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-tier2-search-v1/snapshots/2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed')
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ORIGINAL))
import torch
from experiments.forge.api import task_formulation_context
from experiments.forge.adapters import _models
from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.state import state_digest
from particlegan import ParticleRegularizer, Recipe
from particlegan.gan_loss import GANLoss
from benchmarks.locked_shared import trajectory
from benchmarks.locked_shared.hosts import residual_student
from benchmarks.toy100.problems import sample_real

OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT/'reports/forge/bcap-tier2-search'
RECEIPTS = Path('/home/martyn/dev/ParticleGAN-bcap-tier2-search/reports/forge/attempts')
spec = importlib.util.spec_from_file_location('saved_helpers', EVIDENCE/'probe_failure_states.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)


def tangent(output, parameters, direction):
    cotangent = torch.zeros_like(output, requires_grad=True)
    pull = torch.autograd.grad(output, parameters, cotangent, create_graph=True,
                               retain_graph=True, allow_unused=True)
    product = sum((g*v).sum() for g,v in zip(pull,direction) if g is not None)
    return torch.autograd.grad(product, cotangent, retain_graph=True)[0].detach()


def measures(force, movement):
    f, m = force.detach().flatten(1).double(), movement.detach().flatten(1).double()
    active = f.norm(dim=1)>1e-14
    derivatives = (f*m).sum(1)
    return dict(rms=float(m.square().mean().sqrt()),
        cosine_with_direct_descent=h.cosine(-f,m),
        fraction_active_rows_uphill=float((derivatives[active]>0).double().mean()) if active.any() else None,
        loss_first_order_change=float(derivatives.sum()),
        output_proxy_scale=min(1., max(0., -float(derivatives.sum())/max(float(m.square().mean()),1e-30))))


def trace(loss, output, parameters, optimizers, forward, losses=None):
    old = [p.detach().clone() for p in parameters]
    opt_saved = [deepcopy(o.state_dict()) for o in optimizers]
    gradient = h.grads(loss, parameters)
    force = torch.autograd.grad(loss,output,retain_graph=True)[0].detach()
    # The first shared-kernel comparison uses raw Euclidean parameter descent;
    # its arbitrary scale is immaterial for angles/uphill row fractions.
    raw = tuple(-g for g in gradient)
    raw_motion = tangent(output,parameters,raw)
    # Disposable optimizer clones preserve actual rounded tensors and history.
    clones=[]; directions=[]
    for opt in optimizers:
        copied=deepcopy(opt)
        original_parameters=[p for group in opt.param_groups for p in group['params']]
        copied_parameters=[p for group in copied.param_groups for p in group['params']]
        mapping={id(p):g for p,g in zip(parameters,gradient)}
        for p,q in zip(original_parameters,copied_parameters):q.grad=mapping[id(p)].clone()
        type(opt).step(copied)
        clones.extend(copied_parameters)
        directions.extend((q.detach()-p.detach()).clone() for p,q in zip(original_parameters,copied_parameters))
    linear=tangent(output,parameters,directions)
    roles={id(p):group['role'] for opt in optimizers for group in opt.param_groups for p in group['params']}
    network=[v if roles[id(p)]!='prior' else torch.zeros_like(v) for p,v in zip(parameters,directions)]
    prior=[v-a for v,a in zip(directions,network)]
    network_linear=tangent(output,parameters,network)
    prior_linear=tangent(output,parameters,prior)
    with torch.no_grad():
        before=output.detach().clone()
        for p,a,delta in zip(parameters,old,directions):p.copy_(a+delta)
        finite=forward().detach()-before
        for p,a in zip(parameters,old):p.copy_(a)
    result=dict(loss=float(loss.detach()),direct_force_norm=float(force.norm()),
        parameter_gradient_norm=float(h.flatten(gradient).norm()),
        direct=measures(force,-force),shared_raw_gradient=measures(force,raw_motion),
        actual_dualnorm_linear=measures(force,linear),actual_dualnorm_finite=measures(force,finite),
        nonlinear_relative_error=float((finite-linear).double().norm()/linear.double().norm().clamp_min(1e-30)))
    result['network_linear']=measures(force,network_linear)
    result['prior_linear']=measures(force,prior_linear)
    result['joint_linearity_residual']=float((linear-network_linear-prior_linear).norm()/linear.norm().clamp_min(1e-30))
    if losses:
        # Rebuild graph after reversible finite motion changed tensor versions.
        rebuilt=forward(); parts=losses(rebuilt)
        proposed=[]
        for part in parts.values():
            gs=h.grads(part,parameters); proposals=[]
            mapping={id(p):g for p,g in zip(parameters,gs)}
            for opt in optimizers:
                copied=deepcopy(opt)
                for group,cgroup in zip(opt.param_groups,copied.param_groups):
                    for p,q in zip(group['params'],cgroup['params']):q.grad=mapping[id(p)].clone()
                type(opt).step(copied)
                proposals.extend((q.detach()-p.detach()).clone() for group,cgroup in zip(opt.param_groups,copied.param_groups)
                                 for p,q in zip(group['params'],cgroup['params']))
            proposed.append(h.flatten(proposals))
        total=h.flatten(directions)
        result['individually_normalized_nonadditivity']=float((sum(proposed)-total).norm()/total.norm())
    assert all(torch.equal(p,a) for p,a in zip(parameters,old))
    assert all(state_digest(o.state_dict())==state_digest(s) for o,s in zip(optimizers,opt_saved))
    return result


def conditional(item):
    path=Path(item['path']); assert file_hash(path)==item['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False); digest=state_digest(saved)
    task=item['task']; g,d=h.restore(saved['models'],task); g.float();d.float()
    z=torch.nn.Parameter(saved['models']['prior0']['z'].clone())
    recipe=Recipe(**saved['applied']['recipe'])
    network=recipe.make_generator_optimizer(g.parameters())
    prior=recipe.make_generator_optimizer([dict(params=[z],lr=recipe.lr*recipe.prior_lr_mult,
        betas=recipe.prior_betas or recipe.betas,forge_role='prior')],latent_table=z)
    optimizers=[network,prior]
    for o,s in zip(optimizers,saved['optimizers']['generator']):o.load_state_dict(s)
    prior.set_sampled_rows(z,torch.arange(len(z)))
    slow,target=trajectory.trajectories(); gan=GANLoss('non_saturating')
    def forward():return g(slow,z)
    def parts(fake):
        terms=dict(adversarial=gan.g_loss(d(slow,fake)),
            coverage=trajectory.PROTOCOL['cover_weight']*trajectory._cover(fake,target),
            latent_l2=trajectory.PROTOCOL['particle_l2']*z.square().mean(),
            latent_spread=ParticleRegularizer(weight=trajectory.PROTOCOL['vicreg_weight'])(z))
        if task=='residual_student':
            mask=residual_student.both_land_mask(slow,target,torch.arange(len(slow)))
            terms['paired_residual']=residual_student.RESIDUAL_WEIGHT*(fake[mask]-target[mask]).square().mean()
        return terms
    fake=forward(); losses=parts(fake)
    result=trace(sum(losses.values()),fake,list(g.parameters())+[z],optimizers,forward,parts)
    assert state_digest(saved)==digest and file_hash(path)==item['sha256']
    return dict(**item,cohort='fixed_saved_critic_endpoint',diagnostics=result,
                checkpoint_and_optimizer_unchanged=True,committed_updates=0)


def native():
    aid='db79021404c14c91aa5f38f9f1be40fb'; base=RECEIPTS/aid
    envelope,result,certificate=[read_json(base/(n+'.json')) for n in ('request','result','evidence')]
    for name,expected in certificate['source']['files'].items():
        if name.endswith('.py'):assert file_hash(ORIGINAL/name)==expected,name
    request=envelope['request']; task=request['tasks']['grid100']; row=result['task_results'][0]
    descriptor=row['evidence']['checkpoint']; path=Path(row['evidence']['artifact_root'])/descriptor['path']
    assert file_hash(path)==descriptor['sha256']; saved=torch.load(path,map_location='cuda:0',weights_only=True)
    context=task_formulation_context(request['candidate'],task,request['protocol'],device='cuda:0',root=ORIGINAL)
    g,d=_models(context,task['execution']['model']); trainer=context.build_trainer(g,d,max_steps=7000)
    context.load_state_dict(saved); before=state_digest(context.state_dict()); captured={}
    latent_draw=trainer._sample_training_prior
    def sample(n):
        latent,ids=latent_draw(n);captured['ids']=ids;captured['offset']=latent.detach()-trainer.prior.z[ids].detach()
        return latent,ids
    trainer._sample_training_prior=sample
    def hook(module,args,output):
        if output.requires_grad:captured['fake']=output
    handle=g.register_forward_hook(hook)
    original_backward=torch.Tensor.backward
    def retaining(self,*args,**kwargs):
        kwargs['retain_graph']=True;return original_backward(self,*args,**kwargs)
    original_step=trainer.opt_g.step
    def step(*args,**kwargs):
        def forward():return g(trainer.prior.z[captured['ids']]+captured['offset'])
        fake=captured['fake'];loss=trainer.loss.g_loss(d(fake))
        parameters=[p for group in trainer.opt_g.param_groups for p in group['params']]
        captured['diagnostics']=trace(loss,fake,parameters,[trainer.opt_g],forward)
    trainer.opt_g.step=step;torch.Tensor.backward=retaining
    try:
        trainer.extend_execution(7001)
        data=context.streams.generator('data',component='target',purpose='training')
        trainer.step(sample_real('grid100',context.recipe.batch_size,device='cuda:0',generator=data))
    finally:
        torch.Tensor.backward=original_backward;handle.remove();trainer.opt_g.step=original_step
        trainer._sample_training_prior=latent_draw;trainer.max_steps=7000;context.load_state_dict(saved)
    assert state_digest(context.state_dict())==before and file_hash(path)==descriptor['sha256']
    return dict(task='grid100',cohort='original_source_actual_next_post_D_proposal',path=str(path),
        sha256=descriptor['sha256'],source_digest=certificate['source']['digest'],source_commit=certificate['source']['origin_commit'],
        diagnostics=captured['diagnostics'],all_params_optimizer_buffers_named_rng_restored=True,
        hypothetical_discarded_updates=1,committed_updates=0)


def vector(item):
    path=Path(item['path']);assert file_hash(path)==item['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False);digest=state_digest(saved)
    recipe=Recipe(**saved['recipe']);g,d=h.restore(saved['trainer']['models']);g.float();d.float()
    z=torch.nn.Parameter(saved['trainer']['models']['prior']['z'].clone())
    from particlegan.optim.dualnorm import NormalizedOptimizer
    recorded=saved['trainer']['optimizers'][0]
    groups=[];position=0
    for group in recorded['param_groups']:
        params=[z] if group['role']=='prior' else list(g.parameters())
        groups.append({**group,'params':params})
    opt=NormalizedOptimizer(groups,family='dualnorm',momentum=0,smoothing=.001,convolution='per_offset')
    opt.load_state_dict(recorded)
    rows=torch.arange(min(128,len(z)));sigma=float(saved['trainer']['models']['prior']['sigma'])
    offsets=torch.cat((torch.eye(z.shape[1]),-torch.eye(z.shape[1])))*sigma*(z.shape[1]**.5)
    ids=rows[:,None].expand(-1,len(offsets)).flatten();jitter=offsets[None].expand(len(rows),-1,-1).flatten(0,1)
    opt.set_sampled_rows(z,ids)
    def forward():return g(z[ids]+jitter)
    fake=forward();loss=GANLoss('non_saturating').g_loss(d(fake))
    parameters=[p for group in opt.param_groups for p in group['params']]
    result=trace(loss,fake,parameters,[opt],forward)
    assert state_digest(saved)==digest and file_hash(path)==item['sha256']
    return dict(**item,cohort='fixed_saved_critic_deterministic_kernel_cubature',
        diagnostic_rows=len(rows),cubature_points=len(ids),diagnostics=result,
        all_params_optimizer_buffers_named_rng_restored=True,committed_updates=0)


def main():
    started=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(0).clone()
    original=read_json(EVIDENCE/'failure-state-analysis.json')
    entries=[next(x for x in original['artifact_proofs'] if x['task']==task and x['path'].endswith('provenance-state.pt'))
             for task in ('trajectory','residual_student')]
    rows=[conditional(item) for item in entries];rows.append(native())
    item=next(x for x in original['artifact_proofs'] if x['task']=='vector_unequal_width' and x['path'].endswith('provenance-state.pt'))
    rows.append(vector(item))
    torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda,0)
    atomic_json(OUT/'saved-force-traces.json',dict(schema_version=1,qualification_input=False,
        scope='source-bound endpoint/fixed-critic and discarded public next-update cohorts',
        source_digest=original['source_digest'],rows=rows,committed_training_updates=0,
        global_rng_restored=True,elapsed_seconds=time.monotonic()-started,allowance_seconds=900))
    for row in rows:print(row['task'],row['diagnostics'],flush=True)

if __name__=='__main__':main()
