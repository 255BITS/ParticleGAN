"""Read-only one-virtual-D-step G total-gradient diagnostic.

The virtual critic uses the exact saved post-Adam diagonal metric without
another moment update. Full-chain and detached-critic controls evaluate the
identical virtual endpoint. The D step uses one exact native D minibatch;
the outer G objective uses the actual following G minibatch and a frozen
0.15 spatial stencil. This is a finite surrogate, not a best-response or a
local convergence claim. No original training source is changed.
"""

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.func import functional_call, jvp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from reports.toy100 import pr84_critic_relaxation as fit
from reports.toy100.alternating_curvature_scratch import _metric,_rho
from reports.toy100.functional_b_cap import functional_b_cap
from reports.toy100.pr84_critic_refinement_capture import _sha
from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100.coverage_fixed_eval import fixed_draw,score_support


def flat(values):
    return torch.cat([value.detach().double().flatten() for value in values])


def virtual_parameters(parameters,metric,loss):
    gradient=torch.autograd.grad(loss,tuple(parameters.values()),create_graph=True)
    # Fixed saved metric. Round the virtual correction to the host dtype;
    # no new Adam denominator or moment step is computed or differentiated.
    point={name:p-(m.detach()*g.double()).to(p.dtype)
           for (name,p),m,g in zip(parameters.items(),metric,gradient)}
    return point,gradient


def surrogate(generator,prior,critic,metric,d_batch,g_batch,gan,regularizer,step,*,full_chain):
    params=dict(critic.named_parameters())
    def sharp(x):
        return functional_call(critic,params,(x,))
    fake_d=generator(prior.z[d_batch['indices']])+d_batch['sigma']*d_batch['noise']
    inner=gan.d_loss(sharp(d_batch['real']),sharp(fake_d))
    inner=inner+functional_b_cap(regularizer,sharp,d_batch['real'],fake_d,step)
    virtual,gradient=virtual_parameters(params,metric,inner)
    used=virtual if full_chain else {name:p.detach() for name,p in virtual.items()}
    def opponent(x):
        return functional_call(critic,used,(x,))
    fake_g=generator(prior.z[g_batch['indices']])+g_batch['sigma']*g_batch['noise']
    outer=gan.g_loss(fit.smooth(opponent,fake_g),fit.smooth(opponent,g_batch['real']))
    return outer,dict(inner_loss=float(inner.detach()),
        virtual_parameters=[p.detach().clone() for p in virtual.values()],
        inner_gradient=[p.detach().clone() for p in gradient])


def gradient_at(generator,prior,critic,metric,d_batch,g_batch,gan,regularizer,step,full_chain):
    loss,record=surrogate(generator,prior,critic,metric,d_batch,g_batch,gan,regularizer,step,
                           full_chain=full_chain)
    values=torch.autograd.grad(loss,list(generator.parameters())+list(prior.parameters()))
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError('nonfinite unrolled G field')
    record['loss']=float(loss.detach())
    return [value.detach().clone() for value in values],record


def cosine(a,b):
    a,b=flat(a),flat(b)
    return float((a@b)/(a.norm()*b.norm())) if a.norm()>0 and b.norm()>0 else None


def functional_direction(generator,prior,gradient):
    network=tuple(generator.parameters())
    names=list(dict(generator.named_parameters()))
    def clean(values,z):
        return functional_call(generator,dict(zip(names,values)),(z,))
    _,direction=jvp(clean,(network,prior.z),
        (tuple(-value for value in gradient[:-1]),-gradient[-1]))
    return direction.detach()


def finite_difference(generator,prior,critic,metric,d_batch,g_batch,gan,regularizer,step,gradient):
    params=list(generator.parameters())+list(prior.parameters())
    base=[p.detach().clone() for p in params]
    norm=float(flat(gradient).norm())
    if norm==0:
        return dict(status='EXACT_ZERO_FIELD',rows=[])
    direction=[-g/norm for g in gradient]
    analytic=float(flat(gradient)@flat(direction))
    h0=2e-4*(1+float(flat(base).norm()))
    rows=[]
    try:
        for h in (h0,h0/2):
            losses=[];achieved=[]
            for sign in (-1,1):
                with torch.no_grad():
                    for p,b,u in zip(params,base,direction):p.copy_(b+sign*h*u)
                achieved.append(float(flat([p.detach()-b for p,b in zip(params,base)]).norm()))
                loss,_=surrogate(generator,prior,critic,metric,d_batch,g_batch,gan,regularizer,step,
                                  full_chain=True)
                losses.append(float(loss.detach()))
            estimate=(losses[1]-losses[0])/(2*h)
            absolute=abs(estimate-analytic)
            rows.append(dict(h=h,actual_parameter_norms=achieved,analytic=analytic,
                central=estimate,absolute_error=absolute,relative_error=absolute/max(abs(analytic),1e-30),
                passed=absolute<=max(1e-5,.02*abs(analytic))))
    finally:
        with torch.no_grad():
            for p,b in zip(params,base):p.copy_(b)
    return dict(status='PASS_BOTH_SCALES' if all(row['passed'] for row in rows)
                else 'MISMATCH_OR_NONSMOOTH_CROSSING',rows=rows,
                criterion='both predeclared scales;abs error <= max(1e-5,2% derivative);no adaptive epsilon search')


def proposal(saved,critic,metric,d_batch,g_batch,gan,regularizer,step,full_chain):
    generator,_,prior=fit.modules(saved)
    optimizer=fit.g_optimizer(generator,prior,saved['optimizer_g'])
    params=list(generator.parameters())+list(prior.parameters())
    base=[p.detach().clone() for p in params]
    clean0=generator(prior.z).detach().clone()
    gradient,record=gradient_at(generator,prior,critic,metric,d_batch,g_batch,gan,regularizer,step,full_chain)
    raw_direction=functional_direction(generator,prior,gradient)
    fd=(finite_difference(generator,prior,critic,metric,d_batch,g_batch,gan,regularizer,step,gradient)
        if full_chain else None)
    optimizer.zero_grad()
    for p,g in zip(params,gradient):p.grad=g.clone()
    optimizer.step()
    proposed=[p.detach().clone() for p in params]
    metric_g=_metric(optimizer)
    second,_=gradient_at(generator,prior,critic,metric,d_batch,g_batch,gan,regularizer,step,full_chain)
    rho=_rho(base,proposed,gradient,second,metric_g)
    factor=min(1.,.25/rho) if rho else 1.
    with torch.no_grad():
        for p,b,n in zip(params,base,proposed):p.copy_(torch.lerp(b,n,factor) if factor<1 else n)
        clean1=generator(prior.z).detach().clone()
    third,_=gradient_at(generator,prior,critic,metric,d_batch,g_batch,gan,regularizer,step,full_chain)
    actual_rho=_rho(base,[p.detach() for p in params],gradient,third,metric_g)
    means=mode_hold.ring_means()
    distances=torch.cdist(clean0,means)
    nearest=distances.argmin(1)
    represented=set(nearest[distances.min(1).values<=.21].tolist())
    missing=[i for i in range(len(means)) if i not in represented]
    radial=clean0-means[nearest]
    movement=clean1-clean0
    def direction_row(value):
        result=dict(vectors=value.tolist(),rms=float(value.square().sum(-1).mean().sqrt()),
            inward_particles=int(((value*radial).sum(-1)<0).sum()),
            radial_work=float((value*radial).sum(-1).mean()))
        if missing:
            target=means[missing][torch.cdist(clean0,means[missing]).argmin(1)]-clean0
            result.update(toward_missing_particles=int(((value*target).sum(-1)>0).sum()),
                          toward_missing_work=float((value*target).sum(-1).mean()))
        return result
    indices,noise=fixed_draw(step,clean1)
    return dict(gradient=gradient,virtual_parameters=record.pop('virtual_parameters'),
        inner_gradient=record.pop('inner_gradient'),record=dict(**record,
            raw_gradient_norm=float(flat(gradient).norm()),rho=rho,factor=factor,
            effective_accepted_rho=factor*actual_rho,raw_direction=direction_row(raw_direction),
            accepted_direction=direction_row(movement),finite_difference=fd,
            posthoc_grade_before=score_support(clean0,indices,noise,means),
            posthoc_grade_after=score_support(clean1,indices,noise,means),
            cloned_g_adam_moment_steps=[float(optimizer.state[p]['step']) for p in params]),
        accepted=dict(generator=deepcopy(generator.state_dict()),prior=deepcopy(prior.state_dict()),
                      optimizer=deepcopy(optimizer.state_dict())))


def evaluate_state(pre,saved,critic_state,step,recipe):
    generator,critic,prior=fit.modules(saved)
    critic.load_state_dict(critic_state)
    gan,regularizer=recipe.make_loss(),recipe.make_gradient_penalty()
    rows,_,bank,_,g_batch=fit.banks(pre,saved,generator,prior)
    d_batch=dict(**rows[0],sigma=saved['noise']['output_sigma'])
    d_pre=fit.modules(pre)[1]
    actual_first=fit.gradients(fit.d_loss(d_pre,rows[0],gan,regularizer,step)[0],d_pre)
    expected=[saved['optimizer_d']['state'][idx]['exp_avg']
              for group in saved['optimizer_d']['param_groups'] for idx in group['params']]
    if not all(torch.equal(a,b) for a,b in zip(actual_first,expected)):
        raise RuntimeError('native D first gradient differs from actual Adam moment')
    metric=fit.saved_metric(saved['optimizer_d'])
    critic_before=_sha(critic.state_dict())
    baseline=fit.g_proposal(saved,critic,g_batch,gan,step)[0]
    native_inner=fit.d_loss(critic,rows[0],gan,regularizer,step)[0]
    native_inner_gradient=fit.gradients(native_inner,critic)
    detached=proposal(saved,critic,metric,d_batch,g_batch,gan,regularizer,step,False)
    full=proposal(saved,critic,metric,d_batch,g_batch,gan,regularizer,step,True)
    if (_sha(detached['virtual_parameters'])!=_sha(full['virtual_parameters'])
            or detached['record']['loss']!=full['record']['loss']
            or critic_before!=_sha(critic.state_dict())):
        raise RuntimeError('virtual-endpoint controls or persistent critic changed')
    if (float(native_inner.detach())!=full['record']['inner_loss']
            or not all(torch.equal(a,b) for a,b in zip(native_inner_gradient,full['inner_gradient']))):
        raise RuntimeError('graph-preserving inner loss/gradient differs from the native sharp D objective')
    chain=[a-b for a,b in zip(full['gradient'],detached['gradient'])]
    return dict(step=step,first_native_d_gradient_exact=True,
        functional_inner_value_and_d_gradient_exact=True,
        detached_and_full_virtual_endpoint_exact=True,persistent_critic_unchanged=True,
        field_cosine=cosine(full['gradient'],detached['gradient']),
        chain_norm=float(flat(chain).norm()),
        chain_to_partial_norm=float(flat(chain).norm()/flat(detached['gradient']).norm()),
        baseline=baseline,detached=detached['record'],full_chain=full['record']),dict(
            saved=saved,critic=critic_state,metric=metric,d_batch=d_batch,g_batch=g_batch,
            detached=detached,full=full,chain=chain)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--failure',type=Path,required=True)
    parser.add_argument('--states',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    names=('reports/toy100/pr84_one_step_unroll.py','reports/toy100/functional_b_cap.py',
        'reports/toy100/pr84_critic_relaxation.py','reports/toy100/alternating_curvature_scratch.py',
        'reports/toy100/coverage_fixed_eval.py','particlegan/gan_loss.py','particlegan/grad_regularizers.py')
    declaration=dict(scope='two saved states,onevirtualDstep,no outer training or fits',
        starting_critics={'472':'captured bestfiniteD','1325':'original acceptedD*'},
        virtual_metric='saved post-Adam D metric;fixed,no differentiated or persistentmoment update',
        inner_batch='actual nativeD128pair;full sharpRp plus value-identical graph-preserving cap',
        outer_batch='actual nextnativeG128pair;frozen.15spatialstencil',
        controls=['original fixedD','detached virtualD','fullchain samevirtualD'],
        fd='negativefullgradient direction;h=2e-4*(1+parameterL2),h/2;both2%or1e-5 criterion',
        source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.failure,args.states,args.config)},
        shared_gate_eligible=False,quality_selection=False,best_response_claim=False)
    args.output.mkdir(parents=True,exist_ok=False)
    for name in names:
        out=args.output/'source'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((ROOT/name).read_bytes())
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',**declaration)),flush=True)
    payload=torch.load(args.failure,weights_only=True)
    with gzip.open(args.states,'rb') as stream:states=torch.load(stream,weights_only=True)
    config=json.loads(args.config.read_text());recipe,_,_=declared_recipe(config)
    if recipe.prior_reg!=0:raise ValueError('diagnostic scope requires original zero prior regularizer')
    before=_sha(dict(payload=payload,states=states));rng=torch.get_rng_state().clone()
    rows=[];tensors={}
    with torch.random.fork_rng(devices=[]):
        cases=[(472,payload['pre_step'],payload['post_accepted_d'],payload['best']),
               (1325,states[1325]['pre_step'],states[1325]['post_accepted_d'],
                fit.unwrapped(states[1325]['post_accepted_d']['critic']))]
        for step,pre,saved,critic in cases:
            row,tensor=evaluate_state(pre,saved,critic,step,recipe)
            rows.append(row);tensors[step]=tensor
            print(json.dumps(dict(event='STATE_DONE',**row),allow_nan=False),flush=True)
    if before!=_sha(dict(payload=payload,states=states)) or not torch.equal(rng,torch.get_rng_state()):
        raise RuntimeError('saved input or global RNG mutated')
    result=dict(declaration=declaration,rows=rows,input_state_unchanged=True,global_rng_unchanged=True,
                status='DIAGNOSTIC_COMPLETE',shared_gate_eligible=False)
    torch.save(tensors,args.output/'tensors.pt')
    (args.output/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n')


if __name__=='__main__':main()
