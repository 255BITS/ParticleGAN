"""One saved-state empirical critic-response secant, with one endpoint fit.

This compares partial G fields, not a total Stackelberg gradient. Both
bounded critic fits start from the same captured accepted-D* initialization.
The base result is the already evaluated best finite critic. At the exact
accepted G/prior endpoint, only the fake values in the same frozen paired
bank change. One identical40/80 solver attempt supplies the endpoint critic.
No outer training, controller update, new seed or gain selection occurs.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from reports.toy100 import pr84_critic_relaxation as fit
from reports.toy100.pr84_critic_refinement_finite import finite_trial_fit
from reports.toy100.pr84_critic_refinement_capture import _sha
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe


CAPTURE_SHA='19ceb38a92ba95ffaccd3aafbda55bbd77917765612666d923e986e5523d6965'
STEP=472


def parameters(saved):
    g,_,z=fit.modules(saved)
    return [p.detach().clone() for p in list(g.parameters())+list(z.parameters())]


def field(saved,critic,batch,gan):
    g,_,z=fit.modules(saved)
    fake=g(z.z[batch['indices']])+batch['sigma']*batch['noise']
    loss=gan.g_loss(fit.smooth(critic,fake),fit.smooth(critic,batch['real']))
    values=torch.autograd.grad(loss,list(g.parameters())+list(z.parameters()))
    return [value.detach().clone() for value in values],float(loss.detach())


def flat(values):
    return torch.cat([value.detach().double().flatten() for value in values])


def secant(base,end,delta,metric):
    g0,g1,s,p=map(flat,(base,end,delta,metric))
    denominator=(s.square()/p).sum()
    change=g1-g0
    return dict(metric_displacement_norm=float(denominator.sqrt()),
                own_or_profiled_norm_ratio=float(((p*change.square()).sum()/denominator).sqrt()),
                directional_curvature=float((s@change)/denominator),
                initial_partial_work=float(g0@s),endpoint_partial_work=float(g1@s),
                field_cosine=float((g0@g1)/(g0.norm()*g1.norm())),
                raw_gradient_norm=float(g1.norm()),
                metric_gradient_norm=float((p*g1.square()).sum().sqrt()))


def d_receipt(critic,bank,gan,penalty,metric):
    loss,logistic,regularizer=fit.d_loss(critic,bank,gan,penalty,STEP)
    values=fit.gradients(loss,critic)
    return dict(total_loss=float(loss.detach()),logistic=float(logistic.detach()),
                penalty=float(regularizer.detach()),gradient=fit.norm_receipt(values,metric),
                convergence_claim=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--recovery',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if hashlib.sha256(args.capture.read_bytes()).hexdigest()!=CAPTURE_SHA:
        raise RuntimeError('wrong saved472 failure capture')
    reference=json.loads((args.recovery/'continuation.json').read_text())
    config=json.loads((args.recovery/'config.json').read_text())
    recipe,_,_=declared_recipe(config)
    gan,penalty=recipe.make_loss(),recipe.make_gradient_penalty()
    names=('reports/toy100/pr84_profiled_field472.py','reports/toy100/pr84_critic_relaxation.py',
           'reports/toy100/pr84_critic_refinement_finite.py','reports/toy100/alternating_curvature_scratch.py',
           'reports/toy100/coverage_fixed_eval.py','benchmarks/locked_shared/mode_hold.py',
           'particlegan/gan_loss.py','particlegan/grad_regularizers.py')
    declaration=dict(method='one_empirical_profiled_partial_G_secant',host_update=STEP,
        capture_sha256=CAPTURE_SHA,source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
        endpoint='exact accepted G/prior Adam proposal, existing curvature factor',
        endpoint_fit_initialization='same original accepted-D* as the baseline fit',
        endpoint_fit_budget=dict(iterations=40,attempted_closures=80,attempts=1),
        cached_draws='same1024 real samples/latent indices/output noise; regenerate fake values only',
        metric='post-base G Adam metric, frozen for all secants',stencil_width=.15,
        quality_selection=False,outer_training=False,shared_gate_eligible=False,
        caveat='algorithmic empirical response; neither fit is certified stationary; not a total derivative')
    args.output.mkdir(parents=True,exist_ok=False)
    for name in names:
        path=args.output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',**declaration)),flush=True)
    torch.set_num_threads(1)
    payload=torch.load(args.capture,weights_only=True)
    before=_sha(payload);rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        pre,saved=payload['pre_step'],payload['post_accepted_d']
        if saved['noise']['input_sigma']!=0 or saved['rng']['output'] is not None:
            raise ValueError('diagnosis scoped to captured zero-input global-output-noise state')
        generator,d_start,prior=fit.modules(saved)
        a_rows,b_rows,a_bank,b_bank,g_batch=fit.banks(pre,saved,generator,prior)
        if any(not torch.equal(a_bank[key],payload['bank'][key]) for key in ('real','fake')):
            raise RuntimeError('frozen paired D bank does not reproduce capture')
        best=fit.modules(saved)[1];best.load_state_dict(payload['best'])
        proposal,accepted,unbounded=fit.g_proposal(saved,best,g_batch,gan,STEP)
        old_record=reference['dynamics']['records'][0]
        old_motion=reference['replay']['accepted_movement'][0]
        actual_grade=reference['quality_observations'][0]
        parity=dict(rho=proposal['rho']==old_record['g']['rho'],
                    factor=proposal['factor']==old_record['g']['factor'],
                    width=old_record['critic_refinement']['frozen_g_stencil_width']==fit.WIDTH==.15,
                    clean_rms=proposal['accepted_joint']['rms']==old_motion['clean_output_rms'],
                    hq=proposal['grade_after']['hq']==actual_grade['hq'],
                    modes=proposal['grade_after']['modes']==actual_grade['modes'])
        if not all(parity.values()):
            raise RuntimeError(f'actual accepted G endpoint parity failed: {parity}')
        endpoint=deepcopy(saved)
        endpoint.update(generator=accepted['generator'],prior=accepted['prior'],optimizer_g=accepted['optimizer'])
        unbounded_saved=deepcopy(saved)
        unbounded_saved.update(generator=unbounded['generator'],prior=unbounded['prior'],optimizer_g=unbounded['optimizer'])
        end_g,_,end_z=fit.modules(endpoint)
        sigma=saved['noise']['output_sigma']
        def regenerate(rows):
            with torch.no_grad():
                return dict(real=torch.cat([row['real'] for row in rows]),
                    fake=torch.cat([end_g(end_z.z[row['indices']])+sigma*row['noise'] for row in rows]))
        end_bank,end_heldout=regenerate(a_rows),regenerate(b_rows)
        metric=fit.saved_metric(accepted['optimizer'])
        base_params,end_params,raw_params=map(parameters,(saved,endpoint,unbounded_saved))
        delta=[b-a for a,b in zip(base_params,end_params)]
        delta_raw=[b-a for a,b in zip(base_params,raw_params)]
        g0,l0=field(saved,best,g_batch,gan)
        g_raw,lraw=field(unbounded_saved,best,g_batch,gan)
        g_own,lown=field(endpoint,best,g_batch,gan)
        base_residual=d_receipt(best,a_bank,gan,penalty,payload['metric'])
        endpoint_before=d_receipt(best,end_bank,gan,penalty,payload['metric'])
        endpoint_fit=finite_trial_fit(d_start,end_bank,gan,penalty,STEP,payload['metric'])
        endpoint_after=d_receipt(d_start,end_bank,gan,penalty,payload['metric'])
        g_profile,lprofile=field(endpoint,d_start,g_batch,gan)
        own=secant(g0,g_own,delta,metric)
        profiled=secant(g0,g_profile,delta,metric)
        raw=secant(g0,g_raw,delta_raw,metric)
        for value in (own,profiled):
            value['factor_times_norm_ratio']=proposal['factor']*value['own_or_profiled_norm_ratio']
            value['factor_times_directional_curvature']=proposal['factor']*value['directional_curvature']
        result=dict(declaration=declaration,endpoint_parity=parity,
            base_critic_sha256=_sha(payload['best']),endpoint_critic_sha256=_sha(d_start.state_dict()),
            base_generator_sha256=_sha(saved['generator']),endpoint_generator_sha256=_sha(accepted['generator']),
            unchanged_real_bank=torch.equal(a_bank['real'],end_bank['real']),
            current_proposal=proposal,unbounded_own=raw,accepted_own=own,accepted_profiled=profiled,
            losses=dict(base=l0,unbounded_own=lraw,accepted_own=lown,accepted_profiled=lprofile),
            d_base=base_residual,d_endpoint_before=endpoint_before,d_endpoint_after=endpoint_after,
            d_heldout_endpoint_before=d_receipt(best,end_heldout,gan,penalty,payload['metric']),
            d_heldout_endpoint_after=d_receipt(d_start,end_heldout,gan,penalty,payload['metric']),
            endpoint_fit=endpoint_fit,shared_gate_eligible=False)
        torch.save(dict(accepted=accepted,unbounded=unbounded,g_batch=g_batch,
            base_bank=a_bank,endpoint_bank=end_bank,endpoint_heldout=end_heldout,
            base_critic=payload['best'],endpoint_critic=d_start.state_dict(),
            base_gradient=g0,own_endpoint_gradient=g_own,profiled_endpoint_gradient=g_profile,
            metric=metric,delta=delta),args.output/'tensors.pt')
    if before!=_sha(payload) or not torch.equal(rng,torch.get_rng_state()):
        raise RuntimeError('read-only diagnostic changed saved input or global RNG')
    result.update(input_state_unchanged=True,global_rng_unchanged=True)
    (args.output/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps(dict(event='PROFILED_SECANT_DONE',parity=parity,
        original_factor=proposal['factor'],unbounded_own=raw,accepted_own=own,
        accepted_profiled=profiled,endpoint_fit_calls=endpoint_fit['closure_calls'],
        base_d_residual=base_residual['gradient'],endpoint_d_residual=endpoint_after['gradient']),allow_nan=False),flush=True)


if __name__=='__main__':
    main()
