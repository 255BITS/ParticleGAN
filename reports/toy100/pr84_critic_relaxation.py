"""Bounded, fixed-generator penalized-critic local-fit diagnosis.

Three captured states, one 1024-pair fit bank and separate 1024-pair heldout
bank, one L-BFGS attempt per state, at most 40 iterations/80 closure calls.
No training candidate, target-controlled update or optimality theorem.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.func import functional_call, jvp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPGenerator, SimpleMLPDiscriminator
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from particlegan import ParticlePrior
from reports.toy100.alternating_curvature_scratch import _metric, _rho
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.pr84_prediction_state_filter import state_hash


STATES = (1325, 1530, 1539)
STATE_SHA = '37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47'
WIDTH = .15
MAX_ITER = 40
MAX_CLOSURES = 80


def unwrapped(values):
    return {name.removeprefix('model.'): value for name, value in values.items()}


def modules(saved):
    generator = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2)
    critic = SimpleMLPDiscriminator(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER)
    prior = ParticlePrior(*saved['prior']['z'].shape)
    generator.load_state_dict(unwrapped(saved['generator']))
    critic.load_state_dict(unwrapped(saved['critic']))
    prior.load_state_dict(saved['prior'])
    return generator, critic, prior


def g_optimizer(generator, prior, saved):
    groups = []
    for original, params in zip(saved['param_groups'], (list(generator.parameters()), list(prior.parameters()))):
        groups.append({**{key: deepcopy(value) for key, value in original.items() if key != 'params'},
                       'params': params})
    optimizer = torch.optim.Adam(groups)
    optimizer.load_state_dict(deepcopy(saved))
    return optimizer


def saved_metric(saved):
    result = []
    for group in saved['param_groups']:
        for index in group['params']:
            state = saved['state'][index]
            denominator = (state['exp_avg_sq'] / (1-group['betas'][1]**float(state['step']))).sqrt()+group['eps']
            result.append(group['lr']/denominator.double())
    return result


def banks(saved_pre, saved_accepted, generator, prior):
    if saved_pre['noise']['input_sigma'] != 0 or saved_pre['rng']['output'] is not None:
        raise RuntimeError('diagnosis requires the captured late global-output-noise host')
    stream = torch.Generator()
    stream.set_state(saved_pre['rng']['data'])
    rows = []
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.set_rng_state(saved_pre['rng']['torch'])
        for _ in range(16):
            real = mode_hold.sample_ring(mode_hold.ring_means(), mode_hold.BATCH, mode_hold.SIGMA, stream)
            latent, indices = prior.sample(mode_hold.BATCH, generator=stream)
            clean = generator(latent)
            noise = torch.randn_like(clean)
            fake = clean + saved_pre['noise']['output_sigma']*noise
            rows.append(dict(real=real, fake=fake, indices=indices, noise=noise))
    stream.set_state(saved_accepted['rng']['data'])
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.set_rng_state(saved_accepted['rng']['torch'])
        latent, indices = prior.sample(mode_hold.BATCH, generator=stream)
        clean = generator(latent)
        noise = torch.randn_like(clean)
        real = mode_hold.sample_ring(mode_hold.ring_means(), mode_hold.BATCH, mode_hold.SIGMA, stream)
        g_batch = dict(real=real, indices=indices, noise=noise,
                       sigma=saved_accepted['noise']['output_sigma'])
    def join(values):
        return dict(real=torch.cat([row['real'] for row in values]),
                    fake=torch.cat([row['fake'] for row in values]))
    return rows[:8], rows[8:], join(rows[:8]), join(rows[8:]), g_batch


def d_loss(critic, bank, gan, regularizer, step):
    logistic = gan.d_loss(critic(bank['real']), critic(bank['fake']))
    penalty = regularizer(critic, bank['real'], bank['fake'], step=step)
    return logistic+penalty, logistic, penalty


def gradients(loss, critic):
    values = torch.autograd.grad(loss, tuple(critic.parameters()))
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError('nonfinite fixed-bank critic gradient')
    return [value.detach().clone() for value in values]


def norm_receipt(values, metric):
    return dict(raw_l2=sum(float(x.double().square().sum()) for x in values)**.5,
                raw_linf=max(float(x.abs().max()) for x in values),
                saved_adam_metric_l2=sum(float((p*x.double().square()).sum())
                                         for p,x in zip(metric,values))**.5)


def d_evaluation(critic, bank, batches, gan, regularizer, step, metric):
    loss, logistic, penalty = d_loss(critic, bank, gan, regularizer, step)
    gradient = gradients(loss, critic)
    norms = norm_receipt(gradient, metric)
    activation = {}
    for name in ('real', 'fake'):
        x = bank[name].detach().clone().requires_grad_(True)
        input_gradient = torch.autograd.grad(critic(x).sum(), x)[0]
        slope = (input_gradient.square().sum(1)+1e-12).sqrt()
        activation[name] = dict(fraction_above_cap=float((slope>regularizer.kappa).float().mean()),
                                maximum_slope=float(slope.max()), mean_slope=float(slope.mean()))
    draws = []
    for batch in batches:
        loss_i, _, _ = d_loss(critic, batch, gan, regularizer, step)
        grad_i = gradients(loss_i, critic)
        draws.append(torch.cat([(p.sqrt()*value.double()).flatten() for p,value in zip(metric,grad_i)]))
    stack = torch.stack(draws)
    mean = stack.mean(0)
    variance = float((stack-mean).square().sum(1).sum()/(len(draws)-1))
    signal = max(0., float(mean.square().sum())-variance/len(draws))
    return dict(total_loss=float(loss.detach()), logistic_loss=float(logistic.detach()),
                penalty=float(penalty.detach()), gradient=norms, cap=activation,
                metric_minibatch_coherent_energy=signal/(signal+variance) if signal+variance else None,
                metric_minibatch_signal_energy=signal, metric_minibatch_variance=variance)


def smooth(critic, points):
    values = [critic(points)]
    for dim in range(2):
        offset = torch.zeros_like(points)
        offset[:,dim] = WIDTH
        values.extend((critic(points+offset),critic(points-offset)))
    return torch.stack(values).mean(0)


def g_proposal(saved, critic, batch, gan, step):
    generator, _, prior = modules(saved)
    optimizer = g_optimizer(generator, prior, saved['optimizer_g'])
    params = list(generator.parameters())+list(prior.parameters())
    base = [parameter.detach().clone() for parameter in params]
    base_model = {name: value.detach().clone() for name,value in generator.state_dict().items()}
    base_z = prior.z.detach().clone()
    clean_before = generator(prior.z).detach().clone()

    def loss_at_point():
        fake = generator(prior.z[batch['indices']])+batch['sigma']*batch['noise']
        return gan.g_loss(smooth(critic,fake),smooth(critic,batch['real']))

    optimizer.zero_grad()
    loss = loss_at_point()
    loss.backward()
    first = [parameter.grad.detach().clone() for parameter in params]
    # Exact directional derivative of the clean joint G/prior function, before
    # Adam or its finite proposal. This separates shared-network coupling from
    # the local input gradient and the optimizer's metric.
    names = list(dict(generator.named_parameters()))
    def clean_function(network, z):
        return functional_call(generator,dict(zip(names,network)),(z,))
    _, raw_output_direction = jvp(clean_function,(tuple(base[:-1]),base_z),
                                  (tuple(-value for value in first[:-1]),-first[-1]))
    optimizer.step()
    proposed = [parameter.detach().clone() for parameter in params]
    metric = _metric(optimizer)
    proposal_state = dict(generator=generator.state_dict(), prior=prior.state_dict(),
                          optimizer=deepcopy(optimizer.state_dict()))
    proposal_state = deepcopy(proposal_state)
    next_loss = loss_at_point()
    second = torch.autograd.grad(next_loss,params)
    rho = _rho(base,proposed,first,second,metric)
    factor = min(1.,.25/rho) if rho else 1.
    with torch.no_grad():
        for parameter,old,new in zip(params,base,proposed):
            parameter.copy_(torch.lerp(old,new,factor) if factor<1 else new)
        accepted = generator(prior.z).detach().clone()
        network_only = generator(base_z).detach().clone()
        prior_only = functional_call(generator,base_model,(prior.z,)).detach().clone()
    points = clean_before.detach().clone().requires_grad_(True)
    logits = smooth(critic,points)
    real_logits = smooth(critic,batch['real']).detach()
    input_loss = torch.nn.functional.softplus(real_logits[None,:]-logits[:,None]).mean(1).sum()
    local_direction = -torch.autograd.grad(input_loss,points)[0].detach()
    means = mode_hold.ring_means()
    nearest = torch.cdist(clean_before,means).argmin(1)
    radial = clean_before-means[nearest]
    delta = accepted-clean_before
    indices,noise = fixed_draw(step,accepted)
    def per_row(vector):
        return dict(vectors=vector.tolist(),radial_work=(vector*radial).sum(1).tolist(),
                    rms=float(vector.square().sum(1).mean().sqrt()))
    row = dict(loss=float(loss.detach()),rho=rho,factor=factor,
                clean_before=clean_before.tolist(),clean_accepted=accepted.tolist(),
                nearest_mode=nearest.tolist(),distance_before=radial.norm(dim=1).tolist(),
                distance_after=(accepted-means[nearest]).norm(dim=1).tolist(),
                input_descent=per_row(local_direction),
                raw_joint_parameter_descent=per_row(raw_output_direction.detach()),
                accepted_joint=per_row(delta),network_only=per_row(network_only-clean_before),
                prior_only=per_row(prior_only-clean_before),
                grade_before=score_support(clean_before,indices,noise,means),
                grade_after=score_support(accepted,indices,noise,means))
    return row, dict(generator=deepcopy(generator.state_dict()),prior=deepcopy(prior.state_dict()),
                     optimizer=deepcopy(optimizer.state_dict())), proposal_state


class ClosureBudget(Exception):
    pass


def relax(critic, bank, gan, regularizer, step, metric):
    optimizer = torch.optim.LBFGS(critic.parameters(),lr=1.,max_iter=MAX_ITER,max_eval=MAX_CLOSURES,
                                 tolerance_grad=1e-7,tolerance_change=1e-12,history_size=10,
                                 line_search_fn='strong_wolfe')
    records = []
    best = None
    best_loss = float('inf')
    def closure():
        nonlocal best,best_loss
        if len(records)>=MAX_CLOSURES:
            raise ClosureBudget()
        optimizer.zero_grad()
        loss,logistic,penalty = d_loss(critic,bank,gan,regularizer,step)
        loss.backward()
        gradient = [parameter.grad.detach().clone() for parameter in critic.parameters()]
        value = float(loss.detach())
        if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in gradient):
            raise FloatingPointError('nonfinite local critic fit')
        records.append(dict(closure=len(records)+1,total_loss=value,logistic_loss=float(logistic.detach()),
                            penalty=float(penalty.detach()),gradient=norm_receipt(gradient,metric)))
        if value<best_loss:
            best_loss=value
            best=deepcopy(critic.state_dict())
        return loss
    exhausted=False
    started=time.perf_counter()
    try:
        optimizer.step(closure)
    except ClosureBudget:
        exhausted=True
    if best is None:
        raise RuntimeError('local critic fit evaluated no finite point')
    critic.load_state_dict(best)
    iterations=int(next(iter(optimizer.state.values())).get('n_iter',0))
    return dict(closure_calls=len(records),iterations=iterations,closure_budget_exhausted=exhausted,
                selection='lowest total fixed-bank loss among evaluated points; no quality selection',
                seconds=time.perf_counter()-started,records=records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    source_dir=args.output/'source';source_dir.mkdir()
    files=('reports/toy100/pr84_critic_relaxation.py','reports/toy100/alternating_curvature_scratch.py',
           'reports/toy100/coverage_fixed_eval.py','benchmarks/locked_shared/mode_hold.py',
           'benchmarks/locked_shared/mlp.py','particlegan/gan_loss.py','particlegan/grad_regularizers.py',
           'configs/toy100/constraints_simple_regularization.json')
    hashes={}
    for name in files:
        raw=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
        target=source_dir/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    input_file=args.capture/'selected-states.pt'
    if hashlib.sha256(input_file.read_bytes()).hexdigest()!=STATE_SHA:
        raise RuntimeError('wrong full capture-v2 state sidecar')
    declaration=dict(scope='fixed_generator_local_penalized_critic_fit_diagnostic',shared_gate_eligible=False,
        training_candidate=False,states=STATES,train_pairs=1024,heldout_pairs=1024,batch=128,
        max_lbfgs_iterations=MAX_ITER,max_closure_calls=MAX_CLOSURES,line_search='strong_wolfe',
        lbfgs_history=10,lbfgs_initial_step=1.,no_restarts=True,seed_changes=False,
        selection='lowest finite training-bank total loss among evaluated points',
        g_stencil_width=WIDTH,g_field_scope='same captured G batch, fixed original stencil',
        fit_optimality='local numerical evidence only; never a global or regularized-best-response certificate',
        fit_stationarity_report='raw and saved-metric residuals; tolerance raw gradient infinity norm <=1e-7',
        source=hashes,states_sha256=STATE_SHA)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',**declaration)),flush=True)
    states=torch.load(input_file,weights_only=True);original_hash=state_hash(states)
    recipe,_,_=declared_recipe(json.loads((ROOT/files[-1]).read_text()))
    gan=recipe.make_loss();regularizer=recipe.make_gradient_penalty()
    results=[];tensors={}
    outer_rng=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        for step in STATES:
            captured=states[step];saved=captured['post_accepted_d']
            generator,critic,prior=modules(saved)
            train_batches,heldout_batches,train,heldout,g_batch=banks(
                captured['pre_step'],saved,generator,prior)
            # The first bank row exactly recreates the real original D field
            # before its Adam step; beta1=0 stores that gradient in exp_avg.
            critic.load_state_dict(unwrapped(captured['pre_step']['critic']))
            original_loss,_,_=d_loss(critic,train_batches[0],gan,regularizer,step)
            original_gradient=gradients(original_loss,critic)
            expected=[saved['optimizer_d']['state'][index]['exp_avg']
                      for group in saved['optimizer_d']['param_groups'] for index in group['params']]
            if not all(torch.equal(a,b) for a,b in zip(original_gradient,expected)):
                raise RuntimeError(f'actual first D batch gradient parity failed at {step}')
            critic.load_state_dict(unwrapped(saved['critic']))
            metric=saved_metric(saved['optimizer_d'])
            before=dict(train=d_evaluation(critic,train,train_batches,gan,regularizer,step,metric),
                        heldout=d_evaluation(critic,heldout,heldout_batches,gan,regularizer,step,metric))
            g_before,g_state,proposal=g_proposal(saved,critic,g_batch,gan,step)
            reference=captured['post_bounded_g']
            expected_g=dict(generator=unwrapped(reference['generator']),prior=reference['prior'],
                            optimizer=reference['optimizer_g'])
            if state_hash(g_state)!=state_hash(expected_g):
                raise RuntimeError(f'original bounded G proposal parity failed at {step}')
            reference_unbounded=captured['post_unbounded_g']
            if state_hash(proposal)!=state_hash(dict(generator=unwrapped(reference_unbounded['generator']),
                    prior=reference_unbounded['prior'],optimizer=reference_unbounded['optimizer_g'])):
                raise RuntimeError(f'original unbounded G proposal parity failed at {step}')
            print(json.dumps(dict(event='BASELINE_PARITY',step=step,exact_d_gradient=True,exact_g_proposals=True)),flush=True)
            fit=relax(critic,train,gan,regularizer,step,metric)
            after=dict(train=d_evaluation(critic,train,train_batches,gan,regularizer,step,metric),
                       heldout=d_evaluation(critic,heldout,heldout_batches,gan,regularizer,step,metric))
            g_after,_,_=g_proposal(saved,critic,g_batch,gan,step)
            ratios={split:after[split]['gradient']['saved_adam_metric_l2']/
                          before[split]['gradient']['saved_adam_metric_l2'] for split in ('train','heldout')}
            stationary=after['train']['gradient']['raw_linf']<=1e-7
            row=dict(step=step,baseline_d_gradient_exact=True,baseline_g_proposals_exact=True,
                before=before,after=after,metric_residual_ratio=ratios,fit=fit,
                numerical_fit_status='STATIONARITY_TOLERANCE_MET' if stationary else 'NONCONVERGED_RESIDUAL',
                g_before=g_before,g_after=g_after,
                bank_sha256=state_hash(dict(train=train,heldout=heldout,g_batch=g_batch)),
                fitted_critic_sha256=state_hash(critic.state_dict()))
            results.append(row)
            tensors[step]=dict(train=train,heldout=heldout,g_batch=g_batch,critic=deepcopy(critic.state_dict()))
            (args.output/f'step-{step}.json').write_text(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(dict(event='FIT_DONE',step=step,status=row['numerical_fit_status'],
                closures=fit['closure_calls'],iterations=fit['iterations'],
                train_loss=[before['train']['total_loss'],after['train']['total_loss']],
                heldout_loss=[before['heldout']['total_loss'],after['heldout']['total_loss']],
                residual_ratio=ratios,g_hq=[g_before['grade_after']['hq'],g_after['grade_after']['hq']])),flush=True)
    if state_hash(states)!=original_hash or not torch.equal(torch.get_rng_state(),outer_rng):
        raise RuntimeError('diagnosis changed input saved state or outer RNG')
    torch.save(tensors,args.output/'fitted-critics-and-banks.pt')
    result=dict(declaration=declaration,input_states_unchanged=True,outer_rng_unchanged=True,rows=results)
    (args.output/'summary.json').write_text(json.dumps(result,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
