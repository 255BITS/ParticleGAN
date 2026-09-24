"""One saved-state frozen-feature convex critic/value diagnostic; no training."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch import nn
from torch.func import functional_call, jvp
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100 import pr84_critic_relaxation as fit
from reports.toy100 import pr84_finite_bank_vr_diagnostic as bank_source
from reports.toy100 import pr84_finite_bank_adam_control as native_control
from reports.toy100.convex_readout_value import (
    FrozenFeatures, ReadoutProblem, feature_arrays, fit_readout,
    MAX_ITERATIONS, MAX_CLOSURES, GAP_TOLERANCE,
)
from reports.toy100.functional_b_cap import functional_b_cap
from reports.toy100.pr84_critic_refinement_capture import _sha
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from benchmarks.locked_shared import mode_hold

STEP = 1530
ALPHAS = tuple(2.**-index for index in range(9))
ARMIJO = 1e-4
FD_FRACTIONS = (1e-7, 5e-8)
SOURCES = (
    'reports/toy100/pr84_convex_profiled_value1530.py',
    'reports/toy100/convex_readout_value.py',
    'reports/toy100/functional_b_cap.py',
    'reports/toy100/pr84_critic_relaxation.py',
    'reports/toy100/pr84_finite_bank_vr_diagnostic.py',
    'reports/toy100/pr84_finite_bank_adam_control.py',
    'reports/toy100/alternating_curvature_scratch.py',
    'reports/toy100/pr84_critic_refinement_capture.py',
    'reports/toy100/coverage_fixed_eval.py',
    'benchmarks/locked_shared/mode_hold.py', 'benchmarks/locked_shared/mlp.py',
    'benchmarks/transfer_suite/toy100_compatibility.py',
    'particlegan/gan_loss.py', 'particlegan/grad_regularizers.py',
    'particlegan/particle_prior.py',
    'configs/toy100/constraints_simple_regularization.json',
)


def flattened(values):
    return torch.cat([value.detach().double().flatten() for value in values])


def output_direction(generator, prior, direction):
    names = list(dict(generator.named_parameters()))
    parameters = tuple(generator.parameters())
    def clean(values, latent):
        return functional_call(generator, dict(zip(names, values)), (latent,))
    return jvp(clean, (parameters, prior.z), (tuple(direction[:-1]), direction[-1]))[1].detach()


def direction_record(before, movement):
    # Post-hoc only; never used by the optimizer or its value acceptance.
    centers = mode_hold.ring_means().to(before)
    nearest = torch.cdist(before, centers).argmin(1)
    radial = before-centers[nearest]
    products = (movement*radial).sum(1)
    return dict(rms=float(movement.square().sum(1).mean().sqrt()),
                radial_work=float(products.mean()), inward_particles=int((products<0).sum()),
                per_particle_radial_work=products.tolist(), vectors=movement.tolist())


def masks_at(call, modules):
    masks = []
    def observe(_, values):
        masks.append((values[0].detach() >= 0).flatten().clone())
    with ExitStack() as stack:
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.LeakyReLU):
                    stack.callback(layer.register_forward_pre_hook(observe).remove)
        value = call()
    return value, torch.cat(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    raw = args.states.read_bytes()
    if hashlib.sha256(raw).hexdigest() != fit.STATE_SHA:
        raise RuntimeError('expected exact original capture-v2')
    source_dir = args.output/'source'
    hashes = {}
    for name in SOURCES:
        data = (ROOT/name).read_bytes()
        target = source_dir/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    declaration = dict(method='saved1530_convex_readout_profiled_sharp_D_value',
        shared_gate_eligible=False, training=False, step=STEP, sources=hashes,
        input_sha256=fit.STATE_SHA, banks=16, samples_per_bank=128,
        critic='nonlinear features frozen at native accepted D; final96-dimensional readout only; output bias fixed gauge',
        objective='V=log2-min_readout(sharp Rp logistic D loss + original b_cap) on one fixed paired empirical bank',
        g_gradient='partial -full sharp D objective, including fake-input cap derivative; no optimizer differentiation',
        metric='captured post-bounded-G Adam diagonal; fixed; no new moments or optimizer step',
        solver=dict(method='one LBFGS attempt in invertibly whitened coordinates per point',
                    max_iterations=MAX_ITERATIONS, max_closures=MAX_CLOSURES,
                    gap_tolerance=GAP_TOLERANCE, added_regularizer=None),
        proposal=dict(alphas=ALPHAS, armijo=ARMIJO,
                      accept='trial D-loss lower bound > baseline evaluated upper bound + Armijo predicted decrease'),
        fd=dict(fractions=FD_FRACTIONS, scope='fixed fitted readout partial generator gradient',
                pass_condition='both scales relative error<=1% or absolute error<=1e-7; no branch switches'),
        quality='read-only paired original noisy grade and per-particle radial directions; no acceptance access',
        limitation='finite-bank and frozen-feature class only; numerical global bound is not interval arithmetic or neural best response')
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    captured = torch.load(args.states, weights_only=True)[STEP]
    recipe, _, _ = declared_recipe(json.loads((ROOT/SOURCES[-1]).read_text()))
    gan, cap = recipe.make_loss(), recipe.make_gradient_penalty()
    initial_hash = _sha(captured)
    global_rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]), patch.object(bank_source, 'STEP', STEP):
        pre, accepted = captured['pre_step'], captured['post_accepted_d']
        generator, critic, prior = fit.modules(pre)
        d0, d1, _, _, native_g = fit.banks(pre, accepted, generator, prior)
        drows = d0+d1
        grows = bank_source.g_banks(accepted, generator, prior)
        assert _sha(grows[0]) == _sha(native_g)
        native_gradient = fit.gradients(fit.d_loss(critic, drows[0], gan, cap, STEP)[0], critic)
        expected = [accepted['optimizer_d']['state'][index]['exp_avg']
                    for group in accepted['optimizer_d']['param_groups'] for index in group['params']]
        assert all(torch.equal(a, b) for a, b in zip(native_gradient, expected))
        native, native_state = native_control.mean_adam_update(pre, drows[:1], grows[:1], recipe)
        target = {name: fit.unwrapped(captured['post_bounded_g'][name]) if name in ('generator', 'critic')
                  else captured['post_bounded_g'][name] for name in native_state}
        assert _sha(native_state) == _sha(target)
        print(json.dumps(dict(event='NATIVE_EXACT', grade=native['grade_after'])), flush=True)

        critic.load_state_dict(fit.unwrapped(accepted['critic']))
        generator.double(); critic.double(); prior.double()
        features = FrozenFeatures(critic)
        feature_hash = _sha(features.state_dict())
        initial_readout = critic.net[-1].weight.detach().flatten().clone()
        bias = critic.net[-1].bias.detach().clone()
        parameters = list(generator.parameters())+list(prior.parameters())
        parameter_base = [p.detach().clone() for p in parameters]
        before = generator(prior.z).detach().clone()
        metric = fit.saved_metric(captured['post_bounded_g']['optimizer_g'])
        real = torch.cat([row['real'] for row in drows]).double()
        indices = torch.cat([row['indices'] for row in drows])
        noise = torch.cat([row['noise'] for row in drows]).double()

        def fake_points():
            return generator(prior.z[indices])+.029*noise

        def critic_function(weight):
            return lambda x: F.linear(features(x), weight.unsqueeze(0), bias).squeeze(-1)

        def sharp_loss(weight):
            fake = fake_points()
            evaluate = critic_function(weight)
            return gan.d_loss(evaluate(real), evaluate(fake))+functional_b_cap(cap, evaluate, real, fake, STEP)

        def problem_at_point():
            return ReadoutProblem.from_points(features, real, fake_points().detach(),
                                             coeff=cap.coeff, kappa=cap.kappa)

        base_problem = problem_at_point()
        variable = initial_readout.clone().requires_grad_(True)
        functional_loss = sharp_loss(variable)
        functional_gradient = torch.autograd.grad(functional_loss, variable)[0]
        cached_loss = base_problem.loss(variable)
        cached_gradient = torch.autograd.grad(cached_loss, variable)[0]
        assert torch.allclose(functional_loss, cached_loss, atol=1e-11, rtol=1e-11)
        assert torch.allclose(functional_gradient, cached_gradient, atol=1e-10, rtol=1e-10)
        print(json.dumps(dict(event='CACHE_PARITY', loss_error=float((functional_loss-cached_loss).detach().abs()),
                             gradient_error=float((functional_gradient-cached_gradient).abs().max()),
                             geometry=base_problem.geometry())), flush=True)
        fitted, base_fit = fit_readout(base_problem, initial_readout)
        print(json.dumps(dict(event='BASE_FIT', status=base_fit['status'], calls=base_fit['closure_calls'],
                             loss=base_fit['final_loss'], certificate=base_fit['certificate'])), flush=True)

        fields = {}
        gradient = None
        for name, weight, profile in (
                ('current_nonsaturating_stencil', initial_readout, False),
                ('refitted_nonsaturating_stencil', fitted, False),
                ('refitted_negative_full_sharp_D', fitted, True)):
            evaluate = critic_function(weight)
            loss = -sharp_loss(weight) if profile else gan.g_loss(
                fit.smooth(evaluate, fake_points()), fit.smooth(evaluate, real))
            values = [g.detach() for g in torch.autograd.grad(loss, parameters)]
            delta = [-p*g for p, g in zip(metric, values)]
            fields[name] = dict(loss=float(loss.detach()), gradient_l2=float(flattened(values).norm()),
                raw_direction=direction_record(before, output_direction(generator, prior, [-g for g in values])),
                metric_direction=direction_record(before, output_direction(generator, prior, delta)))
            if profile:
                gradient, proposal = values, delta
        predicted = float(sum((p*g.square()).sum() for p, g in zip(metric, gradient)))

        def assign(alpha):
            with torch.no_grad():
                for p, base, delta in zip(parameters, parameter_base, proposal):
                    p.copy_(base+alpha*delta)

        def cap_masks():
            _, jf = feature_arrays(features, fake_points().detach())
            _, jr = feature_arrays(features, real)
            scores = torch.einsum('nid,d->ni', torch.cat((jr, jf)), fitted)
            return (scores.square().sum(1)+1e-12).sqrt()>cap.kappa

        _, base_masks = masks_at(lambda: -sharp_loss(fitted), (generator, features))
        base_caps = cap_masks()
        fd = []
        for fraction in FD_FRACTIONS:
            losses, signs, cap_signs, achieved = [], [], [], []
            for sign in (-1, 1):
                assign(sign*fraction)
                loss, masks = masks_at(lambda: -sharp_loss(fitted), (generator, features))
                losses.append(float(loss.detach()))
                signs.append(int((masks != base_masks).sum()))
                cap_signs.append(int((cap_masks() != base_caps).sum()))
                actual = flattened([p-b for p, b in zip(parameters, parameter_base)])
                expected = sign*fraction*flattened(proposal)
                achieved.append(float((actual-expected).norm()/expected.norm()) if expected.norm() else 0.)
            central = (losses[1]-losses[0])/(2*fraction)
            error = abs(central+predicted)
            fd.append(dict(fraction=fraction, analytic=-predicted, central=central,
                absolute_error=error, relative_error=error/max(predicted, 1e-30),
                leaky_sign_switches=signs, cap_switches=cap_signs, achieved_direction_errors=achieved,
                passed=error<=max(1e-7, .01*predicted) and not any(signs+cap_signs)))
        assign(0.)
        fd_pass = all(row['passed'] for row in fd)
        print(json.dumps(dict(event='PARTIAL_GRADIENT_AUDIT', passed=fd_pass, rows=fd)), flush=True)

        trials = []
        selected = None
        if base_fit['status'] == 'CERTIFIED_GAP' and fd_pass and predicted > 0:
            for alpha in ALPHAS:
                assign(alpha)
                trial_problem = problem_at_point()
                readout, trial_fit = fit_readout(trial_problem, fitted)
                certificate = trial_fit['certificate']
                bound = (certificate['lower']-base_fit['certificate']['upper']
                         if certificate['available'] else None)
                required = ARMIJO*alpha*predicted
                verified = (trial_fit['status'] == 'CERTIFIED_GAP' and bound is not None
                            and bound > required+1e-12)
                record = dict(alpha=alpha, fit=trial_fit,
                    profiled_decrease_lower_bound=bound, armijo_required=required, verified=verified)
                trials.append(record)
                print(json.dumps(dict(event='VALUE_TRIAL', alpha=alpha, status=trial_fit['status'],
                    calls=trial_fit['closure_calls'], decrease_lower=bound, required=required,
                    verified=verified)), flush=True)
                if verified:
                    selected = dict(alpha=alpha, readout=readout, decrease_lower=bound)
                    break
        if selected is None:
            assign(0.)
        after = generator(prior.z).detach().clone()
        draw_index, draw_noise = fixed_draw(STEP, before.float())
        grades = {name: score_support(points.float(), draw_index, draw_noise, mode_hold.ring_means())
                  for name, points in (('before', before), ('after', after))}
        if _sha(features.state_dict()) != feature_hash:
            raise RuntimeError('frozen features changed')
        tensors = dict(captured=captured, drows=drows, grows=grows, metric=metric,
            features=features.state_dict(), initial_readout=initial_readout, fitted_readout=fitted,
            gradient=gradient, proposal=proposal, before=before, after=after,
            accepted_generator=deepcopy(generator.state_dict()), accepted_prior=deepcopy(prior.state_dict()),
            selected=selected)
        result = dict(declaration=declaration, native_first_D_and_G_banks_exact=True,
            native_single_bank_full_model_Adam_exact=True, cached_loss_and_gradient_parity=True,
            native=native, base_fit=base_fit, fields=fields, partial_gradient_fd=fd,
            trials=trials, accepted_alpha=None if selected is None else selected['alpha'],
            quality=grades, accepted_motion=direction_record(before, after-before),
            moment_updates=0, status='VERIFIED_PROFILED_VALUE_DECREASE' if selected else
                'UNRESOLVED_BASE_OPTIMALITY' if base_fit['status'] != 'CERTIFIED_GAP' else
                'UNRESOLVED_PARTIAL_DERIVATIVE' if not fd_pass else 'NO_VERIFIED_DECREASE',
            shared_gate_eligible=False)
    if _sha(captured) != initial_hash or not torch.equal(global_rng, torch.get_rng_state()):
        raise RuntimeError('saved-state diagnostic changed input or caller RNG')
    result['input_and_caller_rng_unchanged'] = True
    torch.save(tensors, args.output/'tensors.pt')
    result['tensors_sha256'] = hashlib.sha256((args.output/'tensors.pt').read_bytes()).hexdigest()
    (args.output/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE', status=result['status'], quality=grades,
                         accepted_alpha=result['accepted_alpha'])), flush=True)


if __name__ == '__main__':
    main()
