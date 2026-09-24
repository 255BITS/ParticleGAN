"""At most eight own moves on the exact fixed cold472 empirical value.

Includes the already archived first move. No host clock, resampling, new
Adam moments, feature learning, quality gate, or extra fitting budget.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100 import pr84_convex_profiled_value1530 as original
from reports.toy100.convex_readout_value import FrozenFeatures, ReadoutProblem, feature_arrays, fit_readout
from reports.toy100.functional_b_cap import functional_b_cap
from reports.toy100.pr84_critic_refinement_capture import _sha

STEP = 472
MAX_TOTAL_MOVES = 8
METHOD = 'cold472_consecutive_fixed_feature_finite_bank_profiled_value'
REFERENCE = ROOT/'reports/toy100/continuous-evidence/convex-profiled-value-independent/472'


def read(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def attempt_value_move(parameters, metric, readout, base_fit, loss, problem, *, audit=None, observe=None):
    """One fixed-metric move, returning the materialized point or exact rest."""
    base = [p.detach().clone() for p in parameters]
    gradient = [g.detach() for g in torch.autograd.grad(-loss(readout), parameters)]
    proposal = [-p*g for p, g in zip(metric, gradient)]
    predicted = float(sum((p*g.square()).sum() for p, g in zip(metric, gradient)))
    if not math.isfinite(predicted) or not all(torch.isfinite(g).all() for g in gradient):
        raise FloatingPointError('nonfinite generator field')

    def assign(alpha):
        with torch.no_grad():
            for value, before, delta in zip(parameters, base, proposal):
                value.copy_(before+alpha*delta)

    receipt = dict(gradient_l2=float(original.flattened(gradient).norm()),
        proposal_parameter_l2=float(original.flattened(proposal).norm()), predicted=predicted,
        trials=[], accepted_alpha=None, status='EXACT_ZERO_FIELD_REST', derivative_audit=[])
    if predicted == 0:
        return readout, base_fit, receipt
    if not base_fit['certificate']['available']:
        receipt['status'] = 'BASE_CERTIFICATE_UNAVAILABLE'
        return readout, base_fit, receipt
    if audit is not None:
        try:
            receipt['derivative_audit'] = audit(readout, proposal, predicted, base, assign)
        finally:
            assign(0.)
        if not all(row['passed'] for row in receipt['derivative_audit']):
            receipt['status'] = 'UNRESOLVED_PARTIAL_DERIVATIVE'
            return readout, base_fit, receipt
    selected = None
    try:
        for alpha in original.ALPHAS:
            assign(alpha)
            trial_readout, trial_fit = fit_readout(problem(), readout)
            certificate = trial_fit['certificate']
            bound = (certificate['lower']-base_fit['certificate']['upper']
                     if certificate['available'] else None)
            required = original.ARMIJO*alpha*predicted
            verified = bound is not None and bound > required+1e-12
            row = dict(alpha=alpha, fit=trial_fit, profiled_decrease_lower_bound=bound,
                       armijo_required=required, verified=verified)
            receipt['trials'].append(row)
            if observe is not None:
                observe(row)
            if verified:
                selected = (trial_readout, trial_fit)
                receipt.update(accepted_alpha=alpha, certified_decrease_lower_bound=bound,
                               status='VERIFIED_PROFILED_VALUE_DECREASE_WITH_DECLARED_INNER_GAPS')
                break
    finally:
        if selected is None:
            assign(0.)
    if selected is None:
        receipt['status'] = 'NO_VERIFIED_DECREASE_REST'
        return readout, base_fit, receipt
    return *selected, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    reference_bytes = read(REFERENCE/'result.json.gz')
    reference = json.loads(reference_bytes)
    tensor_bytes = read(REFERENCE/'tensors.pt.gz')
    if hashlib.sha256(tensor_bytes).hexdigest() != reference['tensors_sha256']:
        raise RuntimeError('cold first-move tensor payload changed')
    if not (reference['accepted_alpha'] == .25 and reference['input_and_caller_rng_unchanged']
            and reference['native_first_D_and_G_banks_exact']
            and reference['native_D_and_declared_reference_G_endpoint_exact']):
        raise RuntimeError('required first-move/native controls failed')
    tensors = torch.load(BytesIO(tensor_bytes), weights_only=True)
    sources = dict(reference['declaration']['sources'])
    for name, digest in sources.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'fixed first-move source changed: {name}')
    own_name = str(Path(__file__).resolve().relative_to(ROOT))
    sources[own_name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in sources:
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    declaration = dict(method=METHOD, sources=sources, shared_gate_eligible=False, training=False,
        reference_result_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        reference_tensors_sha256=reference['tensors_sha256'],
        maximum_total_moves=MAX_TOTAL_MOVES, previously_executed_moves=1, new_move_budget=7,
        objective=reference['declaration']['objective'], solver=reference['declaration']['solver'],
        proposal=reference['declaration']['proposal'],
        fixed='original sixteen D banks, accepted-D nonlinear features, bias gauge, post-Adam metric and evaluation draw',
        carried='own G/prior parameters and accepted fitted readout; no original-trajectory resets',
        base_fit='reuse the previous accepted evaluated point and its bound; no new base solve',
        moments=0, host_clock='never advanced; every diagnostic uses fixed472 evaluation draws',
        stop='exact zero field, no available base bound, unresolved derivative, no verified decrease, or eight total moves',
        quality='post-hoc only; never controls step selection or stopping',
        scope='one fixed finite objective; numerical float64 bounds, not interval arithmetic or a native-training claim')
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    source_input_hash = _sha(tensors)
    rng = torch.get_rng_state().clone()
    recipe, _, _ = original.declared_recipe(json.loads((ROOT/original.SOURCES[-1]).read_text()))
    gan, cap = recipe.make_loss(), recipe.make_gradient_penalty()
    rows, states = [], []
    with torch.random.fork_rng(devices=[]):
        generator, critic, prior = original.fit.modules(tensors['captured']['post_accepted_d'])
        generator.double(); critic.double(); prior.double()
        generator.load_state_dict(tensors['accepted_generator'])
        prior.load_state_dict(tensors['accepted_prior'])
        features = FrozenFeatures(critic)
        features.load_state_dict(tensors['features'])
        bias = critic.net[-1].bias.detach().clone()
        parameters = list(generator.parameters())+list(prior.parameters())
        metric = deepcopy(tensors['metric'])
        if len(metric) != len(parameters) or any(p.shape != m.shape for p, m in zip(parameters, metric)):
            raise RuntimeError('captured metric parameter order differs')
        immutable = _sha(dict(features=features.state_dict(), bias=bias, metric=metric, drows=tensors['drows']))
        real = torch.cat([row['real'] for row in tensors['drows']]).double()
        indices = torch.cat([row['indices'] for row in tensors['drows']])
        noise = torch.cat([row['noise'] for row in tensors['drows']]).double()
        readout = tensors['selected']['readout'].detach().clone()
        base_fit = deepcopy(next(row['fit'] for row in reference['trials'] if row['verified']))
        index, eval_noise = original.fixed_draw(STEP, tensors['before'].float())

        def clean():
            return generator(prior.z).detach().clone()

        if not torch.equal(clean(), tensors['after']):
            raise RuntimeError('loaded own first-move endpoint differs')

        def fake():
            return generator(prior.z[indices])+.029*noise

        def loss(weight):
            points = fake()
            evaluate = lambda x: F.linear(features(x), weight.unsqueeze(0), bias).squeeze(-1)
            return gan.d_loss(evaluate(real), evaluate(points))+functional_b_cap(cap, evaluate, real, points, STEP)

        def problem():
            return ReadoutProblem.from_points(features, real, fake().detach(), coeff=cap.coeff, kappa=cap.kappa)

        def quality(points):
            return original.score_support(points.float(), index, eval_noise, original.mode_hold.ring_means())

        def save_state(move):
            return dict(move=move, generator=deepcopy(generator.state_dict()), prior=deepcopy(prior.state_dict()),
                        readout=readout.clone(), fit=deepcopy(base_fit), support=clean())

        def audit(weight, proposal, predicted, base, assign):
            def cap_masks():
                _, jf = feature_arrays(features, fake().detach())
                _, jr = feature_arrays(features, real)
                scores = torch.einsum('nid,d->ni', torch.cat((jr, jf)), weight)
                return (scores.square().sum(1)+1e-12).sqrt()>cap.kappa
            _, base_masks = original.masks_at(lambda: -loss(weight), (generator, features))
            base_caps = cap_masks()
            audit_rows = []
            for fraction in original.FD_FRACTIONS:
                values, signs, caps, achieved = [], [], [], []
                for sign in (-1, 1):
                    assign(sign*fraction)
                    value, masks = original.masks_at(lambda: -loss(weight), (generator, features))
                    values.append(float(value.detach()))
                    signs.append(int((masks != base_masks).sum()))
                    caps.append(int((cap_masks() != base_caps).sum()))
                    actual = original.flattened([p-b for p, b in zip(parameters, base)])
                    expected = sign*fraction*original.flattened(proposal)
                    achieved.append(float((actual-expected).norm()/expected.norm()))
                central = (values[1]-values[0])/(2*fraction)
                error = abs(central+predicted)
                audit_rows.append(dict(fraction=fraction, analytic=-predicted, central=central,
                    absolute_error=error, relative_error=error/predicted,
                    leaky_sign_switches=signs, cap_switches=caps, achieved_direction_errors=achieved,
                    passed=error<=max(1e-7, .01*predicted) and not any(signs+caps)))
            return audit_rows

        # The initial move has already been executed, tested and archived.
        first_bound = tensors['selected']['decrease_lower']
        rows.append(dict(move=1, provenance='previously archived independent472 assay',
            accepted_alpha=reference['accepted_alpha'], quality=reference['quality'],
            certified_decrease_lower_bound=first_bound, base_fit=reference['base_fit'], fit=base_fit,
            accepted_motion=reference['accepted_motion'], status=reference['status']))
        states.append(save_state(1))
        initial_upper = reference['base_fit']['certificate']['upper']
        cumulative_bound = first_bound
        stop = 'MAXIMUM_EIGHT_TOTAL_MOVES'
        for move in range(2, MAX_TOTAL_MOVES+1):
            before = clean()
            cached = problem().loss(readout)
            exact_loss = float(loss(readout).detach())
            if abs(float(cached)-exact_loss)>1e-11 or abs(exact_loss-base_fit['final_loss'])>1e-11:
                raise RuntimeError('carried objective/readout no longer matches the accepted point')
            preceding = deepcopy(base_fit)
            def observe(trial):
                print(json.dumps(dict(event='VALUE_TRIAL', move=move, alpha=trial['alpha'],
                    fit_status=trial['fit']['status'], closures=trial['fit']['closure_calls'],
                    decrease_lower=trial['profiled_decrease_lower_bound'], verified=trial['verified'])), flush=True)
            readout, base_fit, receipt = attempt_value_move(parameters, metric, readout, base_fit,
                                                           loss, problem, audit=audit, observe=observe)
            after = clean()
            receipt.update(move=move, base_fit=preceding, fit=deepcopy(base_fit),
                quality=dict(before=quality(before), after=quality(after)),
                accepted_motion=original.direction_record(before, after-before))
            if receipt['accepted_alpha'] is not None:
                cumulative_bound += receipt['certified_decrease_lower_bound']
            receipt['cumulative_decrease_lower_bound'] = cumulative_bound
            receipt['endpoint_decrease_lower_bound'] = base_fit['certificate']['lower']-initial_upper
            rows.append(receipt)
            states.append(save_state(move))
            torch.save(dict(states=states, rows=rows), args.output/'progress.pt')
            print(json.dumps(dict(event='MOVE_DONE', move=move, status=receipt['status'],
                alpha=receipt['accepted_alpha'], quality=receipt['quality']['after'],
                gradient_l2=receipt['gradient_l2'], predicted=receipt['predicted'],
                output_rms=receipt['accepted_motion']['rms'], cumulative_bound=cumulative_bound)), flush=True)
            if receipt['accepted_alpha'] is None:
                stop = receipt['status']
                break
        if immutable != _sha(dict(features=features.state_dict(), bias=bias, metric=metric, drows=tensors['drows'])):
            raise RuntimeError('fixed data, features or metric changed')
        payload = dict(reference=tensors, declaration=declaration, states=states,
                       features=features.state_dict(), bias=bias, metric=metric, drows=tensors['drows'])
    if _sha(tensors) != source_input_hash or not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError('continuation changed input or caller RNG')
    torch.save(payload, args.output/'states.pt')
    new_closures = sum(t['fit']['closure_calls'] for row in rows[1:] for t in row['trials'])
    new_final_gradients = sum(t['fit']['final_gradient_evaluations'] for row in rows[1:] for t in row['trials'])
    result = dict(declaration=declaration, stop=stop, moves=rows,
        accepted_total_moves=sum(row['accepted_alpha'] is not None for row in rows),
        cumulative_decrease_lower_bound=cumulative_bound,
        endpoint_decrease_lower_bound=base_fit['certificate']['lower']-initial_upper,
        before=reference['quality']['before'], after=rows[-1]['quality']['after'],
        new_readout_closures=new_closures, new_readonly_final_gradients=new_final_gradients,
        prior_first_move_closures=reference['base_fit']['closure_calls']+
            sum(t['fit']['closure_calls'] for t in reference['trials']),
        moment_updates=0, host_updates=0, immutable_problem_verified=True,
        source_input_and_caller_rng_unchanged=True, shared_gate_eligible=False,
        states_sha256=hashlib.sha256((args.output/'states.pt').read_bytes()).hexdigest())
    (args.output/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE', stop=stop, accepted_moves=result['accepted_total_moves'],
                         quality=result['after'], cumulative_bound=cumulative_bound)), flush=True)


if __name__ == '__main__':
    main()
