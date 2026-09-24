"""One source-bound actual-Adam finite-bank control at saved update1325.

This advances each cloned player's native Adam state exactly once, preserving
D-then-G order and PR84's original own-curvature bounds and width rule.  It
has no host continuation, persistent snapshot refresh or VR sampling policy.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100 import pr84_finite_bank_vr_diagnostic as probe
from reports.toy100.alternating_curvature_scratch import _metric, _rho
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support


METHOD = 'saved1325_actual_adam_mean16_alternating_control'


def critic_optimizer(critic, saved):
    params = list(critic.parameters())
    offset = 0
    groups = []
    for original in saved['param_groups']:
        size = len(original['params'])
        groups.append({key: deepcopy(value) for key, value in original.items() if key != 'params'} |
                      {'params': params[offset:offset+size]})
        offset += size
    if offset != len(params):
        raise ValueError('critic optimizer parameter order differs')
    optimizer = torch.optim.Adam(groups)
    optimizer.load_state_dict(deepcopy(saved))
    return optimizer


@torch.no_grad()
def smoothing_width(generator, prior, critic):
    points = generator(prior.z).detach()
    squared = 0.
    for axis in range(2):
        displacement = torch.zeros_like(points)
        displacement[:, axis] = 1e-3
        squared = squared + ((critic(points+displacement)-critic(points-displacement))/(2e-3)).square()
    sharpness = float(squared.mean().sqrt())
    if not math.isfinite(sharpness):
        raise FloatingPointError('nonfinite critic sharpness')
    return (min(.15, .5/sharpness) if sharpness > 1e-6 else 0.), sharpness


def smooth(critic, points, width):
    if not width:
        return critic(points)
    values = [critic(points)]
    for axis in range(2):
        displacement = torch.zeros_like(points)
        displacement[:, axis] = width
        values.extend((critic(points+displacement), critic(points-displacement)))
    return torch.stack(values).mean(0)


def mean_adam_update(pre, d_rows, g_rows, recipe):
    """Return a copied post-G/before-EMA state, without consuming random draws."""
    if pre['noise']['input_sigma'] != 0 or pre['noise']['output_sigma'] != .029:
        raise ValueError('single saved late mode-hold state only')
    if recipe.prior_reg != 0:
        raise ValueError('diagnostic excludes auxiliary prior terms')
    generator, critic, prior = probe.fit.modules(pre)
    opt_g = probe.fit.g_optimizer(generator, prior, pre['optimizer_g'])
    opt_d = critic_optimizer(critic, pre['optimizer_d'])
    d_params, g_params = list(critic.parameters()), list(generator.parameters())+list(prior.parameters())
    gan, cap = recipe.make_loss(), recipe.make_gradient_penalty()
    before = generator(prior.z).detach().clone()
    widths = []
    calls = {'d': 0, 'g': 0}

    def field(role, width=0.):
        params, rows = (d_params, d_rows) if role == 'd' else (g_params, g_rows)
        batches = []
        for row in rows:
            fake = generator(prior.z[row['indices']]) + .029*row['noise']
            if role == 'd':
                fake = fake.detach()
                loss = gan.d_loss(critic(row['real']), critic(fake))
                loss = loss + cap(critic, row['real'], fake, step=probe.STEP)
            else:
                loss = gan.g_loss(smooth(critic, fake, width), smooth(critic, row['real'], width))
            batches.append([value.detach().clone() for value in torch.autograd.grad(loss, params)])
            calls[role] += 1
        # Preserve native single-bank arithmetic exactly. For the finite mean,
        # accumulate gradients in float64, then round once to native dtype.
        result = batches[0] if len(batches) == 1 else [
            torch.stack([row[index].double() for row in batches]).mean(0).to(parameter.dtype)
            for index, parameter in enumerate(params)]
        if not all(torch.isfinite(value).all() for value in result):
            raise FloatingPointError('nonfinite mean gradient')
        return result

    dynamics = {}
    for role, optimizer, params, bound in (('d', opt_d, d_params, 3.), ('g', opt_g, g_params, .25)):
        width = 0.
        if role == 'g':
            width, sharp = smoothing_width(generator, prior, critic)
            widths.append(dict(stage='base_g', width=width, sharpness=sharp))
        base = [p.detach().clone() for p in params]
        first = field(role, width)
        optimizer.zero_grad()
        for p, value in zip(params, first):
            p.grad = value.clone()
        optimizer.step()
        proposed = [p.detach().clone() for p in params]
        metric = _metric(optimizer)
        if role == 'g':
            width, sharp = smoothing_width(generator, prior, critic)
            widths.append(dict(stage='proposal_g', width=width, sharpness=sharp))
        second = field(role, width)
        rho = _rho(base, proposed, first, second, metric)
        factor = min(1., bound/rho) if rho else 1.
        with torch.no_grad():
            for p, old, new in zip(params, base, proposed):
                p.copy_(torch.lerp(old, new, factor) if factor < 1 else new)
        moments = [int(optimizer.state[p]['step']) for p in params]
        expected = [int(pre['optimizer_'+role]['state'][index]['step'])+1
                    for group in pre['optimizer_'+role]['param_groups'] for index in group['params']]
        if moments != expected:
            raise RuntimeError('Adam moments did not advance exactly once')
        dynamics[role] = dict(rho=rho, factor=factor, moment_steps=moments,
                              optimizer_steps=1, nominal_rates=[group['lr'] for group in optimizer.param_groups])
    after = generator(prior.z).detach().clone()
    indices, noise = fixed_draw(probe.STEP, before)
    state = dict(generator=deepcopy(generator.state_dict()), critic=deepcopy(critic.state_dict()),
                 prior=deepcopy(prior.state_dict()), optimizer_d=deepcopy(opt_d.state_dict()),
                 optimizer_g=deepcopy(opt_g.state_dict()))
    receipt = dict(dynamics=dynamics, stencil=widths, gradient_calls=calls,
        discarded_native_phase_gradients='omitted in standalone diagnostic; no RNG is consumed by frozen draws',
        grade_before=score_support(before, indices, noise, probe.mode_hold.ring_means()),
        grade_after=score_support(after, indices, noise, probe.mode_hold.ring_means()),
        clean_rms=float((after-before).square().sum(1).mean().sqrt()),
        state_stage='after bounded G, before EMA', full_state_sha256=probe._sha(state),
        all_state_finite=all(torch.isfinite(value).all().item() for module in
                            (generator, critic, prior) for value in module.state_dict().values()))
    return receipt, state


def read(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--banks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    result_path = args.banks/('result.json' if (args.banks/'result.json').exists() else 'result.json.gz')
    tensor_path = args.banks/('tensors.pt' if (args.banks/'tensors.pt').exists() else 'tensors.pt.gz')
    previous = json.loads(read(result_path))
    for name, digest in previous['declaration']['source'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('frozen finite-bank source differs')
    declaration = dict(method=METHOD, step=probe.STEP, banks_per_role=16, outer_updates=1,
        arms=['native first-bank exact control', 'actual new-moment16-bank D-then-G Adam'],
        no_continuation=True, shared_gate_eligible=False, quality_selection=False,
        metric='each native Adam denominator/moment recomputed once from that arm first mean gradient',
        own_bounds=dict(d=3., g=.25), width='original PR84 sharpness rule recomputed before both G fields',
        gradient_mean='float64 per-native-batch gradient accumulation, rounded once to float32',
        bank_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        bank_tensor_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(args.states.read_bytes()).hexdigest(),
        source=previous['declaration']['source'] | {str(Path(__file__).relative_to(ROOT)):
               hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    bundle = torch.load(BytesIO(read(args.states)), weights_only=True)
    if bundle['parent_states_sha256'] != probe.fit.STATE_SHA:
        raise RuntimeError('wrong original capture')
    phases = bundle['states'][probe.STEP]
    banks = torch.load(BytesIO(read(tensor_path)), weights_only=True)
    recipe, _, _ = probe.declared_recipe(json.loads((ROOT/probe.SOURCE_FILES[-1]).read_text()))
    before = probe._sha(dict(phases=phases, banks=banks))
    rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        native, native_state = mean_adam_update(phases['pre_step'], banks['d_rows'][:1],
                                               banks['g_rows'][:1], recipe)
        final = phases['post_bounded_g']
        expected = {key: probe.fit.unwrapped(final[key]) if key in ('generator', 'critic')
                    else final[key] for key in native_state}
        if probe._sha(native_state) != probe._sha(expected):
            raise RuntimeError('native one-bank full model/Adam parity failed')
        print(json.dumps(dict(event='NATIVE_EXACT', grade=native['grade_after'])), flush=True)
        averaged, averaged_state = mean_adam_update(phases['pre_step'], banks['d_rows'], banks['g_rows'], recipe)
    if before != probe._sha(dict(phases=phases, banks=banks)) or not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError('diagnostic mutated input or caller RNG')
    result = dict(declaration=declaration, native_full_model_and_adam_exact=True,
        caller_rng_and_inputs_unchanged=True, native=native, mean16=averaged,
        status='SAVED_POINT_PASS' if averaged['grade_after']['modes'] == 8
               and averaged['grade_after']['hq'] >= .9 else 'SAVED_POINT_FAIL',
        shared_gate_eligible=False)
    torch.save(averaged_state, args.output/'accepted-state.pt')
    (args.output/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(event='DONE', status=result['status'], mean16=averaged)), flush=True)


if __name__ == '__main__':
    main()
