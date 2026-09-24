"""Fixed, finer numerical audit after the initial kink-crossing secants.

The initial two scales remain evidence.  This separate diagnostic uses one
declared smaller pair, and an exact own-G autograd quadratic form.  Neither
changes a training method, data bank, Adam state or step length.
"""

import argparse
from io import BytesIO
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100 import pr84_finite_bank_vr_diagnostic as probe
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support


FRACTIONS = (1e-7, 5e-8)


def own_g_quadratic(field, point, direction):
    field.assign(point)
    values = []
    position = 0
    for parameter in field.g_params:
        values.append(direction[position:position+parameter.numel()].reshape_as(parameter))
        position += parameter.numel()
    zeros = 0
    def inspect(_module, args):
        nonlocal zeros
        zeros += int((args[0] == 0).sum())
    handles = [module.register_forward_pre_hook(inspect)
               for model in (field.generator, field.critic) for module in model.modules()
               if isinstance(module, torch.nn.LeakyReLU)]
    work = 0.
    try:
        for row in field.rows['g']:
            fake = field.generator(field.prior.z[row['indices']]) + field.sigma*row['noise']
            loss = field.gan.g_loss(probe.fit.smooth(field.critic, fake),
                                   probe.fit.smooth(field.critic, row['real']))
            first = torch.autograd.grad(loss, field.g_params, create_graph=True)
            directional = sum((g*v).sum() for g, v in zip(first, values))
            second = torch.autograd.grad(directional, field.g_params)
            work += sum(float((g*v).sum()) for g, v in zip(second, values))/probe.BANKS
    finally:
        for handle in handles:
            handle.remove()
    norm = float((direction.square()/field.metric[field.d_size:]).sum())
    return dict(raw_directional_second_derivative=work,
                metric_rayleigh=work/norm, exact_zero_activation_inputs=zeros,
                scope='own G Hessian; critic fixed; float64 local autograd branch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    initial = json.loads((args.initial/'result.json').read_text())
    source = initial['declaration']['source']
    for name, sha in source.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('initial diagnostic source changed')
    declaration = dict(scope='numerical audit, no candidate or training',
        initial_result_sha256=hashlib.sha256((args.initial/'result.json').read_bytes()).hexdigest(),
        sources=source | {str(Path(__file__).relative_to(ROOT)):
                         hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        finer_fractions=FRACTIONS, fraction_selection='one declared pair after unresolved initial pair',
        dtype='float64', shared_gate_eligible=False)
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    with torch.random.fork_rng(devices=[]):
        if args.states.suffix == '.gz':
            bundle = torch.load(BytesIO(gzip.decompress(args.states.read_bytes())), weights_only=True)
            if bundle['parent_states_sha256'] != probe.fit.STATE_SHA:
                raise RuntimeError('wrong parent capture')
            phases = bundle['states'][probe.STEP]
        else:
            if hashlib.sha256(args.states.read_bytes()).hexdigest() != probe.fit.STATE_SHA:
                raise RuntimeError('wrong parent capture')
            phases = torch.load(args.states, weights_only=True)[probe.STEP]
        tensors = torch.load(args.initial/'tensors.pt', weights_only=True)
        recipe, _, _ = probe.declared_recipe(json.loads((ROOT/probe.SOURCE_FILES[-1]).read_text()))
        field = probe.FixedBankField(phases, tensors['d_rows'], tensors['g_rows'], recipe)
        delta = tensors['captured_delta']
        probe.FRACTIONS = FRACTIONS
        geometry = probe.local_geometry(field, delta)
        pre = own_g_quadratic(field, field.base, delta[field.d_size:])
        point = field.base.clone()
        point[:field.d_size] += delta[:field.d_size]
        accepted_d = own_g_quadratic(field, point, delta[field.d_size:])
        endpoint, bounds = field.alternating_map(field.base)
        field.assign(field.base)
        before = field.generator(field.prior.z).detach().float()
        field.assign(endpoint)
        after = field.generator(field.prior.z).detach().float()
        indices, noise = fixed_draw(probe.STEP, before)
        mean_map = dict(before=score_support(before, indices, noise, probe.mode_hold.ring_means()),
                        after=score_support(after, indices, noise, probe.mode_hold.ring_means()),
                        clean_rms=float((after-before).square().sum(1).mean().sqrt()), bounds=bounds,
                        scope='fixed-metric mean D then mean G, not new-moment Adam')
    result = dict(declaration=declaration, local_geometry=geometry,
        own_g_autograd=dict(pre_d=pre, native_accepted_d=accepted_d),
        mean_alternating_map=mean_map, gradient_calls=field.calls,
        status='NUMERICAL_AUDIT_COMPLETE', shared_gate_eligible=False)
    (args.output/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(event='DONE', own_g=result['own_g_autograd'], mean_map=mean_map)), flush=True)


if __name__ == '__main__':
    main()
