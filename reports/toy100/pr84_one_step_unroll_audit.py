"""Local derivative audit of the frozen one-step-unroll diagnostic.

The native float32 audit is retained. A separate float64 copy evaluates a
predeclared central-difference sequence and records activation/cap switches.
This changes neither the diagnostic surrogate nor any training update.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100 import pr84_one_step_unroll as unroll
from reports.toy100.pr84_critic_refinement_capture import _sha
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe


DOUBLE_EXPONENTS = (0, 2, 4, 6, 8, 10, 12, 14, 16)


class ActivationTrace:
    def __init__(self, generator, critic):
        self.models = {'G': generator, 'D': critic}
        self.calls = defaultdict(lambda: -1)
        self.masks = {}
        self.zeros = {}
        self.minimum_absolute = {}
        self.handles = []

    def __enter__(self):
        for role, model in self.models.items():
            def count(_module, _args, role=role):
                self.calls[role] += 1
            self.handles.append(model.register_forward_pre_hook(count))
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.LeakyReLU):
                    def capture(_module, args, role=role, name=name):
                        values = args[0].detach()
                        key = f'{role}/{self.calls[role]}/{name}'
                        self.masks[key] = (values > 0).clone()
                        self.zeros[key] = int((values == 0).sum())
                        self.minimum_absolute[key] = float(values.abs().min())
                    self.handles.append(module.register_forward_pre_hook(capture))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()


def cast_batch(batch, dtype):
    return {name: (value.to(dtype) if isinstance(value, torch.Tensor)
                   and value.is_floating_point() else value)
            for name, value in batch.items()}


def inspect(generator, prior, critic, metric, d_batch, g_batch, gan, regularizer, step):
    with ActivationTrace(generator, critic) as trace:
        loss, record = unroll.surrogate(generator, prior, critic, metric, d_batch, g_batch,
                                        gan, regularizer, step, full_chain=True)
    params = list(generator.parameters()) + list(prior.parameters())
    gradient = torch.autograd.grad(loss, params)
    fake = generator(prior.z[d_batch['indices']]) + d_batch['sigma'] * d_batch['noise']
    masks = dict(trace.masks)
    cap = {}
    for name, inputs in (('real', d_batch['real']), ('fake', fake)):
        inputs = inputs.detach().clone().requires_grad_(True)
        derivative = torch.autograd.grad(critic(inputs).sum(), inputs)[0]
        norm = (derivative.square().sum(1) + 1e-12).sqrt()
        masks[f'cap/{name}'] = (norm > regularizer.center(step)).detach().clone()
        cap[name] = dict(active=int(masks[f'cap/{name}'].sum()),
                         minimum_margin=float((norm-regularizer.center(step)).abs().min()))
    return dict(loss=float(loss.detach()), gradient=[g.detach().clone() for g in gradient],
                masks=masks, cap=cap, zeros=trace.zeros,
                minimum_absolute=trace.minimum_absolute,
                forward_calls={key: value+1 for key, value in trace.calls.items()},
                virtual=record['virtual_parameters'])


def mask_changes(base, current):
    if base.keys() != current.keys():
        raise RuntimeError('activation-call topology changed')
    counts = defaultdict(int)
    details = {}
    for key in base:
        count = int((base[key] != current[key]).sum())
        if key.startswith('cap/'):
            category = 'cap'
        elif key.startswith('G/'):
            category = 'generator'
        elif int(key.split('/')[1]) < 4:
            category = 'starting_critic'
        else:
            category = 'virtual_critic'
        counts[category] += count
        if count:
            details[key] = count
    return dict(total=sum(counts.values()), by_stage=dict(counts), nonzero_layers=details)


def audit_dtype(tensor, recipe, step, dtype, exponents):
    generator, critic, prior = unroll.fit.modules(tensor['saved'])
    critic.load_state_dict(tensor['critic'])
    generator.to(dtype); critic.to(dtype); prior.to(dtype)
    d_batch, g_batch = (cast_batch(tensor[name], dtype) for name in ('d_batch', 'g_batch'))
    metric = [value.detach().clone().double() for value in tensor['metric']]
    critic_hash, metric_hash = _sha(critic.state_dict()), _sha(metric)
    gan, regularizer = recipe.make_loss(), recipe.make_gradient_penalty()
    params = list(generator.parameters()) + list(prior.parameters())
    base = [parameter.detach().clone() for parameter in params]
    original = inspect(generator, prior, critic, metric, d_batch, g_batch, gan, regularizer, step)
    if dtype == torch.float32 and not all(torch.equal(a, b) for a, b in
            zip(original['gradient'], tensor['full']['gradient'])):
        raise RuntimeError('passive activation observer changed the native full-chain gradient')
    norm = float(unroll.flat(original['gradient']).norm())
    direction = [-value/norm for value in original['gradient']]
    analytic = float(unroll.flat(original['gradient']) @ unroll.flat(direction))
    h0 = 2e-4 * (1 + float(unroll.flat(base).norm()))
    rows = []
    try:
        for exponent in exponents:
            h = h0 * (2. ** -exponent)
            endpoints = []
            for sign in (-1, 1):
                with torch.no_grad():
                    for parameter, old, vector in zip(params, base, direction):
                        parameter.copy_(old + sign*h*vector)
                achieved = unroll.flat([parameter.detach()-old for parameter, old in zip(params, base)])
                expected = sign*h*unroll.flat(direction)
                endpoint = inspect(generator, prior, critic, metric, d_batch, g_batch,
                                   gan, regularizer, step)
                endpoints.append(dict(sign=sign, loss=endpoint['loss'],
                    displacement_norm=float(achieved.norm()),
                    relative_direction_error=float((achieved-expected).norm()/expected.norm()),
                    switches=mask_changes(original['masks'], endpoint['masks']),
                    cap=endpoint['cap']))
            central = (endpoints[1]['loss']-endpoints[0]['loss'])/(2*h)
            error = abs(central-analytic)
            rows.append(dict(exponent=exponent, h=h, analytic=analytic, central=central,
                absolute_error=error, relative_error=error/abs(analytic),
                original_criterion_passed=error <= max(1e-5, .02*abs(analytic)),
                all_masks_unchanged=all(row['switches']['total']==0 for row in endpoints),
                endpoints=endpoints))
    finally:
        with torch.no_grad():
            for parameter, old in zip(params, base):
                parameter.copy_(old)
    if (critic_hash != _sha(critic.state_dict()) or metric_hash != _sha(metric)
            or not all(torch.equal(parameter, old) for parameter, old in zip(params, base))
            or unroll.fit.WIDTH != .15):
        raise RuntimeError('fixed starting state, metric, or stencil changed')
    stable = [row for row in rows if row['all_masks_unchanged']]
    return dict(dtype=str(dtype), h0=h0, gradient_norm=norm,
        cosine_with_native=unroll.cosine(original['gradient'], tensor['full']['gradient']),
        relative_gradient_change_from_native=float((unroll.flat(original['gradient'])-
            unroll.flat(tensor['full']['gradient'])).norm()/unroll.flat(tensor['full']['gradient']).norm()),
        base_loss=original['loss'], forward_calls=original['forward_calls'],
        exact_zero_preactivations=sum(original['zeros'].values()),
        minimum_absolute_preactivation=min(original['minimum_absolute'].values()),
        base_cap=original['cap'], original_masks_sha=_sha(original['masks']),
        native_gradient_exact=(dtype == torch.float32), rows=rows,
        unchanged_mask_rows=len(stable),
        unchanged_mask_rows_all_pass=bool(stable) and all(row['original_criterion_passed'] for row in stable),
        starting_critic_metric_stencil_unchanged=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source_names = ('reports/toy100/pr84_one_step_unroll_audit.py',
        'reports/toy100/pr84_one_step_unroll.py', 'reports/toy100/functional_b_cap.py',
        'benchmarks/locked_shared/mlp.py')
    declaration = dict(scope='derivative audit only;no optimizer or critic fit',
        original_float32_retained=True, native_exponents=[0, 1],
        double_exponents=list(DOUBLE_EXPONENTS), all_declared_scales_evaluated=True,
        h0='2e-4*(1+original G/prior parameter L2 norm)',
        fixed='cached native draws, starting critic, saved double P_D, width .15',
        double_scope='same float32 saved weights/Fourier buffers/draws cast to float64;not a training arm',
        mask_call_order='D0/1 logistic real/fake,D2/3 cap real/fake,D4..8 virtual fake stencil,D9..13 virtual real stencil',
        input_sha=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        config_sha=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names},
        shared_gate_eligible=False)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in source_names:
        out=args.output/'source'/name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((ROOT/name).read_bytes())
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    tensors = torch.load(args.input, weights_only=True)
    recipe, _, _ = declared_recipe(json.loads(args.config.read_text()))
    before = _sha(tensors); rng = torch.get_rng_state().clone()
    rows = []
    with torch.random.fork_rng(devices=[]):
        for step, tensor in tensors.items():
            row = dict(step=step,
                native=audit_dtype(tensor, recipe, step, torch.float32, (0, 1)),
                double=audit_dtype(tensor, recipe, step, torch.float64, DOUBLE_EXPONENTS))
            rows.append(row)
            print(json.dumps(dict(event='STATE_DONE', **row), allow_nan=False), flush=True)
    if before != _sha(tensors) or not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError('input or external RNG changed')
    result=dict(declaration=declaration, rows=rows, input_and_rng_unchanged=True,
                shared_gate_eligible=False)
    (args.output/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
