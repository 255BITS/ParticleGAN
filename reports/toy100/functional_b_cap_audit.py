"""Read-only parity and mixed-derivative audit for virtual-critic b_cap.

Uses the archived native cold-ring update-472 state. The first D minibatch is
checked against its saved Adam first moment; the first G minibatch is checked
against the independent exact-RNG field audit. No optimizer or host step runs.
The only numerical derivative is a central difference along the shared G
network's final output-bias axes at fixed D and fixed saved Adam metric.
"""

import argparse
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path

import torch
from torch.func import functional_call

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPGenerator, SimpleMLPDiscriminator
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from particlegan import ParticlePrior
from reports.toy100 import pr84_cold472_field as field
from reports.toy100.functional_b_cap import functional_b_cap


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_SHA = '19ceb38a92ba95ffaccd3aafbda55bbd77917765612666d923e986e5523d6965'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_gzip(path):
    with gzip.open(path, 'rb') as handle:
        return handle.read()


def modules(saved):
    generator = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN,
                                   mode_hold.N_HIDDEN, 2)
    critic = SimpleMLPDiscriminator(2, mode_hold.HIDDEN,
                                    mode_hold.N_HIDDEN, mode_hold.FOURIER)
    prior = ParticlePrior(*saved['prior']['z'].shape)
    generator.load_state_dict({name.removeprefix('model.'): value
                               for name, value in saved['generator'].items()})
    critic.load_state_dict({name.removeprefix('model.'): value
                            for name, value in saved['critic'].items()})
    prior.load_state_dict(saved['prior'])
    return generator, critic, prior


def saved_metric(saved):
    result = []
    for group in saved['param_groups']:
        for index in group['params']:
            state = saved['state'][index]
            denominator = (state['exp_avg_sq'] /
                           (1-group['betas'][1]**float(state['step']))).sqrt()+group['eps']
            result.append(group['lr']/denominator.double())
    return result


def penalty_parity(regularizer, critic, real, fake, step):
    params = tuple(critic.parameters())
    native = regularizer(critic, real, fake.detach(), step=step)
    functional = functional_b_cap(regularizer, critic, real, fake.detach(), step)
    mapping = dict(critic.named_parameters())
    closure = lambda x: functional_call(critic, mapping, (x,))
    stateless = functional_b_cap(regularizer, closure,
                                  real, fake.detach(), step)
    native_d = torch.autograd.grad(native, params, allow_unused=True)
    functional_d = torch.autograd.grad(functional, params, allow_unused=True)
    stateless_d = torch.autograd.grad(stateless, params, allow_unused=True)
    if not torch.equal(native.detach(), functional.detach()):
        raise AssertionError('native and functional b_cap values differ')
    if not all((a is None and b is None) or
               (a is not None and b is not None and torch.equal(a, b))
               for a, b in zip(native_d, functional_d)):
        raise AssertionError('native and functional b_cap D gradients differ')
    if not torch.equal(native.detach(), stateless.detach()) or not all(
        (a is None and b is None) or
        (a is not None and b is not None and torch.equal(a, b))
        for a, b in zip(native_d, stateless_d)
    ):
        raise AssertionError('functional_call b_cap value or D gradient differs')
    return dict(value=float(native.detach()),
                d_gradient_l2=sum(float(x.double().square().sum())
                                  for x in native_d if x is not None)**.5,
                bitwise_value=True, bitwise_d_gradient=True,
                stateless_functional_call_bitwise=True)


def fixed_metric_mixed(regularizer, critic, generator, prior, real,
                       indices, noise, sigma, metric, step):
    # Finite differencing a small cap-gradient contraction in float32 loses
    # several digits to cancellation. The exact native float32 parity is
    # checked separately above; this local derivative check uses the same
    # captured weights, minibatch and saved metric in float64.
    critic = deepcopy(critic).double()
    generator = deepcopy(generator).double()
    prior = deepcopy(prior).double()
    real = real.double()
    noise = noise.double()
    dparams = tuple(critic.parameters())
    network = list(generator.parameters())
    output_bias = network[-1]
    if tuple(output_bias.shape) != (2,):
        raise AssertionError('the shared G final output bias is not two-dimensional')
    if len(metric) != len(dparams):
        raise AssertionError('saved D metric dimension differs from critic')
    latent = prior.z[indices]

    def fake():
        return generator(latent) + sigma * noise

    base_fake = fake()
    cap = functional_b_cap(regularizer, critic, real, base_fake, step)
    base_d = torch.autograd.grad(cap, dparams, create_graph=True,
                                 allow_unused=True)
    fixed_probe = [None if value is None else value.detach() for value in base_d]

    def contraction(values):
        return sum((p.detach().to(v.dtype) * v * u).sum()
                   for p, v, u in zip(metric, values, fixed_probe)
                   if v is not None and u is not None)

    scalar = contraction(base_d)
    analytical = torch.autograd.grad(scalar, output_bias)[0].detach().clone()
    if not bool(torch.isfinite(analytical).all()):
        raise AssertionError('nonfinite analytic mixed derivative')
    # Native GradRegularizer explicitly disconnects fake = G(z). Its D
    # gradient has no autograd path back to the final G output bias.
    native_cap = regularizer(critic, real, base_fake, step=step)
    native_d = torch.autograd.grad(native_cap, dparams, create_graph=True,
                                   allow_unused=True)
    native_scalar = contraction(native_d)
    native_mixed = torch.autograd.grad(native_scalar, output_bias,
                                       allow_unused=True)[0]
    if native_mixed is not None and not torch.equal(native_mixed, torch.zeros_like(native_mixed)):
        raise AssertionError('frozen cap unexpectedly preserves fake-to-D graph')

    original = output_bias.detach().clone()
    h = 1e-5
    finite = []
    try:
        for axis in range(2):
            values = []
            for sign in (1., -1.):
                with torch.no_grad():
                    output_bias.copy_(original)
                    output_bias[axis] += sign*h
                shifted = fake()
                shifted_cap = functional_b_cap(regularizer, critic, real, shifted, step)
                shifted_d = torch.autograd.grad(shifted_cap, dparams,
                                                allow_unused=True)
                values.append(float(contraction(shifted_d).detach()))
            finite.append((values[0]-values[1])/(2*h))
    finally:
        with torch.no_grad():
            output_bias.copy_(original)
    finite = torch.tensor(finite, dtype=analytical.dtype)
    error = (analytical-finite).abs()
    tolerance = .01*finite.abs().clamp_min(1e-5) + 1e-5
    if not bool((error <= tolerance).all()):
        raise AssertionError(f'host mixed derivative FD mismatch: analytical={analytical} '
                             f'finite={finite}, tolerance={tolerance}')
    return dict(metric='saved_post_adam_D_lr_over_denominator_fixed',
                derivative_precision='float64 copies of captured float32 modules and minibatch',
                probe='weighted dot of cap D gradient with detached base cap D gradient',
                shared_g_parameter='final_output_bias', finite_difference_h=h,
                analytical=analytical.tolist(), finite_difference=finite.tolist(),
                absolute_error=error.tolist(), allowed_error=tolerance.tolist(),
                native_detached_cap_mixed_is_zero=True,
                functional_mixed_nonzero=bool(torch.any(analytical != 0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--field', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raw = load_gzip(args.capture / 'failed-fit.pt.gz')
    if digest(raw) != CAPTURE_SHA:
        raise RuntimeError('frozen failed-fit capture bytes changed')
    payload = torch.load(BytesIO(raw), weights_only=True)
    if payload['host_update'] != 472:
        raise RuntimeError('unexpected saved update')
    config_raw = (args.original / 'config.json').read_bytes()
    recipe, _, _ = declared_recipe(json.loads(config_raw))
    regularizer = recipe.make_gradient_penalty()
    post = payload['post_accepted_d']
    pre = payload['pre_step']
    if post['noise']['input_sigma'] != 0 or post['noise']['output_sigma'] != .029:
        raise RuntimeError('saved native noise law changed')
    if post['rng']['output'] is not None:
        raise RuntimeError('saved native output noise is not the global stream')
    source = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in (
        ROOT / 'reports/toy100/functional_b_cap.py',
        ROOT / 'reports/toy100/functional_b_cap_audit.py',
        ROOT / 'particlegan/grad_regularizers.py',
        ROOT / 'benchmarks/locked_shared/mode_hold.py',
        ROOT / 'reports/toy100/pr84_critic_relaxation.py',
        ROOT / 'reports/toy100/pr84_cold472_field.py')}

    with torch.random.fork_rng(devices=[]):
        generator, accepted_d, prior = modules(post)
        best_d = modules(post)[1]
        best_d.load_state_dict(deepcopy(payload['best']))
        first_d = {name: payload['bank'][name][:mode_hold.BATCH]
                   for name in ('real', 'fake')}
        first_pre_d = modules(pre)[1]
        gan = recipe.make_loss()
        first_loss = (gan.d_loss(first_pre_d(first_d['real']),
                                 first_pre_d(first_d['fake']))
                      + regularizer(first_pre_d, first_d['real'],
                                    first_d['fake'], step=472))
        first_grad = torch.autograd.grad(first_loss, tuple(first_pre_d.parameters()))
        recorded = [post['optimizer_d']['state'][index]['exp_avg']
                    for group in post['optimizer_d']['param_groups']
                    for index in group['params']]
        if not all(torch.equal(a, b) for a, b in zip(first_grad, recorded)):
            raise AssertionError('first D bank does not reproduce saved native Adam gradient')
        parity = {}
        for name, critic in (('accepted_d', accepted_d), ('best_finite_d', best_d)):
            parity[name] = penalty_parity(regularizer, critic,
                                          first_d['real'], first_d['fake'], 472)

        field_receipt = json.loads(load_gzip(args.field / 'diagnosis.json.gz'))
        g_row = field.common_g_batches(post, generator, prior, mode_hold)[0]
        if g_row['sha256'] != field_receipt['batch_sha256'][0]:
            raise AssertionError('first native G minibatch differs from archived field audit')
        metric = saved_metric(post['optimizer_d'])
        mixed = {}
        for name, critic in (('accepted_d', accepted_d), ('best_finite_d', best_d)):
            mixed[name] = fixed_metric_mixed(regularizer, critic, generator, prior,
                                             g_row['real'], g_row['indices'], g_row['noise'],
                                             post['noise']['output_sigma'], metric, 472)
    receipt = dict(scope='read-only native update472 D-bank and G-bank b_cap derivative audit',
                   host_update=472, capture_sha256=digest(raw),
                   config_sha256=digest(config_raw), source=source,
                   first_d_gradient_bitwise_equal=True,
                   first_g_batch_sha256=g_row['sha256'],
                   penalty_parity=parity, mixed_derivative=mixed,
                   runtime=dict(torch=torch.__version__, threads=torch.get_num_threads(),
                                cpu_capability=torch.backends.cpu.get_cpu_capability()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
