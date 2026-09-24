"""Saved-state finite-bank variance and game-geometry diagnosis.

No host continuation or optimizer candidate is run.  Native float32 controls
reproduce the captured D field and G update.  The local map is separately
declared: float64 alternating gradients with the captured fixed Adam metric,
the original own-field bounds, and sixteen completely frozen native-sized
banks.  It is not the Jacobian of Adam's augmented moment state.
"""

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100 import pr84_critic_relaxation as fit
from reports.toy100.pr84_critic_refinement_capture import _sha


STEP = 1325
BANKS = 16
FRACTIONS = (1e-4, 5e-5)
SOURCE_FILES = (
    'reports/toy100/pr84_finite_bank_vr_diagnostic.py',
    'reports/toy100/pr84_critic_relaxation.py',
    'reports/toy100/alternating_curvature_scratch.py',
    'reports/toy100/coverage_fixed_eval.py',
    'reports/toy100/pr84_critic_refinement_capture.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
    'particlegan/gan_loss.py', 'particlegan/grad_regularizers.py',
    'configs/toy100/constraints_simple_regularization.json',
)


def flat(values):
    return torch.cat([value.detach().double().flatten() for value in values])


def control_variate(current, reference, reference_mean):
    """Uniform finite-sum SVRG estimator, not a population-gradient claim."""
    return current - reference + reference_mean


def variance_receipt(rows, metric):
    scaled = rows * metric.sqrt()
    mean = scaled.mean(0)
    variance = (scaled-mean).square().sum(1).mean()
    return dict(mean_metric_norm=float(mean.norm()),
                metric_variance_trace=float(variance),
                mean_squared_metric_norm=float(scaled.square().sum(1).mean()))


def g_banks(saved, generator, prior):
    stream = torch.Generator().set_state(saved['rng']['data'])
    rows = []
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.set_rng_state(saved['rng']['torch'])
        for _ in range(BANKS):
            latent, indices = prior.sample(mode_hold.BATCH, generator=stream)
            noise = torch.randn_like(generator(latent))
            real = mode_hold.sample_ring(mode_hold.ring_means(), mode_hold.BATCH,
                                         mode_hold.SIGMA, stream)
            rows.append(dict(real=real, indices=indices, noise=noise,
                             sigma=saved['noise']['output_sigma']))
    return rows


def joined_g(rows):
    return {key: torch.cat([row[key] for row in rows])
            for key in ('real', 'indices', 'noise')} | {'sigma': rows[0]['sigma']}


class FixedBankField:
    def __init__(self, phases, d_rows, g_rows, recipe):
        self.generator, self.critic, self.prior = fit.modules(phases['pre_step'])
        for module in (self.generator, self.critic, self.prior):
            module.double()
        self.d_params = list(self.critic.parameters())
        self.g_params = list(self.generator.parameters()) + list(self.prior.parameters())
        self.parameters = self.d_params + self.g_params
        self.d_size = sum(p.numel() for p in self.d_params)
        self.base = flat(self.parameters)
        self.metric = flat(fit.saved_metric(phases['post_accepted_d']['optimizer_d'])
                           + fit.saved_metric(phases['post_bounded_g']['optimizer_g']))
        if not torch.isfinite(self.metric).all() or not (self.metric > 0).all():
            raise FloatingPointError('invalid captured metric')
        def cast(row):
            return {key: value.double() if isinstance(value, torch.Tensor)
                    and value.is_floating_point() else value for key, value in row.items()}
        self.rows = {'d': [cast(row) for row in d_rows],
                     'g': [cast(row) for row in g_rows]}
        self.gan, self.cap = recipe.make_loss(), recipe.make_gradient_penalty()
        self.sigma = phases['pre_step']['noise']['output_sigma']
        self.calls = {'d': 0, 'g': 0}

    def assign(self, point):
        offset = 0
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.copy_(point[offset:offset+parameter.numel()].reshape_as(parameter))
                offset += parameter.numel()
        assert offset == len(point)

    def field(self, point, role, *, individual=False):
        self.assign(point)
        params = self.d_params if role == 'd' else self.g_params
        gradients = []
        for row in self.rows[role]:
            fake = self.generator(self.prior.z[row['indices']]) + self.sigma*row['noise']
            if role == 'd':
                fake = fake.detach()
                loss = self.gan.d_loss(self.critic(row['real']), self.critic(fake))
                loss = loss + self.cap(self.critic, row['real'], fake, step=STEP)
            else:
                loss = self.gan.g_loss(fit.smooth(self.critic, fake),
                                      fit.smooth(self.critic, row['real']))
            values = torch.autograd.grad(loss, params)
            gradients.append(flat(values))
            self.calls[role] += 1
        result = torch.stack(gradients)
        if not torch.isfinite(result).all():
            raise FloatingPointError('nonfinite fixed-bank field')
        return result if individual else result.mean(0)

    def joint(self, point, *, individual=False):
        dim = 1 if individual else 0
        return torch.cat([self.field(point, role, individual=individual)
                          for role in ('d', 'g')], dim=dim)

    def alternating_map(self, point):
        """One fixed-metric D-then-G map, including original own-field bounds."""
        value = point.clone()
        receipt = {}
        for role, sl, bound in (('d', slice(0, self.d_size), 3.),
                                ('g', slice(self.d_size, None), .25)):
            first = self.field(value, role)
            p = self.metric[sl]
            proposal = value.clone()
            proposal[sl] -= p*first
            second = self.field(proposal, role)
            displacement = proposal[sl]-value[sl]
            denominator = (displacement.square()/p).sum()
            rho = float(((p*(second-first).square()).sum()/denominator).sqrt()) \
                if denominator else 0.
            factor = min(1., bound/rho) if rho else 1.
            value[sl] += factor*displacement
            receipt[role] = dict(rho=rho, factor=factor)
        if not torch.isfinite(value).all():
            raise FloatingPointError('nonfinite deterministic map')
        return value, receipt


def local_geometry(field, delta):
    base = field.base
    metric = field.metric
    dimensions = {'d_only': slice(0, field.d_size),
                  'g_only': slice(field.d_size, None),
                  'joint': slice(None)}
    rows = []
    for name, sl in dimensions.items():
        direction = torch.zeros_like(delta)
        direction[sl] = delta[sl]
        norm2 = float((direction.square()/metric).sum())
        if norm2 == 0:
            raise ValueError('declared captured direction is zero')
        scales = []
        for h in FRACTIONS:
            plus, minus = base+h*direction, base-h*direction
            fplus, fminus = field.joint(plus), field.joint(minus)
            derivative = (fplus-fminus)/(2*h)
            mapped_plus, rp = field.alternating_map(plus)
            mapped_minus, rm = field.alternating_map(minus)
            tangent = (mapped_plus-mapped_minus)/(2*h)
            achieved = (plus-minus)/(2*h)
            scales.append(dict(fraction_of_captured_update=h,
                achieved_direction_relative_error=float((achieved-direction).norm()/direction.norm()),
                metric_symmetric_rayleigh=float(direction@derivative)/norm2,
                alternating_map_directional_amplification=float(
                    (tangent.square()/metric).sum().sqrt())/(norm2**.5),
                plus_bound=rp, minus_bound=rm))
        keys = ('metric_symmetric_rayleigh', 'alternating_map_directional_amplification')
        agreement = {key: abs(scales[0][key]-scales[1][key])/
                     max(abs(scales[0][key]), abs(scales[1][key]), 1e-12) for key in keys}
        row = dict(direction=name, metric_input_norm=norm2**.5, scales=scales,
                   relative_scale_disagreement=agreement,
                   two_scale_resolved=all(value <= .05 for value in agreement.values()))
        rows.append(row)
        print(json.dumps(dict(event='LOCAL_DIRECTION_DONE', **row)), flush=True)
    return rows


def run(phases, recipe):
    pre, saved, final = (phases[key] for key in
                        ('pre_step', 'post_accepted_d', 'post_bounded_g'))
    if pre['noise']['input_sigma'] != 0 or pre['noise']['output_sigma'] != .029 \
            or pre['rng']['output'] is not None:
        raise ValueError('this one-state diagnostic requires the frozen late noise configuration')
    generator, critic, prior = fit.modules(saved)
    a, b, _, _, actual_g = fit.banks(pre, saved, generator, prior)
    d_rows = a+b
    grows = g_banks(saved, generator, prior)
    if _sha(grows[0]) != _sha(actual_g):
        raise RuntimeError('first G draw differs from native replay')
    gan, cap = recipe.make_loss(), recipe.make_gradient_penalty()
    critic.load_state_dict(fit.unwrapped(pre['critic']))
    actual_d = fit.gradients(fit.d_loss(critic, d_rows[0], gan, cap, STEP)[0], critic)
    expected_d = [saved['optimizer_d']['state'][i]['exp_avg']
                  for group in saved['optimizer_d']['param_groups'] for i in group['params']]
    if not all(torch.equal(a, b) for a, b in zip(actual_d, expected_d)):
        raise RuntimeError('native D gradient differs')
    critic.load_state_dict(fit.unwrapped(saved['critic']))
    native, accepted, _ = fit.g_proposal(saved, critic, grows[0], gan, STEP)
    expected_g = dict(generator=fit.unwrapped(final['generator']), prior=final['prior'],
                      optimizer=final['optimizer_g'])
    if _sha(accepted) != _sha(expected_g):
        raise RuntimeError('native bounded G/Adam update differs')
    averaged, _, _ = fit.g_proposal(saved, critic, joined_g(grows), gan, STEP)
    print(json.dumps(dict(event='NATIVE_AND_MEAN_G_DONE', native=native['grade_after'],
        full_bank=averaged['grade_after'], exact_d_gradient=True, exact_g_and_moments=True)), flush=True)

    field = FixedBankField(phases, d_rows, grows, recipe)
    end_g, end_d, end_prior = fit.modules(final)
    endpoint = flat(list(end_d.parameters())+list(end_g.parameters())+list(end_prior.parameters()))
    delta = endpoint-field.base
    first = field.joint(field.base, individual=True)
    second = field.joint(endpoint, individual=True)
    reference_mean = first.mean(0)
    at_reference = control_variate(first, first, reference_mean)
    at_endpoint = control_variate(second, first, reference_mean)
    exact = torch.equal(at_reference, reference_mean.expand_as(first))
    mean_error = float((at_endpoint.mean(0)-second.mean(0)).abs().max())
    if not exact or mean_error > 1e-12:
        raise RuntimeError('finite-sum control-variate identity failed')
    vr = dict(at_reference_all_components_exact=True,
              endpoint_estimator_mean_max_absolute_error=mean_error,
              native_reference=variance_receipt(first, field.metric),
              controlled_reference=variance_receipt(at_reference, field.metric),
              native_endpoint=variance_receipt(second, field.metric),
              controlled_endpoint=variance_receipt(at_endpoint, field.metric))
    vr['endpoint_variance_ratio'] = (vr['controlled_endpoint']['metric_variance_trace'] /
                                     vr['native_endpoint']['metric_variance_trace'])
    geometry = local_geometry(field, delta)
    field.assign(field.base)
    return dict(native_controls=dict(d_gradient_exact=True, g_proposal_and_moments_exact=True,
                                    first_g_batch_exact=True),
        fixed_native_critic_g_proposals=dict(single_native=native, full_16_bank=averaged),
        finite_sum_control_variate=vr, local_geometry=geometry,
        float64_role_gradient_evaluations=field.calls,
        raw_bank_hashes=dict(d=_sha(d_rows), g=_sha(grows))), \
        dict(d_rows=d_rows, g_rows=grows, base_field=first, endpoint_field=second,
             fixed_metric=field.metric, captured_delta=delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    declaration = dict(method='saved1325_finite_bank_variance_geometry_diagnostic', step=STEP,
        native_sized_banks_per_role=BANKS, pairs_per_bank=mode_hold.BATCH,
        scopes=['native float32 replay and mean-G proposal',
                'float64 fixed-metric alternating bounded map; no Adam moment derivatives'],
        finite_objective='freeze all real values, prior indices and output-noise values per component',
        width=fit.WIDTH, bounds=dict(d=3., g=.25), finite_difference_fractions=FRACTIONS,
        exact_oracle_scope='declared finite bank only; not the online Gaussian expectation',
        no_training=True, shared_gate_eligible=False, quality_selection=False,
        states_sha256=hashlib.sha256(args.states.read_bytes()).hexdigest(),
        source={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES})
    for name in SOURCE_FILES:
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED', **declaration)), flush=True)
    if args.states.suffix == '.gz':
        with gzip.open(args.states, 'rb') as stream:
            bundle = torch.load(stream, weights_only=True)
        if bundle['parent_states_sha256'] != fit.STATE_SHA:
            raise RuntimeError('wrong original full capture')
        phases = bundle['states'][STEP]
    else:
        if declaration['states_sha256'] != fit.STATE_SHA:
            raise RuntimeError('wrong original full capture')
        phases = torch.load(args.states, weights_only=True)[STEP]
    if 'rng' not in phases['post_accepted_d']:
        raise ValueError('input subset omits post-D RNG; use full capture or complete1325 subset')
    before = _sha(phases)
    rng = torch.get_rng_state().clone()
    recipe, _, _ = declared_recipe(json.loads((ROOT/SOURCE_FILES[-1]).read_text()))
    if recipe.prior_reg != 0:
        raise ValueError('declared game has no prior regularizer')
    with torch.random.fork_rng(devices=[]):
        result, tensors = run(phases, recipe)
    if before != _sha(phases) or not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError('read-only diagnostic changed input state or global RNG')
    result.update(declaration=declaration, input_state_unchanged=True, global_rng_unchanged=True,
                  status='DIAGNOSTIC_COMPLETE', shared_gate_eligible=False)
    torch.save(tensors, args.output/'tensors.pt')
    (args.output/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE', vr=result['finite_sum_control_variate'],
                         gradient_calls=result['float64_role_gradient_evaluations'])), flush=True)


if __name__ == '__main__':
    main()
