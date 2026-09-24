"""No-RNG scalar check of a transported critic gradient and raw Adam metric.

This is an algebraic mechanism check, not GAN training or an IID simulation.
The fixed ++-- sequence illustrates a possible bounded realization while the
expectation identities in the receipt use independent symmetric noise.
"""

import json
from pathlib import Path


BETA2 = .999
ETA = .00425
EPS = 1e-8
SHOCK = .35
NOISE = (1., 1., -1., -1.)
BASE_COUNT = 10_000


def transported(current, previous, estimate, bank_count):
    """STORM-style same-bank difference with statistical gain 1/count."""
    return current + (1. - 1. / bank_count) * (estimate - previous)


def adam_denominator(moment, step):
    return (moment / (1. - BETA2 ** step)) ** .5 + EPS


def own_bound_multiplier(raw_normalized_curvature, bound):
    """One-dimensional convex own-field update x'=(1-min(h,bound))*x."""
    return 1. - min(raw_normalized_curvature, bound)


def run():
    estimate = raw_second = tracked_second = total_noise = 0.
    resting = None
    for n in range(1, BASE_COUNT + 1):
        xi = NOISE[(n - 1) % len(NOISE)]
        total_noise += xi
        estimate = transported(xi, xi, estimate, n)  # x remains zero.
        raw_second = BETA2 * raw_second + (1. - BETA2) * xi * xi
        tracked_second = BETA2 * tracked_second + (1. - BETA2) * estimate * estimate
        if n == BASE_COUNT - 1:
            assert abs(estimate - total_noise / n) < 1e-13
            raw_den = adam_denominator(raw_second, n)
            tracked_den = adam_denominator(tracked_second, n)
            resting = dict(bank_count=n, tracked_field=estimate,
                           raw_adam_denominator=raw_den,
                           tracked_adam_denominator=tracked_den,
                           raw_metric_step_magnitude=ETA * abs(estimate) / raw_den,
                           tracked_metric_step_magnitude=ETA * abs(estimate) / tracked_den)

    assert abs(total_noise) < 1e-13
    n = BASE_COUNT + 1
    xi = NOISE[(n - 1) % len(NOISE)]
    old = xi  # evaluate the previous x=0 on the same new bank.
    current = SHOCK + xi
    estimate = transported(current, old, estimate, n)
    raw_second = BETA2 * raw_second + (1. - BETA2) * current * current
    tracked_second = BETA2 * tracked_second + (1. - BETA2) * estimate * estimate
    assert abs(estimate - (SHOCK + xi / n)) < 1e-12
    raw_den = adam_denominator(raw_second, n)
    tracked_den = adam_denominator(tracked_second, n)
    shock = dict(bank_count=n, model_error=SHOCK, tracked_field=estimate,
                 raw_adam_denominator=raw_den,
                 tracked_adam_denominator=tracked_den,
                 raw_metric_step=ETA * estimate / raw_den,
                 tracked_metric_step=ETA * estimate / tracked_den,
                 d_bound_rho_raw=ETA / raw_den,
                 d_bound_rho_tracked=ETA / tracked_den,
                 d_bound_factor_raw=min(1., 3. / (ETA / raw_den)),
                 d_bound_factor_tracked=min(1., 3. / (ETA / tracked_den)))

    # For iid xi=+-1, at a fixed x=0 the tracker is the sample mean.
    # The raw squared gradient has expectation 1. For a continuously varying
    # x, this finite-sample identity does not assert vanishing tracker noise.
    expectation = dict(field='F(x)=x, g(x,xi)=x+xi, iid xi=+-1 equiprobable',
                       tracker_at_fixed_x='x+mean(xi_1,...,xi_n)',
                       tracker_noise_variance='1/n',
                       raw_second_moment_at_true_root=1.,
                       response_to_any_model_delta='exact additive delta on same bank')

    # The host's D bound 3 does not ensure contraction for a scalar convex
    # field if the unbounded normalized own-curvature h reaches 3. The G bound
    # .25 does. A bilinear D-then-G game has zero own curvature, so neither
    # bound acts and its alternating-map determinant is exactly one.
    h = 3.
    toy_geometry = dict(scalar_h=h,
        scalar_d_bound_multiplier=own_bound_multiplier(h, 3.),
        scalar_g_bound_multiplier=own_bound_multiplier(h, .25),
        bilinear_d_then_g_determinant=1.,
        bilinear_d_then_g_trace=2. - ETA * ETA,
        bilinear_d_then_g_spectral_radius=1.)
    assert toy_geometry['scalar_d_bound_multiplier'] == -2.
    assert toy_geometry['scalar_g_bound_multiplier'] == .75
    assert abs(toy_geometry['bilinear_d_then_g_trace']) < 2.
    assert resting['raw_metric_step_magnitude'] < 1e-5
    assert resting['tracked_metric_step_magnitude'] > 1e-3
    assert shock['d_bound_factor_raw'] == shock['d_bound_factor_tracked'] == 1.
    return dict(scope='deterministic algebraic illustration; no GAN update, no RNG',
                beta1=0., beta2=BETA2, epsilon=EPS, nominal_rate=ETA,
                statistical_gain='1 / number of distinct observed banks',
                expectation=expectation, resting_realization=resting,
                model_error_shock=shock, geometry=toy_geometry)


if __name__ == '__main__':
    receipt = run()
    print(json.dumps(receipt, indent=2))
