"""Deterministic population RpGAN counterexample for unequal Gaussian widths.

This is one-dimensional output-coordinate best-response arithmetic, not host
training. The two fake centers are +/-a, each with Gaussian output noise s;
the real target is N(0, t**2). The unrestricted, unregularized Rp logistic
critic is D*=log(p/q_a). G differentiates through its samples with D* frozen.
"""

import json

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq
from scipy.special import expit, logsumexp, ndtr


REAL_STD = 0.07
FAKE_STD = 0.029
HQ_RADIUS = 3 * REAL_STD


def population(a: float, order: int = 160) -> dict:
    nodes, weights = hermgauss(order)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    real = REAL_STD * nodes
    fake = a + FAKE_STD * nodes  # Negative-center term is equal by symmetry.

    def critic(x):
        pieces = np.stack((
            -0.5 * ((x - a) / FAKE_STD) ** 2,
            -0.5 * ((x + a) / FAKE_STD) ** 2,
        ))
        log_q = logsumexp(pieces, axis=0) - np.log(2 * FAKE_STD * np.sqrt(2 * np.pi))
        log_p = -0.5 * (x / REAL_STD) ** 2 - np.log(REAL_STD * np.sqrt(2 * np.pi))
        return log_p - log_q

    def critic_slope(x):
        return (x * (1 / FAKE_STD**2 - 1 / REAL_STD**2)
                - a / FAKE_STD**2 * np.tanh(a * x / FAKE_STD**2))

    real_score = critic(real)
    fake_score = critic(fake)
    pair_weights = weights[:, None] * weights[None, :]
    discriminator_loss = np.sum(
        pair_weights * np.logaddexp(0, fake_score[None, :] - real_score[:, None]))
    real_factor = np.sum(
        weights[:, None] * expit(real_score[:, None] - fake_score[None, :]), axis=0)
    # This is the gradient-descent direction for the positive center a when D
    # is held fixed. The negative center gives the same radial force.
    generator_field = np.sum(weights * real_factor * critic_slope(fake))
    hq_1d = ndtr((HQ_RADIUS - a) / FAKE_STD) - ndtr((-HQ_RADIUS - a) / FAKE_STD)
    return dict(
        separation=a,
        generator_descent_field=float(generator_field),
        discriminator_loss=float(discriminator_loss),
        discriminator_advantage=float(np.log(2) - discriminator_loss),
        fake_score_slope_rms=float(np.sqrt(np.sum(weights * critic_slope(fake) ** 2))),
        hq_1d=float(hq_1d),
    )


def main():
    root_80 = brentq(lambda a: population(a, 80)["generator_descent_field"], .01, .1)
    root_160 = brentq(lambda a: population(a, 160)["generator_descent_field"], .01, .1)
    result = dict(
        model="one_dimensional_two_equal_fake_gaussians_versus_one_real_gaussian",
        real_std=REAL_STD,
        fake_std=FAKE_STD,
        hq_radius=HQ_RADIUS,
        root_order_80=root_80,
        root_order_160=root_160,
        below=population(root_160 - .001),
        at=population(root_160),
        above=population(root_160 + .001),
        scope="population unregularized best-response field; no b_cap, MLP, Adam, or shared G",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
