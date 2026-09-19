"""Analytic sanity checks for the component-overlap diagnostic."""
import os

import pytest
import torch

from experiments.probe_cifar_ae_overlap import latent_geometry


pytestmark = pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1',
                                reason='explicit CUDA opt-in required')


def test_identical_components_have_one_bit_of_ambiguity():
    r = latent_geometry(torch.zeros(2, 2, device='cuda'), 1., 2, 32768)
    assert abs(r['posterior_entropy_bits']-1.) < 1e-6
    assert abs(r['wrong_nearest_fraction']-.5) < .01
    assert r['different_parent_wrong_count'] == 0
    assert r['parent_posterior_entropy_bits'] == 0


def test_two_gaussians_match_analytic_bayes_error_and_noise_ordering():
    means = torch.tensor([[-1., 0.], [1., 0.]], device='cuda')
    baseline = latent_geometry(means, 1., 1, 32768)
    smaller = latent_geometry(means, .5, 1, 32768)
    # Equal isotropic priors: nearest-center error Phi(-distance/(2*sigma)).
    assert abs(baseline['wrong_nearest_fraction']-.1586553) < .008
    assert abs(smaller['wrong_nearest_fraction']-.0227501) < .004
    assert 0 < smaller['posterior_entropy_bits'] < baseline['posterior_entropy_bits'] < 1
    assert baseline['same_parent_wrong_count'] == 0
    assert baseline['median_nearest_distance'] == 2
