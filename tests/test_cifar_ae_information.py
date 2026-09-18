"""CPU tests of mathematical definitions and train/validation/test separation."""
import math
import torch

from experiments.probe_cifar_ae_information import (
    conditional_information, density_coverage, fit_decoder, score_decoder,
    variance_decomposition,
)


def test_density_coverage_matches_brute_force_and_strict_boundaries():
    # Real radii are 2, 2, 3. At k1 fake0 is inside only first ball;
    # fake2 lies exactly on first radius and inside second, fake100 nowhere.
    real = torch.tensor([[0.], [2.], [5.]], dtype=torch.float64)
    fake = torch.tensor([[0.], [2.], [100.]], dtype=torch.float64)
    result = density_coverage(real, fake, k=1, chunk=1)
    assert result['density'] == 2/3 and result['coverage'] == 2/3
    stream = torch.Generator().manual_seed(21)
    real = torch.randn(19, 7, generator=stream, dtype=torch.float64)
    fake = torch.randn(13, 7, generator=stream, dtype=torch.float64)
    rr = torch.cdist(real, real)
    rr.fill_diagonal_(float('inf'))
    radius = rr.kthvalue(5, dim=1).values
    inside = torch.cdist(real, fake) < radius[:, None]
    for chunk in (1, 4, 100):
        result = density_coverage(real, fake, k=5, chunk=chunk)
        assert abs(result['density']-float(inside.sum())/(5*len(fake))) < 1e-12
        assert abs(result['coverage']-float(inside.any(1).double().mean())) < 1e-12


def test_nested_variance_adds_up_and_distinguishes_noise_from_siblings():
    parents = torch.tensor([-3., 3.])[:, None, None, None]
    siblings = torch.tensor([-2., 2.])[None, :, None, None]
    noise = torch.tensor([-1., 1.])[None, None, :, None]
    result = variance_decomposition(parents+siblings+noise)
    assert abs(result['between_parents_fraction']-9/14) < 1e-12
    assert abs(result['between_siblings_fraction']-4/14) < 1e-12
    assert abs(result['within_child_fraction']-1/14) < 1e-12
    zero = variance_decomposition(torch.zeros(2, 1, 3, 4))
    assert zero['total_variance_trace'] == 0


def test_decodable_bits_separable_clusters_and_null_controls():
    torch.set_num_threads(2)
    stream = torch.Generator().manual_seed(113)
    centers = torch.eye(4, dtype=torch.float64)*5
    def sample(n):
        return centers[None, :, None]+torch.randn(4, 4, n, 4, generator=stream, dtype=torch.float64)*.3
    result = conditional_information(sample(64), sample(32), sample(128))
    assert result['observed']['decodable_bits'] > 1.95
    assert abs(result['identical_clones']['decodable_bits']) < 1e-10
    assert result['shuffled_labels']['decodable_bits'] < .05
    assert result['available_sibling_bits'] == 2


def test_test_data_cannot_change_fitted_decoder_and_negative_bits_are_retained():
    train = torch.tensor([[[-2.], [-1.8]], [[2.], [1.8]]])
    validation = train.clone()
    model = fit_decoder(train, validation)
    selected = (model['temperature'], model['shrinkage'], model['validation_ce_nats'])
    good = score_decoder(model, train)
    bad = score_decoder(model, train.flip(0))
    assert good['decodable_bits'] > .9 and bad['decodable_bits'] < 0
    assert selected == (model['temperature'], model['shrinkage'], model['validation_ce_nats'])


def test_single_child_has_zero_information_and_parent_variance_survives():
    x = torch.arange(24).reshape(2, 1, 3, 4).float()
    result = conditional_information(x, x+1, x+2)
    for name in ('observed', 'shuffled_labels', 'identical_clones'):
        assert result[name]['decodable_bits'] == 0
        assert result[name]['test_accuracy'] == 1
    variance = variance_decomposition(x)
    assert variance['between_siblings_fraction'] == 0
    assert variance['between_parents_fraction'] > 0
    assert variance['within_child_fraction'] > 0
