import math
import torch

from experiments.measure_cifar_particle_variation import pair_metrics


def test_known_distinct_triplet():
    pixels = torch.tensor([[[0], [1], [2]]], dtype=torch.uint8)
    features = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
    result = pair_metrics(pixels, features)
    assert result['pair_pixel_mse'].item() == 2
    assert math.isclose(result['pair_feature_cosine'].item(), 2/3, abs_tol=1e-7)
    assert result['identical_pair_fraction'].item() == 0
    assert result['unique_draws'].item() == 3


def test_duplicates_counted_per_input_not_across_inputs():
    pixels = torch.tensor([[[5], [5], [5]], [[0], [0], [1]]], dtype=torch.uint8)
    features = torch.tensor([[[1., 0.]] * 3, [[0., 1.]] * 3])
    result = pair_metrics(pixels, features)
    torch.testing.assert_close(result['unique_draws'], torch.tensor([1., 2.]))
    torch.testing.assert_close(result['identical_pair_fraction'], torch.tensor([1., 1/3]))
    torch.testing.assert_close(result['pair_pixel_mse'], torch.tensor([0., 2/3]))
    torch.testing.assert_close(result['pair_feature_cosine'], torch.zeros(2))
