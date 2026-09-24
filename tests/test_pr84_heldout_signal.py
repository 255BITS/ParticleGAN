"""Analytic checks for the fixed-state directional diagnostic."""

import math

import torch

from reports.toy100.pr84_heldout_signal import _energy_fraction, _quality_direction


def test_coherent_energy_removes_finite_batch_noise_floor():
    fixed = [torch.tensor([1., 2.]) for _ in range(16)]
    balanced = [torch.tensor([1., 0.]) if i % 2 else torch.tensor([-1., 0.])
                for i in range(16)]
    assert math.isclose(_energy_fraction(fixed)["bias_corrected_coherent_fraction"], 1.)
    assert _energy_fraction(balanced)["bias_corrected_coherent_fraction"] == 0.


def test_metric_direction_can_disagree_with_raw_direction():
    # Offline quality gradient q=(1,1).  The game gradient g=(-2,1)
    # initially points outward under -g; a positive anisotropic metric can
    # emphasize its inward coordinate.  The diagnostic must report both signs.
    quality = [torch.tensor([1., 1.])]
    game = [torch.tensor([-2., 1.])]
    metric = [torch.tensor([.1, 10.], dtype=torch.double)]
    row = _quality_direction(quality, game, metric, ["network"])["network"]
    assert row["raw_direction_cosine"] > 0
    assert row["adam_metric_direction_cosine"] < 0
