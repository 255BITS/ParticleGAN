import itertools

import torch

from reports.toy100.chamfer_discrete_reallocation import (
    _cost, best_real_relocation, greedy_real_reallocate,
)


def test_global_single_move_matches_brute_force_and_does_not_draw_rng():
    real = torch.tensor([[-2., 0.], [-1.9, .1], [0., 2.], [2., 0.]])
    points = torch.tensor([[-1.95, .02], [-1.90, -.02], [2.02, .01]])
    original_points = points.clone()
    rng = torch.random.get_rng_state().clone()
    choice = best_real_relocation(real, points)
    brute = []
    for donor, sample in itertools.product(range(len(points)), range(len(real))):
        trial = points.clone()
        trial[donor] = real[sample]
        brute.append((_cost(real, trial), donor, sample))
    expected = min(brute)
    assert (choice["donor"], choice["real_sample"]) == expected[1:]
    assert abs(choice["objective_if_relocated"] - expected[0]) < 1e-12
    assert torch.equal(points, original_points)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_greedy_reallocation_lowers_same_sampled_objective_without_mutation():
    real = torch.tensor([[-2., 0.], [-1.9, .1], [0., 2.], [2., 0.]])
    points = torch.tensor([[-1.95, .02], [-1.90, -.02], [2.02, .01]])
    original_points = points.clone()
    updated, receipt = greedy_real_reallocate(real, points)
    assert receipt["moves"]
    assert len(receipt["moves"]) <= len(points)
    assert all(row["decrease"] > 0 for row in receipt["moves"])
    assert receipt["final_objective"] < receipt["input_objective"]
    assert abs(_cost(real, updated) - receipt["final_objective"]) < 1e-12
    assert torch.equal(points, original_points)


def test_invalid_nonfinite_or_single_particle_is_rejected():
    real = torch.zeros(2, 2)
    points = torch.zeros(2, 2)
    try:
        best_real_relocation(real, points[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("single particle unexpectedly accepted")
    points[0, 0] = float("nan")
    try:
        best_real_relocation(real, points)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("nonfinite point unexpectedly accepted")
