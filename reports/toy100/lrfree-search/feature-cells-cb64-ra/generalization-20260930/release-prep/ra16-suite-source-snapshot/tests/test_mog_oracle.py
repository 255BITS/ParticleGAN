"""Known-answer checks for the frozen decoded-center oracle."""
import torch

from experiments.analyze_mog_oracle import center_errors


def test_oracle_finds_global_best_and_coordinate_averaged_regret():
    x = torch.tensor([[1., 0.], [9., 2.]])
    decoded = torch.tensor([[0., 0.], [10., 0.], [100., 100.]])
    best, error, selected = center_errors(x, decoded, torch.tensor([1, 0]))
    assert best.tolist() == [0, 1]
    torch.testing.assert_close(error, torch.tensor([.5, 2.5], dtype=torch.float64))
    torch.testing.assert_close(selected, torch.tensor([40.5, 42.5], dtype=torch.float64))
    assert ((selected - error) == 40).all()


def test_different_ids_can_have_zero_regret_and_unrouted_oracle_works():
    x = torch.tensor([[1., 0.]])
    decoded = torch.tensor([[0., 0.], [0., 0.]])
    best, error, selected = center_errors(x, decoded, torch.tensor([1]))
    assert best.item() == 0
    assert torch.equal(error, selected)
    best_only, error_only, no_selection = center_errors(x, decoded)
    assert torch.equal(best, best_only) and torch.equal(error, error_only)
    assert no_selection is None
