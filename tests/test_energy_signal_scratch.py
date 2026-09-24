"""Small numerical checks for the data-space proposal signal."""

from unittest.mock import patch

import pytest
import torch

from reports.toy100.energy_signal_scratch import EnergySignalRecorder, energy_score


def _case(gradient, target=0.):
    network = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        network.weight.fill_(1.)
    prior = torch.nn.Module()
    prior.z = torch.nn.Parameter(torch.ones(1, 1))
    optimizer = torch.optim.Adam([
        dict(params=list(network.parameters()), lr=.1, _comparison_prior=False),
        dict(params=[prior.z], lr=.1, _comparison_prior=True),
    ], betas=(0., .9))
    for parameter in (network.weight, prior.z):
        parameter.grad = torch.full_like(parameter, gradient)
    host = dict(host_kind="benchmarks.locked_shared.mode_hold",
                generator=network, prior=prior, opt_g=optimizer,
                opt_d=object(), latent=prior.z[torch.zeros(4, dtype=torch.long)],
                real_g=torch.full((4, 1), target),
                _=torch.zeros(4, dtype=torch.long))
    return host


@pytest.mark.parametrize("gradient,accepted", [(1., True), (-1., False)])
def test_energy_signal_keeps_only_data_improving_joint_proposals(gradient, accepted):
    host = _case(gradient)
    optimizer = host["opt_g"]
    network, prior = host["generator"], host["prior"]
    rng = torch.get_rng_state().clone()
    recorder = EnergySignalRecorder()
    with patch("reports.toy100.energy_signal_scratch._host_locals", return_value=host):
        recorder.step(optimizer, torch.optim.Adam.step)
    row = recorder.receipt["rows"][0]
    assert row["accepted_by_signal"] is accepted
    assert row["proposal_kept"] is accepted
    assert all(after < before for before, after in zip(row["before"], row["after"])) is accepted
    expected = .9 if accepted else 1.
    assert float(network.weight.detach()) == pytest.approx(expected)
    assert float(prior.z.detach()) == pytest.approx(expected)
    assert all(optimizer.state[p]["step"] == 1 for p in (network.weight, prior.z))
    assert torch.equal(torch.get_rng_state(), rng)


def test_energy_score_uses_distribution_not_target_centers():
    real = torch.zeros(4, 1)
    near = torch.full((4, 1), .5)
    far = torch.full((4, 1), 1.)
    assert energy_score(real, near) < energy_score(real, far)
    with pytest.raises(ValueError):
        energy_score(real, far[:2])


def test_backtracking_keeps_largest_improving_fraction_and_full_moments():
    host = _case(1., target=.95)
    optimizer = host["opt_g"]
    recorder = EnergySignalRecorder(backtracking=True)
    with patch("reports.toy100.energy_signal_scratch._host_locals", return_value=host):
        recorder.step(optimizer, torch.optim.Adam.step)
    row = recorder.receipt["rows"][0]
    assert not row["accepted_by_signal"]
    assert row["accepted_scale"] == .5
    assert row["fractional_diagnostic"][0]["improves_both"]
    assert float(host["generator"].weight.detach()) == pytest.approx(.95)
    assert float(host["prior"].z.detach()) == pytest.approx(.95)
    assert all(optimizer.state[p]["step"] == 1 for group in optimizer.param_groups
               for p in group["params"])
