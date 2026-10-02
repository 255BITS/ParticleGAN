"""Discriminating distribution controls for conditional source coverage."""
import numpy as np
import torch

from benchmarks.toy_audit import source_conditional_scoring as scoring
from benchmarks.toy_audit.source_conditional_capture import owned
from benchmarks.toy_audit.source_demos import isolated, state_hash
from lib.denoising_toy import GaussianGrid
from lib.sparse_toy import SparseMixedToy
from lib.sparse_metrics import convergence_bar


def test_exact_law_controls_reject_sparsity_width_symbol_and_posterior_defects():
    torch.set_num_threads(1)
    controls = scoring.calibration(SparseMixedToy, GaussianGrid)["controls"]
    assert len(controls) == 17
    assert all(control["passed"] == control["expected_pass"] for control in controls)
    for control in controls:
        if control["family"] == "sparse" and control["control"] in ("tiny_inactive_smear", "zero_width_centres"):
            assert convergence_bar(control["metrics"], 64)["bar_all"]
            assert not control["passed"]


def test_observational_mode_changes_preserve_gradients_and_owned_random_streams():
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout())
    model.train()
    model[0].eval()
    optimizer = torch.optim.Adam(model.parameters())
    stream = torch.Generator().manual_seed(17)
    model(torch.randn(8, 4)).sum().backward()
    owners = dict(G=model, opt_G=optimizer, train_gen=stream)
    before = state_hash(owned(owners))
    with isolated((model,)):
        model(torch.randn(8, 4))
        np.random.randn(3)
    assert state_hash(owned(owners)) == before


def test_four_class_posterior_oracle_is_conditioned_and_remains_multimodal():
    toy = GaussianGrid(classes=4)
    observation = torch.zeros(1, 2)
    for cls in range(4):
        weights, means, variance = toy.posterior(observation, torch.tensor([cls]), .5)
        assert (weights > .01).sum() > 1
        global_mode = torch.cdist(means[0], toy.means).argmin(1)
        assert bool((toy.labels[global_mode] == cls).all())
        assert 0 < float(variance.flatten()[0]) < toy.std**2
