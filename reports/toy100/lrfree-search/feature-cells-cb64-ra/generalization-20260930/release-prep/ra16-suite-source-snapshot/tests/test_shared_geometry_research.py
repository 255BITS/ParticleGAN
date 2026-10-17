"""Invariants for the D-only geometry research variants."""
import torch

from benchmarks.transfer_suite import shared_geometry_research as geometry
from benchmarks.transfer_suite import shared_geometry_residual_research as residual
from benchmarks.transfer_suite import shared_geometry_readout_research as readout
from benchmarks.transfer_suite import shared_pointnorm_research as pointnorm
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.shared_variants import architecture_spec


REFERENCE = dict(name='reference_ln_beta4', implementation='shared_pointnorm_v1',
                 width=96, layers=3, fourier=0, activation='softplus',
                 softplus_beta=4., normalization='layernorm',
                 normalization_scope='all', layernorm_eps=1e-5,
                 rmsnorm_eps=1e-5, output_scale=1.)


def test_geometry_cards_are_discriminator_only_and_differentiable():
    original = next(job['spec'] for job in plan() if job['spec']['name'] == 'vector_unequal_mass')
    for module in (geometry, residual, readout):
        for card in module.ARCHITECTURES:
            changed = architecture_spec(original, module.variant(card))
            assert changed['name'] == original['name']
            assert changed['means'] == original['means']
            assert changed['covariances'] == original['covariances']
            assert changed['masses'] == original['masses']
            torch.manual_seed(0)
            model = module.constructor(card)(2, card['width'], card['layers'], 0)
            x = torch.tensor([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5],
                              [1.5, 1.5], [0., 0.]], requires_grad=True)
            output = model(x)
            gradient = torch.autograd.grad(output.sum(), x)[0]
            assert output.shape == (5,)
            assert torch.isfinite(output).all()
            assert torch.isfinite(gradient).all()


def test_residual_pathways_preserve_lead_at_initialization_and_receive_gradients():
    x = torch.tensor([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5],
                      [1.5, 1.5], [0., 0.]])
    for card in residual.ARCHITECTURES:
        torch.manual_seed(0)
        base = pointnorm.SharedPointnormCritic(2, 96, 3, 0, architecture=REFERENCE)
        expected_state = torch.random.get_rng_state()
        torch.manual_seed(0)
        model = residual.SharedGeometryResidualCritic(2, 96, 3, 0, architecture=card)
        assert torch.equal(torch.random.get_rng_state(), expected_state)
        assert torch.equal(model(x), base(x))
        for first, second in zip(model.layers, base.layers):
            assert torch.equal(first.weight, second.weight)
            assert torch.equal(first.bias, second.bias)
        assert torch.equal(model.head.weight, base.head.weight)
        assert torch.equal(model.head.bias, base.head.bias)
        model(x).square().sum().backward()
        added = [p.grad for name, p in model.named_parameters()
                 if any(tag in name for tag in ('injections', 'feature_head', 'activation_coefficients'))]
        assert added and all(grad is not None and torch.isfinite(grad).all()
                             and grad.abs().sum() > 0 for grad in added)


def test_readout_variants_keep_identical_trunk_and_finite_tail_gradients():
    for card in readout.ARCHITECTURES:
        torch.manual_seed(0)
        base = pointnorm.SharedPointnormCritic(2, 96, 3, 0, architecture=REFERENCE)
        expected_state = torch.random.get_rng_state()
        torch.manual_seed(0)
        model = readout.SharedGeometryReadoutCritic(2, 96, 3, 0, architecture=card)
        assert torch.equal(torch.random.get_rng_state(), expected_state)
        for first, second in zip(model.layers, base.layers):
            assert torch.equal(first.weight, second.weight)
            assert torch.equal(first.bias, second.bias)
        assert torch.equal(model.head.weight, base.head.weight)
        assert torch.equal(model.head.bias, base.head.bias)
        x = torch.tensor([[-10., -10.], [0., 0.], [10., 10.]], requires_grad=True)
        score = model(x)
        gradient = torch.autograd.grad(score.sum(), x)[0]
        assert torch.isfinite(score).all() and torch.isfinite(gradient).all()
