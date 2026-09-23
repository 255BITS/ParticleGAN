"""Public batch-distance critic must reproduce the archived rare-mass witness."""
import pytest
import torch

from benchmarks.transfer_suite.shared_batch_feature_research import ARCHITECTURES, constructor
from particlegan import BatchDistanceDiscriminator, GradientPenalty


def paired_critics():
    card = next(card for card in ARCHITECTURES
                if card['name'] == 'batchfeat_center6_distance_head')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        research = constructor(card)(2, 96, 3, 0)
        torch.manual_seed(0)
        public = BatchDistanceDiscriminator()
    return research, public


def test_default_initialization_scores_and_first_second_derivatives_are_identical():
    torch.set_num_threads(1)
    research, public = paired_critics()
    assert sum(p.numel() for p in public.parameters()) == 19013
    assert list(research.state_dict()) == list(public.state_dict())
    for name, value in research.state_dict().items():
        assert torch.equal(value, public.state_dict()[name]), name

    points = torch.tensor([[0., 0.], [.12, .04], [-.18, .21], [.35, -.16],
                           [-.25, -.12], [1.2, -.4]], requires_grad=True)
    for reference, promoted in ((research.pairwise_features, public.pairwise_features),
                                (research, public)):
        original, current = reference(points), promoted(points)
        assert torch.equal(original, current)
        first_original = torch.autograd.grad(original.sum(), points, create_graph=True)[0]
        first_current = torch.autograd.grad(current.sum(), points, create_graph=True)[0]
        assert torch.equal(first_original, first_current)
        second_original = torch.autograd.grad(first_original.square().sum(), points)[0]
        second_current = torch.autograd.grad(first_current.square().sum(), points)[0]
        assert torch.equal(second_original, second_current)


def test_native_active_bcap_value_and_parameter_gradients_are_identical():
    torch.set_num_threads(1)
    research, public = paired_critics()
    with torch.no_grad():
        for critic in (research, public):
            critic.head.weight[0, -4:] = torch.tensor([5., 2., 1., .5])
    real = torch.tensor([[0., 0.], [.12, .03], [-.07, .10], [.15, -.11]])
    fake = torch.tensor([[.02, -.03], [-.10, .08], [.09, .10], [.20, -.02]])
    penalty = GradientPenalty('b_cap', coeff=6., kappa=1.25)
    old, new = penalty(research, real, fake, step=1), penalty(public, real, fake, step=1)
    assert old > 0 and torch.equal(old, new)
    old.backward()
    new.backward()
    for (old_name, old_parameter), (new_name, new_parameter) in zip(
            research.named_parameters(), public.named_parameters()):
        assert old_name == new_name
        if old_parameter.grad is None:
            assert new_parameter.grad is None
        else:
            assert torch.equal(old_parameter.grad, new_parameter.grad), old_name
    assert public.head.weight.grad[0, -4:].abs().sum() > 0


def test_public_critic_is_batch_equivariant_stateless_and_supports_custom_dimensions():
    public = BatchDistanceDiscriminator(in_dim=3, hidden_dim=8, n_hidden=2,
                                        scales=(.2, 1.))
    x = torch.tensor([[0., 0., 0.], [.1, -.2, .3], [-.3, .5, .2]])
    permutation = torch.tensor([2, 0, 1])
    public.train()
    train_scores = public(x)
    public.eval()
    assert torch.equal(public(x), train_scores)
    torch.testing.assert_close(public(x[permutation]), train_scores[permutation],
                               rtol=1e-6, atol=1e-6)
    assert public.pairwise_features(x).shape == (3, 2)
    assert not any('running_' in name for name in public.state_dict())


@pytest.mark.parametrize('kwargs', [
    {'in_dim': 0}, {'hidden_dim': 0}, {'n_hidden': 0},
    {'scales': ()}, {'scales': (0.,)}, {'scales': (float('nan'),)},
    {'scales': (True,)}, {'beta': 0.}, {'beta': 'six'},
    {'eps': -1.}, {'eps': float('nan')},
])
def test_invalid_public_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BatchDistanceDiscriminator(**kwargs)


def test_invalid_input_shape_is_rejected():
    critic = BatchDistanceDiscriminator()
    with pytest.raises(ValueError, match='nonempty'):
        critic(torch.empty(0, 2))
    with pytest.raises(ValueError, match='nonempty'):
        critic(torch.ones(3, 1))
