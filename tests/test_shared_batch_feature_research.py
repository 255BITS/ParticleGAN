"""Behavioral safety checks for the archived batch-distance discriminator."""
import torch

from particlegan import GradientPenalty
from benchmarks.transfer_suite.shared_batch_feature_research import ARCHITECTURES, constructor


def distance_critic():
    card = next(card for card in ARCHITECTURES if card['name'] == 'batchfeat_center6_distance_head')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return constructor(card)(2, 96, 3, 0)


def test_scores_are_permutation_equivariant_and_train_eval_stateless():
    torch.set_num_threads(1)
    critic = distance_critic()
    x = torch.tensor([[0., 0.], [.12, .04], [-.18, .21], [.35, -.16], [-.25, -.12]])
    order = torch.tensor([3, 0, 4, 1, 2])
    state = {name: value.clone() for name, value in critic.state_dict().items()}

    critic.train()
    train_scores = critic(x)
    critic.eval()
    eval_scores = critic(x)
    permuted_scores = critic(x[order])

    torch.testing.assert_close(eval_scores, train_scores, rtol=0, atol=0)
    torch.testing.assert_close(permuted_scores, eval_scores[order], rtol=1e-6, atol=1e-6)
    assert all(torch.equal(value, critic.state_dict()[name]) for name, value in state.items())
    assert not any('running_mean' in name or 'running_var' in name for name in state)


def test_score_has_a_differentiable_neighbor_effect():
    critic = distance_critic()
    with torch.no_grad():
        critic.head.weight.zero_()
        critic.head.weight[0, -4] = 1.
        critic.head.bias.zero_()
    x = torch.tensor([[0., 0.], [.12, .04], [-.19, .17]], requires_grad=True)
    score = critic(x)
    gradient = torch.autograd.grad(score[0], x)[0]

    assert torch.isfinite(gradient).all()
    assert gradient[1].abs().sum() > 1e-4
    assert gradient[2].abs().sum() > 1e-4
    assert gradient[0].abs().sum() > 1e-4


def test_singleton_distance_is_finite_zero_and_excludes_self_pair():
    critic = distance_critic()
    singleton = torch.tensor([[.3, -.2]], requires_grad=True)
    feature = critic.pairwise_features(singleton)
    assert feature.shape == (1, 4)
    assert torch.isfinite(feature).all()
    assert torch.count_nonzero(feature) == 0
    assert torch.isfinite(critic(singleton)).all()

    # With exactly one neighbor at distance .5, the widest kernel's local
    # mean-square distance is .25. A self-pair in the denominator halves it.
    pair = torch.tensor([[0., 0.], [.5, 0.]])
    widest = critic.pairwise_features(pair)[:, -1]
    assert torch.all((widest > .249) & (widest < .251))


def test_native_bcap_double_backward_and_coincident_hessian_are_finite():
    torch.set_num_threads(1)
    critic = distance_critic()
    with torch.no_grad():
        critic.head.weight[0, -4:] = torch.tensor([5., 2., 1., .5])
    penalty = GradientPenalty('b_cap', coeff=6., kappa=1.25)
    real = torch.tensor([[0., 0.], [.12, .03], [-.07, .10], [.15, -.11]])
    fake = torch.tensor([[.02, -.03], [-.10, .08], [.09, .10], [.20, -.02]])
    term = penalty(critic, real, fake, step=1)
    assert torch.isfinite(term) and term > 0
    term.backward()
    gradients = [parameter.grad for parameter in critic.parameters()]
    # Biases that vanish under centering or input differentiation need not
    # receive a gradient from b_cap alone.
    assert any(grad is not None for grad in gradients)
    assert all(grad is None or torch.isfinite(grad).all() for grad in gradients)
    assert critic.head.weight.grad[0, -4:].abs().sum() > 0

    critic.zero_grad(set_to_none=True)
    coincident = torch.zeros(4, 2)
    coincident_term = penalty(critic, coincident, coincident, step=1)
    assert torch.isfinite(coincident_term)
    coincident_term.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in critic.parameters())

    # The distance feature is smooth even when two inputs coincide. Its
    # neighbor coupling has a finite, nonzero second input derivative there.
    with torch.no_grad():
        critic.head.weight.zero_()
        critic.head.weight[0, -4] = 1.
        critic.head.bias.zero_()
    x = torch.zeros(3, 2, requires_grad=True)
    first = torch.autograd.grad(critic(x)[0], x, create_graph=True)[0]
    second = torch.autograd.grad(first[1, 0], x)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert second[0, 0].abs() > 1e-4
