import math
import torch
from experiments.train_mog_vae import posterior_kl, two_draw_loss


def test_kl_matches_torch_distributions():
    logits = torch.tensor([[.3, -.4, .7], [-1., .1, 2.]], dtype=torch.float64)
    logq = logits.log_softmax(1)
    u = torch.tensor([[.2, -.8], [1., 0.]], dtype=torch.float64)
    logs = torch.tensor([[.3, -.4], [.2, 0.]], dtype=torch.float64)
    kc, kl = posterior_kl(logq, u, logs, True)
    dist = torch.distributions
    expected_c = dist.kl_divergence(dist.Categorical(logits=logits), dist.Categorical(logits=torch.zeros_like(logits)))
    expected_l = dist.kl_divergence(dist.Normal(u, logs.exp()), dist.Normal(torch.zeros_like(u), torch.ones_like(u))).sum(1)
    torch.testing.assert_close(kc, expected_c)
    torch.testing.assert_close(kl, expected_l)
    _, zero = posterior_kl(logq, u, logs, False)
    assert zero.eq(0).all()


def test_two_draw_gradient_exact_enumeration():
    # Enumerate every iid pair, proving expected surrogate gradient matches
    # the exact categorical expectation, including cost path derivatives.
    logits = torch.tensor([.2, -.4, .7], dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(.6, dtype=torch.float64, requires_grad=True)
    cost = (theta - torch.tensor([-1., .5, 2.], dtype=torch.float64)).square()
    logq = logits.log_softmax(0); q = logq.exp()
    exact = torch.autograd.grad((q * cost).sum(), (logits, theta), retain_graph=True)
    estimated = [torch.zeros_like(logits), torch.zeros_like(theta)]
    for a in range(3):
        for b in range(3):
            loss = two_draw_loss(cost[[a, b]][None], logq[[a, b]][None])
            torch.testing.assert_close(loss, cost[[a, b]].mean())
            gradients = torch.autograd.grad(loss, (logits, theta), retain_graph=True)
            for i, gradient in enumerate(gradients):
                estimated[i] += q[a].detach() * q[b].detach() * gradient
    for actual, expected in zip(estimated, exact):
        torch.testing.assert_close(actual, expected)


def test_scaled_gaussian_elbo_in_two_dimensions():
    x = torch.tensor([[.5, -.2]], dtype=torch.float64)
    y = torch.tensor([[.4, .3]], dtype=torch.float64)
    tau = .1
    negative_log_likelihood = -torch.distributions.Normal(y, tau).log_prob(x).sum(1).mean()
    mse = (y - x).square().mean()
    torch.testing.assert_close(tau**2 * negative_log_likelihood, mse + tau**2 * math.log(2 * math.pi * tau**2))
