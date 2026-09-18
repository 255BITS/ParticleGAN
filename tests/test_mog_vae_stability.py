import math
import torch
from particlegan import MoGParticlePrior
from experiments.train_mog_vae_stability import Encoder, sampled_codes, variational_kl


def test_hard_forward_matches_selected_prior_component_and_has_query_surrogate():
    torch.manual_seed(12)
    e = Encoder(8).double()
    p = MoGParticlePrior(num_particles=8, z_dim=2, sigma_rel=.025, device='cpu').double()
    x = torch.tensor([[1., 2.], [-2., 1.]], dtype=torch.float64)
    seed = 19
    z, ids, logq, u, logs = sampled_codes(e, x, p, {'posterior':'hard', 'temperature':.25},
                                         torch.Generator().manual_seed(seed), 4)
    eps = torch.randn((2, 4, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(z, p.means()[ids] + p.sigma * eps)
    assert torch.equal(ids, ids[:, :1].expand_as(ids))
    assert torch.equal(logq.exp().sum(1), torch.ones(2, dtype=torch.float64))
    assert (logq.exp() > 0).sum(1).eq(1).all()
    kc, kl = variational_kl(logq, u, logs, 'hard')
    expected = torch.distributions.kl_divergence(
        torch.distributions.Categorical(probs=logq.exp()),
        torch.distributions.Categorical(probs=torch.ones_like(logq)/8))
    torch.testing.assert_close(kc, expected)
    assert kl.eq(0).all() and not kc.requires_grad
    z.square().mean().backward()
    grad = e.net.net[-1].weight.grad
    assert torch.isfinite(grad).all() and grad[:2].abs().sum() > 0
    assert grad[2:].eq(0).all()  # no learned offset or variance in hard family


def test_joint_log_density_ratio_is_constant_despite_learned_centers():
    for centers in [torch.tensor([[0., 1.], [2., 3.]]), torch.tensor([[4., 5.], [-2., 3.]])]:
        sigma = .2
        z = centers + torch.tensor([[.1, -.2], [.3, .1]])
        local = torch.distributions.Normal(centers, sigma).log_prob(z).sum(1)
        log_q_joint = local  # hard posterior mass one on selected index
        log_p_joint = -math.log(7) + local  # K=7, same local Gaussian
        torch.testing.assert_close(log_q_joint-log_p_joint, torch.full((2,), math.log(7)))
