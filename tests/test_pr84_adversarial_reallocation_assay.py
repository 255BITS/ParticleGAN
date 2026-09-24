import torch
from particlegan.gan_loss import GANLoss
from reports.toy100.pr84_adversarial_reallocation_assay import loss, replace_fake, select


class Affine(torch.nn.Module):
    def __init__(self, bias=0., flat=False):
        super().__init__()
        self.bias = bias
        self.flat = flat

    def forward(self, x):
        return x[:, :1] * (0. if self.flat else 1.) + self.bias


def case():
    fake = torch.tensor([[-1., 0.], [0., 0.], [-1., 0.], [1., 0.]], dtype=torch.float64)
    batch = dict(indices=torch.tensor([0, 1, 0, 2]), real=torch.zeros(4, 2, dtype=torch.float64),
                 sigma=.029, noise=torch.zeros(4, 2, dtype=torch.float64))
    candidates = torch.tensor([[2., .4], [-2., .1]], dtype=torch.float64)
    return fake, batch, candidates


def test_exact_separable_enumeration_matches_full_original_loss():
    fake, batch, candidates = case()
    gan = GANLoss('logistic', 'rp')
    selected, target, scores = select(Affine(), fake, batch, candidates, 3, gan)
    expected = torch.stack([torch.stack([
        loss(Affine(), replace_fake(fake, batch, donor, point), batch['real'], gan)
        for point in candidates]) for donor in range(3)])
    torch.testing.assert_close(scores, expected, rtol=1e-14, atol=1e-14)
    assert selected['selected'] and selected['donor'] == 0 and selected['candidate'] == 0
    torch.testing.assert_close(target, candidates[0])


def test_constant_critic_rests_and_does_not_consume_rng_or_mutate_data():
    fake, batch, candidates = case()
    old = fake.clone(), batch['indices'].clone(), batch['noise'].clone(), candidates.clone()
    rng = torch.get_rng_state().clone()
    selected, _, _ = select(Affine(flat=True), fake, batch, candidates, 3, GANLoss('logistic', 'rp'))
    assert not selected['selected']
    assert torch.equal(rng, torch.get_rng_state())
    for before, after in zip(old, (fake, batch['indices'], batch['noise'], candidates)):
        assert torch.equal(before, after)


def test_common_critic_bias_and_unobserved_particle_do_not_change_selection():
    fake, batch, candidates = case()
    a, target_a, scores_a = select(Affine(), fake, batch, candidates, 4, GANLoss('logistic', 'rp'))
    b, target_b, scores_b = select(Affine(7.), fake, batch, candidates, 4, GANLoss('logistic', 'rp'))
    assert (a['donor'], a['candidate']) == (b['donor'], b['candidate'])
    torch.testing.assert_close(target_a, target_b)
    torch.testing.assert_close(scores_a, scores_b, rtol=1e-14, atol=1e-14)
    assert torch.all(scores_a[3] == a['native_loss_before'])
