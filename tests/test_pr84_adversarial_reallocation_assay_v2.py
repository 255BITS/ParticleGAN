import pytest
import torch
from particlegan.gan_loss import GANLoss
from reports.toy100.pr84_adversarial_reallocation_assay import loss, replace_fake
from reports.toy100.pr84_adversarial_reallocation_assay_v2 import select


class Critic(torch.nn.Module):
    def __init__(self, squeezed):
        super().__init__()
        self.squeezed = squeezed

    def forward(self, x):
        value = x[:, :1]
        return value[:, 0] if self.squeezed else value


@pytest.mark.parametrize('squeezed', [True, False])
def test_baseline_and_every_proposal_equal_original_paired_host_loss(squeezed):
    fake = torch.tensor([[-1., 0.], [0., 0.], [-1., 0.], [1., 0.]], dtype=torch.float64)
    batch = dict(indices=torch.tensor([0, 1, 0, 2]),
                 real=torch.tensor([[2., 0.], [1., 0.], [0., 0.], [-1., 0.]], dtype=torch.float64),
                 sigma=.029, noise=torch.zeros(4, 2, dtype=torch.float64))
    candidates = torch.tensor([[2., .4], [-2., .1]], dtype=torch.float64)
    critic, gan = Critic(squeezed), GANLoss('logistic', 'rp')
    selected, _, scores = select(critic, fake, batch, candidates, 4, gan)
    native = loss(critic, fake, batch['real'], gan)
    assert selected['native_loss_before'] == float(native)
    expected = torch.stack([torch.stack([
        loss(critic, replace_fake(fake, batch, donor, target), batch['real'], gan)
        for target in candidates]) for donor in range(4)])
    torch.testing.assert_close(scores, expected, atol=1e-14, rtol=1e-14)
    proposed = replace_fake(fake, batch, selected['donor'], candidates[selected['candidate']])
    # The affine critic's symmetric stencil equals its original affine value.
    direct_before = torch.nn.functional.softplus(batch['real'][:, 0] - fake[:, 0]).mean()
    direct_after = torch.nn.functional.softplus(batch['real'][:, 0] - proposed[:, 0]).mean()
    assert abs(selected['native_loss_before'] - float(direct_before)) < 1e-14
    assert abs(selected['native_loss_after'] - float(direct_after)) < 1e-14
    assert selected['paired_baseline_exact']
    assert torch.all(scores[3] == native)
