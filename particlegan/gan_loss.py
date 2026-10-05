"""Adversarial objectives on raw critic scores, independent of regularization."""
import torch
import torch.nn.functional as F


class GANLoss:
    """A scalar GAN objective, built by ``recipe.make_loss()``.

    ``relativistic`` preserves the paired logistic RpGAN objective. The other
    objectives are ``non_saturating`` logistic, ``hinge``, ``wasserstein`` and
    ``least_squares`` (real/fake targets 1/0, generator target 1, half factors).
    Both methods return losses to minimize and preserve input gradient paths;
    the caller owns detaching scores and choosing a critic regularizer.
    """

    LOSSES = ("relativistic", "non_saturating", "hinge", "wasserstein", "least_squares")

    def __init__(self, loss="relativistic"):
        if loss not in self.LOSSES:
            raise ValueError(f"unknown adversarial loss {loss!r}; choose {', '.join(self.LOSSES)}")
        self.loss = loss

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Critic loss (minimized)."""
        if self.loss == "relativistic":
            return F.softplus(-(real_logits - fake_logits)).mean()
        if self.loss == "non_saturating":
            return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
        if self.loss == "hinge":
            return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
        if self.loss == "wasserstein":
            return fake_logits.mean() - real_logits.mean()
        return .5 * ((real_logits - 1).square().mean() + fake_logits.square().mean())

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor = None) -> torch.Tensor:
        """Generator loss; only ``relativistic`` requires real scores."""
        if self.loss == "relativistic":
            if real_logits is None:
                raise ValueError("RpGAN requires real_logits in g_loss")
            return F.softplus(-(fake_logits - real_logits)).mean()
        if self.loss == "non_saturating":
            return F.softplus(-fake_logits).mean()
        if self.loss in ("hinge", "wasserstein"):
            return -fake_logits.mean()
        return .5 * (fake_logits - 1).square().mean()
