"""Adversarial objectives on raw critic scores, independent of regularization."""
import torch
import torch.nn.functional as F
import math


class GANLoss:
    """A scalar GAN objective, built by ``recipe.make_loss()``.

    ``relativistic`` preserves the paired logistic RpGAN objective. The other
    objectives are ``non_saturating`` logistic, ``hinge``, ``wasserstein`` and
    ``least_squares`` (real/fake targets 1/0, generator target 1, half factors).
    Both methods return losses to minimize and preserve input gradient paths;
    the caller owns detaching scores and choosing a critic regularizer.
    """

    LOSSES = ("relativistic", "non_saturating", "hinge", "wasserstein", "least_squares")

    def __init__(self, loss="relativistic", *, labels=(0.0, 1.0, 1.0)):
        if loss not in self.LOSSES:
            raise ValueError(f"unknown adversarial loss {loss!r}; choose {', '.join(self.LOSSES)}")
        self.loss = loss
        if (not isinstance(labels, (list, tuple)) or len(labels) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in labels)):
            raise ValueError("least-squares labels must be three finite numbers (fake, real, generator)")
        self.labels = tuple(float(v) for v in labels)
        if loss != "least_squares" and self.labels != (0.0, 1.0, 1.0):
            raise ValueError("nondefault loss labels require least_squares")

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
        a, b, _ = self.labels
        return .5 * ((real_logits - b).square().mean() + (fake_logits - a).square().mean())

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
        return .5 * (fake_logits - self.labels[2]).square().mean()

    def joint_g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        """Generator/encoder loss for a joint BiGAN critic.

        Both scores have trainable inputs: fake pairs ``(G(z), z)`` must look
        real, while encoded real pairs ``(x, E(x))`` must look fake. Scalar
        GANs use ``g_loss`` instead. The paired default is exactly its original
        expression; unpaired losses add the reversed-label real-stream term.
        """
        if real_logits is None:
            raise ValueError("joint_g_loss requires real_logits for the encoder stream")
        if self.loss == "relativistic":
            return self.g_loss(fake_logits, real_logits)
        if self.loss == "non_saturating":
            return self.g_loss(fake_logits) + F.softplus(real_logits).mean()
        if self.loss in ("hinge", "wasserstein"):
            return self.g_loss(fake_logits) + real_logits.mean()
        return self.g_loss(fake_logits) + .5 * (real_logits - self.labels[0]).square().mean()
