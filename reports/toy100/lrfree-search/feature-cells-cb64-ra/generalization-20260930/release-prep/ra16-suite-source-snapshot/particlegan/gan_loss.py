"""The adversarial loss: relativistic pairing (RpGAN) with the logistic link."""
import torch
import torch.nn.functional as F


class GANLoss:
    """RpGAN logistic loss (Jolicoeur-Martineau), built by ``recipe.make_loss()``.

    The critic scores each real against the fake in the same batch row:
    ``d_loss = E[softplus(-(D(real) - D(fake)))]`` and
    ``g_loss = E[softplus(-(D(fake) - D(real)))]``. Pairing needs real and
    fake logits of the same shape in both calls.
    """

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Critic loss (minimized)."""
        return F.softplus(-(real_logits - fake_logits)).mean()

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor = None) -> torch.Tensor:
        """Generator loss (minimized); ``real_logits`` is required."""
        if real_logits is None:
            raise ValueError("RpGAN requires real_logits in g_loss")
        return F.softplus(-(fake_logits - real_logits)).mean()
