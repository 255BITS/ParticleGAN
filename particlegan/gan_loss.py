"""The adversarial loss: relativistic pairing (RpGAN) with the logistic link."""
import torch
import torch.nn.functional as F


class GANLoss:
    """RpGAN logistic loss (Jolicoeur-Martineau), built by ``recipe.make_loss(opt_d)``.

    The critic scores each real against the fake in the same batch row:
    ``d_loss = E[softplus(-(D(real) - D(fake)))]`` and
    ``g_loss = E[softplus(-(D(fake) - D(real)))]``. Pairing needs real and
    fake logits of the same shape in both calls.

    ``controller`` (set by ``recipe.make_loss(opt_d)``) is the recipe
    optimizers' LR controller: each call also reports its detached value,
    from which the generator optimizer reads the game payoff. Without it the
    loss is a plain function.
    """

    def __init__(self, controller=None):
        self.controller = controller

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Critic loss (minimized)."""
        loss = F.softplus(-(real_logits - fake_logits)).mean()
        if self.controller is not None:
            self.controller.record_critic_loss(loss)
        return loss

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor = None) -> torch.Tensor:
        """Generator loss (minimized); ``real_logits`` is required."""
        if real_logits is None:
            raise ValueError("RpGAN requires real_logits in g_loss")
        loss = F.softplus(-(fake_logits - real_logits)).mean()
        if self.controller is not None:
            self.controller.record_generator_loss(loss)
        return loss
