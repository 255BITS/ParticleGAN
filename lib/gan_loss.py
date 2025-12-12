import torch
import torch.nn.functional as F


class GANLoss:
    """
    Common GAN loss utilities.
    Supports standard (vanilla) GANs and Relativistic GANs (RpGAN, RaGAN).
    
    Args:
        loss_type (str): 'hinge', 'logistic', 'wasserstein', 'lsgan'.
        mode (str):
            - 'vanilla': Standard GAN (Goodfellow et al).
            - 'rp': Relativistic Pairing (Jolicoeur-Martineau).
                    D(real, fake) = D(real) - D(fake).
            - 'ra': Relativistic Average.
                    D(real, fake) = D(real) - mean(D(fake)).
    """

    def __init__(
        self,
        loss_type: str = "hinge",
        mode: str = "vanilla",
        label_smoothing: float = 0.0,
        label_flip_prob: float = 0.0,
    ) -> None:
        self.loss_type = loss_type.lower()
        self.mode = mode.lower()
        self.label_smoothing = float(label_smoothing)
        self.label_flip_prob = float(label_flip_prob)

        # Map string to kernel methods
        kernels = {
            "hinge": (self._d_hinge, self._g_hinge),
            "logistic": (self._d_logistic, self._g_logistic),
            "wasserstein": (self._d_wasserstein, self._g_wasserstein),
            "lsgan": (self._d_lsgan, self._g_lsgan),
        }
        
        if self.loss_type not in kernels:
            raise ValueError(f"Unknown GAN loss type: {self.loss_type}")
            
        self._kernel_d, self._kernel_g = kernels[self.loss_type]

    def d_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Discriminator loss (minimization objective)."""
        if self.mode == "vanilla":
            return self._kernel_d(real_logits, fake_logits)
        elif self.mode == "rp":
            # RpGAN: D input is (real - fake)
            return self._kernel_d(real_logits - fake_logits, None)
        elif self.mode == "ra":
            # RaGAN: D(real - mean(fake)) and D(fake - mean(real))
            return (
                self._kernel_d(real_logits - torch.mean(fake_logits, dim=0, keepdim=True), None)
                + self._kernel_d(None, fake_logits - torch.mean(real_logits, dim=0, keepdim=True))
            ) * 0.5
        else:
            raise ValueError(f"Unknown gan mode: {self.mode}")

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor = None) -> torch.Tensor:
        """Generator loss (minimization objective)."""
        if self.mode == "vanilla":
            return self._kernel_g(fake_logits)
        elif self.mode == "rp":
            if real_logits is None:
                raise ValueError("RpGAN requires real_logits in g_loss")
            # RpGAN G: Symmetric to D. G wants (fake - real) to be 'real'.
            return self._kernel_g(fake_logits - real_logits)
        elif self.mode == "ra":
            if real_logits is None:
                raise ValueError("RaGAN requires real_logits in g_loss")
            return (
                self._kernel_g(fake_logits - torch.mean(real_logits, dim=0, keepdim=True))
                + self._kernel_g(real_logits - torch.mean(fake_logits, dim=0, keepdim=True), invert=True)
            ) * 0.5
        else:
            raise ValueError(f"Unknown gan mode: {self.mode}")

    # -------------------------
    #  Loss Kernels
    # -------------------------

    # Each kernel handles standard (real/fake) or relativistic (logits/None)
    # When fake_logits is None, it means we are in 'relativistic' single-term mode (D(r-f)).
    
    def _d_hinge(self, real_logits, fake_logits):
        if fake_logits is None: # Relativistic case: E[max(0, 1 - (r-f))]
            return F.relu(1.0 - real_logits).mean()
        # Vanilla case
        loss_real = F.relu(1.0 - real_logits).mean()
        loss_fake = F.relu(1.0 + fake_logits).mean()
        return loss_real + loss_fake

    def _g_hinge(self, fake_logits, invert=False):
        # Vanilla: -E[fake]
        # Rp/Ra: -E[fake - real]. (Linear hinge generator).
        # If invert=True (for RaGAN 2nd term), we want to minimize 'real' score -> maximize 'real'.
        # Note: Symmetric Hinge G is often just -mean.
        return -fake_logits.mean() if not invert else fake_logits.mean()

    def _d_wasserstein(self, real_logits, fake_logits):
        if fake_logits is None: # Relativistic: E[fake - real] (minimized) -> E[-(real-fake)]
            return -real_logits.mean()
        # Vanilla: E[fake] - E[real]
        return fake_logits.mean() - real_logits.mean()

    def _g_wasserstein(self, fake_logits, invert=False):
        return -fake_logits.mean() if not invert else fake_logits.mean()
    
    def _d_lsgan(self, real_logits, fake_logits):
        if fake_logits is None: # Relativistic: E[(real - fake - 1)^2]
            return 0.5 * ((real_logits - 1.0) ** 2).mean()
        # Vanilla
        return 0.5 * (((real_logits - 1.0) ** 2).mean() + (fake_logits ** 2).mean())

    def _g_lsgan(self, fake_logits, invert=False):
        if invert: # RaGAN 2nd term: real should look fake (0)
            return 0.5 * (fake_logits ** 2).mean()
        # Vanilla/Rp: fake should look real (1)
        return 0.5 * ((fake_logits - 1.0) ** 2).mean()

    def _d_logistic(self, real_logits, fake_logits):
        if fake_logits is None:
            # Relativistic: E[softplus(-(real - fake))].
            # Minimizing this maximizes (real - fake).
            return F.softplus(-real_logits).mean()

        # Vanilla with smoothing
        if self.label_smoothing > 0.0:
            real_targets = torch.full_like(real_logits, 1.0 - self.label_smoothing)
            fake_targets = torch.full_like(fake_logits, self.label_smoothing)
        else:
            real_targets = torch.ones_like(real_logits)
            fake_targets = torch.zeros_like(fake_logits)

        # Optional label flipping on a per-element basis.
        if self.label_flip_prob > 0.0:
            flip_real = torch.rand_like(real_targets) < self.label_flip_prob
            flip_fake = torch.rand_like(fake_targets) < self.label_flip_prob
            real_targets = torch.where(flip_real, 1.0 - real_targets, real_targets)
            fake_targets = torch.where(flip_fake, 1.0 - fake_targets, fake_targets)

        loss_real = F.binary_cross_entropy_with_logits(real_logits, real_targets)
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
        return loss_real + loss_fake

    def _g_logistic(self, fake_logits, invert=False):
        # Vanilla: E[softplus(-fake)] -> Maximize fake score.
        if invert:
            # RaGAN 2nd term: Real images should look fake. Maximize softplus(real).
            # min E[softplus(real)]
            return F.softplus(fake_logits).mean()
        return F.softplus(-fake_logits).mean()


def get_gan_loss(
    loss_type: str = "hinge",
    mode: str = "vanilla",
    label_smoothing: float = 0.0,
    label_flip_prob: float = 0.0,
) -> GANLoss:
    """
    Convenience factory that mirrors pytorch-style helpers.

    mode: 'vanilla' (Standard), 'rp' (Pairing), 'ra' (Average)
    """
    return GANLoss(
        loss_type=loss_type,
        mode=mode,
        label_smoothing=label_smoothing,
        label_flip_prob=label_flip_prob,
    )
