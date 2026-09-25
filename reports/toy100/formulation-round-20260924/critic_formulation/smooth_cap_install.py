# Candidate mechanism: smooth radial cap with the original unit scale.
# phi(n) = (sqrt(n^2 + kappa^2) - kappa)^2.
# Positive pressure for every n>0, quartic near zero, quadratic at large n.
# No changes to network calls, penalty sample locations, RNG, or Adam.
from particlegan.grad_regularizers import GradientPenalty
_original_phi = GradientPenalty._phi

def smooth_cap_phi(self, norm, center):
    if self.arm != 'b_cap':
        return _original_phi(self, norm, center)
    return (torch.sqrt(norm.square() + center * center) - center).square()

GradientPenalty._phi = smooth_cap_phi
