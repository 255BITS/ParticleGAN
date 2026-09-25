# R1 only: preserve the original real-point coefficient; zero fake-point pressure.
# Retain the zero-weight fake graph to preserve baseline forward/backward work
# and exact native CUDA random draw order for this equal-compute comparison.
from particlegan.grad_regularizers import GradientPenalty
_original_penalty = GradientPenalty.penalty

def r1_only_penalty(self, D, x_real, x_fake, step=1, generator=None, collect_stats=True):
    if self.arm != 'a_r1r2' or (self.lazy_k > 1 and step % self.lazy_k != 0):
        return _original_penalty(self, D, x_real, x_fake, step, generator, collect_stats)
    coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
    real_squared = self._grad_norm(D, x_real, squared=True)
    fake_squared = self._grad_norm(D, x_fake, squared=True)
    penalty = (coefficient / 2.0) * (real_squared.mean() + 0.0 * fake_squared.mean())
    stats = {'applied': True, 'pen': float(penalty.detach()), 'center': 0.0} if collect_stats else {}
    return penalty, stats

GradientPenalty.penalty = r1_only_penalty
