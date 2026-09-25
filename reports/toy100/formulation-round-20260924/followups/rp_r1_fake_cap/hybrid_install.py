from particlegan.grad_regularizers import GradientPenalty
_original_penalty = GradientPenalty.penalty

def hybrid_penalty(self, D, x_real, x_fake, step=1, generator=None, collect_stats=True):
    if self.arm != 'a_r1r2' or (self.lazy_k > 1 and step % self.lazy_k != 0):
        return _original_penalty(self, D, x_real, x_fake, step, generator, collect_stats)
    coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
    real_squared = self._grad_norm(D, x_real, squared=True)
    fake_norm = self._grad_norm(D, x_fake, squared=False)
    fake_cap = (fake_norm - self.kappa).relu().square()
    penalty = (coefficient / 2.0) * (real_squared.mean() + fake_cap.mean())
    stats = {'applied': True, 'pen': float(penalty.detach()), 'fake_cap': self.kappa} if collect_stats else {}
    return penalty, stats

GradientPenalty.penalty = hybrid_penalty
