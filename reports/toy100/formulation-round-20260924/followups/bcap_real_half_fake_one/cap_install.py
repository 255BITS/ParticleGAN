from particlegan.grad_regularizers import GradientPenalty
_original_penalty = GradientPenalty.penalty

def asymmetric_cap(self, D, x_real, x_fake, step=1, generator=None, collect_stats=True):
    if self.arm != 'b_cap' or (self.lazy_k > 1 and step % self.lazy_k != 0):
        return _original_penalty(self, D, x_real, x_fake, step, generator, collect_stats)
    coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
    real_norm = self._grad_norm(D, x_real, squared=False)
    fake_norm = self._grad_norm(D, x_fake, squared=False)
    penalty = (coefficient / 2.0) * ((real_norm - .5 * self.kappa).relu().square().mean() + (fake_norm - self.kappa).relu().square().mean())
    stats = {'applied': True, 'pen': float(penalty.detach()), 'real_cap': .5 * self.kappa, 'fake_cap': self.kappa} if collect_stats else {}
    return penalty, stats

GradientPenalty.penalty = asymmetric_cap
