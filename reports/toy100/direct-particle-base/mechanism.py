"""Rp hybrid with input-gradient RMS units, independent of host identity."""
from particlegan.grad_regularizers import GradientPenalty
_original_penalty = GradientPenalty.penalty
receipt = {'calls': 0, 'dimensions': {}, 'extra_critic_forwards': 0}

def scaled_penalty(self, D, x_real, x_fake, step=1, generator=None, collect_stats=True):
    if self.arm != 'a_r1r2' or (self.lazy_k > 1 and step % self.lazy_k != 0):
        return _original_penalty(self, D, x_real, x_fake, step, generator, collect_stats)
    coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
    dimension = x_real[0].numel()
    assert dimension == x_fake[0].numel()
    real_squared = self._grad_norm(D, x_real, squared=True) / dimension
    fake_norm = self._grad_norm(D, x_fake, squared=False) / dimension ** 0.5
    fake_cap = (fake_norm - self.kappa).relu().square()
    penalty = (coefficient / 2.0) * (real_squared.mean() + fake_cap.mean())
    receipt['calls'] += 1
    receipt['dimensions'][str(dimension)] = receipt['dimensions'].get(str(dimension), 0) + 1
    stats = {'applied': True, 'pen': float(penalty.detach()), 'fake_cap': self.kappa} if collect_stats else {}
    return penalty, stats

GradientPenalty.penalty = scaled_penalty
