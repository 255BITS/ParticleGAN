"""Explicit fixed-kernel particle prior for the isolated Atlas experiment.

Locations, uniform row IDs and optimizer ownership remain the original table.
The Gaussian kernel acts before the generator; policy/DV12/output noise are
separate controls and are neither enabled nor disabled by this module.
"""
import torch
from torch import nn

from .particle_prior import MoGParticlePrior, _nonnegative_scalar


class NoisyParticlePrior(MoGParticlePrior):
    """Raw trainable centers plus an explicit fixed isotropic latent kernel.

``sigma`` has raw latent-coordinate units and is always caller supplied.
The positive-width sampling law is the existing unstandardized MoG law.
Zero width preserves pure row indexing and consumes no Gaussian-kernel RNG.
Birth/death owners may mutate ``z`` in place while retaining this same prior,
its fixed kernel buffers, row indices and location-optimizer alias.
"""

    def __init__(self, num_particles=20_000, z_dim=4, init_std=1.0,
                 device=None, dtype=None, learnable=True, generator=None,
                 *, sigma, standardize=False):
        if standardize is not False:
            raise ValueError('NoisyParticlePrior requires raw unstandardized row-local centers')
        super().__init__(num_particles, z_dim, init_std, device, dtype,
                         learnable, generator, sigma=sigma, standardize=False)

    @classmethod
    def from_table(cls, table, *, sigma):
        """Wrap the SAME caller-owned location Parameter without an RNG draw.

This is the direct-coordinate boundary: no latent initialization, network,
row selection, table copy or optimizer is created by wrapping the bank.
"""
        sigma = _nonnegative_scalar(sigma, 'sigma')
        if (type(table) is not nn.Parameter or table.ndim != 2
                or min(table.shape) < 1 or not table.is_floating_point()
                or not torch.isfinite(table.detach()).all()):
            raise ValueError('table must be a finite floating 2D nn.Parameter')
        prior = cls.__new__(cls)
        nn.Module.__init__(prior)
        prior.init_std = 0.0
        prior.register_parameter('z', table)
        prior.sigma_rel = 0.0
        prior.standardize = False
        prior.register_buffer('sigma', table.new_zeros(()))
        prior.register_buffer('d0', table.new_zeros(()))
        prior.set_sigma(sigma)
        return prior

    def set_extra_state(self, state):
        if state.get('standardize') is not False or state.get('sigma_rel') != 0:
            raise ValueError('NoisyParticlePrior checkpoint must retain the explicit raw fixed-kernel law')
        super().set_extra_state(state)

    def kernel_contract(self):
        """Describe the actual kernel without sampling or altering the prior."""
        return {'schema_version': 1, 'kind': 'noisy_particle_cloud',
                'code_path': 'particlegan.noisy_particle_prior.NoisyParticlePrior',
                'equation': 'z[index] + sigma * epsilon',
                'epsilon_distribution': 'independent_standard_normal',
                'sigma': float(self.sigma.detach()), 'sigma_units': 'raw_latent_coordinates',
                'standardize': False, 'learned_width': False,
                'row_weights': 'uniform', 'fixed_first_n': 'fixes_indices_only',
                'zero_sigma_consumes_no_kernel_rng': True}
