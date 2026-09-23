"""Read-only analytic checks for the shared-c6 rare-case mechanism audit.

This script constructs static tensors and differentiates the native losses.
It performs no optimizer step, benchmark training, gate change, or toy run.
"""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from particlegan.grad_regularizers import GradientPenalty
from particlegan.vicreg_loss import ParticleRegularizer


SOURCES = (
    'particlegan/vicreg_loss.py', 'particlegan/grad_regularizers.py',
    'particlegan/gan_loss.py', 'particlegan/particle_prior.py',
    'benchmarks/transfer_suite/vector_tasks.py', 'lib/toy_models.py',
)


def latent_spread_counterexample():
    # Eight point clusters, 32 identical particles each. Every coordinate has
    # global standard deviation above one and global off-diagonal covariance 0.
    z = torch.cat([sign*2*torch.eye(4)[axis].repeat(32, 1)
                   for axis in range(4) for sign in (-1., 1.)], dim=0)
    z.requires_grad_(True)
    loss = ParticleRegularizer(weight=.05)(z)
    loss.backward()
    centered = z.detach()-z.detach().mean(dim=0)
    cov = centered.T@centered/(len(z)-1)
    offdiag = cov-torch.diag(torch.diag(cov))
    return dict(particles=len(z), dimensions=z.shape[1],
                global_std=z.detach().std(dim=0).tolist(),
                global_offdiagonal_covariance_max=float(offdiag.abs().max()),
                per_cluster_variance_max=float(z.detach()[:32].var(dim=0).max()),
                regularizer=float(loss.detach()),
                latent_gradient_abs_max=float(z.grad.abs().max()))


class LinearCritic(torch.nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([slope, 0.], dtype=torch.float32))

    def forward(self, x):
        return x@self.weight


def one_sided_cap_check():
    x = torch.tensor([[0., 0.], [1., 1.]])
    rows = []
    for slope in (.5, 1.25, 2.):
        critic = LinearCritic(slope)
        penalty = GradientPenalty('b_cap', coeff=6., kappa=1.25)(critic, x, x)
        penalty.backward()
        rows.append(dict(input_gradient_norm=slope, penalty=float(penalty.detach()),
                         parameter_gradient=critic.weight.grad.tolist()))
    return rows


def energy_distance_gradient_check():
    # Generic two-cluster 1D example. Generated values are closer together
    # than the real values, so the unbiased energy-gradient should spread them.
    real = torch.tensor([[-.18], [.18]]).repeat(16, 1)
    fake = torch.tensor([[-.02], [.02]]).repeat(16, 1).requires_grad_()
    offdiag = ~torch.eye(len(fake), dtype=torch.bool)
    energy = (2*torch.cdist(fake, real).mean()
              - torch.cdist(fake, fake)[offdiag].mean()
              - torch.cdist(real, real)[offdiag].mean())
    energy.backward()
    return dict(batch=len(fake), energy=float(energy.detach()),
                negative_group_mean_gradient=float(fake.grad[::2].mean()),
                positive_group_mean_gradient=float(fake.grad[1::2].mean()),
                interpretation='gradient descent moves the negative group lower and positive group higher')


def main():
    torch.set_num_threads(1)
    output = dict(kind='analytic_read_only_no_training',
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in SOURCES},
                  latent_spread=latent_spread_counterexample(),
                  b_cap=one_sided_cap_check(),
                  proposed_energy_term=energy_distance_gradient_check())
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
