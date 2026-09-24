"""Graph-preserving b_cap for read-only virtual-critic response diagnostics.

The frozen regularizer detaches both real and fake coordinates before taking
an input gradient. That is correct for its ordinary D update, but a virtual
D update differentiated back through fake = G(z) would omit the penalty's
mixed G-to-D term. This helper changes only that fake-coordinate detach.
It never replaces the production regularizer or performs an optimizer step.
"""

import torch
from torch.nn import functional as F

from particlegan.grad_regularizers import _score_scalar


def functional_b_cap(regularizer, D, real, fake, step):
    """Return the native sharp b_cap value with its fake-to-D graph intact.

    ``D`` may be a module or a callable wrapping ``functional_call`` with
    virtual D parameters. The real branch remains detached exactly as in the
    original D loss. For a numerically identical detached fake, both the
    value and D-parameter gradient match the frozen regularizer.
    """
    if (regularizer.arm != 'b_cap' or regularizer.method != 'autograd'
            or regularizer.norm != 'l2' or regularizer.lazy_k != 1
            or regularizer.target_anneal != 'none'):
        raise ValueError('functional b_cap is scoped to native constant autograd L2 b_cap')
    if not torch.is_grad_enabled():
        raise RuntimeError('functional b_cap requires autograd for its mixed fake-to-D term')

    def input_norm(x, *, preserve):
        tracked = (x.clone() if preserve else x.detach().clone()).requires_grad_(True)
        logits = D(tracked)
        input_grad = torch.autograd.grad(_score_scalar(logits), tracked,
                                         create_graph=True)[0]
        return torch.sqrt(input_grad.pow(2).flatten(1).sum(dim=1) + 1e-12)

    real_norm = input_norm(real, preserve=False)
    fake_norm = input_norm(fake, preserve=True)
    cap = regularizer.center(step)
    return (regularizer.coeff / 2.0) * (
        F.relu(real_norm - cap).pow(2).mean()
        + F.relu(fake_norm - cap).pow(2).mean())
