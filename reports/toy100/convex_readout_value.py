"""Convex fixed-feature Rp/cap readout, with a coercive-sublevel gap bound.

This is a diagnostic helper, not a training adapter. No penalty is added:
whitening changes coordinates in the existing readout objective only.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.func import jvp
from torch.nn import functional as F


CAP_EPS = 1e-12
MAX_ITERATIONS = 100
MAX_CLOSURES = 200
GAP_TOLERANCE = 1e-7


class FrozenFeatures(nn.Module):
    def __init__(self, critic):
        super().__init__()
        if not isinstance(critic.net[-1], nn.Linear) or critic.net[-1].out_features != 1:
            raise ValueError('expected a scalar final affine readout')
        self.net = deepcopy(critic.net[:-1])
        self.fourier = critic.fourier
        if self.fourier:
            self.register_buffer('freqs', critic.freqs.detach().clone())
        self.requires_grad_(False)

    def forward(self, x):
        values = x
        if self.fourier:
            phase = x.unsqueeze(-1)*self.freqs
            values = torch.cat((values, phase.sin().flatten(1), phase.cos().flatten(1)), dim=1)
        return self.net(values)


def feature_arrays(features, points):
    """Each point is independent; two batch JVPs give its input Jacobian."""
    columns = []
    for axis in range(points.shape[1]):
        direction = torch.zeros_like(points)
        direction[:, axis] = 1
        value, derivative = jvp(features, (points,), (direction,))
        columns.append(derivative.detach())
    return value.detach(), torch.stack(columns, dim=1)


class ReadoutProblem:
    def __init__(self, delta, jacobian, *, coeff=1., kappa=1.):
        if delta.dtype != torch.float64 or jacobian.dtype != torch.float64:
            raise ValueError('the convex diagnostic uses float64 throughout')
        if coeff <= 0 or kappa <= CAP_EPS**.5:
            raise ValueError('positive native cap with a free constant critic required')
        self.delta, self.jacobian = delta.detach(), jacobian.detach()
        self.coeff, self.kappa = coeff, kappa
        self.gram = torch.einsum('nid,nie->de', jacobian, jacobian)/len(jacobian)
        self.eigenvalues, self.eigenvectors = torch.linalg.eigh(self.gram)
        threshold = 64*torch.finfo(torch.float64).eps*len(self.gram)*float(self.eigenvalues[-1])
        self.full_rank = bool(self.eigenvalues[0] > threshold > 0)
        # A floor changes only the invertible solver coordinate map. A
        # deficient S does NOT receive a global gap certificate.
        floor = max(threshold, torch.finfo(torch.float64).tiny)
        eigenvalues = self.eigenvalues.clamp_min(floor)
        self.root = (self.eigenvectors*eigenvalues.sqrt())@self.eigenvectors.T
        self.inverse_root = (self.eigenvectors*eigenvalues.rsqrt())@self.eigenvectors.T
        self.rank_threshold = threshold

    @classmethod
    def from_points(cls, features, real, fake, *, coeff=1., kappa=1.):
        real_phi, real_jac = feature_arrays(features, real)
        fake_phi, fake_jac = feature_arrays(features, fake)
        return cls(fake_phi-real_phi, torch.cat((real_jac, fake_jac)), coeff=coeff, kappa=kappa)

    def loss(self, weight):
        logistic = F.softplus(self.delta@weight).mean()
        input_gradient = torch.einsum('nid,d->ni', self.jacobian, weight)
        norm = (input_gradient.square().sum(1)+CAP_EPS).sqrt()
        cap = self.coeff*F.relu(norm-self.kappa).square().mean()
        return logistic+cap

    def certificate(self, weight, gradient, loss, upper):
        if not self.full_rank:
            return dict(available=False, reason='input-Jacobian Gram is not numerically full rank')
        # relu(sqrt(||Jw||²+eps)-kappa)² >= .5||Jw||²-kappa².
        # An optimum has loss<=upper and thus ||w*||_S<=radius.
        radius = math.sqrt(2*(upper/self.coeff+self.kappa**2))
        dual2 = float(((self.eigenvectors.T@gradient).square()/self.eigenvalues).sum())
        lower = max(0., loss-float(gradient@weight)-radius*math.sqrt(max(dual2, 0.)))
        gap = upper-lower
        tolerance = 1e-10*max(1., abs(upper), abs(lower))
        if gap < -tolerance:
            raise FloatingPointError('computed convex lower bound exceeds an evaluated upper bound')
        return dict(available=True, lower=lower, upper=upper, gap=max(gap, 0.),
                    sublevel_radius=radius, dual_gradient_norm=math.sqrt(max(dual2, 0.)),
                    scope='convex global bound under the finite cached objective; float64 arithmetic, not interval arithmetic')

    def geometry(self):
        return dict(input_gram_full_rank=self.full_rank,
                    eigenvalue_min=float(self.eigenvalues[0]), eigenvalue_max=float(self.eigenvalues[-1]),
                    numerical_rank=int((self.eigenvalues>self.rank_threshold).sum()),
                    rank_threshold=self.rank_threshold,
                    whitening_error=float((self.inverse_root@self.gram@self.inverse_root-
                                           torch.eye(len(self.gram), dtype=torch.float64)).abs().max()),
                    bias='omitted: exact additive output gauge in Rp logits and input-gradient cap')


class _StopFit(Exception):
    pass


def fit_readout(problem, initial):
    """One L-BFGS attempt; at most200 closures, no objective regularization."""
    u = (problem.root@initial.detach()).clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([u], lr=1., max_iter=MAX_ITERATIONS,
        max_eval=MAX_CLOSURES, history_size=50, tolerance_grad=1e-12,
        tolerance_change=1e-15, line_search_fn='strong_wolfe')
    rows, best = [], None
    reason = 'OPTIMIZER_RETURN'

    def closure():
        nonlocal best, reason
        if len(rows) >= MAX_CLOSURES:
            reason = 'CLOSURE_BUDGET'
            raise _StopFit
        optimizer.zero_grad()
        weight = problem.inverse_root@u
        loss = problem.loss(weight)
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(u.grad).all():
            reason = 'NONFINITE_TRIAL'
            rows.append(dict(call=len(rows)+1, finite=False))
            if best is None:
                raise FloatingPointError('initial convex readout field is nonfinite')
            raise _StopFit
        gradient = problem.root@u.grad.detach()
        value = float(loss.detach())
        upper = min(math.log(2.), value, best['loss'] if best else math.inf)
        certificate = problem.certificate(weight.detach(), gradient, value, upper)
        row = dict(call=len(rows)+1, finite=True, loss=value,
                   gradient_l2=float(gradient.norm()), whitened_gradient_l2=float(u.grad.norm()),
                   weight_l2=float(weight.detach().norm()), certificate=certificate)
        rows.append(row)
        if best is None or value < best['loss']:
            best = dict(weight=weight.detach().clone(), gradient=gradient.clone(), loss=value,
                        certificate=certificate, call=len(rows))
        if certificate['available'] and certificate['gap'] <= GAP_TOLERANCE:
            # The current point can certify a previously evaluated better one.
            best['certificate'] = certificate
            reason = 'GAP_REACHED'
            raise _StopFit
        return loss

    try:
        optimizer.step(closure)
    except _StopFit:
        pass
    if best is None:
        raise RuntimeError('readout solver evaluated no finite point')
    # Recompute certificate at the returned best state, without an extra
    # optimizer attempt. It is a read-only gradient evaluation and counted.
    weight = best['weight'].clone().requires_grad_(True)
    final_loss = problem.loss(weight)
    final_gradient = torch.autograd.grad(final_loss, weight)[0].detach()
    final_certificate = problem.certificate(weight.detach(), final_gradient, float(final_loss.detach()),
                                          min(math.log(2.), float(final_loss.detach())))
    passed = final_certificate['available'] and final_certificate['gap'] <= GAP_TOLERANCE
    return weight.detach(), dict(status='CERTIFIED_GAP' if passed else 'NOT_CERTIFIED',
        termination=reason, closure_calls=len(rows), final_gradient_evaluations=1,
        lbfgs_iterations=int(optimizer.state[u].get('n_iter', 0)),
        best_call=best['call'], initial_loss=rows[0].get('loss'), final_loss=float(final_loss.detach()),
        raw_gradient_l2=float(final_gradient.norm()),
        whitened_gradient_l2=float((problem.inverse_root@final_gradient).norm()),
        certificate=final_certificate, geometry=problem.geometry(), records=rows)
