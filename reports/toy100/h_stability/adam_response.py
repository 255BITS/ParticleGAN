"""Fixed Adam response proposals; no new training signal or rate schedule."""
from contextlib import contextmanager
from unittest.mock import patch

import torch
import critic_signal
import continuous_candidates
from continuous_candidates import candidate_update
from selected_h_extension import extended_signal_policy

PROPOSALS = {
    'eps_all_1m': dict(network_eps=.001, prior_eps=.001),
    'eps_gp_1m': dict(network_eps=1e-8, prior_eps=.001, g_eps=.001),
    'eps_d_1m': dict(network_eps=1e-8, prior_eps=1e-8, d_eps=.001),
    'tensor_g': dict(network_eps=1e-8, prior_eps=1e-8),
    'tensor_gp': dict(network_eps=1e-8, prior_eps=1e-8),
    'tensor_all': dict(network_eps=1e-8, prior_eps=1e-8),
    'eps_net_1m': dict(network_eps=.001, prior_eps=1e-8),
    'split_prior': dict(network_eps=1e-8, prior_eps=1e-8),
    'eps_net_split_prior': dict(network_eps=.001, prior_eps=1e-8),
}
GEOMETRY_ROLES = {'tensor_g': ('g',), 'tensor_gp': ('g', 'prior'),
                  'tensor_all': ('g', 'prior', 'd')}


class BlockObserver(continuous_candidates._EpsilonObserver):
    """Keep Adam's moments; pool second moments over each parameter tensor.

    The fixed partition is determined by the frozen host's parameter tensors.
    Every direction has the positive multiplier 1/(sqrt(mean(v_hat))+epsilon).
    No gradient calls, history resets, target information or gains are added.
    """
    def __init__(self, eps, roles):
        super().__init__(eps)
        self.roles = roles
        self.receipt['moment_geometry'] = dict(
            partition='parameter_tensor', roles=list(roles),
            formula='delta = -lr * m_hat / (sqrt(mean(v_hat)) + eps)',
            positive_all_directions=True, state='unchanged coordinate Adam moments',
            additional_gradient_evaluations=0, updates=[])

    def step(self, optimizer, original_step, closure=None):
        def block_step(opt):
            selected = []
            for group in opt.param_groups:
                role = self._group_role(group, self.optimizer_roles[id(opt)])
                if role in self.roles:
                    selected.extend((p, p.detach().clone(), group, role)
                                    for p in group['params'] if p.grad is not None)
            result = original_step(opt)
            with torch.no_grad():
                denominators = []
                for p, anchor, group, role in selected:
                    state = opt.state[p]
                    step = int(state['step'])
                    beta1, beta2 = group['betas']
                    denominator = (state['exp_avg_sq'].mean() / (1-beta2**step)).sqrt() + group['eps']
                    p.copy_(anchor).addcdiv_(state['exp_avg'], denominator,
                        value=-group['lr']/(1-beta1**step))
                    denominators.append(dict(role=role, shape=list(p.shape),
                        denominator=float(denominator), lr=float(group['lr']), step=step))
                self.receipt['moment_geometry']['updates'].append(denominators)
            return result
        return super().step(optimizer, block_step, closure)


class SplitObserver(continuous_candidates._EpsilonObserver):
    """Orthogonal centroid/relative particle blocks with separate Adam moments.

    Per latent coordinate, P = I_centroid/denom_c + I_relative/denom_r is
    positive definite. Both denominators are positive; neither subspace freezes.
    Cold moments start at zero. When screening an older state, new block moments
    start from its mean coordinate second moment, without resetting Adam state.
    Same-policy checkpoints retain the two additional tensors exactly.
    """
    def __init__(self, eps):
        super().__init__(eps)
        self.receipt['moment_geometry'] = dict(
            partition='particle centroid and orthogonal relative space per latent coordinate',
            roles=['prior'], positive_all_directions=True,
            formula='delta=-lr*(mean(m_hat)/denom_c + (m_hat-mean(m_hat))/denom_r)',
            cold_initialization='zero block moments',
            diagnostic_initialization='new block moments inherit mean coordinate exp_avg_sq',
            state_keys=['response_centroid_sq', 'response_relative_sq'], updates=[])

    def step(self, optimizer, original_step, closure=None):
        def split_step(opt):
            selected = []
            for group in opt.param_groups:
                role = self._group_role(group, self.optimizer_roles[id(opt)])
                if role != 'prior':
                    continue
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if p.ndim < 2 or group['betas'][0] != 0.:
                        raise ValueError('split geometry requires particle axis and beta1=0')
                    old = opt.state.get(p, {}).get('exp_avg_sq')
                    initial = torch.zeros_like(p[:1]) if old is None else old.mean(0, keepdim=True).clone()
                    selected.append((p, p.detach().clone(), group, initial))
            result = original_step(opt)
            with torch.no_grad():
                rows = []
                for p, anchor, group, initial in selected:
                    state = opt.state[p]; step = int(state['step']); beta2 = group['betas'][1]
                    centroid = p.grad.mean(0, keepdim=True)
                    relative = p.grad-centroid
                    vc = state.setdefault('response_centroid_sq', initial.clone())
                    vr = state.setdefault('response_relative_sq', initial.clone())
                    vc.mul_(beta2).addcmul_(centroid, centroid, value=1-beta2)
                    vr.mul_(beta2).add_(relative.square().mean(0, keepdim=True), alpha=1-beta2)
                    dc = (vc/(1-beta2**step)).sqrt()+group['eps']
                    dr = (vr/(1-beta2**step)).sqrt()+group['eps']
                    direction = centroid/dc + relative/dr
                    p.copy_(anchor).add_(direction, alpha=-group['lr'])
                    rows.append(dict(step=step, lr=float(group['lr']), shape=list(p.shape),
                        centroid_denom_min=float(dc.min()), relative_denom_min=float(dr.min())))
                self.receipt['moment_geometry']['updates'].append(rows)
            return result
        return super().step(optimizer, split_step, closure)


@contextmanager
def response_policy(options):
    options = dict(options)
    name = options.pop('adam_response')
    eps = PROPOSALS[name]
    # Reuse the observer's ownership, full applied-rate trace and numerical Adam.
    # This replaces its epsilon declaration, never nests two optimizer adapters.
    observer = continuous_candidates._EpsilonObserver
    factory = (lambda values: BlockObserver(values, GEOMETRY_ROLES[name])) if name in GEOMETRY_ROLES else observer
    if name in ('split_prior', 'eps_net_split_prior'):
        factory = SplitObserver
    with patch.object(continuous_candidates, '_EpsilonObserver', factory), \
         patch.object(critic_signal, 'observe_adam', lambda _: candidate_update(eps)):
        with extended_signal_policy(options) as receipt:
            receipt['adam_response'] = dict(name=name, epsilon=eps,
                formula=(receipt.get('moment_geometry', {}).get('formula') or
                         'm_hat / (sqrt(v_hat) + fixed_positive_epsilon)'),
                new_adversarial_signal=False, fixed_positive_preconditioner=True,
                generator_objective='inherited logistic relativistic discriminator objective only')
            receipt['host_extension']['archived_candidate_unchanged'] = False
            yield receipt
