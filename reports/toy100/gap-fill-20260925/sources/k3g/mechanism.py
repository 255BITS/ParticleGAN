"""K3G: K3 (one critic rule, a_r1r2 -> b_cap keyed on the critic's applied LR) with the SAME matured-tensor guard applied to
every optimizer's network tensors (critic, generator, host networks), not only the critic. Particle tables are excluded:
registered ParticlePrior parameters (response.prior_ids) and direct-particle groups (group['_comparison_prior']).

pen = s * a_r1r2(RMS units) + (1 - s) * b_cap, where
  r = lr the critic's Adam group actually applied on its last step / max applied so far (global post-step hook),
  s = max(0, min(1, 2r) - 2f) / (1 - 2f), f = NETWORK_FLOOR = .01 (frozen network_lr_floor), s < 1e-6 -> 0.
s == 1 runs the a_r1r2 code path unchanged; s == 0 runs b_cap exactly (no residual R1 at the floor).
Guard: before each Adam step (any optimizer), a non-particle tensor with >= 200 prior updates has g *= C / q when
q = RMS(g) / sqrt(mean(v_hat_prev)) > C, C = 5 (beta1 = 0, so q is the Adam update size in lr units).
No task names: the critic is found from D (module, bound method or closure) and its optimizer from the hooks.
"""
import torch
import torch.nn.functional as F
import response
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
from particlegan.grad_regularizers import GradientPenalty

NETWORK_FLOOR = 0.01
GUARD_C = 5.0
GUARD_MATURITY = 200
_original_penalty = GradientPenalty.penalty
receipt = {'calls': 0, 'dimensions': {}, 'extra_critic_forwards': 0, 'a_only': 0, 'blend': 0, 'b_only': 0,
           'first_blend_step': None, 'first_b_only_step': None, 'unresolved_critic': 0,
           'guard_steps': 0, 'guard_clips': 0, 'guard_clips_critic': 0, 'guard_clips_other': 0, 'guard_clip_steps': [], 'guard_max_q': 0.0, 's_trace': []}
_critic_ids = set()        # ids of every tensor seen as a critic parameter
_applied = {}              # id(param) -> lr applied on its optimizer's last step
_max_applied = {}          # id(param) -> max applied lr so far
_updates = {}              # id(param) -> completed Adam updates
_critic_opt = {}           # id(opt) -> bool
_guard_open = set()        # optimizers whose pre-hook ran and post-hook has not yet


def _modules_in(obj, depth=0):
    if isinstance(obj, torch.nn.Module):
        return [obj]
    if depth > 2:
        return []
    found = []
    owner = getattr(obj, '__self__', None)
    if owner is not None:
        found += _modules_in(owner, depth + 1)
    for cell in (getattr(obj, '__closure__', None) or ()):
        try:
            found += _modules_in(cell.cell_contents, depth + 1)
        except ValueError:
            pass
    return found


def _critic_params(D):
    params = []
    for module in _modules_in(D):
        params += [p for p in module.parameters() if p.requires_grad]
    return params


def _is_critic(opt):
    key = id(opt)
    if key not in _critic_opt:
        _critic_opt[key] = any(id(p) in _critic_ids for g in opt.param_groups for p in g['params'])
    return _critic_opt[key]


def _pre(opt, args, kwargs):
    if id(opt) in _guard_open:
        return None
    _guard_open.add(id(opt))
    critic = bool(_critic_ids) and _is_critic(opt)
    receipt['guard_steps'] += int(critic)
    for group in opt.param_groups:
        if group.get('_comparison_prior', False):
            continue
        beta2 = group['betas'][1]
        for p in group['params']:
            n = _updates.get(id(p), 0)
            state = opt.state.get(p)
            if id(p) in response.prior_ids or p.grad is None or n < GUARD_MATURITY or not state or 'exp_avg_sq' not in state:
                continue
            v_hat = state['exp_avg_sq'] / (1.0 - beta2 ** n)
            q = float(p.grad.square().mean().sqrt() / v_hat.mean().sqrt())
            receipt['guard_max_q'] = max(receipt['guard_max_q'], q)
            if q > GUARD_C:
                p.grad.mul_(GUARD_C / q)
                receipt['guard_clips'] += 1
                receipt['guard_clips_critic' if critic else 'guard_clips_other'] += 1
                if len(receipt['guard_clip_steps']) < 60:
                    receipt['guard_clip_steps'].append(['D' if critic else 'G', receipt['guard_steps'], list(p.shape), round(q, 3)])
    return None


def _post(opt, args, kwargs):
    _guard_open.discard(id(opt))
    for group in opt.param_groups:
        lr = float(group['lr'])
        for p in group['params']:
            if p.grad is not None:
                _updates[id(p)] = _updates.get(id(p), 0) + 1
            _applied[id(p)] = lr
            _max_applied[id(p)] = max(_max_applied.get(id(p), 0.0), lr)


register_optimizer_step_pre_hook(_pre)
register_optimizer_step_post_hook(_post)


def _share(D):
    params = _critic_params(D)
    if not params:
        receipt['unresolved_critic'] += 1
        return 1.0
    for p in params:
        if id(p) not in _critic_ids:
            _critic_ids.add(id(p))
            _critic_opt.clear()
    key = id(params[0])
    if key not in _applied or _max_applied[key] <= 0:
        return 1.0
    r = _applied[key] / _max_applied[key]
    s = max(0.0, min(1.0, 2.0 * r) - 2.0 * NETWORK_FLOOR) / (1.0 - 2.0 * NETWORK_FLOOR)
    return 0.0 if s < 1e-6 else min(1.0, s)


def _a_r1r2(self, D, x_real, x_fake, coefficient):
    dimension = x_real[0].numel()
    assert dimension == x_fake[0].numel()
    real_squared = self._grad_norm(D, x_real, squared=True) / dimension
    fake_norm = self._grad_norm(D, x_fake, squared=False) / dimension ** 0.5
    fake_cap = (fake_norm - self.kappa).relu().square()
    receipt['dimensions'][str(dimension)] = receipt['dimensions'].get(str(dimension), 0) + 1
    return (coefficient / 2.0) * (real_squared.mean() + fake_cap.mean())


def _b_cap(self, D, x_real, x_fake, coefficient):
    assert self.target_anneal == 'none' and self.norm == 'l2'
    n_r = self._grad_norm(D, x_real, squared=False)
    n_f = self._grad_norm(D, x_fake, squared=False)
    return (coefficient / 2.0) * (F.relu(n_r - self.kappa).pow(2).mean() + F.relu(n_f - self.kappa).pow(2).mean())


def blended_penalty(self, D, x_real, x_fake, step=1, generator=None, collect_stats=True):
    if self.arm != 'a_r1r2' or (self.lazy_k > 1 and step % self.lazy_k != 0):
        return _original_penalty(self, D, x_real, x_fake, step, generator, collect_stats)
    coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
    s = _share(D)
    receipt['calls'] += 1
    if s >= 1.0:
        receipt['a_only'] += 1
        penalty = _a_r1r2(self, D, x_real, x_fake, coefficient)
    elif s <= 0.0:
        receipt['b_only'] += 1
        if receipt['first_b_only_step'] is None:
            receipt['first_b_only_step'] = [receipt['calls'], step]
        penalty = _b_cap(self, D, x_real, x_fake, coefficient)
    else:
        receipt['blend'] += 1
        receipt['extra_critic_forwards'] += 2
        if receipt['first_blend_step'] is None:
            receipt['first_blend_step'] = [receipt['calls'], step]
        penalty = s * _a_r1r2(self, D, x_real, x_fake, coefficient) + (1.0 - s) * _b_cap(self, D, x_real, x_fake, coefficient)
    if receipt['calls'] % 100 == 0 and len(receipt['s_trace']) < 200:
        receipt['s_trace'].append([receipt['calls'], round(s, 5)])
    stats = {'applied': True, 'pen': float(penalty.detach()), 'fake_cap': self.kappa, 'share_r1r2': s} if collect_stats else {}
    return penalty, stats


GradientPenalty.penalty = blended_penalty
