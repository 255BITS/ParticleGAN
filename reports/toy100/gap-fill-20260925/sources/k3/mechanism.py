"""P1 stepkey: one critic rule, keyed on the critic's own applied Adam step size, independent of host identity.

    pen = s * a_r1r2 + (1 - s) * b_cap
    s   = clip((min(1, 2 r) - 2 f) / (1 - 2 f), 0, 1)
    r   = LR the critic's last Adam step applied / max LR it has applied, f = .01 (frozen base network_lr_floor)

a_r1r2 is the RMS-unit zero-centered real R1 plus fake cap (the session a_r1r2 mechanism, bit-exact at s = 1);
b_cap is the library one-sided L2 cap relu(|g| - kappa)^2 on both sides (bit-exact at s = 0).
While the critic step is large the zero-centered real term damps allocation; at the floor the critic is free
to keep a slope at real data. Before the first critic step s = 1.

Guard (matured, critic only): before each critic Adam step, a tensor with >= 200 prior steps gets
g *= C / q when q = RMS(g) / sqrt(mean vhat_prev) > C, C = 5. q is the would-be update in LR units.
"""
import json
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
import torch.nn.functional as F
from particlegan.grad_regularizers import GradientPenalty

_original_penalty = GradientPenalty.penalty
FLOOR = 0.01
GUARD_C = 5.0
GUARD_MATURITY = 200
receipt = {'calls': 0, 'dimensions': {}, 'extra_critic_forwards': 0, 'a_calls': 0, 'blend_calls': 0, 'b_calls': 0,
           'first_blend_call': None, 'first_b_call': None, 's_last': 1.0, 'critic_steps': 0, 'critic_lr_max': None,
           'critic_lr_last': None, 'guard_clips': 0, 'guard_clip_steps': [], 'guard_q_max': 0.0}
_state = {'pending': False, 'critic': None}


def _event(**kw):
    print(json.dumps(dict(event='P1', **kw)), flush=True)


def _share():
    lr_max, lr = receipt['critic_lr_max'], receipt['critic_lr_last']
    if not lr_max:
        return 1.0
    r = lr / lr_max
    return min(1.0, max(0.0, (min(1.0, 2.0 * r) - 2.0 * FLOOR) / (1.0 - 2.0 * FLOOR)))


def _post_step(opt, args, kwargs):
    if _state['critic'] is None and _state['pending']:
        _state['critic'] = opt
        _event(what='critic_optimizer', groups=len(opt.param_groups), lr=opt.param_groups[0]['lr'])
    _state['pending'] = False
    if opt is _state['critic']:
        lr = float(opt.param_groups[0]['lr'])
        receipt['critic_steps'] += 1
        receipt['critic_lr_last'] = lr
        receipt['critic_lr_max'] = lr if receipt['critic_lr_max'] is None else max(receipt['critic_lr_max'], lr)


def _pre_step(opt, args, kwargs):
    if opt is not _state['critic']:
        return
    for group in opt.param_groups:
        beta2 = group['betas'][1]
        for p in group['params']:
            state = opt.state.get(p)
            if p.grad is None or not state or 'exp_avg_sq' not in state:
                continue
            t = float(state['step'])
            if t < GUARD_MATURITY:
                continue
            vhat = state['exp_avg_sq'].mean() / (1.0 - beta2 ** t)
            q = float(p.grad.pow(2).mean().sqrt() / vhat.sqrt())
            receipt['guard_q_max'] = max(receipt['guard_q_max'], q)
            if q > GUARD_C:
                p.grad.mul_(GUARD_C / q)
                receipt['guard_clips'] += 1
                if len(receipt['guard_clip_steps']) < 200:
                    receipt['guard_clip_steps'].append(receipt['critic_steps'] + 1)
                if receipt['guard_clips'] <= 20:
                    _event(what='guard_clip', critic_step=receipt['critic_steps'] + 1, q=q, numel=p.numel())


register_optimizer_step_pre_hook(_pre_step)
register_optimizer_step_post_hook(_post_step)


def scaled_penalty(self, D, x_real, x_fake, step=1, generator=None, collect_stats=True):
    if self.arm != 'a_r1r2' or (self.lazy_k > 1 and step % self.lazy_k != 0):
        return _original_penalty(self, D, x_real, x_fake, step, generator, collect_stats)
    _state['pending'] = True
    coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
    dimension = x_real[0].numel()
    assert dimension == x_fake[0].numel()
    s = _share()
    receipt['calls'] += 1
    receipt['dimensions'][str(dimension)] = receipt['dimensions'].get(str(dimension), 0) + 1
    receipt['s_last'] = s
    if s == 1.0:
        receipt['a_calls'] += 1
        real_squared = self._grad_norm(D, x_real, squared=True) / dimension
        fake_norm = self._grad_norm(D, x_fake, squared=False) / dimension ** 0.5
        fake_cap = (fake_norm - self.kappa).relu().square()
        penalty = (coefficient / 2.0) * (real_squared.mean() + fake_cap.mean())
    else:
        self.arm = 'b_cap'
        try:
            center = self.center(step)
        finally:
            self.arm = 'a_r1r2'
        if s == 0.0:
            receipt['b_calls'] += 1
            if receipt['first_b_call'] is None:
                receipt['first_b_call'] = receipt['calls']
                _event(what='first_b_call', call=receipt['calls'], critic_step=receipt['critic_steps'])
            n_r = self._grad_norm(D, x_real, squared=False)
            n_f = self._grad_norm(D, x_fake, squared=False)
            penalty = (coefficient / 2.0) * (F.relu(n_r - center).pow(2).mean() + F.relu(n_f - center).pow(2).mean())
        else:
            receipt['blend_calls'] += 1
            if receipt['first_blend_call'] is None:
                receipt['first_blend_call'] = receipt['calls']
                _event(what='first_blend_call', call=receipt['calls'], critic_step=receipt['critic_steps'], s=s)
            sq_r = self._grad_norm(D, x_real, squared=True)
            n_r = torch.sqrt(sq_r + 1e-12)
            n_f = self._grad_norm(D, x_fake, squared=False)
            a = (sq_r / dimension).mean() + (n_f / dimension ** 0.5 - self.kappa).relu().square().mean()
            b = F.relu(n_r - center).pow(2).mean() + F.relu(n_f - center).pow(2).mean()
            penalty = (coefficient / 2.0) * (s * a + (1.0 - s) * b)
    stats = {'applied': True, 'pen': float(penalty.detach()), 'fake_cap': self.kappa, 's': s} if collect_stats else {}
    return penalty, stats


GradientPenalty.penalty = scaled_penalty
