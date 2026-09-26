"""K3P: K3 plus an anchored R1 that damps the critic without biasing its equilibrium.

pen = s * a_r1r2 + (1 - s) * (b_cap + prox),  prox = mean ||grad_x D(x_r) - grad_x Dbar(x_r)||^2 / d  (RMS units, reals),
Dbar = EMA of the critic's parameters (decay ANCHOR_DECAY per critic step, started at the first blended call).
Zero-centered R1 anchors the critic's input gradient to 0 (damped but flat at data: imprecise centers); b_cap is inactive
below kappa, so at the floor the critic is unregularized and undamped. prox anchors the gradient to the critic's own recent
average instead: zero at a stationary critic, restoring during oscillation. At s == 1 this is bit-exact a_r1r2 (toys).
Rest of the K3 description:

pen = s * a_r1r2 + (1 - s) * b_cap, both from the same input gradients, with
  a_r1r2 = R1 on reals + one-sided fake cap, RMS units (the session mechanism, bit-exact at s == 1)
  b_cap  = relu(||g|| - kappa)^2 on reals and fakes, L2 units (GradRegularizer 'b_cap', exact at s == 0)
  s = max(0, min(1, 2r) - 2f) / (1 - 2f),  r = critic's applied LR on its last Adam step / max applied LR,
  f = FLOOR, the declared network LR floor, so s == 0 exactly at the floor.
Critic guard: before each critic Adam step, a tensor with >= GUARD_MIN_STEPS prior steps whose gradient RMS
exceeds GUARD_C * sqrt(mean bias-corrected v) is scaled down to that ratio (no-op multiply by 1.0 otherwise).
The critic is the first optimizer stepped after a penalty call. No host or task identity is read.
"""
import atexit, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook
from particlegan.grad_regularizers import GradientPenalty, _score_scalar

FLOOR = 0.01
GUARD_C = 5.0
GUARD_MIN_STEPS = 200
ANCHOR_DECAY = 0.999
_original_penalty = GradientPenalty.penalty
receipt = {'calls': 0, 'dimensions': {}, 'extra_critic_forwards': 0, 'rule': 'k3', 'floor': FLOOR,
           'guard_c': GUARD_C, 'guard_min_steps': GUARD_MIN_STEPS, 'pure_a_calls': 0, 'blend_calls': 0,
           'pure_b_calls': 0, 'first_blend_call': None, 'first_pure_b_call': None, 'critic_steps': 0,
           'critic_parameters': None, 'lr_max': None, 'lr_last': None, 's_trace': [], 'rule': 'k3p', 'anchor_decay': ANCHOR_DECAY,
           'anchor_started_call': None, 'prox_trace': []}
_state = {'ema': None, 'critic': None, 'pending': False, 'lr_max': 0.0, 'lr_last': None, 'depth': 0, 'clips': [], 'critic_ref': None}


def handover_weight():
    if _state['lr_last'] is None or _state['lr_max'] <= 0.0:
        return 1.0
    r = _state['lr_last'] / _state['lr_max']
    return max(0.0, min(1.0, 2.0 * r) - 2.0 * FLOOR) / (1.0 - 2.0 * FLOOR)


def _critic_params():
    return [p for g in _state['critic_ref'].param_groups for p in g['params']]


def anchored_gradient_gap(D, x_real, g, dimension):
    """mean ||g - grad_x Dbar(x_real)||^2 / d; Dbar = parameter EMA, evaluated by swapping .data (restored before return)."""
    if _state['critic_ref'] is None:
        return g.new_zeros(())
    params = _critic_params()
    if _state['ema'] is None:
        _state['ema'] = [p.detach().clone() for p in params]
        receipt['anchor_started_call'] = receipt['calls']
        return g.new_zeros(())
    saved = [p.data for p in params]
    try:
        for p, e in zip(params, _state['ema']):
            p.data = e
        xb = x_real.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            gb = torch.autograd.grad(_score_scalar(D(xb)), xb)[0].detach()
    finally:
        for p, v in zip(params, saved):
            p.data = v
    return (g - gb).pow(2).flatten(1).sum(dim=1).mean() / dimension


def scaled_penalty(self, D, x_real, x_fake, step=1, collect_stats=True, *, generator=None, ema_critic=None):
    # The current K3P penalty takes collect_stats positionally and ema_critic
    # by keyword, and it has no arm. vector_unequal_mass always passes
    # ema_critic. Delegate that call to the penalty this replacement captured.
    if not hasattr(self, "arm"):
        return _original_penalty(self, D, x_real, x_fake, step, collect_stats, ema_critic=ema_critic)
    if self.arm != 'a_r1r2' or (self.lazy_k > 1 and step % self.lazy_k != 0):
        return _original_penalty(self, D, x_real, x_fake, step, collect_stats, ema_critic=ema_critic)
    _state['pending'] = True
    coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
    dimension = x_real[0].numel()
    assert dimension == x_fake[0].numel()
    s = handover_weight()
    receipt['calls'] += 1
    receipt['dimensions'][str(dimension)] = receipt['dimensions'].get(str(dimension), 0) + 1
    if s >= 1.0:
        real_squared = self._grad_norm(D, x_real, squared=True) / dimension
        fake_norm = self._grad_norm(D, x_fake, squared=False) / dimension ** 0.5
        fake_cap = (fake_norm - self.kappa).relu().square()
        penalty = (coefficient / 2.0) * (real_squared.mean() + fake_cap.mean())
        receipt['pure_a_calls'] += 1
    else:
        x = x_real.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(_score_scalar(D(x)), x, create_graph=True)[0]
        sq_r = g.pow(2).flatten(1).sum(dim=1)
        n_r = torch.sqrt(sq_r + 1e-12)
        n_f = self._grad_norm(D, x_fake, squared=False)
        prox = anchored_gradient_gap(D, x_real, g, dimension)
        if s > 0.0:
            a_term = (sq_r / dimension).mean() + (n_f / dimension ** 0.5 - self.kappa).relu().square().mean()
            b_term = F.relu(n_r - self.kappa).pow(2).mean() + F.relu(n_f - self.kappa).pow(2).mean() + prox
            penalty = (coefficient / 2.0) * (s * a_term + (1.0 - s) * b_term)
            receipt['blend_calls'] += 1
            if receipt['first_blend_call'] is None:
                receipt['first_blend_call'] = receipt['calls']
        else:
            penalty = (coefficient / 2.0) * (F.relu(n_r - self.kappa).pow(2).mean() + F.relu(n_f - self.kappa).pow(2).mean() + prox)
            receipt['pure_b_calls'] += 1
            if receipt['first_pure_b_call'] is None:
                receipt['first_pure_b_call'] = receipt['calls']
        if receipt['calls'] % 50 == 0:
            receipt['prox_trace'].append([receipt['calls'], float(prox.detach())])
    if receipt['calls'] == 1 or receipt['calls'] % 50 == 0:
        receipt['s_trace'].append([receipt['calls'], round(s, 6)])
    stats = {'applied': True, 'pen': float(penalty.detach()), 'fake_cap': self.kappa, 's': s} if collect_stats else {}
    return penalty, stats


def _guard(opt, args, kwargs):
    if _state['critic'] is None or id(opt) != _state['critic']:
        return
    _state['depth'] += 1
    if _state['depth'] != 1:
        return
    flags = []
    for group in opt.param_groups:
        beta2 = group['betas'][1]
        for p in group['params']:
            st = opt.state.get(p)
            if p.grad is None or not st or 'exp_avg_sq' not in st:
                continue
            t = st['step']
            vhat = st['exp_avg_sq'].mean() / (1.0 - beta2 ** t)
            ratio = p.grad.square().mean().sqrt() / vhat.clamp_min(1e-30).sqrt()
            clip = (t >= GUARD_MIN_STEPS) & (ratio > GUARD_C)
            p.grad.mul_(torch.where(clip, GUARD_C / ratio, torch.ones_like(ratio)))
            flags.append(clip)
    if flags:
        _state['clips'].append(torch.stack(flags).sum())


def _record(opt, args, kwargs):
    if _state['critic'] is None:
        if not _state['pending']:
            return
        _state['critic'] = id(opt)
        _state['critic_ref'] = opt
        receipt['critic_parameters'] = sum(p.numel() for g in opt.param_groups for p in g['params'])
        _state['depth'] = 1
    if id(opt) != _state['critic']:
        return
    if _state['depth'] == 1:
        if _state['ema'] is not None:
            with torch.no_grad():
                for e, p in zip(_state['ema'], _critic_params()):
                    e.mul_(ANCHOR_DECAY).add_(p.detach(), alpha=1.0 - ANCHOR_DECAY)
        lr = max(float(g['lr']) for g in opt.param_groups)
        _state['lr_last'] = lr
        _state['lr_max'] = max(_state['lr_max'], lr)
        receipt['critic_steps'] += 1
    _state['depth'] = max(0, _state['depth'] - 1)


def _write_receipt():
    receipt['lr_max'] = _state['lr_max']
    receipt['lr_last'] = _state['lr_last']
    receipt['final_s'] = handover_weight()
    if _state['clips']:
        per_step = torch.stack(_state['clips']).cpu()
        steps = torch.nonzero(per_step > 0).flatten().tolist()
        receipt['guard_clipped_tensors'] = int(per_step.sum())
        receipt['guard_clip_steps'] = [s + 1 for s in steps][:200]
        receipt['guard_clip_step_count'] = len(steps)
    else:
        receipt['guard_clipped_tensors'] = 0
        receipt['guard_clip_steps'] = []
        receipt['guard_clip_step_count'] = 0
    if '--output' in sys.argv:
        out = Path(sys.argv[sys.argv.index('--output') + 1])
        if out.is_dir():
            (out / 'mechanism-receipt.json').write_text(json.dumps(receipt, indent=1, default=str) + '\n')


GradientPenalty.penalty = scaled_penalty
register_optimizer_step_pre_hook(_guard)
register_optimizer_step_post_hook(_record)
atexit.register(_write_receipt)
