"""p3_floor_reopen: p2 dwell, plus a floor that does not chase a post-shift rise.

p1 showed critic cosine is already strongly negative by step 400, while only
four modes exist, so a reversal dwell freezes a precise 7-mode cover and a
later reopen toward full rate erases it. This rule leaves the K3P initial rate
and pure R1 in place until the generator+prior gradient RMS falls below 0.05
of its own peak. That 20x drop is the collapse measured between steps 700 and
800 (about 0.015 to 0.00025) when quality first reached HQ 1. Acquisition noise
is the copied K3P schedule, because removing it left the run at 7 modes.
After the dwell, the reversible band is only 0.10 to 0.25 of the initial rate:
0.25 is below the 0.66 fraction that erased modes at step 3600, and 0.10 is
the holding fraction. Reopening requires the gradient RMS to exceed five times
its slow quiet average for a short confirmation. The critic mix locks at the
EMA anchor once the dwell rate is reached and does not follow later rate
changes. Anchor strength is 1 + relu(-cosine). No budget, score, target, or
change time is read.
"""
import atexit, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook
from particlegan.grad_regularizers import GradientPenalty, _score_scalar
import precision_policy

FLOOR = 0.01
GUARD_C = 5.0
GUARD_MIN_STEPS = 200
ANCHOR_DECAY = 0.999
RATIO_DWELL = 0.05
REOPEN_FACTOR = 5.0
CONFIRM_DECAY = 0.9
SLEW = 0.9
PEAK_MIN = 0.005
QUIET_DECAY = 0.98
_original_penalty = GradientPenalty.penalty
receipt = {'calls': 0, 'dimensions': {}, 'extra_critic_forwards': 0, 'floor': FLOOR,
           'guard_c': GUARD_C, 'guard_min_steps': GUARD_MIN_STEPS, 'pure_a_calls': 0, 'blend_calls': 0,
           'pure_b_calls': 0, 'first_blend_call': None, 'first_pure_b_call': None, 'critic_steps': 0,
           'critic_parameters': None, 'lr_max': None, 'lr_last': None, 's_trace': [],
           'rule': 'p3_floor_reopen', 'anchor_decay': ANCHOR_DECAY,
           'anchor_started_call': None, 'prox_trace': [],
           'ratio_dwell': RATIO_DWELL, 'reopen_factor': REOPEN_FACTOR,
           'adapt_fraction': precision_policy.ADAPT, 'dwell_fraction': precision_policy.DWELL,
           'controller_trace': []}
_state = {'ema': None, 'critic': None, 'pending': False, 'lr_max': 0.0, 'lr_last': None, 'depth': 0, 'clips': [],
          'critic_ref': None, 'grad_previous': None, 'restoring_strength': 1.0}


def handover_weight():
    # Explicitly not last/max learning rate. After precision locks, the anchor stays on.
    return float(precision_policy.state['s'])


def _critic_params():
    return [p for g in _state['critic_ref'].param_groups for p in g['params']]


def _flat_grad(opt):
    grads = [p.grad.detach().flatten() for group in opt.param_groups for p in group['params'] if p.grad is not None]
    if not grads:
        return None
    current = torch.cat(grads)
    assert current.device.type == 'cuda' and current.dtype == torch.float32
    return current


def _critic_strength(opt):
    current = _flat_grad(opt)
    if current is None:
        return
    previous = _state['grad_previous']
    _state['grad_previous'] = current.clone()
    if previous is None:
        return
    cosine = float(F.cosine_similarity(current, previous, dim=0, eps=1e-12))
    _state['restoring_strength'] = 1.0 + min(1.0, max(0.0, -cosine))


def _generator_signal(opt):
    current = _flat_grad(opt)
    if current is None:
        return
    rms = float(current.square().mean().sqrt())
    ctrl = precision_policy.state
    ctrl['g_rms'] = rms
    ctrl['g_peak'] = max(ctrl['g_peak'], rms)
    if ctrl['g_quiet'] is None:
        ctrl['g_quiet'] = rms
    else:
        ctrl['g_quiet'] = QUIET_DECAY * ctrl['g_quiet'] + (1.0 - QUIET_DECAY) * rms
    peak = ctrl['g_peak']
    quiet = ctrl['g_quiet']
    ratio = (quiet / peak) if peak > 0.0 else 1.0
    phase = ctrl['phase']
    if phase == 'acquire':
        # Quiet average, not one step: p1's generator RMS dipped under 0.05 of
        # its peak near step 500 while only a few modes existed.
        cool = peak >= PEAK_MIN and ratio < RATIO_DWELL
        ctrl['confirm'] = CONFIRM_DECAY * ctrl['confirm'] + (1.0 - CONFIRM_DECAY) * (1.0 if cool else 0.0)
        if ctrl['confirm'] > 0.5:
            ctrl['phase'] = 'dwell'
            ctrl['confirm'] = 0.0
            ctrl['g_floor'] = quiet
    elif phase == 'dwell':
        # p2 compared RMS with a quiet EMA that chased the post-shift rise, so
        # generator RMS could climb 14x (0.00026 to 0.0036) and the rate stayed
        # at the dwell. The floor creeps upward only, and is reset when dwelling.
        if ctrl['g_floor'] is None:
            ctrl['g_floor'] = quiet
        elif rms < ctrl['g_floor']:
            ctrl['g_floor'] = 0.98 * ctrl['g_floor'] + 0.02 * rms
        else:
            ctrl['g_floor'] = 0.999 * ctrl['g_floor'] + 0.001 * rms
        hot = ctrl['g_floor'] > 0.0 and rms > REOPEN_FACTOR * ctrl['g_floor']
        ctrl['confirm'] = CONFIRM_DECAY * ctrl['confirm'] + (1.0 - CONFIRM_DECAY) * (1.0 if hot else 0.0)
        if ctrl['confirm'] > 0.5:
            ctrl['phase'] = 'adapt'
            ctrl['confirm'] = 0.0
    else:
        cool = peak >= PEAK_MIN and ratio < RATIO_DWELL
        ctrl['confirm'] = CONFIRM_DECAY * ctrl['confirm'] + (1.0 - CONFIRM_DECAY) * (1.0 if cool else 0.0)
        if ctrl['confirm'] > 0.5:
            ctrl['phase'] = 'dwell'
            ctrl['confirm'] = 0.0
            ctrl['g_floor'] = quiet
    target = {'acquire': 1.0, 'dwell': precision_policy.DWELL, 'adapt': precision_policy.ADAPT}[ctrl['phase']]
    ctrl['multiplier'] = SLEW * ctrl['multiplier'] + (1.0 - SLEW) * target
    if not ctrl['precision_locked']:
        ctrl['s'] = min(1.0, max(0.0, (ctrl['multiplier'] - precision_policy.DWELL) / (1.0 - precision_policy.DWELL)))
        if ctrl['phase'] == 'dwell' and ctrl['multiplier'] < precision_policy.DWELL + 0.02:
            ctrl['precision_locked'] = True
            ctrl['s'] = 0.0
    else:
        ctrl['s'] = 0.0
    if receipt['calls'] % 50 == 0:
        receipt['controller_trace'].append(dict(
            call=receipt['calls'], rms=rms, peak=ctrl['g_peak'], quiet=ctrl['g_quiet'], floor=ctrl['g_floor'],
            ratio=ratio, phase=ctrl['phase'], confirm=ctrl['confirm'],
            multiplier=ctrl['multiplier'], s=ctrl['s'], precision_locked=ctrl['precision_locked'],
            restoring_strength=_state['restoring_strength']))


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
            receipt['extra_critic_forwards'] += 1
    finally:
        for p, v in zip(params, saved):
            p.data = v
    return (g - gb).pow(2).flatten(1).sum(dim=1).mean() / dimension


def scaled_penalty(self, D, x_real, x_fake, step=1, generator=None, collect_stats=True):
    if self.arm != 'a_r1r2' or (self.lazy_k > 1 and step % self.lazy_k != 0):
        return _original_penalty(self, D, x_real, x_fake, step, generator, collect_stats)
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
        prox = anchored_gradient_gap(D, x_real, g, dimension) * _state['restoring_strength']
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
            receipt['prox_trace'].append([receipt['calls'], float(prox.detach()), round(_state['restoring_strength'], 6)])
    if receipt['calls'] == 1 or receipt['calls'] % 50 == 0:
        receipt['s_trace'].append([receipt['calls'], round(s, 6), round(precision_policy.state['multiplier'], 6), precision_policy.state['phase']])
    stats = {'applied': True, 'pen': float(penalty.detach()), 'fake_cap': self.kappa, 's': s} if collect_stats else {}
    return penalty, stats


def _guard(opt, args, kwargs):
    if _state['critic'] is not None and id(opt) != _state['critic']:
        with torch.no_grad():
            _generator_signal(opt)
        return
    if _state['critic'] is None or id(opt) != _state['critic']:
        return
    _state['depth'] += 1
    if _state['depth'] != 1:
        return
    with torch.no_grad():
        _critic_strength(opt)
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
    receipt['final_multiplier_state'] = precision_policy.state['multiplier']
    receipt['final_phase'] = precision_policy.state['phase']
    receipt['precision_locked'] = precision_policy.state['precision_locked']
    receipt['final_g_rms'] = precision_policy.state['g_rms']
    receipt['final_g_peak'] = precision_policy.state['g_peak']
    receipt['final_confirm'] = precision_policy.state['confirm']
    receipt['multiplier_trace'] = precision_policy.state['multiplier_trace']
    receipt['final_multiplier'] = precision_policy.state['last_multiplier']
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
