"""ap3: after the anchor latches, a magnitude reopen restores a partial rate only.

ap1 reopened to full rate and the early penalty and stuck on 4 precise modes.
ap2 kept the anchor (s=0) but reopened to full rate: modes moved (up to 7) and
oscillated, and the deadline still failed. Cold acquisition is unchanged: rate
gain and mixing gain stay tied until mixing gain first hits 0.

A later reopen sets rate gain to 0.2, not 1. Applied multipliers are then
network 0.01+0.99*0.2 = 0.208 and prior 0.05+0.95*0.2 = 0.24. Mixing gain stays
0, so the cap and the EMA anchor stay on. That partial rate is held for 800
critic steps, and the peak is reset to the slow RMS estimate 200 steps in.
Critic cosine, quality scores, targets, shift time, and the training budget are
not read. Host cosine arguments are ignored.

Input and output noise stay on the driver horizon of 1200 (labeled ablation).
"""
import atexit, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook
from particlegan.grad_regularizers import GradientPenalty, _score_scalar

FLOOR = 0.01
PRIOR_FLOOR = 0.05
GUARD_C = 5.0
GUARD_MIN_STEPS = 200
ANCHOR_DECAY = 0.999
WARMUP = 200
PEAK_DECAY = 0.9997
FAST_DECAY = 0.9
SLOW_DECAY = 0.99
QUIET_LEVEL = 0.25
REOPEN_LEVEL = 0.5
QUIET_DWELL = 250
CLOSE_DECAY = 0.99
SURPRISE_RATIO = 3.0
SURPRISE_DWELL = 3
REFIT_STEPS = 800
PEAK_RESET_LAG = 200
REOPEN_GAIN = 0.2
_original_penalty = GradientPenalty.penalty
receipt = {'calls': 0, 'dimensions': {}, 'extra_critic_forwards': 0, 'floor': FLOOR,
           'guard_c': GUARD_C, 'guard_min_steps': GUARD_MIN_STEPS, 'pure_a_calls': 0, 'blend_calls': 0,
           'pure_b_calls': 0, 'first_blend_call': None, 'first_pure_b_call': None, 'critic_steps': 0,
           'critic_parameters': None, 'lr_max': None, 'lr_last': None, 's_trace': [], 'rule': 'ap3_partial_reopen',
           'anchor_decay': ANCHOR_DECAY, 'anchor_started_call': None, 'prox_trace': [], 'phase_trace': [],
           'signal': 'critic_grad_rms_vs_post_warmup_peak', 'cosine_used': False,
           'peak_decay': PEAK_DECAY, 'quiet_level': QUIET_LEVEL, 'reopen_level': REOPEN_LEVEL,
           'quiet_dwell': QUIET_DWELL, 'close_decay': CLOSE_DECAY, 'surprise_ratio': SURPRISE_RATIO,
           'refit_steps': REFIT_STEPS, 'reopen_gain': REOPEN_GAIN, 'warmup': WARMUP,
           'noise': 'inherited K3P horizon ablation via the driver',
           'rate_rule': 'network=0.01+0.99*rate_gain prior=0.05+0.95*rate_gain; after first s=0, reopen sets rate_gain=0.2',
           'mix_rule': 's follows rate gain until it first hits 0; later reopens leave s at 0'}
_state = {'ema': None, 'critic': None, 'pending': False, 'lr_max': 0.0, 'lr_last': None, 'depth': 0, 'clips': [],
          'critic_ref': None, 'rate_gain': 1.0, 'mix_gain': 1.0, 'peak': None, 'fast': None, 'slow': None,
          'quiet': 0, 'shock': 0, 'closed': False, 'level': 1.0, 'surprise': 1.0, 'warm': False,
          'ever_anchored': False, 'refit_until': 0, 'peak_reset_at': 0}


def handover_weight():
    """K3P map on the mixing gain. Does not read learning rates or step counts."""
    r = _state['mix_gain']
    return max(0.0, min(1.0, 2.0 * r) - 2.0 * FLOOR) / (1.0 - 2.0 * FLOOR)


def phase_multipliers(*args, **kwargs):
    """Drop-in for schedule.policy_multipliers. Ignores step, horizon, and anneal."""
    gain = _state['rate_gain']
    if gain >= 1.0:
        return 1.0, 1.0
    prior_floor = float(args[3]) if len(args) >= 4 and args[3] is not None else PRIOR_FLOOR
    net_floor = kwargs.get('network_lr_floor', None)
    net_floor = FLOOR if net_floor is None else float(net_floor)
    return (net_floor + (1.0 - net_floor) * gain,
            prior_floor + (1.0 - prior_floor) * gain)


def _install_phase_rates():
    from benchmarks.toy100 import schedule
    original = schedule.policy_multipliers
    schedule.policy_multipliers = phase_multipliers
    for module in list(sys.modules.values()):
        if module is None:
            continue
        name = getattr(module, '__name__', '')
        if not name.startswith(('particlegan', 'benchmarks')):
            continue
        for attr, value in list(vars(module).items()):
            if value is original:
                setattr(module, attr, phase_multipliers)


_install_phase_rates()


def _critic_params():
    return [p for g in _state['critic_ref'].param_groups for p in g['params']]


def _grad_rms(opt):
    total = None
    count = 0
    for group in opt.param_groups:
        for parameter in group['params']:
            if parameter.grad is None:
                continue
            term = parameter.grad.detach().square().sum()
            total = term if total is None else total + term
            count += parameter.grad.numel()
    if total is None or count == 0:
        return None
    return float(torch.sqrt(total / count))


def _update_phase(rms):
    if rms is None or not rms > 0.0:
        return
    if _state['fast'] is None:
        _state['fast'] = rms
        _state['slow'] = rms
        _state['level'] = 1.0
        _state['surprise'] = 1.0
        return
    fast = FAST_DECAY * _state['fast'] + (1.0 - FAST_DECAY) * rms
    slow = SLOW_DECAY * _state['slow'] + (1.0 - SLOW_DECAY) * rms
    _state['fast'] = fast
    _state['slow'] = slow
    _state['surprise'] = fast / slow if slow > 0.0 else 1.0
    if receipt['critic_steps'] < WARMUP:
        return
    if not _state['warm']:
        _state['peak'] = slow
        _state['warm'] = True
        _state['quiet'] = 0
        _state['shock'] = 0
        _state['closed'] = False
        _state['rate_gain'] = 1.0
        _state['mix_gain'] = 1.0
    if _state['peak_reset_at'] and receipt['critic_steps'] >= _state['peak_reset_at']:
        _state['peak'] = _state['slow']
        _state['peak_reset_at'] = 0
    peak = max(_state['peak'] * PEAK_DECAY, rms)
    level = rms / peak
    surprise = _state['surprise']
    _state['peak'] = peak
    _state['level'] = level
    if surprise >= SURPRISE_RATIO:
        _state['shock'] += 1
    else:
        _state['shock'] = 0
    reopen = level >= REOPEN_LEVEL or _state['shock'] >= SURPRISE_DWELL
    if reopen:
        _state['quiet'] = 0
        if _state['ever_anchored'] and (_state['closed'] or _state['rate_gain'] < REOPEN_GAIN):
            _state['rate_gain'] = REOPEN_GAIN
            _state['closed'] = False
            _state['refit_until'] = receipt['critic_steps'] + REFIT_STEPS
            _state['peak_reset_at'] = receipt['critic_steps'] + PEAK_RESET_LAG
            return
        if not _state['ever_anchored']:
            _state['rate_gain'] = 1.0
            _state['mix_gain'] = 1.0
            _state['closed'] = False
            return
    if _state['ever_anchored'] and receipt['critic_steps'] < _state['refit_until']:
        _state['rate_gain'] = REOPEN_GAIN
        _state['closed'] = False
        _state['quiet'] = 0
        return
    if level < QUIET_LEVEL:
        _state['quiet'] += 1
        if _state['quiet'] >= QUIET_DWELL:
            _state['closed'] = True
    if _state['closed']:
        _state['rate_gain'] = max(0.0, _state['rate_gain'] * CLOSE_DECAY)
        if not _state['ever_anchored']:
            _state['mix_gain'] = _state['rate_gain']
            if _state['mix_gain'] <= FLOOR:
                _state['mix_gain'] = 0.0
                _state['ever_anchored'] = True


def _phase_row():
    net, prior = phase_multipliers(0, 1, 0.6, PRIOR_FLOOR, 1600, network_lr_floor=FLOOR)
    return dict(critic_steps=receipt['critic_steps'], calls=receipt['calls'],
                rate_gain=round(_state['rate_gain'], 6), mix_gain=round(_state['mix_gain'], 6),
                s=round(handover_weight(), 6), level=None if _state['level'] is None else round(_state['level'], 6),
                surprise=None if _state['surprise'] is None else round(_state['surprise'], 6),
                quiet=_state['quiet'], shock=_state['shock'], closed=_state['closed'], warm=_state['warm'],
                ever_anchored=_state['ever_anchored'], refit_until=_state['refit_until'],
                network_multiplier=net, prior_multiplier=prior, lr_last=_state['lr_last'])


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
    receipt['extra_critic_forwards'] += 1
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
        row = _phase_row()
        receipt['s_trace'].append([receipt['calls'], row['s']])
        receipt['phase_trace'].append(row)
        print(json.dumps(dict(phase=row)), flush=True)
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
        _update_phase(_grad_rms(opt))
    _state['depth'] = max(0, _state['depth'] - 1)


def _write_receipt():
    receipt['lr_max'] = _state['lr_max']
    receipt['lr_last'] = _state['lr_last']
    receipt['final_s'] = handover_weight()
    receipt['final_rate_gain'] = _state['rate_gain']
    receipt['final_mix_gain'] = _state['mix_gain']
    receipt['final_ever_anchored'] = _state['ever_anchored']
    receipt['final_closed'] = _state['closed']
    receipt['final_level'] = _state['level']
    receipt['final_surprise'] = _state['surprise']
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
