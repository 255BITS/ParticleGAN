"""rp1: reversible partial-rate close, no forced dwell, horizon-free noise.

Parent is pinned K3P. Cold acquisition matches AP3: critic-gradient RMS versus
a post-warmup peak, full rate and the early penalty until 250 quiet steps,
then both decay. After the mixing gain first hits 0 it stays there, so the
gradient cap and the 0.999 EMA anchor stay on.

AP3 then held rate gain 0.2 for 800 critic steps and reset the RMS peak to the
slow estimate 200 steps into that window. On the measured shift the level was
already below the quiet threshold by step 2500, and the deadline misses were
the later precision wobble while that partial rate was still forced. This rule
does not arm that dwell and does not replace the peak. A later RMS reopen sets
rate gain to 0.2 only. When the level falls below the quiet threshold for 250
steps, the same 0.99 decay used on the cold path closes the rate. A new reopen
can raise it again. The early penalty is not restored, and the rate is not
returned to the base.

Input noise falls from 0.5 to 0 over a declared 120 updates. Output noise rises
from 0 to 0.029 over a declared 240 updates. Those lengths equal K3P's 0.1 and
0.2 fractions of a 1200-update noise horizon, stored as constants. The total
training budget, the noise_horizon argument, quality, targets, and the shift
time are not read. Host cosine arguments are ignored.
"""
import atexit, json, math, sys
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
REOPEN_GAIN = 0.2
INPUT_ZERO_BY_STEP = 120
OUTPUT_FULL_BY_STEP = 240
_original_penalty = GradientPenalty.penalty
receipt = {'calls': 0, 'dimensions': {}, 'extra_critic_forwards': 0, 'floor': FLOOR,
           'guard_c': GUARD_C, 'guard_min_steps': GUARD_MIN_STEPS, 'pure_a_calls': 0, 'blend_calls': 0,
           'pure_b_calls': 0, 'first_blend_call': None, 'first_pure_b_call': None, 'critic_steps': 0,
           'critic_parameters': None, 'lr_max': None, 'lr_last': None, 's_trace': [], 'rule': 'rp1_signal_close',
           'anchor_decay': ANCHOR_DECAY, 'anchor_started_call': None, 'prox_trace': [], 'phase_trace': [],
           'signal': 'critic_grad_rms_vs_post_warmup_peak', 'cosine_used': False,
           'peak_decay': PEAK_DECAY, 'quiet_level': QUIET_LEVEL, 'reopen_level': REOPEN_LEVEL,
           'quiet_dwell': QUIET_DWELL, 'close_decay': CLOSE_DECAY, 'surprise_ratio': SURPRISE_RATIO,
           'refit_steps': 0, 'peak_reset': False, 'reopen_gain': REOPEN_GAIN, 'warmup': WARMUP,
           'input_zero_by_step': INPUT_ZERO_BY_STEP, 'output_full_by_step': OUTPUT_FULL_BY_STEP,
           'noise': 'absolute 120-step input anneal and 240-step output warmup; total budget and noise_horizon ignored',
           'rate_rule': 'network=0.01+0.99*rate_gain prior=0.05+0.95*rate_gain; after first s=0, reopen sets rate_gain=0.2 then 0.99 decay after 250 quiet steps',
           'mix_rule': 's follows rate gain until it first hits 0; later reopens leave s at 0',
           'budget_dependencies': [
               'driver still requires the noise_horizon argument to equal 1200; the schedule does not read it',
               'config network_lr_horizon_cap is still passed by the host; phase_multipliers ignores step, total, and cap',
           ]}
_state = {'ema': None, 'critic': None, 'pending': False, 'lr_max': 0.0, 'lr_last': None, 'depth': 0, 'clips': [],
          'critic_ref': None, 'rate_gain': 1.0, 'mix_gain': 1.0, 'peak': None, 'fast': None, 'slow': None,
          'quiet': 0, 'shock': 0, 'closed': False, 'level': 1.0, 'surprise': 1.0, 'warm': False,
          'ever_anchored': False, 'refit_until': 0}
_old_input_noise = None
_old_output_noise = None
_noise_patched = False


def absolute_input_noise(peak, completed_steps, total_steps, end_fraction):
    """Peak at step 0, zero at INPUT_ZERO_BY_STEP. total_steps is not used."""
    if isinstance(peak, bool) or not math.isfinite(peak) or peak < 0:
        raise ValueError("input noise std must be finite and nonnegative")
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("invalid input noise step count")
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("invalid input noise step count")
    if isinstance(end_fraction, bool) or not math.isfinite(end_fraction) or not 0 < end_fraction <= 1:
        raise ValueError("input noise anneal end must be in (0, 1]")
    return float(peak * max(0.0, 1.0 - completed_steps / INPUT_ZERO_BY_STEP))


def absolute_output_noise(peak, completed_steps, total_steps, warmup_fraction=0.0):
    """Rise to peak over OUTPUT_FULL_BY_STEP when warmup is requested.

    A zero warmup stays constant, matching the original contract. total_steps
    and a positive warmup fraction do not set the window.
    """
    if isinstance(peak, bool) or not math.isfinite(peak) or peak < 0:
        raise ValueError("output noise std must be finite and nonnegative")
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("invalid output noise step count")
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("invalid output noise step count")
    if (isinstance(warmup_fraction, bool) or not math.isfinite(warmup_fraction)
            or not 0 <= warmup_fraction <= 1):
        raise ValueError("output noise warmup must be a finite fraction in [0, 1]")
    if warmup_fraction == 0:
        return float(peak)
    return float(peak * min(1.0, completed_steps / OUTPUT_FULL_BY_STEP))


def _k3p_reference_noise(step):
    """K3P curve at noise horizon 1200, for the receipt audit only."""
    return (0.5 * max(0.0, 1.0 - step / 120.0),
            0.029 * min(1.0, step / 240.0))


def _patch_noise():
    global _old_input_noise, _old_output_noise, _noise_patched
    import benchmarks.toy100.models as toy_models
    if _old_input_noise is None:
        _old_input_noise = toy_models.linear_input_noise
        _old_output_noise = toy_models.linear_output_noise
    toy_models.linear_input_noise = absolute_input_noise
    toy_models.linear_output_noise = absolute_output_noise
    sites = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        for attr, value in list(vars(module).items()):
            if value is _old_input_noise:
                setattr(module, attr, absolute_input_noise)
                sites.append(f"{getattr(module, '__name__', '?')}.{attr}:input")
            elif value is _old_output_noise:
                setattr(module, attr, absolute_output_noise)
                sites.append(f"{getattr(module, '__name__', '?')}.{attr}:output")
    audit = []
    for step in (0, 60, 119, 120, 240, 1200):
        for horizon in (1200, 4800):
            got_in = absolute_input_noise(0.5, step, horizon, 0.1)
            got_out = absolute_output_noise(0.029, step, horizon, 0.2)
            ref_in, ref_out = _k3p_reference_noise(step)
            if got_in != ref_in or got_out != ref_out:
                raise RuntimeError(f"noise schedule diverges from the 1200-horizon K3P curve at step {step}")
        audit.append(dict(step=step, input=absolute_input_noise(0.5, step, 1200, 0.1),
                          output=absolute_output_noise(0.029, step, 1200, 0.2),
                          same_at_horizon_4800=True))
    receipt['noise_horizon_audit'] = audit
    receipt['noise_patch_sites'] = sites
    _noise_patched = True


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
_patch_noise()


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
    peak = max(_state['peak'] * PEAK_DECAY, rms)
    level = rms / peak if peak > 0.0 else 1.0
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
        if _state['ever_anchored']:
            if _state['closed'] or _state['rate_gain'] < REOPEN_GAIN:
                _state['rate_gain'] = REOPEN_GAIN
                _state['closed'] = False
            return
        _state['rate_gain'] = 1.0
        _state['mix_gain'] = 1.0
        _state['closed'] = False
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
    if not _noise_patched:
        _patch_noise()
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
    receipt['final_quiet'] = _state['quiet']
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
