"""KA2 (kal_asym lane): ASYMMETRIC stateful alpha on the R2 base.

KA2 slow-attack/fast-release: attack k=1/60 (false-alarm guard: needs ~sustained surprise over tens of calls; ignores R2's spurious pre-shift release at step ~2275); release k=0.5 (re-anchor within ~2-3 calls once surprise clears: agility). Tests guard+agility corner.

Lineage: R1 displacement-band mechanism file; same B3/K3P base (800-call pure-A
warmup, s=0.5 LR-decoupled, K3P guard C=5/200, EMA decay 0.999, config/latent/
response byte-identical to K3P). Change: the release signal is the critic
Adam SECOND-MOMENT SURPRISE, not realized displacement, not prox/median.

Signal: at each critic Adam step, per tensor surprise_n = RMS(grad_n) /
sqrt(mean bias-corrected exp_avg_sq_n) (the same guard ratio K3P already
computes; ordinary optimizer signal; no budget/shift/metric/target reads).
sur_hist collects the median-over-tensors surprise of the latest completed
critic step at each blended penalty call. base = median of the first 24
post-warmup blended surprises. ratio = median(last 24) / base. Hysteretic
band: W==1 and ratio > 3.0 -> W=0 (release); W==0 and ratio < 1.75 -> W=1
(re-anchor). Cold start: sur_hist < 25 forces W=1 (quiet by construction).
Settled hold: gradients match their running second moment -> surprise ~1 ->
W=1. Post-shift transient: stale second moment underestimates new gradients
-> surprise spikes and stays high while relearning -> W stays 0. Scale-free:
independent of the LR floor, so no high-LR vs floored-scale mismatch.

penalty = (coeff/2) * (s*A + (1-s)*(B + W*prox)), s=0.5 fixed (LR-decoupled).
STATEFUL alpha drives EMA every critic step (no W gate on updates); guarded
re-seed after 60 consecutive W==0 blended calls WITH ratio > 3 retained.
Asymmetric-alpha change (this lane): alpha itself is STATEFUL with different
attack and release speeds. Instantaneous target t=g(ratio): 0 at ratio<=1,
linear 1->3, saturated 1 above 3.0 (same band numbers as R2's validated W
band). Per blended call with a fresh ratio:
  alpha <- alpha + (t-alpha)*k, k=K_ATK if t>alpha else K_REL.
EMA decay = 1 - alpha*(1-0.90): alpha=0 frozen memory, alpha=1 fast tracking.
EMA updates EVERY critic step (no binary iff-W==1 gate); W band drives only
the blend penalty. Guarded reseed (60-streak) retained from R2.
LR-decoupling: no LR/floor/total-step reads; constant critic LR keeps
surprise dynamics (hence ratio/alpha/W) alive.

Remaining budget dependency (labeled ablation): inherited LR/noise schedules
(floors .01/.05, network horizon cap 1600) unchanged from K3P.


Remaining budget dependency (labeled ablation): inherited LR/noise schedules
(floors .01/.05, network horizon cap 1600) unchanged from K3P.
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
ANCHOR_DECAY_MIN = 0.90
GAIN_LO = 1.0
GAIN_HI = 3.0
K_ATK = (1.0/60.0)
K_REL = 0.5
S_FIX = 0.5
WARMUP_CALLS = 800
SHORT_WINDOW = 24
GATE_MIN_SAMPLES = 25
REL_HI = 3.0
REL_LO = 1.75
RESEED_STREAK = 60
BASE_WINDOW = 24
HIST_CAP = 400
_original_penalty = GradientPenalty.penalty
receipt = {'calls': 0, 'dimensions': {}, 'extra_critic_forwards': 0, 'rule': 'ka2-slowatk-fastrel',
           'floor': FLOOR, 'guard_c': GUARD_C, 'guard_min_steps': GUARD_MIN_STEPS,
           'pure_a_calls': 0, 'blend_calls': 0, 'pure_b_calls': 0,
           'first_blend_call': None, 'first_pure_b_call': None, 'critic_steps': 0,
           'critic_parameters': None, 'lr_max': None, 'lr_last': None, 's_trace': [],
           'anchor_decay': ANCHOR_DECAY, 'anchor_started_call': None, 'prox_trace': [],
           's_fix': S_FIX, 'lr_decoupled': True, 'w_trace': [], 'sur_trace': [],
           'sur_base': None, 'ema_updates': 0, 'ema_skips': 0, 'ema_reseeds': 0, 'alpha_trace': []}
_state = {'ema': None, 'critic': None, 'pending': False, 'lr_max': 0.0, 'lr_last': None,
          'depth': 0, 'clips': [], 'critic_ref': None, 'last_sur': None,
          'sur_hist': [], 'sur_base': None, 'w': 1.0, 'low_streak': 0, 'alpha': 0.0, 'last_ratio': None}


def handover_weight():
    return S_FIX


def _critic_params():
    return [p for g in _state['critic_ref'].param_groups for p in g['params']]


def anchored_gradient_gap(D, x_real, g, dimension):
    """mean ||g - grad_x Dbar(x_real)||^2 / d; Dbar = parameter EMA, swapped via .data (restored)."""
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


def _gain_target(ratio):
    """Instantaneous target t=g(ratio): 0 at ratio<=1, linear 1->3, 1 above."""
    if ratio is None:
        return 0.0
    if ratio <= GAIN_LO:
        return 0.0
    if ratio >= GAIN_HI:
        return 1.0
    return (ratio - GAIN_LO) / (GAIN_HI - GAIN_LO)


def _alpha_step(ratio):
    """Asymmetric first-order alpha update. Returns (alpha, decay)."""
    t = _gain_target(ratio)
    a = _state['alpha']
    k = K_ATK if t > a else K_REL
    a = a + (t - a) * k
    if a < 0.0:
        a = 0.0
    elif a > 1.0:
        a = 1.0
    _state['alpha'] = a
    return a, 1.0 - a * (1.0 - ANCHOR_DECAY_MIN)



def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def _sur_gate():
    """Hysteretic moment-surprise band gate. Returns (W, ratio_or_None)."""
    hist = _state['sur_hist']
    if len(hist) < GATE_MIN_SAMPLES:
        return 1.0, None
    if _state['sur_base'] is None and len(hist) >= BASE_WINDOW:
        _state['sur_base'] = _median(hist[:BASE_WINDOW])
        receipt['sur_base'] = _state['sur_base']
    base = _state['sur_base']
    if not base or base <= 0.0:
        return 1.0, None
    ratio = _median(hist[-SHORT_WINDOW:]) / base
    w = _state['w']
    if w >= 1.0 and ratio > REL_HI:
        w = 0.0
    elif w <= 0.0 and ratio < REL_LO:
        w = 1.0
    _state['w'] = w
    return w, ratio


def scaled_penalty(self, D, x_real, x_fake, step=1, generator=None, collect_stats=True):
    if self.arm != 'a_r1r2' or (self.lazy_k > 1 and step % self.lazy_k != 0):
        return _original_penalty(self, D, x_real, x_fake, step, generator, collect_stats)
    _state['pending'] = True
    coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
    dimension = x_real[0].numel()
    assert dimension == x_fake[0].numel()
    s = S_FIX
    receipt['calls'] += 1
    receipt['dimensions'][str(dimension)] = receipt['dimensions'].get(str(dimension), 0) + 1
    if receipt['calls'] + 1 <= WARMUP_CALLS:
        real_squared = self._grad_norm(D, x_real, squared=True) / dimension
        fake_norm = self._grad_norm(D, x_fake, squared=False) / dimension ** 0.5
        fake_cap = (fake_norm - self.kappa).relu().square()
        penalty = (coefficient / 2.0) * (real_squared.mean() + fake_cap.mean())
        receipt['pure_a_calls'] += 1
        w = 1.0
        _state['w'] = w
        if receipt['calls'] == 1 or receipt['calls'] % 50 == 0:
            receipt['s_trace'].append([receipt['calls'], round(s, 6)])
        stats = {'applied': True, 'pen': float(penalty.detach()), 'fake_cap': self.kappa, 's': s, 'w': w} if collect_stats else {}
        return penalty, stats
    x = x_real.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(_score_scalar(D(x)), x, create_graph=True)[0]
    sq_r = g.pow(2).flatten(1).sum(dim=1)
    n_r = torch.sqrt(sq_r + 1e-12)
    n_f = self._grad_norm(D, x_fake, squared=False)
    prox = anchored_gradient_gap(D, x_real, g, dimension)
    prox_val = float(prox.detach())
    if _state['last_sur'] is not None:
        _state['sur_hist'].append(float(_state['last_sur']))
        if len(_state['sur_hist']) > HIST_CAP:
            del _state['sur_hist'][:len(_state['sur_hist']) - HIST_CAP]
    w, ratio = _sur_gate()
    _state['last_ratio'] = ratio
    _alpha_step(ratio)
    a_term = (sq_r / dimension).mean() + (n_f / dimension ** 0.5 - self.kappa).relu().square().mean()
    b_term = F.relu(n_r - self.kappa).pow(2).mean() + F.relu(n_f - self.kappa).pow(2).mean() + w * prox
    penalty = (coefficient / 2.0) * (s * a_term + (1.0 - s) * b_term)
    receipt['blend_calls'] += 1
    if w < 0.5 and ratio is not None and ratio > REL_HI:
        _state['low_streak'] += 1
    else:
        _state['low_streak'] = 0
    if receipt['first_blend_call'] is None:
        receipt['first_blend_call'] = receipt['calls']
    if receipt['calls'] % 50 == 0:
        receipt['prox_trace'].append([receipt['calls'], prox_val])
        receipt['w_trace'].append([receipt['calls'], round(w, 6)])
        receipt['sur_trace'].append([receipt['calls'], float(_state['sur_hist'][-1]) if _state['sur_hist'] else None,
                                     round(ratio, 4) if ratio is not None else None])
    if receipt['calls'] == 1 or receipt['calls'] % 50 == 0:
        receipt['s_trace'].append([receipt['calls'], round(s, 6)])
        receipt['alpha_trace'].append([receipt['calls'], round(_state['alpha'], 4), _state['last_ratio'] if _state['last_ratio'] is None else round(_state['last_ratio'], 4)])
    stats = {'applied': True, 'pen': float(penalty.detach()), 'fake_cap': self.kappa, 's': s, 'w': w, 'alpha': _state['alpha']} if collect_stats else {}
    return penalty, stats


def _surprise_of(opt):
    """Median-over-tensors gradient-RMS / sqrt(mean bias-corrected v). Pure read of Adam state."""
    vals = []
    with torch.no_grad():
        for group in opt.param_groups:
            beta2 = group['betas'][1]
            for p in group['params']:
                st = opt.state.get(p)
                if p.grad is None or not st or 'exp_avg_sq' not in st:
                    continue
                t = st['step']
                if t < 1:
                    continue
                vhat = st['exp_avg_sq'].mean() / (1.0 - beta2 ** t)
                r = p.grad.square().mean().sqrt() / vhat.clamp_min(1e-30).sqrt()
                vals.append(float(r.detach()))
    if not vals:
        return None
    return _median(vals)


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
        sur = _surprise_of(opt)
        if sur is not None:
            _state['last_sur'] = sur
        if _state['low_streak'] >= RESEED_STREAK and _state['ema'] is not None:
            with torch.no_grad():
                for e, p in zip(_state['ema'], _critic_params()):
                    e.copy_(p.detach())
            receipt['ema_reseeds'] += 1
            _state['low_streak'] = 0
        if _state['ema'] is not None:
            a = _state['alpha']
            decay = 1.0 - a * (1.0 - ANCHOR_DECAY_MIN)
            if decay >= 1.0:
                receipt['ema_skips'] += 1
            else:
                with torch.no_grad():
                    for e, p in zip(_state['ema'], _critic_params()):
                        e.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
                receipt['ema_updates'] += 1
        lr = max(float(g['lr']) for g in opt.param_groups)
        _state['lr_last'] = lr
        _state['lr_max'] = max(_state['lr_max'], lr)
        receipt['critic_steps'] += 1
    _state['depth'] = max(0, _state['depth'] - 1)


def _write_receipt():
    receipt['lr_max'] = _state['lr_max']
    receipt['lr_last'] = _state['lr_last']
    receipt['final_s'] = handover_weight()
    receipt['final_w'] = _state['w']
    receipt['final_sur_base'] = _state['sur_base']
    receipt['sur_hist_len'] = len(_state['sur_hist'])
    receipt['final_alpha'] = _state['alpha']
    receipt['final_ratio'] = _state['last_ratio']
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
