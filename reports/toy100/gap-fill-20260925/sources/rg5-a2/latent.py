"""Bounded current-gradient-anchored coherence damping for sparse latent tables; direct response byte-identical.
u_i = (1 - (1 - rho_i) / 2) * g_i: history removes at most half of the current response.  rho_i = (1 + cos(g_i, h_i)) / 2 in [0, 1], h_i = row i's last observed gradient
(rho_i = 1 without history).  Inactive rows get u_i = 0 (no transport).  Applied only to registered
ParticlePrior tables with a missing row this step AND cumulative row-observation rate < 1/2;
otherwise the exact parent Adam step (betas (0,.999), state untouched).  Parent v from raw g."""
import torch
import response
B1 = .5
receipt = {'formula': 'u=(1-(1-rho)/2)*g in [g/2,g], rho=(1+cos(g,h_last_observed))/2, rho=1 without history; scope: missing row and cumulative observed rate<1/2; v parent',
           'calls': 0, 'scoped_calls': 0, 'rows': [], 'tables': {}}
stats = {}  # id(p) -> [observed row-steps, total row-steps]


def begin(opt):
    saved = []
    for group in opt.param_groups:
        latent = [p for p in group['params'] if id(p) in response.prior_ids and p.grad is not None]
        if not latent:
            continue
        assert len(latent) == len(group['params'])
        receipt['calls'] += 1
        if len(latent) != 1 or latent[0].dim() != 2:
            continue
        p = latent[0]
        g = p.grad.detach()
        with torch.no_grad():
            norm = g.square().sum(-1).sqrt()
            active = norm > 0
            count = int(active.sum())
            s = stats.setdefault(id(p), [0, 0])
            s[0] += count; s[1] += active.numel()
            rate = s[0] / s[1]
            sparse = count < active.numel()
            state = opt.state[p]
            if sparse and 'anchor_prev' not in state:
                state['anchor_prev'] = torch.zeros_like(p)
            if 'anchor_prev' not in state:
                continue
            h = state['anchor_prev']
            scoped = sparse and rate < .5
            info = dict(call=receipt['calls'], rows=active.numel(), active_fraction=count / active.numel(), cumulative_rate=rate, scoped=scoped)
            if scoped:
                hn = h.square().sum(-1).sqrt()
                has = active & (hn > 0)
                cos = (g * h).sum(-1) / (norm * hn).clamp_min(1e-30)
                rho = torch.where(has, .75 + .25 * cos, torch.ones_like(cos))  # 1-(1-(1+cos)/2)/2
                step = float(state['step']) + 1.
                bc1 = 1. - B1 ** step
                # native lerp with weight .5: m = g - (g - pre)*.5 = rho*bc1*g, so the update is rho*g/(sqrt(v_hat)+eps)
                state['exp_avg'].copy_(g * (2. * bc1 * rho - 1.).unsqueeze(-1))
                saved.append((group, group['betas'], state['exp_avg'], g.clone()))
                group['betas'] = (B1, group['betas'][1])
                receipt['scoped_calls'] += 1
                info.update(rho_mean_active=float(rho[active].mean()), history_fraction_active=float(has.sum()) / max(count, 1))
            h[active] = g[active]
            if receipt['calls'] <= 20 or receipt['calls'] % 50 == 0:
                receipt['rows'].append(info)
            receipt['tables'][str(active.numel())] = dict(cumulative_rate=rate, scoped_last=scoped)
    return saved


@torch.no_grad()
def end(saved):
    for group, betas, m, g in saved:
        group['betas'] = betas
        m.copy_(g)  # parent beta1=0 state holds the raw gradient


GUARD = dict(C=5.0, MATURE=200, ROLES=('g', 'd'), G_GATE='reversal')
# ---- step-ratio guard (appended to the arm latent; runs before the arm rule) ----
# r = RMS(g) / sqrt(mean(v_hat_prev)) per network tensor: the would-be Adam update size in lr units (beta1 = 0).
# Clip: g *= min(1, C / r) when the optimizer has taken >= MATURE steps and its role is in ROLES.
# G_GATE 'reversal': a generator tensor is clipped only when cos(g, g_prev) < 0 (period-2 overshoot), never on a
# coherent step change. Particle tables (registered ParticlePrior parameters) are never touched.
# No targets, labels or evaluations are read. With ROLES = () this block only records statistics.
import torch as _gt
import response
import particlegan.training as _gtr
_g_init = _gtr.GANTrainer.__init__
def _g_tag(self, *args, **kwargs):
    _g_init(self, *args, **kwargs)
    self.opt_g._guard_role = 'g'
    self.opt_d._guard_role = 'd'
_gtr.GANTrainer.__init__ = _g_tag
guard_receipt = {'guard': GUARD, 'clips': {'g': 0, 'd': 0}, 'clip_steps': {'g': [], 'd': []}, 'untagged_calls': 0, 'untagged_roles': {}, 'rows': []}
_g_calls = {}


@_gt.no_grad()
def _guard(opt):
    role = getattr(opt, '_guard_role', None)
    if role is None:
        # optimizers built outside GANTrainer (continuous probe hosts): the host's own role rule,
        # generator = the optimizer with several parameter groups or registered particle tables
        guard_receipt['untagged_calls'] += 1
        role = 'g' if len(opt.param_groups) > 1 or any(id(p) in response.prior_ids for g in opt.param_groups for p in g['params']) else 'd'
        guard_receipt['untagged_roles'][role] = guard_receipt['untagged_roles'].get(role, 0) + 1
    t_prev = _g_calls.get(id(opt), 0)
    _g_calls[id(opt)] = t_prev + 1
    if t_prev < 1:
        return
    rs, cs, ps = [], [], []
    for group in opt.param_groups:
        if group.get('_comparison_prior', False):
            continue
        b2 = group['betas'][1]
        for p in group['params']:
            if p.grad is None or id(p) in response.prior_ids or p not in opt.state or 'exp_avg_sq' not in opt.state[p]:
                continue
            st = opt.state[p]
            g = p.grad
            vhat = st['exp_avg_sq'].mean() / (1. - b2 ** t_prev)
            r = g.square().mean().sqrt() / vhat.clamp_min(1e-30).sqrt()
            m = st['exp_avg']
            c = (g * m).sum() / (g.norm() * m.norm()).clamp_min(1e-30)
            rs.append(r); cs.append(c); ps.append(p)
    if not rs:
        return
    r = _gt.stack(rs); c = _gt.stack(cs)
    active = role in GUARD['ROLES'] and t_prev >= GUARD['MATURE']
    if active:
        hit = r > GUARD['C']
        if role == 'g' and GUARD['G_GATE'] == 'reversal':
            hit = hit & (c < 0)
        scale = _gt.where(hit, GUARD['C'] / r.clamp_min(1e-30), _gt.ones_like(r))
        for p, s in zip(ps, scale):
            p.grad.mul_(s)
    stats = _gt.stack([r.max(), r.median(), c.min(), c[r.argmax()]] + ([hit.sum().float()] if active else [])).tolist()
    n = int(stats[4]) if active else 0
    if n:
        guard_receipt['clips'][role] += n
        guard_receipt['clip_steps'][role].append(t_prev)
    guard_receipt['rows'].append((role, t_prev, round(stats[0], 5), round(stats[1], 5), round(stats[2], 4), round(stats[3], 4), n))


_arm_begin = begin
def begin(opt):
    _guard(opt)
    return _arm_begin(opt)
