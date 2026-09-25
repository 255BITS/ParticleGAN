"""Attribution control: no latent rule. The selected base's exact parent Adam step (native100.py hook is a no-op)."""
receipt = {'formula': 'none: control for a2_bounded_damp attribution', 'calls': 0, 'scoped_calls': 0, 'rows': [], 'tables': {}}
stats = {}


def begin(opt):
    return []


def end(saved):
    pass


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
