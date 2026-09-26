"""Bounded current-gradient-anchored coherence damping for sparse latent tables; direct response byte-identical.
u_i = (1 - (1 - rho_i) / 2) * g_i: history removes at most half of the current response.  rho_i = (1 + cos(g_i, h_i)) / 2 in [0, 1], h_i = row i's last observed gradient
(rho_i = 1 without history).  Inactive rows get u_i = 0 (no transport).  Applied only to registered
ParticlePrior tables with a missing row this step AND cumulative row-observation rate < 1/2;
otherwise the exact parent Adam step (betas (0,.999), state untouched).  Parent v from raw g."""
import os
import torch
import response
B1 = .5
receipt = {'formula': 'u=(1-(1-rho)/2)*g in [g/2,g], rho=(1+cos(g,h_last_observed))/2, rho=1 without history; scope: missing row and cumulative observed rate<1/2; v parent',
           'calls': 0, 'scoped_calls': 0, 'rows': [], 'tables': {}}
stats = {}  # id(p) -> [observed row-steps, total row-steps]


def _diag_prior(p, *, scoped, active_fraction, rate, rho_mean):
    if not os.environ.get('K3P_DIAG_TRAJ'):
        return
    from benchmarks.toy100.diag_traj import note
    z = p.detach()
    row = z.norm(dim=1)
    note(a2_scoped=bool(scoped),
         a2_active_fraction=round(float(active_fraction), 6),
         a2_cumulative_rate=round(float(rate), 6),
         a2_rho_mean=None if rho_mean is None else round(float(rho_mean), 6),
         prior_row_norm_mean=round(float(row.mean()), 6),
         prior_row_norm_std=round(float(row.std(unbiased=False)), 6),
         prior_coord_std=round(float(z.std(unbiased=False)), 6))


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
                _diag_prior(p, scoped=False, active_fraction=count / active.numel(), rate=rate, rho_mean=None)
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
            _diag_prior(p, scoped=scoped, active_fraction=info['active_fraction'], rate=rate,
                        rho_mean=info.get('rho_mean_active'))
    return saved


@torch.no_grad()
def end(saved):
    for group, betas, m, g in saved:
        group['betas'] = betas
        m.copy_(g)  # parent beta1=0 state holds the raw gradient
