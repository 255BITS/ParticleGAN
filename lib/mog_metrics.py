"""Metrics for the fixed-sigma MoG study; no training RNG is consumed."""
import math
import numpy as np
import torch
from scipy.spatial import cKDTree
from lib.toy_metrics import per_mode_core_ratio
from lib.toy_models import sample_100gaussians


@torch.no_grad()
def sample_metrics(x):
    coords = torch.arange(10, device=x.device, dtype=x.dtype) - 4.5
    centers = torch.cartesian_prod(coords, coords)
    distance, nearest = torch.cdist(x, centers).min(1)
    hq = distance <= .09
    counts = torch.bincount(nearest[hq], minlength=100).double()
    p = counts / counts.sum().clamp_min(1)
    positive = p > 0
    out = dict(modes=int((counts >= 10).sum()), hq=float(hq.double().mean()),
               bridge=float((~hq).double().mean()),
               kl_balance=float((p[positive] * (100*p[positive]).log()).sum()) if hq.any() else None,
               share_min=float(p.min()*100), share_max=float(p.max()*100),
               width=per_mode_core_ratio(x, min_count=50)['per_mode_core_ratio'])
    out['width_audited_modes'] = int((torch.bincount(nearest, minlength=100) >= 50).sum())
    return out, nearest, hq


@torch.no_grad()
def geometry(prior):
    if not hasattr(prior, 'means'):
        return {key: None for key in ('d_med', 'r_eff', 'k_nbrs', 'raw_std', 'sigma', 'd0', 'bridge_heuristic')}
    points = prior.means().detach().cpu().double().numpy()
    tree = cKDTree(points)
    d = float(np.median(tree.query(points, k=2)[0][:, 1]))
    k = float(np.mean(tree.query_ball_point(points, 1.5*d, return_length=True)-1))
    sigma = float(prior.sigma)
    r = sigma / d if d > 0 else None
    bound = k * .5 * math.erfc(1 / (2*r*math.sqrt(2))) if r else 0.0
    return dict(d_med=d, r_eff=r, k_nbrs=k, raw_std=float(prior.z.std()),
                sigma=sigma, d0=float(prior.d0), bridge_heuristic=bound)


def component_metrics(idx, nearest, hq, n_components, detail=False):
    counts = torch.bincount(idx[hq]*100+nearest[hq], minlength=n_components*100).reshape(n_components, 100)
    total = torch.bincount(idx, minlength=n_components)
    good = counts.sum(1)
    valid = good > 0
    purity = counts.max(1).values.double() / good.clamp_min(1)
    majority = counts.argmax(1)
    allocation = torch.bincount(majority[valid], minlength=100)
    bridge = 1 - good.double() / total.clamp_min(1)
    out = dict(purity_mean=float(purity[valid].mean()) if valid.any() else None,
               purity_below_09=int(((purity < .9) & valid).sum()),
               components_no_hq=int((~valid).sum()), components_unsampled=int((total == 0).sum()),
               alloc_empty=int((allocation == 0).sum()),
               bridge_i_mean=float(bridge[total > 0].mean()))
    if not detail:
        return out, None
    purity, valid, bridge, total, majority = [v.cpu().tolist() for v in (purity, valid, bridge, total, majority)]
    detail = dict(purity_i=[v if ok else None for v, ok in zip(purity, valid)],
                  bridge_i=[v if count else None for v, count in zip(bridge, total)],
                  majority_mode=[v if ok else None for v, ok in zip(majority, valid)],
                  alloc=allocation.cpu().tolist())
    return out, detail


@torch.no_grad()
def evaluate(g, prior, n, seed, initial_raw_std=None, component_detail=False, pass_criteria=None):
    device = next(g.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed+999)
    real_rng = torch.Generator(device=device).manual_seed(seed+1999)
    z, idx = prior.sample(n, generator=rng)
    # Fixed chunk size bounds memory without changing RNG consumption.
    x = torch.cat([g(chunk) for chunk in z.split(20000)])
    real = sample_100gaussians(n, device, generator=real_rng)
    out, nearest, hq = sample_metrics(x)
    reference, _, _ = sample_metrics(real)
    out.update({key+'_real': value for key, value in reference.items()})
    for key in ('modes','hq','bridge','width','kl_balance','share_min','share_max'):
        den = reference[key]
        out[key+'_ratio'] = out[key]/den if den and out[key] is not None else None
    out.update(geometry(prior))
    out['raw_std_ratio'] = out['raw_std']/initial_raw_std if initial_raw_std and out['raw_std'] is not None else None
    r = getattr(prior, 'sigma_rel', 0)
    out['r_eff_drift_flag'] = bool(r and (out['r_eff'] is None or not r/1.5 <= out['r_eff'] <= r*1.5))
    out['raw_std_drift_flag'] = bool(out['raw_std_ratio'] is not None and not .5 <= out['raw_std_ratio'] <= 2)
    detail = None
    if idx is not None:
        stats, detail = component_metrics(idx, nearest, hq, prior.num_particles, detail=component_detail)
        out.update(stats)
        if component_detail and float(prior.sigma) == 0:
            codes = prior.sample(prior.num_particles, fixed_first_n=True)[0]
            atom_x = torch.cat([g(chunk) for chunk in codes.split(20000)])
            deterministic, assignment, good = sample_metrics(atom_x)
            out.update({'deterministic_'+key: value for key, value in deterministic.items()})
            _, detail = component_metrics(torch.arange(prior.num_particles, device=device), assignment, good, prior.num_particles, detail=True)
    else:
        out.update({key: None for key in ('purity_mean','purity_below_09','components_no_hq','components_unsampled','alloc_empty','bridge_i_mean')})
    out.update(pass_metrics(out, pass_criteria))
    return out, detail, x, real


def allocation_null(n, draws=1000):
    rng = np.random.default_rng(1729+n)
    counts = rng.multinomial(n, np.full(100, .01), size=draws)
    p = counts / n
    kl = np.sum(p * np.log(np.maximum(p, 1e-300)*100), axis=1)
    return dict(n=n, draws=draws, kl_balance=float(kl.mean()),
                empty_modes=float((counts == 0).sum(1).mean()), kl_std=float(kl.std()))


def pass_metrics(metrics, criteria=None):
    """Score against frozen thresholds; never derive thresholds from pilot runs."""
    def qualifies(hq_min, width_min, width_max, kl_max):
        values = [metrics.get(k) for k in ('hq_ratio', 'width_ratio', 'kl_balance')]
        return bool(metrics.get('modes') == 100 and all(
            v is not None and math.isfinite(v) for v in values) and
            values[0] >= hq_min and width_min <= values[1] <= width_max and
            values[2] <= kl_max)
    strict = qualifies(.98, .9, 1.1, .01)
    out = {'passed_strict': strict, 'passed': strict}
    if criteria is not None:
        required = {'hq_ratio_min', 'width_ratio_min', 'width_ratio_max', 'kl_balance_max'}
        if set(criteria) != required or not all(math.isfinite(v) for v in criteria.values()):
            raise ValueError('pass criteria must contain four finite thresholds')
        if not (0 <= criteria['hq_ratio_min'] and 0 <= criteria['width_ratio_min'] <= criteria['width_ratio_max']
                and criteria['kl_balance_max'] >= 0):
            raise ValueError('invalid pass thresholds')
        passed = qualifies(criteria['hq_ratio_min'], criteria['width_ratio_min'],
                           criteria['width_ratio_max'], criteria['kl_balance_max'])
        out.update(passed=passed, passed_baseline=passed)
    return out
