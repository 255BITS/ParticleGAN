"""Cheap geometric filter of one nonlocal signal, not a GAN training result.

Predeclared: tau=1, Euler step=.1 (values studied by He et al. 2026), 200
updates, float64, converged uniform couplings. No parameter/seed scan. The
12-to-8 law deliberately mirrors the host's mass mismatch, without noise or
a generator. Clean HQ is a proxy here, not the production sampled metric.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch


@torch.no_grad()
def coupling(x, y, tau=1., tolerance=1e-10, max_iterations=2000):
    logits = -.5 * torch.cdist(x, y).square() / tau
    u = torch.zeros(len(x), dtype=x.dtype)
    v = torch.zeros(len(y), dtype=y.dtype)
    la, lb = -math.log(len(x)), -math.log(len(y))
    for iteration in range(max_iterations):
        u = la - torch.logsumexp(logits + v[None], dim=1)
        v = lb - torch.logsumexp(logits + u[:, None], dim=0)
        plan = torch.exp(logits + u[:, None] + v[None])
        residual = max(float((plan.sum(1) - 1 / len(x)).abs().max()),
                       float((plan.sum(0) - 1 / len(y)).abs().max()))
        if residual <= tolerance:
            return plan, residual, iteration + 1
    raise RuntimeError(f"Sinkhorn marginal residual {residual} exceeds {tolerance}")


def field(x, y):
    cross, rx, ix = coupling(x, y)
    own, ro, io = coupling(x, x)
    return len(x) * (cross @ y - own @ x), max(rx, ro), max(ix, io)


def grade(x, centers):
    dist = torch.cdist(x, centers)
    return dict(modes=int((dist.min(0).values <= .21).sum()),
                clean_hq=float((dist.min(1).values <= .21).double().mean()))


def run(distinct=False):
    torch.set_num_threads(1)
    theta = torch.arange(8, dtype=torch.float64) * (2 * math.pi / 8)
    target = 3 * torch.stack((theta.cos(), theta.sin()), 1)
    assignments = torch.tensor([0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7])
    passing_cloud = target[assignments].clone()
    # A deterministic symmetry check, not a seed search: exact coincident
    # points cannot split under a permutation-equivariant deterministic field.
    angle = torch.arange(12, dtype=torch.float64) * (2 * math.pi / 12)
    offset = (.029 * torch.stack((angle.cos(), angle.sin()), 1)
              if distinct else torch.zeros_like(passing_cloud))
    passing_cloud = passing_cloud + offset
    matched, residual, _ = field(passing_cloud, passing_cloud)
    assert float(matched.abs().max()) == 0.
    output = dict(method="one_fixed_sinkhorn_cross_minus_self_field", tau=1.,
                  distinct=distinct, offset_radius=.029 if distinct else 0.,
                  step=.1, steps=200, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  matched_field_max=float(matched.abs().max()),
                  matched_marginal_residual=residual,
                  scope="clean free-particle diagnostic, not a host acquisition or stability gate",
                  source="https://arxiv.org/abs/2603.12366", cases={})
    for name, x in (("passing_12_to_8", passing_cloud.clone()),
                    ("missing_6", target[torch.where(assignments == 6, 5, assignments)].clone() + offset)):
        initial = grade(x, target)
        trace = []
        max_residual = 0.
        for step in range(1, 201):
            drift, residual, iterations = field(x, target)
            max_residual = max(max_residual, residual)
            x = x + .1 * drift
            trace.append(dict(step=step, **grade(x, target),
                              drift_max=float(drift.norm(dim=1).max()),
                              marginal_residual=residual, iterations=iterations))
        output['cases'][name] = dict(initial=initial, final=grade(x, target),
            first_failed_quality=next((r['step'] for r in trace if r['modes'] < 8 or r['clean_hq'] < .9), None),
            min_clean_hq=min(r['clean_hq'] for r in trace),
            max_marginal_residual=max_residual, trace=trace)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distinct', action='store_true')
    args = parser.parse_args()
    began = time.perf_counter()
    result = run(distinct=args.distinct)
    result['seconds'] = time.perf_counter() - began
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({**{k: v for k, v in result.items() if k != 'cases'},
        'cases': {k: {n: v for n, v in c.items() if n != 'trace'}
                  for k, c in result['cases'].items()}}), flush=True)
