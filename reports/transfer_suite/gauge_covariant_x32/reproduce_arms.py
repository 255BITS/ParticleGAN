"""Gauge-complete ×32 two_broad: fixed lengths vs kernels×32 + init_std×32."""
from __future__ import annotations
import json, gzip, hashlib, time, sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from particlegan import get_recipe, ParticlePrior

ROOT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from benchmarks.transfer_suite import shared_discriminator_search as architecture
from benchmarks.transfer_suite import shared_batch_feature_research as batchfeat
from benchmarks.transfer_suite import vector_tasks
from benchmarks.transfer_suite.compare_defaults import plan, ema_verdict
from benchmarks.transfer_suite.shared_variants import architecture_spec
from benchmarks.transfer_suite.protocol import test_verdict

SCALE = 32.0
BASE_INIT_STD = 0.5


def make_job():
    base = next(j for j in plan() if j['spec']['name'] == 'vector_two_broad')
    job = deepcopy(base)
    spec = job['spec']
    s = SCALE
    spec['means'] = [[s * m[0], s * m[1]] for m in spec['means']]
    spec['covariances'] = [
        [[s * s * c[0][0], s * s * c[0][1]], [s * s * c[1][0], s * s * c[1][1]]]
        for c in spec['covariances']
    ]
    spec['name'] = f'vector_two_broad_gauge_x{int(SCALE)}'
    spec['family'] = 'separated_broad_gauge_covariant'
    spec['importance_reason'] = (
        'Stationary isotropic unit change of two_broad; absolute kernel scales and '
        'ParticlePrior init_std must both track the gauge.')
    spec['limitations'] = (
        f'Geometry is vector_two_broad after global ×{SCALE}; control scales kernels and init_std.')
    job['architecture'] = 'break_candidate'
    job['reference'] = None
    job['reference_sha256'] = None
    return job


def make_scaled_prior(std):
    def scaled_prior(n, z_dim, init_std=BASE_INIT_STD, generator=None, **kwargs):
        return ParticlePrior(n, z_dim, init_std=std, generator=generator, **kwargs)
    return scaled_prior


def run_arm(name, card, out, init_std):
    out.mkdir(parents=True, exist_ok=True)
    job = make_job()
    recipe_card = json.loads((ROOT / 'recipe.json').read_text())
    recipe = get_recipe(**recipe_card['overrides']).replace(name=recipe_card['name'])
    architecture_spec(job['spec'], batchfeat.variant(card))
    torch.set_num_threads(1)
    print(f'START {name} kernels={card["kernel_scales"]} init_std={init_std}', flush=True)
    t0 = time.perf_counter()
    with patch.object(architecture, 'recipe', lambda: recipe), \
         patch.object(architecture, 'constructor', batchfeat.constructor), \
         patch.object(architecture, 'variant', batchfeat.variant), \
         patch.object(vector_tasks, 'ParticlePrior', make_scaled_prior(init_std)):
        payload = architecture.episode(job, deepcopy(card))
    payload['arm'] = name
    payload['scale'] = SCALE
    payload['init_std'] = init_std
    payload['seconds_wall'] = time.perf_counter() - t0
    payload['verdict'] = test_verdict(payload['spec'], payload['result'])
    payload['ema_verdict'] = ema_verdict(payload['spec'], payload['result'])
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + '\n').encode()
    (out / f'{name}.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    summary = dict(
        arm=name,
        status=payload['verdict']['status'],
        suffix=payload['verdict'].get('convergence', {}).get('passing_suffix'),
        shortfall=payload['verdict'].get('shortfall'),
        live=payload['result'].get('live'),
        error=payload['result'].get('error'),
        seconds=payload['result'].get('seconds'),
        kernel_scales=card['kernel_scales'],
        init_std=init_std,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    (out / f'{name}.summary.json').write_text(json.dumps(
        {k: summary[k] for k in summary if k != 'live'}, indent=2) + '\n')
    print(json.dumps(dict(event='DONE', **{k: summary[k] for k in summary if k != 'live'}), default=str), flush=True)
    print(json.dumps(dict(event='LIVE', arm=name, live=summary['live']), default=str), flush=True)
    return summary


def main():
    out = ROOT / 'runs' / time.strftime('run-%Y%m%d-%H%M%S')
    winner = json.loads((ROOT / 'winner_card.json').read_text())
    control = json.loads((ROOT / 'control_card.json').read_text())
    results = [
        run_arm('winner_fixed_gauge_x32', winner, out, init_std=BASE_INIT_STD),
        run_arm('control_gauge_x32', control, out, init_std=BASE_INIT_STD * SCALE),
    ]
    (out / 'index.json').write_text(json.dumps(results, indent=2) + '\n')
    print('WROTE', out, flush=True)


if __name__ == '__main__':
    main()
