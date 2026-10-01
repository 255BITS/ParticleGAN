"""Fixed folded turnover/copy diagnostic and saved native2D kernel density."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

from diagnose_folded import ROOT, OLD, PREV, common, geometry, setup, sha
from diagnose_saved import fixed_displacement, exact_displacement
from particlegan.feature_cells import BoundedLatentGeometry


@torch.no_grad()
def main():
    shared = setup()
    results = dict(scope='CPU small causal diagnostics; saved CPU native state is not canonical GPU acceptance',
                   folded_copies=[], native_density={})
    for backend in ('fixed', 'bounded_local_dv12'):
        ev = common.Evaluator('geometry', 2048, 'fold', 2, 'trained600', shared)
        if backend == 'bounded_local_dv12':
            shared.cb = importlib.import_module('particlegan.feature_cells')
            shared.cb_recipe = importlib.import_module('particlegan.recipes').Recipe
        full = common.full_scope(ev, 'cb64_ra', shared)
        # Correct the reused observer's historical label for this private law.
        full['native_jitter'] = {'kernel':backend, 'bandwidth':ev.bandwidth,
                                'cap':'.05 fixed' if backend == 'fixed' else 'bounded half nearest other'}
        row = dict(kernel=backend, original_final_gate=full['final']['unsupported_mass'] <= .01
                   and full['final']['rare_ratio'] >= .90 and full['final']['mass_tv'] <= .015
                   and full['planted_repeated_fraction'] <= .20, result=full)
        results['folded_copies'].append(row)
        print(json.dumps(dict(event='folded_copies', kernel=backend, final=full['final'],
                              ordinary_moves=full['ordinary_moves'], isolation_moves=full['isolation_moves'],
                              original_final_gate=row['original_final_gate'])), flush=True)
    path = OLD/'validation'/'screen'/'grid100'/'final-state.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    state = saved.get('trainer', saved)
    prior = SimpleNamespace(z=state['models']['prior']['z'])
    latent = prior.z[:256]
    bandwidth = state['controller']['latent_bandwidth']
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(1234))
    kernel = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256)
    exact = exact_displacement(latent, noise, prior, bandwidth)
    bounded = kernel.displacement(latent, prior, bandwidth, noise)
    rows = []
    for label, delta in (('fixed', fixed_displacement(latent, noise)), ('exact_dv12', exact), ('bounded_local_dv12', bounded)):
        rows.append(dict(kernel=label, delta_rms=float(delta.square().mean().sqrt()),
                         norm_mean=float(delta.norm(dim=1).mean()), norm_max=float(delta.norm(dim=1).max())))
    results['native_density'] = dict(checkpoint_sha256=sha(path), particles=len(prior.z),
             query_rows=len(latent), seed=1234, rows=rows, bounded_work=dict(kernel.work),
             exact_bounded_delta_max=float((exact-bounded).abs().max()),
             controller_bandwidth_mean=float(bandwidth.mean()),
             geometry_source_sha256=sha(ROOT/'pkg-CB64-RA2'/'particlegan'/'feature_cells.py'))
    print(json.dumps(dict(event='native_density', **results['native_density'])), flush=True)
    results['cuda_initialized'] = torch.cuda.is_initialized()
    assert not results['cuda_initialized']
    (ROOT/'copies-density-results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
