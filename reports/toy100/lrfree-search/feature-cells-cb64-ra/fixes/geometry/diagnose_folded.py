"""One fixed folded family: repaired centers, matched noise, no retraining."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True

import hashlib
import importlib
import json
import time
import types
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
OLD = Path('/ml2/hypergan/gan-attempts/feature-cells-config-20260929')
PREV = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929')
sys.path.insert(0, str(ROOT/'pkg-CB64-RA2'))
from particlegan.feature_cells import BoundedLatentGeometry
from diagnose_saved import fixed_displacement, exact_displacement
sys.path.insert(0, str(OLD/'geometry'))
import run_validation as common
sys.path.insert(0, str(PREV/'geometry_a'))
import toy_family as geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    common.namespace('geometry_baseline', OLD/'pkg-CB64-RA'/'particlegan')
    common.namespace('geometry_reference', common.REFERENCE)
    return SimpleNamespace(cb=importlib.import_module('geometry_baseline.feature_cells'),
        ref=importlib.import_module('geometry_reference.birth_death'),
        cb_recipe=importlib.import_module('geometry_baseline.recipes').Recipe,
        ref_recipe=importlib.import_module('geometry_reference.recipes').Recipe,
        geometry=geometry, bundle=geometry.load_bundle(PREV/'geometry_a'/'bundle.pt'),
        config=json.loads(common.CONFIG.read_text()), ref_config=json.loads(common.REF_CONFIG.read_text()))


@torch.no_grad()
def main():
    shared = setup()
    results = dict(scope='CPU fixed-fixture geometry diagnostic, no acceptance claim',
                   seed=geometry.SEED, cases=[], input_sha256={str(p): sha(p) for p in
                   (PREV/'geometry_a'/'bundle.pt', PREV/'geometry_a'/'toy_family.py',
                    OLD/'geometry'/'run_validation.py', common.CONFIG, Path(__file__))})
    for dim in (2, 128):
        ev = common.Evaluator('geometry', 2048, 'fold', dim, 'trained600', shared)
        trainer, bd = common.make_trainer(ev, 'cb64_ra', shared)
        flags, child, parent, detail, _ = common.mechanical_select(ev, 'cb64_ra', trainer, bd, shared)
        centers = ev.z.clone(); centers[child] = centers[parent]
        prior = SimpleNamespace(z=centers)
        noise = torch.randn(centers.shape, generator=torch.Generator().manual_seed(geometry.SEED))
        # Preserve this fixture's original controller bandwidth, not an oracle.
        bandwidth = trainer.controller.latent_bandwidth
        bounded = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256)
        exact = exact_displacement(centers, noise, prior, bandwidth)
        candidate = bounded.displacement(centers, prior, bandwidth, noise)
        for name, delta in (('clean', torch.zeros_like(centers)),
                            ('fixed', fixed_displacement(centers, noise)),
                            ('exact_dv12', exact), ('bounded_local_dv12', candidate)):
            start = time.perf_counter()
            metrics = ev.evaluate(centers+delta, child)
            row = dict(dim=dim, kernel=name, metrics=metrics,
                       detector=ev.detector(flags), parent_receipt=ev.parents(ev.z, child, parent, detail),
                       delta_rms=float(delta.square().mean().sqrt()),
                       delta_norm_mean=float(delta.norm(dim=1).mean()),
                       original_final_geometry_gate=metrics['unsupported_mass'] <= .01
                           and metrics['rare_ratio'] >= .90 and metrics['mass_tv'] <= .015,
                       score_seconds=time.perf_counter()-start)
            results['cases'].append(row)
            print(json.dumps(dict(event='case', **row)), flush=True)
        results['cases'][-1]['geometry_work'] = dict(bounded.work)
        results['cases'][-1]['exact_vs_bounded_delta_max'] = float((exact-candidate).abs().max())
        assert ev.state_hash() == ev.initial_state_hash
    results['cuda_initialized'] = torch.cuda.is_initialized()
    assert not results['cuda_initialized']
    (ROOT/'folded-results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(dict(event='complete', cuda_initialized=False)), flush=True)


if __name__ == '__main__':
    main()
