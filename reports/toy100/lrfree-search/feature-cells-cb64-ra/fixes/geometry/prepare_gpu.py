"""Prepare immutable CPU tensors for root's one-slot CUDA kernel diagnostic."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
import json
from pathlib import Path
import torch
from types import SimpleNamespace
from diagnose_folded import ROOT, OLD, PREV, common, geometry, setup, sha


def main():
    target = ROOT/'gpu-inputs.pt'
    if target.exists():
        raise RuntimeError('GPU input artifact already exists')
    shared = setup(); cases = []
    for dim in (2, 128):
        ev = common.Evaluator('geometry', 2048, 'fold', dim, 'trained600', shared)
        trainer, bd = common.make_trainer(ev, 'cb64_ra', shared)
        _, child, parent, _, _ = common.mechanical_select(ev, 'cb64_ra', trainer, bd, shared)
        centers = ev.z.clone(); centers[child] = centers[parent]
        cases.append(dict(name=f'fold{dim}', prior=centers, query=centers,
            bandwidth=torch.as_tensor(ev.bandwidth), target=ev.target, seed=geometry.SEED,
            noise=torch.randn(centers.shape, generator=torch.Generator().manual_seed(geometry.SEED))))
    native = OLD/'validation'/'screen'/'grid100'/'final-state.pt'
    state = torch.load(native, map_location='cpu', weights_only=False)
    state = state.get('trainer', state)
    points = state['models']['prior']['z']
    cases.append(dict(name='native_cpu_saved_density', prior=points, query=points[:256],
        bandwidth=state['controller']['latent_bandwidth'], seed=1234,
        noise=torch.randn((256, 2), generator=torch.Generator().manual_seed(1234))))
    runs = ROOT.parent.parent/'feature-cells-cuda-retest-20260929'/'learned'/'training'/'mnist'
    sources = [native, PREV/'geometry_a'/'bundle.pt', PREV/'geometry_a'/'toy_family.py',
               OLD/'geometry'/'run_validation.py', common.CONFIG]
    for variant, step in (('E22', 0), ('E22', 2000), ('CB64-RA', 2000)):
        path = runs/variant/f'checkpoint-{step:04d}.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)['trainer']
        points = state['models']['prior']['z']
        cases.append(dict(name=f'mnist_{variant}_{step}', prior=points, query=points[:256],
            bandwidth=state['controller']['latent_bandwidth'], seed=314259,
            noise=torch.randn((256, 128), generator=torch.Generator().manual_seed(314259))))
        sources.append(path)
    torch.save(dict(cases=cases, config=json.loads(common.CONFIG.read_text())), target)
    receipt = dict(input_sha256=sha(target), cpu_only=True, cuda_initialized=torch.cuda.is_initialized(),
                   scope='matched saved CPU tensors/noise for one CUDA geometry diagnostic; no training or native acceptance',
                   cases=[dict(name=c['name'], n=len(c['prior']), dim=c['prior'].shape[1],
                               query_rows=len(c['query']), seed=c['seed']) for c in cases],
                   read_only_source_sha256={str(p):sha(p) for p in sources})
    assert not receipt['cuda_initialized']
    (ROOT/'GPU-INPUTS.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
