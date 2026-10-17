"""Root-scheduled CUDA geometry diagnostic; no training or acceptance rerun."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parent
UUID = 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
ORACLE = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929/geometry_a/toy_family.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact_delta(latent, prior, width, noise):
    nearest = latent.new_full((len(latent),), float('inf'))
    for centers in prior.z.detach().split(max(1, 2048//latent.shape[1])):
        distance = (latent.detach()[:, None]-centers[None]).square().sum(-1)
        distance.masked_fill_(distance == 0, float('inf'))
        nearest = torch.minimum(nearest, distance.min(1).values)
    nearest = nearest.sqrt()
    radius = torch.where(torch.isfinite(nearest), .5*nearest, torch.zeros_like(nearest))
    delta = width*noise
    return delta*(radius/delta.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)[:, None]


def fixed_delta(noise):
    delta = .025*noise
    return delta*(.05/delta.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)[:, None]


def oracle_functions():
    parsed = ast.parse(ORACLE.read_text())
    score = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'oracle_modes')
    model = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'Generator')
    semantic = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'semantic')
    space = dict(torch=torch)
    exec(compile(ast.Module(body=[score, semantic], type_ignores=[]), str(ORACLE), 'exec'), space)
    return space['oracle_modes'], space['semantic']


@torch.no_grad()
def copy_check(cls, recipe_class, config, device, points, width):
    n, dim = points.shape
    prior = SimpleNamespace(z=torch.nn.Parameter(points.clone()))
    ema = SimpleNamespace(z=torch.nn.Parameter(points.clone()+10., requires_grad=False))
    row_values = torch.arange(n*dim, device=device, dtype=points.dtype).reshape(n, dim)
    moments = dict(exp_avg=row_values.clone(), exp_avg_sq=row_values+1., max_exp_avg_sq=row_values+2.)
    opt = SimpleNamespace(state={prior.z:moments}, latent_history=row_values+3.)
    recipe = recipe_class(**dict(config, num_particles=n, z_dim=dim))
    critic = torch.nn.Sequential(torch.nn.Linear(dim, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).to(device)
    trainer = SimpleNamespace(prior=prior, ema_prior=ema, opt_g=opt, recipe=recipe, D=critic,
             device=device, dtype=points.dtype, controller=SimpleNamespace(variant='dv12',
             latent_bandwidth=width, latent_applications=[]))
    bd = cls(trainer, 90229)
    child = torch.arange(n-32, n, device=device); parent = torch.arange(32, device=device)
    before, ema_before = points.clone(), ema.z.clone()
    before_moments = {k:v.clone() for k,v in moments.items()}; before_history = opt.latent_history.clone()
    draw = torch.Generator(device=device).set_state(bd.stream.get_state())
    expected = bd.latent_geometry.displacement(before[parent], prior, width,
                      torch.randn((32, dim), device=device, generator=draw))
    bd._move(trainer, child, parent)
    checks = dict(prior=torch.equal(prior.z[child], before[parent]+expected),
                  ema=torch.equal(ema.z[child], ema_before[parent]+expected),
                  history=torch.equal(opt.latent_history[child], before_history[parent]),
                  optimizer=all(torch.equal(moments[k][child], v[parent]) for k,v in before_moments.items()),
                  private_rng=torch.equal(bd.stream.get_state(), draw.get_state()))
    saved = bd.state_dict(); bd.load_state_dict(saved)
    checks['cache_cleared_on_load'] = not bd.latent_geometry._entries
    require(all(checks.values()), f'CUDA copy contract failed: {checks}')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=ROOT/'pkg-CB64-RA2')
    args = parser.parse_args()
    target = ROOT/'gpu-kernel-result.json'
    require(not target.exists(), 'GPU result already exists')
    receipt = json.loads((ROOT/'GPU-INPUTS.json').read_text())
    require(sha(ROOT/'gpu-inputs.pt') == receipt['input_sha256'], 'GPU input tensors changed')
    for path, expected in receipt['read_only_source_sha256'].items():
        require(sha(path) == expected, f'Original input changed: {path}')
    source_map = {str(p):sha(p) for p in sorted((args.package_root/'particlegan').glob('*.py'))}
    sys.path.insert(0, str(args.package_root))
    from particlegan.feature_cells import BoundedLatentGeometry, FeatureCellBirthDeath
    from particlegan.recipes import Recipe
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                       check=True, capture_output=True, text=True).stdout
    physical = dict(line.strip().split(', ', 1) for line in inventory.strip().splitlines())
    require(physical['0'] == UUID, 'Physical GPU0 UUID mismatch')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One CUDA GPU required')
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    cuda_uuid = 'GPU-'+str(properties.uuid).removeprefix('GPU-').lower()
    require(cuda_uuid == UUID, 'Visible cuda0 is not frozen physical GPU0')
    torch.cuda.set_per_process_memory_fraction(.2)
    torch.manual_seed(90229); torch.cuda.manual_seed(90229)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device = torch.device('cuda:0')
    inputs = torch.load(ROOT/'gpu-inputs.pt', map_location='cpu', weights_only=False)
    oracle_modes, semantic = oracle_functions()
    result = dict(scope='CUDA saved-input kernel/copy diagnostic; no retraining or original quality acceptance',
                  physical_gpu=0, cuda_uuid=cuda_uuid, input_sha256=receipt['input_sha256'],
                  script_sha256=sha(Path(__file__)), package_source_sha256=source_map,
                  runtime=dict(torch=str(torch.__version__), gpu=properties.name, cpu_threads=2,
                               memory_fraction=.2, deterministic=True, tf32=False), cases=[])
    for case in inputs['cases']:
        points, query, width, noise = (case[k].to(device) for k in ('prior', 'query', 'bandwidth', 'noise'))
        prior = SimpleNamespace(z=points)
        kernel = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256)
        torch.cuda.synchronize(); start=time.perf_counter()
        proposed = kernel.displacement(query, prior, width, noise)
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
        replay = kernel.displacement(query, prior, width, noise)
        require(torch.equal(proposed, replay), 'CUDA derived-cache replay mismatch')
        deltas = dict(fixed=fixed_delta(noise), exact_dv12=exact_delta(query, prior, width, noise),
                      bounded_local_dv12=proposed)
        row = dict(name=case['name'], seconds=seconds, geometry_work=dict(kernel.work),
                   cached_draw_bit_exact=True, kernels={})
        for name, delta in deltas.items():
            detail=dict(delta_rms=float(delta.square().mean().sqrt()), norm_mean=float(delta.norm(dim=1).mean()),
                        norm_max=float(delta.norm(dim=1).max()))
            if case['name'].startswith('fold'):
                modes=oracle_modes(semantic(SimpleNamespace(mechanism='fold'), query+delta))
                mass=torch.bincount(modes, minlength=5).double()/len(modes)
                target_mass=torch.cat((case['target'].to(device),mass.new_zeros(1)))
                detail.update(support=float(1-mass[-1]), rare_ratio=float(mass[3]/target_mass[3]),
                              mass_tv=float((mass-target_mass).abs().sum()/2))
            row['kernels'][name]=detail
        require(kernel.work['max_candidates'] <= 64 and kernel.work['max_query_rows'] <= 256,
                'Latent geometry work bound exceeded')
        result['cases'].append(row)
        print(json.dumps(dict(event='kernel_case', **row)), flush=True)
    example=next(c for c in inputs['cases'] if c['name']=='mnist_E22_0')
    result['copy_checks']=copy_check(FeatureCellBirthDeath, Recipe, inputs['config'], device,
                                   example['prior'].to(device), example['bandwidth'].to(device))
    result['source_unchanged']=all(sha(p)==expected for p,expected in source_map.items())
    require(result['source_unchanged'], 'Package source changed during CUDA diagnostic')
    result.update(status='PASS', peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(event='complete', status='PASS', path=str(target), copy_checks=result['copy_checks'])), flush=True)


if __name__=='__main__':
    main()
