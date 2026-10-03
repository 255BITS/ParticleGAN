"""CPU matched saved-noise diagnosis of corrected CUDA toy regression."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parent
FIXES = ROOT.parent.parent
BASELINE = Path('/ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/training')
CORRECTED = FIXES/'validation'/'learned'/'training'
MODELS = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation')
PKG = FIXES/'pkg-CB64-RA2'
sys.path.insert(0, str(MODELS))
from models_metrics import networks, oracle_centres
sys.path.insert(0, str(PKG))
from particlegan.feature_cells import BoundedLatentGeometry
from particlegan.birth_death import ParticleBirthDeath


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def describe(x):
    x = x.double().flatten()
    return dict(min=float(x.min()), q05=float(torch.quantile(x, .05)),
                median=float(x.median()), mean=float(x.mean()),
                q95=float(torch.quantile(x, .95)), max=float(x.max()))


@torch.no_grad()
def exact_radius(query, points):
    nearest = query.new_full((len(query),), float('inf'))
    nearest_ids = torch.full((len(query),), -1, dtype=torch.long)
    for start in range(0, len(points), max(1, 2048//points.shape[1])):
        block = points[start:start+max(1, 2048//points.shape[1])]
        distance = (query[:, None]-block[None]).square().sum(2)
        distance.masked_fill_(distance == 0, float('inf'))
        value, index = distance.min(1)
        replace = value < nearest
        nearest = torch.minimum(nearest, value)
        nearest_ids[replace] = index[replace]+start
    radius = nearest.sqrt()*.5
    radius = torch.where(torch.isfinite(radius), radius, torch.zeros_like(radius))
    return radius, nearest_ids


def displacement(noise, width, radius):
    delta = width*noise
    fraction = (radius/delta.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)
    return delta*fraction[:, None]


@torch.no_grad()
def score_samples(raw):
    centers = oracle_centres()
    distance, mode = torch.cdist(raw, centers).min(1)
    accepted = distance <= .09
    mass = torch.bincount(mode[accepted], minlength=25).double()/len(raw)
    return dict(precision=float(accepted.float().mean()), coverage=int((mass >= .01).sum()),
                centre_distance=float(distance.mean()), unsupported_mass=float((~accepted).float().mean()),
                mass_tv=float((mass-.04).abs().sum()/2+(~accepted).double().mean()/2),
                supported_mass=mass.tolist())


@torch.no_grad()
def inspect_case(path, variant, noise, output_noise):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    state = saved['trainer']
    table_stationary = state['lr_settle'][0][1]['last_decisive'] == -1
    served = table_stationary and state['recipe']['serve_average'] > 0
    pairs = [('training', 'G', 'prior')]
    if served:
        pairs.append(('served_average', 'ema_G', 'ema_prior'))
    case = dict(variant=variant, step=state['completed_steps'], checkpoint_sha256=sha(path),
                saved_device=state['device'], table_stationary=table_stationary,
                served_pair='ema_G/ema_prior' if served else 'G/prior', tables=[])
    record = saved['record']; bd = record['diagnostics']['birth_death']
    case['original_cuda_record'] = dict(
        precision=record['metrics']['precision'], coverage=record['metrics']['coverage'],
        clean_precision=record['metrics']['clean_particle_centres']['precision'],
        clean_coverage=record['metrics']['clean_particle_centres']['coverage'],
        output_sigma=record['diagnostics']['output_sigma'], counters=bd['counters'],
        last={k:bd['last'].get(k) for k in ('iso_flagged','ordinary_moves','discoveries',
                  'ordinary_excess_cells','ordinary_deficit_cells','ordinary_inaccessible_birth_quota')}
            if bd['last'] else None)
    for label, g_name, p_name in pairs:
        with torch.random.fork_rng(devices=[]):
            G, _ = networks('toy')
        G.load_state_dict(state['models'][g_name]); G.eval()
        points = state['models'][p_name]['z']
        prior = SimpleNamespace(z=points)
        width = state['controller']['latent_bandwidth']
        kernel = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256)
        bounded, local_width = kernel._local_geometry(points, prior)
        exact, nearest_ids = exact_radius(points, points)
        legacy = ParticleBirthDeath._nearest_other(None, points, points)
        row = dict(table=label, model=g_name, prior=p_name, points_sha256=tensor_sha(points),
                   coordinate_spread=describe(points.std(0, unbiased=False)),
                   exact_unique_rows=len(torch.unique(points, dim=0)),
                   radius=dict(exact=describe(exact), reference_shortlist=describe(legacy), bounded=describe(bounded)),
                   radius_ratio=describe(bounded/exact.clamp_min(1e-20)),
                   exact_radius_hit_fraction=float(torch.isclose(bounded, exact, rtol=1e-5, atol=1e-6).float().mean()),
                   reference_zero_despite_nonidentical=int(((legacy == 0) & (exact > 0)).sum()),
                   local_coordinate_width=describe(local_width), bandwidth=describe(width),
                   work=dict(kernel.work), kernels={})
        eps = torch.finfo(points.dtype).eps
        # Equality in coordinate space can hide distinct full-vector rows.
        row['coordinate_unique_fraction'] = describe(torch.tensor(
            [len(torch.unique(points[:, axis]))/len(points) for axis in range(points.shape[1])]))
        row['nearest_pair_coordinate_difference'] = describe((points-points[nearest_ids.clamp_min(0)]).abs())
        clean = G(points)
        sigma = record['diagnostics']['output_sigma']
        row['clean_metrics'] = score_samples(clean)
        for name, delta in (
                ('current_bounded', displacement(noise, torch.minimum(width, local_width), bounded)),
                ('exact_dv12', displacement(noise, width, exact)),
                ('exact_cap_current_width', displacement(noise, torch.minimum(width, local_width), exact)),
                ('bounded_cap_reference_width', displacement(noise, width, bounded)),
                ('historical_fixed', displacement(noise, .025, noise.new_full((len(noise),), .05))),
                ('zero', torch.zeros_like(noise))):
            raw = G(points+delta)
            change = raw-clean
            row['kernels'][name] = dict(delta_rms=float(delta.square().mean().sqrt()),
                 delta_norm=describe(delta.norm(dim=1)), generator_delta_rms=float(change.square().mean().sqrt()),
                 generator_delta_norm=describe(change.norm(dim=1)), clean_output_metrics=score_samples(raw),
                 matched_output_noise_metrics=score_samples(raw+sigma*output_noise))
        case['tables'].append(row)
    return case


@torch.no_grad()
def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sources = [Path(__file__), ROOT/'PROTOCOL.md', MODELS/'models_metrics.py',
               PKG/'particlegan'/'feature_cells.py', PKG/'particlegan'/'birth_death.py',
               FIXES/'geometry'/'gpu-inputs.pt', FIXES/'geometry'/'GPU-INPUTS.json']
    cases = [(CORRECTED/'toy'/'CB64-RA2'/'checkpoint-0000.pt','CB64-RA2')]
    for step in (1000, 2000):
        cases += [(BASELINE/'toy'/variant/f'checkpoint-{step:04d}.pt',variant) for variant in ('E22','CB64-RA')]
        cases += [(CORRECTED/'toy'/'CB64-RA2'/f'checkpoint-{step:04d}.pt','CB64-RA2')]
    hashes = {str(p):sha(p) for p in sources+[p for p,_ in cases]}
    inputs = torch.load(FIXES/'geometry'/'gpu-inputs.pt', map_location='cpu', weights_only=False)
    noise = next(c for c in inputs['cases'] if c['name'] == 'fold128')['noise'][:1024]
    output_noise = next(c for c in inputs['cases'] if c['name'] == 'fold2')['noise'][:1024]
    result = dict(scope='CPU inference on fixed CUDA toy checkpoints and existing saved noise; no quality verdict',
                  source_sha256=hashes, latent_noise_sha256=tensor_sha(noise),
                  output_noise_sha256=tensor_sha(output_noise), noise_rows=1024, new_seeds=0, cases=[])
    for path, variant in cases:
        begin = time.perf_counter()
        row = inspect_case(path, variant, noise, output_noise)
        row['seconds'] = time.perf_counter()-begin
        result['cases'].append(row)
        print(json.dumps(dict(event='case',variant=variant,step=row['step'],seconds=row['seconds'],
             tables=[dict(table=t['table'],radius_exact=t['radius']['exact']['mean'],
                 radius_bounded=t['radius']['bounded']['mean'],radius_ratio=t['radius_ratio']['median'],
                 kernels={k:dict(rms=v['delta_rms'],G_rms=v['generator_delta_rms'],
                              precision=v['matched_output_noise_metrics']['precision'],
                              coverage=v['matched_output_noise_metrics']['coverage'])
                          for k,v in t['kernels'].items()}) for t in row['tables']])), flush=True)
    result.update(cuda_initialized=torch.cuda.is_initialized(),
                  sources_unchanged=hashes == {str(p):sha(p) for p in sources+[p for p,_ in cases]})
    assert not result['cuda_initialized'] and result['sources_unchanged']
    (ROOT/'kernel-diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
