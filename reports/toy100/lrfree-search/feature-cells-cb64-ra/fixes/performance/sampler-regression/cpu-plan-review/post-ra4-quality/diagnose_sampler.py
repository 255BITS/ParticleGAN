"""CPU saved-input sampler/EMA diagnostic; no training or quality verdict."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT.parent / 'feature-cells-cuda-retest-20260929'
MODELS = ROOT.parent / 'scaling-portability-20260929/validation'
PACKAGE = ROOT / 'pkg-CB64-RA4'
sys.path.insert(0, str(MODELS))
from models_metrics import networks, oracle_centres
sys.path.insert(0, str(PACKAGE))
from particlegan.feature_cells import BoundedLatentGeometry, LatentLineage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def describe(tensor):
    values = tensor.detach().double().flatten()
    return dict(min=float(values.min()), median=float(values.median()), mean=float(values.mean()),
                q05=float(torch.quantile(values, .05)), q95=float(torch.quantile(values, .95)),
                max=float(values.max()))


@torch.no_grad()
def exact_radius(points):
    output = points.new_full((len(points),), float('inf'))
    for centers in points.split(max(1, 2048 // points.shape[1])):
        distance = (points[:, None] - centers[None]).square().sum(-1)
        distance.masked_fill_(distance == 0, float('inf'))
        output = torch.minimum(output, distance.min(1).values)
    output = output.sqrt() * .5
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


def delta(noise, width, radius):
    raw = width * noise
    return raw * (radius / raw.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)[:, None]


@torch.no_grad()
def score(samples):
    distance, mode = torch.cdist(samples, oracle_centres()).min(1)
    accepted = distance <= .09
    mass = torch.bincount(mode[accepted], minlength=25).double() / len(samples)
    return dict(precision=float(accepted.float().mean()), coverage=int((mass >= .01).sum()),
                mass_tv=float((mass - .04).abs().sum() / 2 + (~accepted).double().mean() / 2),
                centre_distance=float(distance.mean()))


@torch.no_grad()
def inspect(path, variant, noise, output_noise):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    state, record = saved['trainer'], saved['record']
    bandwidth = state['controller']['latent_bandwidth']
    tester = state['lr_settle'][0][1]
    lineage = None
    if variant == 'CB64-RA4':
        graph = state['birth_death']['lineage_neighbors']
        lineage = LatentLineage(len(graph), graph.shape[1], torch.device('cpu'))
        lineage.validate(graph)
        lineage.neighbors.copy_(graph)
    result = dict(variant=variant, step=state['completed_steps'], checkpoint=str(path),
        checkpoint_sha256=sha(path), bandwidth=describe(bandwidth),
        table_tester={key:tester.get(key) for key in ('s', 'b', 'last_decisive')},
        ema_rate=tester['s'] / (state['recipe']['serve_average'] * tester['b']),
        original_emitted=record['metrics'], output_sigma=record['diagnostics']['output_sigma'],
        original_latent_applications=record['diagnostics']['controller'].get('latent_applications'),
        original_birth_last=record['diagnostics']['birth_death']['last'], tables=[])
    rows = torch.arange(len(noise))
    geometry = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256, lineage=lineage)
    radii, widths, perturbations = {}, {}, {}
    for name, model_key, prior_key in (('live', 'G', 'prior'), ('ema', 'ema_G', 'ema_prior')):
        points = state['models'][prior_key]['z']
        prior = SimpleNamespace(z=points)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(state['cpu_rng'])
            model, _ = networks('toy')
        model.load_state_dict(state['models'][model_key]); model.eval()
        exact = exact_radius(points)
        bounded, local = geometry._local_geometry(points, prior, rows=rows)
        fresh = points.std(0, unbiased=False) * len(points) ** (-1. / points.shape[1])
        if variant == 'CB64-RA4':
            used_radius, width = bounded, torch.minimum(bandwidth, local)
            current = geometry.displacement(points, prior, bandwidth, noise, rows=rows)
            no_graph = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256)
            unlinked = no_graph.displacement(points, prior, bandwidth, noise)
            fresh_delta = geometry.displacement(points, prior, fresh, noise, rows=rows)
        else:
            used_radius, width = exact, bandwidth
            current = delta(noise, bandwidth, exact)
            unlinked = current
            fresh_delta = delta(noise, fresh, exact)
        radii[name], widths[name], perturbations[name] = used_radius, width, current
        clean = model(points)
        row = dict(table=name, coordinate_spread=describe(points.std(0, unbiased=False)),
            exact_radius=describe(exact), bounded_radius=describe(bounded), applied_radius=describe(used_radius),
            radius_hit_fraction=float(torch.isclose(bounded, exact, rtol=1e-5, atol=1e-6).float().mean()),
            applied_width=describe(width), fresh_controller_width=describe(fresh),
            global_cap_coordinate_fraction=float((bandwidth < local).float().mean()),
            geometry_work=dict(geometry.work), clean=score(clean),
            delta_rms=float(current.square().mean().sqrt()),
            generator_delta_rms=float((model(points + current) - clean).square().mean().sqrt()),
            current_latent_only=score(model(points + current)),
            current_latent_and_output_noise=score(model(points + current) + result['output_sigma'] * output_noise),
            no_latent_with_same_output_noise=score(clean + result['output_sigma'] * output_noise),
            no_lineage_delta_rms=float(unlinked.square().mean().sqrt()),
            no_lineage_with_same_output_noise=score(model(points + unlinked) + result['output_sigma'] * output_noise),
            fresh_bandwidth_delta_bits_same=torch.equal(current, fresh_delta),
            fresh_bandwidth_delta_rms=float(fresh_delta.square().mean().sqrt()),
            fresh_bandwidth_with_same_output_noise=score(model(points + fresh_delta) + result['output_sigma'] * output_noise))
        result['tables'].append(row)
    # The current row-copy implementation reuses the live displacement in EMA.
    # This fixed-noise probe checks the two current priors' own kernel bounds.
    live_delta = perturbations['live']
    ema_radius = radii['ema']
    ratio = live_delta.norm(dim=1) / ema_radius.clamp_min(1e-20)
    result['copy_ema_geometry_probe'] = dict(
        scope='all saved rows as possible parents; not historical GPU copy reconstruction',
        current_live_delta_over_ema_radius=describe(ratio),
        ema_bound_violation_rows=int((ratio > 1.00001).sum()),
        paired_delta_bound_violation_rows=int((perturbations['ema'].norm(dim=1) > ema_radius * 1.00001 + 1e-20).sum()),
        live_to_ema_radius_ratio=describe(radii['live'] / ema_radius.clamp_min(1e-20)),
        shared_noise_delta_difference_rms=float((live_delta - perturbations['ema']).square().mean().sqrt()))
    return result


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = [(lane / f'checkpoint-{step:04d}.pt', variant)
        for variant, lane in (('E22', BASE / 'learned/training/toy/E22'),
                             ('CB64-RA4', ROOT / 'validation-ra4/learned/training/toy/CB64-RA4'))
        for step in (1250, 1500, 1750, 2000)]
    inputs_path = ROOT / 'geometry/gpu-inputs.pt'
    sources = [Path(__file__), inputs_path, MODELS / 'models_metrics.py',
               PACKAGE / 'particlegan/feature_cells.py', PACKAGE / 'particlegan/birth_death.py',
               PACKAGE / 'particlegan/continuous.py', PACKAGE / 'particlegan/training.py'] + [p for p, _ in paths]
    hashes = {str(p):sha(p) for p in sources}
    fixtures = torch.load(inputs_path, map_location='cpu', weights_only=False)['cases']
    noise = next(x for x in fixtures if x['name'] == 'fold128')['noise'][:1024]
    output_noise = next(x for x in fixtures if x['name'] == 'fold2')['noise'][:1024]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope=__doc__, source_sha256=hashes,
        latent_noise_sha256=tensor_sha(noise), output_noise_sha256=tensor_sha(output_noise),
        new_seeds=0, optimizer_updates=0, cuda_initialized=False, cases=[])
    for path, variant in paths:
        row = inspect(path, variant, noise, output_noise)
        result['cases'].append(row)
        print(json.dumps(dict(variant=variant, step=row['step'],
            tables=[{k:t[k] for k in ('table', 'clean', 'delta_rms', 'generator_delta_rms',
                'current_latent_and_output_noise', 'no_lineage_with_same_output_noise',
                'global_cap_coordinate_fraction', 'fresh_bandwidth_delta_bits_same')} for t in row['tables']],
            copy_ema_geometry_probe=row['copy_ema_geometry_probe'])), flush=True)
    result['cuda_initialized'] = torch.cuda.is_initialized()
    result['sources_unchanged'] = hashes == {str(p):sha(p) for p in sources}
    assert not result['cuda_initialized'] and result['sources_unchanged']
    (HERE / 'sampler-diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
