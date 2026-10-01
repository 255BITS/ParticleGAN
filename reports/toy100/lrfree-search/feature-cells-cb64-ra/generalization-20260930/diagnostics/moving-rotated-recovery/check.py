"""Read-only CPU geometry and partial-witness isolation from frozen checkpoints."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
LANE = STUDY / 'validation-ra14-r2'
RUN = LANE / 'moving/rotated100'
PACKAGE = STUDY / 'pkg-RA14-replay'
sys.path.insert(0, str(PACKAGE))

import numpy as np
import torch
from scipy.stats import ncx2
from particlegan.feature_cells import BoundedLatentGeometry, LatentLineage
from particlegan.output_moments import (fit_projection, OutputObservation, freeze_moment,
    FixedOutputMoment, odd_witness, group_means)

torch.set_num_threads(1)
assert not torch.cuda.is_initialized()


def deny_cuda(*args, **kwargs):
    raise AssertionError('CUDA forbidden in CPU recovery diagnostic')


torch.cuda._lazy_init = deny_cuda
cpu_rng_before = torch.get_rng_state().clone()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    frozen = json.loads((LANE / 'SOURCE-FREEZE.json').read_text())
    for path, expected in frozen['hashes'].items():
        assert sha(path) == expected, path
    return dict(status='VALID', files=len(frozen['hashes']),
                package_sha256=frozen['package_sha256'],
                source_freeze_sha256=sha(LANE / 'SOURCE-FREEZE.json'))


before = verify()
frames = np.load(RUN / 'frames.npz')
base_centers = frames['centers'].astype(np.float64)
verdict = json.loads((RUN / 'frames.npz.verdict.json').read_text())
bar = .9 * verdict['pre_turn_hq']
inputs = {str(path): sha(path) for path in [RUN / f'checkpoint-{s:06d}.pt' for s in (500, 1000, 1500)] +
          [RUN / 'frames.npz', RUN / 'frames.npz.verdict.json', RUN / 'adapted_runner.py', RUN / 'COMPLETION.json']}


def rotation(angle):
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


def nearest(q, centers):
    dist2 = ((q[:, None, :] - centers[None, :, :]) ** 2).sum(2)
    ids = dist2.argmin(1)
    return ids, np.sqrt(dist2[np.arange(len(q)), ids])


def geometry(state, generator, prior, centers):
    W = state['models'][generator]['weight'].double().numpy()
    b = state['models'][generator]['bias'].double().numpy()
    z = state['models'][prior]['z'].double().numpy()
    q = z @ W.T + b
    ids, r = nearest(q, centers)
    counts = np.bincount(ids, minlength=100)
    weights = counts / counts.sum()
    means = np.array([q[ids == i].mean(0) if counts[i] else centers[i] for i in range(100)])
    errors = means - centers
    covariances = np.array([np.cov(q[ids == i], rowvar=False, bias=True) if counts[i] > 1 else np.zeros((2, 2))
                           for i in range(100)])
    expected_hq = ncx2.cdf((.09 / .029) ** 2, 2, (r / .029) ** 2)
    u, singular_values, vh = np.linalg.svd(W)
    polar = u @ vh
    xm = (means * weights[:, None]).sum(0)
    ym = (centers * weights[:, None]).sum(0)
    u, _, vh = np.linalg.svd((means - xm).T @ (weights[:, None] * (centers - ym)))
    rigid = u @ vh
    corrected = (q - xm) @ rigid + ym
    _, corrected_r = nearest(corrected, centers)
    summary = dict(generator=generator, prior=prior, W=W.tolist(), bias=b.tolist(),
        generator_polar_degrees=math.degrees(math.atan2(polar[1, 0], polar[0, 0])),
        generator_singular_values=singular_values.tolist(), prior_mean=z.mean(0).tolist(),
        prior_covariance=np.cov(z, rowvar=False, bias=True).tolist(),
        output_mean=q.mean(0).tolist(), output_covariance=np.cov(q, rowvar=False, bias=True).tolist(),
        clean_hq=float((r <= .09).mean()), clean_radius_p50_p90_p99=np.quantile(r, [.5, .9, .99]).tolist(),
        expected_hq_output_noise_only=float(expected_hq.mean()),
        mode_mean_offset_rms=float(np.sqrt((weights * (errors ** 2).sum(1)).sum())),
        missing_raw_modes=np.flatnonzero(counts == 0).tolist(),
        best_rigid_correction_degrees=math.degrees(math.atan2(rigid[0, 1], rigid[0, 0])),
        best_rigid_correction_expected_hq=float(ncx2.cdf((.09 / .029) ** 2, 2, (corrected_r / .029) ** 2).mean()),
        mass_tv=float(.5 * np.abs(weights - .01).sum()),
        clean_within_mode_covariance_trace_mean=float((weights * np.trace(covariances, axis1=1, axis2=2)).sum()),
        modes=[dict(mode=i, rows=int(counts[i]), mean=means[i].tolist(), target=centers[i].tolist(),
                    offset=errors[i].tolist(), covariance=covariances[i].tolist(),
                    expected_hq=float(expected_hq[ids == i].mean()) if counts[i] else 0.) for i in range(100)])
    return summary, q


def kernel_bound(state, generator, prior, centers):
    z = state['models'][prior]['z']
    W = state['models'][generator]['weight'].double().numpy()
    q = z.double().numpy() @ W.T + state['models'][generator]['bias'].double().numpy()
    settings = state['birth_death']['settings']
    lineage = LatentLineage(len(z), settings['lineage_degree'], 'cpu')
    lineage.neighbors = state['birth_death']['lineage_neighbors'].clone()
    kernel = BoundedLatentGeometry(rank=settings['latent_rank'], neighbors=settings['latent_neighbors'],
                                  chunk=settings['chunk'], lineage=lineage)
    radius, _ = kernel._local_geometry(z, SimpleNamespace(z=z), rows=torch.arange(len(z)))
    output_radius = radius.numpy() * np.linalg.norm(W, 2)
    _, distance = nearest(q, centers)
    lower = ncx2.cdf((.09 / .029) ** 2, 2, ((distance + output_radius) / .029) ** 2).mean()
    # The lower bound uses one nearest ball. The upper bound sums every ball,
    # so it remains valid even when the perturbation changes the nearest mode.
    all_distances = np.sqrt(((q[:, None, :] - centers[None, :, :]) ** 2).sum(2))
    upper_rows = ncx2.cdf((.09 / .029) ** 2, 2,
                        (np.maximum(all_distances - output_radius[:, None], 0) / .029) ** 2).sum(1)
    upper = np.minimum(upper_rows, 1.).mean()
    return dict(generator=generator, prior=prior, latent_radius_mean=float(radius.mean()),
                latent_radius_p50_p90_p99=np.quantile(radius.numpy(), [.5, .9, .99]).tolist(),
                expected_hq_bounds=[float(lower), float(upper)],
                interpretation='Uniform table expectation with output Gaussian; triangle bound covers every bounded latent perturbation, without RNG draws')


rows = []
states = {}
for step in (500, 1000, 1500):
    state = torch.load(RUN / f'checkpoint-{step:06d}.pt', map_location='cpu', weights_only=False)
    assert state['schema'] == 4 and state['completed_steps'] == step
    states[step] = state
    centers = base_centers @ rotation(math.radians((step // 500 - 1) * 30)).T
    fast, qfast = geometry(state, 'G', 'prior', centers)
    average, qaverage = geometry(state, 'ema_G', 'ema_prior', centers)
    hybrid_g, _ = geometry(state, 'G', 'ema_prior', centers)
    hybrid_z, _ = geometry(state, 'ema_G', 'prior', centers)
    row = dict(step=step, target_degrees=(step // 500 - 1) * 30,
        original_gate=verdict['periods'][step // 500 - 1], served_source=state['policy']['served_source'],
        fast=fast, averaged=average, live_G_average_prior_expected_hq=hybrid_g['expected_hq_output_noise_only'],
        average_G_live_prior_expected_hq=hybrid_z['expected_hq_output_noise_only'],
        fast_average_output_rms=float(np.sqrt(((qfast - qaverage) ** 2).sum(1).mean())),
        surprise=state['surprise']['log'], guard=state['reopen_guard'],
        paired_average=state['birth_death']['last']['paired_average'],
        mean_transport=state['birth_death']['last']['mean_transport'],
        mean_witness_fires=state['birth_death']['counters']['mean_witness_fires'],
        mean_moves=state['birth_death']['counters']['mean_moves'],
        birth_death_moves=state['birth_death']['counters']['moves'])
    rows.append(row)
    print(json.dumps(dict(step=step, served=row['served_source'], original_hq=row['original_gate']['hq'],
        expected_fast_hq=fast['expected_hq_output_noise_only'], expected_average_hq=average['expected_hq_output_noise_only'],
        generator_degrees=fast['generator_polar_degrees'], local_mean_offset_rms=fast['mode_mean_offset_rms'],
        mean_fires=row['mean_witness_fires'], mean_moves=row['mean_moves'])), flush=True)

final = states[1500]
target = base_centers @ rotation(math.radians(60)).T
bounds = [kernel_bound(final, g, z, target) for g, z in [('G', 'prior'), ('ema_G', 'ema_prior')]]

# The exact ephemeral critic chart was not checkpointed. This adapter isolates
# the all-group veto in raw geometry; evaluator labels are diagnostic only.
class RawGeometry:
    valid_metric = True
    rank = 2
    duplicate_fraction = 0.
    mass_groups = 100
    cells = 128

    def transform(self, x):
        return x.double()

    def _assign_metric(self, x):
        distances = torch.cdist(x, torch.from_numpy(target)).square()
        values, ids = distances.min(1)
        return ids, values

    def _mass_topology(self):
        return torch.arange(100)


snapshot = RawGeometry()
real = final['birth_death']['reservoir'].double()
ema = final['models']['ema_prior']['z'].double() @ final['models']['ema_G']['weight'].double().T + final['models']['ema_G']['bias'].double()
projection, reason = fit_projection(real[0::2])
assert reason is None
fixed, reason = freeze_moment(snapshot, real[0::2], real[0::2],
                            OutputObservation(ema, projection.transform(ema)), projection)
assert fixed is None and reason == 'missing_EMA_group'
even_groups = snapshot._assign_metric(real[0::2])[0]
metric = projection.transform(real[0::2])
centers, counts = group_means(metric, even_groups, 100)
assert bool((counts >= 2).all())
squares = (metric - centers[even_groups]).square().sum(1)
ss = torch.zeros(100, dtype=torch.float64)
ss.index_add_(0, even_groups, squares)
scales = (ss / counts / projection.rank).sqrt()
radius = math.sqrt(projection.rank / .05)


def psi(x, groups):
    z = (x - centers[groups]) / scales[groups, None]
    return z * (radius / z.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)[:, None]


even_means, _ = group_means(psi(metric, even_groups), even_groups, 100)
ema_groups = snapshot._assign_metric(ema)[0]
ema_means, ema_counts = group_means(psi(projection.transform(ema), ema_groups), ema_groups, 100)
active = ema_counts > 0
residual = even_means - ema_means
lengths = residual.norm(dim=1)
directions = torch.where((active & (lengths > 0))[:, None],
    residual / lengths.clamp_min(1e-30)[:, None], torch.zeros_like(residual))
weights = counts.double() / counts.sum()
partial = FixedOutputMoment(projection, centers, scales, even_means, ema_means, ema_counts,
                          weights * active, directions, radius, projection.rank, 2, 128)
witness = odd_witness(snapshot, partial, real[1::2], real[1::2])
assert witness['valid'] and witness['fires'] and witness['observations'] == 10000
assert bool((directions[~active] == 0).all())
assert active.sum() == 99 and (~active).nonzero().flatten().tolist() == [99]
isolation = dict(status='PASS', scope='Raw-space isolation, not original critic chart reproduction',
    original_function_reason=reason, active_groups=int(active.sum()), missing_raw_groups=[99],
    original_weights_sum=float(weights.sum()), retained_weights_sum=float((weights * active).sum()),
    weights_renormalized=False, active_set_frozen_before_odd_data=True, witness=witness,
    limitation='Exact original critic chart/group IDs require a captured runtime chart; raw mode99 is not identified with a critic group')
np.savez_compressed(ROOT / 'isolation-arrays.npz', target_centers=target, real_fifo=real.numpy(),
    ema_outputs=ema.numpy(), even_groups=even_groups.numpy(), ema_groups=ema_groups.numpy(),
    centers=centers.numpy(), scales=scales.numpy(), even_means=even_means.numpy(),
    ema_means=ema_means.numpy(), ema_counts=ema_counts.numpy(), weights=weights.numpy(),
    active=active.numpy(), directions=directions.numpy())
(ROOT / 'raw-space-isolation.json').write_text(json.dumps(isolation, indent=2, sort_keys=True) + '\n')
assert torch.equal(cpu_rng_before, torch.get_rng_state())
assert not torch.cuda.is_initialized()
assert verify() == before
for path, expected in inputs.items():
    assert sha(path) == expected, path
receipt = dict(status='PASS_DIAGNOSTIC_ONLY', actual_quality_status='FAIL', original_verdict=verdict,
    original_quality_bar=bar, checkpoints=rows, final_kernel_bounds=bounds,
    raw_space_isolation=isolation, source_integrity_before=before, source_integrity_after=verify(),
    retained_input_hashes=inputs, cuda_initialized=False, training_updates=0, model_constructions=0,
    RNG_draws=0, CPU_RNG_unchanged=True, host_process_signals=0,
    limitations=['Noise-only expected HQ is a deterministic uniform-row expectation; it is not a re-evaluation of the original CUDA draw.',
                 'Per-row motion across checkpoints includes birth/death replacements; row IDs do not certify particle lineage.',
                 'Raw-space partial witness verifies bounded-score algebra, not original ephemeral critic-chart reconstruction.'],
    artifacts={p.name: sha(p) for p in [Path(__file__), ROOT / 'isolation-arrays.npz', ROOT / 'raw-space-isolation.json']})
(ROOT / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(dict(event='raw_partial_witness', original_veto=reason, active_groups=99,
                      lower_bound=witness['lower_bound'], all_odd_observations=witness['observations'])), flush=True)
print(json.dumps(dict(event='complete', diagnostic='PASS', quality='FAIL', source='VALID',
                      CUDA=False, RNG_draws=0, training_updates=0, kernel_bounds=bounds)), flush=True)
