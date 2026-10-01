"""Independent NumPy score audit of an explicitly non-authoritative raw chart.

No checkpoint loads, model calls, package imports, RNG draws or GPU operations.
"""
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
OTHER = HERE.parent / 'moving-rotated-recovery'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


inputs = {str(p): sha(p) for p in (OTHER / 'isolation-arrays.npz', OTHER / 'raw-space-isolation.json')}
expected = json.loads((OTHER / 'raw-space-isolation.json').read_text())
with np.load(OTHER / 'isolation-arrays.npz', allow_pickle=False) as arrays:
    data = {key: arrays[key].copy() for key in arrays.files}
odd = data['real_fifo'][1::2]
distances = ((odd[:, None] - data['target_centers'][None]) ** 2).sum(2)
groups = distances.argmin(1)
active = data['ema_counts'] > 0
assert np.array_equal(active, data['active'])
assert np.all(data['directions'][~active] == 0)
assert np.all(np.linalg.norm(data['directions'], axis=1) <= 1 + 1e-14)
weights = data['weights'] * active
assert math.isclose(data['weights'].sum(), 1., abs_tol=1e-14)
assert weights.sum() < 1
radius = math.sqrt(data['centers'].shape[1] / .05)
z = (odd - data['centers'][groups]) / data['scales'][groups, None]
psi = z * np.minimum(1., radius / np.maximum(np.linalg.norm(z, axis=1), 1e-30))[:, None]
values = (data['directions'][groups] * (psi - data['ema_means'][groups])).sum(1)
assert np.all(values[~active[groups]] == 0)
assert np.all(np.abs(values) <= 2 * radius + 1e-10)
assert len(values) == len(odd) == 10000
alpha = .05 / (3 * 128 + 3)
t = math.log(2 / alpha)
mean = float(values.mean())
variance = float(values.var(ddof=1))
variance_penalty = math.sqrt(2 * variance * t / len(values))
range_penalty = (7 / 3) * (4 * radius) * t / (len(values) - 1)
lcb = mean - variance_penalty - range_penalty
for key, actual in [('mean', mean), ('variance_ddof1', variance), ('lower_bound', lcb),
                    ('variance_penalty', variance_penalty), ('range_penalty', range_penalty)]:
    assert math.isclose(actual, expected['witness'][key], rel_tol=1e-12, abs_tol=1e-12), key
assert lcb > 0
assert all(sha(Path(path)) == digest for path, digest in inputs.items())
receipt = dict(status='PASS_CPU_ALGEBRA_ONLY', scope=expected['scope'],
    authoritative=False, original_critic_chart_reproduced=False,
    observations=len(values), inactive_zero_observations=int((~active[groups]).sum()),
    active_groups=int(active.sum()), missing_raw_groups=np.flatnonzero(~active).tolist(),
    original_even_mass=float(data['weights'].sum()), retained_even_mass=float(weights.sum()),
    weights_renormalized=False, mean=mean, variance_ddof1=variance, lower_bound=lcb,
    scalar_min=float(values.min()), scalar_max=float(values.max()),
    radius=radius, known_range=4 * radius, alpha=alpha, multiplicity=387,
    variance_penalty=variance_penalty, range_penalty=range_penalty,
    model_calls=0, scorer_calls=0, checkpoint_loads=0, training_updates=0,
    gpu_operations=0, rng_draws=0, input_sha256=inputs,
    verdict='APPROVE conditional bounded-score algebra and source-only causal test',
    requirements=[
        'Freeze active groups using even-real and pre-action EMA support before odd data or count actions.',
        'Set both inactive weights and inactive directions to zero without renormalization.',
        'Retain every odd row, including zero-direction observations, and unchanged alpha and 4r range.',
        'Exclude initially inactive groups from pair and packet eligibility before division by EMA count.',
        'After count actions, veto if any initially active group is empty; never expand the frozen mask.',
        'Retain complete raw-output axes up to8, shared action budget and95percent serving coherence.',
        'Activate only feature backend after existing typed R1 fires; default and no-fire code remain equivalent.'],
    limitations=[
        'Raw-space isolation uses evaluator-only target centers; production must use its own frozen critic chart.',
        'Original ephemeral critic chart is absent from saved checkpoint, so raw group99 is not a critic group assertion.',
        'Bounded conditional iid algebra is not a new population guarantee for an adaptive trained chart/FIFO.',
        'Positive witness does not guarantee legal packets, serving coherence or an original quality-gate pass.'])
with (HERE / 'PEER-REVIEW.json').open('x') as handle:
    handle.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: receipt[key] for key in ('status', 'active_groups', 'observations',
    'inactive_zero_observations', 'retained_even_mass', 'lower_bound', 'gpu_operations')}, sort_keys=True))
