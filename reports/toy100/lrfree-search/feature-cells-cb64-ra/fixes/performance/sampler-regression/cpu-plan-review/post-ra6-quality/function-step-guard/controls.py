"""Three deterministic mechanical controls, without another saved-state sweep."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from guard_design import LocalMotionChart, interpolate_parameters

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
output = HERE / 'controls.json'
if output.exists(): raise SystemExit('Preserve existing controls.')
source_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (Path(__file__), HERE / 'guard_design.py')}
rng = torch.get_rng_state().clone()


class Geometry:
    def __init__(self):
        self.valid_metric, self.rank, self.cells, self.duplicate_fraction, self.mass_groups = True, 2, 1, 0., 1
        self.real_representatives = torch.zeros(1, 2, dtype=torch.float64)
        self.cell_scale = torch.ones(1, dtype=torch.float64)
        self.reference_counts = torch.tensor([16])
    def transform(self, values): return values.double()
    def _assign_metric(self, values): return torch.zeros(len(values), dtype=torch.long), values.square().sum(1)
    def count_categories(self, values): return torch.zeros(len(values), dtype=torch.long)
    def support(self, values):
        return torch.zeros(len(values), dtype=torch.bool), torch.ones(len(values)), values.square().sum(1).sqrt()
    def _mass_topology(self): return torch.zeros(1, dtype=torch.long)


row = torch.arange(32, dtype=torch.float64)
real = torch.stack(((row - 15.5) / 16, .01 * row.remainder(3)), 1)
baseline = torch.zeros(8, 2, dtype=torch.float64)
chart = LocalMotionChart(Geometry(), real, baseline)
assert chart.active
wide = chart.inspect(baseline + torch.tensor([.1, 0.], dtype=torch.float64))
narrow = chart.inspect(baseline + torch.tensor([0., .1], dtype=torch.float64))
assert narrow['normalized_motion_q95'] > wide['normalized_motion_q95']
count = []
def candidate(fraction):
    count.append(fraction)
    return baseline + torch.tensor([0., .3 * fraction], dtype=torch.float64)
fraction, attempts = chart.choose(candidate)
assert fraction == .25 and count == [1., .5, .25]
count = []
def unsafe(fraction):
    count.append(fraction)
    return baseline + 100.
rejected, rejected_attempts = chart.choose(unsafe)
assert rejected == 0. and count == [1., .5, .25, .125]
before = dict(x=torch.tensor([1., -1., 1e-20], dtype=torch.float32))
after = dict(x=torch.nextafter(before['x'], torch.full_like(before['x'], float('inf'))))
assert torch.equal(interpolate_parameters(before, after, 0.)['x'], before['x'])
assert torch.equal(interpolate_parameters(before, after, 1.)['x'], after['x'])
try:
    LocalMotionChart(Geometry(), real, torch.zeros(129, 2))
except ValueError: pass
else: raise AssertionError('uncapped probes accepted')
assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
assert source_before == {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_before}
result = dict(status='PASS', controls=3, directional_covariance=dict(wide=wide, narrow=narrow),
    accepted_fraction=fraction, accepted_attempts=attempts,
    four_trial_rejection=dict(fraction=rejected, attempts=rejected_attempts),
    exact_parameter_endpoints=True, probe_cap_rejected=True, cpu_rng_unchanged=True,
    cuda_initialized=False, new_seeds=0, new_training_steps=0, production_changed=False,
    source_sha256=source_before)
output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(status='PASS', output=str(output))), flush=True)
