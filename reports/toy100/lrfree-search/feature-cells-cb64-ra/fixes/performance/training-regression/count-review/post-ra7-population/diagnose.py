"""Reserved population-flux design: saved reads and a tiny fixed CPU example.

No generator evaluation, model gradient, optimizer update, random sample or seed.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import torch

torch.set_num_threads(1)
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'validation-cb64-ra7/learned/training/toy/CB64-RA7'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE / 'observations.json').exists()
rng = torch.get_rng_state().clone()
inputs = {}
for name, expected in {
    'pkg-CB64-RA7/particlegan/continuous.py': '071ebb1b8b85c2ff4166557499dc2b71b5ec41e62aeb9d6dfc34bedc10160195',
    'pkg-CB64-RA7/particlegan/training.py': '8961dc8596230bc0ff5f153606694aa21745e77f5577a0fc93664d9107d5b005',
    'pkg-CB64-RA7/particlegan/feature_cells.py': 'eee5469b420d9c750d9ad015172af58052e8aab161d93d4f16fc323be12f9245',
    'quality/ra7/READY.json': 'f300089ce4fece3a812b890d4060e0567e833dddd236e5a955993f65ef15e1dd',
    'validation-cb64-ra7/source-freeze.json': '1724fc603970338b0ae94d58d6055a734e3cd424fcd514a53bb1ca9297426b3d',
    'performance/training-regression/count-review/ra7-prospective/FINAL-FROZEN.json': '36df2b0c6c754b4eca0daf0fa09eada081b649eda0f3d29fa4723753a2a3dda2',
}.items():
    path = ROOT / name
    assert sha(path) == expected, name
    inputs[str(path)] = expected

def completion(pair, n):
    pair = pair.double()
    finite = torch.isfinite(pair)
    m = int(finite.sum())
    total = float(pair[finite].sum())
    # This is an identification bound, not a confidence interval. Every
    # discarded row could have any cosine in [-1, 1]. No missing-at-random law.
    return dict(observed_rows=m, absent_rows=n-m,
                survivor_mean=(total/m if m else None),
                full_population_identification_interval=[(total-(n-m))/n,
                                                         (total+(n-m))/n])

saved = []
for step, expected in {
    500: '132f9351b3c5857b0e434c7d9b294de12084bc09f8012e01c53159e9e78f7612',
    1000: '5ca98bdfbe400930d42f52306e6fa0eac330b6d87a8f6ae5254594daa5c7edb2',
    2000: '94ee1c70def3f8256140ec23a2f7c9ef997699ca1a63815ce9a51bd1e85db3c2',
}.items():
    path = RUN / f'checkpoint-{step:04d}.pt'
    assert sha(path) == expected
    inputs[str(path)] = expected
    trainer = torch.load(path, map_location='cpu', weights_only=False)['trainer']
    table, generator = trainer['lr_settle'][0][1], trainer['lr_settle'][0][0]
    n = table['rows']
    pair_bounds = {}
    participation = {}
    for name in ('r_b', 'r_2b'):
        pairs = table[name]
        pair_bounds[name] = [completion(pair, n) for pair in pairs]
        participation[name] = int((torch.isfinite(torch.stack(pairs)).sum(0) >= 2).sum()) if pairs else 0
    z, ema_z = trainer['models']['prior']['z'], trainer['models']['ema_prior']['z']
    latent_pair_rms = float((z.double()-ema_z.double()).square().mean().sqrt())
    saved.append(dict(step=step, required_rows=n-math.floor(.05*n),
        current_window_participation=participation, pair_completion_bounds=pair_bounds,
        table_s=table['s'], table_b=table['b'], table_stamp=table['last_decisive'],
        population_active=table['population_active'], last_population=table['last_population'],
        generator_s=generator['s'], generator_stamp=generator['last_decisive'],
        generator_stationary_decisions=generator['counts']['stationary'],
        cumulative_moves=trainer['birth_death']['counters']['moves'],
        paired_latent_rms_descriptive_only=latent_pair_rms))
    assert sha(path) == expected

# Extract just the immutable base tester. This invokes no model/optimizer and
# keeps its real row-incarnation rebase semantics in the fixed demonstration.
source = (ROOT / 'pkg-CB64-RA7/particlegan/continuous.py').read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SettleTest')
ns = dict(torch=torch, math=math, deepcopy=deepcopy)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<immutable-SettleTest>', 'exec'), ns)
tester = ns['SettleTest']()
n, d, moves, blocks = 64, 2, 3, 20
x = torch.stack(((torch.arange(n, dtype=torch.float64)-32)/32,
                 ((torch.arange(n) % 2)*2-1).double()/2), dim=1)
tester.rows = n
tester.begin([x])
def moments(a):
    y = a.tanh()
    return torch.cat((y.mean(0), y.square().mean(0)))
flux, reaction_flux, accounting_error = [], [], []
for k in range(blocks):
    before = moments(x)
    x[:, 0] += (1 if k % 2 == 0 else -1)/32
    after_coordinate = moments(x)
    tester.observe([x], ratio=1, step=k+1)
    rows = (torch.arange(moves)+moves*k) % n
    # A three-row permutation preserves the population but replaces the
    # corresponding incarnations. New rows receive no ancestor pair evidence.
    x[rows] = x[rows.roll(1)].clone()
    tester.rebase([x], rows)
    after_reaction = moments(x)
    flux.append(after_coordinate-before)
    reaction_flux.append(after_reaction-after_coordinate)
    accounting_error.append(float((after_reaction-before-flux[-1]-reaction_flux[-1]).abs().max()))
cosines = [float((a@b)/(a.norm()*b.norm())) for a,b in zip(flux[::2], flux[1::2])]
participating = int((torch.isfinite(torch.stack(tester.r_b)).sum(0) >= 2).sum())
required = n-math.floor(.05*n)
assert participating < required
assert min(cosines) < -.999999999 and max(cosines) < -.999999999
assert max(float(v.abs().max()) for v in reaction_flux) < 1e-14
assert max(accounting_error) < 1e-14
synthetic = dict(rows=n, dimension=d, replaced_rows_per_block=moves, blocks=blocks,
    required_current_row_participants=required, current_row_participants=participating,
    all_population_flux_pairs=len(cosines), all_population_flux_cosines=cosines,
    maximum_reaction_flux=float(torch.stack(reaction_flux).abs().max()),
    maximum_flux_accounting_error=max(accounting_error),
    note='Every coordinate moves; deterministic permutations invalidate rows but preserve all population observables.')

# First moments can miss mode spreading even without replacement. Finite
# first/second moments are still not a full-distribution stationarity theorem.
a = torch.tensor([[-1.],[1.]], dtype=torch.float64)
b = 2*a
assert float(a.mean()) == float(b.mean())
cancellation = dict(first_moments=[float(a.mean()),float(b.mean())],
                    bounded_second_moments=[float(a.tanh().square().mean()),float(b.tanh().square().mean())])

# Tiny exact null check of a proposed bounded conditional-mean betting law.
# No random draws. For X in [-1,1], E[X|past]>=0, predictable lambda in [0,1],
# E[1-lambda*X|past]<=1. The nonnegative product/mixture is a supermartingale.
lambdas = tuple(Fraction(k,4) for k in (1,2,3,4))
alpha, threshold = Fraction(1,80), Fraction(80)
products = [Fraction(1)]*4
first_negative_crossing = None
for k in range(1,41):
    products = [v*(1+l) for v,l in zip(products,lambdas)]
    if sum(products)/4 >= threshold:
        first_negative_crossing=k
        break
assert first_negative_crossing is not None
hits, final_mean = 0, Fraction(0)
depth = 12
for values in itertools.product((-1,1), repeat=depth):
    products, hit = [Fraction(1)]*4, False
    for value in values:
        products = [v*(1-l*value) for v,l in zip(products,lambdas)]
        hit |= sum(products)/4 >= threshold
    hits += hit
    final_mean += sum(products)/4
null_probability = Fraction(hits,2**depth)
assert null_probability <= alpha and final_mean/(2**depth) == 1
betting = dict(lambdas=[str(l) for l in lambdas], alpha=str(alpha), threshold=int(threshold),
    fixed_all_negative_crossing_pairs=first_negative_crossing,
    fair_null_sequences=2**depth, null_anytime_crossings=hits,
    null_anytime_probability=str(null_probability), exact_null_final_expectation=1,
    null='Conditional population-flux cosine mean >=0, not full-density equilibrium.',
    limits='Epoch/scale allocation and certificate expiry remain necessary; this is not production implementation.')

assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
for path, expected in inputs.items():
    assert sha(path) == expected
out = dict(status='VALID',scope='Reserved alternative, not a candidate acceptance or production-law implementation',
    saved=saved, synthetic=synthetic, mean_cancellation_counterexample=cancellation,
    bounded_betting_example=betting, input_sha256=inputs,
    script_sha256=sha(__file__), frozen_sources_and_inputs_unchanged=True,
    global_rng_unchanged=True, cuda_initialized=False, model_forward_calls=0,
    model_gradient_calls=0, optimizer_calls=0, new_seeds=0, new_quality_emissions=0)
(HERE/'observations.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
for v in saved:
    print(json.dumps({k:v[k] for k in ('step','required_rows','current_window_participation','table_s','table_b','generator_s','generator_stamp')}),flush=True)
print(json.dumps(dict(synthetic=synthetic,betting=betting)),flush=True)
