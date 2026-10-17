"""Saved-state sample-size and mean-force spectra; no gradient/optimizer run."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import hashlib
import json
import math
from pathlib import Path
import torch

torch.set_num_threads(1)
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OUT = Path(__file__).resolve().parent
RUNS = dict(RA4=ROOT / 'validation-ra4/learned/training/toy/CB64-RA4',
    E22=Path('/ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/training/toy/E22'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = [ROOT / 'pkg-CB64-RA4/particlegan' / name for name in ('row_evidence.py', 'training.py')]
files += [base / f'checkpoint-{step:04d}.pt' for base in RUNS.values() for step in (1250, 2000)]
files += [base / 'config.json' for base in RUNS.values()]
before = {str(p): sha(p) for p in files}
rng = torch.get_rng_state().clone()
records = []
for run, base in RUNS.items():
    for step in (1250, 2000):
        checkpoint = torch.load(base / f'checkpoint-{step:04d}.pt', map_location='cpu', weights_only=False)
        ev = checkpoint['trainer']['row_evidence']
        d = ev['M'].shape[1]
        neff = ev['W'].double().square() / ev['S'].double().clamp_min(1e-30)
        mean = ev['M'].double() / ev['W'].double().clamp_min(1e-30)[:, None]
        singular = torch.linalg.svdvals(mean)
        eigen = singular.square()
        distribution = eigen / eigen.sum().clamp_min(1e-300)
        cumulative = distribution.cumsum(0)
        energy_rank = lambda q: int((cumulative < q).sum()) + 1
        source_epsilon = torch.finfo(ev['M'].dtype).eps
        numerical_rank = int((singular > source_epsilon * max(mean.shape) * singular.max()).sum())
        normalized_persistence = mean.square().sum(1) / (ev['Qs'].double()
            / ev['W'].double().clamp_min(1e-30)).clamp_min(1e-300)
        touched = ev['W'] > 0
        records.append(dict(run=run, step=step, population=len(neff), latent_dimension=d,
            mature_rows_original=int((neff >= 3*d).sum()), current_flags=int(ev['flag'].sum()),
            touched_rows=int(touched.sum()), neff_min=float(neff.min()), neff_median=float(neff.median()),
            neff_max=float(neff.max()),
            maturity_counts_for_hypothetical_fixed_dimension={str(k): int((neff >= 3*k).sum())
                for k in (1,2,4,8,16,32,33,128)},
            mean_force_spectrum=dict(scope='cross-row M/W spectrum, not within-row touch covariance',
                numerical_rank=numerical_rank, energy_rank90=energy_rank(.90), energy_rank95=energy_rank(.95),
                energy_rank99=energy_rank(.99), entropy_effective_rank=float((-(distribution*
                    distribution.clamp_min(1e-300).log()).sum()).exp()),
                first8_mean_energy=float(distribution[:8].sum()), first32_mean_energy=float(distribution[:32].sum())),
            mean_force_to_touch_energy_median=float(normalized_persistence[touched].median())
                if bool(touched.any()) else None,
            table_ema_rate=checkpoint['trainer']['lr_settle'][0][1]['s'] /
                (4*checkpoint['trainer']['lr_settle'][0][1]['b'])))

lam = 1/50
effective_cap = (2-lam)/lam
keep = 1-lam
finite_neff = lambda t: ((1-keep**t)/(1-keep))**2 / ((1-keep**(2*t))/(1-keep**2))
minimum_touches = {}
for k in (1,2,4,8,16,32):
    minimum_touches[str(k)] = next(t for t in range(1, 10000) if finite_neff(t) >= 3*k)
assert effective_cap == 99 and finite_neff(1000) < 99
assert all(r['mature_rows_original'] == 0 for r in records)
assert all(r['maturity_counts_for_hypothetical_fixed_dimension']['33'] == 0 for r in records)
assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
assert before == {str(p): sha(p) for p in files}
receipt = dict(status='VALID', diagnostic_status='COMPLETE_SAVED_STATE_ONLY',
    records=records, input_source_sha256=before,
    sample_size=dict(default_window=50, asymptotic_neff=effective_cap, original_dimension=128,
        original_required_neff=384, largest_finite_touch_maturable_integer_dimension=32,
        minimum_touches_at_existing_window=minimum_touches,
        expected_distinct_uniform_row_touches_over2000=2000*(1-(1-1/1024)**128),
        expected_linear_draws_per_row_over2000=2000*128/1024),
    identifiability=dict(projected_first_moment='recoverable for a specified fixed projection from M',
        projected_second_moment='unidentifiable from scalar Qs; history/covariance absent',
        within_row_intrinsic_gradient_rank='unidentifiable',
        cross_row_mean_rank='descriptive, cannot replace per-row null dimension',
        historical_projected_p_values='not reconstructed'),
    recommendation=dict(current_action='Keep RA5 RowEvidence unchanged until prospective quality establishes a remaining need.',
        possible_future_mechanism='A generic projection fixed independently of row gradient histories, with new projected accumulators and a declared projected null law.',
        excluded_shortcut='Do not change d to a fitted rank or reuse Qs as a projected squared norm.',
        state_requirement='Persist projection identity, projected M/Qs and maturity policy; old full-dimension state must not silently resume as projected evidence.',
        interpretation='A projected null test can detect force in its subspace; absence of discoveries cannot certify equilibrium in omitted directions.',
        null_limit='Original trace-only variance assumes isotropic components and independent touches. Reducing dimension alone does not establish either assumption; scaled median calibration is not an exact conditional null.'),
    new_gradients=0, optimizer_updates=0, new_seeds=0, cuda_initialized=False,
    global_rng_unchanged=True, quality_verdict=None)
assert not (OUT / 'receipt.json').exists() and not (OUT / 'FROZEN.json').exists()
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
lines=['# Saved RowEvidence dimension feasibility', '',
    'The current window50 caps effective sample size at99; d128 needs384. No saved RA4 or E22 row can mature under that law. Rank33 also cannot mature at finite touches; rank32 can.', '',
    '| Run/step | Median/max n_eff | Mature for fixed k8 / k16 / k32 | Mean-spectrum 95% rank |',
    '|---|---:|---:|---:|']
for r in records:
    maturity=r['maturity_counts_for_hypothetical_fixed_dimension']
    lines.append(f"| {r['run']}/{r['step']} | {r['neff_median']:.2f}/{r['neff_max']:.2f} | "
        f"{maturity['8']} / {maturity['16']} / {maturity['32']} | {r['mean_force_spectrum']['energy_rank95']} |")
lines += ['', 'These maturity counts only use saved W/S. They do not estimate projected flags or a corrected quality result.', '',
    'A fixed projection could reduce the sample requirement. However, scalar Qs stores total128-coordinate energy: projected variance and within-row gradient rank cannot be reconstructed. The spectrum of means across rows is descriptive and does not supply the within-row null dimension. Fitting a projection/rank to the same tested means would also introduce selection.', '',
    'A later candidate would need an independently fixed generic projection, new projected statistics and explicit checkpoint identity. It would preserve full-dimensional optimizer updates and test force only in that subspace. The current trace-variance null assumes isotropic components and independent touches; projection does not establish those assumptions. No projection or null-law change is justified as a passing correction by these saved states.', '',
    'Recommendation: keep RA5 evidence unchanged and await its prospective quality result. This receipt has no new gradients, optimizer updates, sampling, seeds or CUDA contexts. Frozen inputs were hashed before and after; global RNG is unchanged.']
(OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
frozen=dict(status='VALID', files={str(OUT / name):sha(OUT / name) for name in
    ('diagnose.py','receipt.json','REPORT.md')}, input_source_sha256=before)
(OUT / 'FROZEN.json').write_text(json.dumps(frozen, indent=2)+'\n')
print(json.dumps(dict(status='VALID', receipt=str(OUT/'receipt.json'), receipt_sha256=sha(OUT/'receipt.json'),
    frozen_sha256=sha(OUT/'FROZEN.json'), records=[dict(run=r['run'],step=r['step'],
        mature_k8=r['maturity_counts_for_hypothetical_fixed_dimension']['8'],
        mean_rank95=r['mean_force_spectrum']['energy_rank95']) for r in records]),indent=2))
