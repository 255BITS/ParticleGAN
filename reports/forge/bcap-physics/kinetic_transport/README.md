# Kinetic transport: same-batch quantile forces

**The transport signal repairs rare allocation at the endpoint, but solves no
additional sustained task.** Both arms pass **2/5 mutually runnable tasks**.
The candidate records two PASS, three numerical FAIL and one explicit BLOCKED;
the matched winner records three PASS and three FAIL including its separate
two-pole fixture. All 11 runnable jobs complete once for **541.166 paid worker
seconds**, with zero retries or remaining reservations.

This is a source-bound mechanism diagnostic, with no ordinary Tier 2,
calibration, or public-default promotion credit. The parent maintains the single
cross-track comparison; the task table below is a readout, not another leaderboard.

## Completed results

| Unchanged task | Matched winner | Transport candidate | Measurement and retained failure |
| --- | --- | --- | --- |
| Two-pole | PASS, 17/24 checks | BLOCKED | Frozen public-components fixture does not consume the new sample-space signal; no substitution |
| Gaussian smoke | PASS, 3/24 independently confirmed states | PASS, 13/24 confirmed states | First confirmed acquisition 375→84; endpoint KS .071842→.026218; smoke requires any confirmed pass, not terminal quality |
| Gaussian stability | FAIL: 2/72 stationary, 0/24 shifted hold | FAIL: 33/72 stationary, 12/24 shifted hold | Both miss deadline reacquisition and complete retention; final KS .320623→.090741, candidate mean error .238351 sigma>.2 |
| Unequal mass | FAIL: 0/24, suffix 0 | FAIL: 11/24, suffix 2 | Minimum mass ratio .207520→.961538; covariance 3.691653→.773642; candidate last-five eigen floors fail at 1050/1100 (.084535/.096804<.15) |
| Unequal width | FAIL: 0/24, suffix 0 | FAIL: 0/24, suffix 0 | Mass TV .291504→.022705 and SW1 .295865→.041591 improve; full covariance 6.287559→7.038571 still fails≤.85 |
| Two broad, regression guardrail | PASS: 22/24, suffix 22 | PASS: 24/24, suffix 24 | Covariance .385581→.244338 and SW1 .134590→.040533; full gate retained |

[Exact final metrics and temporal failures](results.json),
[provenance and matched-condition proof](provenance.json),
[saved-state diagnostics](saved-state-diagnostics.json), and
[scorer controls](scorer-controls.json) retain the complete numeric readout.
On the five shared executable cells, PASS count remains 2/5 for both arms.
The unsupported sixth candidate cell cannot be counted as a trained failure or
silently removed from the declared common-core scope.

### Mass moves; density stability remains unsolved

The deterministic saved-center census changes unequal mass from
`142/94/19/1` to **`144/73/34/5`**. Five rare centers predict mass `.019531`,
close to target `.02`; the candidate's served endpoint supplies 96/4096 draws
(mass `.023438`), versus control 17/4096 (`.004150`). Thus this is a measured
redistribution through the trained map/prior, without reweighting or renewal.
The preregistered minimum-ratio≥.5 prediction is observed. The stronger claim
of sustained redistribution and correct local density fails: 1050 and 1100
violate the minimum variance gate, breaking the required five-check suffix.

Unequal width changes its center counts from `95/13/108/40` to
**`62/63/65/66`**, nearly the intended equal allocation. Yet narrow components'
full covariance errors are **18.2922 and 9.4500**, versus core-only .3889 and
.1919. Global HQ .974609 and full core-average covariance .248193 pass their
respective descriptive bounds while the retained full average 7.038571 fails.
Better occupancy does not eliminate boundary spill. The full-covariance≤.85
prediction is falsified. These are measured endpoint decompositions, not proof
that a particular quantile pairing caused the stray outputs.

Gaussian improves acquisition and the frequency of later passes, but continues
to fluctuate. The candidate fails 39/72 stationary checks and 12/24 postdeadline
shift checks; all 48 frozen no-update shift controls fail. Its endpoint width
ratio 1.002610 is accurate while KS .090741 and mean error .238351 still fail.
This distinguishes improved response from a continuous learner.

Posthoc reconstruction of all 1200 unequal-mass training batches matches the
checkpointed data RNG exactly. The rarest nearest component appears 2.58 times
per batch on average and is absent in 85/1200 batches (7.083%). This agrees with
the predeclared finite-batch competing explanation. It is an analysis of the
fixed seen batches; nearest-component labels are never supplied to training.
The experiment does not establish that rare-batch absence causes the failed
variance checks, or that transport alone explains improved Gaussian retention.

### Recommendation and transfer limits

**Stop the exact weight 1, 32-direction candidate as a global repair.** Preserve
its useful allocation result and the opt-in capability, while retaining the
winning control and ordinary qualification history. A useful next read-only
question is whether saved transport/adversarial gradients reinforce shared
center motion while suppressing rare local covariance, versus intermittent
boundary spill under normalized updates. Inspect those saved fields before
preregistering any successor; this round launches no tuning, extra continuation,
new seed, sampler or promotion.

The automatic Forge study decision remains **incomplete** because the original
two-pole host cannot execute the candidate, even though its endpoint mass
prediction is observed. This is a declared compatibility limit, not an incomplete
worker. All runnable tasks and their full gates are finished. Ordinary required
tiers remain unqualified. No image was trained in this subset; pixel geometry,
high-dimensional finite projections, conditional identity and native100 do not
inherit these vector gains. The passing broad-vector guardrail alone cannot
establish global applicability.

## Hypothesis and density equations

The learned prior is
`rho_z(z) = (1/M) sum_i N(z; a_i, sigma^2 I)` with fixed equal weights and fixed
local noise. Moving locations conserve each atom's mass. For a smooth time
interpolation of locations, the prior flux is exactly
`j_z = (1/M) sum_i N(z; a_i, sigma^2 I) * dot(a_i)`, so
`partial_t rho_z + div(j_z) = 0`, with `v_z = j_z/rho_z`.
No Brownian diffusion or birth/death source term is introduced. If the map is locally
invertible, its pushforward obeys a continuity equation
`partial_t rho_x + div(rho_x v_x) = 0`, with
`v_x(G(z)) = partial_t G(z) + J_G(z) v_z(z)`.
For a noninjective generator, the flux is the conditional average of these
velocities over preimages; a single globally invertible density formula is not
assumed. Changing the map also changes the local covariance, approximately
`sigma^2 J_G J_G^T`. Thus more latent centers allocated to a rare mode can repair
mass without repairing contracted local clouds, and moving a shared network can
simultaneously push unrelated centers across mode boundaries.

The saved winner's unequal-mass census is `142/94/19/1`; its one rare center
predicts mass `.00391`, close to served `.00415` versus target `.02`.
Its minimum local variance ratio `.01194` also predicts the failing served
`.00907`. Unequal-width has `95/13/108/40` centers, underallocating the second
mode. Stray samples account for 92.6%/97.5% of selected components' full
variance, so a core covariance improvement cannot replace the full gate.
These are original source-bound observations in
[failure-state-analysis.json](../../bcap-tier2-search/failure-state-analysis.json)
and [failure analysis](../../bcap-tier2-search/FAILURE_ANALYSIS.md), not new controls.

Hypothesis: local critic fields and normalized shared-map response can preserve
wrong allocations; an explicit equal-mass transport force from the training law
can redistribute outputs across projected quantile boundaries. This is a
physical analogy guiding a test, not a theorem about DualNorm GAN dynamics.

## One global candidate and explicit trainer delta

`kinetic_transport_sliced_v1` adds

`L_G = L_non_saturating + lambda * L_transport`,

`L_transport = (1/(K n s_R^2)) sum_(k,j)
  [ sort_j(u_k^T x_fake) - sort_j(u_k^T x_real) ]^2`.

Here `lambda=1`, `K=32`, and `s_R^2` is the detached mean centered coordinate
variance of the same G-phase real batch (machine epsilon floor). For scalar
outputs, one direction gives exact empirical quadratic Wasserstein distance.
For 2D outputs, fixed angles `k*pi/32` cover the unoriented circle. A fixed
normalized trigonometric frame generalizes the public function to higher
dimensions, without claiming rotation invariance or full OT fidelity there.
Sorting supplies a piecewise gradient; away from ties each fake point receives
`grad_x L = 2/(K n s_R^2) sum_k u_k [u_k^T x - matched_quantile_k]`.
The network and sampled-prior forces are its pullbacks through `J_theta G^T`
and `J_z G^T`. Repeated sampled rows accumulate forces before the unchanged
row-normalized response. DualNorm's polar/row normalization means these steps
are not the continuous Wasserstein gradient flow itself. Ties are a
nondifferentiable boundary.
All real tensors and the scale are detached. No held-out samples, component
centers, target labels, masses, evaluator projections, or thresholds enter training.

The signal uses the already consumed G fake batch and G-phase real tensor.
These frozen Forge vector/Gaussian adapters supply the same per-update target
batch to D and G; this historical batch law is preserved exactly. Public
GANTrainer also accepts a separate `generator_real`, but this study does not
change the adapters to enable it.
It adds no random draws or persistent state; all original constructor, critic
data, G data, latent, MoG kernel, penalty, model and evaluation streams retain
their checkpoint contracts. Equal atom weights, task sigma, sampling law,
architecture, batch sequence, update budget and scoring cadence stay frozen.
Only the reusable trainer objective changes. G/prior receive the auxiliary
gradient through the public trainer; D's objective is unchanged. The public
`Recipe.kinetic_transport_loss` also exposes the signal to callers that own
their batches. Existing frozen public-components hosts do not call it and
therefore explicitly refuse a nonzero transport recipe.

Everything else is the saved winner: non-saturating loss, full DualNorm,
momentum 0, smoothing `.001`, per-offset convolutions, G/E `.012`, D `.018`,
prior `.030`, constant floors 1, BCAP coefficient 1/cap 1 every update, no input
or output training noise, prior regularizer, or EMA. Forge resolves the full
saved configuration through its versioned API; `get_recipe('bcap')` alone is
not the provenance contract. Default fields 0/32 preserve old checkpoint and
mechanism projections. This candidate changes a training signal, not rates,
sampling or target-task conditions.

## Literature and competing explanations

This is an application of existing sliced transport methods, not a newly
invented optimal-transport algorithm. [Deshpande et al., CVPR 2018](https://arxiv.org/abs/1803.11188)
study generative learning from one-dimensional projected distribution matching.
Our finite deterministic directions and normalized auxiliary mixture differ
from their random-projection training law. [Feydy et al., AISTATS 2019](https://proceedings.mlr.press/v89/feydy19a.html)
study debiased entropic transport divergences. We use sorted 1D transport and
do not import Sinkhorn guarantees or claim that our empirical loss is unbiased.
[Lu, Lu and Nolen, 2019](https://arxiv.org/abs/1905.09863) add a nonlocal
birth/death term to Langevin dynamics. Our atom masses and identities stay fixed;
there is no birth/death source term or clone/renewal move.

Repository history already tests [prior/D pacing](../../dualnorm-pacing-v2/README.md):
twenty whole pacing configurations did not improve the baseline required count;
positive network momentum regressed. Archived [paired birth/death](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/paired-bd-graft/README.md)
improved mass TV but failed the original native gate; a later
[sigdata paired graft](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/sigdata-bdpair/README.md)
passed 2/3 noisy native tasks, whereas row-EM renewal reported 3/3 noisy and
0/3 clean. Their prior, initialization, noise and serving laws are separate.
[Gaussian saved-state diagnosis](../../gaussian1d-diagnosis/README.md) establishes
representational capacity, not a training pass. Newer [prior magnitude #319](https://github.com/255BITS/ParticleGAN/pull/319),
[network magnitude #320](https://github.com/255BITS/ParticleGAN/pull/320), and
[combined magnitude #321](https://github.com/255BITS/ParticleGAN/pull/321) improve
some retention but fail continuous learning and introduce ring regressions.
This experiment repeats none of their caps, timing changes or sampling repairs.

Competing explanation: finite minibatch transport is noisy for rare events
and can interpolate between true clusters. With batch 128 and rare mass `.02`,
about `.98^128 = .0753` of G-phase real batches contain no rare observation.
Finite projections can miss dependence; the normalized optimizer can amplify
small empirical mismatches even at a good fit. Shared-map covariance spill or
local contraction may therefore worsen while aggregate mass improves. A new
mass signal cannot guarantee quiet dynamics or true component covariance.

## Frozen question, prediction, falsifier and costs

Unchanged task definitions: `two_pole`, `gaussian1d_smoke`, its own dependent
`gaussian1d_stability`, `vector_unequal_mass`, `vector_unequal_width`, and passing
`vector_two_broad`. The candidate's two-pole cell is explicitly **BLOCKED**:
the original stored-critic/zero-coordinate fixture has no sample-space loss
consumer. Its control still runs. No substitute fixture or ignored mechanism
earns candidate credit. The broad vector is the executed passing guardrail.
Gaussian stability can run only if that recipe's own full smoke passes.

All vectors run 1200 updates, 24 scored observations, 4096 clean/live draws and
the original five terminal passing checks. Gates retain normalized SW1≤.18,
mass TV≤.15, HQ≥.85, full component covariance error≤.85 and minimum eigenvalue
ratio≥.15; unequal mass additionally requires minimum mass ratio≥.25.
Gaussian retains the full acquisition/confirmation, 72 stationary checks,
deadline reacquisition and 24 postdeadline hold checks.

Before training, predict rare minimum mass ratio≥.5 (at least `.01` observed
mass in the rarest mode), versus original `.2075`; ratio<.5 falsifies that
signature. Additionally predict unequal-width covariance error≤.85 and retain
the full two-broad PASS. These signatures are diagnostic expectations; the
complete sustained numerical task gates are authoritative. Even an observed
mass prediction cannot establish the wider redistribution-and-fidelity claim.

One candidate plus one matched winner, one scientific round, seed 0, no tuning,
extra continuation, seeds or scientific retries. Full maximum reservation is
6420 seconds per recipe, 12840 total, below the 14400 track allowance. Actual
candidate reservations omit the unsupported 300-second two-pole cell. Both
studies share campaign `kinetic_transport_v1`; the control's historical-parent
study comparison admits the existing winner but launches no parent run.
Independent runnable peers finish even if Gaussian smoke fails. Any failed or
incomplete outcome terminates this revision for a final readout.

## Evidence, validation and reproduction

Use the project Python 3.12 and shared queue; the parent owns the only drain.

```sh
PYTHONPATH=. .venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue plan kinetic_transport_sliced_v1 --study kinetic_transport_candidate_v1 --show-boundaries
PYTHONPATH=. .venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue enqueue kinetic_transport_sliced_v1 --study kinetic_transport_candidate_v1
PYTHONPATH=. .venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue enqueue bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36 --study kinetic_transport_control_v1
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue/kinetic_transport_v1/progress.jsonl
```

Executed scientific commit: `3fb7d4f98190589f399a8328fa11a19c591d9cb3`;
shared source digest: `d6fa25a96d87ad1b3eb7a43619088c564e655500869f9b97073c182282d330f7`.
Candidate revision: `24bacd26b777b7bb90214c0254f92861d20b93bd8c2f83fce4ed59ee76e49997`;
matched winner revision: `c140b8522f30cae109998c05dae7fc2b9ccefeb320e4151a6308098af1f8a1a7`.
Original archived winner revision remains
`dfe88a2ee15fb9d83ffdc5c8a25d698686b73d4e7c63b6b0e35efb0d64e94359`,
under original source `2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`.
Its passes remain original-cohort evidence. The new matched control reproduces
its selected numeric failures and vector allocations; it is not silently given
archived qualification credit.

Both new arms use Python 3.12.13, Torch 2.14.0+cu130, NumPy 2.5.2 and RTX A6000,
with deterministic execution, one Torch thread and TF32 disabled. All five
matched tasks have equal initial tensors/prior bindings and final named RNG
states; Gaussian recorded data digests match, and all three vectors additionally
reconstruct exact target-batch digests from the unchanged data stream. Recipe
comparison finds only effective transport weight 0→1; default fields are omitted
from archived Recipe projections and resolved explicitly through public defaults.
Complete consumed states, source-frozen requests, logs and samples stay in the
shared artifact queue. Saved-center census and batch replay are separately
labelled CPU diagnostics; they add no model sampling or optimizer updates.

Actual new outer updates: candidate 9600, control 9680, 19280 total. Paid cost is
248.327309 candidate plus 292.838209 control = 541.165518 worker seconds.
Declared full reservation ceiling 12840; executed jobs reserve 12540 in total
(the blocked 300-second cell consumes none), remaining reservation 0, scientific
retries 0. Shared GPU contention and startup are included in this accounting;
these costs are not speed-comparison evidence. Focused software tests take
1.34+5.82 seconds, with an earlier 1.41-second fixture failure repaired;
software/media/scorer work adds no paid scientific worker or training diagnostic.
Even conservatively adding these checks leaves the 14400 track ceiling intact.

[138 focused software checks](software-verification.json) and Forge validation
pass. Four independent target-law oracles pass the exact scalar/vector bounds;
four destructive point-collapse controls fail. Publication reproduces **432**
saved primary metric sets exactly and renders 11 nine-frame GIFs, selected by
fixed scoring indices without grade-based substitution. Scalar confirmations
and frozen shift comparisons retain their original certified grader checks.
An initial publication audit rejected omission of baseline default fields;
the repaired effective-value comparison preserves the original raw recipe
projection and confirms exactly weight 0→1. No training was repeated.

| Task | Matched control actual-training GIF | Candidate actual-training GIF |
| --- | --- | --- |
| Two-pole | [Goal metrics](control-two_pole.gif) | BLOCKED |
| Gaussian smoke | [Target and draws](control-gaussian1d_smoke.gif) | [Target and draws](candidate-gaussian1d_smoke.gif) |
| Gaussian stability | [Hold and shift](control-gaussian1d_stability.gif) | [Hold and shift](candidate-gaussian1d_stability.gif) |
| Unequal mass | [Density and gates](control-vector_unequal_mass.gif) | [Density and gates](candidate-vector_unequal_mass.gif) |
| Unequal width | [Density and gates](control-vector_unequal_width.gif) | [Density and gates](candidate-vector_unequal_width.gif) |
| Two broad | [Density and gates](control-vector_two_broad.gif) | [Density and gates](candidate-vector_two_broad.gif) |

[Media receipts](media.json) bind each GIF to saved observation bytes. Reproduce
publication without training or model sampling:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python reports/forge/bcap-physics/kinetic_transport/publish.py
```

Re-enqueue commands above describe scientific replay, not another authorized
round. Bulk stdout, per-update streams and checkpoints remain outside Git;
only final metrics, compact receipts, source and actual-training GIFs are committed.
PR: [#358](https://github.com/255BITS/ParticleGAN/pull/358), stacked on
[#356](https://github.com/255BITS/ParticleGAN/pull/356).
