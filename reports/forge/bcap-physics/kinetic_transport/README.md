# Kinetic transport: same-batch quantile forces

This preregistered mechanism diagnostic tests one structural trainer candidate
against the exact winning BCAP recipe. **Results are pending.** It supplies no
ordinary Tier 2 qualification, calibration, or public-default promotion credit.
The parent campaign maintains the single current comparison leaderboard.

## Hypothesis and density equations

The learned prior is
`rho_z(z) = (1/M) sum_i N(z; a_i, sigma^2 I)` with fixed equal weights and fixed
local noise. Moving locations conserve each atom's mass. If the map is locally
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
Sorting supplies a piecewise gradient; ties are a nondifferentiable boundary.
All real tensors and the scale are detached. No held-out samples, component
centers, target labels, masses, evaluator projections, or thresholds enter training.

The signal uses the already consumed G fake batch and independent G real batch.
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
momentum0, smoothing `.001`, per-offset convolutions, G/E `.012`, D `.018`,
prior `.030`, constant floors1, BCAP coefficient1/cap1 every update, no input
or output training noise, prior regularizer, or EMA. Forge resolves the full
saved configuration through its versioned API; `get_recipe('bcap')` alone is
not the provenance contract. Default fields0/32 preserve old checkpoint and
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
and can interpolate between true clusters. With batch128 and rare mass `.02`,
about `.98^128 = .0753` of G real batches contain no rare observation.
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

All vectors run1200 updates, 24 scored observations,4096 clean/live draws and
the original five terminal passing checks. Gates retain normalized SW1≤.18,
mass TV≤.15, HQ≥.85, full component covariance error≤.85 and minimum eigenvalue
ratio≥.15; unequal mass additionally requires minimum mass ratio≥.25.
Gaussian retains the full acquisition/confirmation,72 stationary checks,
deadline reacquisition and24 postdeadline hold checks.

Before training, predict rare minimum mass ratio≥.5 (at least `.01` observed
mass in the rarest mode), versus original `.2075`; ratio<.5 falsifies that
signature. Additionally predict unequal-width covariance error≤.85 and retain
the full two-broad PASS. These signatures are diagnostic expectations; the
complete sustained numerical task gates are authoritative. Even an observed
mass prediction cannot establish the wider redistribution-and-fidelity claim.

One candidate plus one matched winner, one scientific round, seed0, no tuning,
extra continuation, seeds or scientific retries. Full maximum reservation is
6420 seconds per recipe,12840 total, below the14400 track allowance. Actual
candidate reservations omit the unsupported300-second two-pole cell. Both
studies share campaign `kinetic_transport_v1`; the control's historical-parent
study comparison admits the existing winner but launches no parent run.
Independent runnable peers finish even if Gaussian smoke fails. Any failed or
incomplete outcome terminates this revision for a final readout.

## Reproduction and pending outputs

Use the project Python3.12 and shared queue; the parent owns the only drain.

```sh
PYTHONPATH=. .venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue plan kinetic_transport_sliced_v1 --study kinetic_transport_candidate_v1 --show-boundaries
PYTHONPATH=. .venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue enqueue kinetic_transport_sliced_v1 --study kinetic_transport_candidate_v1
PYTHONPATH=. .venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue enqueue bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36 --study kinetic_transport_control_v1
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue/kinetic_transport_v1/progress.jsonl
```

Compact results, receipt bindings, scorer controls, matched-condition checks,
saved-state diagnostics and actual-training GIFs will be added after completion.
Bulk logs, per-update metrics, samples and checkpoints remain outside Git.
