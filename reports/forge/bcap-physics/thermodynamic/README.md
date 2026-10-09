# Thermodynamic track: selective optimistic BCAP dynamics

Selective network optimism **fails as a repair**. Across the six unchanged
scoped tasks it records **2 PASS, 3 FAIL and 1 BLOCKED**, versus the matched
winner's **3 PASS and 3 FAIL**. It loses confirmed Gaussian acquisition and
severely regresses mode hold, while retaining both passing guardrails. All
11 executable jobs finish; the blocked stability continuation consumes no
training. This mechanism diagnostic confers no ordinary tier qualification,
calibration, promotion or public-default claim.


## Completed numerical comparison

[Final metrics](results.json), [verified provenance](provenance.json), the
[candidate readout](../../records/readout-f48985fb0a66208013724f4f.json) and
[control readout](../../records/readout-e00b36764951ef208e534a57.json) preserve
the exact contracts, source and verdicts. Results below use every original
numerical bound together; terminal values alone do not determine the grade.

| Unchanged task | Optimistic candidate | Matched winning control | Required interpretation |
| --- | --- | --- | --- |
| Two-pole | PASS; active mean .957914, median critic input gradient .168937; suffix 17 | PASS; mean .958502, gradient .952346; suffix 17 | Mean >= .3, gradient <= 1; five terminal full passes |
| Gaussian smoke | FAIL; 2/24 primary passes, **zero independently confirmed**; final KS .082485, mean error .203229 sigma, width ratio 1.013995 | PASS; 3/24 primary passes, confirmed at 375/875; final KS .071842, mean error .125898, width 1.063043 | Any full primary + independent same-state pass; all 1,000 updates complete. Endpoint need not pass |
| Gaussian stability | **BLOCKED, unmeasured:** own smoke failed; zero new updates/cost | FAIL; stationary 2/72, shifted hold 0/24, deadline reacquisition FAIL; final KS .320623, mean error .255238, width .662330 | Own passing smoke producer; all 72 stationary and 24 shift-hold checks, plus deadline suffix |
| Mode hold | FAIL; **3/8 modes, quality .129883**, full passes 0/24, suffix 0 | FAIL; endpoint 8 modes/quality .989258, full passes 10/24, suffix 3 | All 8 modes and quality >= .90, five terminal full passes |
| Two broad vector components | PASS; covariance error .240319, minimum eigen ratio .600889, SW .088421, mass TV .052490, quality .991943; suffix 18 | PASS; covariance .385581, eigen .400066, SW .134590, TV .062988, quality .988770; suffix 22 | Full covariance <= .85, eigen >= .15, SW <= .18, TV <= .15, quality >= .85; five terminal full passes |
| Conditional trajectory | FAIL; identity MSE .241836, full passes 0/24 | FAIL; identity MSE .239862, full passes 0/24 | Paired identity MSE <= .02 |

All measured sample-count/finite-state requirements remain satisfied. The
candidate's broad-vector endpoint covariance and distribution distance improve,
but its passing suffix shortens from 22 to 18. The unchanged full guardrail still
passes. The lower two-pole input-gradient median is not a distribution or game
stability result. Conditional identity does not improve; its competing
coverage objective remains unchanged.

Gaussian acquisition width variation narrows: candidate range .793465–2.433668
versus control .772801–3.191114; total variation 5.367664 versus 7.166739 across
24 primary checks. KS total variation is 1.365794 versus 1.526892. These are
acquisition stability proxies, not stationary retention or measurements of
antisymmetric game flow. Two isolated primary full passes cannot replace the
failed independent confirmations. The catastrophic mode-hold regression,
from 8 to 3 modes and quality .989258 to .129883, prevents accepting optimism even
if a coarse oscillation proxy improves.

The matched control reproduces the original Gaussian endpoint KS
`.3206228939608994`, the 2/72 stationary and 0/24 shift-hold outcomes, and mode
hold's three-check terminal suffix. This corroborates the source-bound recipe
reconstruction; archived passes are not added to this new diagnostic count.

**Forecast binding limitation:** the frozen studies name final metric `ks`,
while these Gaussian receipts expose `cdf_ks`. Consequently both automatic
forecast outcomes are `incomplete`; the candidate also has no stability result
because its smoke prerequisite failed. The immutable declarations and
forecast receipts are preserved. The control's measured `cdf_ks=.320623`
independently fails the actual task's .05 bound. This report does not label an
unmeasured candidate stability value as FAIL or silently repair preregistration
after training. Use the exact emitted metric name in future declarations.

## Accounting, verification and actual-training media

The two recipes reserved 12,840 seconds under the 14,400 ceiling. Measured paid
worker cost is **494.946963 seconds**: candidate 172.663206 and control 322.283757.
Remaining active reservations are zero. No scientific retry, seed-only run,
extra timing arm, parameter tuning or follow-on training occurred. Shared GPU
contention and startup costs are included; these costs are not speed evidence.

Scientific source commit
`1503419419e3581cc84657aef5efe7465ea17671`, digest
`5dd9416b67f955d731ef39f9d161a4bb403f0fe8466192c7ef4f20f999b1f1ef`, is identical
for both arms. Runtime is CPython 3.12.13, PyTorch 2.14.0, NumPy 2.5.2 and
SciPy 1.17.1 on the requested RTX A6000 CUDA cohort. Candidate revision is
`37de3dc839254b12f9c961e0b1b4dc17855bcc854688808f1985f8ea6a1e2e90`;
matched control revision is
`6eb2b2fbb09d85edefa1f070f45c3e19cd4b63b512152284384f5d3bfc0b3302`.
These current identities do not replace the archived winner's original revision.

All 11 original certificates and checkpoint/artifact hashes verify. The five
mutually executed tasks have equal initial-model proofs and final named stream
state hashes, with zero unintended RNG deviations. Final stream parity is a
state audit, not an independent bytewise consumed-batch digest. Two-pole retains
its explicit fixed native fixture; other task initialization policies remain
unchanged. Checkpoints preserve all consumed streams and normalized direction
history. Candidate cached network field counts are two-pole 4, Gaussian 12,
mode hold 16, broad-vector 12 and trajectory 6; control counts are zero. Prior
rows acquire no optimistic history. Only one last normalized field is retained:
a measured successive-field angle/amplification or antisymmetric-Jacobian
census is **unavailable**, so the cause of mode collapse is not established.

75 focused software checks pass, covering numerical formula/current rates,
ordinary-first-step and zero/missing gradient semantics, convolution layout,
prior row ownership, exact checkpoint replay, nonfinite/mismatched cache
rejection, field ownership, structural admission and Forge studies. Forge
validation and summary-only memory freshness checks pass. The scoped bilinear
algebra test confirms the stated eigenvalue/energy calculation and supplies no
neural-training qualification.

All 11 GIFs use certified actual observations, verified saved sample arrays where
retained, and fixed numerical bounds. Export adds zero updates or sampling;
[media receipts](media-index.json) bind every frame selection to its original
training evidence. The blocked continuation has no fabricated media.

| Task | Candidate GIF | Control GIF |
| --- | --- | --- |
| Two-pole | [Actual training](media/candidate/two_pole.gif) | [Actual training](media/control/two_pole.gif) |
| Gaussian smoke | [Actual training](media/candidate/gaussian1d_smoke.gif) | [Actual training](media/control/gaussian1d_smoke.gif) |
| Gaussian stability | BLOCKED | [Actual training](media/control/gaussian1d_stability.gif) |
| Mode hold | [Actual training](media/candidate/mode_hold.gif) | [Actual training](media/control/mode_hold.gif) |
| Broad components | [Actual training](media/candidate/vector_two_broad.gif) | [Actual training](media/control/vector_two_broad.gif) |
| Conditional trajectory | [Actual measurements](media/candidate/trajectory.gif) | [Actual measurements](media/control/trajectory.gif) |

## Recommendation

Stop this exact network-optimism coefficient 1 recipe. Retain the winner as a
matched experimental control, with its existing failures and blocked promotion.
The two passing guardrails rule out blanket numerical failure, while lost
Gaussian confirmation and mode collapse reject this proposed repair. They do
not falsify all circulation-based methods or prove that circulation caused the
baseline's failures. The nonlinear, alternating, selectively corrected field
and continuing prior motion differ from the scoped simultaneous bilinear model.

Before another timing proposal, inspect saved field-history/Jacobian evidence
for rotational dominance and noisy direction amplification, binding any probes
to their original source and sampling cohort. Such an investigation needs a
separate bounded declaration. No additional training or diagnostic model calls
were made. Correct the forecast metric namespace before
admitting the next study. No coefficient grid or automatic extragradient
continuation follows from this negative result.

## Local theory and its limits

Write the coupled descending game field as
`F(w)=(grad_D L_D, grad_G L_G, grad_z L_G)`; signs are absorbed into each
player's minimized loss. Locally `J=dF/dw=S+A`, with
`S=(J+J^T)/2`, `A=(J-J^T)/2`. The symmetric component supplies local
contraction/expansion; the antisymmetric component supplies circulation.
For the continuous field `dot(w)=-F(w)`, the residual energy
`R=||F||^2/2` obeys `dot(R)=-F^T S F`, because `F^T A F=0`.
This is a mathematical stability diagnostic, not physical heat or entropy.
There is no literal thermodynamic law for this GAN.

For a scoped bilinear rotation `F(w)=A w`, `A^T=-A`, ordinary simultaneous
Euler updates multiply energy in a frequency block by `1+(eta*omega)^2`.
Optimistic updates `w_(t+1)=w_t-2 eta F(w_t)+eta F(w_(t-1))` have roots
`lambda=(1-2 i a +/- sqrt(1-4a^2))/2`, `a=eta*omega`.
For `0<|a|<1/2`, their squared moduli are
`(1 +/- sqrt(1-4a^2))/2`, both below one. Thus a precisely scoped
prediction is asymptotic damping instead of Euler energy growth. The
initial ordinary step can expand; monotonic step-by-step energy is not claimed.
A dimensionless contraction surrogate is `-log(R_next/R_current)`;
its interpretation needs the fixed local model and excludes zero residual.

Neural non-saturating BCAP is not a bilinear monotone game. Its normalized
field is nonlinear, its updates alternate D then G/prior, and only networks
receive optimism. The theorem above does not carry over. Nor do scalar width
oscillations prove nonzero antisymmetric Jacobian flow. We test the resulting
trainer, while describing observed KS/width variation as stability proxies,
not measured entropy production or an identified game circulation tensor.

## One substantive trainer delta

The candidate [declaration](../../../../configs/forge/ideas/thermodynamic-bcap-optimism-v1.json)
adds public `Recipe.thermodynamic_optimism=1`. Compute each network's ordinary
unit-rate DualNorm direction `u_t`, including aspect/kernel factors and
smoothing .001; apply `delta=-eta*(2*u_t-u_previous)` thereafter. The first
active gradient uses `delta=-eta*u_t`. G/E/D share the same rule, coefficient
and fixed settings across all tasks. There is one cached normalized field per
network parameter; state validation binds it to the coefficient and checks its
shape, dtype and finiteness. A missing gradient consumes no history. A zero
current field allows the explicit previous-field correction.

Prior/table roles retain the exact sampled-row rule, including unsampled-row
immobility. No sampled prior cache is introduced: an asynchronously sampled
previous row field would be stale by a different interval on every row.
This is selective network optimism in the coupled network/prior game, not
an implementation of fully simultaneous three-player optimistic descent.

The control is the exact saved configuration
`bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36`
resolved in this same source/runtime: non-saturating BCAP coefficient1/cap1
on every update, full zero-momentum DualNorm smoothing .001 with per-offset
convolutions, constant G/E .012, D .018 and prior .030, zero input/output
training noise and zero EMA. Every other resolved winner setting is inherited.
Archived source digest `2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`
and revision `dfe88a2ee15fb9d83ffdc5c8a25d698686b73d4e7c63b6b0e35efb0d64e94359`
retain their original 7/21 Tier 2 result; those are not matched new-control credit.

## Prior evidence and primary literature

[The winner analysis](../../bcap-tier2-search/FAILURE_ANALYSIS.md) records Gaussian
stationary width .354–2.492, 2/72 stationary full passes and mode hold's
late quality dip to .8694. [Pacing](../../dualnorm-pacing-v2/README.md) and
[Gaussian diagnosis](../../gaussian1d-diagnosis/README.md) motivate a response
to direction history rather than another unchanged rate or seed experiment.
The saved-state analysis also finds conditional coverage opposing identity;
optimism might leave that competing objective unresolved.

[Daskalakis et al., Training GANs with Optimism](https://arxiv.org/abs/1711.00141)
propose optimistic game optimization. This track applies that family of
previous-direction correction to normalized network fields; it does not claim
novelty for optimism. [Gidel et al.](https://arxiv.org/abs/1802.10551) study
variational inequalities and extrapolation methods.
[Balduzzi et al.](https://proceedings.mlr.press/v80/balduzzi18a.html) motivate
the symmetric/antisymmetric game-Jacobian decomposition. Their assumptions
and algorithms do not certify this stochastic selectively corrected trainer.

The older [Optimistic Adam screen](../../../toy100/optimistic-adam-scratch.md)
failed all six declared rows under two different cores. The original
[game-dynamics attempts](../../../toy100/formulation-round-20260924/game_dynamics/conclusions.md)
also retain failures and different CPU-fixture initialization contracts.
Those are negative prior evidence, not reasons to rerun an unchanged method.
Our full smoothed DualNorm field, network-only correction, exact winning
non-saturating recipe and public initializer define a different operator/cohort.

Newer source-bound [network magnitude PR320](https://github.com/255BITS/ParticleGAN/pull/320),
[prior magnitude PR319](https://github.com/255BITS/ParticleGAN/pull/319), and
[combined magnitude PR321](https://github.com/255BITS/ParticleGAN/pull/321)
all failed the Gaussian continuous-learning gate and regressed ring. The combined
cached-past arm uses a temporary joint preview from its previous field, evaluates
a fresh field there, restores the base, then corrects. This candidate instead
evaluates the existing alternating field once at actual parameters and directly
combines current/previous normalized directions. It introduces neither fixed
spectral cap .1, prior cap .001, nor a preview evaluation. These studies differ
in source, prior and timing and supply no matched qualification reuse.

## Prediction, competing explanation and finite protocol

Before training, predict Gaussian stability terminal KS <= .05, improved
stationary full-pass count and narrower width variation; mode hold must acquire
five terminal passing checks, and the broad-vector/two-pole guards must retain
full passes. The study's machine signature is Gaussian terminal KS <= .05;
KS > .05 falsifies that signature. Every unchanged sustained task bound remains
necessary, including conditional trajectory identity MSE <= .02.

The competing explanation is continued sampled-prior motion, non-Gaussian
latent allocation or task-owned objective competition. Independent noisy
unit directions with coordinate variance `sigma^2` would give
`Var(2u_t-u_previous)=5 sigma^2` (or `(5-4rho)*sigma^2` with lag correlation
rho). Optimism may therefore amplify the disturbance it was meant to correct.
An improved endpoint alongside failed retention or guardrail regression does
not establish repair or identify circulation as the failure's cause.

[Diagnostic view](../../../../configs/forge/views/thermodynamic-circulation-v1.json)
contains unchanged two_pole, gaussian1d_smoke, gaussian1d_stability, mode_hold,
vector_two_broad and trajectory. All task budgets/gates remain intact; Gaussian
stability requires its own passing smoke checkpoint. A smoke failure must leave
that continuation BLOCKED. This is an explicit diagnostic scope, not a smaller
ordinary-tier denominator. Each recipe reserves 6,420 seconds; the two recipes
reserve 12,840 within the shared campaign ceiling 14,400, including any paid
repair/diagnostic. At most one substantive candidate and one matched control;
no scientific retries, parameter search, extra seed or continuation. Stop and
publish regardless of result.

Protocol seed0, public deterministic initialization, architectures, target/data
laws, prior widths/learnability, D/G seen batches, sampling, update budgets and
evaluation cadence remain fixed within each task. Public components and
GANTrainer consume/checkpoint their isolated constructor/data/prior/noise/eval
streams. Clean live sampling is retained. The completed actual-training GIFs and numerical
readouts are linked above. The parent maintains the sole campaign
leaderboard; this report only contains task comparisons.

## Reproduction and logs

Use the pinned scientific source and Python3.12 `.venv`; the public package must
resolve to this worktree. Source/runtime requests freeze at enqueue. Planning:

```sh
.venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue plan thermodynamic-bcap-optimism-v1 --study thermodynamic-circulation-candidate-v1 --show-boundaries
```

The same dedicated queue receives both declarations; only the parent's shared
coordinator executes them. Do not start a second drain. Tail
`/mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue/events.jsonl` and
`/mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue/thermodynamic-circulation-v1/progress.jsonl`.
Bulk logs/JSONL/checkpoints stay in the artifact filesystem. Compact outcomes,
provenance and media are committed here. Re-export/verify without training:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python reports/forge/bcap-physics/thermodynamic/thermodynamic_publish.py --queue /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue --certificates /home/martyn/dev/ParticleGAN/reports/forge/attempts
.venv/bin/python -m experiments.forge compile --summaries-only
```

The central queue/source snapshots and artifact paths in provenance are the
original local archive. Mirrored original envelopes remain ignored under
`reports/forge/attempts/`; they are not force-added to Git.
