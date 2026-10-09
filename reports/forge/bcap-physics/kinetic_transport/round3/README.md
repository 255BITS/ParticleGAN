# Round 3: finite proposal descent

**Reject this exact finite-descent successor.** It passes **2/5 mutually runnable
tasks**, versus **3/5** for matched local-v2; both retain one genuine two-pole
BLOCKED cell. All ten full-budget jobs complete once for **344.233370 paid worker
seconds**, with zero scientific retries and no remaining reservations. The
backtracking rule works numerically, but loses the sustained unequal-mass PASS
and worsens full unequal-width covariance.

## Completed matched results

| Unchanged task | Local-v2 primary control | Finite-descent successor | Complete outcome |
| --- | --- | --- | --- |
| Two-pole | BLOCKED | BLOCKED | Neither frozen component host consumes this sample-space mechanism; no launch |
| Gaussian smoke | PASS, 13/24 confirmed states | PASS, 12/24 confirmed states | First confirmation 84→125; any-confirmed-state acquisition gate retained |
| Gaussian stability | FAIL, 28/72 stationary, 11/24 shifted hold | FAIL, 37/72 stationary, 10/24 shifted hold | Both miss deadline reacquisition; count forecasts≥50/≥16 fail |
| Unequal mass | PASS, 13/24 checks, suffix 5 | FAIL, 16/24 checks, suffix 0 | Full covariance .522577→1.123510; sustained repair lost |
| Unequal width | FAIL, 0/24 checks | FAIL, 0/24 checks | Full covariance 2.480438→3.859733; forecast≤1.25 falsified, original≤.85 still fails |
| Two broad | PASS, 24/24 checks, suffix 24 | PASS, 24/24 checks, suffix 24 | Entire trajectory and final models/optimizer/RNG states match bitwise |

[Final metrics and temporal failures](results.json), [source and protocol
receipts](provenance.json), and [saved-center/data diagnostics](saved-state-diagnostics.json)
bind these outcomes. [Archived local-v2 parity](predecessor-parity.json) verifies
all five new primary-control trajectories, endpoint metrics, graders, final model
tensors, base optimizer states and named RNG states against their original
round-two identities. These are fresh matched current-source receipts; archived
evidence is neither substituted nor awarded qualification credit.

The [complete backtracking audit](backtracking-audit.json) verifies **all 9,600
actual candidate updates** and **11,542 replay evaluations**. Every applied value
satisfies its finite same-batch bound; **1,517 full proposals increased the batch
loss and were contracted**, with **zero rejected proposals**. Contractions occur
on 156/1000 smoke, 956/5000 stability, 31/1200 unequal-mass and 374/1200 width
updates. Broad accepts all 1,200 full proposals; its [inactive parity receipt](inactive-broad-parity.json)
confirms exact preservation of the matched trajectory and final state. No
additional sampling or critic updates occur in these checks.

Unequal mass improves the number of isolated passing observations but breaks the
required terminal suffix. At 1050 its minimum eigen ratio `.130087` falls below
`.15`; at 1200 full component covariance exceeds `.85`. Rare centers increase
6→7 and served rare count 114→126/4096, against target mass `.02`. Rare spill
increases `.140351→.269841` and rare full covariance `.470663→2.986301`.
Overall mass TV slightly improves `.016426→.015986`, so occupancy alone does not
explain or repair this quality failure. Minimum mass ratio `.877028` still meets
the auxiliary forecast≥.5; the required sustained-PASS forecast is falsified.

Unequal width gives mixed local results. Narrow full covariance errors change
`4.787480/4.601254→11.900118/3.121494`, and their spill fractions change
`.042753/.062749→.100410/.020747`. The first spill forecast fails while the
second improves. Four-sigma core-average covariance improves `.289171→.179413`
and HQ `.955078→.967041`, but full covariance worsens and all 24 complete checks
fail. Reduced core error cannot establish repaired tails. Centers change
`61/64/65/66→62/62/65/67`; mass remains populated rather than collapsing.

Gaussian final KS worsens `.060653→.082014`, and final mean error
`.071086→.202504` also exceeds its `.2` bound. Width ratio `1.001564` passes.
Smoke's candidate final KS `.050318` exceeds `.05`, despite earlier independent
same-state confirmations satisfying the declared acquisition gate. Stability
restores each arm's **exact 1,000-update checkpoint from its qualifying smoke
run**, not a selected best checkpoint; all streams and histories continue
unchanged. Thus smoke PASS does not assert that every state or its endpoint
passes quality. Both continuations complete their original 5,000 new updates.

**Keep local-v2's scoped rare-density evidence; stop this exact backtracking
revision as a repair.** The experiment falsifies the sufficiency of same-batch
finite descent for these served-quality goals. It does not prove which changing
batch feature, critic motion or optimizer-history effect causes the regressions.
No weight tuning, seed experiment, automatic successor, continuation or promotion
follows. Native/image/conditional transfer remains unmeasured; no ordinary Tier2,
calibration or default-adoption claim follows. The parent maintains the one
current goal leaderboard.

## Source, cost and verification

Both arms execute scientific commit
`2e42afee1da43bbefb510cc8d01f22a2840b483f` and source digest
`6d5401c4d567480d4a4ec197dd7ace9ac5bea3db9b14a15b9789507216340eb8`.
Candidate revision is `946b56de446aad0b362661c0110da16a34098fa74fff4a88b2eaa53249d75236`;
control revision is `5f1aa24fb58837b8f1b2de1150ee4925db1d035bea77fe0366fad4924634d81a`.
Later publication changes only reports/reproduction sources, with no unmeasured
trainer correction presented as trained evidence.

Forge's automatic study outcomes remain `incomplete` because the declared
two-pole capability is unsupported, despite every runnable worker completing.
The candidate's measured falsifier is explicitly satisfied. Both request
subscriptions are concluded; no missing-evidence request authorizes more work.

Initial models/prior, final consumed stream states, actual data batches, fixed
prior/sampling laws, update clocks and complete gates match. Vector batch replay
matches all 1,200 consumed data batches per arm. Each arm completes 9,600 base outer
updates, 19,200 total. Python 3.12.13, Torch 2.14.0, NumPy 2.5.2, RTX A6000,
deterministic execution, one Torch thread and TF32 off are source-bound. The
single global consumed recipe delta is the backtracking boolean.

Candidate paid 180.877021 + control 163.356348 = **344.233370 seconds**.
Declared campaign ceiling 12,840, executed full reservations 12,240, diagnostic
allowance 120 and track ceiling 14,400 remain intact. The successful saved-state
probe separately measures 2.029498 GPU seconds; its earlier metadata failure
adds no proposal/draw and remains covered by the conservative 120-second allowance.
No reservation, drain worker or status watcher remains. Device sharing supplies
accounting, not an optimizer-speed comparison. CPU software/media work adds no
scientific qualification.

[141 distinct focused software checks](software-verification.json) pass; the
final six backtracking checks overlap that set. They verify finite overshoot,
full-bit preservation, exact rejection, consumed-noise replay, unchanged critic
and RNGs, base optimizer state preservation, buffer purity and exact checkpoint
continuation. Four initial sampler fixtures used an invalid constructor keyword;
correcting the fixtures required no trainer change or scientific retry. Forge
declarations validate. [Compatible scorer controls](scorer-controls-reuse.json)
retain four oracle PASS/four collapse FAIL with exact original receipt/scorer
hashes and unchanged laws/gates. Original compact records and the current
technique inventory remain bytewise unchanged; memory refreshes use summaries
only, preserving qualification. Publication exactly reproduces
**432 saved primary metric sets** and renders **10 actual-training GIFs**, each
with nine fixed observation frames. No image inspection determines the grades.

| Task | Local-v2 actual-training GIF | Finite-descent actual-training GIF |
| --- | --- | --- |
| Gaussian smoke | [Acquisition](control-gaussian1d_smoke.gif) | [Acquisition](candidate-gaussian1d_smoke.gif) |
| Gaussian stability | [Hold and shift](control-gaussian1d_stability.gif) | [Hold and shift](candidate-gaussian1d_stability.gif) |
| Unequal mass | [Density and gates](control-vector_unequal_mass.gif) | [Density and gates](candidate-vector_unequal_mass.gif) |
| Unequal width | [Density and gates](control-vector_unequal_width.gif) | [Density and gates](candidate-vector_unequal_width.gif) |
| Two broad | [Density and gates](control-vector_two_broad.gif) | [Density and gates](candidate-vector_two_broad.gif) |

[Media receipts](media.json) retain the saved-input hashes. Reproduce publication
and verification without training or sampling:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python reports/forge/bcap-physics/kinetic_transport/round3/publish.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python reports/forge/bcap-physics/kinetic_transport/round3/verify.py
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/kinetic_transport/logs/progress.log
```

The mechanism, forecasts and stopping contract below were pushed before
training and retain their original values.

This authorized comparison tests one successor of local-density-v2 against
**local-density-v2 as its one primary matched control**. The sole global recipe
delta is `kinetic_transport_backtrack: false → true`. Loss weights, base
optimizer, rates, critic updates, task resources and all numerical gates remain
unchanged. The [round-two result](../round2/README.md) retains its original
source, forecasts and sustained unequal-mass PASS. Archived winner and earlier
transport measurements provide context, never a third matched arm.

## Saved-state motivation

[Endpoint response diagnostics](prior-response-diagnostics.json) verify the
certified round-two local-v2 checkpoints and exactly replay each final consumed
real batch. One **new**, explicitly scoped 128-row/jitter draw from cloned final
streams supplies each frozen endpoint probe. These are not reconstructions of
the final training G rows. Four in-memory optimizer proposals consume 512 probe
draws, no critic updates and no mutations of archived checkpoints/streams; the
successful GPU probe takes 2.029498 seconds. An earlier failed invocation stopped
on a missing metadata key before constructing proposals or drawing samples.

All four joint parameter directional derivatives are negative. Nevertheless,
the full Gaussian-stability proposal increases composite loss by `.003044`,
and the unequal-mass proposal by `.009017`; halving accepts both. Broad and
unequal-width proposals accept full scale, although width's predicted slope
`−.180899` yields only `−.008843` finite loss change. This limited frozen-state
evidence supports checking actual finite response, not claiming that every
training update overshoots. Output-space versus parameter-space gradient
cosines also change substantially: broad adversarial/local cosine changes
`−.367745 → +.227717`. Shared-map pullbacks can change apparent force conflict.

The competing explanation is a noisy or incomplete objective. Real anchors
change each batch; finite local features can miss distant tails. Descent on
that loss may worsen served covariance, rare variance or Gaussian KS. The
critic and optimizer histories also continue changing. A shorter proposal can
suppress useful progress even when it removes finite overshoot.

## Mechanism and scope

Let `p` concatenate the current generator parameters and learned prior
locations. Construct the **unchanged actual rounded DualNorm proposal** `p*`,
and set `δ=p*−p`, `s=⟨∇Φ(p),δ⟩`. With the post-D critic fixed, the existing
same-batch objective is

`Φ = L_adversarial + L_transport + L_local`.

Try `α∈{1, 1/2, 1/4, …, 1/128}` in order and accept the first finite value
satisfying

`Φ(p+αδ) ≤ Φ(p) + 10⁻⁴ α min(s,0)`.

If all eight trials fail, restore `p` exactly. A full accepted proposal retains
its original parameter bits. Base optimizer states and clocks advance once;
they are not rewound after contraction or rejection. This controller has no
persistent state and consumes no extra randomness. Every consumed stream remains
checkpointed. Per-update scales and loss bounds are emitted as tail-friendly
JSON events in local worker logs and will receive a compact publication audit.

Replay uses the **actual consumed G rows and actual detached MoG additive
jitter**, exposed by optional public `MoGParticlePrior.sample(return_noise=True)`.
The real tensor and post-D critic stay fixed. Extra generator forwards use cloned
buffers, so BatchNorm statistics do not update again. The admitted deterministic
MLP hosts have zero additive training noise. Known stochastic generator modules
are refused when enabled. The default false path and two-value sampler API retain
their original operations; arbitrary custom stochastic modules are outside this
diagnostic's support claim.

This follows the established sufficient-decrease/backtracking idea of
[Armijo (1966)](https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-s.pdf).
That paper's fixed smooth minimization assumptions do not establish convergence
for a changing GAN objective, finite empirical sorted transport or this bounded
eight-trial controller. The directly checked claim is finite same-batch
nonincrease. Held-out task gates determine scientific success. No physical
analogy supplies an additional guarantee.

## Frozen comparison and forecasts

Both source-bound arms use protocol seed0 and the public deterministic
initializer. Each task's architecture, target/data law, seen batches, fixed
prior widths/weights/components, sampling, update allowance and evaluation
cadence remain fixed. Frozen Gaussian/vector adapters reuse the same real
tensor for D and G. Constructor, data, training and evaluation streams remain
isolated. Both Gaussian continuations require and restore their own passing
smoke checkpoint. Clean/live evidence remains separate from noisy or EMA.

| Unchanged task | Per-arm full reservation | Forecast and complete gate |
| --- | ---: | --- |
| Two-pole | 300 s declared, no launch | Both frozen component hosts remain genuinely BLOCKED; identity/zero fixture remains a separate cohort |
| Gaussian smoke | 120 s | Retain complete own confirmed smoke PASS |
| Gaussian stability | 600 s | At least 50/72 stationary and 16/24 shifted passing checks; original full retention/deadline gate remains authoritative |
| Unequal mass | 1800 s | Retain full sustained PASS, terminal suffix≥5 and endpoint minimum mass ratio≥.5 |
| Unequal width | 1800 s | Final `component_covariance_error≤1.25` versus archived local-v2 2.480438; `>1.25` falsifies the primary prediction. Original full bound remains≤.85. Narrow spill fractions should not exceed .042753/.062749; report core, full tails, mass and eigen ratios separately |
| Two broad | 1800 s | Retain full sustained PASS |

The source-frozen studies are `kinetic_transport_candidate_round3` and
`kinetic_transport_control_round3`, campaign `kinetic_transport_round3`,
view `kinetic_transport_round3_diagnostic`. Transport-v1 appears only as the
control study's admission reference; only local-v2 and its successor execute.
Declared full campaign allowance is 12840 seconds, with 12240 runnable full
reservations. A conservative 120-second saved-state diagnostic allowance
includes both probe launches; the fresh track ceiling is 14400 seconds including
diagnostics and any retries. One sharing-enabled worker runs this track's own
bounded queue. This is a mechanism diagnostic and confers no ordinary Tier2,
calibration, native/image transfer or default-adoption credit.

Run every admitted job through the unchanged full budget and full gates.
Stop this one revision and publish all results afterward; no sweep, seed
repeat, automatic successor, continuation or promotion follows. A favorable
surrogate forecast cannot substitute for complete task PASS. The parent owns
the single current goal leaderboard; this report is a scoped comparison.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/kinetic_transport/logs/drain.log
```

Worker logs, per-update events, checkpoints and saved arrays remain outside Git
under this track's queue. Publication will retain compact final metrics,
source/protocol/cost receipts, scorer-control identities and actual-training GIFs.
