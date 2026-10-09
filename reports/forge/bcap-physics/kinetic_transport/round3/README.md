# Round 3: finite proposal descent

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
