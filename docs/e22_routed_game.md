# Routed E22: penalty units, represented mass and game qualification

The pooled-token example now applies KA2 in token units while retaining one
RpGAN score per conditioning context. Routed DV12 uses represented mass and
support, rather than the number of row slots. These corrections preserve
independent-row E22's regularization, perturbation and parity contract.

Clean learned-feature improvement remains the routed structural criterion.
It does not certify repair of an adversarial game. The new diagnostic example
tests candidates under the noisy game and records later training behavior;
its results are observations, never proposal scores or acceptance inputs.

## Pooled critic units

For `D(x) = mean_t f(x_t)`, the derivative of each token includes `1/T`.
Applying native KA2 to `[B,T,C]` also divides A's real squared-gradient term
by `T*C`. Repeating an identical token therefore weakens R1 by `T²`: at 128
tokens, the old example applied a term 16,384 times smaller than at one token.
This is an example-level mismatch with KA2's per-input unit convention.

[`TokenPenaltyView`](../examples/e22_routed_sites.py) reshapes `[B*T,C]` back
into `[B,T,C]`, calls the complete contextual critic, and repeats its pooled
scores `T` times. Only the penalty receives this view. Adversarial real/fake
scores retain their original context grouping. Repeating the scores cancels
the derivative's `1/T`; KA2 then treats `C` as the input dimension for each
token's gradient norm. This also gives the fake caps, blended B caps and EMA
gradient proximity consistent token units. Tokens remain correlated uses of
one context in the routing evidence; their number never inflates its ESS.

Regression tests compare one, eight and 128 repeated tokens, including critic
parameter gradients, active caps, nonzero EMA proximity and lazy penalties.
Actual paired training crosses the 799 pure-A calls and first blend at call
800, with exact checkpoint continuation from call 798.

New example checkpoints record `penalty_units="token"`. The supported
`"context"` option reproduces the former example's units. Missing unit metadata
means context units when reading a historical application configuration; a
restore into a different unit convention rejects before modifying state.
The CLI rebuilds the stored convention and rejects an explicit conflicting
override. This application compatibility does not erase the DV12 version
boundary described below.

## DV12 law under an equivalent bank representation

The routed prior measure is proportional to `exp(log_mass)` on each row's
latent position. Inactive rows have `log_mass=-inf`. Exact duplicate active
positions form one atom and sum their masses before normalization.
For normalized atom masses `p_a`, coordinate mean `mu`, and latent dimension
`d`, its cell width is:

```text
variance = sum_a p_a * (z_a - mu)^2
effective_atoms = 1 / sum_a p_a^2
width = sqrt(variance) * effective_atoms^(-1/d)
```

The existing controller initialization and `.01` bandwidth refresh use this
width. Per-site DV12 draws remain independent and occur after routing, with
the same mixed-code shapes and ordering. Local clipping searches active
support atoms only. An inactive outlier cannot change the width or clip
radius. Replacing an inactive row by an exact parent duplicate with half the
parent's mass leaves the measure, support and noisy function unchanged, up
to floating-point arithmetic. Table positions and all declared row state
must duplicate together to preserve the router's function as well.

The geometry describes the bank's prior measure, not observed token usage.
An antisymmetric split changes that measure intentionally. Retiring a row
with nonzero mass also changes the function; neither operation is an
equivalent representation and neither receives an invariance guarantee.

CPU and CUDA regressions exercise refreshed bandwidth, clean/noisy complete
two-site outputs, tied key/value gradients, summed parent/child gradients,
activation-checkpoint replay, exact recovery, and selected fast/averaged
table and mass buffers. Independent-row DV12 retains its original arithmetic.

`RoutedRows.to_dict()` and routed policy recovery state record
`routed_geometry="mass_atoms_v1"`. The marker is accepted by the constructor
for configuration round trips; unsupported versions reject. Routed
checkpoints from the former raw-row geometry reject before restoring any
owner. Exact old-law recovery requires the prior release. To adopt the new
law from an old run, initialize a fresh policy and explicitly copy compatible
model/table/router weights; rebuild optimizer/controller/evidence state.
That is a new trajectory, not exact resume. Current-version checkpoints
resume all training and private replay streams exactly.

## Replaying the noisy game without changing it

[`e22_routed_game.py`](../examples/e22_routed_game.py) captures explicit model,
table, controller, critic optimizer and paired KA2 penalty ownership. Pass
private latent and paired-noise generator states. Each candidate/replicate
uses its own copy of the same bundle and the same draw states:

1. Rerun the complete critic-side generator model with per-site DV12 and a
   paired real/fake error-noise draw. Evaluate the copied next KA2 call and
   critic game-plus-penalty gradient.
2. Rerun the complete generator-side model with its separate DV12 draws and
   separate paired noise, retaining the table and network gradient paths.
3. Report score behavior, RpGAN payoffs, clean/noisy feature discrepancy,
   gradient norms, mass-normalized gradient spectrum and context usage/ESS.

The critic weights are held fixed between these diagnostic roles. No
diagnostic optimizer step occurs. This is a fixed-critic gradient-response
probe, not a simulation of an adaptive complete training update. The copied
models use evaluation mode; this matches the example's deterministic linear,
tanh and frozen BF16 forward, but does not certify a caller's training-mode
dropout or mutable-buffer behavior.

`prospective_bandwidth=False` freezes this update's controller bandwidth;
`True` additionally performs one candidate-specific width refresh on each
controller copy. Both search the candidate's own represented support. The
current/prospective distinction must be kept when interpreting a row move.

```python
from e22_routed_game import capture_game

# Rebuild the ordinary application and load policy.state_dict() first.
# Save these caller-owned diagnostic states alongside that checkpoint.
replay = capture_game(
    models=models, table=table, controller=policy.controller,
    critic_optimizer=critic_optimizer, penalty=policy.penalty,
    rows=rows, output_sigma=policy.output_sigma(),
    latent_rng_states=saved_latent_states,
    paired_rng_states=saved_paired_states,
    penalty_units="token",
)
result = replay.compare(
    protected_context, protected_targets, before_candidate, after_candidate,
    residual_scale=error_scale, split_rows=(parent, child),
)
```

Reconstructing this capture after ordinary checkpoint recovery reproduces the
same diagnostics. It consumes no training/global RNG, duplicates no DV12 or
row evidence records, and modifies no live model, penalty clock, gradients,
optimizer or module modes. Tests also compare instrumented training against
unobserved training across real moves and require complete state equality.

## Benchmark contract

[`e22_routed_support.py`](../examples/e22_routed_support.py) retains fixed and
movable-bank controls and offers two source/time-conditioned tasks. `pooled`
uses the existing API-initialized edit example. `transitions` requires moving
spatial transitions from two bank-only nonlinear residual branches; their
output factors start deliberately at zero, as in a neutral low-rank adapter.
All remaining trainable projections and the R2 bank use
`init.deterministic_orthogonal_` with fixed role keys. Both sequential sites
share one 128×4 bank and frozen BF16 host layers.

```bash
PYTHONPATH=. python -u examples/e22_routed_support.py --task pooled \
    --compare --steps 1200 --tokens 128 --particles 128 --device cuda \
    --max-context-harm 1e-4 --include-default-full --game-diagnostics \
    --receipt /tmp/e22-pooled-game.json > /tmp/e22-pooled-game.log
tail -f /tmp/e22-pooled-game.log
```

The `1e-4` setting is an explicit benchmark allowance. The additional full
arm uses the library's zero feature-harm default. New API-initialized example
runs also default to zero; the historical conformance fixture explicitly
retains its `1e-4` allowance. Checkpoints record the resolved allowance and
the CLI honors it on recovery. Neither allowance bounds output RMSE.

Each arm shares initial weights, batches, paired base noise and the four
ordered DV12 draw shapes/states per update. Private diagnostics use four
matched Monte Carlo replicates, not a seed sweep. Candidate observations are
captured only after clean structural selection; game/output measurements do
not choose or veto moves. Fit, protected and final validation contexts remain
separate. Clean held-out error, feature gains and wall time are reported
together. Timings exclude diagnostic replay/capture and synchronize updates;
they describe a shared GPU, not isolated throughput.

Temporal gradient and usage observations help test persistence. A single
feature gain, instantaneous payoff improvement, gradient norm or effective
rank is insufficient to establish game repair. Such a claim also needs
consistent later behavior under critic adaptation and benefit against the
matched movable-bank control.

## Recorded 1,200-update comparisons

The [compact receipt](e22_routed_game_results.json) records the exact source
hashes, commands, all 100-update checks and four-draw paired move deltas.
The commands regenerate full raw receipts. Each task has one matched fixed
initialization; there is no seed or hyperparameter sweep. All arms crossed
the actual KA2 blend and served fast weights at their recorded checkpoints.
Elapsed times are synchronized updates on a shared A6000; diagnostic replay
and capture are excluded.

| Task | Bank/control arm | Final clean RMSE | Maximum token L2 error | Mean RMSE at 800–1,200 checks | Update time | Splits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pooled edit | Fixed | .006118 | .024153 | .006345 | 23.72 s | 0 |
| Pooled edit | Movable, controls off | **.005882** | .025523 | .009395 | 25.36 s | 0 |
| Pooled edit | Full, zero-harm default | **.005882** | .025523 | .009395 | 74.33 s | 0 |
| Pooled edit | Full, explicit `1e-4` allowance | .006053 | **.022196** | **.006208** | 72.66 s | 40 |
| Moving transitions | Fixed | **.003773** | **.023807** | **.004549** | 25.65 s | 0 |
| Moving transitions | Movable, controls off | .004242 | .034118 | .004759 | 25.47 s | 0 |
| Moving transitions | Full, zero-harm default | .004314 | .029280 | .004700 | 74.37 s | 5 |
| Moving transitions | Full, explicit `1e-4` allowance | .004301 | .031305 | .004777 | 72.70 s | 45 |

The corrected API-initialized pooled task now exercises structural moves with
the explicit allowance. Its zero-harm arm accepts none and matches the
movable validation metrics at every recorded check. The allowance arm has
slightly worse endpoint RMSE and lower maximum error, while its final five
checks have lower mean RMSE. Endpoint and trajectory summaries answer
different questions; neither is sufficient to declare a game-repair winner.

The neutral, bank-dependent moving-transition task accepts five moves with
the zero-harm default, at updates 2, 9, 11, 23 and 25. The CUDA conformance
test resumes exactly across the first accepted move with its full 128×4 bank,
128 tokens, token-unit penalty and frozen BF16 layers. The fixed bank has the
lowest endpoint and later-check errors here. Moving or restructuring the bank
is not established as this task's limiting capacity.

### What the noisy evidence establishes

The clean guard accepted every committed move. With the same critic,
per-site DV12 and paired noise, the four-draw mean responses are:

| Task/control arm | Splits | Noisy G loss worsened | Noisy feature error worsened | Parent flagged persistent at commit | Mass-gradient rank increased / decreased |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled, explicit allowance | 40 | 20 | 18 | 0 | 17 / 23 |
| Transitions, zero default | 5 | 0 | 0 | 1 | 2 / 3 |
| Transitions, explicit allowance | 45 | 11 | 8 | 0 | 26 / 19 |

All five default transition moves improve the immediate noisy payoff and
feature probe. Their gradient spectrum moves in both directions; later
comparisons do not establish a persistent conditioning or support repair.
In the pooled allowance run, half the accepted moves worsen noisy generator
loss, and 18 worsen noisy feature distance despite improving clean distance.
Clean evidence therefore cannot be substituted for noisy-game evidence.

`RoutedEvidence.flag` combines exposure, deletion effect, observation count
and temporal gradient persistence. Structural eligibility currently uses
fresh deletion effects and enough effective contexts, without requiring this
flag. Several transition moves also occur before a full persistence window
could form. The instantaneous deletion effects are refreshed at each probe;
they are not a persistent-support certificate. Mass-weighted usage and
gradient-spectrum observations do not prove missing support or an optimizer
conditioning bottleneck by themselves.

A separate regression constructs a clean-feature gain of `.184438` while
matched noisy generator loss worsens by `.110502`. Another proves the entire
instrumented trajectory remains identical to ordinary training, including
accepted moves and checkpoints. The instrumentation diagnoses the current
adaptation; it introduces no game-health acceptance law.

Retain fixed and movable baselines for integration validation. These results
qualify the unit correction, mass-invariant noise law, real row moves and
recovery. They leave **persistent game repair unestablished**. A future
structural criterion needs a validated temporal game statistic and consistent
improvement after critic adaptation; changing the selector to output MSE or
an instantaneous generator payoff would not supply that evidence.

Validation: the full CPU suite passed 1,348 tests with 18 subtests. Latest
game/token/capacity checks passed another focused set of 36 tests; all 27
CUDA conformance cases ran and passed, including noise initialization,
late-split transport, BF16 whole-model replay and default spatial move resume.
The installed wheel's 20 package sources match the checkout, and independent,
paired, whole-model and noisy-game examples run outside it with exact recovery.
Native independent E22 golden behavior and research configurations are unchanged.
