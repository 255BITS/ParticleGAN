# Winner implementation audit and next hypotheses

Reviewed the actual saved `match_shuffle25` 2k and `match_shuffle25_5k`
checkpoints, archived training source, current implementation, public recipe,
loss/regularizer, runtime loop, and metrics. No correctness bug was identified
in the winning path. This is a bounded review, not a proof that every optional
configuration is correct. No training code, checkpoint, or recipe was changed.

## Verified behavior

- D owns one GRU32 memory per trajectory. G has no persistent private state.
- G reads D's actual memory. Its shared point reader makes a proposal; a learned
  residual adapter takes `(M, proposal)` and produces a translated read; the
  same reader emits the final point. Translation never writes back to M.
- Runtime emits `x = G(z, M, clock)` then `M = D.writer.write(M, x)`.
  Exactly one final-point write, fixed particle per trajectory, no expert after
  handoff. Cold state is zero. Warm time begins at prefix length.
- Real-prefix snapshots are strictly before targets. Candidate scoring is
  read-only. Real and fake candidate scores use the same cached judging state.
- Point training replaces at most the final prefix observation on eligible
  examples: probability .5, strength .25 after a 500-step ramp. G reads that
  state; D point loss equally averages clean and explored judging contexts.
- The independent pair branch uses two G outputs and one full generated write.
  It does not continue the point-feedback branch. Its D head judges the pair
  against the original pre-pair memory.
- D phase detaches generated candidates/proposals. Point feedback still trains
  D's writer through its explored judging state. Pair D loss trains the real
  prefix writer, but not the intervening generated write inside detached fake
  generation. G phase differentiates through the frozen writer to the earlier
  G output. D is rebuilt/re-encoded after its optimizer step before G training.
- Both fake and real score branches remain connected in G's paired loss when
  their context depends on G; a common context-only score offset cancels.
- Winning mismatch loss ranks the actual next real point above another
  episode's next real point, using clean-prefix M and the same point head G
  uses. It trains D, not G directly. Donors exclude the same episode.
- Aggregate D adversarial weights are .60 point / .20 pair / .20 mismatch;
  G weights are .75 point / .25 pair. G also gets the default particle spread
  regularizer. No prediction/repair MSE, stability auxiliary, clipping, or EMA.
- Public recipe matches the saved resolved recipe exactly: logistic RpGAN,
  exact autograd B-cap, coefficient/cap 1, applied every update. Clock has six
  dyadic bands; G sees clock, D's score heads do not. Training clocks are 0–63.

Relevant implementation: `memory_handoff_scout.py` (contexts, phases, pair loss),
`memory_recent.py` (proposal adapter/clock), `memory_scout.py` (GRU/reader/cold
runtime), `memory_local_objectives.py` (mismatch), `memory_core_scout.py` (warm
runtime/fidelity), and `particlegan/{recipes,gan_loss,grad_regularizers}.py`.

## Checks and new measurements

57 tests passed across handoff, local objectives/recovery, clock, dynamics,
core, and process diagnostics. These cover causality, gradient ownership,
active B-cap, fixed-particle/timing behavior, exact resume, and metric controls.

`audit.py` loads saved weights and performs evaluation only on CUDA0. Both
checkpoints reproduce all 128 saved cold/prefix8/prefix32 1024-step paths
**bit-for-bit**. Training's full-write pair exactly reproduces the first two
runtime points; the clean point path exactly reproduces the first point.
Current defaults therefore preserve the winning runtime behavior.

The 5k winner deteriorates within clock support, not only after extrapolation:

| Autonomous points after prefix32 | Mean radial RMSE / radius | Radial criterion passes /128 | All existing thresholds /128 |
|---|---:|---:|---:|
| 1 | .0519 | 113 | 40 |
| 8 | .2641 | 17 | 7 |
| 32 | .4504 | 1 | 0 |
| 64 | .5448 | 0 | 0 |
| 1024 | .9120 | 0 | 0 |

Short-window threshold counts are diagnostics, not full-circle successes.
All 128 first points meet the existing startup-position tolerance, but only
44 meet the one-step angular-speed tolerance. Thus startup is plausible but
already imperfect. At 32 generated points (absolute clocks32–63), no path
meets all thresholds. Unseen clock phases cannot be the sole explanation.
These are the fixed selection panel and learned particles, not a new held-out
validation set. The 2k reference shows the same qualitative early degradation.

B-cap has another precise limitation here: the winning candidate head is a
LeakyReLU MLP over concatenated `(x, M)`, hence piecewise affine in these inputs.
Within an activation region, its candidate gradient is constant in M. The
candidate-gradient penalty therefore has zero M derivative almost everywhere.
The checkpoint audit confirms an active penalty with zero writer gradient on
the clean point panel. The GAN losses still train the writer. This is expected
for this architecture/domain, not an autograd bug. B-cap does not constrain the
Jacobian of `M -> W(M, G(z, M, clock))` or guarantee restorative dynamics.

Reproduce: `.venv/bin/python reports/memory-handoff/winner_audit/audit.py`.
Details: `results.json`; earlier independent CPU B-cap check: `bcap.json`.
No training jobs launched or queued.

## What may be missing: hypotheses, not established causes

1. **Preserve predictive identity through writes.** D is rewarded for judging
   local candidates; that does not explicitly require a generated write to
   preserve everything about which process is being continued. Proposal repair
   can improve a read, but cannot reconstruct information once lost. Existing
   process interventions show output dependence fading; they do not yet tell us
   whether the information disappears from M or G stops using it.

2. **Correct behavior when composed.** A useful next-point score and even one
   successful feedback step do not verify a stable repeated transition. We have
   trained one-write contexts and tested several perturbation schemes already;
   repeating those unchanged is not a new solution. Global contraction is also
   the wrong blanket target: distinct valid processes must remain distinct.

3. **Particle–memory compatibility.** On real-prefix training anchors, z is
   sampled independently of the episode; in autonomous execution M is produced
   by that same fixed z. Pair training covers one such correlated update, but
   not the resulting long-run joint distribution. This is an additional
   hypothesis, not evidence that fixed particles are inherently unsuitable.

4. **Clock support remains a secondary gap.** Current training uses only part
   of the lowest-frequency clock cycle (~201 steps). Earlier origin scouts
   failed, and the new early-error result rules out clock extrapolation as a
   complete diagnosis. Revisit only with a matched current-winner control.

## Suggested order, not a queued sweep

First measure process decodability from frozen M at depths0/1/8/32/128, comparing
real and autonomous histories with held-out episodes and multiple probe
capacities. Use radius/speed/direction labels for evaluation only. Failed simple
decoding is not proof of information absence. A complementary fixed-clock,
swapped-state read can assess G's response to still-informative states.

If predictive information is lost, test **future-conditioned history ranking
after one generated write**: score the original episode's future observations
at several offsets against other episodes under the updated state. That asks
the writer to preserve more than immediate continuation compatibility, without
generating a trajectory or matching raw memory with MSE. Keep the adversarial
point head and compare clean/explored controls. This extends round13's one-step
ranking; it differs from round12's direct future G queries. A separate head can
still store information G ignores, so require downstream read/use diagnostics.

If information remains but G ignores it, focus on shared predictive readout or
conditioning changes and matched memory interventions, instead of adding more
storage. Another possible mechanism is learned write protection within D-owned
M, but slow/fast memory and private G GRUs have already been tested without a
win; a new version needs a new incentive, not just another size/rate sweep.

Local training is not disproved. If M retained sufficient predictive information
and the learned conditional transition remained correct under repeated use,
full trajectory training would not be intrinsically required. The open problem
is how to enforce or test those properties with bounded local objectives.
