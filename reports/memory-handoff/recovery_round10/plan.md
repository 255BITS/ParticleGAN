# Local adversarial recovery: round10

Goal: draw circles autonomously or make measurable progress without full generated
training rollouts. Sixteen 2k scouts, both A6000 GPUs, the existing completion-only
queue and shared log. No MSE training objectives, clipping, EMA, seed sweeps,
private G memory, or changes to the public API exact B-cap defaults. Standard
particle regularizer once per update. All architecture/recipe defaults match round9.

## Hypotheses and controls

- Proposal repair plus shared judging trains D's writer through explored histories.
  Compare clean, shared, and mixed shared weights .25/.5/.75 at strength .25.
  A no-adapter mixed control tests whether repair adds value.
- Uniform replacement U(0,.5) has the same mean .25 as fixed mild replacement.
  Compare both clean and mixed judging. A 75% mild .25 / 25% full mixture has mean
  .4375; include fixed .4375 as its mean-matched control. Ramp all over500 updates.
  Draw strengths once per update with a dedicated resumable RNG, reuse for D/G.
- Local transition discrimination: optional D pair head receives the two candidate
  points and D's real-prefix memory from before either point. Real pair is from X;
  fake pair is G(z,M,t), then G(z,D.write(M,first),t+1), using the same z.
  This is one generated write and two outputs. Empty-prefix examples train the
  first two points from M=0. Pair loss weights .25/.5 mix with the point GAN;
  B-cap penalties use the same convex weights, in2 versus4 candidate coordinates.
  There is no regression target or third generated point. Pair and point-feedback
  branches are independent, never chained. Combined scouts cost up to4 full G
  calls /8 internal proposal-reader calls per D or G phase. Runtime remains two
  reader calls, one point, and one write. Pair heads are training-only.

Clean/shared scores always use the same causal memory for real, fake, and penalty
within each view. Mixed losses average separate GAN views, rather than averaging
their memories or logits. During D phase generated samples are detached. During G
phase D is frozen; gradients traverse its writer to the earlier G output. D alone
trains writer parameters. Pair judging trains the prefix writer; it does not by
itself add D writer gradients through the generated write. Shared point judging
provides that route in combined scouts.

Configs: experiments/configs/memory_handoff/recovery_round10/*.json.
Fresh proposal_clean_s25 reproduces the previous candidate to verify unchanged
legacy computation; older clock_control/proposal_control/shared_s25 are comparison
baselines, not new seed runs. All scout sources frozen before launch.

## Continuous evaluation and promotion

Retain all original cold-circle and warm original-orbit passes at256/1024 as
primary metrics, plus radial error, signed speed, direction, phase-position error,
stopping, and coverage. Do not equate a continuous score with a solved circle.

New warm orbit quality Q is the average over time/particles of:

    1 / ((1 + (relative radial error / .1)^2)
         * (1 + (per-step signed angular error / .03)^2))

Reference is the true orbit, fitted offline from its clean initial32 points.
Q lies in[0,1]; perfect orbit motion scores1 and stationary/wrong-direction motion
is penalized. It does not measure absolute phase, so retain startup and position
errors. Both prefixes8/32, early32 and late256 windows are reported. No extra
training or evaluation rollouts are needed: use existing saved paths.

Also report fraction of good steps (radial<.1 and absolute signed angular error
<.03), initial good-step streak, longest consecutive good arc in turns, and
quarter/full-turn fractions. These expose partial arcs before collapse. Cold
self-fit quality holds the initial32-point fit fixed with radius/speed bounds;
it is separate from reference fidelity. Validate metrics with ideal, stationary,
reversed, drifting and tiny-circle synthetic paths before viewing new results.

Completed-only selection: rank primary passes first, then worst-prefix Q, with
late quality and other errors as guardrails. At most two exact2k->5k extensions
with unchanged10k schedule. If passes remain zero, require >=20% Q improvement
at BOTH prefixes over the fresh proposal_clean_s25 control, no decrease in late
quality at either prefix, no >5% radial regression at either prefix, no >2pp
direction regression, and no >1pp increase in stopping. Prefer distinct mechanisms
among qualifiers and report matched-control comparisons as well. If no scout
qualifies, do not extend merely for a marginal local-prediction improvement.

Run recurrence/history and adapter-bypass diagnostics on leading completed
checkpoints as time permits; metrics guide conclusions, not images. Learned
evaluation particles and in-distribution radii do not establish generalization.

Log: `tail -F runs/memory_path/core_round1/train.log`
