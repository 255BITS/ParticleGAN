# Bounded D-underfit audit: rotated100 at 14k

**The saved critic has measurable remaining descent under the original regularized objective. Restoring the pre-cut LR is not demonstrably better on that objective.** No full GAN candidate was launched by this audit.

The last saved D stationarity decision cut scale 1/8 to 1/16 at 13991. Starting from the 14k checkpoint, two shadow critics each received 250 updates against the frozen saved generator law. One retained the post-cut D LR .000265625; the other restored the pre-cut scale, LR .000531250. Training batches were paired between arms. The original candidate KA2 penalty/optimizer was used in both; all G/prior/noise/controller state remained frozen.

## Independent confirmation

Each number below is a paired difference versus the untouched saved critic, evaluated on 128 fresh independent batches of 2048. Negative loss differences indicate improvement. Intervals are normal 95% Monte Carlo intervals conditional on this checkpoint, calibration and training seed.

| Shadow at 250 updates | D logistic change | Fixed-context KA2 penalty change | Fixed-context total objective change |
|---|---:|---:|---:|
|post_cut|-4.35e-05 [-5.804e-05, -2.896e-05]|7.024e-06 [6.935e-06, 7.114e-06]|-3.647e-05 [-5.102e-05, -2.193e-05]|
|pre_cut|-5.312e-05 [-7.309e-05, -3.315e-05]|1.654e-05 [1.642e-05, 1.667e-05]|-3.658e-05 [-5.655e-05, -1.66e-05]|

The direct pre-cut minus post-cut total-objective difference is -1.026e-07 (interval -7.476e-06 to 7.271e-06). It does not resolve a benefit from restoring the higher rate. The higher-rate arm lowers raw logistic loss more, but incurs an offsetting larger fixed-context penalty.

| Shadow | Final center derivative | Final shape derivative | HQ modes favoring/opposing/uncertain for shape |
|---|---:|---:|---:|
|post_cut|-1.256e-05|-3.31e-06|67 / 24 / 9|
|pre_cut|-1.581e-05|-5.4e-06|63 / 27 / 10|

Negative diagnostic derivatives favor the corresponding correction under the local generator loss, with fixed membership/jitter. The pre-cut arm has a more favorable shape derivative, but this oracle geometry is evaluation-only and must not become a controller input. These scores do not predict a passing GAN continuation.

At 24 updates neither arm showed a statistically resolved full-objective improvement. The 250-update budget did resolve improvement in both monitoring and independent confirmation. A very short shadow test would therefore have been inconclusive at this checkpoint.

## Fixed-context objective verification

The held-out total objective uses the exact candidate penalty implementation, with the saved baseline EMA anchor, penalty record, controller values, coefficient and phase. The penalty record is restored before and after every held-out call. Consequently both candidate critics are judged against the same penalty context; the teacher does not move during evaluation. The shadow training paths retain ordinary adaptive KA2 state.

Two repeated paired baseline evaluations were bitwise identical, including penalty values. Full optimizer/anchor/record state equality was verified after every held-out evaluation. Source-checkpoint SHA256 and frozen G/prior/sigma/controller state equality were verified after training. These receipts are in `manifest.json` and `completion.json`.

## Scope and next decision

This result supports challenging a displacement-based STATIONARY verdict when a bounded same-law shadow continuation demonstrates held-out descent. It supports testing a cut veto, not automatically resetting D to base LR. The audit does not establish which rate is best, whether extra D updates should replace a cut veto, or whether the effect generalizes across training seeds/tasks.

The requested grid100 arm could not be run: neither st5 QR grid100 nor st7 grid100 saved final-state.pt. Only the rotated100 14k native state exists. No full trajectory was reconstructed.

The full read-only artifacts are under
`/ml2/hypergan/lrfree-20260926/reports/st5-critic-underfit-audit/`:
`manifest.json`, `observations.jsonl`, `training.jsonl`, paired baseline/arm
NPZ observations, saved 24/250-step shadow critics, and `completion.json`.
The reproduction script is
`/ml2/hypergan/lrfree-20260926/reports/st5-critic-underfit-audit.py`.
This summary is adapted from the harness's `summary.md` to make paths
explicit. The original checkpoint remained unchanged.
