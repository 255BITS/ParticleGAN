# Precision wave3 review: RP7–RP9

None of these candidates qualifies for release. RP7 and RP8 both succeed on their measured public recovery rings, then fail the same two broader tasks. RP9 fails those cheap tasks first, so its ring is correctly unrun. The successful rings remain valid evidence; broader rejection does not erase them.

| Candidate | Initial ring acquisition/retention | Shifted ring recovery/retention | Tiny mode_hold | Intensity image | Reported run wall time |
|---|---|---|---|---|---|
| RP7 |600; 181/181 |+340; 187/187 |0/24, suffix 0; final 0 modes |3/24, suffix 2 |563.76s |
| RP8 |520; 189/189 |+290; 192/192 |0/24, suffix 0; final 6 modes |10/24, suffix 3 |470.09s |
| RP9 |NOT_RUN |NOT_RUN |0/24, suffix 0; final 6 modes |7/24, suffix 1 |104.92s |

Ring retention counts begin at actual arrival within each target segment; both measured rings have zero subsequent departures. The broader tasks retain their separately declared final five passing-check requirement. RP8 image's miss at 525 breaks an earlier passing streak, despite its passing endpoint; RP9 image misses 400,475,575 after first arrival 375. All observations and failures are archived. Total reported quality-window wall time is 1,138.77s (18.98min), including those workers' measurement work; this excludes regression tests, source review, other attempts and the separate released K3P reference. RP7 used 428.95s ring/92.49s tiny/42.33s image; RP8 used 347.52/86.23/36.34; RP9 used 74.77/30.15 for tiny/image.

RP7 measures a second secant direction and solves a projected two-dimensional implicit response. RP8 uses the same three native-Adam fields but fits the full linear residual with bounded forward coefficients. Across its 6,400 accepted updates, all 6,400 rank diagnostics are rank2, no zero update or half-plane activation occurs, and retained residual arithmetic checks agree within FP32 tolerance. That stronger local numerical fit does not establish support coverage. RP9 returns to the original two-field secant update and adds bounded prior noise from predictor/corrector disagreement; it also fails to establish all eight tiny-task modes. These are observed limits, not proof of a single causal explanation.

All three use learner-owned reversible precision, total_steps=None and fixed one-time360/720 noise initialization. No target-quality oracle or evaluator horizon was found in the frozen learner routes. Model/task/stream/scorer bindings remain frozen. Both RP7/RP8 tiny runs and RP9 preserve all 1,200 caller batch/cursor receipts from the existing task. Candidate regularization differences stay explicit: the historical image prior_weight=.05 is not the public candidate prior_reg0. The tiny 12-particle/z4/batch128 host remains distinct from the 20k/z2/2048 public recovery ring.

The rollback-test masking issue is fixed: earlier simultaneous faults let an early G forward hide the later precision-evidence exception. Separate cases now reach both sites and assert full checkpoint equality/context restoration. RP7's corrected receipt is 79 passes; RP8 records 84; RP9 records 86. Test source and learner source are distinguished: learner ZIPs are sealed, while test files are separately retained audit snapshots. RP8 source was preserved before the active checkout advanced to RP9. RP9 adds a private seed 6 stream correctly to checkpoint/rollback, but its broad initial JSON omits the separately promised per-stream hash receipt; source/checkpoint inclusion must not be overstated as independently verified initial RNG parity.

No candidate completed its own stationary, long, repeated-change, budget-prefix or CUDA cross-process replay qualification after these early failures. No ancestor result is inherited. The public GANTrainer path is demonstrated; eight custom hosts with conditional inputs, auxiliary objectives or multiple optimizers still need faithful library-owned transactions. Standalone finite-horizon LR helpers do not implement the continuous controller route. Passing mechanics tests and successful rings do not close those API gaps.

Next review order: keep frozen mode_hold and intensity ahead of expensive ring/long windows, complete every started window, and preserve all misses. Avoid another richer secant/residual solve or another particle-disagreement-noise variation as a duplicate of this wave. A single exact released K3P mode_hold reference is source-feasible and would provide the missing matched finite baseline before another mechanism proposal. It needs no package change: explicit benchmark horizon 1200, released noise 120/240 and LR schedules, native lazyCPU Adam counters, released EMA, and an external full-step serial context. It must retain the tiny host's canonical CUDA init, shared data/latent order, separate latent 9/output 402+step observation streams and unchanged final five scorer. This source plan does not authorize or claim a completed run.

Detailed records: `rp7-completed-audit.{md,json}`, `rp8-completed-audit.{md,json}`, `rp9-completed-audit.{md,json}`, and `k3p-mode-hold-feasibility.{md,json}`. All broader source/metrics archives are new isolated dirs; shared results were appended separately by the supervisor. No merge, default promotion, extra model launch or GPU work was performed by this audit.


## Root-reviewed next attempt

The predecessor driver1657545 has exited. Continue the authorized search with
one external `gpt-6-astra`/max session, one GPU worker, and at most three distinct
new mechanisms, API-RP10 onward. No seed experiments or coefficient grid. Finish
with a reviewable report after the cap; no merge/default promotion. Eventual
integration targets develop.

Run ONE supporting exact released K3P mode_hold reference before the new quality
mechanisms. Root authorizes this measured comparison, not a fourth candidate.
Use the prepared bundle at
`/ml2/hypergan/gan-attempts/continuous-api-20260926/supervisor-support/k3p-mode-hold-reference`.
It is being prepared by ka2_tests and independently reviewed by api_constant_runner.
Do not execute it until root confirms independent review and the sealed manifest;
preparation/read-only analysis of new mechanisms may continue meanwhile. Do not
modify or tune the reference to improve its score. Keep original release code,
CPU Adam scalar counters, released float-buffer EMA, all frozen tiny-host streams,
full-step serial execution, and the explicit1200 benchmark horizon/noise120/240.
The result is supporting scheduled-reference evidence, not default eligibility.
No extra GPU worker or child model. Preserve all24 observations, full raw initial/
final states, per-update rates/noise, caller-stream receipts and package hashes.

Then pursue a new mechanism grounded in the retained failure evidence. Full
quality belongs to the exact new package: neither RP5's ten broader passes nor
RP7/RP8's successful rings transfer. The next candidate must first address the
small-host coverage/image failures without weakening scoring, using evaluator
labels/centers, task-specific policy, target-change hints, caller phases or a
predetermined ending. Avoid duplicating the current constant lane's bounded
nonlinear residual iterations or data lane's learned latent-width smoothing.
The default requirement and COMMON.md/evaluation-protocols.json remain unchanged.
