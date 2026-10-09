# Round 5: component centers and kernel jitter

**Reject this exact prior-only routing replacement and retain local-v2’s scoped rare-density repair.** The candidate completes **2 PASS / 4 FAIL**, versus matched local-v2 **3 PASS / 3 FAIL**. All twelve jobs complete at their original full update budgets, including two 7k native runs. Paid worker time is **2952.960023 seconds**, with zero retries or remaining reservations. [PR376](https://github.com/255BITS/ParticleGAN/pull/376) is ready and unmerged.

## Complete matched results

| Original unchanged task | Exact local-v2 control | Prior-only transport |
| --- | --- | --- |
| gaussian1d_smoke | PASS; first confirmed 167; 13/24 primary checks | PASS; first confirmed 125; 4/24 primary checks |
| gaussian1d_stability | FAIL; stationary 28/72, shifted 11/24, deadline FAIL | FAIL; stationary 10/72, shifted 9/24, deadline FAIL |
| vector_unequal_mass | PASS; covariance 0.522577; 13/24, suffix 5 | FAIL; covariance 1.361399; 9/24, suffix 0 |
| vector_unequal_width | FAIL; covariance 2.480438; 0/24, suffix 0 | FAIL; covariance 0.843668; 8/24, suffix 1 |
| vector_two_broad | PASS; covariance 0.226564; 24/24, suffix 24 | PASS; covariance 0.733819; 23/24, suffix 23 |
| grid100 | FAIL; holdout precision 0.219730, 0/5 terminal checks | FAIL; holdout precision 0.262610, 0/5 terminal checks |

[Complete metrics and final-window failures](results.json), [source, protocol and cost receipts](provenance.json), [current center/kernel and uncensored tail diagnostics](current-component-diagnostics.json), and [exact archived-control parity](predecessor-parity.json) bind these outcomes. This is a scoped diagnostic comparison; the single current goal leaderboard and original ordinary qualifications are unchanged.

The width forecast≤1.25 is observed, and its final average covariance also satisfies the original .85 endpoint bound. It has eight passing scheduled observations but only one terminal pass; five are required. All four modes remain present, so this gain is not the missing-mode averaging artifact of the rejected tail-moment package. Its final counts are [1170, 816, 1109, 1001]; mass TV 0.056396, HQ 0.894287, core covariance 0.255154. The first narrow component still has full covariance error 2.349925 and spill .176923; the lower average is not uniform tail fidelity.

The required preservation forecast fails: unequal mass loses its sustained PASS. Full covariance worsens 0.522577→1.361399, and the suffix falls 5→0. Candidate counts [2322, 1141, 506, 127] retain every component; mass TV 0.027900 passes. This is a shape/retention regression rather than total rare-mode disappearance. Broad remains PASS, while its full covariance worsens 0.226564→0.733819 and suffix becomes 23. No specialist passes are pooled.

Gaussian final KS improves 0.060653→0.027805; the candidate endpoint passes every scalar bound. Nevertheless stationary passes drop 28/72→10/72, shifted hold drops 11/24→9/24, and deadline reacquisition fails. Both continuations restore their own exact eligible smoke state and complete all 5000 new updates with the original schedule and streams. Endpoint accuracy is not retention.

## Native and component diagnostics

Native full quality/coverage/accuracy remains FAIL in both arms. Independent holdout precision is 0.219730→0.262610; the original bound is .97. Undefined accuracy shape fields remain unavailable, never zeros or surrogate passes. The separate uncensored audit uses every saved final 20k clean/live draw, with nearest-cell assignment and population covariance. It supplies diagnostics, not a new gate or replacement for the original independent holdout.

| Native saved final-law diagnostic | Local-v2 | Prior-only |
| --- | ---: | ---: |
| Original 20k precision | 0.214350 | 0.259100 |
| Original 20k modes | 15.000000 | 17.000000 |
| Original 20k min_hq_mode_mass | 0.000000 | 0.000150 |
| Original 20k mass_tv | 0.084650 | 0.152300 |
| Original 20k min_cov_eig_ratio | 0.000000 | 0.062586 |
| Original 20k max_cov_eig_ratio | 4.948608 | 3.444655 |
| Original 20k min_radial_median_ratio | 0.000000 | 0.855297 |
| Original 20k max_radial_median_ratio | 2.496600 | 2.440764 |
| Uncensored component_covariance_error | 11.743818 | 8.595152 |
| Uncensored component_min_eigen_ratio | 0.633517 | 0.464993 |
| Uncensored global_spill | 0.785600 | 0.740900 |
| Uncensored max_component_spill | 1.000000 | 0.968085 |
| Uncensored uncensored_radial_ks | 0.799763 | 0.781358 |

| Unequal-width fixed-center diagnostic | Local-v2 | Prior-only |
| --- | ---: | ---: |
| Narrow component 0: center_output_trace_over_target | 5.122888 | 2.864610 |
| Narrow component 0: conditional_jitter_trace_over_target | 0.057369 | 0.212744 |
| Narrow component 1: center_output_trace_over_target | 4.328984 | 0.802564 |
| Narrow component 1: conditional_jitter_trace_over_target | 0.020522 | 0.162395 |

Both narrow center clouds improve in this endpoint census, while their conditional kernel-jitter contributions increase. The average between-conditional-mean fraction falls .993170→.928961. This separates the measured contributor changes; it does not reconstruct or causally attribute the entire training history.

The [saved preselection census](saved-component-diagnostics.json) establishes that local-v2 vector tails are dominated by the center population: the two narrow center trace ratios are 5.122888/4.328984, against jitter .057369/.020522. The nonlinear quadrature mean between fraction is .993170. This supports investigating individual location motion, without attributing the entire training failure to G. The current census reports both arms and keeps fixed center assignment, assignment migration and nonlinear integration approximations explicit. Exact center outputs and finite quadrature covariance algebra are distinct from the approximate Gaussian conditional moments. Vector radial KS uses uncensored whitened nearest-cell radii against χ²₂; overlap can alter that reference, so it is diagnostic only.

**Stop this exact global routing candidate.** The width endpoint benefit is real under the matched law, but rare-density preservation and full temporal gates fail. Retain the source-bound local-v2 positives and their limitations. A possible useful question is how to preserve shared-map allocation while controlling individual center tails, but this study authorizes no second candidate, sweep, extension, seed repeat or promotion. Conditional hosts, images, words/rings and ordinary full Tier2 transfer are unmeasured. Original unsupported two-pole contracts retain their archived BLOCKED identity and are outside this six-task question.

## Source, verification and reproduction

Both arms execute commit `f59b0495c451871a89b777fab890bf72093958cd` and digest `ae77a5d503f693e7ab3d4fbde0c335fa451c6370c5b3be46a741ae7d5cc55648`. Every exercised scientific file and frozen snapshot is verified unchanged. Later edits only clarify diagnostic captions or add publication/reproduction files. The sole consumed global recipe delta is `kinetic_transport_prior_only: false→true`, with local weight1, sliced weight1/32 directions, tail weight0 and backtracking off. Both use the exact winning FullDualNorm/rate configuration; bare historical BCAP Adam is not the control.

All six matched final stream registries, initial models/prior, fixed task laws and resources are equal. Vector data replay verifies all 1200 target batches per arm and their final streams. All three frozen scalar/vector/native adapters reuse one actual real tensor for D/G. No extra forward/draw, target oracle, learned width/weight change or clean-to-noisy serving switch enters the candidate. The five retained local-v2 tasks exactly reproduce their original complete metric trajectories, final model/optimizer tensors and streams; native local-v2 is a new diagnostic cell, not archived qualification reuse.

[Verification](software-verification.json) records **94 distinct checks**, **12 actual-training GIFs** and **432 reproduced scalar/vector primary metric sets**. [Compatible scorer controls](scorer-controls-reuse.json) retain five oracle PASS/five collapse FAIL under their original identities. No generated model samples are added during publication; rescoring recreates the frozen scorer’s deterministic target-reference draws. The 15 tiny software outer updates are a separate fixture cohort, not quality evidence. Memory refreshes use summaries only and all prior qualification/inventory/telemetry blobs remain byte-identical.

Main executed full reservations are **19440 seconds**, within the **21600-second track ceiling** after the declared 600-second saved-analysis and 120-second software allowances. Saved and current static diagnostics add zero optimizer updates or generated sampling draws; their measured method times remain separate from worker costs. The two initial diagnostic refusals occurred before updates and have no instrumented method timing; they remain inside the shared conservative allowance. GPU contention is accounting, not speed superiority. No retry, active subscription reservation, worker or monitor remains. Bulk stdout/JSONL/JUnit/checkpoints/tensor dumps remain on the artifact drive.

| Goal | Local-v2 actual training | Prior-only actual training |
| --- | --- | --- |
| gaussian1d_smoke | [GIF](control-gaussian1d_smoke.gif) | [GIF](candidate-gaussian1d_smoke.gif) |
| gaussian1d_stability | [GIF](control-gaussian1d_stability.gif) | [GIF](candidate-gaussian1d_stability.gif) |
| vector_unequal_mass | [GIF](control-vector_unequal_mass.gif) | [GIF](candidate-vector_unequal_mass.gif) |
| vector_unequal_width | [GIF](control-vector_unequal_width.gif) | [GIF](candidate-vector_unequal_width.gif) |
| vector_two_broad | [GIF](control-vector_two_broad.gif) | [GIF](candidate-vector_two_broad.gif) |
| grid100 | [GIF](control-grid100.gif) | [GIF](candidate-grid100.gif) |

[Media input receipts](media.json) certify the saved observations. Reproduce reporting without training:

```sh
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/component_tails/round5/publish.py
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/component_tails/round5/analyze.py
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python -m experiments.forge compile --summaries-only
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/component_tails/round5/verify.py
```

The frozen declarations below retain the pre-run predictions and stopping rules.

## Preregistered rationale and protocol

This bounded diagnostic tests one global trainer change against **exact retained
local-v2**, with tail weight zero and finite backtracking disabled. No ordinary
qualification or default promotion follows. The parent owns the current goal
leaderboard; this report contains only the scoped matched comparison.

## Evidence before selection

[Saved diagnostics](saved-component-diagnostics.json) verify original requests,
checkpoint bytes and state digests. They enumerate every output G(c_i), then
compute nonlinear conditional kernel moments with five Gauss-Hermite nodes per
latent axis (625 per four-dimensional vector kernel, 25 per native kernel).
The center population is exhaustive; Gaussian integration is approximate.
Assignments are fixed by center output for this separate diagnostic, with
weighted migration reported. The actual served-law metrics and uncensored full
gates remain authoritative. No RNG advances or training updates occur.

Local-v2's unequal-width narrow components have center-output trace ratios
**5.122888 and 4.328984**, against conditional jitter trace ratios **.057369 and
.020522**. Between-kernel conditional means account for **99.317%** of the
average fixed-assignment covariance trace. Broad and rare-mass averages are
99.803% and 98.675%. In native grid100 the average between fraction is only
44.625%, so its kernel deformation remains substantial; this is a distinct
guardrail, not evidence that one center controller must repair every task.

[Round-four tail-moment collapse](../../transport_tails/round4/README.md) and
[role-motion diagnostics](https://github.com/255BITS/ParticleGAN/blob/871b76af18388e70595db1eb9d685f7a8de8af70/reports/forge/bcap-physics/role_motion/round4/README.md)
retain their original identities. The original finite G/prior proposals are
hash-bound in the new census: the tail probe freezes the critic, reuses the last
consumed target batch and draws the next latent batch from cloned final streams.
It is not the next complete D/G update. The role-motion study separately retains
actual next-update probes. Neither attributes every actual training update.
Two early diagnostic scripts refused before
updates (latent dimension and nested native artifact-path assumptions); the
corrected exhaustive analysis took 1.527 seconds; the compact-archive pass took
2.419 seconds. Both and the initial refusals share one 600-second allowance.

## One mechanism and its limits

Set `Recipe.kinetic_transport_prior_only=True`. For the **identical** consumed
G-phase real tensor, fake tensor and sliced/local scalar losses, backpropagate
the ordinary adversarial plus declared prior-regularization term to G and prior,
then restrict the existing transport backward call to trainable prior parameters.
The default false path retains the original single backward call. This changes
gradient routing only: no new center target, labels, covariance, sigma, geometry,
extra forward, draw, optimizer, rate, architecture, or serving law enters training.
The prior retains fixed kernel width/uniform weights and learnable locations.

The scientific hypothesis is that individual learned locations can rearrange
the dominant center population without forcing the shared map to absorb that
same density signal. Its latent pullback still depends on the G Jacobian;
removing G's transport force may damage initial allocation or adversarial
stability. Finite normalized updates need not descend or converge.

[Liutkus et al. (2019)](https://proceedings.mlr.press/v97/liutkus19a.html) study
nonparametric sliced-Wasserstein measure flows. Their direct particle/diffusion
algorithm motivates separating particle motion from a shared network but does
not establish a theorem for our normalized latent pullback. The conditional
covariance decomposition follows by expanding around each conditional mean;
its algebra is exact for the finite quadrature law, while the nonlinear Gaussian
moments and fixed component assignments remain approximations to serving.

## Frozen scope, forecasts and stopping

Ready schema-v3 candidate `component_prior_transport_r5`, studies
`component_tails_candidate_round5` / `component_tails_control_round5`, and
diagnostic view `component_tails_round5_diagnostic` admit exactly two matched
arms under campaign `component_tails_round5`. Sliced-v1 is the control's admission
reference only; it is not a third arm. Both global recipes retain winning BCAP
DualNorm, smoothing .001/momentum0/per-offset, non-saturating loss, G .012,
D .018, prior .030, coefficient/cap1, zero additive training noise/EMA,
sliced weight1/32 projections and local weight1. Only the routing switch changes.

| Original task | Per-arm full reservation | Forecast and authoritative outcome |
| --- | ---: | --- |
| Gaussian smoke | 120 s | Retain independently confirmed full acquisition PASS; finish 1000 updates |
| Gaussian stability | 600 s | Restore own eligible smoke state; full retention and deadline reacquisition required |
| Unequal mass | 1800 s | Preserve sustained rare-density PASS and terminal suffix≥5 |
| Unequal width | 1800 s | Primary final full covariance≤1.25, >1.25 falsifies forecast; original sustained gate≤.85 unchanged |
| Two broad | 1800 s | Preserve full sustained PASS |
| Grid100 | 3600 s | Complete original 7000 updates and all five terminal quality/coverage/accuracy checks |

Missing modes/counts, mass TV, uncensored covariance, core covariance, spill,
eigen/radial shape and temporal suffix must all be reported. An endpoint scalar
forecast never replaces the complete gate. All independent runnable jobs complete;
a failed own smoke blocks only its required continuation.

Full main reservations are **9720 per arm / 19440 total**, plus **600 seconds**
for all saved analysis attempts and **120 seconds** for tiny software checks:
20160≤21600 track ceiling. No capacity test, sweep, second candidate, seed-only
run, extra-budget continuation or automatic promotion is authorized. Remaining
1440 seconds can cover only an execution repair whose full allowance fits.
GPU contention measures accounting rather than speed superiority.

Protocol seed0/public deterministic initializer, actual seen data batches,
architecture/target, prior law, committed update limits and evaluation cadence
are matched. Constructor, target, training/noise and evaluation streams are
isolated/checkpointed. The frozen Gaussian/vector adapters reuse one actual real
tensor for D/G; native retains its own frozen data law. Clean/live grading stays
separate from noisy/EMA. Existing task cards and qualification snapshots remain
unchanged. No fixed identity/zero cohort is silently substituted.

## Execution and publication

Scientific source and ready declarations are pushed before paid training. Public
Queue/drain runs disable full-compilation callbacks and stop after this campaign.
Logs/checkpoints/JSONL stay on the artifact drive. Compact receipts, complete
metrics and actual-training GIFs will be published here, even for a negative result.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/component_tails/logs/driver.log
```
