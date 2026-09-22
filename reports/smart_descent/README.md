# Smart descent — expanded experiments

**A learned feedback addition to cosine now passes all 29 live bounds and
sustains all nine behavioral toys. It is not a reliable replacement for cosine
or Adam.** Direct fitting without a clock did not produce a policy that sustains
all nine toys. Per-tensor gradient descent improved substantially over ordinary
SGD, but no tested SGD rule passed both of its development tasks.

Keep the supported cosine baseline and production defaults. Retain the learned
policy as a research comparison: its feedback has a measurable benefit on the
development suite, and it succeeds on one new architecture/task where cosine
fails. It does not dominate the transfer cases or establish a wall-time speedup.

## Full behavioral leaderboard

All rows use the same cap formulation, original host budgets and seed 0. Live
weights determine scores; EMA is separate. The ten independent shared checks
all pass. Configuration/posture checks are excluded.

| Controller | Live bounds | Sustained toys | Ring modes / HQ | Ring confirmation | Mean confirmation / budget |
| --- | ---: | ---: | --- | ---: | ---: |
| Cosine baseline | 29/29 | 9/9 | 8/8 / 100% | 1,050 | 0.5850 |
| **Learned feedback + cosine** | **29/29** | **9/9** | **8/8 / 100%** | **1,150** | **0.5712** |
| Same coefficients, feedback disabled | 29/29 | 8/9 | 7/8 / 100% | — | — |
| Same coefficients, cosine removed | 27/29 | 8/9 | 5/8 / 66.77% | — | — |
| Constant LR control for direct fitting | 29/29 | 8/9 | 8/8 / 100% | — | — |

The original regression gate permits seven ring modes. Sustained success
requires all eight modes, HQ≥90%, and at least five final passing observations
in the complete 24-point curve. A final PASS alone is insufficient.

The selected policy is `lr_g01_p05`. Relative to cosine, trajectory confirmation
improves from 167 to 134 and residual-student confirmation from 167 to 117, but
ring confirmation moves from 1,050 to 1,150. The other six confirmation steps
are unchanged. Mean normalized confirmation improves **2.36%**; the ring itself
gets slower. This is a tradeoff, not a win on every speed metric.

Measured total suite time is 21.24s for cosine and 21.18s for feedback; summed
time to confirmation is 13.95s versus 13.73s. The numerically identical LR-only
ablation takes 20.63s total, a larger variation than the apparent gain.
**These timings do not establish a dependable speedup.** They include setup,
metric evaluation and controller work. Later checkpoints were still evaluated
to verify stability; no early-stop rule is implemented.

[Full comparison, EMA, timing and curves](frozen_evaluation/README.md) ·
[Runnable policy](frozen_evaluation/policy.json) ·
[Reviewed fitting/evaluation provenance](frozen_evaluation/provenance.json).

## Fresh transfer after policy freeze

The following task/architecture combinations were declared before search and
first evaluated after the final challenger was frozen. No transfer score was
used to select or refit that policy. Each task has 1,600 updates.

| Task | Cosine: modes / HQ / sustained | Feedback: modes / HQ / sustained | Feedback disabled: modes / HQ / sustained |
| --- | --- | --- | --- |
| Six-mode ring, width64/depth2 | 5/6 / 74.12% / No | **6/6 / 90.99% / Yes** | 6/6 / 92.16% / No |
| Sixteen-mode grid, width128/depth3 | 12/16 / 77.56% / No | 13/16 / 76.76% / No | **16/16 / 90.06% / Yes** |
| Eight-mode ellipse, width64, R1+R2 | 7/8 / 100% / No | 8/8 / 93.33% / No | 6/8 / 81.88% / No |

Feedback sustains the six-mode ring from step 1,134, confirmed at 1,400; its last
eight scheduled observations pass. The ellipse has only four final passing
observations, below the required five. Disabling feedback sustains the grid
instead. Both feedback and its ablation therefore sustain **1/3** transfer
tasks, on different tasks; cosine sustains 0/3. Feedback does not show a uniform
transfer advantage.

## Searches and negative results

| Study | Tested scope | Outcome |
| --- | --- | --- |
| [LR plus regularization feedback](multi_control/README.md) | 34 distinct policies, ring screening followed by full suite | Three nonzero policies sustain all nine; all slower by normalized confirmation than cosine |
| [LR-only feedback on cosine](lr_only/README.md) | 21 new policies plus reused cosine control; every candidate runs all nine | Three nonzero policies sustain all nine; selected candidate shown above |
| [Direct fit without a clock or cosine](standalone/README.md) | 22 distinct policies, including constant and warm-start controls; every candidate runs all nine | No policy sustains all nine; constant LR remains the search winner |
| [Stronger clock-free damping](damping/README.md) | 12 additional responses to gradient growth/innovation, applied to G, D or both | None sustains the ring; the other eight toys are screened and cannot earn a PASS |
| [Common-LR SGD](sgd/common_lr/README.md) | 16 G/D rate pairs × constant/cosine × two tasks; then 16 learned policies × two tasks | 36/64 rate-sweep episodes become nonfinite; learned search selects zero coefficients |
| [Per-tensor descent](sgd/per_tensor/README.md) | 16 G/D relative-step pairs × two tasks | All finite; coverage improves substantially, but no shared sustained pass |
| [Per-tensor RMS memory/decay](sgd/rms_decay/README.md) | Eight conditions × two tasks | Cosine yields a sustained ring4 pass while grid9 fails; no condition sustains both |

The Adam searches examine **55 distinct cosine-backed policies** in the first
two stages and **33 nonzero clock-free candidates** in the later fit/probes.
The latter tests go beyond merely removing cosine from a policy fitted with it.
They use no progress feature: learning rates respond only to gradient history.
Their two newly reserved transfer cases were not evaluated because no learned
clock-free candidate qualified on all nine development toys.

These results limit the tested controller families and search ranges. They do
not prove that a larger learned optimizer, richer observations, or a different
training procedure cannot work.

## What the controller learned and what remains unresolved

The controller is a small equation with G/D outputs, positive bounded LR
multipliers, and a moving mean gradient. It observes gradient RMS change,
alignment, innovation, and the opposing role's last normalized gradient RMS.
It receives no task identity, formulation name, mode label or evaluation score.
The multi-control version can also set regularization strength for the next
loss computation; the selected policy has zero regularization coefficients.
All coefficients are fitted by black-box parameter search, not by backpropagating
through an unrolled GAN.

On the full development suite, suppressing feedback loses a ring mode. Thus the
new experiment establishes a local benefit from feedback that the first study
did not establish. The transfer tradeoff and dependence on cosine remain.

The SGD diagnostics expose a scale problem: at initialization a shared LR gives
the generator output bias roughly 2,700 times the relative update of its first
weight matrix. Per-tensor rates address that disparity and improve coverage.
They do not resolve the shared-configuration problem. This is evidence about
the update scales, not proof that they alone caused the original collapse.

## Dataset, source and reproduction

Every attempted policy, failure, feature/action trace, metric curve and separate
EMA result is retained. Large JSON files are compressed without changing their
original bytes; [archive_manifest.json](archive_manifest.json) records their
SHA256s. Each phase has an exact source bundle. The SGD phases have their own
manifests. Parent snapshots and duplicated controls are explicitly identified;
they are reused records, not independent training examples.

These are observations, actions and outcomes. They are not resumable model
checkpoints or labels for the optimal action. All GAN initializations use seed 0;
only the separate controller-proposal RNG changes between search stages.

The original frozen evaluation retains its original metadata and sources. A
review subsequently tightened future freeze records to bind the selected
search's transfer declaration and separate fitting, numerical and evaluation
source hashes. The reviewed provenance file selects exactly the same weights;
no numerical training or evaluation result was replaced.

[Method and runnable commands](../../benchmarks/smart_descent/README.md) ·
[Raw-gradient SGD method](../../benchmarks/learned_lr/SGD_README.md) ·
[Original learned-LR study](../learned_lr/README.md).

Validation: **17 contract tests pass**, covering exact cosine/Adam parity,
unscaled SGD updates, causal regularization, scalar RMS memory, feedback
ablation, group ratios, host hook restoration and freeze provenance. The new
cosine control reproduces all nine historical final metrics exactly. The
existing unrelated particle-native CI failure is documented in
[ci_status.md](../ci_status.md); its threshold remains unchanged.
