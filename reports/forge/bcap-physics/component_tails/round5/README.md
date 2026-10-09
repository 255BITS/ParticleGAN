# Round 5: component centers and kernel jitter

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
retain their original identities. The exact original finite G/prior next-batch
probes are hash-bound in the new census. They are endpoint probes, not attribution
of every actual training update. Two early diagnostic scripts refused before
updates (latent dimension and nested native artifact-path assumptions); the
corrected exhaustive analysis completed in 1.527 seconds under one shared
600-second allowance.

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
