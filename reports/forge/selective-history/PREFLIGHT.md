# Selective generator history: frozen preflight specification

Registered before candidate saved-state outputs on 2026-10-15. This implements
the user-approved [review and staged plan](/ml2/hypergan/combination-review-20261014/PLAN.md).
Base commit: `4728b8b030237aa00ef2188a8782cea247991c32`. No trained pass is
implied by software checks or counterfactual probes.

## Frozen equation

The original source manifest is
`9bc9e6ae14ccd13455bf8805bd68b9e3affd6e09e112a06be7a97b368248d61f`;
`gradient_signal.py` has SHA256
`b7fd16b80604e7b0982aa426f0723b95ffa1d0637375a8e933598f2fbe16fdeb`.
It attenuates the normalized update through the learning rate. Its role-level
float64 estimator uses decay 7/8, bias-corrected means, effective sample count,
unbiased variance, debiased nonnegative signal and fixed denominator epsilon
1e-16. Preserve this exact recurrence; means estimate signal and never replace
the current direction.

The new **gradient-space adaptation** is explicitly different:

- B: current direction-only BCAP/DualNorm.
- T: B plus global and local-v2 transport weights 1/1, 32 projections.
- W: T with `g = rho(total_history, total_gradient) * total_gradient`.
- S: T with `g = rho(adversarial_history, adversarial_gradient) *
  adversarial_gradient + (total_gradient - adversarial_gradient)`.

W and S modify only effective generator-player slots, including direct generated
coordinates, excluding encoder and learned latent-prior slots. Both apply before
ONE existing optimizer proposal; direction blending uses the original current
protected normals. New runs start with empty estimator state. No history from
whole-gradient checkpoints may initialize S. No task-specific switching.

## State roster and history construction

Freeze original checkpoint identities from
[source recovery](/ml2/hypergan/combination-review-20261014/EXECUTION-SOURCE-RECOVERY.json)
and [missing originals](/ml2/hypergan/combination-review-20261014/EXECUTION-MISSING-ORIGINALS.json).
Required windows are Gaussian acquisition step1000, stationary/pre-shift4000,
shifted-hold6000, unequal-mass1200, late two-pole, failed late Ring, and
conditional trajectory/residual. Include an explicitly synthetic repeated-real
zero-variance numerical panel. Synthetic panels never fill missing original
windows. Recover exact original states before selecting any additional window.

The retained Gaussian/rare originals have no gradient time series. For each
available state, hold model parameters and original optimizer state fixed and
observe **32 successive original-size training batches**, starting at its saved
named RNG streams. Only cloned observation streams and empty diagnostic history
advance. Do not update G, D, prior, scheduler or original estimator. Use the
original target law at each saved phase, including the shifted target after its
original switch. Original target labels/evaluator assignments cannot enter the
loss. Record exactly consumed stream states and batches. This fixed-state batch
panel measures gradient noise and update-map behavior, not temporal training
history or a reconstructed historical trajectory.

Use each of the 32 panels for a separately restored counterfactual T/W/S update;
no counterfactual becomes the next panel's initial state. Histories advance only
from their designated frozen-state observed channel. Preserve original packets
read-only. An explicit state mapping must distinguish this intervention from a
strict original continuation. Historical post-normalization attenuation may be
reported as a separate reference, never called W.

## Measurements and acceptance

Record adversarial, transport and remaining gradients; channel norms and
pairwise inner products; rho, signal, variance and effective count; pre-map
gradient, normalized proposal, blended update and actual rounded displacement;
original protected-gradient inner products; fresh transport and identity
objective values before/after finite motion; output mean/covariance; role map;
and any nonfinite values. Preserve per-panel data outside Git with compact
summaries and hashes in the report. No best-panel selection.

1. Verify exact source/artifact hashes and original constructor/load round trip.
   Restore twice, compute the same next-update operation on independent clones,
   and require identical tensors/counters/streams on the declared runtime.
   These disposable one-step diagnostics are not campaign training or grades.
2. Require default-off parity, complete active checkpoint/resume, exact recovered
   estimator parity, unmodified raw protected normals, and explicit player roles.
   Missing decomposition or ambiguous shared ownership is unsupported.
3. Require finite gradients, estimator values, update maps and outputs, including
   the degenerate panel. A nonfinite panel rejects admission.
4. Require S's effect to survive into actual rounded generator displacement on
   at least one noninitial real-state panel. Report the number of identical
   T/S panels and displacement ratios on every window. If every actual map is
   identical, reject the coupling. W's scalar cancellation is a control result,
   not evidence for S.
5. Reject a panel that reverses a fresh useful correction: where T strictly
   descends a protected identity or transport gradient, S may not ascend it
   beyond `64 * dtype_epsilon * ||gradient|| * ||displacement||`. Also report
   finite objective increases separately; first-order descent is not a finite
   improvement guarantee. A failing window rejects advancement; do not tune the
   equation or delete the window.
6. Missing required original windows, failed strict restore or unexecuted checks
   leave preflight **BLOCKED**, never PASS. Available-state diagnostics can still
   complete. They cannot authorize the paid pole/Ring screen by themselves.

## Budget and progression

Software and disposable probes share the approved 1,800-second execution cap;
record process wall time separately from development and recovery elapsed time.
No campaign training is authorized until all required preflight gates pass.
Then the first stage reserves exactly four arms times (300-second pole plus
300-second Ring) = 2,400 seconds, with both independent jobs completed per arm.
Only S passing both complete original gates advances. The remaining conditional
science ceiling is as declared in the approved plan: 108,600 seconds total,
including this screen. Do not spend unused capacity on new equations or seeds.
