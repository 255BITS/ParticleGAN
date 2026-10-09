# Round 4: tail-sensitive local transport

This separately registered bounded diagnostic compares one successor of
**local-v2** with **local-v2 as the single primary matched control**. The sole
global recipe delta is `kinetic_transport_tail_weight: 0 → 1`; finite
backtracking remains disabled. No result confers ordinary qualification or
default-adoption credit. The parent owns the one current goal leaderboard.

## Saved evidence and mechanism

The [round-two repair](../../kinetic_transport/round2/README.md) and
[failed round-three backtracking](../../kinetic_transport/round3/README.md)
retain their original identities. The new
[saved endpoint probe](saved-tail-diagnostics.json) checks all original receipt
and checkpoint hashes, replays final consumed target batches and verifies their
final named data streams. It examines six v2/v3 vector checkpoints, all saved
served samples, and one additional 128-row draw per checkpoint from cloned final
streams. Six in-memory optimizer proposals add **zero training updates**, consume
768 probe draws, and take **2.550579 GPU seconds**. These are separate frozen-state
cohorts, not reconstructed final training updates.

Local-v2 has rare centers 6/256 versus v3's 7/256. Its full width covariance error
2.480438 greatly exceeds its core error .289171; v3's full error worsens to
3.859733 while its core improves to .179413. The saved rare-component tail
kernel force norm is .0000595 against sliced .0078963, and only 43.75% of local
tail forces point inward. The total rare-tail force points outward on every
saved sample in this one final-batch probe. Generator-only width motion has
finite curvature remainder .129247, versus .012133 for prior-only motion;
joint remainder .172057. Signs and magnitudes depend on this batch and critic.
They support investigating tail sensitivity, without establishing the cause of
the trained failure or asserting that every actual step overshoots.

For detached real anchors `y_i`, let `h_i²` be their fourth-other-neighbor squared
distance (or `n−1` for fewer than five samples), with the existing numerical
variance floor; let `b² = median_i h_i²`. Define

```
a_i(x) = (1 + ||x−y_i||²/b²)^−2
w_i(x) = a_i(x) / sum_j a_j(x)
phi_i(x) = w_i(x) ||x−y_i||²/h_i²
p_i = mean_real phi_i, q_i = mean_fake phi_i
L_tail = mean_i ((q_i−p_i)/(p_i + 1/n))²
```

Rational normalized weights are evaluated as a log-weight softmax. Along an
escaping ray, each weight tends to `1/n`, so each feature grows quadratically.
This mathematical property prevents the entire feature vector vanishing for
remote samples; it does **not** guarantee correct component covariance,
convergence, inward force on every row, or finite-step descent. Anchors,
neighbor widths, bandwidth and reference moments are detached. Fake weights
remain differentiable. Only the same real/fake training tensors are consumed.
There are no extra draws, labels, evaluator centers, known target covariance,
prior changes, stateful statistics or serving changes.

[Gretton et al. (2012)](https://www.jmlr.org/papers/v13/gretton12a.html) motivate
matching feature expectations; this finite data-dependent normalized feature
family supplies no characteristic-kernel or population test guarantee.
[Srinivasan et al. (2025)](https://proceedings.mlr.press/v267/srinivasan25a.html)
show that explicitly polynomial Stein features can detect selected moments for
Gaussian targets. Their score-based theorem does not apply here; it motivates
distinguishing finite moment detection from distribution convergence. The
mechanism and its escaping-ray property above are our own diagnostic hypothesis.

## Frozen comparison, predictions and stopping

Ready candidate `transport_tail_moments_r4`, studies
`transport_tails_candidate_round4` / `transport_tails_control_round4`, campaign
`transport_tails_round4`, and view `transport_tails_round4_diagnostic` freeze
one complete global configuration per arm. Both arms retain the exact archived
winner rates, DualNorm smoothing .001, per-offset convolution, zero momentum,
non-saturating adversarial loss, prior regularization zero, BCAP coefficient/cap1,
local-v2 weight1 and sliced weight1/32 directions. Bare historical BCAP Adam is
not used. Archived winner and v3 results provide context only.

| Unchanged task | Per-arm reservation | Prediction and authoritative gate |
| --- | ---: | --- |
| Two-pole | 300 s declared, no launch | Genuine public-component capability blocker; fixed identity/zero cohort is not replaced |
| Gaussian smoke | 120 s | Preserve own full confirmed acquisition PASS |
| Gaussian stability | 600 s | Improve stationary/shift passing counts over 28/72 and 11/24; full retention/deadline gate remains authoritative |
| Unequal mass | 1800 s | Preserve full sustained PASS and terminal suffix≥5; report rare mass/core/full covariance/spill separately |
| Unequal width | 1800 s | Full final covariance error≤1.25 (original gate≤.85); >1.25 falsifies primary forecast; report narrow spill and eigen/core tradeoffs |
| Two broad | 1800 s | Preserve full sustained PASS |

Full declared reservations are **6420 seconds per arm / 12840 total**; two blocked
cells spend zero, so runnable reservations total 12240 if both own smokes pass.
The conservative **120-second saved-probe allowance** and software costs fit
the **14400-second track ceiling**. No capacity run, sweep, seed repeat, second
candidate, unchanged continuation or automatic promotion is authorized.

Protocol seed0, public deterministic initialization, target/data law, seen
batch sequence, architecture, prior width/mass/locations, update allowance,
sampling and scoring cadence are fixed within each task. Frozen scalar/vector
adapters reuse one actual real tensor for D and G. Constructor, data, training
noise and evaluation streams are isolated and checkpointed. Stability restores
each arm's own exact eligible smoke state; a failed own smoke blocks it.
Clean/live scores are separate from noisy/EMA. Run every independent admitted
job at the unchanged full budget; stop and report this revision after completion.
Unsupported cells remain visible and never grant qualification credit.

Competing explanations are fluctuating anchors, sparse-batch amplification,
rational partition leakage across components, and normalized shared-generator
motion. A tail moment can improve the endpoint while harming core eigenvalues,
rare allocation or the temporal suffix. Favorable surrogate behavior cannot
replace the full gate. No convergence or physical analogy claim is made.

## Execution

Scientific source and ready declarations are pushed before enqueue. Both arms
execute on the same scientific source/runtime through public ParticleGAN API.
Worker logs, per-update streams, checkpoints and tensors stay under the local
artifact root; compact receipts and actual-training GIFs will be published here.
Public `Queue(..., on_completion=None)` and bounded `drain(..., allow_sharing=True,
watch=False)` prevent full-compile callbacks from regrading archived evidence.
GPU sharing affects costs, not optimizer-speed evidence.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/transport_tails/logs/driver.log
```
