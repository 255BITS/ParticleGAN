# Alchemy: conserved soft-cell moment distillation

This is a preregistered round2 mechanism diagnostic. **No trained result yet.**
The parent owns the single current goal leaderboard. This report preserves one
candidate, the exact winning primary control, and complete original task gates.
No ordinary qualification, calibrated screen or default-adoption claim follows.

## Mechanism and scope

The analogy is separation and recombination of conserved material. The actual
mechanism is a finite, label-free distribution witness, not a physical reaction.
The critic continues supplying the winning non-saturating adversarial force;
an additional real-batch moment frame makes local errors directly differentiable.
It is distinct from a travel limiter, quantile pairing, projection or pairwise
kernel MMD. Finite moment matching is established prior art: [GMMN](https://proceedings.mlr.press/v37/li15.html)
and [McGAN](https://proceedings.mlr.press/v70/mroueh17a.html) motivate distribution
and mean/covariance signals respectively. We claim a particular conservative
soft-cell decomposition and normalization, not invention of moment matching.

For a batch of n real vectors in dimension d, pick K=min(8,n) farthest-first
anchors starting at its first row. Let S=E_P||x-E_P x||², with S=1 for an exact
zero-variance batch, and T=S/(4K). Freeze real-derived
r_k(x)=softmax_k(-||x-a_k||²/T), so sum_k r_k(x)=1.
Set p_k=E_P r_k, mu_k=E_P[r_k x]/p_k,
s_k²=max(E_P[r_k||x-mu_k||²]/p_k, S/(16n²)),
u_k=(x-mu_k)/s_k, C_k=E_P[r_k u_k u_k^T]/p_k.
Tiny positive mass floors handle floating-point underflow; numerical residual
subtraction gives identical empirical batches an exactly zero loss.

The G-phase generated batch Q defines m_k=E_Q r_k and residuals
b0_k=m_k-p_k, b1_k=E_Q[r_k u_k]-E_P[r_k u_k],
b2_k=E_Q[r_k(u_k u_k^T-C_k)]-E_P[r_k(u_k u_k^T-C_k)].
The added loss, at fixed weight1, is

    L = sum_k (b0_k² + ||b1_k||²/d + ||b2_k||_F²/d²) / max(p_k,1/n).

The mass residual conserves signed material exactly: sum_k b0_k=0. In exact
arithmetic b1_k=m_k(mu_Qk-mu_k)/s_k and
b2_k=m_k[(Cov_Qk+(mu_Qk-mu_k)(mu_Qk-mu_k)^T)/s_k²-C_k].
Thus location and conditional second moments are separated from pure occupancy.
Location still contributes to the second moment, and soft cells overlap; the
three residuals are not asserted statistically orthogonal. Real partition
construction stays outside autograd, while fake responsibilities remain
fully differentiable. Empirical equality gives zero loss/gradient. Translation,
orthogonal rotation and common nonzero unit scaling preserve the mathematical
loss when anchor selections are unique and the pooled variance is positive;
ties and the exact atomic fallback qualify this equivariance.

The learned prior remains the same fixed-width, uniform MoG. Every component
retains its weight and identity; only the existing network/prior parameters
receive the combined gradient. No target component labels, analytic centers,
widths, independent data draws, reservoir, row renewal or new RNG are used.
The finite frame is not characteristic: correct mass/mean/covariance can still
hide non-Gaussian tails. Farthest-first extremes, minibatch cell changes and
rare-sample absence can generate noisy or adverse normalized motion.

## Evidence and frozen forecast

Before choosing the candidate, [saved-evidence.json](saved-evidence.json) verifies
original endpoint sample hashes and applies the frame without training or random
draws. Original winner center allocation is 142/94/19/1 for unequal mass and
95/13/108/40 for unequal width. Analytic target information appears only in
separately labelled diagnostic controls, never in the trainer. Empirical null,
collapse, width doubling, shift and missing-component controls probe the witness.
Its numerical values are descriptive and do not replace the task scorers.

Forecast before full training: unequal-width final `component_covariance_error`
<=.85 (primary signature; >.85 falsifies), rare-mode `min_mass_ratio`>=.5,
broad-vector full sustained PASS retained, Gaussian smoke independently confirmed
PASS and final stability `cdf_ks`<=.05. The primary forecast is stronger than an
endpoint mass improvement and could fail even if the decomposition supplies a
useful gradient. Original thresholds and five-terminal-check vector suffix stay
unchanged. These predictions do not relax the full gates.

One global trainer delta: `Recipe.distillation_weight=1`,
`Recipe.distillation_cells=8`; all exact winner overrides retained. The winner
is the saved `bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36`
declaration, not bare preset defaults. Constant G/E .012, D .018, prior .030,
DualNorm smoothing .001/momentum0/per_offset, non-saturating loss,
BCAP cap1/coefficient1/every update, floors1, no additive noise or EMA remain.

The six unchanged tasks are two-pole, Gaussian smoke and its own-state stability,
unequal mass, unequal width, and the genuinely passing two-broad guardrail.
The frozen two-pole public-components fixture cannot consume the new trainer
signal and stays explicitly BLOCKED for the candidate. The control executes its
original separate zero/stored-weight fixture. No native100 task fits alongside
this complete subset: each vector task reserves1800s, Gaussian120+600s and
fixture300s, totaling6420s per arm /12840s declared; the fresh ceiling14400s
leaves1560s for diagnostics or genuine errors, not another candidate or sweep.
No claim extends to image, conditional, native or full Tier2 qualification.

Protocol seed0, public deterministic initializer, architecture, prior widths,
weights and components, task laws, update budget, evaluation cadence and batch
sequence stay fixed. Forge's frozen vector/Gaussian adapter reuses one actual
real tensor for D/G; that historical law is preserved. Constructor, data,
training-noise and evaluation streams stay isolated and checkpointed. The
witness is stateless and consumes no RNG. Live clean sampling is authoritative.

## Execution and review

Ready declarations: [candidate](../../../../configs/forge/ideas/alchemy_distillation_r2_v1.json),
[candidate study](../../../../configs/forge/studies/alchemy_distillation_r2_candidate_v1.json),
[control study](../../../../configs/forge/studies/alchemy_distillation_r2_control_v1.json),
[diagnostic view](../../../../configs/forge/views/alchemy_distillation_r2_diagnostic.json).
Source and declarations will be pushed in a draft PR before full matched runs.
Queue uses a frozen snapshot and `on_completion=None`; no full reducer or
historical qualification rewrite. Both arms are enqueued before the bounded
one-worker shared-GPU drain. Publication will render actual-training GIFs and
retain complete final metrics, source receipts, costs and failures.

Tail all research logs:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/alchemy/logs/drain.log
```

Bulk logs, raw streams and checkpoints stay outside Git. Reproduction uses the
shared Python3.12 environment with this worktree on PYTHONPATH. Deterministic
witness controls and public checkpoint/stream continuation are covered by
`tests/test_distillation.py`; full training will decide scientific success.
