# Alchemy: conserved soft-cell moment distillation

**Reject this exact global candidate: it loses the passing broad-vector guardrail
and leaves both difficult mixture tasks at zero passing observations.** The
candidate records **1 PASS / 4 FAIL / 1 BLOCKED**, versus the source-matched
winner's **3 PASS / 3 FAIL**. On five mutually executable tasks the comparison
is **1/5 versus 2/5 PASS**; the candidate's only PASS is Gaussian acquisition.
There is no new sustained passing task. All 11 runnable jobs finish once for
**638.740393 paid worker seconds**, with zero retries or remaining reservations.

This is a completed round2 mechanism diagnostic. The parent owns the single
current goal leaderboard; this task table is a readout. No ordinary qualification,
calibrated screen or default adoption follows. The original winner retains its
source-bound [7/21 Tier2 result](../../technique-inventory.md).

## Full unchanged gates and measured results

| Task | Matched winner | Distillation candidate | What the complete measurement establishes |
| --- | --- | --- | --- |
| Two-pole | PASS, 17/24 checks | BLOCKED | Original public-components fixture does not consume this signal; no replacement or paid attempt |
| Gaussian smoke | PASS, 3/24 confirmed states | PASS, 10/24 confirmed states | First confirmed step375→167; endpoint CDF KS .071842→.042492; any confirmed acquisition supplies smoke PASS |
| Gaussian stability | FAIL: stationary2/72, shifted hold0/24 | FAIL: stationary13/72, shifted hold4/24 | Deadline reacquisition FAIL in both; endpoint KS .320623→.088941 still exceeds .05 |
| Unequal mass | FAIL, 0/24, suffix0 | FAIL, 0/24, suffix0 | Covariance3.691653→.619860, but minimum mass ratio .207520→0 and counts3574/522/0/0 |
| Unequal width | FAIL, 0/24, suffix0 | FAIL, 0/24, suffix0 | Covariance6.287559→.609777 satisfies the primary forecast; mass TV .291504→.527344, counts0/0/912/3184 and minimum eigen ratio0 fail |
| Two broad, passing guardrail | PASS, 22/24, suffix22 | FAIL, 0/24, suffix0 | Counts2306/1790→4096/0, mass TV .062988→.5, SW1 .134590→.568760 and minimum eigen ratio0 fail |

[Final metrics and temporal gate failures](results.json),
[exact source/recipe/data/initialization proofs](provenance.json),
[reused byte-verified scorer controls](scorer-controls.json),
[deterministic saved-center/batch diagnostics](saved-state-diagnostics.json),
and [11 actual-training GIF receipts](media.json) preserve the result.
The [candidate broad-vector GIF](candidate-vector_two_broad.gif) illustrates its
numerically established regression; media adds no training or qualification.

### An observed forecast does not rescue a failed mechanism

The preregistered unequal-width covariance forecast is observed (.609777≤.85),
while the rare-mass≥.5, retained broad-vector PASS and final Gaussian KS≤.05
forecasts fail. Gaussian independently confirmed acquisition is observed.
The covariance average alone can be below .85 while two components are empty:
the unchanged minimum-eigenvalue and mass gates catch that failure. Full mixture
bounds fail at all24 observations in every candidate vector task. Neither better
conditional shape among surviving modes nor HQ1.0 supplies distribution coverage.

The [posthoc witness diagnostic](final-witness-diagnostics.json) applies the same
finite frame to saved endpoint samples using a separately labelled deterministic
4096-point target batch. Unequal-width shape residual falls .827296→.020229 while
mass residual rises .650280→1.742366. Broad-vector mass residual rises
.257052→1.091849. Thus even this witness's own occupancy term exposes errors
that the trained combined dynamics did not correct. Candidate mass-gradient RMS
is nonzero (.000916 for width, .001105 for broad), so a vanished total mass force
is not established. These are output-space diagnostic derivatives, not actual
last-minibatch or network/prior gradients, and establish no causal attribution.

For a fixed frame, L is a squared finite-feature mean discrepancy, equivalently
an MMD with its induced finite-rank kernel. Its specific separation into soft
occupancy and centered polynomial moments distinguishes the implementation from
pairwise Gaussian-kernel matching; it is not a new general discrepancy family.
The mass derivative uses grad r_k=(2r_k/T)(a_k−sum_j r_j a_j), making an empty
faraway cell's correction potentially weak. That algebra motivates a competing
explanation, not proof of the observed training cause. Adaptive minibatch frames,
adversarial conflict and normalized shared-network response remain unresolved.

**Recommendation: stop weight1/cells8 as a global repair and retain the winning
control.** Preserve the opt-in code and negative evidence for review. Inspect
actual saved parameter-space mass/location/shape gradients before considering
any separately bounded successor; this report authorizes no new candidate,
tuning, seed repeat, continuation or adoption. Native100, images, conditional
objectives and the full ordinary Tier2 suite were not measured.

The automatic candidate study decision remains `incomplete` because its declared
two-pole cell is unsupported, while its primary numerical prediction is observed.
This is compatibility incompleteness, not an incomplete worker. The control
study's same covariance signature is falsified by6.287559; it does not invalidate
the winner's original archived qualification. Frozen forecasts and studies stay
unchanged; their [pretraining source](https://github.com/255BITS/ParticleGAN/blob/e5b607481eb5f2cbb2e09781acbefd678192f8e3/reports/forge/bcap-physics/alchemy/README.md)
remains reviewable.

## Mechanism and scope

The analogy is separation and recombination of conserved material. The actual
mechanism is a finite, label-free distribution witness, not a physical reaction.
The critic continues supplying the winning non-saturating adversarial force;
an additional real-batch moment frame makes local errors directly differentiable.
Its finite polynomial frame differs from a travel limiter, quantile pairing,
projection or pairwise Gaussian-kernel implementation. Finite moment matching is established prior art: [GMMN](https://proceedings.mlr.press/v37/li15.html)
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
[PR363](https://github.com/255BITS/ParticleGAN/pull/363) was opened as a draft and
scientific commit `e5b607481eb5f2cbb2e09781acbefd678192f8e3` was pushed before
both arms were enqueued. Both executed source digest
`587006be44148b1a2adfd905bd01d5631061ed4206c27c85e7751cc2a3f29d5a`;
all 1,201 frozen scientific source files still match publication bytes.
Candidate revision `523df0e2d667fb2b1fb59e80127bfb96525a7d8c6c0ce705bb3122de7c05d377`
and matched control revision `b80c8204aa299cd190869dd05ff26b422786499213476f2c67d8df084e889945`
are new-source identities. Winner measurements reproduce their archived endpoints
but form this new-source matched control, not an independent seed replication.
Publication commits change reporting and the local runner only; no unmeasured
scientific fix is presented as trained.

Declared reservations are 12840s; executed full allowances are 12540s because the
candidate's unsupported fixture spends nothing. Paid workers total 638.740393s.
Deterministic CPU diagnostic costs appear separately in their receipts; no extra
scientific training, capacity probe or paid retry was launched. GPU0 stacking
was authorized, with one active worker; contention supplies no optimizer speed
claim. The bounded drain has stopped with zero active reservations/watchers.
It used `on_completion=None`. Readouts use summaries-only compilation and retain
original qualification. Generated goal boards remain parent-owned.

154 focused software checks pass, including the public trainer, same-stream
matched arms, checkpoints, analytic null/collapse controls, finite derivatives,
unique-anchor equivariance, legacy serialization and Forge boundaries/studies.
Forge declaration validation passes. A local runner initially supplied
`cuda:0` where Queue requires physical index`0`; it failed before creating any
worker, was corrected in a separate reporting commit and its raw error log is
retained. No scientific attempt failed to execute.

Tail all research logs:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/alchemy/queue/events.jsonl
```

Bulk logs, raw streams and checkpoints stay outside Git. Reproduction uses the
shared Python3.12 environment with this worktree on PYTHONPATH. Deterministic
witness controls and public checkpoint/stream continuation are covered by
`tests/test_distillation.py`; the completed full training establishes this revision's rejection.


Reproduce reporting without new training:

```sh
PYTHONPATH=$PWD /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/alchemy/publish.py
PYTHONPATH=$PWD /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/alchemy/diagnose_final.py
```

Restore bulk artifacts from the receipt paths first. Fresh scientific execution
requires a separately admitted source binding; these concluded study IDs must
not be reset. `run_queue.py` is the bounded runner used after both ready arms were
frozen. The raw logs and full checkpoints remain in the dedicated local queue.
