# Direction-only baseline repair: frozen conditional proposal

This approved proposal is a separate bounded follow-up to the five Phase 3
studies, prepared from `9005ed73a09a740174f590fbfd543440935040eb`. It has no
paid attempts, queue admissions, or production edits. Admission requires root's
explicit confirmation that all five studies are ineligible under their declared
replacement rule. A completed original Tier 1 failure establishes ineligibility;
it does not require retrying that scientific failure. Every study must still
finish its remaining jobs and publish its complete evidence. Root owns admission
and worker scheduling for this pair; no duplicate admission is authorized.

The candidate changes exactly one complete-recipe field:
`constraint_geometry_mode: none -> direction_blend`. The control inherits the
selected `bcap-three-phase-incumbent-v1` unchanged. Transport weights and
`prior_reg` remain zero, SVD remains native, BCAP cap remains 1, momentum remains
zero, and every other resolved recipe field matches. The precommit historical
[preparation proof](preparation.json)
verifies 1,206 scientific file hashes, the original digest
`d944540c70e280d8f981367368792920a70bbec0d919d681b4bdf0770a96f59f`, and
identical task conditions across both arms. Technique ownership records differ
only in the intentional mode value and its source. No task source rebinding was
needed. Both arms compute READY in read-only planning; that is not admission.

The existing algorithm first applies one original DualNorm update, then measures
its actual rounded displacement Δ against the protected gradients. If every
gradient has nonpositive dot product with Δ, it retains the already-applied
tensors bitwise. On conflict, let a_i be unit protected normals, C their common
non-ascent cone, and a_bar their mean. The full-scale replacement is
`d = (projection_C(Δ) - ||Δ|| a_bar / ||a_bar||) / 2`. Opposed normals cause a
zero displacement, while the original optimizer clock still advances once.
This is the unchanged archived implementation. It checks no finite objective
values and adds no paired supervision, target draws, or evaluator probes.

| Original consumer | Total generator-side objective | Existing protected signals | Expected effect without transport |
|---|---|---|---|
| Gaussian smoke/stability, ring, vectors, native100 | GAN + zero-weight prior regularizer | GAN | Aligned ideal direction; expect no blends |
| Word smoke/own hold | One joint G/E/prior GAN + zero-weight prior regularizer | The same joint GAN | Does not separately protect generation and inverse reconstruction; expect no blends |
| Two-pole | GAN + fixed host particle-L2 | GAN | Empirically exact inactive parity in Phase 1; not a sole-GAN identity |
| Trajectory | GAN + original set-cover, particle-L2 and spread | GAN | May remove cover/spread displacement that increases conditional adversarial loss |
| Residual student | Same terms + existing masked paired residual | GAN and active paired residual | May retain correct conditional assignment while moving the set |
| Unused-token hold | Original GAN branch + masked unused-slot hold | GAN branch and hold | Can change existing conflict; Tier 1 preservation requires measurement |
| AE/GAN hold | Reconstruction + generated GAN/cover + original regularization | Reconstruction and original GAN/cover branch | Can change existing branch conflict; Tier 1 preservation requires measurement |
| Cover-leftover / mid-scale (outside the proposed 16 tasks) | Original GAN and active paired cover, plus existing host regularizers | GAN and cover | Supported existing hooks; no grade predicted in this study |

The single-loss expectation is analytical, not an unconditional bitwise theorem
for floating-point full steps. The actual `gradient @ rounded_displacement > 0`
criterion remains authoritative. The base zero-momentum smoothed polar and row
directions descend their own ideal gradient, but rounding can change the measured
dot product. Even a blended non-ascent derivative does not guarantee that a
finite loss decreases, or protect against the next critic update. The generated
optimizer joins the original G/E/prior groups only for common displacement
protection, preserving their rates, group laws, and sampled-row ownership.

The strongest historical support is specific. At publication
`fed122eed540a100a83d0b2614895f35d6c9e5c9`, the direction-only trajectory
variant passes with MSE .000258480 and terminal suffix 19, versus winner FAIL
.239862/suffix 0. Residual passes MSE .000249089, own-pad success 1, wrong-pad
rate 0, suffix 21, versus winner FAIL .0610360, .5/.5, suffix 0. The direction
arm records 12 trajectory and 55 residual blends, no stalls, and zero active
transport calls. Its original measured source is `593748efef6d1760315723beba46f1e95ffaa75b`,
not the later publication commit. [Compact historical projection](prior-evidence.json)
retains original task IDs, source/digest, result blob/SHA and attempt identities.

The current trajectory/residual loops and direction/common-descent modules are
byte-identical to that archived publication. Conditional variants and current
original cards have the same architecture, initialization, prior, update budget,
sampling and gates; their IDs, cohort, source bindings and consumer annotations
differ. Current zero-weight consumers return the original loss immediately,
where the historical consumer recorded inactive calls. This is useful transfer
evidence, with no transfer of grades or checkpoints. Phase 1 independently
records exact two-pole tensors and observations for incumbent/projection-only,
with suffix 17. Neither source establishes current direction-only 6/6 Tier 1.

Runtime remains a material falsifier. Protection computes an extra
`autograd.grad` before the original backward and clones/flattens G/E/prior
parameters; it performs no additional SVD. The saved word profile measures
about 1.25 ms/update for protected-gradient binding, approximately 25 seconds
over 20,001 updates before other wrapper overhead. The fresh native incumbent
finished in 755.73 seconds against the unchanged 900-second allowance. These
different source/contention measurements do not predict a deadline pass.
A timeout remains INCOMPLETE and blocks that arm's own word hold; there is no
backend substitution or allowance extension. Existing projection, row-ownership,
checkpoint/resume, inactive compatibility and runner controls already cover the
software mechanism, so no duplicate tests or quality reruns were made here.

The [input spec](spec.json) and [generated registration](registration.json)
declare the same six original Tier 1 and ten Tier 2 questions used by the shared
paired runner. Each arm reserves 22,920 seconds, the pair 45,840 seconds, with
48,000 paid ceiling and a separate 300-second software allowance. Protocol seed
0 and public deterministic initialization remain fixed. Both own Gaussian
stability and word hold require that same arm's complete passing producer and
exact compatible state; no control checkpoint can be borrowed. This separate
diagnostic view preserves all original gates and grants no ordinary qualification.

Prediction: preserve 6/6 Tier 1 and repair the complete trajectory and residual
gates, including five terminal passing checks, trajectory/residual identity MSE
≤ .02, residual success ≥ 1 and wrong-pad rate ≤ 0. Expect scalar/vector/native
and word numerical paths to match their paired control when the actual dot
criterion stays inactive. Gaussian retention and vector density failures are
expected to remain, since their sole protected GAN is unchanged. A research
replacement still requires at least one complete paired Tier 2 FAIL→PASS,
preservation of every baseline Tier 2 PASS, and no unresolved paired comparisons.
Any Tier 1 non-PASS, missed sustained conditional gate, regression of a baseline
PASS, or incomplete dependency falsifies eligibility; favorable endpoints cannot
rescue it. Finite-step curvature, critic dynamics and host incompatibility remain
competing explanations for conditional failure.

The reviewed declarations and report are frozen together on this isolated
branch. [Read-only reproduction check](verify_preparation.py) resolves the complete pair
and verifies the sole recipe delta and original scientific digest. The shared
runner's source guard verifies the exact committed HEAD before admission;
report-only origin changes retain the original scientific digest. Root must
confirm the ineligibility predicate before executing these existing helper calls:

```sh
# Root's explicit admission predicate must be satisfied before these commands.
archive=/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/baseline-repairs/projection
python reports/forge/bcap-projection-baseline-review/verify_preparation.py
python reports/forge/bcap-three-phase/phase3.py --submit \
  --registration reports/forge/bcap-projection-baseline-review/registration.json \
  --artifacts "$archive" --source-commit "$(git rev-parse HEAD)" \
  > "$archive/logs/submit.log" 2>&1
python reports/forge/bcap-three-phase/phase3.py --drain \
  --registration reports/forge/bcap-projection-baseline-review/registration.json \
  --artifacts "$archive" --source-commit "$(git rev-parse HEAD)" --gpus 0,1 \
  > "$archive/logs/phase3-driver.log" 2>&1
tail -F "$archive/logs/phase3-driver.log"
```

Admission/execution are pending. After completion, publish certified saved-only
metrics, grading reasons, attempt/cost history, initial/prior/stream proofs,
own-checkpoint dependencies, and actual-training GIFs with the existing guarded
publisher. Keep bulk evidence outside Git and the current technique inventory
as the single goal leaderboard. No additional candidate, seeds, tuning, automatic
retry, merge, or public-default change is proposed.
