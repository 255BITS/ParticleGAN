<!-- Generated Forge family report -->

# BCAP dualnorm_D_only

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [capped-input-gradients](../technique-inventory.md#tag-capped-input-gradients) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions)

## Technique overview

BCAP with public Recipe.optimizer_family=dualnorm_D_only. The focused study changes only the optimizer rule and its rates from current pure BCAP. It retains the relativistic objective, fixed real/fake cap penalty and each task-owned auxiliary objective. Outcomes remain pending until compatible ordinary receipts are measured.

## Simplified pseudocode

G is the generator, E an encoder when present, D the critic, P the task-declared prior, g the loss gradient, m a momentum buffer, mu its coefficient, eta_P the player step size and epsilon the numerical floor. G and E share one player for global normalization. All updates descend their existing public losses.

```text
For each unchanged task update:
  Compute the existing critic loss plus fixed BCAP penalty; backpropagate D.
  Apply the selected optimizer rule to D.
  Compute the existing G/E/prior loss through the updated critic, retaining task auxiliary terms.
  Record actual G-sampled prior rows when the selected optimizer consumes a row mask.
  Apply the selected optimizer rule to G/E/prior.
  Apply the existing schedule multiplier and score with the frozen task sampling/evaluation law.
Selected rule: Apply the dualnorm matrix/vector rule only to D, with mu=0.5. Generator, encoder and latent prior use native Adam at the fixed incumbent rates and moments. Direct-coordinate controls also keep their native-Adam generator path.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Current BCAP paired relativistic logistic; joint hosts retain their explicit two-stream generator/encoder objective and existing auxiliary terms. |
| Optimizer | Apply the dualnorm matrix/vector rule only to D, with mu=0.5. Generator, encoder and latent prior use native Adam at the fixed incumbent rates and moments. Direct-coordinate controls also keep their native-Adam generator path. |
| Learning rates and annealing | Normalized etaD=1.5*eta for eta in {0.003, 0.01, 0.03, 0.1}. Adam G/E base is fixed at 0.00425 and latent prior at 0.0085. The BCAP baseline schedule has network/prior floors 1, so its declared schedule multiplier stays constant; changing the optimizer does not change that shape. |
| Parameter-gradient clipping | No added parameter-gradient clipping. Normalization defines the optimizer step; the existing BCAP term remains a loss penalty on critic input gradients. |
| Critic penalties and anchors | Unchanged b_cap coefficient 1 and cap 1, applied to real/fake L2 input-gradient norms; no added critic anchor. |
| Damping and update guards | Existing BCAP disables A2, direct-particle gain and critic spike guard; no replacement intervention is introduced. |
| Training and sampling noise | Zero additive training input/output noise. Task-declared prior sampling, named streams and data batches remain unchanged. |
| Parameter averaging and serving | Current BCAP EMA decay 0 and live serving/scoring; this focused study does not substitute the pasted historical EMA/cosine example recipe. |

## Configuration differences

- The initial optimizer campaign has exactly 41 whole recipes and reserves 103320 seconds (2520 per recipe); every recipe retains all six required Tier 1 tasks plus the existing diagnostic.
- Each candidate uses one global configuration across tasks, protocol seed 0 and the public deterministic initializer. Independent runnable current-tier peers finish despite another required failure.
- No measured result or qualification is inferred from these mechanism descriptions. Existing task failures, archived source cohorts and the incumbent remain intact.
- Five-seed inference, the R1/R2 axis, native examples and width/depth transfer are deferred by the focused study scope; P1-P6 cannot be statistically falsified by this screen.
- Consult the single current goal leaderboard reports/forge/technique-inventory.md and the registered search for compatible measured results.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlegan/optim/dualnorm.py](../../../particlegan/optim/dualnorm.py)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/gan_loss.py](../../../particlegan/gan_loss.py)
- [particlegan/grad_regularizers.py](../../../particlegan/grad_regularizers.py)
- [configs/forge/ideas/bcap-dualnorm-d-only-v1.json](../../../configs/forge/ideas/bcap-dualnorm-d-only-v1.json)
- [configs/forge/searches/bcap-optim-dualnorm-d-only-tier1-v1.json](../../../configs/forge/searches/bcap-optim-dualnorm-d-only-tier1-v1.json)
- [configs/forge/campaigns/bcap-dualnorm-tier1-v1.json](../../../configs/forge/campaigns/bcap-dualnorm-tier1-v1.json)
- [reports/forge/technique-inventory.md](../technique-inventory.md)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

**Configuration detail for [BCAP](bcap-pure.md).** Its optimizer or settings do not create a separate solution family. This page preserves the original configuration evidence and diagnostics.

<a name="cohort-cuda-1bf9d7d34422"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [bcap-dualnorm-d-only · 3305345f128e](../../../configs/forge/configurations/bcap-dualnorm-d-only--3305345f128eaaef274d5e0932575a068cd40c59f65d4cfad4051322d282c864.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `6269a18ac4f82564cb16ba19afa4b3dd2f836a2b4085fbe5aeb81a35a453e895`. Candidate revision: `819567a0fb1f32b5fac54bc4c8e55ddaf17b94de6231cf36f31b0a6710c8808d`. Runtime cohort: `e72f1270e1714be910fcca8e3b837f68b50ff539d3fd34295e3b810b87d4edaa`.

[Frozen numerical evidence](../technique-evidence/7c4e188f1ee495526c2decf6f02a45f392123aaf4f33f1f6fb9a398fe3ac05e7.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: current_measurement. Preserve the round's pre-run whole candidate choice in its freshly measured CUDA source cohort; no task pooling, outcome-based recipe reselection, calibration or default adoption.

</details>

Complete current Tier 1 measurement in: discriminator_stability. PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [2(*)/23](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3(*)/30](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [2(*)/6](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [2(*)/29](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [2(*)/24](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [2(*)/24](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [2(*)/22](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) | [0(*)/7](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1) | [0/0](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-2) | [0/0](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-3) | [0(*)/7](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-1bf9d7d34422-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | PASS | matches recorded run |
| [five_word_joint_smoke](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [ring16_acquisition](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [five_word_joint_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 3**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **4 / 19 / 7**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 8**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | required | FAIL | matches recorded run |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [ring16_acquisition](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | required | FAIL | matches recorded run |
| [five_word_joint_smoke](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | required | UNKNOWN | matches recorded run |
| [five_word_joint_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 2**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **7 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-1bf9d7d34422-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.0296894 | <= 0.35 | PASS |
| recon_mse | 0.0119997 | <= 0.05 | PASS |

Recorded terminal passing observations: **22**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/a0f4fa819c2c472ba481b2810b6794e5.json)

Recorded conditions: mog prior (sigma 0.025); generated_and_reconstructed_prior_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = encoder, generator, prior, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 250 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); generated_and_reconstructed_prior_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = encoder, generator, prior, discriminator; rng isolation = True.

Declared budget: 250 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_generated_and_reconstructed_prior_with_scheduled_output_noise; weights state_selected; output noise public_recipe_schedule.

<a name="cohort-cuda-1bf9d7d34422-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: PASS**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. declared state/horizon/cadence/restart comparisons agree; source audit bound

Actual task device: `0` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| parity_comparisons | 4 |

[Compact metrics and receipt provenance](../technique-receipts/169224a78d9b4d51afb0c3a609829e26.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| u_kept | >= 0.85 |
| content_kept | >= 0.75 |
| leak_ratio | <= 0.2 |
| pole_rel_err_plus | <= 0.2 |
| pole_rel_err_minus | <= 0.2 |
| same_dir | <= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 800 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: exact optimizer updates = True; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_generated_and_paired_reconstructed_prior_without_output_noise; weights state_selected; output noise clean.

[Explanation and existing training artifacts](../five-word-joint/README.md)

<a name="cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = False.

Declared budget: 4000 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

Dependencies: five_word_joint_smoke (checkpoint).

[Explanation and existing training artifacts](../five-word-tier-split/README.md)

<a name="cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../five-word-tier-split/README.md)

<a name="cohort-cuda-1bf9d7d34422-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

[Explanation and existing training artifacts](../../toy_audit/api_contract/gaussian1d/README.md)

<a name="cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full pass with independent same-state confirmation

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.159188 |
| finite_fraction | 1 |
| mean | 1.84765 |
| mean_error_sigma | 0.304706 |
| sample_count | 4096 |
| std | 0.431782 |
| std_ratio | 0.863563 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/0e345a04c9dd4bf99051d47a09a36318.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../gaussian-smoke-tier-split/README.md)

<a name="cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2).

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

Declared budget: 6000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: gaussian1d_smoke (checkpoint).

[Explanation and existing training artifacts](../gaussian-smoke-tier-split/README.md)

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100"></a>

### grid100

**grid100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: particle_cloud prior (sigma 0); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_bars4"></a>

### img_bars4

**img_bars4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| concept_cos_plus | >= 0.85 |
| concept_cos_minus | >= 0.85 |
| concept_mag_plus | >= 0.75 |
| concept_mag_plus | <= 1.25 |
| concept_mag_minus | >= 0.75 |
| concept_mag_minus | <= 1.25 |
| identity_at_0 | >= 0.85 |
| identity_at_mid | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 800 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-mode_hold"></a>

### mode_hold

**mode_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-residual_student"></a>

### residual_student

**residual_student: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |
| success_rate | >= 1 |
| wrong_pad_rate | <= 0 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 20.285 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 1.68061 | >= 0.15 | PASS |
| hq | 0.491943 | >= 0.85 | FAIL |
| mass_tv | 0.0410156 | <= 0.15 | PASS |
| modes | 13 | >= 16 | FAIL |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f5a6afb6e65a4de5919832041ea4f359.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 96 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-ring_extension"></a>

### ring_extension

**ring_extension: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

Declared budget: 7500 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: ring_hold (checkpoint).

<a name="cohort-cuda-1bf9d7d34422-experiment-ring_hold"></a>

### ring_hold

**ring_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

Declared budget: 7500 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-1bf9d7d34422-experiment-rotated100"></a>

### rotated100

**rotated100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: rotated100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-staggered100"></a>

### staggered100

**staggered100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: staggered100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) · [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-1bf9d7d34422-experiment-trajectory"></a>

### trajectory

**trajectory: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-1bf9d7d34422-experiment-two_pole"></a>

### two_pole

**two_pole: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 1.2109 | <= 1 | FAIL |
| mean_abs | 0.293039 | >= 0.3 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/4e51092d751c47c8b74042c3e6e7157c.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = prior, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 80 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [k3p_two_pole_horizon / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

[Explanation and existing training artifacts](../k3p-two-pole-horizon-v1/README.md)

<a name="cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [k3p_two_pole_horizon / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

[Explanation and existing training artifacts](../k3p-two-pole-horizon-v1/README.md)

<a name="cohort-cuda-1bf9d7d34422-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 80 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_learned_particles_and_critic_gradient; weights state_selected; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-unipolar"></a>

### unipolar

**unipolar: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| cover | >= 0.85 |
| off_caption | <= 0.05 |
| neu_hold | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.987808 | >= 0.85 | PASS |
| unused_hold | 0.999157 | >= 0.85 | PASS |

Recorded terminal passing observations: **10**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/5fc2548464e44126aeedae2d0e5c78f2.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = generator, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 200 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `ef8ccde8d1fa54af8bfce01c044e3671de8131c980eb4e8022d12ffc8caf51d8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator; rng isolation = True.

Declared budget: 200 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_learned_parameter_measurement; weights state_selected; output noise not_applied_to_measurement.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [bcap-dualnorm-d-only · 3305345f128e](../../../configs/forge/configurations/bcap-dualnorm-d-only--3305345f128eaaef274d5e0932575a068cd40c59f65d4cfad4051322d282c864.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `d276c5a7344fab6ec5de7b314d3982af5b0ef8027c8b89c01376ae366844a9cb`. Candidate revision: `2e09a54ee6f53133528ad974e1e2174744459b014f3bed3907eda942b5e44dc4`. Runtime cohort: `598a38b2e3c994b9ecabc3eccd85eedf9a1164e8a90bf8117af77f9b3569641f`.

[Frozen numerical evidence](../technique-evidence/83f549889bf36b44bda98c8fe4c0f1600ef0cdd1d12c529f9e480ffcfdc93512.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Preserve the exact archived revision-7 measurement and its original policy qualification; no current-policy measurement or default-adoption credit.

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) | [0(*)/1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-3) | [2(*)/23](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation) |
| [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous) | [3/4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) | [0(*)/7](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | [3(*)/30](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous) |
| [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability) | [3(*)/6](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | [0(*)/21](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | [0(*)/2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) | [3(*)/29](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability) |
| [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) | [0(*)/2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) | [2(*)/24](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison) |
| [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) | [0(*)/2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | [2(*)/24](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer) |
| [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | [0/0](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-3) | [2(*)/22](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage) | [0(*)/7](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1) | [0/0](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-2) | [0/0](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-3) | [0(*)/7](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-c195899a64af-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) | PASS | matches recorded run |
| [five_word_joint_smoke](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | PASS | matches recorded run |
| [ring16_acquisition](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-two_pole) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [five_word_joint_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_bars4) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mode_hold) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-residual_student) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-trajectory) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unipolar) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100_14k) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring_extension) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring_hold) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100_14k) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100_14k) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-3), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-c195899a64af-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 3**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **4 / 19 / 7**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 8**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | required | PASS | matches recorded run |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [ring16_acquisition](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | required | FAIL | matches recorded run |
| [five_word_joint_smoke](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | required | UNKNOWN | matches recorded run |
| [five_word_joint_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 2**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-c195899a64af-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | FAIL | matches recorded run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **7 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-c195899a64af-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.0241582 | <= 0.35 | PASS |
| recon_mse | 0.00339826 | <= 0.05 | PASS |

Recorded terminal passing observations: **20**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/a4e39db19cff4678ae915ad2f653eae6.json)

Recorded conditions: mog prior (sigma 0.025); generated_and_reconstructed_prior_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = encoder, generator, prior, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 250 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); generated_and_reconstructed_prior_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-c195899a64af-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = encoder, generator, prior, discriminator; rng isolation = True.

Declared budget: 250 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_generated_and_reconstructed_prior_with_scheduled_output_noise; weights state_selected; output noise public_recipe_schedule.

<a name="cohort-cuda-c195899a64af-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: PASS**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. declared state/horizon/cadence/restart comparisons agree; source audit bound

Actual task device: `0` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| parity_comparisons | 4 |

[Compact metrics and receipt provenance](../technique-receipts/933c99cc47324661b91130b6c8dd69fc.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| u_kept | >= 0.85 |
| content_kept | >= 0.75 |
| leak_ratio | <= 0.2 |
| pole_rel_err_plus | <= 0.2 |
| pole_rel_err_minus | <= 0.2 |
| same_dir | <= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 800 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-c195899a64af-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: exact optimizer updates = True; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_generated_and_paired_reconstructed_prior_without_output_noise; weights state_selected; output noise clean.

[Explanation and existing training artifacts](../five-word-joint/README.md)

<a name="cohort-cuda-c195899a64af-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = False.

Declared budget: 4000 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

Dependencies: five_word_joint_smoke (checkpoint).

[Explanation and existing training artifacts](../five-word-tier-split/README.md)

<a name="cohort-cuda-c195899a64af-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../five-word-tier-split/README.md)

<a name="cohort-cuda-c195899a64af-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

[Explanation and existing training artifacts](../../toy_audit/api_contract/gaussian1d/README.md)

<a name="cohort-cuda-c195899a64af-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full pass with independent same-state confirmation

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.0735022 |
| finite_fraction | 1 |
| mean | 1.92232 |
| mean_error_sigma | 0.15535 |
| sample_count | 4096 |
| std | 0.51886 |
| std_ratio | 1.03772 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/daa0baee3ffc469693f6411586b8ae3c.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../gaussian-smoke-tier-split/README.md)

<a name="cohort-cuda-c195899a64af-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2).

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

Declared budget: 6000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: gaussian1d_smoke (checkpoint).

[Explanation and existing training artifacts](../gaussian-smoke-tier-split/README.md)

<a name="cohort-cuda-c195899a64af-experiment-grid100"></a>

### grid100

**grid100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-c195899a64af-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: particle_cloud prior (sigma 0); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-img_bars4"></a>

### img_bars4

**img_bars4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| concept_cos_plus | >= 0.85 |
| concept_cos_minus | >= 0.85 |
| concept_mag_plus | >= 0.75 |
| concept_mag_plus | <= 1.25 |
| concept_mag_minus | >= 0.75 |
| concept_mag_minus | <= 1.25 |
| identity_at_0 | >= 0.85 |
| identity_at_mid | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 800 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-c195899a64af-experiment-mode_hold"></a>

### mode_hold

**mode_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-residual_student"></a>

### residual_student

**residual_student: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |
| success_rate | >= 1 |
| wrong_pad_rate | <= 0 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-c195899a64af-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 21.1383 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 2.96989 | >= 0.15 | PASS |
| hq | 0.516113 | >= 0.85 | FAIL |
| mass_tv | 0.0715332 | <= 0.15 | PASS |
| modes | 15 | >= 16 | FAIL |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/7504ae7533fb4d63a46009e8ede83b7d.json)

Recorded conditions: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 96 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-ring_extension"></a>

### ring_extension

**ring_extension: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

Declared budget: 7500 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: ring_hold (checkpoint).

<a name="cohort-cuda-c195899a64af-experiment-ring_hold"></a>

### ring_hold

**ring_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

Declared budget: 7500 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-c195899a64af-experiment-rotated100"></a>

### rotated100

**rotated100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: rotated100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-staggered100"></a>

### staggered100

**staggered100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: staggered100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-3) · [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-c195899a64af-experiment-trajectory"></a>

### trajectory

**trajectory: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-c195899a64af-experiment-two_pole"></a>

### two_pole

**two_pole: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.237631 | <= 1 | PASS |
| mean_abs | 0.257395 | >= 0.3 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/42bc1c50e3704936852f226107e8b076.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = prior, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 80 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-c195899a64af-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [k3p_two_pole_horizon / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

[Explanation and existing training artifacts](../k3p-two-pole-horizon-v1/README.md)

<a name="cohort-cuda-c195899a64af-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [k3p_two_pole_horizon / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

[Explanation and existing training artifacts](../k3p-two-pole-horizon-v1/README.md)

<a name="cohort-cuda-c195899a64af-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 80 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_learned_particles_and_critic_gradient; weights state_selected; output noise not_applied_to_measurement.

<a name="cohort-cuda-c195899a64af-experiment-unipolar"></a>

### unipolar

**unipolar: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| cover | >= 0.85 |
| off_caption | <= 0.05 |
| neu_hold | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-c195899a64af-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.985216 | >= 0.85 | PASS |
| unused_hold | 0.989573 | >= 0.85 | PASS |

Recorded terminal passing observations: **10**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/3ad0f0878664432ca77c5ed332b56900.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = generator, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 200 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-c195899a64af-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `ef8ccde8d1fa54af8bfce01c044e3671de8131c980eb4e8022d12ffc8caf51d8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator; rng isolation = True.

Declared budget: 200 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_learned_parameter_measurement; weights state_selected; output noise not_applied_to_measurement.

<a name="cohort-cuda-c195899a64af-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [bcap-dualnorm-d-only · 3305345f128e](../../../configs/forge/configurations/bcap-dualnorm-d-only--3305345f128eaaef274d5e0932575a068cd40c59f65d4cfad4051322d282c864.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `c5c60a8018e144c447325f3705dd398d04069a91bb294eaaa76d37084ade67cc`. Candidate revision: `f02bd4cbfe6c9b6f720ed99b1c628eb785aa23a06cd33dc1d80d0f8844c69e62`. Runtime cohort: `c3afbd30cd6bb0c6c79516597a4a9f8cec34a5c9aa0af4f4325285d1c3d548e6`.

[Frozen numerical evidence](../technique-evidence/0239350844b13b7e346b446dbd668e1469625f55755e4989e4f0660e780cff6e.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Preserve the exact archived revision-5 measurement and its original policy qualification; no current-policy measurement or default-adoption credit.

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) | [0(*)/1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) | [2(*)/23](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation) |
| [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous) | [3/4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) | [0(*)/7](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | [3(*)/30](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous) |
| [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability) | [2(*)/6](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | [0(*)/21](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | [0(*)/2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) | [2(*)/29](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability) |
| [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) | [0(*)/2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) | [2(*)/24](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison) |
| [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) | [0(*)/2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | [2(*)/24](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer) |
| [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage) | [2/3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | [0(*)/19](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | [0/0](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-3) | [2(*)/22](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) | [0(*)/7](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1) | [0/0](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2) | [0/0](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3) | [0(*)/7](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-0d83d78027c5-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | changed since run |
| [clockfree_audit_measurement_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | PASS | matches recorded run |
| [five_word_joint_smoke](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | FAIL | changed since run |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | FAIL | changed since run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [five_word_joint_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | [adaptation](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-3), [clockfree_continuous](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 3**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **4 / 19 / 7**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |
| [clockfree_audit_measurement_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | required | PASS | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 8**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | required | UNKNOWN | recorded definition unavailable |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |
| [ring16_acquisition](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | required | FAIL | changed since run |
| [five_word_joint_smoke](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | diagnostic | PASS | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 2**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **7 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. Test-definition changes describe differences from the recorded run, independently of whether it was executed. Earlier verdicts are preserved.

<a name="cohort-cuda-0d83d78027c5-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00295198 | <= 0.35 | PASS |
| recon_mse | 0.00607732 | <= 0.05 | PASS |

Recorded terminal passing observations: **12**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/081d6122848c44909c3d28b52abd939f.json)

Recorded conditions: mog prior (sigma 0.025); generated_and_reconstructed_prior_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = encoder, generator, prior, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 250 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); generated_and_reconstructed_prior_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-0d83d78027c5-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = encoder, generator, prior, discriminator; rng isolation = True.

Declared budget: 250 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_generated_and_reconstructed_prior_with_scheduled_output_noise; weights state_selected; output noise public_recipe_schedule.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: PASS**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. declared state/horizon/cadence/restart comparisons agree; source audit bound

Actual task device: `0` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| parity_comparisons | 4 |

[Compact metrics and receipt provenance](../technique-receipts/f16994dff74442c88ea05f66b1404841.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| u_kept | >= 0.85 |
| content_kept | >= 0.75 |
| leak_ratio | <= 0.2 |
| pole_rel_err_plus | <= 0.2 |
| pole_rel_err_minus | <= 0.2 |
| same_dir | <= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 800 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: exact optimizer updates = True; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_generated_and_paired_reconstructed_prior_without_output_noise; weights state_selected; output noise clean.

[Explanation and existing training artifacts](../five-word-joint/README.md)

<a name="cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold"></a>

### five_word_joint_hold

**five_word_joint_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_hold.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = False.

Declared budget: 4000 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

Dependencies: five_word_joint_smoke (checkpoint).

[Explanation and existing training artifacts](../five-word-tier-split/README.md)

<a name="cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke"></a>

### five_word_joint_smoke

**five_word_joint_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../five-word-tier-split/README.md)

<a name="cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

[Explanation and existing training artifacts](../../toy_audit/api_contract/gaussian1d/README.md)

<a name="cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke"></a>

### gaussian1d_smoke

**gaussian1d_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

Declared budget: 1000 updates; timeout 120 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../gaussian-smoke-tier-split/README.md)

<a name="cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability"></a>

### gaussian1d_stability

**gaussian1d_stability: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

Declared budget: 6000 updates; timeout 600 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: gaussian1d_smoke (checkpoint).

[Explanation and existing training artifacts](../gaussian-smoke-tier-split/README.md)

<a name="cohort-cuda-0d83d78027c5-experiment-grid100"></a>

### grid100

**grid100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: grid100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: particle_cloud prior (sigma 0); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_bars4"></a>

### img_bars4

**img_bars4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| concept_cos_plus | >= 0.85 |
| concept_cos_minus | >= 0.85 |
| concept_mag_plus | >= 0.75 |
| concept_mag_plus | <= 1.25 |
| concept_mag_minus | >= 0.75 |
| concept_mag_minus | <= 1.25 |
| identity_at_0 | >= 0.85 |
| identity_at_mid | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 800 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-mode_hold"></a>

### mode_hold

**mode_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-residual_student"></a>

### residual_student

**residual_student: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |
| success_rate | >= 1 |
| wrong_pad_rate | <= 0 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-0d83d78027c5-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 30.5062 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 2.31101 | >= 0.15 | PASS |
| hq | 0.375977 | >= 0.85 | FAIL |
| mass_tv | 0.0505371 | <= 0.15 | PASS |
| modes | 13 | >= 16 | FAIL |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/daa3b4b7045e4ecdb5ac149d677cee4e.json)

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 96 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.1); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-ring_extension"></a>

### ring_extension

**ring_extension: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

Declared budget: 7500 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: ring_hold (checkpoint).

<a name="cohort-cuda-0d83d78027c5-experiment-ring_hold"></a>

### ring_hold

**ring_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

Declared budget: 7500 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-0d83d78027c5-experiment-rotated100"></a>

### rotated100

**rotated100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: rotated100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-staggered100"></a>

### staggered100

**staggered100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 14000 updates; timeout 7200 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: staggered100 (checkpoint), clockfree_audit (gate).

<a name="cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.97 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

Declared budget: 7000 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) · [clockfree_continuous / Tier 3](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-0d83d78027c5-experiment-trajectory"></a>

### trajectory

**trajectory: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-0d83d78027c5-experiment-two_pole"></a>

### two_pole

**two_pole: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.583821 | <= 1 | PASS |
| mean_abs | 0.252038 | >= 0.3 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/a2d60cc61e7b43b4a7119efae9c1a645.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = prior, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 80 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [k3p_two_pole_horizon / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

[Explanation and existing training artifacts](../k3p-two-pole-horizon-v1/README.md)

<a name="cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [k3p_two_pole_horizon / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 800 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_particles_and_critic_gradient; weights live; output noise not_applied_to_measurement.

[Explanation and existing training artifacts](../k3p-two-pole-horizon-v1/README.md)

<a name="cohort-cuda-0d83d78027c5-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

Declared budget: 80 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_learned_particles_and_critic_gradient; weights state_selected; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-unipolar"></a>

### unipolar

**unipolar: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| cover | >= 0.85 |
| off_caption | <= 0.05 |
| neu_hold | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.979091 | >= 0.85 | PASS |
| unused_hold | 0.99775 | >= 0.85 | PASS |

Recorded terminal passing observations: **10**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c7a2a28fb59540e98450d42036cb1975.json)

Recorded conditions: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = generator, discriminator; mechanism exercised = True; rng isolation = True.

Declared budget: 200 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); learned_parameter_measurement; weights live; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `ef8ccde8d1fa54af8bfce01c044e3671de8131c980eb4e8022d12ffc8caf51d8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator; rng isolation = True.

Declared budget: 200 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_learned_parameter_measurement; weights state_selected; output noise not_applied_to_measurement.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1600 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Used by: [formulation_comparison / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap-dualnorm-d-only.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 1200 updates; timeout 1800 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

## Historical and diagnostic evidence

Separate configurations, API variants and serving laws retain their own scopes and supply no cells above.

- [Other configurations and original evidence bindings](../technique-inventory.json)
- [Compiled experiment memory](../EXPERIMENT_MEMORY.md)

## Refresh

```sh
python reports/forge/regenerate_technique_inventory.py
```

This page is generated alongside the leaderboard. Register new source evidence before refreshing; editing a page cannot change a verdict or earn qualification.

## References

This technique is repository-specific; no dedicated paper reference is declared. Its implementation and recipe sources are linked above.
