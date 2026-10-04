<!-- Generated Forge family report -->

# Atlas — experiment results

[← Family leaderboard](../technique-inventory.md)

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

**Full original Atlas — fresh common-26 diagnostic: two_pole FAIL; completed 1/26; remaining 25 NOT_RUN.** mean_abs 0.00244565 >= 0.3 (FAIL); grad_med 0.010897 <= 1 (PASS). The accepted first case completed 80 updates and 24 ordinary live observations at seed 0. [Verified first-case result and goal GIF](../common26-first-two-pole-full-atlas-20261004/README.md) · [Pinned result, full Recipe and source](../common26-first-two-pole-full-atlas-20261004/results.json). This full original configuration is separate from the canonical Atlas configuration selected in the recorded table. The requested continuation uses the original revision-3 common-26 gates and continues after numerical FAIL; the remaining cases are pending adapter and budget resolution. No selected-table cells, prerequisite credit, default adoption or speed ranking are awarded.

<a name="cohort-cuda-7f9c23eb0e27"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [atlas](../../../configs/forge/ideas/atlas.json).

Recorded qualification: **tier 0**, discriminator_stability revision 3. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `7306340bac0a4ea67ea7b080513116457b7d72a8cb0d0c1f0727eaae6db38185`. Candidate revision: `506319d4621e284ded8897c02670c7c51c2d50c759e6a459b39848114e346667`. Runtime cohort: `e198762f715ad7070912b7940542c3688b7f3b0d695b91f28e17129bdca24cba`.

[Frozen numerical evidence](../technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Retain the exact recorded incumbent; its outcomes are historical best observed evidence where a registered search exists. Alternatives from another source or runtime are unranked.

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation) | [0(*)/3](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) | [0(*)/19](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) | [0(*)/1](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-3) | [0(*)/23](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation) |
| [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous) | [0(*)/4](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) | [0(*)/19](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) | [0(*)/6](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | [0(*)/29](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous) |
| [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability) | [0(*)/6](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | [0(*)/19](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | [0(*)/2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) | [0(*)/27](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability) |
| [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison) | [0(*)/3](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) | [0(*)/19](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) | [0(*)/2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3) | [0(*)/24](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison) |
| [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer) | [0(*)/3](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) | [0(*)/19](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) | [0(*)/2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) | [0(*)/24](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer) |
| [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage) | [0(*)/3](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | [0(*)/19](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | [0/0](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-3) | [0(*)/22](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage) |

\* indicates incomplete results, including changed or unbound current contracts.

<a name="cohort-cuda-7f9c23eb0e27-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ae_gan_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | BLOCKED | CHANGED |
| [clockfree_audit](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit) | [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) | UNKNOWN | unbound |
| [five_word_joint_acquisition](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_acquisition) | [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | BLOCKED | CHANGED |
| [gaussian1d_acquisition](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_acquisition) | [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | unbound |
| [ring16_acquisition](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition) | [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | BLOCKED | CHANGED |
| [two_pole](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | BLOCKED | CHANGED |
| [unused_token_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | BLOCKED | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [cover_leftover](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [grid100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | matches |
| [img_bars4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [img_blobs4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [img_intensity2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [img_stripes2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [mid_scale_identity](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [mode_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [residual_student](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [rotated100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | matches |
| [staggered100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | matches |
| [trajectory](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [unipolar](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [vector_anisotropic](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [vector_overlap](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [vector_spiral](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [vector_two_broad](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [vector_unequal_mass](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |
| [vector_unequal_width](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | BLOCKED | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [grid100_14k](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [ring_extension](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) | BLOCKED | matches |
| [ring_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3), [discriminator_stability](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3), [formulation_comparison](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3), [host_profile_transfer](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) | BLOCKED | matches |
| [rotated100_14k](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [staggered100_14k](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [target_shift_recovery](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery) | [adaptation](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-3), [clockfree_continuous](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-7f9c23eb0e27-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | CHANGED |
| [unused_token_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | CHANGED |
| [ae_gan_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | CHANGED |
| [residual_student](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | CHANGED |
| [unipolar](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | CHANGED |
| [cover_leftover](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | CHANGED |
| [mid_scale_identity](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | CHANGED |
| [mode_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | CHANGED |
| [vector_two_broad](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | CHANGED |
| [vector_unequal_mass](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | BLOCKED | CHANGED |
| [vector_unequal_width](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | BLOCKED | CHANGED |
| [vector_anisotropic](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | BLOCKED | CHANGED |
| [vector_overlap](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | BLOCKED | CHANGED |
| [vector_spiral](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | CHANGED |
| [img_stripes2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | CHANGED |
| [img_bars4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | BLOCKED | CHANGED |
| [img_blobs4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | BLOCKED | CHANGED |
| [img_intensity2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | BLOCKED | CHANGED |
| [grid100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | matches |
| [rotated100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | matches |
| [staggered100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | matches |

<a name="cohort-cuda-7f9c23eb0e27-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [target_shift_recovery](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery) | required | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 2**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **4 / 19 / 6**.

Calibration: **provisional**. Phase D historical calibration remains required

Additional eligibility requirements:

- Capability: named_rng
- Capability: checkpoint
- Claim learning: clockfree
- Claim shared_settings: True

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | CHANGED |
| [unused_token_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | CHANGED |
| [ae_gan_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | CHANGED |
| [clockfree_audit](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit) | required | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | CHANGED |
| [residual_student](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | CHANGED |
| [unipolar](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | CHANGED |
| [cover_leftover](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | CHANGED |
| [mid_scale_identity](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | CHANGED |
| [mode_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | CHANGED |
| [vector_two_broad](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | CHANGED |
| [vector_unequal_mass](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | BLOCKED | CHANGED |
| [vector_unequal_width](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | BLOCKED | CHANGED |
| [vector_anisotropic](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | BLOCKED | CHANGED |
| [vector_overlap](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | BLOCKED | CHANGED |
| [vector_spiral](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | CHANGED |
| [img_stripes2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | CHANGED |
| [img_bars4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | BLOCKED | CHANGED |
| [img_blobs4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | BLOCKED | CHANGED |
| [img_intensity2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | BLOCKED | CHANGED |
| [grid100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | matches |
| [rotated100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | matches |
| [staggered100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | matches |

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | required | BLOCKED | matches |
| [ring_extension](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | BLOCKED | matches |
| [grid100_14k](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_14k) | required | UNKNOWN | unbound |
| [rotated100_14k](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_14k) | required | UNKNOWN | unbound |
| [staggered100_14k](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_14k) | required | UNKNOWN | unbound |
| [target_shift_recovery](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery) | required | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 4**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 19 / 2**.

Calibration: **provisional**. Expanded six-task Tier 1 placement is provisional and requires bounded calibration. Revision 3 and prior profiles retain their original tasks and evidence; a standalone scalar pass gives no whole-view/default credit.

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [gaussian1d_acquisition](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_acquisition) | required | UNKNOWN | unbound |
| [two_pole](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | CHANGED |
| [unused_token_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | CHANGED |
| [ae_gan_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | CHANGED |
| [ring16_acquisition](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition) | required | BLOCKED | CHANGED |
| [five_word_joint_acquisition](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_acquisition) | required | BLOCKED | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | CHANGED |
| [residual_student](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | CHANGED |
| [unipolar](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | CHANGED |
| [cover_leftover](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | CHANGED |
| [mid_scale_identity](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | CHANGED |
| [mode_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | CHANGED |
| [vector_two_broad](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | CHANGED |
| [vector_unequal_mass](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | BLOCKED | CHANGED |
| [vector_unequal_width](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | BLOCKED | CHANGED |
| [vector_anisotropic](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | BLOCKED | CHANGED |
| [vector_overlap](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | BLOCKED | CHANGED |
| [vector_spiral](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | CHANGED |
| [img_stripes2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | CHANGED |
| [img_bars4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | BLOCKED | CHANGED |
| [img_blobs4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | BLOCKED | CHANGED |
| [img_intensity2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | BLOCKED | CHANGED |
| [grid100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | matches |
| [rotated100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | matches |
| [staggered100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | matches |

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | required | BLOCKED | matches |
| [ring_extension](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | BLOCKED | matches |

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | CHANGED |
| [unused_token_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | CHANGED |
| [ae_gan_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | CHANGED |
| [residual_student](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | CHANGED |
| [unipolar](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | CHANGED |
| [cover_leftover](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | CHANGED |
| [mid_scale_identity](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | CHANGED |
| [mode_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | CHANGED |
| [vector_two_broad](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | CHANGED |
| [vector_unequal_mass](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | BLOCKED | CHANGED |
| [vector_unequal_width](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | BLOCKED | CHANGED |
| [vector_anisotropic](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | BLOCKED | CHANGED |
| [vector_overlap](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | BLOCKED | CHANGED |
| [vector_spiral](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | CHANGED |
| [img_stripes2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | CHANGED |
| [img_bars4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | BLOCKED | CHANGED |
| [img_blobs4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | BLOCKED | CHANGED |
| [img_intensity2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | BLOCKED | CHANGED |
| [grid100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | matches |
| [rotated100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | matches |
| [staggered100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | matches |
| [img_intensity2_residual16](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | unbound |
| [vector_two_broad_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_mass_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_width_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | unbound |
| [vector_anisotropic_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | unbound |
| [vector_overlap_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap_published) | diagnostic | UNKNOWN | unbound |
| [vector_spiral_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral_published) | diagnostic | UNKNOWN | unbound |
| [img_stripes2_residual16](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | unbound |
| [img_bars4_residual16](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | unbound |
| [img_blobs4_residual16](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_square_named_v1](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [rotated100_affine_square_named_v1](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [staggered100_affine_square_named_v1](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_paired_laws_v1](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | unbound |
| [grid100_release07_cloud_named_v1](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | required | BLOCKED | matches |
| [ring_extension](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | BLOCKED | matches |

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | CHANGED |
| [unused_token_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | CHANGED |
| [ae_gan_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | CHANGED |
| [residual_student](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | CHANGED |
| [unipolar](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | CHANGED |
| [cover_leftover](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | CHANGED |
| [mid_scale_identity](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | CHANGED |
| [mode_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | CHANGED |
| [vector_two_broad](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | CHANGED |
| [vector_unequal_mass](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | BLOCKED | CHANGED |
| [vector_unequal_width](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | BLOCKED | CHANGED |
| [vector_anisotropic](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | BLOCKED | CHANGED |
| [vector_overlap](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | BLOCKED | CHANGED |
| [vector_spiral](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | CHANGED |
| [img_stripes2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | CHANGED |
| [img_bars4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | BLOCKED | CHANGED |
| [img_blobs4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | BLOCKED | CHANGED |
| [img_intensity2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | BLOCKED | CHANGED |
| [grid100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | matches |
| [rotated100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | matches |
| [staggered100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | matches |
| [img_intensity2_residual16](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | unbound |
| [vector_two_broad_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_mass_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_width_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | unbound |
| [vector_anisotropic_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | unbound |
| [vector_overlap_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap_published) | diagnostic | UNKNOWN | unbound |
| [vector_spiral_published](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral_published) | diagnostic | UNKNOWN | unbound |
| [img_stripes2_residual16](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | unbound |
| [img_bars4_residual16](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | unbound |
| [img_blobs4_residual16](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_square_named_v1](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [rotated100_affine_square_named_v1](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [staggered100_affine_square_named_v1](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | required | BLOCKED | matches |
| [ring_extension](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | BLOCKED | matches |

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | unbound |
| [two_pole_800_schedule800_diagnostic_v1](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage"></a>

## quality_coverage

**quality_coverage — revision 2**. [View declaration](../../../configs/forge/views/quality_coverage.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 0**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | BLOCKED | CHANGED |
| [unused_token_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | BLOCKED | CHANGED |
| [ae_gan_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | BLOCKED | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | BLOCKED | CHANGED |
| [residual_student](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | BLOCKED | CHANGED |
| [unipolar](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | BLOCKED | CHANGED |
| [cover_leftover](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | BLOCKED | CHANGED |
| [mid_scale_identity](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | BLOCKED | CHANGED |
| [mode_hold](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | BLOCKED | CHANGED |
| [vector_two_broad](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | BLOCKED | CHANGED |
| [vector_unequal_mass](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | BLOCKED | CHANGED |
| [vector_unequal_width](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | BLOCKED | CHANGED |
| [vector_anisotropic](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | BLOCKED | CHANGED |
| [vector_overlap](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | BLOCKED | CHANGED |
| [vector_spiral](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | BLOCKED | CHANGED |
| [img_stripes2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | BLOCKED | CHANGED |
| [img_bars4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | BLOCKED | CHANGED |
| [img_blobs4](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | BLOCKED | CHANGED |
| [img_intensity2](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | BLOCKED | CHANGED |
| [grid100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | BLOCKED | matches |
| [rotated100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | BLOCKED | matches |
| [staggered100](atlas.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | BLOCKED | matches |

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. CHANGED means the declared execution or evaluator differs from the recorded task; its earlier verdict is preserved.

<a name="cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). ae_gan_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; ae_gan_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). cover_leftover: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; cover_leftover: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_acquisition"></a>

### five_word_joint_acquisition

**five_word_joint_acquisition: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_acquisition.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). five_word_joint_acquisition: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; five_word_joint_acquisition: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1).

Recorded conditions: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

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
Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = True.

Declared budget: 20001 updates; timeout 900 seconds.

Current measurement: particle_cloud prior (sigma 0); generated_and_paired_reconstructed_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../five-word-joint/README.md)

<a name="cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_acquisition"></a>

### gaussian1d_acquisition

**gaussian1d_acquisition: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_acquisition.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1).

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

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

[Explanation and existing training artifacts](../../toy_audit/api_contract/gaussian1d/README.md)

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100"></a>

### grid100

**grid100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Current contract: **matches**. Current task contract matches the recorded conditions. grid100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; grid100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_14k"></a>

### grid100_14k

**grid100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_14k.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_paired_laws_v1"></a>

### grid100_affine_paired_laws_v1

**grid100_affine_paired_laws_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_paired_laws_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_square_named_v1"></a>

### grid100_affine_square_named_v1

**grid100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_affine_square_named_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100_release07_cloud_named_v1"></a>

### grid100_release07_cloud_named_v1

**grid100_release07_cloud_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100_release07_cloud_named_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_bars4"></a>

### img_bars4

**img_bars4: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). img_bars4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; img_bars4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_bars4_residual16"></a>

### img_bars4_residual16

**img_bars4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4_residual16.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_blobs4"></a>

### img_blobs4

**img_blobs4: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). img_blobs4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; img_blobs4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_blobs4_residual16"></a>

### img_blobs4_residual16

**img_blobs4_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4_residual16.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_intensity2"></a>

### img_intensity2

**img_intensity2: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). img_intensity2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; img_intensity2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_intensity2_residual16"></a>

### img_intensity2_residual16

**img_intensity2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2_residual16.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_stripes2"></a>

### img_stripes2

**img_stripes2: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). img_stripes2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; img_stripes2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_residual16"></a>

### img_stripes2_residual16

**img_stripes2_residual16: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2_residual16.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 600 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); enumerated_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity"></a>

### mid_scale_identity

**mid_scale_identity: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). mid_scale_identity: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; mid_scale_identity: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-mode_hold"></a>

### mode_hold

**mode_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). mode_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; mode_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-residual_student"></a>

### residual_student

**residual_student: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). residual_student: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; residual_student: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition"></a>

### ring16_acquisition

**ring16_acquisition: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). ring16_acquisition: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; ring16_acquisition: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1).

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
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-ring_extension"></a>

### ring_extension

**ring_extension: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Current contract: **matches**. Current task contract matches the recorded conditions. ring_extension: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; ring_extension: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ring_hold"></a>

### ring_hold

**ring_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Current contract: **matches**. Current task contract matches the recorded conditions. ring_hold: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; ring_hold: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-rotated100"></a>

### rotated100

**rotated100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Current contract: **matches**. Current task contract matches the recorded conditions. rotated100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; rotated100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-rotated100_14k"></a>

### rotated100_14k

**rotated100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_14k.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-rotated100_affine_square_named_v1"></a>

### rotated100_affine_square_named_v1

**rotated100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100_affine_square_named_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-staggered100"></a>

### staggered100

**staggered100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Current contract: **matches**. Current task contract matches the recorded conditions. staggered100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; staggered100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-staggered100_14k"></a>

### staggered100_14k

**staggered100_14k: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_14k.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-staggered100_affine_square_named_v1"></a>

### staggered100_affine_square_named_v1

**staggered100_affine_square_named_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100_affine_square_named_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery"></a>

### target_shift_recovery

**target_shift_recovery: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/target_shift_recovery.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [adaptation / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-3) · [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-7f9c23eb0e27-experiment-trajectory"></a>

### trajectory

**trajectory: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). trajectory: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; trajectory: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

Recorded conditions: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

Current pass criteria:

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

Declared budget: 400 updates; timeout 1800 seconds.

Current measurement: particle_cloud prior (sigma 0); conditional_prior_centers_with_scheduled_output_noise; weights live; output noise public_recipe_schedule.

<a name="cohort-cuda-7f9c23eb0e27-experiment-two_pole"></a>

### two_pole

**two_pole: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-two_pole_800_schedule800_diagnostic_v1"></a>

### two_pole_800_schedule800_diagnostic_v1

**two_pole_800_schedule800_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [k3p_two_pole_horizon / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-two_pole_800_schedule80_diagnostic_v1"></a>

### two_pole_800_schedule80_diagnostic_v1

**two_pole_800_schedule80_diagnostic_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [k3p_two_pole_horizon / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-unipolar"></a>

### unipolar

**unipolar: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). unipolar: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; unipolar: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold"></a>

### unused_token_hold

**unused_token_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). unused_token_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; unused_token_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). vector_anisotropic: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; vector_anisotropic: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic_published"></a>

### vector_anisotropic_published

**vector_anisotropic_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic_published.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_overlap"></a>

### vector_overlap

**vector_overlap: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). vector_overlap: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; vector_overlap: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_overlap_published"></a>

### vector_overlap_published

**vector_overlap_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap_published.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_spiral"></a>

### vector_spiral

**vector_spiral: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). vector_spiral: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; vector_spiral: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_spiral_published"></a>

### vector_spiral_published

**vector_spiral_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral_published.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad"></a>

### vector_two_broad

**vector_two_broad: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). vector_two_broad: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; vector_two_broad: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad_published"></a>

### vector_two_broad_published

**vector_two_broad_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad_published.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass"></a>

### vector_unequal_mass

**vector_unequal_mass: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). vector_unequal_mass: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; vector_unequal_mass: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass_published"></a>

### vector_unequal_mass_published

**vector_unequal_mass_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass_published.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width"></a>

### vector_unequal_width

**vector_unequal_width: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). vector_unequal_width: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation; vector_unequal_width: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width_published"></a>

### vector_unequal_width_published

**vector_unequal_width_published: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width_published.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

- [Original Atlas recipe and serving-law evidence](../continuous-baseline-20261003/README.md)
- [fresh_retest](../pr223-original-full-retest-stopped17-20261004/README.md)
- [native3_continuation](../pr223-native3-first-invalid-20261004/README.md)
- [native3_repaired_continuation](../pr223-native3-repaired-20261004/README.md)
- [Atlas diagnostic](../atlas-current-gpu-diagnostics-native-v2-20261003/README.md)
- [Atlas diagnostic](../atlas-named-gpu-diagnostics-native-v3-20261003/README.md)
- [Atlas diagnostic](../atlas-named-gpu-diagnostics-native-v3-20261003/README.md)
- [Atlas diagnostic](../atlas-named-gpu-diagnostics-native-v4b-20261004/README.md)
- [Atlas diagnostic](../atlas-named-gpu-diagnostics-native-v4b-20261004/README.md)
- [Atlas diagnostic](../atlas-named-gpu-diagnostics-native-v4b-20261004/README.md)
- [Atlas diagnostic](../atlas-word-retained-context-20261004/README.md)
- [C6 baseline selection and retained diagnosis](../c6-baseline-debug-20261003/README.md)
- [Word half-base rate contrast](../word-half-base-20261004/README.md)
- [atlas19_original](../continuous-baseline-20261003/README.md)
- [c6_hold](../continuous-baseline-20261003/README.md)
- [critic_balance](../critic-balance-20261003/README.md)
- [generator_step](../generator-step-20261003/README.md)
- [Other configurations and original evidence bindings](../technique-inventory.json)
- [Compiled experiment memory](../EXPERIMENT_MEMORY.md)

## Refresh

```sh
python reports/forge/regenerate_technique_inventory.py
```

This page is generated alongside the leaderboard. Register new source evidence before refreshing; editing a page cannot change a verdict or earn qualification.
