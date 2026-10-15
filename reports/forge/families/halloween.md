<!-- Generated Forge family report -->

# Halloween optimizer/loss transfer

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [constant-learning-rate](../technique-inventory.md#tag-constant-learning-rate) · [tensorflow-v1-adam](../technique-inventory.md#tag-tensorflow-v1-adam)

## Technique overview

Halloween transfers the inspected historical G/D Adam settings and [-1,1,1] least-squares labels through the public API. Original HyperGAN execution and trained success remain unbound. This new family is unmeasured.

## Simplified pseudocode

G is the generator, E an optional encoder, D the critic and P the task-owned prior; t counts optimizer applications.

```text
Draw task-owned real and latent batches from isolated streams.
Minimize half the mean of (D(real)-1)^2 plus (D(fake)+1)^2 for D.
Minimize half the mean of (D(fake)-1)^2 for G; joint hosts add the reversed real-stream encoder term.
Apply role-specific dense legacy Adam, incrementing each optimizer application clock.
Save role clocks, powers, moments, model weights and every consumed stream.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Least squares with fake/real/G labels (-1,1,1); joint encoder stream is explicit. |
| Optimizer | Dense TensorFlow-v1 Adam formula implemented in PyTorch; different G/D betas and epsilon, no AMSGrad or weight decay. |
| Learning rates and annealing | Constant rates: G .008020980209802098, D .003947939479394794. Prior multiplier 2 with its own (0,.999) betas and 1e-8 epsilon. Original decay is unbound; explicit exponential whole-update schedules are optional new variants. |
| Parameter-gradient clipping | No parameter-gradient clipping. |
| Critic penalties and anchors | Zero additional critic penalty in the transfer preset. |
| Damping and update guards | No A2 damping, critic guard, anchor or direct-particle response. |
| Training and sampling noise | No additive training input/output noise. Task-owned MoG kernel noise and clean sampling remain explicit. |
| Parameter averaging and serving | Live parameters; no averaged serving. |

## Configuration differences

- Unmeasured structural transfer: no default adoption or original-result credit.
- One global trainer configuration per matched task comparison. Architecture, data, prior, initialization, budget and sampling belong to tasks.
- See docs/forge-search-spaces.md for finite random subsets, conditional categories and immutable manifests.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlegan/tensorflow_adam.py](../../../particlegan/tensorflow_adam.py)
- [particlegan/gan_loss.py](../../../particlegan/gan_loss.py)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [configs/forge/ideas/halloween-optimizer-loss-v1.json](../../../configs/forge/ideas/halloween-optimizer-loss-v1.json)
- [reports/forge/hypergan-search-audit/halloween-optimizer-loss.json](../hypergan-search-audit/halloween-optimizer-loss.json)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

**Historical cohort navigation.** This original recipe/prior row is retained for existing links. [Current GAN v3 solution family](release07-gan-v3.md) uses one whole selected configuration and task-declared priors; these historical cells are not pooled into it.

<a name="cohort-cuda-c195899a64af"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [halloween-optimizer-loss-v1](../../../configs/forge/ideas/halloween-optimizer-loss-v1.json).

Recorded qualification: **tier 0**, discriminator_stability revision 9. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `d276c5a7344fab6ec5de7b314d3982af5b0ef8027c8b89c01376ae366844a9cb`. Candidate revision: `dd0600f49a549bab20b83a56b1e16978a34c97769e7f2d67645f738720de452f`. Runtime cohort: `c48d91df811e2899e94cc493df3bb19f51e75514dd9c232c1d399b24c13b4cc1`.

[Frozen numerical evidence](../technique-evidence/83f549889bf36b44bda98c8fe4c0f1600ef0cdd1d12c529f9e480ffcfdc93512.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Current-source pre-run whole candidate choice is partially BLOCKED or unmeasured; retain exact display evidence with no complete current-measurement, qualification or default-adoption credit.

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation) | [0/3 (3 not run)](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-1) | [0/19 (19 not run)](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) | [0/1 (1 not run)](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-3) | [0/23 (23 not run)](halloween.md#cohort-cuda-c195899a64af-adaptation) |
| [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous) | [0/4 (4 not run)](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) | [0/19 (19 not run)](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) | [0/7 (7 not run)](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | [0/30 (30 not run)](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous) |
| [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability) | [0/6 (6 not run)](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | [0/21 (21 not run)](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | [0/2 (2 not run)](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) | [0/29 (29 not run)](halloween.md#cohort-cuda-c195899a64af-discriminator_stability) |
| [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison) | [0/3 (3 not run)](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) | [0/19 (19 not run)](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) | [0/2 (2 not run)](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) | [0/24 (24 not run)](halloween.md#cohort-cuda-c195899a64af-formulation_comparison) |
| [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer) | [0/3 (3 not run)](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) | [0/19 (19 not run)](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) | [0/2 (2 not run)](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | [0/24 (24 not run)](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer) |
| [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage) | [0/3 (3 not run)](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | [0/19 (19 not run)](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | [0/0](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-3) | [0/22 (22 not run)](halloween.md#cohort-cuda-c195899a64af-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage) | [0/7 (7 not run)](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1) | [0/0](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-2) | [0/0](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-3) | [0/7 (7 not run)](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage) |

Not run counts missing executions; incompatible counts preflight-blocked formulation/task contracts without execution. These counts are separate and remain in the required denominator. PASS and FAIL both count as executed. Attempted errors retain their status and cause; changes to today's test definition are shown separately from execution coverage.

<a name="cohort-cuda-c195899a64af-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | UNKNOWN | changed since run |
| [clockfree_audit_measurement_v1](halloween.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) | UNKNOWN | changed since run |
| [five_word_joint_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | UNKNOWN | changed since run |
| [ring16_acquisition](halloween.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | UNKNOWN | changed since run |
| [two_pole](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | UNKNOWN | changed since run |
| [unused_token_hold](halloween.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [five_word_joint_hold](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](halloween.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](halloween.md#cohort-cuda-c195899a64af-experiment-grid100_14k) | [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](halloween.md#cohort-cuda-c195899a64af-experiment-ring_extension) | [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | UNKNOWN | changed since run |
| [ring_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ring_hold) | [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3), [discriminator_stability](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3), [formulation_comparison](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3), [host_profile_transfer](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | UNKNOWN | changed since run |
| [rotated100_14k](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100_14k) | [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100_14k) | [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](halloween.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | [adaptation](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-3), [clockfree_continuous](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-c195899a64af-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](halloween.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](halloween.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1"></a>

## bcap-develop-integration-deeper-diagnostic-v1

**bcap-develop-integration-deeper-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1"></a>

## bcap-projection-baseline-repair-diagnostic-v1

**bcap-projection-baseline-repair-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [two_pole](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole) | diagnostic | UNKNOWN | changed since run |
| [unused_token_hold](halloween.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | diagnostic | UNKNOWN | changed since run |
| [ae_gan_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | diagnostic | UNKNOWN | changed since run |
| [ring16_acquisition](halloween.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-bcap-tier1-stability-diagnostic-v1"></a>

## bcap-tier1-stability-diagnostic-v1

**bcap-tier1-stability-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-bcap-tier1-stability-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-bcap-tier1-stability-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-bcap-tier1-stability-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-bcap-tier1-stability-repairs-diagnostic-v1"></a>

## bcap-tier1-stability-repairs-diagnostic-v1

**bcap-tier1-stability-repairs-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-bcap-tier1-stability-repairs-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-bcap-tier1-stability-repairs-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-bcap-tier1-stability-repairs-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-bcap_convolution_images"></a>

## bcap_convolution_images

**bcap_convolution_images — revision 1**. [View declaration](../../../configs/forge/views/bcap_convolution_images.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

<a name="cohort-cuda-c195899a64af-bcap_convolution_images-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-bcap_convolution_images-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-bcap_convolution_images-tier-3"></a>

### Tier 3

No experiments assigned.

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
| [two_pole](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](halloween.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |
| [clockfree_audit_measurement_v1](halloween.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](halloween.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | changed since run |
| [ring_extension](halloween.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | changed since run |
| [grid100_14k](halloween.md#cohort-cuda-c195899a64af-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](halloween.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 9**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | required | UNKNOWN | changed since run |
| [two_pole](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](halloween.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |
| [ring16_acquisition](halloween.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | required | UNKNOWN | changed since run |
| [five_word_joint_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](halloween.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | required | UNKNOWN | changed since run |
| [five_word_joint_hold](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | changed since run |
| [ring_extension](halloween.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](halloween.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | changed since run |
| [img_intensity2_residual16](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](halloween.md#cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](halloween.md#cohort-cuda-c195899a64af-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](halloween.md#cohort-cuda-c195899a64af-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | changed since run |
| [ring_extension](halloween.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](halloween.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | changed since run |
| [img_intensity2_residual16](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](halloween.md#cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | changed since run |
| [ring_extension](halloween.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | UNKNOWN | changed since run |
| [unused_token_hold](halloween.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | UNKNOWN | changed since run |
| [ae_gan_hold](halloween.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1"></a>

## recipe-prior-refactor-k3p-diagnostic-v1

**recipe-prior-refactor-k3p-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1"></a>

## recipe-prior-refactor-ka2-diagnostic-v1

**recipe-prior-refactor-ka2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1"></a>

## recipe-prior-refactor-r1r2-diagnostic-v1

**recipe-prior-refactor-r1r2-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1"></a>

## recipe-prior-refactor-release07-gan-v3-diagnostic-v1

**recipe-prior-refactor-release07-gan-v3-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [five_word_joint_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | diagnostic | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | diagnostic | UNKNOWN | changed since run |
| [gaussian1d_stability](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | diagnostic | UNKNOWN | changed since run |
| [five_word_joint_hold](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | diagnostic | UNKNOWN | recorded definition unavailable |
| [trajectory](halloween.md#cohort-cuda-c195899a64af-experiment-trajectory) | diagnostic | UNKNOWN | changed since run |
| [residual_student](halloween.md#cohort-cuda-c195899a64af-experiment-residual_student) | diagnostic | UNKNOWN | changed since run |
| [unipolar](halloween.md#cohort-cuda-c195899a64af-experiment-unipolar) | diagnostic | UNKNOWN | changed since run |
| [cover_leftover](halloween.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | diagnostic | UNKNOWN | changed since run |
| [mid_scale_identity](halloween.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | diagnostic | UNKNOWN | changed since run |
| [mode_hold](halloween.md#cohort-cuda-c195899a64af-experiment-mode_hold) | diagnostic | UNKNOWN | changed since run |
| [vector_two_broad](halloween.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_mass](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | diagnostic | UNKNOWN | changed since run |
| [vector_unequal_width](halloween.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | diagnostic | UNKNOWN | changed since run |
| [vector_anisotropic](halloween.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | diagnostic | UNKNOWN | changed since run |
| [vector_overlap](halloween.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | diagnostic | UNKNOWN | changed since run |
| [vector_spiral](halloween.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | diagnostic | UNKNOWN | changed since run |
| [img_stripes2](halloween.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | diagnostic | UNKNOWN | changed since run |
| [img_bars4](halloween.md#cohort-cuda-c195899a64af-experiment-img_bars4) | diagnostic | UNKNOWN | changed since run |
| [img_blobs4](halloween.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | diagnostic | UNKNOWN | changed since run |
| [img_intensity2](halloween.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | diagnostic | UNKNOWN | changed since run |
| [grid100](halloween.md#cohort-cuda-c195899a64af-experiment-grid100) | diagnostic | UNKNOWN | changed since run |
| [rotated100](halloween.md#cohort-cuda-c195899a64af-experiment-rotated100) | diagnostic | UNKNOWN | changed since run |
| [staggered100](halloween.md#cohort-cuda-c195899a64af-experiment-staggered100) | diagnostic | UNKNOWN | changed since run |

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-c195899a64af-selected-family-coverage-diagnostic-v1"></a>

## selected-family-coverage-diagnostic-v1

**selected-family-coverage-diagnostic-v1 — revision 1**. [View declaration](../../../configs/forge/views/selected-family-coverage-diagnostic-v1.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-c195899a64af-selected-family-coverage-diagnostic-v1-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](halloween.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-selected-family-coverage-diagnostic-v1-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-c195899a64af-selected-family-coverage-diagnostic-v1-tier-3"></a>

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](halloween.md#cohort-cuda-c195899a64af-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](halloween.md#cohort-cuda-c195899a64af-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](halloween.md#cohort-cuda-c195899a64af-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](halloween.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](halloween.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](halloween.md#cohort-cuda-c195899a64af-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](halloween.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

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

**ae_gan_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 1](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

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

Execution: **not run**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `607689de7b76e774f30f3d0cc26f5c5524b40d36ed86e353f29c29134a5a8c8b`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [selected-family-coverage-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-selected-family-coverage-diagnostic-v1-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 1](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **not run**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Execution: **not run**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

**gaussian1d_smoke: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap_convolution_images / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap_convolution_images-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

**ring16_acquisition: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [discriminator_stability / Tier 1](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

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

Execution: **not run**.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [clockfree_continuous / Tier 3](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Execution: **not run**.

Used by: [adaptation / Tier 3](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-3) · [clockfree_continuous / Tier 3](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-c195899a64af-experiment-trajectory"></a>

### trajectory

**trajectory: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

**two_pole: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 1](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [bcap-tier1-stability-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-tier1-stability-diagnostic-v1-tier-1) · [bcap-tier1-stability-repairs-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-tier1-stability-repairs-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

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

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](halloween.md#cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1).

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

Execution: **not run**.

Used by: [k3p_two_pole_horizon / Tier 1](halloween.md#cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1).

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

Execution: **not run**.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

**unused_token_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 1](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 1](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

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

Execution: **not run**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `d12cabad3799d268a92325ee128599c171ea5515d474971be4f6d4cb7850c65a`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](halloween.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **not run**.

Used by: [adaptation / Tier 2](halloween.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [bcap-develop-integration-deeper-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-develop-integration-deeper-diagnostic-v1-tier-1) · [bcap-projection-baseline-repair-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-bcap-projection-baseline-repair-diagnostic-v1-tier-1) · [clockfree_continuous / Tier 2](halloween.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](halloween.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](halloween.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) · [recipe-prior-refactor-k3p-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-k3p-diagnostic-v1-tier-1) · [recipe-prior-refactor-ka2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-ka2-diagnostic-v1-tier-1) · [recipe-prior-refactor-r1r2-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-r1r2-diagnostic-v1-tier-1) · [recipe-prior-refactor-release07-gan-v3-diagnostic-v1 / Tier 1](halloween.md#cohort-cuda-c195899a64af-recipe-prior-refactor-release07-gan-v3-diagnostic-v1-tier-1).

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

Execution: **not run**.

Used by: [formulation_comparison / Tier 2](halloween.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](halloween.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

- [Inspected TensorFlow v1.3 Adam source](https://github.com/tensorflow/tensorflow/blob/9e76bf324f6bac63137a02bb6e6ec9120703ea9b/tensorflow/python/training/adam.py). Dense Adam epsilon and bias-correction law
