<!-- Generated Forge family report -->

# K3P without training output noise

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions) · [structural-ablation](../technique-inventory.md#tag-structural-ablation)

## Technique overview

Despite its broad historical family label, this variant removes only additive noise on generated outputs in the discriminator and generator training objectives. It retains critic-input noise and any noise intrinsic to the task's prior. The critic penalty, optimizer controls and learning-rate schedules remain. Clean public sampling was already separate from noisy training.

## Simplified pseudocode

G: generator; D: critic; D_bar: exponential moving average of the critic; x: real samples; z: sampled latent input; d: number of coordinates per critic input. norm is the Euclidean norm, mean averages samples, relu(a)=max(a,0), and softplus(a)=log(1+exp(a)). The training sketch shows the scalar GAN host; joint/auxiliary hosts add task-owned objectives. Remove additive generated-output training noise. stop_gradient detaches generated values from G and the prior during the critic update. When critic-input noise is enabled, D_bar is evaluated through a matching fresh-noise wrapper for the anchor term.

```text
Start from the K3P recipe and apply: generated_output_noise_std = 0
For each training iteration:
  Set G/D learning rates from their network schedule; set prior rate from its full-budget schedule.
  Draw real x and latent z from the task; fake = stop_gradient(G(z)) for the critic update.
  Evaluate D through the scheduled input-noise wrapper (fresh noise per forward); do not add generated-output noise.
  L_D = mean(softplus(D(fake) - D(x))).
  g_r = gradient of D(x) with respect to x; g_f = gradient of D(fake) with respect to fake.
  A = mean(norm(g_r)^2 / d) + mean(relu(norm(g_f)/sqrt(d) - cap)^2).
  r = last applied critic LR / largest applied critic LR; f = network LR floor.
  s = max(0, min(1, 2*r) - 2*f) / (1 - 2*f); before the first critic step, s = 1.
  If s < 1 first occurs, copy D into D_bar and set proximal_term = 0; subsequently use
    proximal_term = anchor_weight * mean(norm(g_r - gradient(D_bar,x))^2 / d).
  B = mean(relu(norm(g_r)-cap)^2) + mean(relu(norm(g_f)-cap)^2) + proximal_term.
  Every k-th penalty call, add k*coefficient/2 * (s*A + (1-s)*B); add zero on other calls (default k=1).
  Backpropagate L_D; after warmup, scale each critic gradient tensor down if its RMS exceeds
    guard_ratio * RMS expected from Adam's bias-corrected second moment; take the critic Adam step.
  Once the anchor has started, update D_bar as 0.999*D_bar + 0.001*D after each critic step.
  Resample latent input and recompute differentiable fake and critic scores using the updated D; retain declared training noise.
  L_G = mean(softplus(D(x) - D(fake))); backpropagate through G and learned prior with D fixed.
  If some latent rows have zero gradient, the cumulative observed-row fraction is below max_rate, and Adam history exists:
    scale observed-row Adam responses by 0.75+0.25*cos(current,last observed gradient); use 1 without row history.
  Otherwise leave the latent-row Adam response unchanged.
  Retain any host-eligible direct-particle response gain; update G and prior.
  Maintain any configured moving averages; score with the task's declared weights and sampling law.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Paired relativistic logistic by default: critic mean softplus(D(fake)-D(real)); generator reverses that score difference. Task-owned joint or reconstruction terms remain separate. |
| Optimizer | Inherited K3P Adam wrappers for critic and generator/prior, with recipe-specific moments and per-role rates. This ablation does not switch to native Adam without interventions. |
| Learning rates and annealing | Inherited scheduled training. G/D use the network cosine schedule, capped at a configured horizon; the learned prior uses its full-budget cosine schedule. In the public baseline, rates hold through 60% of their horizon before decaying to their floors. Resolved recorded configurations own exact rates and horizons. |
| Parameter-gradient clipping | Adaptive per-tensor critic gradient scaling remains via the spike guard; it is clipping-like and uses Adam second-moment history rather than a fixed global norm threshold. No generic global gradient-norm clipping is added by this candidate. Input-gradient penalties regularize the objective separately. |
| Critic penalties and anchors | Enabled: learning-rate-dependent K3P blend of early real-gradient regularization/fake caps and late real/fake caps plus a critic-gradient anchor. With lazy cadence k, apply k times this penalty every k-th call and zero otherwise; default k=1. |
| Damping and update guards | A2 acts only on eligible sparse latent rows: responses are scaled within [0.5,1] according to consecutive gradient agreement, while Adam's second moment still tracks the raw gradient. The critic spike guard remains. Direct sample-particle gain is a separate inherited, host-dependent mechanism; it does not automatically apply to every learned prior. A2 activates only when some rows have zero gradient, the cumulative observed-row fraction is below the configured max_rate (baseline 0.5), and Adam state exists; missing row history gives scale 1. |
| Training and sampling noise | Generated-output training noise is disabled. Scheduled critic-input noise remains; task-declared learned-MoG kernel noise also remains. Clean and noisy evaluation are separate cohorts. Critic-input noise is independently drawn on each forward of the wrapped critic, including gradient-penalty evaluation. |
| Parameter averaging and serving | The critic anchor and optional generator/prior moving averages remain. Live clean scoring stays separate from averaging and from any separately declared noisy evaluation. |

## Configuration differences

- The candidate makes one named mechanism removal; it does not replace task priors, initialization, training budgets, or numerical criteria.
- Settings above describe the declared ablation and public K3P baseline. Each recorded result retains its executed recipe/source; current implementation descriptions do not requalify historical measurements.
- Penalty laziness, exact noise magnitudes, role-specific rates, moment settings, and host-eligible controls must be read from the selected configuration bindings rather than assumed to be identical across tasks.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [configs/forge/ideas/k3p-no-output-noise-diagnostic.json](../../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/k3p.py](../../../particlegan/k3p.py)
- [particlegan/grad_regularizers.py](../../../particlegan/grad_regularizers.py)
- [particlegan/gan_loss.py](../../../particlegan/gan_loss.py)
- [particlegan/training.py](../../../particlegan/training.py)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

<a name="cohort-cuda-0d83d78027c5"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [k3p-no-output-noise-diagnostic](../../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json).

Recorded qualification: **tier 0**, discriminator_stability revision 5. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `21ec7e3f89402b5d0c77669d5fba0e36bcf4e690069cfdceab383520006c525f`. Candidate revision: `897508cbe0391de6cd90c7042c8991a15a35971dd85d1d86f2bc95754f84fbee`. Runtime cohort: `03c79722db9abe1daa6c02f9a6377f95e5b2368df608d4dc46b064be3e46eb33`.

[Frozen numerical evidence](../technique-evidence/ddde64ee936114becac42863847a51e88ec0301bad9f7f69767d0a2fbc3f3d69.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: current_measurement. Complete current Tier 1 measurement at one frozen recipe and executed source. FAIL completes a measurement; calibration and confirmation remain separate.

</details>

Complete current Tier 1 measurement in: discriminator_stability; additional scoped probes: clockfree_audit_measurement_v1. PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation) | [2/3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) | [0(*)/19](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) | [0(*)/1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) | [2(*)/23](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation) |
| [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous) | [2/4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | [0(*)/19](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) | [0(*)/7](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | [2(*)/30](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous) |
| [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability) | [3(*)/6](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | [0(*)/19](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | [0(*)/2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) | [3(*)/27](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability) |
| [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison) | [2/3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) | [0(*)/19](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) | [0(*)/2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) | [2(*)/24](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison) |
| [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer) | [2/3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) | [0(*)/19](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) | [0(*)/2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | [2(*)/24](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer) |
| [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage) | [2/3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | [0(*)/19](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | [0/0](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-3) | [2(*)/22](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) | [0(*)/7](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1) | [0/0](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2) | [0/0](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3) | [0(*)/7](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) |

\* indicates incomplete results, including changed or unbound current contracts.

<a name="cohort-cuda-0d83d78027c5-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ae_gan_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | matches |
| [clockfree_audit_measurement_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | FAIL | matches |
| [five_word_joint_acquisition](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition) | [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | PASS | CHANGED |
| [gaussian1d_acquisition](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition) | [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | FAIL | matches |
| [ring16_acquisition](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | FAIL | matches |
| [two_pole](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | FAIL | matches |
| [unused_token_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [cover_leftover](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [grid100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_bars4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_blobs4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_intensity2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_stripes2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [mid_scale_identity](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [mode_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [residual_student](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [rotated100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [staggered100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [trajectory](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [unipolar](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_anisotropic](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_overlap](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_spiral](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_two_broad](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_unequal_mass](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_unequal_width](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [clockfree_audit](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [grid100_14k](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [ring_extension](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches |
| [ring_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches |
| [rotated100_14k](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [staggered100_14k](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [target_shift_recovery](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | [adaptation](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-3), [clockfree_continuous](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | matches |
| [unused_token_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [target_shift_recovery](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | unbound |

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

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | matches |
| [unused_token_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |
| [clockfree_audit_measurement_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | required | FAIL | matches |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [clockfree_audit](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | required | UNKNOWN | unbound |
| [ring_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |
| [grid100_14k](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | required | UNKNOWN | unbound |
| [rotated100_14k](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | required | UNKNOWN | unbound |
| [staggered100_14k](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | required | UNKNOWN | unbound |
| [target_shift_recovery](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 5**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 19 / 2**.

Calibration: **provisional**. Expanded six-task Tier 1 placement is provisional and requires bounded calibration. Revision 3 and prior profiles retain their original tasks and evidence; a standalone scalar pass gives no whole-view/default credit.

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [gaussian1d_acquisition](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition) | required | FAIL | matches |
| [two_pole](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | matches |
| [unused_token_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |
| [ring16_acquisition](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | required | FAIL | matches |
| [five_word_joint_acquisition](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition) | required | PASS | CHANGED |
| [clockfree_audit_measurement_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | diagnostic | FAIL | matches |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | matches |
| [unused_token_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |
| [img_intensity2_residual16](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | unbound |
| [vector_two_broad_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_mass_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_width_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | unbound |
| [vector_anisotropic_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | unbound |
| [vector_overlap_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | unbound |
| [vector_spiral_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | unbound |
| [img_stripes2_residual16](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | unbound |
| [img_bars4_residual16](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | unbound |
| [img_blobs4_residual16](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_square_named_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [rotated100_affine_square_named_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [staggered100_affine_square_named_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_paired_laws_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | unbound |
| [grid100_release07_cloud_named_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | matches |
| [unused_token_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |
| [img_intensity2_residual16](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | unbound |
| [vector_two_broad_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_mass_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_width_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | unbound |
| [vector_anisotropic_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | unbound |
| [vector_overlap_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | unbound |
| [vector_spiral_published](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | unbound |
| [img_stripes2_residual16](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | unbound |
| [img_bars4_residual16](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | unbound |
| [img_blobs4_residual16](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_square_named_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [rotated100_affine_square_named_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [staggered100_affine_square_named_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | unbound |
| [two_pole_800_schedule800_diagnostic_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | unbound |

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

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | matches |
| [unused_token_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

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

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [two_pole_tier1_policy_selected_cloud_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [unused_token_hold_tier1_policy_selected_cloud_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [clockfree_audit_tier1_policy_selected_cloud_v1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-0d83d78027c5-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. CHANGED means the declared execution or evaluator differs from the recorded task; its earlier verdict is preserved.

<a name="cohort-cuda-0d83d78027c5-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Current contract: **matches**. Current task contract matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00133647 | <= 0.35 | PASS |
| recon_mse | 0.00642514 | <= 0.05 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/9daa17b88d3949438b8cb9887e64b442.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-training-noise/9daa17b88d3949438b8cb9887e64b442/ae_gan_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [clockfree_continuous / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Current contract: **matches**. Current task contract matches the recorded conditions. step_label changed the update or common-prefix state

Actual task device: `0` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/39568145a2a24c5291060e6ab48b23a8.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-training-noise/39568145a2a24c5291060e6ab48b23a8/clockfree_audit_measurement_v1.gif); 4 saved observations; no new optimizer updates or sampling draws.

Recorded clock parity diagnostics:

| Condition | Exact state digest equality |
| --- | --- |
| evaluation_cadence | equal |
| horizon | different |
| restart | equal |
| step_label | different |

Recorded unexplained clock dependencies: **3**.

- learning-rate annealing depends on completed steps and horizon
- input-noise annealing depends on completed steps and horizon
- critic guard releases at a fixed minimum update count

[Certified parity digests and source audit](../tier1-completion/scoped-evidence.json). These display diagnostics preserve the recorded gate FAIL.

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

<a name="cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition"></a>

### five_word_joint_acquisition

**five_word_joint_acquisition: PASS**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_acquisition.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| minimum_reconstruction_token_probability | 0.999977 | >= 0.9 | PASS |
| modes | 5 | == 5 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| sample_count | 1024 | >= 1024 | PASS |

Recorded terminal passing observations: **17**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/531e90cf76cf4e45815ec1786e3a9f3c.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-training-noise/531e90cf76cf4e45815ec1786e3a9f3c/five_word_joint_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

<a name="cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `419dbbd4aa5116e093e543cd7cacc904185408806cefbc8758004edabcecd048`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition"></a>

### gaussian1d_acquisition

**gaussian1d_acquisition: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_acquisition.json).

Current contract: **matches**. Current task contract matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.0314938 | <= 0.05 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.0296138 | <= 0.2 | PASS |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 1.02223 | <= 1.2 | PASS |

Recorded terminal passing observations: **4**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/fca8da49a5a14f7da875e55c90e22347.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-training-noise/fca8da49a5a14f7da875e55c90e22347/gaussian1d_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

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

<a name="cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-grid100"></a>

### grid100

**grid100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [clockfree_continuous / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 7.69553 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.319935 | >= 0.15 | PASS |
| hq | 0.904297 | >= 0.85 | PASS |
| mass_tv | 0.0852051 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/06e5d0bec7fb43d283699ffe8b5ed989.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-training-noise/06e5d0bec7fb43d283699ffe8b5ed989/ring16_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

<a name="cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [clockfree_continuous / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [clockfree_continuous / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [clockfree_continuous / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [clockfree_continuous / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [adaptation / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) · [clockfree_continuous / Tier 3](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-0d83d78027c5-experiment-trajectory"></a>

### trajectory

**trajectory: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.130802 | <= 1 | PASS |
| mean_abs | 0.264117 | >= 0.3 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/09504ecb601543ba80058d46bc456757.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-training-noise/09504ecb601543ba80058d46bc456757/two_pole.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [k3p_two_pole_horizon / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [k3p_two_pole_horizon / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.951601 | >= 0.85 | PASS |
| unused_hold | 0.998174 | >= 0.85 | PASS |

Recorded terminal passing observations: **11**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/37c259fdac504f92b4636d50e94d9461.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-training-noise/37c259fdac504f92b4636d50e94d9461/unused_token_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `ef8ccde8d1fa54af8bfce01c044e3671de8131c980eb4e8022d12ffc8caf51d8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [formulation_comparison / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-training-noise.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base optimizer; the spike guard, A2 rule and K3P handover are repository-specific additions.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Paired relativistic adversarial loss.
- [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406). Related work on zero-centered input-gradient regularization; it is not a paper specifying the full K3P mechanism or this ablation.
