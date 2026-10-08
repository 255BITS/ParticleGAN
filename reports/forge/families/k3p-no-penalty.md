<!-- Generated Forge family report -->

# K3P without critic penalty

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions) · [structural-ablation](../technique-inventory.md#tag-structural-ablation)

## Technique overview

K3P without critic penalty sets the full critic regularization coefficient to zero. That removes both its early gradient regularization and its later caps and anchor contribution from the objective. K3P's optimizer interventions remain, so this is a test of critic regularization rather than plain Adam training.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior, $x$ a real batch, $z\sim P$ a latent batch and $y$ the generated batch, including any declared output noise. $C$ is the critic actually evaluated: $C=D$ without input noise; otherwise every forward evaluates $D(u+\epsilon_{\mathrm{in}})$ with a fresh draw. $\mathbb E$ means a sample mean, $\lVert\cdot\rVert_2$ is the Euclidean norm, $(a)_+=\max(a,0)$ and $\operatorname{softplus}(a)=\log(1+e^a)$. The scalar sketch detaches $y$ during the critic update and resamples differentiable $y$ after that update; joint and auxiliary hosts retain task-owned objectives. $h_i$ is latent row $i$'s current gradient, $h_i^{\mathrm{prev}}$ its last observed gradient, $\rho_i$ the A2 response multiplier and $\Delta z_i$ the corresponding row update. $\operatorname{cos}$ denotes the implementation's bounded cosine similarity, including its handling of zero-length history. For the guard, $g_p$ is a critic tensor gradient, $v_p$ its Adam second moment, $\tau_p$ its stored step count and $c_{\mathrm{guard}}$ its threshold.

**Default paired adversarial loss**

$$
\begin{aligned}\ell_D&=\mathbb E\!\left[\operatorname{softplus}(C(y)-C(x))\right],\\\ell_G&=\mathbb E\!\left[\operatorname{softplus}(C(x)-C(y))\right].\end{aligned}
$$

These scalar losses are minimized. The critic objective is $\ell_D+R$; the generator adds only its declared auxiliary objectives. Critic fakes are detached, and generator fakes and both scores are recomputed after the critic step. $C$ includes any declared fresh input noise.

**Removed critic penalty**

$$
\lambda=0,\qquad R=0,\qquad L_D=\ell_D.
$$

This ablation removes all input-gradient and anchor loss contributions. The inherited K3P Adam wrappers, spike guard, eligible A2 and host-bound direct-particle controls remain; this is not a switch to native Adam.

**A2 sparse-row response**

$$
\begin{aligned}\rho_i&=\begin{cases}0.75+0.25\operatorname{cos}(h_i,h_i^{\mathrm{prev}}),&\text{with row history},\\1,&\text{without row history},\end{cases}\\\Delta z_i^{\mathrm{A2}}&=\rho_i\Delta z_i^{\mathrm{Adam}}.\end{aligned}
$$

A2 acts only on an eligible row-local table with existing Adam state, some zero-gradient rows, and cumulative observed-row fraction below the configured cutoff (baseline $0.5$). Then $\rho_i\in[0.5,1]$. Adam still accumulates the raw gradient second moment; unsupported or ineligible hosts keep their ordinary response.

**Adam-state critic spike guard**

$$
\begin{aligned}v_p^{\mathrm{RMS}}&=\frac{\operatorname{mean}(v_p)}{1-\beta_2^{\tau_p}},\\u_p&=\frac{\operatorname{RMS}(g_p)}{\sqrt{\max(v_p^{\mathrm{RMS}},10^{-30})}},\\g_p^{\mathrm{guarded}}&=\begin{cases}\dfrac{c_{\mathrm{guard}}}{u_p}g_p,&\tau_p\ge\tau_{\min}\text{ and }u_p>c_{\mathrm{guard}},\\g_p,&\text{otherwise}.\end{cases}\end{aligned}
$$

This clipping rule applies per critic tensor only after its Adam history reaches the declared warmup (baseline $\tau_{\min}=200$ steps). $g_p$ is its parameter gradient, $v_p$ the stored second moment, $\tau_p$ its Adam step count and $c_{\mathrm{guard}}=5$ the baseline threshold. With AMSGrad use its stored running maximum. Before warmup the guard leaves gradients unchanged.

## Simplified pseudocode

```text
Start from the K3P recipe and apply: coefficient = 0; penalty = 0
For each training iteration:
  Set G/D learning rates from their network schedule; set prior rate from its full-budget schedule.
  Draw real x and latent z from the task; fake = stop_gradient(G(z)) for the critic update.
  Evaluate D through the scheduled input-noise wrapper (fresh noise per forward); add generated-output training noise to fake.
  Compute the paired critic adversarial loss shown above.
  Do not add any critic gradient penalty to L_D.
  Backpropagate L_D; after warmup, scale each critic gradient tensor down if its RMS exceeds
    guard_ratio * RMS expected from Adam's bias-corrected second moment; take the critic Adam step.
  Resample latent input and recompute differentiable fake and critic scores using the updated D; retain declared training noise.
  Compute the paired generator adversarial loss from the updated critic; backpropagate through G and learned prior with D fixed.
  If some latent rows have zero gradient, the cumulative observed-row fraction is below max_rate, and Adam history exists:
    Scale observed-row Adam responses using the A2 cosine-agreement rule; leave rows without history unchanged.
  Otherwise leave the latent-row Adam response unchanged.
  Retain any host-eligible direct-particle response gain; update G and prior.
  Maintain any configured moving averages; score with the task's declared weights and sampling law.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Paired relativistic logistic by default: critic $\mathbb E[\operatorname{softplus}(C(y)-C(x))]$; generator reverses that score difference. Task-owned joint or reconstruction terms remain separate. |
| Optimizer | Inherited K3P Adam wrappers for critic and generator/prior, with recipe-specific moments and per-role rates. This ablation does not switch to native Adam without interventions. |
| Learning rates and annealing | Inherited scheduled training. $G$ and $D$ use the network cosine schedule, capped at a configured horizon; the learned prior uses its full-budget cosine schedule. In the public baseline, rates hold through $60\%$ of their horizon before decaying to their floors. Resolved recorded configurations own exact rates and horizons. |
| Parameter-gradient clipping | Adaptive per-tensor critic gradient scaling remains via the spike guard; it is clipping-like and uses Adam second-moment history rather than a fixed global norm threshold. No generic global gradient-norm clipping is added by this candidate. |
| Critic penalties and anchors | Disabled: coefficient zero removes the entire K3P penalty, including the anchor contribution. The optimizer can retain anchor bookkeeping, which contributes no penalty to the objective. |
| Damping and update guards | A2 acts only on eligible sparse latent rows: responses are scaled within $[0.5,1]$ according to consecutive gradient agreement, while Adam's second moment still tracks the raw gradient. The critic spike guard remains. Direct sample-particle gain is a separate inherited, host-dependent mechanism; it does not automatically apply to every learned prior. A2 activates only when some rows have zero gradient, the cumulative observed-row fraction is below the configured max_rate (baseline $0.5$), and Adam state exists; missing row history gives scale $1$. |
| Training and sampling noise | Inherited scheduled critic-input noise and generated-output training noise. Prior sampling noise belongs to the task. Public clean sampling does not automatically include the generated-output training noise. Critic-input noise is independently drawn on each forward of the wrapped critic, including gradient-penalty evaluation. |
| Parameter averaging and serving | No effective critic-anchor force because the whole penalty is zero. Critic-anchor bookkeeping and generator/prior moving averages may remain; they do not change the live scoring law. |

## Configuration differences

- The candidate makes one named mechanism removal; it does not replace task priors, initialization, training budgets, or numerical criteria.
- Settings above describe the declared ablation and public K3P baseline. Each recorded result retains its executed recipe/source; current implementation descriptions do not requalify historical measurements.
- Penalty laziness, exact noise magnitudes, role-specific rates, moment settings, and host-eligible controls must be read from the selected configuration bindings rather than assumed to be identical across tasks.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [configs/forge/ideas/forge-no-critic-penalty.json](../../../configs/forge/ideas/forge-no-critic-penalty.json)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/k3p.py](../../../particlegan/k3p.py)
- [particlegan/grad_regularizers.py](../../../particlegan/grad_regularizers.py)
- [particlegan/gan_loss.py](../../../particlegan/gan_loss.py)
- [particlegan/training.py](../../../particlegan/training.py)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

**Configuration detail for [K3P](k3p.md).** Its optimizer or settings do not create a separate solution family. This page preserves the original configuration evidence and diagnostics.

<a name="cohort-cuda-1bf9d7d34422"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [forge-no-critic-penalty](../../../configs/forge/ideas/forge-no-critic-penalty.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `6269a18ac4f82564cb16ba19afa4b3dd2f836a2b4085fbe5aeb81a35a453e895`. Candidate revision: `16605dbcd6cddb5d25147cd3e7f736152a4466b8ec831f2c08d119acf77846a2`. Runtime cohort: `ea17fc1a20c76725e065e73a94f1c766bd13ae6aec625dd108486c29ca77b1c2`.

[Frozen numerical evidence](../technique-evidence/7c4e188f1ee495526c2decf6f02a45f392123aaf4f33f1f6fb9a398fe3ac05e7.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: current_measurement. Preserve the round's pre-run whole candidate choice in its freshly measured CUDA source cohort; no task pooling, outcome-based recipe reselection, calibration or default adoption.

</details>

Complete current Tier 1 measurement in: discriminator_stability. PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3(*)/23](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3(*)/30](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [3(*)/6](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [3(*)/29](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3(*)/24](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3(*)/24](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3(*)/22](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) | [0(*)/7](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1) | [0/0](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-2) | [0/0](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-3) | [0(*)/7](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-1bf9d7d34422-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | FAIL | matches recorded run |
| [five_word_joint_smoke](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [ring16_acquisition](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [two_pole](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [five_word_joint_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [trajectory](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | [adaptation](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | required | FAIL | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 8**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | required | FAIL | matches recorded run |
| [two_pole](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [ring16_acquisition](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | required | FAIL | matches recorded run |
| [five_word_joint_smoke](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | required | UNKNOWN | matches recorded run |
| [five_word_joint_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

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

Used by: [adaptation / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00691416 | <= 0.35 | PASS |
| recon_mse | 0.00685878 | <= 0.05 | PASS |

Recorded terminal passing observations: **21**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/0f0f6707ff9a462c8766b9039bbddf4b.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. step_label changed the update or common-prefix state

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/43aa366a7d5444c1a33ce9f3a22fde89.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2).

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

Used by: [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.10327 |
| finite_fraction | 1 |
| mean | 2.07333 |
| mean_error_sigma | 0.146655 |
| sample_count | 4096 |
| std | 0.444676 |
| std_ratio | 0.889353 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/d98386bc78ba41538fc1010d966095a0.json)

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

Used by: [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 77.6766 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 1.56968 | >= 0.15 | PASS |
| hq | 0.351562 | >= 0.85 | FAIL |
| mass_tv | 0.199707 | <= 0.15 | FAIL |
| modes | 8 | >= 16 | FAIL |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d4f13271aada4ce29d90e5abc3578ae9.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) · [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.942821 | <= 1 | PASS |
| mean_abs | 0.426749 | >= 0.3 | PASS |

Recorded terminal passing observations: **9**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/ca711ee50161415eb53bbc3413abe79b.json)

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.898723 | >= 0.85 | PASS |
| unused_hold | 0.967419 | >= 0.85 | PASS |

Recorded terminal passing observations: **5**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/3334fda9a1f045598c18bfcac0084c33.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Runtime: **cuda**. Selected configuration: [forge-no-critic-penalty](../../../configs/forge/ideas/forge-no-critic-penalty.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `d276c5a7344fab6ec5de7b314d3982af5b0ef8027c8b89c01376ae366844a9cb`. Candidate revision: `c670878da61da2e52110dadeac5f9c8f7acd2b0e8819f9ad9e781cb531695e4b`. Runtime cohort: `ecd8071163b8f0d4b5e2e8a2d66ddcbb87b4bcd38ae0e33b62d2f4bc2ae41ae3`.

[Frozen numerical evidence](../technique-evidence/83f549889bf36b44bda98c8fe4c0f1600ef0cdd1d12c529f9e480ffcfdc93512.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Preserve the exact archived revision-7 measurement and its original policy qualification; no current-policy measurement or default-adoption credit.

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation) | [3/3](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) | [0(*)/1](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-3) | [3(*)/23](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation) |
| [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous) | [3/4](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) | [0(*)/7](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | [3(*)/30](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous) |
| [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability) | [3(*)/6](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | [0(*)/21](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | [0(*)/2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) | [3(*)/29](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability) |
| [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison) | [3/3](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) | [0(*)/2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) | [3(*)/24](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison) |
| [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer) | [3/3](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) | [0(*)/2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | [3(*)/24](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer) |
| [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage) | [3/3](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | [0/0](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-3) | [3(*)/22](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage) | [0(*)/7](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1) | [0/0](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-2) | [0/0](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-3) | [0(*)/7](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-c195899a64af-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) | FAIL | matches recorded run |
| [five_word_joint_smoke](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [ring16_acquisition](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [two_pole](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-two_pole) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [five_word_joint_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_bars4) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mode_hold) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-residual_student) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [trajectory](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-trajectory) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unipolar) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100_14k) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring_extension) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring_hold) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100_14k) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100_14k) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | [adaptation](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-3), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-c195899a64af-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | required | FAIL | matches recorded run |

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 8**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | required | FAIL | matches recorded run |
| [two_pole](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [ring16_acquisition](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | required | FAIL | matches recorded run |
| [five_word_joint_smoke](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | required | UNKNOWN | matches recorded run |
| [five_word_joint_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

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

Used by: [adaptation / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00691416 | <= 0.35 | PASS |
| recon_mse | 0.00685878 | <= 0.05 | PASS |

Recorded terminal passing observations: **21**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/b1b7e305c18e4d17803724f1fd86292b.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. step_label changed the update or common-prefix state

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/e2108d4981fe4339963259472f16d019.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2).

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

Used by: [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

**gaussian1d_smoke: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full pass with independent same-state confirmation

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.10327 |
| finite_fraction | 1 |
| mean | 2.07333 |
| mean_error_sigma | 0.146655 |
| sample_count | 4096 |
| std | 0.444676 |
| std_ratio | 0.889353 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/a15027675a7049669fe3af6ffcfa4b6e.json)

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

Used by: [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 77.6766 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 1.56968 | >= 0.15 | PASS |
| hq | 0.351562 | >= 0.85 | FAIL |
| mass_tv | 0.199707 | <= 0.15 | FAIL |
| modes | 8 | >= 16 | FAIL |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/a6816dfa5cad42db9f84d713d0a12e99.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-3) · [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Used by: [adaptation / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.942821 | <= 1 | PASS |
| mean_abs | 0.426749 | >= 0.3 | PASS |

Recorded terminal passing observations: **9**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/06614ea85f434847a59ff5976f983f2b.json)

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.898723 | >= 0.85 | PASS |
| unused_hold | 0.967419 | >= 0.85 | PASS |

Recorded terminal passing observations: **5**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/e1692a5048944855b4440f1bea9ae918.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Runtime: **cuda**. Selected configuration: [forge-no-critic-penalty](../../../configs/forge/ideas/forge-no-critic-penalty.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `21ec7e3f89402b5d0c77669d5fba0e36bcf4e690069cfdceab383520006c525f`. Candidate revision: `51e07b68506b8759bcc2369f1ca4d655d375119eda5a320eeae8eb858970d8d8`. Runtime cohort: `861df90816c5c76aec34000815d1d0c6cc59bd390d5a99cbf6693064c327e776`.

[Frozen numerical evidence](../technique-evidence/ddde64ee936114becac42863847a51e88ec0301bad9f7f69767d0a2fbc3f3d69.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Retain the exact complete measurement under its original joint-word evaluator source binding. The current v4 evaluator contract differs; this archived evidence grants no current-measurement or default-adoption claim.

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation) | [1/3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) | [0(*)/1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) | [1(*)/23](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation) |
| [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous) | [1/4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) | [0(*)/7](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | [1(*)/30](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous) |
| [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability) | [1(*)/6](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | [0(*)/21](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | [0(*)/2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) | [1(*)/29](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability) |
| [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison) | [1/3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) | [0(*)/2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) | [1(*)/24](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison) |
| [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer) | [1/3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) | [0(*)/2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | [1(*)/24](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer) |
| [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage) | [1/3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | [0(*)/19](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | [0/0](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-3) | [1(*)/22](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) | [0(*)/7](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1) | [0/0](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2) | [0/0](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3) | [0(*)/7](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-0d83d78027c5-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | changed since run |
| [clockfree_audit_measurement_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | FAIL | matches recorded run |
| [five_word_joint_smoke](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | FAIL | changed since run |
| [two_pole](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | FAIL | changed since run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | FAIL | changed since run |

<a name="cohort-cuda-0d83d78027c5-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [five_word_joint_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [grid100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [trajectory](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | [adaptation](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-3), [clockfree_continuous](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | FAIL | changed since run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | FAIL | changed since run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |
| [clockfree_audit_measurement_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | required | FAIL | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 8**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | required | UNKNOWN | recorded definition unavailable |
| [two_pole](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | FAIL | changed since run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |
| [ring16_acquisition](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | required | FAIL | changed since run |
| [five_word_joint_smoke](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | FAIL | changed since run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | FAIL | changed since run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | FAIL | changed since run |
| [unused_token_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | FAIL | changed since run |
| [ae_gan_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

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

Used by: [adaptation / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00270075 | <= 0.35 | PASS |
| recon_mse | 0.00403404 | <= 0.05 | PASS |

Recorded terminal passing observations: **16**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/01313af6d0b1452b9d9e60f95a57e1d7.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-penalty/01313af6d0b1452b9d9e60f95a57e1d7/ae_gan_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. step_label changed the update or common-prefix state

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/2c34e6b34c1844958037b5cc399dcfaa.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-penalty/2c34e6b34c1844958037b5cc399dcfaa/clockfree_audit_measurement_v1.gif); 4 saved observations; no new optimizer updates or sampling draws.

Recorded clock parity diagnostics:

| Condition | Exact state digest equality |
| --- | --- |
| evaluation_cadence | equal |
| horizon | different |
| restart | equal |
| step_label | different |

Recorded unexplained clock dependencies: **4**.

- learning-rate annealing depends on completed steps and horizon
- input-noise annealing depends on completed steps and horizon
- output-noise warmup depends on completed steps and horizon
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

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2).

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

Used by: [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

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

Used by: [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 145.578 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.356492 | >= 0.15 | PASS |
| hq | 0.320068 | >= 0.85 | FAIL |
| mass_tv | 0.229004 | <= 0.15 | FAIL |
| modes | 8 | >= 16 | FAIL |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/f4c4560d3c214491914c067adc0782fb.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-penalty/f4c4560d3c214491914c067adc0782fb/ring16_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) · [clockfree_continuous / Tier 3](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 1.2546 | <= 1 | FAIL |
| mean_abs | 0.447748 | >= 0.3 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d54d98951bf34e7990d47fced9234d37.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-penalty/d54d98951bf34e7990d47fced9234d37/two_pole.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

**unused_token_hold: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.88019 | >= 0.85 | PASS |
| unused_hold | 0.973274 | >= 0.85 | PASS |

Recorded terminal passing observations: **4**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/8ca22bcbde054570956f3e5c7e9d14a7.json)

[Actual-training GIF](../tier1-completion/media/k3p-no-penalty/8ca22bcbde054570956f3e5c7e9d14a7/unused_token_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p-no-penalty.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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
