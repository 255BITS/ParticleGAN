<!-- Generated Forge family report -->

# BCAP with K3P

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [capped-input-gradients](../technique-inventory.md#tag-capped-input-gradients) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions)

## Technique overview

BCAP with K3P replaces the K3P critic penalty with the fixed BCAP real/fake gradient cap, while retaining the K3P training machinery. It therefore combines a simple cap loss with scheduled learning rates, a critic gradient-spike guard, sparse latent-row damping and the declared training noise.

![Real and generated samples share a critic; a soft penalty discourages excessive input-gradient slopes and adds to the critic loss.](assets/bcap-explainer.png)

Conceptual illustration of the BCAP critic-loss mechanism, not measured training results. Generator/prior updates and K3P optimizer controls are described below.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior, $x$ a real batch, $z\sim P$ a latent batch and $y$ the generated batch, including any declared output noise. $C$ is the critic actually evaluated: $C=D$ without input noise; otherwise every forward evaluates $D(u+\epsilon_{\mathrm{in}})$ with a fresh draw. $\mathbb E$ means a sample mean, $\lVert\cdot\rVert_2$ is the Euclidean norm, $(a)_+=\max(a,0)$ and $\operatorname{softplus}(a)=\log(1+e^a)$. The scalar sketch detaches $y$ during the critic update and resamples differentiable $y$ after that update; joint and auxiliary hosts retain task-owned objectives. $g_r=\nabla_x C(x)$ and $g_f=\nabla_y C(y)$ are per-sample input gradients; $d$ counts critic-input coordinates. $\lambda$ is penalty strength and $\kappa$ the cap threshold. $h_i$ is latent row $i$'s current gradient, $h_i^{\mathrm{prev}}$ its last observed gradient, $\rho_i$ the A2 response multiplier and $\Delta z_i$ the corresponding row update. $\operatorname{cos}$ denotes the implementation's bounded cosine similarity, including its handling of zero-length history. For the separate spike guard, $g_p$ is a critic parameter gradient, $v_p$ its Adam second moment, $\tau_p$ its stored step count and $c_{\mathrm{guard}}$ its threshold.

**Default paired adversarial loss**

$$
\begin{aligned}\ell_D&=\mathbb E\!\left[\operatorname{softplus}(C(y)-C(x))\right],\\\ell_G&=\mathbb E\!\left[\operatorname{softplus}(C(x)-C(y))\right].\end{aligned}
$$

These scalar losses are minimized. The critic objective is $\ell_D+R$; the generator adds only its declared auxiliary objectives. Critic fakes are detached, and generator fakes and both scores are recomputed after the critic step. $C$ includes any declared fresh input noise.

**Fixed capped input-gradient penalty**

$$
R_{\mathrm{BCAP}}=\frac{\lambda}{2}\left\{\mathbb E\!\left[(\lVert g_r\rVert_2-\kappa)_+^2\right]+\mathbb E\!\left[(\lVert g_f\rVert_2-\kappa)_+^2\right]\right\}.
$$

The cap is a soft loss penalty on real and fake critic input-gradient norms, without division by input dimension. It does not directly clip critic parameter gradients or weights. The canonical BCAP cards use $\lambda=1$ and $\kappa=1$; source-bound selected recipes may differ.

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
For each training update:
  Apply the declared split cosine learning-rate schedules to network and prior parameter groups.
  Wrap D so each forward adds fresh scheduled critic-input noise.
  Sample real x and latent z; form fake y=G(z) with declared generated-output training noise.
  Critic objective = mean softplus(D(stop_gradient(y))-D(x)) + BCAP.
  Compute BCAP from the real and fake input-gradient norms using the coefficient and cap shown above.
  For penalty derivatives, detach samples from G and treat them as differentiable critic inputs; the penalty updates D only.
  Backpropagate; after the guard warmup, shrink each critic tensor’s gradient if its RMS exceeds the configured multiple of Adam’s historical RMS.
  Take the K3P critic Adam step. The BCAP penalty does not use K3P’s early/late penalty blend or EMA anchor.
  With D fixed, resample latent z and recompute differentiable fake y and both critic scores.
  Generator objective = mean softplus(D(x)-D(y)) + task-owned auxiliary losses; backpropagate.
  A2 is active only for an eligible latent table when some rows have zero gradient and the cumulative observed-row fraction is below its configured threshold.
  With Adam history, scale each active row response using A2 gradient agreement; rows without history stay unchanged.
  Take the K3P generator/prior Adam step; direct sample-particle hosts may use their declared coherent-gradient response gain.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Canonical paired relativistic logistic on raw critic scores: $D$ minimizes $\mathbb E[\operatorname{softplus}(C(y)-C(x))]$; $G$ minimizes the reverse. Host-owned auxiliary losses remain explicit. |
| Optimizer | K3PCriticAdam and K3PGeneratorAdam, built through public recipe factories. Base Adam moments default to $(\beta_1,\beta_2)=(0,0.999)$; the wrappers apply critic guards and eligible latent/direct-particle interventions. |
| Learning rates and annealing | Annealing is retained. Canonical network LR holds for $60\%$ of min(task budget, $1600$ steps), then cosine-decays to $1\%$ and holds. Prior LR holds for $60\%$ of the full task budget, then cosine-decays to $5\%$. Canonical global LR $0.00425$, D multiplier $1$, prior multiplier $2$; recorded searches change role multipliers. Resolve the selected recipe for exact horizons/floors. |
| Parameter-gradient clipping | Adaptive tensorwise critic gradient clipping via the spike guard: after at least $200$ prior Adam steps per tensor, gradient RMS is limited to $5$ times the bias-corrected second-moment RMS. This is separate from BCAP and is not a single global norm clip. |
| Critic penalties and anchors | Fixed BCAP real/fake squared one-sided L2 cap; canonical $\lambda=1$ and $\kappa=1$, every update. The historical search includes $\lambda=0.5$ alternatives. K3P’s LR-dependent penalty handover and EMA-anchor penalty are bypassed by reg_arm=b_cap. |
| Damping and update guards | A2 requires an eligible sparse learnable latent table, some zero-gradient rows on the current update, and cumulative observed-row fraction below latent_damping_max_rate (canonical $0.5$). When Adam state and row history exist, it scales the row response by $0.75+0.25\operatorname{cos}(h_i,h_i^{\mathrm{prev}})$, within $[0.5,1]$; rows without history use $1$. Adam’s second moment still tracks the raw gradient. Direct sample-particle hosts may additionally use coherent-gradient gain and dedicated moments; ordinary learned-prior networks do not automatically receive that gain. |
| Training and sampling noise | Canonical recipe retains critic input noise, starting at std $0.5$ and linearly decaying to zero over the first $10\%$ of training, and generated-output noise warming toward std $0.029$ over the first $20\%$. Task/source-specific adaptations determine actual served and training noise; clean and noisy cohorts remain separate. Critic-input noise is sampled afresh by the noise-wrapped critic on each forward, including penalty evaluation; generated-output noise is separately added to fake samples. |
| Parameter averaging and serving | Canonical generator EMA decay=$0.995$ may be maintained by the host, but the card declares live-weight scoring. Fixed BCAP bypasses critic-anchor evaluation, even if an EMA critic copy exists in optimizer state. |

## Configuration differences

- The pseudocode describes the family mechanism; the selected configuration and each task determine architectures, initialization, prior, sampling, update count and task-owned auxiliary losses.
- The cap is a soft penalty on gradients with respect to critic inputs, not a hard bound on model-parameter gradients. Its L2 norm is not divided by input dimension.
- Configuration alternatives are complete recipes; passing cells from different recipes or sources are not combined.
- BCAP with K3P is an editorial rename of family ID bcap; the candidate remains k3p-bcap-matched-v1. It is distinct from the native-Adam BCAP family (ID bcap-pure).
- Canonical values describe the reusable baseline. Selected historical configurations can change penalty coefficient and role rates; source-bound measurement receipts remain authoritative.
- The declared A2 capability may be blocked on an incompatible host; do not infer that damping ran merely from the family name.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [configs/forge/ideas/k3p-bcap-matched-v1.json](../../../configs/forge/ideas/k3p-bcap-matched-v1.json)
- [configs/forge/searches/bcap-tier1-refresh-v1.json](../../../configs/forge/searches/bcap-tier1-refresh-v1.json)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/k3p.py](../../../particlegan/k3p.py)
- [particlegan/grad_regularizers.py](../../../particlegan/grad_regularizers.py)
- [particlegan/training.py](../../../particlegan/training.py)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

<a name="cohort-cuda-0d83d78027c5"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [bcap · 08689a73c551](../../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json).

Recorded qualification: **tier 0**, discriminator_stability revision 5. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `21ec7e3f89402b5d0c77669d5fba0e36bcf4e690069cfdceab383520006c525f`. Candidate revision: `1f1d81a0aa3f60e2a3f3187dacfa29e7688bb6509b16daf439036ed5f9951c4e`. Runtime cohort: `de1faf2ca5cd1417f6fec32350070a74ded53fef94424a96667e51be3886d769`.

[Frozen numerical evidence](../technique-evidence/ddde64ee936114becac42863847a51e88ec0301bad9f7f69767d0a2fbc3f3d69.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: current_measurement. Complete current Tier 1 measurement at one frozen recipe and executed source. FAIL completes a measurement; calibration and confirmation remain separate.

</details>

Complete current Tier 1 measurement in: discriminator_stability; additional scoped probes: clockfree_audit_measurement_v1. PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation) | [3/3](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) | [0(*)/19](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) | [0(*)/1](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) | [3(*)/23](bcap.md#cohort-cuda-0d83d78027c5-adaptation) |
| [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous) | [3/4](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | [0(*)/19](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) | [0(*)/7](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | [3(*)/30](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous) |
| [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability) | [4(*)/6](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | [0(*)/19](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | [0(*)/2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) | [4(*)/27](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability) |
| [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison) | [3/3](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) | [0(*)/19](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) | [0(*)/2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) | [3(*)/24](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison) |
| [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer) | [3/3](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) | [0(*)/19](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) | [0(*)/2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | [3(*)/24](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer) |
| [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage) | [3/3](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | [0(*)/19](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | [0/0](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-3) | [3(*)/22](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) | [0(*)/7](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1) | [0/0](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2) | [0/0](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3) | [0(*)/7](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) |

\* indicates incomplete results, including changed or unbound current contracts.

<a name="cohort-cuda-0d83d78027c5-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ae_gan_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | matches |
| [clockfree_audit_measurement_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | FAIL | matches |
| [five_word_joint_acquisition](bcap.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition) | [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | PASS | CHANGED |
| [gaussian1d_acquisition](bcap.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition) | [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | FAIL | matches |
| [ring16_acquisition](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | FAIL | matches |
| [two_pole](bcap.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | matches |
| [unused_token_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [cover_leftover](bcap.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [grid100](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_bars4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_blobs4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_intensity2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_stripes2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [mid_scale_identity](bcap.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [mode_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [residual_student](bcap.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [rotated100](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [staggered100](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [trajectory](bcap.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [unipolar](bcap.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_anisotropic](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_overlap](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_spiral](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_two_broad](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_unequal_mass](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_unequal_width](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [clockfree_audit](bcap.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [grid100_14k](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [ring_extension](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches |
| [ring_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches |
| [rotated100_14k](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [staggered100_14k](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [target_shift_recovery](bcap.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | [adaptation](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-3), [clockfree_continuous](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](bcap.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](bcap.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](bcap.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](bcap.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](bcap.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](bcap.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [target_shift_recovery](bcap.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | unbound |

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
| [two_pole](bcap.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |
| [clockfree_audit_measurement_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | required | FAIL | matches |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](bcap.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](bcap.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](bcap.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](bcap.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](bcap.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [clockfree_audit](bcap.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | required | UNKNOWN | unbound |
| [ring_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |
| [grid100_14k](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | required | UNKNOWN | unbound |
| [rotated100_14k](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | required | UNKNOWN | unbound |
| [staggered100_14k](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | required | UNKNOWN | unbound |
| [target_shift_recovery](bcap.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 5**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 19 / 2**.

Calibration: **provisional**. Expanded six-task Tier 1 placement is provisional and requires bounded calibration. Revision 3 and prior profiles retain their original tasks and evidence; a standalone scalar pass gives no whole-view/default credit.

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [gaussian1d_acquisition](bcap.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition) | required | FAIL | matches |
| [two_pole](bcap.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |
| [ring16_acquisition](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | required | FAIL | matches |
| [five_word_joint_acquisition](bcap.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition) | required | PASS | CHANGED |
| [clockfree_audit_measurement_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | diagnostic | FAIL | matches |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](bcap.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](bcap.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](bcap.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](bcap.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](bcap.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](bcap.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](bcap.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](bcap.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](bcap.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](bcap.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](bcap.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |
| [img_intensity2_residual16](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | unbound |
| [vector_two_broad_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_mass_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_width_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | unbound |
| [vector_anisotropic_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | unbound |
| [vector_overlap_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | unbound |
| [vector_spiral_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | unbound |
| [img_stripes2_residual16](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | unbound |
| [img_bars4_residual16](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | unbound |
| [img_blobs4_residual16](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_square_named_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [rotated100_affine_square_named_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [staggered100_affine_square_named_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_paired_laws_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | unbound |
| [grid100_release07_cloud_named_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](bcap.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](bcap.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](bcap.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](bcap.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](bcap.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](bcap.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |
| [img_intensity2_residual16](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | unbound |
| [vector_two_broad_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_mass_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_width_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | unbound |
| [vector_anisotropic_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | unbound |
| [vector_overlap_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | unbound |
| [vector_spiral_published](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | unbound |
| [img_stripes2_residual16](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | unbound |
| [img_bars4_residual16](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | unbound |
| [img_blobs4_residual16](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_square_named_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [rotated100_affine_square_named_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [staggered100_affine_square_named_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | unbound |
| [two_pole_800_schedule800_diagnostic_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | unbound |

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
| [two_pole](bcap.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](bcap.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](bcap.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](bcap.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](bcap.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](bcap.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](bcap.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](bcap.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](bcap.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](bcap.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](bcap.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](bcap.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [two_pole_tier1_policy_selected_cloud_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [unused_token_hold_tier1_policy_selected_cloud_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [clockfree_audit_tier1_policy_selected_cloud_v1](bcap.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |

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

Used by: [adaptation / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00857694 | <= 0.35 | PASS |
| recon_mse | 0.00370552 | <= 0.05 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/99b2e68810ba4379a860e467e8a818ca.json)

[Actual-training GIF](../tier1-completion/media/bcap/99b2e68810ba4379a860e467e8a818ca/ae_gan_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Current contract: **matches**. Current task contract matches the recorded conditions. step_label changed the update or common-prefix state

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/5732e526ff764977af157e194f8a5749.json)

[Actual-training GIF](../tier1-completion/media/bcap/5732e526ff764977af157e194f8a5749/clockfree_audit_measurement_v1.gif); 4 saved observations; no new optimizer updates or sampling draws.

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

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| minimum_reconstruction_token_probability | 0.996322 | >= 0.9 | PASS |
| modes | 5 | == 5 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| sample_count | 1024 | >= 1024 | PASS |

Recorded terminal passing observations: **18**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/8b88bf0491144e35b637e68eda852310.json)

[Actual-training GIF](../tier1-completion/media/bcap/8b88bf0491144e35b637e68eda852310/five_word_joint_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.174963 | <= 0.05 | FAIL |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.417016 | <= 0.2 | FAIL |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 1.08263 | <= 1.2 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d0638ad5ce5a47e5b2fcb00b369768b2.json)

[Actual-training GIF](../tier1-completion/media/bcap/d0638ad5ce5a47e5b2fcb00b369768b2/gaussian1d_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [discriminator_stability / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 4.457 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.134501 | >= 0.15 | FAIL |
| hq | 0.938477 | >= 0.85 | PASS |
| mass_tv | 0.12207 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d8185ca486b54af79ef422395eac8065.json)

[Actual-training GIF](../tier1-completion/media/bcap/d8185ca486b54af79ef422395eac8065/ring16_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

Used by: [clockfree_continuous / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) · [clockfree_continuous / Tier 3](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Current contract: **matches**. Current task contract matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.753464 | <= 1 | PASS |
| mean_abs | 0.434628 | >= 0.3 | PASS |

Recorded terminal passing observations: **9**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/bde4745f31864cc08c42880d0fa36272.json)

[Actual-training GIF](../tier1-completion/media/bcap/bde4745f31864cc08c42880d0fa36272/two_pole.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [k3p_two_pole_horizon / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.947523 | >= 0.85 | PASS |
| unused_hold | 0.99325 | >= 0.85 | PASS |

Recorded terminal passing observations: **13**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/329006823c804cdfbd01ab414d10d1cd.json)

[Actual-training GIF](../tier1-completion/media/bcap/329006823c804cdfbd01ab414d10d1cd/unused_token_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](bcap.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](bcap.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base optimizer. Repository-specific guards and damping are separate mechanisms.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Paired relativistic logistic objective; this implementation pairs scores, rather than subtracting batch-average scores.
- [On the regularization of Wasserstein GANs](https://arxiv.org/abs/1709.08894). Related work on one-sided gradient penalties. The repository BCAP kernel uses real/fake inputs directly and is not a reproduction of this paper or its sampling scheme.
