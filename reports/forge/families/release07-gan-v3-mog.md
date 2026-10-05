<!-- Generated Forge family report -->

# GAN v3 release 0.7 (MoG)

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [capped-input-gradients](../technique-inventory.md#tag-capped-input-gradients) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [historical-cohort](../technique-inventory.md#tag-historical-cohort) · [learning-rate-annealing](../technique-inventory.md#tag-learning-rate-annealing)

## Technique overview

This historical entry retains the released v0.7 GAN v3 recipe adapted to the matched learned-MoG native host: latent dimension 2, batch 2048 and task-bound MoG sampling. It uses the same fixed critic-gradient cap, latent spread objective and full-horizon cosine as the release family. Its exact source/host results remain historical; the reusable task-adapted card is the current family declaration.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior, $x$ a real batch, $z\sim P$ a latent batch and $y$ the generated batch, including any declared output noise. $C$ is the critic actually evaluated: $C=D$ without input noise; otherwise every forward evaluates $D(u+\epsilon_{\mathrm{in}})$ with a fresh draw. $\mathbb E$ means a sample mean, $\lVert\cdot\rVert_2$ is the Euclidean norm, $(a)_+=\max(a,0)$ and $\operatorname{softplus}(a)=\log(1+e^a)$. The scalar sketch detaches $y$ during the critic update and resamples differentiable $y$ after that update; joint and auxiliary hosts retain task-owned objectives. $g_r=\nabla_x C(x)$ and $g_f=\nabla_y C(y)$ are per-sample input gradients; $d$ counts critic-input coordinates. $\lambda$ is penalty strength and $\kappa$ the cap threshold. $R_{\mathrm{prior}}$ is the VICReg-inspired latent spread term, $d_z$ the latent dimension, $\Sigma_z$ the unbiased sample covariance and $\epsilon=10^{-4}$ the variance stabilizer. $t$ counts completed updates, $T$ is the declared recipe horizon, $q=t/T$, $\eta^{(0)}$ a base LR and $a(q)$ its multiplier. EMA denotes a parameter exponential moving average.

**Default paired adversarial loss**

$$
\begin{aligned}\ell_D&=\mathbb E\!\left[\operatorname{softplus}(C(y)-C(x))\right],\\\ell_G&=\mathbb E\!\left[\operatorname{softplus}(C(x)-C(y))\right].\end{aligned}
$$

These scalar losses are minimized. The critic objective is $\ell_D+R$; the generator adds only its declared auxiliary objectives. Critic fakes are detached, and generator fakes and both scores are recomputed after the critic step. $C$ includes any declared fresh input noise.

**Fixed capped input-gradient penalty**

$$
R_{\mathrm{BCAP}}=\frac{\lambda}{2}\left\{\mathbb E\!\left[(\lVert g_r\rVert_2-\kappa)_+^2\right]+\mathbb E\!\left[(\lVert g_f\rVert_2-\kappa)_+^2\right]\right\}.
$$

The full release recipe fixes $\lambda=6$ and $\kappa=1.25$, applied every critic update. This is the same soft L2 cap mechanism with stronger and wider settings; K3P handover, critic proximity and optimizer interventions are disabled.

**Latent spread auxiliary objective**

$$
\begin{aligned}R_{\mathrm{prior}}&=\frac{1}{d_z}\sum_{a=1}^{d_z}\left(1-\sqrt{(\Sigma_z)_{aa}+10^{-4}}\right)_+\\&\quad+\frac{1}{d_z}\sum_{a\ne b}(\Sigma_z)_{ab}^2,\\L_G&=\ell_G+0.05R_{\mathrm{prior}}.\end{aligned}
$$

For scalar hosts that expose this auxiliary, use the unbiased sample covariance of the supplied latent batch. It discourages standard deviations below $1$ and off-diagonal covariance, without requiring a Gaussian topology. Batches with fewer than two rows return zero. Behavioral hosts retain their explicitly bound auxiliary objectives.

**Full-horizon learning-rate cosine**

$$
a(q)=\begin{cases}1,&q\le0.6,\\0.05+\dfrac{0.95}{2}\left[1+\cos\!\left(\pi\min\!\left(1,\dfrac{q-0.6}{0.4}\right)\right)\right],&q>0.6,\end{cases}\qquad \eta(t)=\eta^{(0)}a(t/T).
$$

Hold each role at its base rate for $60\%$ of the declared recipe horizon, then decay to a $5\%$ floor and hold. The reusable task-adapted card resolves $T$ explicitly; the historical native MoG/cloud cards retain $T=7000$. An external execution cap does not redefine the schedule.

## Simplified pseudocode

```text
For each training iteration:
  Use the historical 7000-step recipe horizon: hold LR through 60%, then cosine-decay toward 5% of its base value.
  Draw real x and task-prior z; fake = detach(G(z)).
  Compute the paired critic adversarial loss shown above.
  g_r = gradient(D(x),x); g_f = gradient(D(fake),fake), using detached input copies.
  Add the fixed real/fake input-gradient penalty shown above.
  Backpropagate; take the critic Adam step with guard and gradient anchor disabled.
  Freeze D parameters; draw fresh z, regenerate fake and recompute both scores.
  Compute the paired generator adversarial loss from the updated critic.
  On scalar latent-table hosts, add .05 * R_prior to L_G; behavioral hosts retain their original objectives.
  Backpropagate; update G/prior with A2 and direct response gain disabled.
  Maintain the configured .995 generator/prior EMA where the host implements it.
  Score with the frozen task's declared weights, prior and noise law; ordinary gates use clean live outputs.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Paired relativistic logistic, with the fixed real/fake cap regularizer below. Scalar hosts add the declared $0.05$ latent variance/covariance term; behavioral hosts keep their task-owned reconstruction, conditional or direct-particle objectives. |
| Optimizer | K3P-derived Adam wrappers selected by the explicit fixed b_cap arm. $\beta_1=0$, $\beta_2=0.99$, AMSGrad off; guard, anchor, A2 and direct gain are disabled. Observer bookkeeping remains. This recipe is not selected through optimizer_family='adam', despite having its intervention switches off. Direct sample-particle hosts can retain separately declared response moments even when response gain is disabled; inspect the host binding. |
| Learning rates and annealing | Original reference base LR $0.00425$, critic multiplier $1$, latent-prior multiplier $2$. The historical selected whole variant uses $0.006375$ and prior multiplier $1$. Both retain the $7000$-step recipe horizon, a $60\%$ hold followed by cosine decay to $5\%$, with no separate short network horizon. This native card fixes its resource fields; the later task-adapted successor delegates them explicitly without rewriting the historical recipe. |
| Parameter-gradient clipping | No critic spike guard or global gradient-norm clipping. The cap penalizes excess critic input-gradient magnitude in the loss; it does not clamp critic parameter gradients or weights. |
| Critic penalties and anchors | Fixed $\lambda=6$ and L2 cap $\kappa=1.25$ on both real and fake samples, applied every step. No RMS normalization, K3P handover or critic-gradient proximity term. |
| Damping and update guards | A2, critic spike guard, critic anchor and direct sample-particle gain are all disabled. This distinguishes the full release recipe from a fixed BCap penalty swapped into an otherwise intact K3P recipe. |
| Training and sampling noise | No additive critic-input or generated-output training noise. The task's prior still owns its latent sampling noise: learned-MoG and particle-cloud hosts remain distinct. Paired noisy/EMA diagnostics do not replace the required clean/live result. |
| Parameter averaging and serving | Generator/prior EMA decay $0.995$ is configured where the host implements it; selected ordinary gates use live weights. Behavioral hosts may own no scored EMA or latent table. No critic-gradient anchor average is active. |

## Configuration differences

- Each result retains its executed source, recipe, task prior, initialization, budget and sampling law. These descriptions do not change or requalify recorded measurements.
- Task-owned objectives and active components matter: direct sample particles, learned latent rows and a generator network are different parameter roles. A declared recipe switch does not imply that every host can apply it.
- The retained selected variant is release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c at historical source 2899099048c0a9987eb8720214abfce56d86d92a. It uses LR .006375 and prior multiplier 1; the original native card used .00425 and multiplier 2.
- The historical reference fixes z_dim=2, batch_size=2048 and 7000-step recipe horizon. Its learned-MoG binding records sigma .025 and standardize=false; task initialization and affine native host remain part of the experiment.
- The original native card's fixed resource fields were incompatible with behavioral toy hosts. A later explicit task-adapted successor delegates those fields; that successor does not rewrite this historical evidence.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [configs/forge/ideas/release07-gan-v3-mog-v1.json](../../../configs/forge/ideas/release07-gan-v3-mog-v1.json)
- [configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json](../../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json)
- [reports/forge/studies/FORMULATION_COMPARISON_CONTINUATION.md](../studies/FORMULATION_COMPARISON_CONTINUATION.md)
- [reports/forge/RELEASE07_TASK_ADAPTATION_READOUT.md](../RELEASE07_TASK_ADAPTATION_READOUT.md)
- [particlegan/grad_regularizers.py](../../../particlegan/grad_regularizers.py)
- [particlegan/vicreg_loss.py](../../../particlegan/vicreg_loss.py)
- [https://github.com/255BITS/ParticleGAN/blob/2899099048c0a9987eb8720214abfce56d86d92a/particlegan/recipes.py](https://github.com/255BITS/ParticleGAN/blob/2899099048c0a9987eb8720214abfce56d86d92a/particlegan/recipes.py)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

**Historical cohort navigation.** This original recipe/prior row is retained for existing links. [Current GAN v3 solution family](release07-gan-v3.md) uses one whole selected configuration and task-declared priors; these historical cells are not pooled into it.

<a name="cohort-cuda-7f9c23eb0e27"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [release07-gan-v3-mog · 1e266b5a2986](../../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json).

Recorded qualification: **tier 0**, discriminator_stability revision 5. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `7306340bac0a4ea67ea7b080513116457b7d72a8cb0d0c1f0727eaae6db38185`. Candidate revision: `47630167ae31039bb209a2245097e4b9d5b2379ae0c526d4d3aa57e6204b0f33`. Runtime cohort: `f99998f9b0fa4054fac701e7ea0b5aaa8d5986b332d48ee8447f5878cd92cf3e`.

[Frozen numerical evidence](../technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Retain the exact recorded incumbent; its outcomes are historical best observed evidence where a registered search exists. Alternatives from another source or runtime are unranked.

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation) | [3(*)/3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) | [0(*)/19](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) | [0(*)/1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-3) | [3(*)/23](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation) |
| [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous) | [3(*)/4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) | [0(*)/19](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) | [0(*)/7](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | [3(*)/30](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous) |
| [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability) | [3(*)/6](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | [0(*)/19](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) | [0(*)/2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) | [3(*)/27](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability) |
| [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison) | [3(*)/3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) | [0(*)/19](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) | [0(*)/2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3) | [3(*)/24](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison) |
| [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer) | [3(*)/3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) | [0(*)/19](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) | [0(*)/2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) | [3(*)/24](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer) |
| [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage) | [3(*)/3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | [0(*)/19](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | [0/0](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-3) | [3(*)/22](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage) | [0(*)/7](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1) | [0/0](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-2) | [0/0](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-3) | [0(*)/7](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage) |

\* indicates incomplete results, including changed or unbound current contracts.

<a name="cohort-cuda-7f9c23eb0e27-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ae_gan_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | PASS | CHANGED |
| [clockfree_audit_measurement_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) | UNKNOWN | unbound |
| [five_word_joint_acquisition](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_acquisition) | [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | CHANGED |
| [gaussian1d_acquisition](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_acquisition) | [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | UNKNOWN | unbound |
| [ring16_acquisition](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition) | [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) | FAIL | CHANGED |
| [two_pole](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | PASS | CHANGED |
| [unused_token_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1) | PASS | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [cover_leftover](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [grid100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_bars4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [img_blobs4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [img_intensity2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [img_stripes2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [mid_scale_identity](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [mode_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [residual_student](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [rotated100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | matches |
| [staggered100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | matches |
| [trajectory](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [unipolar](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [vector_anisotropic](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [vector_overlap](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [vector_spiral](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [vector_two_broad](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [vector_unequal_mass](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |
| [vector_unequal_width](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2), [quality_coverage](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2) | UNKNOWN | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [clockfree_audit](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit) | [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [grid100_14k](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_14k) | [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [ring_extension](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) | UNKNOWN | matches |
| [ring_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3), [discriminator_stability](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3), [formulation_comparison](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3), [host_profile_transfer](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3) | UNKNOWN | matches |
| [rotated100_14k](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_14k) | [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [staggered100_14k](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_14k) | [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [target_shift_recovery](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery) | [adaptation](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-3), [clockfree_continuous](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-7f9c23eb0e27-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | PASS | CHANGED |
| [unused_token_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | PASS | CHANGED |
| [ae_gan_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | PASS | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | UNKNOWN | CHANGED |
| [residual_student](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | UNKNOWN | CHANGED |
| [unipolar](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | UNKNOWN | CHANGED |
| [cover_leftover](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | UNKNOWN | CHANGED |
| [mid_scale_identity](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | UNKNOWN | CHANGED |
| [mode_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | UNKNOWN | CHANGED |
| [vector_two_broad](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | UNKNOWN | CHANGED |
| [vector_unequal_mass](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | UNKNOWN | CHANGED |
| [vector_unequal_width](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | UNKNOWN | CHANGED |
| [vector_anisotropic](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | UNKNOWN | CHANGED |
| [vector_overlap](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | UNKNOWN | CHANGED |
| [vector_spiral](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | UNKNOWN | CHANGED |
| [img_stripes2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | UNKNOWN | CHANGED |
| [img_bars4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | UNKNOWN | CHANGED |
| [img_blobs4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | UNKNOWN | CHANGED |
| [img_intensity2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | UNKNOWN | CHANGED |
| [grid100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-7f9c23eb0e27-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [target_shift_recovery](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery) | required | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous"></a>

## clockfree_continuous

**clockfree_continuous — revision 3**. [View declaration](../../../configs/forge/views/clockfree_continuous.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **4 / 19 / 7**.

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
| [two_pole](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | PASS | CHANGED |
| [unused_token_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | PASS | CHANGED |
| [ae_gan_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | PASS | CHANGED |
| [clockfree_audit_measurement_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_measurement_v1) | required | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | UNKNOWN | CHANGED |
| [residual_student](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | UNKNOWN | CHANGED |
| [unipolar](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | UNKNOWN | CHANGED |
| [cover_leftover](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | UNKNOWN | CHANGED |
| [mid_scale_identity](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | UNKNOWN | CHANGED |
| [mode_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | UNKNOWN | CHANGED |
| [vector_two_broad](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | UNKNOWN | CHANGED |
| [vector_unequal_mass](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | UNKNOWN | CHANGED |
| [vector_unequal_width](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | UNKNOWN | CHANGED |
| [vector_anisotropic](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | UNKNOWN | CHANGED |
| [vector_overlap](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | UNKNOWN | CHANGED |
| [vector_spiral](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | UNKNOWN | CHANGED |
| [img_stripes2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | UNKNOWN | CHANGED |
| [img_bars4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | UNKNOWN | CHANGED |
| [img_blobs4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | UNKNOWN | CHANGED |
| [img_intensity2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | UNKNOWN | CHANGED |
| [grid100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [clockfree_audit](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit) | required | UNKNOWN | unbound |
| [ring_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | UNKNOWN | matches |
| [grid100_14k](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_14k) | required | UNKNOWN | unbound |
| [rotated100_14k](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_14k) | required | UNKNOWN | unbound |
| [staggered100_14k](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_14k) | required | UNKNOWN | unbound |
| [target_shift_recovery](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-target_shift_recovery) | required | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 5**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 19 / 2**.

Calibration: **provisional**. Expanded six-task Tier 1 placement is provisional and requires bounded calibration. Revision 3 and prior profiles retain their original tasks and evidence; a standalone scalar pass gives no whole-view/default credit.

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [gaussian1d_acquisition](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_acquisition) | required | UNKNOWN | unbound |
| [two_pole](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | PASS | CHANGED |
| [unused_token_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | PASS | CHANGED |
| [ae_gan_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | PASS | CHANGED |
| [ring16_acquisition](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition) | required | FAIL | CHANGED |
| [five_word_joint_acquisition](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_acquisition) | required | UNKNOWN | CHANGED |
| [clockfree_audit_measurement_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_measurement_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | UNKNOWN | CHANGED |
| [residual_student](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | UNKNOWN | CHANGED |
| [unipolar](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | UNKNOWN | CHANGED |
| [cover_leftover](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | UNKNOWN | CHANGED |
| [mid_scale_identity](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | UNKNOWN | CHANGED |
| [mode_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | UNKNOWN | CHANGED |
| [vector_two_broad](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | UNKNOWN | CHANGED |
| [vector_unequal_mass](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | UNKNOWN | CHANGED |
| [vector_unequal_width](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | UNKNOWN | CHANGED |
| [vector_anisotropic](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | UNKNOWN | CHANGED |
| [vector_overlap](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | UNKNOWN | CHANGED |
| [vector_spiral](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | UNKNOWN | CHANGED |
| [img_stripes2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | UNKNOWN | CHANGED |
| [img_bars4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | UNKNOWN | CHANGED |
| [img_blobs4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | UNKNOWN | CHANGED |
| [img_intensity2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | UNKNOWN | CHANGED |
| [grid100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | PASS | CHANGED |
| [unused_token_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | PASS | CHANGED |
| [ae_gan_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | PASS | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | UNKNOWN | CHANGED |
| [residual_student](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | UNKNOWN | CHANGED |
| [unipolar](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | UNKNOWN | CHANGED |
| [cover_leftover](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | UNKNOWN | CHANGED |
| [mid_scale_identity](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | UNKNOWN | CHANGED |
| [mode_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | UNKNOWN | CHANGED |
| [vector_two_broad](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | UNKNOWN | CHANGED |
| [vector_unequal_mass](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | UNKNOWN | CHANGED |
| [vector_unequal_width](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | UNKNOWN | CHANGED |
| [vector_anisotropic](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | UNKNOWN | CHANGED |
| [vector_overlap](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | UNKNOWN | CHANGED |
| [vector_spiral](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | UNKNOWN | CHANGED |
| [img_stripes2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | UNKNOWN | CHANGED |
| [img_bars4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | UNKNOWN | CHANGED |
| [img_blobs4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | UNKNOWN | CHANGED |
| [img_intensity2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | UNKNOWN | CHANGED |
| [grid100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | UNKNOWN | matches |
| [img_intensity2_residual16](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | unbound |
| [vector_two_broad_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_mass_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_width_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | unbound |
| [vector_anisotropic_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | unbound |
| [vector_overlap_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap_published) | diagnostic | UNKNOWN | unbound |
| [vector_spiral_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral_published) | diagnostic | UNKNOWN | unbound |
| [img_stripes2_residual16](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | unbound |
| [img_bars4_residual16](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | unbound |
| [img_blobs4_residual16](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_square_named_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [rotated100_affine_square_named_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [staggered100_affine_square_named_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_paired_laws_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | unbound |
| [grid100_release07_cloud_named_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | PASS | CHANGED |
| [unused_token_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | PASS | CHANGED |
| [ae_gan_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | PASS | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | UNKNOWN | CHANGED |
| [residual_student](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | UNKNOWN | CHANGED |
| [unipolar](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | UNKNOWN | CHANGED |
| [cover_leftover](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | UNKNOWN | CHANGED |
| [mid_scale_identity](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | UNKNOWN | CHANGED |
| [mode_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | UNKNOWN | CHANGED |
| [vector_two_broad](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | UNKNOWN | CHANGED |
| [vector_unequal_mass](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | UNKNOWN | CHANGED |
| [vector_unequal_width](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | UNKNOWN | CHANGED |
| [vector_anisotropic](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | UNKNOWN | CHANGED |
| [vector_overlap](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | UNKNOWN | CHANGED |
| [vector_spiral](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | UNKNOWN | CHANGED |
| [img_stripes2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | UNKNOWN | CHANGED |
| [img_bars4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | UNKNOWN | CHANGED |
| [img_blobs4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | UNKNOWN | CHANGED |
| [img_intensity2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | UNKNOWN | CHANGED |
| [grid100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | UNKNOWN | matches |
| [img_intensity2_residual16](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | unbound |
| [vector_two_broad_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_mass_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_width_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | unbound |
| [vector_anisotropic_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | unbound |
| [vector_overlap_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap_published) | diagnostic | UNKNOWN | unbound |
| [vector_spiral_published](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral_published) | diagnostic | UNKNOWN | unbound |
| [img_stripes2_residual16](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | unbound |
| [img_bars4_residual16](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | unbound |
| [img_blobs4_residual16](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_square_named_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [rotated100_affine_square_named_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [staggered100_affine_square_named_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | unbound |
| [two_pole_800_schedule800_diagnostic_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | unbound |

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
| [two_pole](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole) | required | PASS | CHANGED |
| [unused_token_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold) | required | PASS | CHANGED |
| [ae_gan_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold) | required | PASS | CHANGED |

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-trajectory) | required | UNKNOWN | CHANGED |
| [residual_student](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-residual_student) | required | UNKNOWN | CHANGED |
| [unipolar](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unipolar) | required | UNKNOWN | CHANGED |
| [cover_leftover](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-cover_leftover) | required | UNKNOWN | CHANGED |
| [mid_scale_identity](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mid_scale_identity) | required | UNKNOWN | CHANGED |
| [mode_hold](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-mode_hold) | required | UNKNOWN | CHANGED |
| [vector_two_broad](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_two_broad) | required | UNKNOWN | CHANGED |
| [vector_unequal_mass](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_mass) | required | UNKNOWN | CHANGED |
| [vector_unequal_width](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_unequal_width) | required | UNKNOWN | CHANGED |
| [vector_anisotropic](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic) | required | UNKNOWN | CHANGED |
| [vector_overlap](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_overlap) | required | UNKNOWN | CHANGED |
| [vector_spiral](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-vector_spiral) | required | UNKNOWN | CHANGED |
| [img_stripes2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_stripes2) | required | UNKNOWN | CHANGED |
| [img_bars4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_bars4) | required | UNKNOWN | CHANGED |
| [img_blobs4](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_blobs4) | required | UNKNOWN | CHANGED |
| [img_intensity2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-img_intensity2) | required | UNKNOWN | CHANGED |
| [grid100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-7f9c23eb0e27-quality_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-tier1_policy_coverage"></a>

## tier1_policy_coverage

**tier1_policy_coverage — revision 1**. [View declaration](../../../configs/forge/views/tier1_policy_coverage.json).

Separately scoped cohort. This ordinary lane retains its own required gates and execution policy; its measurements are excluded from family totals and give no parent-cohort credit.

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **7 / 0 / 0**.

Calibration: **undeclared**. Calibration and robustness are separate from recorded task passes.

<a name="cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [two_pole_tier1_policy_selected_cloud_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [unused_token_hold_tier1_policy_selected_cloud_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [clockfree_audit_tier1_policy_selected_cloud_v1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |

<a name="cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-2"></a>

### Tier 2

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-3"></a>

### Tier 3

No experiments assigned.

<a name="cohort-cuda-7f9c23eb0e27-experiments"></a>

## Experiment metrics and pass criteria

One evidence entry per experiment is shared by its view rows. CHANGED means the declared execution or evaluator differs from the recorded task; its earlier verdict is preserved.

<a name="cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold"></a>

### ae_gan_hold

**ae_gan_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) · [clockfree_continuous / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00199085 | <= 0.35 | PASS |
| recon_mse | 0.0131624 | <= 0.05 | PASS |

Recorded terminal passing observations: **23**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/7cb6a3d82e7d4bf2b44253f76e6f3f39.json)

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1"></a>

### ae_gan_hold_tier1_policy_selected_cloud_v1

**ae_gan_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit"></a>

### clockfree_audit

**clockfree_audit: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [clockfree_continuous / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-7f9c23eb0e27-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

**five_word_joint_acquisition: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_acquisition.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [discriminator_stability / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_acquisition"></a>

### gaussian1d_acquisition

**gaussian1d_acquisition: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_acquisition.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Used by: [discriminator_stability / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-grid100"></a>

### grid100

**grid100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2).

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

**img_bars4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

**img_blobs4: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

**img_intensity2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

**img_stripes2: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

**mid_scale_identity: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

**mode_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

**residual_student: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

**ring16_acquisition: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 12.8113 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.345435 | >= 0.15 | PASS |
| hq | 0.800537 | >= 0.85 | FAIL |
| mass_tv | 0.117188 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/da27d343916c4c99be2d372a27dbdb17.json)

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1"></a>

### ring16_acquisition_tier1_policy_selected_cloud_v1

**ring16_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-ring_extension"></a>

### ring_extension

**ring_extension: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3).

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

**ring_hold: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-3).

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

**rotated100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

**staggered100: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-3) · [clockfree_continuous / Tier 3](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-7f9c23eb0e27-experiment-trajectory"></a>

### trajectory

**trajectory: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

**two_pole: PASS**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) · [clockfree_continuous / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.488181 | <= 1 | PASS |
| mean_abs | 0.455219 | >= 0.3 | PASS |

Recorded terminal passing observations: **11**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/046d0735b04d4f299f3cbb92b765fa87.json)

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

Used by: [k3p_two_pole_horizon / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-k3p_two_pole_horizon-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-two_pole_tier1_policy_selected_cloud_v1"></a>

### two_pole_tier1_policy_selected_cloud_v1

**two_pole_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-unipolar"></a>

### unipolar

**unipolar: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

**unused_token_hold: PASS**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-1) · [clockfree_continuous / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.900335 | >= 0.85 | PASS |
| unused_hold | 0.990177 | >= 0.85 | PASS |

Recorded terminal passing observations: **8**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/c83eea7e71ce4193ad9755cb9ce9c125.json)

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-unused_token_hold_tier1_policy_selected_cloud_v1"></a>

### unused_token_hold_tier1_policy_selected_cloud_v1

**unused_token_hold_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Current contract: **unbound**. No recorded task contract binds this cell to the current declaration. No recorded result for this selected configuration and source.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `ef8ccde8d1fa54af8bfce01c044e3671de8131c980eb4e8022d12ffc8caf51d8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-7f9c23eb0e27-experiment-vector_anisotropic"></a>

### vector_anisotropic

**vector_anisotropic: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

**vector_overlap: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

**vector_spiral: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

**vector_two_broad: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

**vector_unequal_mass: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

**vector_unequal_width: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). no compatible result

Used by: [adaptation / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-adaptation-tier-2) · [clockfree_continuous / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](release07-gan-v3-mog.md#cohort-cuda-7f9c23eb0e27-host_profile_transfer-tier-2).

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

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base Adam optimizer.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Related paired relativistic adversarial objective.
- [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906). Inspiration for latent variance/covariance regularization; no full VICReg objective is claimed.
- [On the regularization of Wasserstein GANs](https://arxiv.org/abs/1709.08894). Related work on one-sided input-gradient penalties. The repository evaluates caps directly on real/fake inputs; it does not reproduce the paper's full training or sampling algorithm.
