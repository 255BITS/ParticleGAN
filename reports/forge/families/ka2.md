<!-- Generated Forge family report -->

# KA2

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions)

## Technique overview

KA2 replaces K3P's learning-rate-driven handover with a critic-local controller. It uses the early penalty for 799 applied calls, then mixes early and late penalties equally. Adam moment surprise releases or restores the critic-gradient anchor and controls how quickly its averaged critic follows the live critic. It shares K3P's spike guard and eligible sparse latent-row damping.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior, $x$ a real batch, $z\sim P$ a latent batch and $y$ the generated batch, including any declared output noise. $C$ is the critic actually evaluated: $C=D$ without input noise; otherwise every forward evaluates $D(u+\epsilon_{\mathrm{in}})$ with a fresh draw. $\mathbb E$ means a sample mean, $\lVert\cdot\rVert_2$ is the Euclidean norm, $(a)_+=\max(a,0)$ and $\operatorname{softplus}(a)=\log(1+e^a)$. The scalar sketch detaches $y$ during the critic update and resamples differentiable $y$ after that update; joint and auxiliary hosts retain task-owned objectives. $g_r=\nabla_x C(x)$ and $g_f=\nabla_y C(y)$ are per-sample input gradients; $d$ counts critic-input coordinates. $\lambda$ is penalty strength and $\kappa$ the cap threshold. $\bar C$ evaluates the critic parameter EMA with the same fresh-input-noise law as $C$. $A$ is the early RMS-scaled penalty, $B_{\mathrm{cap}}$ the late L2 cap term and $P_{\mathrm{anchor}}$ the dimension-normalized gradient proximity. $w$ is the configured anchor weight. $n$ counts applied penalty calls, $W\in\{0,1\}$ releases or restores the anchor, and $\alpha\in[0,1]$ is the moment-surprise tracking state. $\delta$ is critic EMA decay and $\theta$ the live critic parameters. Moment surprise compares critic gradient RMS with completed Adam second-moment RMS. $h_i$ is latent row $i$'s current gradient, $h_i^{\mathrm{prev}}$ its last observed gradient, $\rho_i$ the A2 response multiplier and $\Delta z_i$ the corresponding row update. $\operatorname{cos}$ denotes the implementation's bounded cosine similarity, including its handling of zero-length history.

**Default paired adversarial loss**

$$
\begin{aligned}\ell_D&=\mathbb E\!\left[\operatorname{softplus}(C(y)-C(x))\right],\\\ell_G&=\mathbb E\!\left[\operatorname{softplus}(C(x)-C(y))\right].\end{aligned}
$$

These scalar losses are minimized. The critic objective is $\ell_D+R$; the generator adds only its declared auxiliary objectives. Critic fakes are detached, and generator fakes and both scores are recomputed after the critic step. $C$ includes any declared fresh input noise.

**Early, capped and anchor terms**

$$
\begin{aligned} A&=\mathbb E\!\left[\frac{\lVert g_r\rVert_2^2}{d}\right]+\mathbb E\!\left[\left(\frac{\lVert g_f\rVert_2}{\sqrt d}-\kappa\right)_+^2\right],\\ B_{\mathrm{cap}}&=\mathbb E\!\left[(\lVert g_r\rVert_2-\kappa)_+^2\right]+\mathbb E\!\left[(\lVert g_f\rVert_2-\kappa)_+^2\right],\\ P_{\mathrm{anchor}}&=\mathbb E\!\left[\frac{\lVert g_r-\nabla_x\bar C(x)\rVert_2^2}{d}\right].\end{aligned}
$$

$A$ uses RMS input-gradient units; $B_{\mathrm{cap}}$ uses unnormalized L2 caps. When the anchor first starts, copy the live critic and set $P_{\mathrm{anchor}}=0$ for that call. Subsequent calls evaluate its EMA; its fresh input-noise draws follow the same law as the live critic.

**KA2 applied-call handover**

$$
R_{\mathrm{KA2}}=\begin{cases}\dfrac{\lambda}{2}A,&n<800,\\\dfrac{\lambda}{2}\left[\dfrac12 A+\dfrac12\left(B_{\mathrm{cap}}+WwP_{\mathrm{anchor}}\right)\right],&n\ge800.\end{cases}
$$

The first $799$ applied penalty calls use $A$ alone. Starting at call $800$, the equal blend operates even at constant LR. Moment-surprise ratios release $W$ above $3$ and restore it below $1.75$ once reference history exists. Lazy skips do not increment $n$; an applied lazy penalty includes its cadence multiplier.

**KA2 adaptive critic average**

$$
\delta_t=1-0.1\alpha_t,\qquad \bar\theta_t=\delta_t\bar\theta_{t-1}+(1-\delta_t)\theta_t.
$$

After the critic Adam step, the controller records surprise and updates the started anchor using its last penalty-call tracking decision. Here $\delta_t\in[0.9,1]$: calm tracking can freeze the average, while surprise speeds it up. First activation and sustained-surprise reseeding copy the live critic instead of taking this ordinary EMA update.

## Simplified pseudocode

```text
For each training iteration:
  Set role-specific learning rates and noise from the resolved recipe or policy.
  Draw real x and latent z; fake = detach(G(z) + generated-output training noise).
  Dn(u) evaluates D(u + fresh critic-input noise on each forward); D_bar_n uses the same noise law with fresh draws.
  Compute the paired critic adversarial loss shown above.
  g_r = gradient(Dn(x), x); g_f = gradient(Dn(fake), fake), using detached input copies.
  Compute the early RMS-scaled real-gradient/fake-cap term A shown above.
  Compute the late unnormalized real/fake cap term B_caps shown above.
  Increment the applied penalty-call counter; lazy skips do not advance it.
  Calls 1..799: apply the early term A only.
  From call 800:
    When the anchor first becomes active, copy D into D_bar and set proximity = 0 for that call.
    On later blended calls, measure real-input gradient proximity to the noise-wrapped critic average.
    ratio = recent median critic moment surprise / its initial reference median.
    Release W to 0 when ratio > 3; restore W to 1 when ratio < 1.75.
    Use the equal early/late KA2 blend with the current anchor release weight.
  Add penalty every configured k-th step, multiplying its strength by k for lazy application.
  Backpropagate L_D; apply the critic spike guard, then take the critic Adam step.
  After Adam, record moment surprise; update/reseed the critic EMA using the last penalty-call tracking decision.
  Freeze D parameters; draw a fresh latent batch and recompute fake and both critic scores.
  Compute the paired generator adversarial loss from the updated critic.
  Backpropagate L_G through G and the learned prior.
  For an eligible sparse latent table with existing Adam state: if some rows received no gradient and
    cumulative observed-row fraction < max_rate, multiply each row's Adam response by rho.
    Use the A2 cosine-agreement multiplier shown above; without row history use an unchanged response.
  Apply direct sample-particle response gain only when the host explicitly binds that separate parameter group.
  Take the generator/prior Adam step; maintain configured averages; score with the task's declared law.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Paired relativistic logistic on raw scores: $D$ minimizes $\mathbb E[\operatorname{softplus}(C(y)-C(x))]$; $G$ reverses the difference. Input noise wraps the critic in both losses and the penalty; fresh noise is drawn per forward. Joint and auxiliary losses are host-owned. |
| Optimizer | KA2CriticAdam supplies private penalty-call, moment-surprise and adaptive critic-EMA state. The generator/prior uses the shared K3PGeneratorAdam. Selected $\beta_1=0$, $\beta_2=0.999$ and AMSGrad off. KA2 checkpoints retain their own controller identity. |
| Learning rates and annealing | The public base rate is $0.00425$, critic multiplier $1$ and latent-prior multiplier $2$. The selected whole configuration uses $0.006375$ and prior multiplier $1$. $G$ and $D$ hold through $60\%$ of min(resolved task horizon,$1600$), then cosine-decay to $1\%$; the prior follows its full resolved horizon to $5\%$. Hosts may explicitly bind another horizon or transition. |
| Parameter-gradient clipping | The spike guard is adaptive per-tensor clipping: after $200$ prior Adam steps, scale a critic tensor's gradient down when its RMS exceeds $5$ times the RMS predicted by Adam's bias-corrected second moment. It is separate from the input-gradient loss. The family adds no fixed global norm clip. |
| Critic penalties and anchors | $\lambda=1$ and $\kappa=1$. The early $A$ and late $B_{\mathrm{cap}}$ terms use the same RMS/L2 units as K3P. KA2 applies $\lambda A/2$ for the first $799$ penalty calls, then $\frac{\lambda}{2}[\frac12 A+\frac12(B_{\mathrm{cap}}+WwP_{\mathrm{anchor}})]$. This handover remains active at constant LR. Recent/baseline surprise releases $W$ above $3$ and restores it below $1.75$. |
| Damping and update guards | A2 is sparse latent-row damping, with $\rho_i$ in $[0.5,1]$ and $\texttt{max\_rate}=0.5$. It changes the row's Adam response while preserving the raw second-moment update. Nonstandardized row-local priors can expose this hook; standardized or unsupported hosts cannot. Direct sample-particle gain is a separate host-bound rule, up to $2$ times LR when centered gradients agree. |
| Training and sampling noise | Base critic-input standard deviation $0.5$ decreases linearly to zero over the first $10\%$ of the resolved horizon. Generated-output training noise rises from zero to $0.029$ over its first $20\%$. Latent MoG sampling noise is task-owned. Clean gates remove generated-output noise according to their declared sampling law. |
| Parameter averaging and serving | The critic anchor starts at the first blended call. Its EMA decay adapts between $0.9$ and $1$ using moment surprise; sustained released/high-surprise calls can reseed it. This differs from K3P's fixed $0.999$ critic EMA. Generator/prior averaging remains a separate configured $0.995$ state; ordinary selected gates score live weights. |

## Configuration differences

- Each result retains its executed source, recipe, task prior, initialization, budget and sampling law. These descriptions do not change or requalify recorded measurements.
- Task-owned objectives and active components matter: direct sample particles, learned latent rows and a generator network are different parameter roles. A declared recipe switch does not imply that every host can apply it.
- Selected ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f at source 928b485ffbe6e17d79b41d307ae8b6275489b37a uses base LR .006375 and prior multiplier 1, versus public defaults .00425 and 2.
- KA2's controller is independent of LR; it still has a fixed applied-call warmup. Its ordinary recipe also retains explicit LR/noise schedules. This should not be described as an entirely clock-free formulation.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [configs/forge/ideas/ka2.json](../../../configs/forge/ideas/ka2.json)
- [configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json](../../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/ka2.py](../../../particlegan/ka2.py)
- [particlegan/k3p.py](../../../particlegan/k3p.py)
- [particlegan/grad_regularizers.py](../../../particlegan/grad_regularizers.py)
- [particlegan/policy.py](../../../particlegan/policy.py)
- [https://github.com/255BITS/ParticleGAN/blob/928b485ffbe6e17d79b41d307ae8b6275489b37a/particlegan/ka2.py](https://github.com/255BITS/ParticleGAN/blob/928b485ffbe6e17d79b41d307ae8b6275489b37a/particlegan/ka2.py)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

<a name="cohort-cuda-0d83d78027c5"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [ka2 · 093c6f2bd417](../../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json).

Recorded qualification: **tier 0**, discriminator_stability revision 5. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `21ec7e3f89402b5d0c77669d5fba0e36bcf4e690069cfdceab383520006c525f`. Candidate revision: `a851463cd36ee1a5b8cd796f34a47bf215e33e00fe09c1232c2c3588353bc892`. Runtime cohort: `fb17123bbc348e7f21c906529039fcbc098648b5cf2fe1c0f2d8e1cdcf72a416`.

[Frozen numerical evidence](../technique-evidence/ddde64ee936114becac42863847a51e88ec0301bad9f7f69767d0a2fbc3f3d69.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: current_measurement. Complete current Tier 1 measurement at one frozen recipe and executed source. FAIL completes a measurement; calibration and confirmation remain separate.

</details>

Complete current Tier 1 measurement in: discriminator_stability; additional scoped probes: clockfree_audit_measurement_v1. PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation) | [3/3](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) | [0(*)/19](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) | [0(*)/1](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) | [3(*)/23](ka2.md#cohort-cuda-0d83d78027c5-adaptation) |
| [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous) | [3/4](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | [0(*)/19](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) | [0(*)/7](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | [3(*)/30](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous) |
| [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability) | [4(*)/6](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | [0(*)/19](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | [0(*)/2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) | [4(*)/27](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability) |
| [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison) | [3/3](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) | [0(*)/19](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) | [0(*)/2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) | [3(*)/24](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison) |
| [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer) | [3/3](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) | [0(*)/19](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) | [0(*)/2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | [3(*)/24](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer) |
| [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage) | [3/3](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | [0(*)/19](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | [0/0](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-3) | [3(*)/22](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) | [0(*)/7](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1) | [0/0](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2) | [0/0](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3) | [0(*)/7](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) |

\* indicates incomplete results, including changed or unbound current contracts.

<a name="cohort-cuda-0d83d78027c5-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ae_gan_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | matches |
| [clockfree_audit_measurement_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | FAIL | matches |
| [five_word_joint_acquisition](ka2.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition) | [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | FAIL | CHANGED |
| [gaussian1d_acquisition](ka2.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition) | [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | FAIL | matches |
| [ring16_acquisition](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | PASS | matches |
| [two_pole](ka2.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | matches |
| [unused_token_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [cover_leftover](ka2.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [grid100](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_bars4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_blobs4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_intensity2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [img_stripes2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [mid_scale_identity](ka2.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [mode_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [residual_student](ka2.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [rotated100](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [staggered100](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [trajectory](ka2.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [unipolar](ka2.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_anisotropic](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_overlap](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_spiral](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_two_broad](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_unequal_mass](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |
| [vector_unequal_width](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Current contract |
| --- | --- | --- | --- |
| [clockfree_audit](ka2.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [grid100_14k](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [ring_extension](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches |
| [ring_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches |
| [rotated100_14k](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [staggered100_14k](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |
| [target_shift_recovery](ka2.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | [adaptation](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-3), [clockfree_continuous](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](ka2.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](ka2.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](ka2.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](ka2.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](ka2.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](ka2.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [target_shift_recovery](ka2.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | unbound |

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
| [two_pole](ka2.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |
| [clockfree_audit_measurement_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | required | FAIL | matches |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](ka2.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](ka2.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](ka2.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](ka2.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](ka2.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [clockfree_audit](ka2.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | required | UNKNOWN | unbound |
| [ring_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |
| [grid100_14k](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | required | UNKNOWN | unbound |
| [rotated100_14k](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | required | UNKNOWN | unbound |
| [staggered100_14k](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | required | UNKNOWN | unbound |
| [target_shift_recovery](ka2.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 5**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 19 / 2**.

Calibration: **provisional**. Expanded six-task Tier 1 placement is provisional and requires bounded calibration. Revision 3 and prior profiles retain their original tasks and evidence; a standalone scalar pass gives no whole-view/default credit.

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [gaussian1d_acquisition](ka2.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition) | required | FAIL | matches |
| [two_pole](ka2.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |
| [ring16_acquisition](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | required | PASS | matches |
| [five_word_joint_acquisition](ka2.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition) | required | FAIL | CHANGED |
| [clockfree_audit_measurement_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | diagnostic | FAIL | matches |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](ka2.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](ka2.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](ka2.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](ka2.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](ka2.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](ka2.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](ka2.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](ka2.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](ka2.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](ka2.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](ka2.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |
| [img_intensity2_residual16](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | unbound |
| [vector_two_broad_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_mass_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_width_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | unbound |
| [vector_anisotropic_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | unbound |
| [vector_overlap_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | unbound |
| [vector_spiral_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | unbound |
| [img_stripes2_residual16](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | unbound |
| [img_bars4_residual16](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | unbound |
| [img_blobs4_residual16](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_square_named_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [rotated100_affine_square_named_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [staggered100_affine_square_named_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_paired_laws_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | unbound |
| [grid100_release07_cloud_named_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole](ka2.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](ka2.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](ka2.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](ka2.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](ka2.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](ka2.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |
| [img_intensity2_residual16](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | unbound |
| [vector_two_broad_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_mass_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | unbound |
| [vector_unequal_width_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | unbound |
| [vector_anisotropic_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | unbound |
| [vector_overlap_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | unbound |
| [vector_spiral_published](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | unbound |
| [img_stripes2_residual16](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | unbound |
| [img_bars4_residual16](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | unbound |
| [img_blobs4_residual16](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | unbound |
| [grid100_affine_square_named_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [rotated100_affine_square_named_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |
| [staggered100_affine_square_named_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | unbound |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [ring_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches |
| [ring_extension](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches |

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | unbound |
| [two_pole_800_schedule800_diagnostic_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | unbound |

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
| [two_pole](ka2.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | matches |
| [unused_token_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | matches |
| [ae_gan_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | matches |

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Current contract |
| --- | --- | --- | --- |
| [trajectory](ka2.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | matches |
| [residual_student](ka2.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | matches |
| [unipolar](ka2.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | matches |
| [cover_leftover](ka2.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | matches |
| [mid_scale_identity](ka2.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | matches |
| [mode_hold](ka2.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches |
| [vector_two_broad](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches |
| [vector_unequal_mass](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches |
| [vector_unequal_width](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches |
| [vector_anisotropic](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches |
| [vector_overlap](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches |
| [vector_spiral](ka2.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches |
| [img_stripes2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches |
| [img_bars4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches |
| [img_blobs4](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches |
| [img_intensity2](ka2.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches |
| [grid100](ka2.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches |
| [rotated100](ka2.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches |
| [staggered100](ka2.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [two_pole_tier1_policy_selected_cloud_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [unused_token_hold_tier1_policy_selected_cloud_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |
| [clockfree_audit_tier1_policy_selected_cloud_v1](ka2.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | unbound |

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

Used by: [adaptation / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00557699 | <= 0.35 | PASS |
| recon_mse | 0.00439529 | <= 0.05 | PASS |

Recorded terminal passing observations: **21**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/bf6d669e859d4c00bda87284addca604.json)

[Actual-training GIF](../tier1-completion/media/ka2/bf6d669e859d4c00bda87284addca604/ae_gan_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Current contract: **matches**. Current task contract matches the recorded conditions. step_label changed the update or common-prefix state

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/ce6eec65990e41269dd54b438e68625c.json)

[Actual-training GIF](../tier1-completion/media/ka2/ce6eec65990e41269dd54b438e68625c/clockfree_audit_measurement_v1.gif); 4 saved observations; no new optimizer updates or sampling draws.

Recorded clock parity diagnostics:

| Condition | Exact state digest equality |
| --- | --- |
| evaluation_cadence | equal |
| horizon | different |
| restart | equal |
| step_label | different |

Recorded unexplained clock dependencies: **5**.

- learning-rate annealing depends on completed steps and horizon
- input-noise annealing depends on completed steps and horizon
- output-noise warmup depends on completed steps and horizon
- KA2 switches from pure A to blended penalty at call 800
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

Used by: [tier1_policy_coverage / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Current contract: **matches**. Current task contract matches the recorded conditions. no compatible result

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

**five_word_joint_acquisition: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_acquisition.json).

Current contract: **CHANGED**. Current coverage is stale: changed evaluation (gates or sampling law). recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| minimum_reconstruction_token_probability | 0.947841 | >= 0.9 | PASS |
| modes | 5 | == 5 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| reconstruction_exact | 1 | == 1 | PASS |
| sample_count | 1024 | >= 1024 | PASS |

Recorded terminal passing observations: **1**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/859d609d7dee4dfab70bb52bd6dcc1dc.json)

[Actual-training GIF](../tier1-completion/media/ka2/859d609d7dee4dfab70bb52bd6dcc1dc/five_word_joint_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [discriminator_stability / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.027005 | <= 0.05 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.0171821 | <= 0.2 | PASS |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 0.971122 | <= 1.2 | PASS |

Recorded terminal passing observations: **3**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/0a80066804bc4760bafd5110710304a7.json)

[Actual-training GIF](../tier1-completion/media/ka2/0a80066804bc4760bafd5110710304a7/gaussian1d_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

**ring16_acquisition: PASS**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Current contract: **matches**. Current task contract matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.636137 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.319289 | >= 0.15 | PASS |
| hq | 0.935303 | >= 0.85 | PASS |
| mass_tv | 0.0810547 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **5**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/764fcd21c0de4d95a8bb898609dfaa72.json)

[Actual-training GIF](../tier1-completion/media/ka2/764fcd21c0de4d95a8bb898609dfaa72/ring16_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

Used by: [clockfree_continuous / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) · [clockfree_continuous / Tier 3](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.121668 | <= 1 | PASS |
| mean_abs | 0.428871 | >= 0.3 | PASS |

Recorded terminal passing observations: **7**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/02b97792b1e24d6f98f8931ca77cf5f6.json)

[Actual-training GIF](../tier1-completion/media/ka2/02b97792b1e24d6f98f8931ca77cf5f6/two_pole.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [k3p_two_pole_horizon / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.992696 | >= 0.85 | PASS |
| unused_hold | 0.999316 | >= 0.85 | PASS |

Recorded terminal passing observations: **15**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/49dcdd59005a4413afc2145348bdfa31.json)

[Actual-training GIF](../tier1-completion/media/ka2/49dcdd59005a4413afc2145348bdfa31/unused_token_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](ka2.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](ka2.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base optimizer. Guard, damping and controller rules are repository additions.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Related paired relativistic adversarial objective.
- [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406). Zero-centered input-gradient regularization; not the full repository formulation.
