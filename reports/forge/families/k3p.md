<!-- Generated Forge family report -->

# K3P

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions)

## Technique overview

K3P couples a changing critic penalty to the critic's applied learning rate. At high rates it suppresses real-data input gradients and caps fake gradients; as the rate falls it adds real/fake caps and pulls the critic's input gradients toward an averaged critic. Its Adam wrappers also limit critic gradient spikes and damp inconsistent updates to sparsely sampled latent rows. These extra rules are repository-specific.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior, $x$ a real batch, $z\sim P$ a latent batch and $y$ the generated batch, including any declared output noise. $C$ is the critic actually evaluated: $C=D$ without input noise; otherwise every forward evaluates $D(u+\epsilon_{\mathrm{in}})$ with a fresh draw. $\mathbb E$ means a sample mean, $\lVert\cdot\rVert_2$ is the Euclidean norm, $(a)_+=\max(a,0)$ and $\operatorname{softplus}(a)=\log(1+e^a)$. The scalar sketch detaches $y$ during the critic update and resamples differentiable $y$ after that update; joint and auxiliary hosts retain task-owned objectives. $g_r=\nabla_x C(x)$ and $g_f=\nabla_y C(y)$ are per-sample input gradients; $d$ counts critic-input coordinates. $\lambda$ is penalty strength and $\kappa$ the cap threshold. $\bar C$ evaluates the critic parameter EMA with the same fresh-input-noise law as $C$. $A$ is the early RMS-scaled penalty, $B_{\mathrm{cap}}$ the late L2 cap term and $P_{\mathrm{anchor}}$ the dimension-normalized gradient proximity. $w$ is the configured anchor weight. $r$ is last applied critic LR divided by its largest recorded LR; $f$ is the resolved network LR floor fraction; $s$ is the early-penalty weight. $h_i$ is latent row $i$'s current gradient, $h_i^{\mathrm{prev}}$ its last observed gradient, $\rho_i$ the A2 response multiplier and $\Delta z_i$ the corresponding row update. $\operatorname{cos}$ denotes the implementation's bounded cosine similarity, including its handling of zero-length history.

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

**K3P learning-rate handover**

$$
\begin{aligned}s&=\frac{\max\!\left(0,\min(1,2r)-2f\right)}{1-2f},\\R_{\mathrm{K3P}}&=\frac{\lambda}{2}\left[sA+(1-s)\left(B_{\mathrm{cap}}+wP_{\mathrm{anchor}}\right)\right].\end{aligned}
$$

Before the first critic step, $s=1$. At $s=1$ evaluate $A$ alone; start the anchor only when $s<1$. Constant critic LR keeps the early term active. Lazy cadence $k$ applies $kR_{\mathrm{K3P}}$ every $k$th scheduled call and zero otherwise.

**A2 sparse-row response**

$$
\begin{aligned}\rho_i&=\begin{cases}0.75+0.25\operatorname{cos}(h_i,h_i^{\mathrm{prev}}),&\text{with row history},\\1,&\text{without row history},\end{cases}\\\Delta z_i^{\mathrm{A2}}&=\rho_i\Delta z_i^{\mathrm{Adam}}.\end{aligned}
$$

A2 acts only on an eligible row-local table with existing Adam state, some zero-gradient rows, and cumulative observed-row fraction below the configured cutoff (baseline $0.5$). Then $\rho_i\in[0.5,1]$. Adam still accumulates the raw gradient second moment; unsupported or ineligible hosts keep their ordinary response.

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
  r = last applied critic LR / largest recorded critic LR; f = resolved network LR floor.
  Compute the early-penalty blend weight from the applied critic LR ratio and network floor; use s=1 before the first critic step.
  If s < 1:
    When the anchor first becomes active, copy D into D_bar and set proximity = 0 for that call.
    On later blended calls, measure real-input gradient proximity to the noise-wrapped critic average.
  Add the early/late blended penalty shown above using the active anchor weight.
    At s=1 evaluate A alone; no anchor is evaluated or started.
    Lazy cadence k applies the penalty every k-th step at k times its declared strength.
  Backpropagate L_D; apply the critic spike guard, then take the critic Adam step and update its record/anchor.
  Freeze D parameters; draw a fresh latent batch and recompute fake and both critic scores.
  Compute the paired generator adversarial loss from the updated critic.
  Backpropagate L_G through G and the learned prior.
  For an eligible sparse latent table with existing Adam state: if some rows received no gradient and
    cumulative observed-row fraction < max_rate, multiply each row's Adam response by rho.
    Use the A2 cosine-agreement multiplier shown above; without row history use an unchanged response.
  Apply direct sample-particle response gain only when the host explicitly binds that separate parameter group.
  Take the generator/prior Adam step; maintain configured averages; score with the task's declared law.
```


## Configuration differences

- Each result retains its executed source, recipe, task prior, initialization, budget and sampling law. These descriptions do not change or requalify recorded measurements.
- Task-owned objectives and active components matter: direct sample particles, learned latent rows and a generator network are different parameter roles. A declared recipe switch does not imply that every host can apply it.
- The current selected candidate is k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c, recorded at source 928b485ffbe6e17d79b41d307ae8b6275489b37a. Its .006375 rate and prior multiplier 1 differ from the canonical public defaults.
- The critic's EMA anchor is a training regularizer; the generator/prior EMA is a separate optional evaluation state. Neither supplies another configuration's gate credit.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [configs/forge/ideas/k3p.json](../../../configs/forge/ideas/k3p.json)
- [configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json](../../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/grad_regularizers.py](../../../particlegan/grad_regularizers.py)
- [particlegan/k3p.py](../../../particlegan/k3p.py)
- [particlegan/gan_loss.py](../../../particlegan/gan_loss.py)
- [particlegan/policy.py](../../../particlegan/policy.py)
- [https://github.com/255BITS/ParticleGAN/blob/928b485ffbe6e17d79b41d307ae8b6275489b37a/particlegan/k3p.py](https://github.com/255BITS/ParticleGAN/blob/928b485ffbe6e17d79b41d307ae8b6275489b37a/particlegan/k3p.py)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

<a name="cohort-cuda-1bf9d7d34422"></a>

## CUDA results

Runtime: **cuda**. Selected configuration: [k3p · 0b37e98a01e3](../../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `6269a18ac4f82564cb16ba19afa4b3dd2f836a2b4085fbe5aeb81a35a453e895`. Candidate revision: `ee4b65dd6dbe019a8099ccd05013eee2876acea1cc7c04904c14fc53eefd835e`. Runtime cohort: `0474d5313a8fd0ae863081733be2556dfd70a9639cf940149c736d37358e396f`.

[Frozen numerical evidence](../technique-evidence/7c4e188f1ee495526c2decf6f02a45f392123aaf4f33f1f6fb9a398fe3ac05e7.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: current_measurement. Preserve the round's pre-run whole candidate choice in its freshly measured CUDA source cohort; no task pooling, outcome-based recipe reselection, calibration or default adoption.

</details>

Complete current Tier 1 measurement in: discriminator_stability. PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.

## Best recorded configuration

Selected by recorded required passes, then completed measurements. Each count comes from this one complete configuration. Source differences preserve separate evidence contracts; the selection does not establish a controlled win or default adoption.

Recorded trainer recipe; task-owned architecture, prior, initialization, budget and sampling remain in the experiment receipts below. Null role overrides inherit the shared value. Optimizer parameters only apply to optimizers that consume them.

| Setting | Selected value |
| --- | --- |
| Adversarial loss | loss=relativistic; loss_labels=[0.0, 1.0, 1.0] |
| Optimizer | optimizer_family=formulation; optimizer_momentum=0.0 |
| Learning rates | lr=0.006375; d_lr_mult=1.0; prior_lr_mult=1.0 |
| Rate schedule | lr_schedule=cosine; lr_floor=0.05; network_lr_floor=0.01; network_lr_horizon_cap=1600 |
| Critic penalty | reg_arm=None; reg_coeff=1.0; reg_kappa=1.0; reg_every=1; reg_anchor_weight=1.0 |
| Damping and guards | d_guard_ratio=5.0; latent_damping_max_rate=0.5; direct_particle_gain=True |
| Training noise | input_noise_std=0.5; output_noise_std=0.029; output_noise_mode=fixed |
| Averaging | ema_decay=0.995; serve_average=0.0 |

Base network rate: **0.006375**; critic rate: **0.006375**; prior rate: **0.006375** before any declared schedule or host adaptation.

Selected optimizer rule: K3PCriticAdam and K3PGeneratorAdam wrap PyTorch Adam. Selected moments are $\beta_1=0$, $\beta_2=0.999$, AMSGrad off. The wrappers include critic observation/anchor updates, spike guarding, eligible A2 and optional direct-particle response.

<details>
<summary>Other recorded configurations in this family</summary>

These are whole configurations under their original sources. Their individual passing cells do not contribute to the selected result.

| Configuration | Required passes | Executed source | Display selection |
| --- | ---: | --- | --- |
| [k3p · 0b37e98a01e3](k3p.md) | 19(*)/151 | `6269a18ac4f8` | Selected |
| [k3p-no-output-noise-diagnostic](k3p-no-training-noise.md) | 14(*)/151 | `6269a18ac4f8` | Alternative |
| [k3p-a2-off-native-diagnostic](k3p-no-a2.md) | 13(*)/151 | `6269a18ac4f8` | Alternative |

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3(*)/23](k3p.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3(*)/30](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [4/6](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/20](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [4(*)/28](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3(*)/24](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3(*)/24](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3(*)/22](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) | [0(*)/7](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1) | [0/0](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-2) | [0/0](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-3) | [0(*)/7](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-1bf9d7d34422-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | FAIL | matches recorded run |
| [five_word_joint_acquisition](k3p.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition) | [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [gaussian1d_smoke](k3p.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | PASS | matches recorded run |
| [ring16_acquisition](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [two_pole](k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](k3p.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [gaussian1d_stability](k3p.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [trajectory](k3p.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [ring_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | [adaptation](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3), [clockfree_continuous](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](k3p.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | required | FAIL | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 7**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 20 / 2**.

Calibration: **provisional**. Revision7 smoke/stability separation is provisional and requires bounded calibration before default adoption. Historical Gaussian acquisition evidence retains its original sigma.025 cohort and five-terminal gate.

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](k3p.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | required | PASS | matches recorded run |
| [two_pole](k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [ring16_acquisition](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | required | FAIL | matches recorded run |
| [five_word_joint_acquisition](k3p.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition) | required | FAIL | matches recorded run |
| [clockfree_audit_measurement_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](k3p.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | required | UNKNOWN | matches recorded run |
| [trajectory](k3p.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | UNKNOWN | matches recorded run |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

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

Used by: [adaptation / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00723454 | <= 0.35 | PASS |
| recon_mse | 0.00358433 | <= 0.05 | PASS |

Recorded terminal passing observations: **11**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d0dea0262ec74da39fff2118533f7ed0.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. step_label changed the update or common-prefix state

Actual task device: `0` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/58848edcb0a145f593e6c27662beac8b.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition"></a>

### five_word_joint_acquisition

**five_word_joint_acquisition: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| mass_tv | 0.0189453 | <= 0.1 | PASS |
| minimum_reconstruction_token_probability | 0.000107317 | >= 0.9 | FAIL |
| modes | 5 | == 5 | PASS |
| quality_fraction | 1 | >= 0.95 | PASS |
| reconstruction_exact | 0 | == 1 | FAIL |
| sample_count | 1024 | >= 1024 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/b71bf91ff3234ece991ef2dd5a1020fd.json)

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

<a name="cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-1bf9d7d34422-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

**gaussian1d_smoke: PASS**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. any scheduled full pass with independent same-state confirmation

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.0346106 |
| finite_fraction | 1 |
| mean | 1.96819 |
| mean_error_sigma | 0.0636115 |
| sample_count | 4096 |
| std | 0.475962 |
| std_ratio | 0.951925 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/c8e242a0e03a4c7d9488f0fc3762a442.json)

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

Used by: [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [discriminator_stability / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 2.70369 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.442025 | >= 0.15 | PASS |
| hq | 0.779053 | >= 0.85 | FAIL |
| mass_tv | 0.0629883 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/38b7de44cdc3478bad29445209a8c4d8.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) · [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.187403 | <= 1 | PASS |
| mean_abs | 0.447848 | >= 0.3 | PASS |

Recorded terminal passing observations: **8**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d8f917001d51447abfdaf7c2372b94cd.json)

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.996966 | >= 0.85 | PASS |
| unused_hold | 0.999836 | >= 0.85 | PASS |

Recorded terminal passing observations: **16**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/e320e05e3878484ca26059040e370f2f.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Runtime: **cuda**. Selected configuration: [k3p · 0b37e98a01e3](../../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `d276c5a7344fab6ec5de7b314d3982af5b0ef8027c8b89c01376ae366844a9cb`. Candidate revision: `654453f7e21d30e45d54b20bb43833fe4e97b0fc83122cb605579b7a50282502`. Runtime cohort: `06a5c0ef12e139273d97ffbdca58e5bb0e23e40710e0253d95aa45b7ef57288e`.

[Frozen numerical evidence](../technique-evidence/83f549889bf36b44bda98c8fe4c0f1600ef0cdd1d12c529f9e480ffcfdc93512.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Preserve the exact archived revision-7 measurement and its original policy qualification; no current-policy measurement or default-adoption credit.

</details>

## Best recorded configuration

Selected by recorded required passes, then completed measurements. Each count comes from this one complete configuration. Source differences preserve separate evidence contracts; the selection does not establish a controlled win or default adoption.

Recorded trainer recipe; task-owned architecture, prior, initialization, budget and sampling remain in the experiment receipts below. Null role overrides inherit the shared value. Optimizer parameters only apply to optimizers that consume them.

| Setting | Selected value |
| --- | --- |
| Adversarial loss | loss=relativistic; loss_labels=[0.0, 1.0, 1.0] |
| Optimizer | optimizer_family=formulation; optimizer_momentum=0.0 |
| Learning rates | lr=0.006375; d_lr_mult=1.0; prior_lr_mult=1.0 |
| Rate schedule | lr_schedule=cosine; lr_floor=0.05; network_lr_floor=0.01; network_lr_horizon_cap=1600 |
| Critic penalty | reg_arm=None; reg_coeff=1.0; reg_kappa=1.0; reg_every=1; reg_anchor_weight=1.0 |
| Damping and guards | d_guard_ratio=5.0; latent_damping_max_rate=0.5; direct_particle_gain=True |
| Training noise | input_noise_std=0.5; output_noise_std=0.029; output_noise_mode=fixed |
| Averaging | ema_decay=0.995; serve_average=0.0 |

Base network rate: **0.006375**; critic rate: **0.006375**; prior rate: **0.006375** before any declared schedule or host adaptation.

Selected optimizer rule: K3PCriticAdam and K3PGeneratorAdam wrap PyTorch Adam. Selected moments are $\beta_1=0$, $\beta_2=0.999$, AMSGrad off. The wrappers include critic observation/anchor updates, spike guarding, eligible A2 and optional direct-particle response.

<details>
<summary>Other recorded configurations in this family</summary>

These are whole configurations under their original sources. Their individual passing cells do not contribute to the selected result.

| Configuration | Required passes | Executed source | Display selection |
| --- | ---: | --- | --- |
| [k3p · 0b37e98a01e3](k3p.md) | 19(*)/151 | `d276c5a7344f` | Selected |
| [k3p-no-output-noise-diagnostic](k3p-no-training-noise.md) | 14(*)/151 | `d276c5a7344f` | Alternative |
| [k3p-a2-off-native-diagnostic](k3p-no-a2.md) | 13(*)/151 | `d276c5a7344f` | Alternative |

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation) | [3/3](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-1) | [0(*)/19](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) | [0(*)/1](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-3) | [3(*)/23](k3p.md#cohort-cuda-c195899a64af-adaptation) |
| [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous) | [3/4](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) | [0(*)/19](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) | [0(*)/7](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | [3(*)/30](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous) |
| [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability) | [4/6](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | [0(*)/20](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | [0(*)/2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) | [4(*)/28](k3p.md#cohort-cuda-c195899a64af-discriminator_stability) |
| [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison) | [3/3](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) | [0(*)/19](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) | [0(*)/2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) | [3(*)/24](k3p.md#cohort-cuda-c195899a64af-formulation_comparison) |
| [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer) | [3/3](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) | [0(*)/19](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) | [0(*)/2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | [3(*)/24](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer) |
| [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage) | [3/3](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | [0(*)/19](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | [0/0](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-3) | [3(*)/22](k3p.md#cohort-cuda-c195899a64af-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage) | [0(*)/7](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1) | [0/0](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-2) | [0/0](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-3) | [0(*)/7](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-c195899a64af-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](k3p.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) | FAIL | matches recorded run |
| [five_word_joint_acquisition](k3p.md#cohort-cuda-c195899a64af-experiment-five_word_joint_acquisition) | [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | FAIL | changed since run |
| [gaussian1d_smoke](k3p.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | PASS | matches recorded run |
| [ring16_acquisition](k3p.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | FAIL | matches recorded run |
| [two_pole](k3p.md#cohort-cuda-c195899a64af-experiment-two_pole) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](k3p.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [gaussian1d_stability](k3p.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-c195899a64af-experiment-grid100) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-c195899a64af-experiment-img_bars4) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-c195899a64af-experiment-mode_hold) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-c195899a64af-experiment-residual_student) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [trajectory](k3p.md#cohort-cuda-c195899a64af-experiment-trajectory) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-c195899a64af-experiment-unipolar) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](k3p.md#cohort-cuda-c195899a64af-experiment-grid100_14k) | [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](k3p.md#cohort-cuda-c195899a64af-experiment-ring_extension) | [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [ring_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ring_hold) | [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3), [discriminator_stability](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3), [formulation_comparison](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3), [host_profile_transfer](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100_14k) | [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100_14k) | [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | [adaptation](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-3), [clockfree_continuous](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-c195899a64af-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](k3p.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [clockfree_audit_measurement_v1](k3p.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | required | FAIL | matches recorded run |

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](k3p.md#cohort-cuda-c195899a64af-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 7**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 20 / 2**.

Calibration: **provisional**. Revision7 smoke/stability separation is provisional and requires bounded calibration before default adoption. Historical Gaussian acquisition evidence retains its original sigma.025 cohort and five-terminal gate.

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](k3p.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | required | PASS | matches recorded run |
| [two_pole](k3p.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |
| [ring16_acquisition](k3p.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | required | FAIL | matches recorded run |
| [five_word_joint_acquisition](k3p.md#cohort-cuda-c195899a64af-experiment-five_word_joint_acquisition) | required | FAIL | changed since run |
| [clockfree_audit_measurement_v1](k3p.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](k3p.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | required | UNKNOWN | matches recorded run |
| [trajectory](k3p.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p.md#cohort-cuda-c195899a64af-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p.md#cohort-cuda-c195899a64af-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p.md#cohort-cuda-c195899a64af-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p.md#cohort-cuda-c195899a64af-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p.md#cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](k3p.md#cohort-cuda-c195899a64af-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](k3p.md#cohort-cuda-c195899a64af-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p.md#cohort-cuda-c195899a64af-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p.md#cohort-cuda-c195899a64af-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p.md#cohort-cuda-c195899a64af-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p.md#cohort-cuda-c195899a64af-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p.md#cohort-cuda-c195899a64af-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p.md#cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](k3p.md#cohort-cuda-c195899a64af-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](k3p.md#cohort-cuda-c195899a64af-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | PASS | matches recorded run |
| [unused_token_hold](k3p.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | PASS | matches recorded run |
| [ae_gan_hold](k3p.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | PASS | matches recorded run |

<a name="cohort-cuda-c195899a64af-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | UNKNOWN | matches recorded run |
| [unipolar](k3p.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | UNKNOWN | matches recorded run |
| [cover_leftover](k3p.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | UNKNOWN | matches recorded run |
| [mode_hold](k3p.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-c195899a64af-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | UNKNOWN | matches recorded run |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-c195899a64af-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-c195899a64af-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-c195899a64af-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-c195899a64af-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

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

Used by: [adaptation / Tier 1](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00685861 | <= 0.35 | PASS |
| recon_mse | 0.00427632 | <= 0.05 | PASS |

Recorded terminal passing observations: **21**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/47a4406793ab447dad28d85dfcc77b0f.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. step_label changed the update or common-prefix state

Actual task device: `1` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/168933ecccf2453193220fad65edd038.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

<a name="cohort-cuda-c195899a64af-experiment-five_word_joint_acquisition"></a>

### five_word_joint_acquisition

**five_word_joint_acquisition: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/five_word_joint_acquisition.json).

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| mass_tv | 0.4 | <= 0.1 | FAIL |
| minimum_reconstruction_token_probability | 0 | >= 0.9 | FAIL |
| modes | 3 | == 5 | FAIL |
| quality_fraction | 1 | >= 0.95 | PASS |
| reconstruction_exact | 0 | == 1 | FAIL |
| sample_count | 1024 | >= 1024 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/129e9d051ab349538633d6b85d3d8258.json)

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

<a name="cohort-cuda-c195899a64af-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1"></a>

### five_word_joint_acquisition_tier1_policy_selected_cloud_v1

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-c195899a64af-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Actual task device: `0` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

Recorded final metrics:

| Metric | Measured |
| --- | ---: |
| cdf_ks | 0.0307608 |
| finite_fraction | 1 |
| mean | 1.96764 |
| mean_error_sigma | 0.0647124 |
| sample_count | 4096 |
| std | 0.484358 |
| std_ratio | 0.968716 |
| step | 1000 |

[Compact metrics and receipt provenance](../technique-receipts/2211d306d84045f3ac208111add2d516.json)

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

Used by: [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [discriminator_stability / Tier 1](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 2.59688 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 0.436038 | >= 0.15 | PASS |
| hq | 0.783447 | >= 0.85 | FAIL |
| mass_tv | 0.0449219 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/3684bb53a93448bfa436af7932ce366d.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-3) · [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.187403 | <= 1 | PASS |
| mean_abs | 0.447847 | >= 0.3 | PASS |

Recorded terminal passing observations: **8**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/fbd7476f43154bd3a16142f86771f529.json)

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p.md#cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p.md#cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.996966 | >= 0.85 | PASS |
| unused_hold | 0.999836 | >= 0.85 | PASS |

Recorded terminal passing observations: **16**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/81ad7903834b4d688f2ba561b9005b14.json)

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Runtime: **cuda**. Selected configuration: [k3p · 0b37e98a01e3](../../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `21ec7e3f89402b5d0c77669d5fba0e36bcf4e690069cfdceab383520006c525f`. Candidate revision: `de743d96953782a0f6079843dbe6a10a1145226c1b83660bee550c9c45bce71d`. Runtime cohort: `942ae9802bd828bc36d093df0b2bebecfcbe074fe8c8a45c44ab9a4ffd5fcc01`.

[Frozen numerical evidence](../technique-evidence/ddde64ee936114becac42863847a51e88ec0301bad9f7f69767d0a2fbc3f3d69.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Retain the exact complete measurement under its original joint-word evaluator source binding. The current v4 evaluator contract differs; this archived evidence grants no current-measurement or default-adoption claim.

</details>

## Best recorded configuration

Selected by recorded required passes, then completed measurements. Each count comes from this one complete configuration. Source differences preserve separate evidence contracts; the selection does not establish a controlled win or default adoption.

Recorded trainer recipe; task-owned architecture, prior, initialization, budget and sampling remain in the experiment receipts below. Null role overrides inherit the shared value. Optimizer parameters only apply to optimizers that consume them.

| Setting | Selected value |
| --- | --- |
| Optimizer | optimizer_family=formulation |
| Learning rates | lr=0.006375; d_lr_mult=1.0; prior_lr_mult=1.0 |
| Rate schedule | lr_floor=0.05; network_lr_floor=0.01; network_lr_horizon_cap=1600 |
| Critic penalty | reg_arm=None; reg_coeff=1.0; reg_kappa=1.0; reg_every=1; reg_anchor_weight=1.0 |
| Damping and guards | d_guard_ratio=5.0; latent_damping_max_rate=0.5; direct_particle_gain=True |
| Training noise | input_noise_std=0.5; output_noise_std=0.029; output_noise_mode=fixed |
| Averaging | ema_decay=0.995; serve_average=0.0 |

Base network rate: **0.006375**; critic rate: **0.006375**; prior rate: **0.006375** before any declared schedule or host adaptation.

Selected optimizer rule: K3PCriticAdam and K3PGeneratorAdam wrap PyTorch Adam. Selected moments are $\beta_1=0$, $\beta_2=0.999$, AMSGrad off. The wrappers include critic observation/anchor updates, spike guarding, eligible A2 and optional direct-particle response.

<details>
<summary>Other recorded configurations in this family</summary>

These are whole configurations under their original sources. Their individual passing cells do not contribute to the selected result.

| Configuration | Required passes | Executed source | Display selection |
| --- | ---: | --- | --- |
| [k3p · 0b37e98a01e3](k3p.md) | 19(*)/151 | `21ec7e3f8940` | Selected |
| [k3p-no-output-noise-diagnostic](k3p-no-training-noise.md) | 13(*)/151 | `21ec7e3f8940` | Alternative |
| [k3p-a2-off-native-diagnostic](k3p-no-a2.md) | 12(*)/151 | `21ec7e3f8940` | Alternative |

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation) | [3/3](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) | [0(*)/19](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) | [0(*)/1](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) | [3(*)/23](k3p.md#cohort-cuda-0d83d78027c5-adaptation) |
| [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous) | [3/4](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | [0(*)/19](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) | [0(*)/7](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | [3(*)/30](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous) |
| [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability) | [4(*)/6](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | [0(*)/20](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | [0(*)/2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) | [4(*)/28](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability) |
| [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison) | [3/3](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) | [0(*)/19](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) | [0(*)/2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) | [3(*)/24](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison) |
| [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer) | [3/3](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) | [0(*)/19](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) | [0(*)/2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | [3(*)/24](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer) |
| [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage) | [3/3](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | [0(*)/19](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | [0/0](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-3) | [3(*)/22](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) | [0(*)/7](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1) | [0/0](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2) | [0/0](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3) | [0(*)/7](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-0d83d78027c5-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | changed since run |
| [clockfree_audit_measurement_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | FAIL | matches recorded run |
| [five_word_joint_acquisition](k3p.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition) | [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | FAIL | changed since run |
| [gaussian1d_smoke](k3p.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | PASS | changed since run |
| [two_pole](k3p.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | changed since run |
| [unused_token_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](k3p.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [gaussian1d_stability](k3p.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [grid100](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [mid_scale_identity](k3p.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [mode_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [residual_student](k3p.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [rotated100](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [trajectory](k3p.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [unipolar](k3p.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | changed since run |
| [vector_anisotropic](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [ring_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | UNKNOWN | matches recorded run |
| [rotated100_14k](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | [adaptation](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-3), [clockfree_continuous](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](k3p.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |
| [clockfree_audit_measurement_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | required | FAIL | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](k3p.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |
| [grid100_14k](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](k3p.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 7**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 20 / 2**.

Calibration: **provisional**. Revision7 smoke/stability separation is provisional and requires bounded calibration before default adoption. Historical Gaussian acquisition evidence retains its original sigma.025 cohort and five-terminal gate.

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](k3p.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | required | UNKNOWN | recorded definition unavailable |
| [two_pole](k3p.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |
| [ring16_acquisition](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | required | PASS | changed since run |
| [five_word_joint_acquisition](k3p.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition) | required | FAIL | changed since run |
| [clockfree_audit_measurement_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | diagnostic | FAIL | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](k3p.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](k3p.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](k3p.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |
| [img_intensity2_residual16](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | UNKNOWN | matches recorded run |
| [ring_extension](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | UNKNOWN | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](k3p.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | PASS | changed since run |
| [unused_token_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | PASS | changed since run |
| [ae_gan_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | PASS | changed since run |

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](k3p.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | UNKNOWN | changed since run |
| [residual_student](k3p.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | UNKNOWN | changed since run |
| [unipolar](k3p.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | UNKNOWN | changed since run |
| [cover_leftover](k3p.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | UNKNOWN | changed since run |
| [mid_scale_identity](k3p.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | UNKNOWN | changed since run |
| [mode_hold](k3p.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | UNKNOWN | matches recorded run |
| [vector_two_broad](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | UNKNOWN | matches recorded run |
| [vector_unequal_mass](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | UNKNOWN | matches recorded run |
| [vector_unequal_width](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | UNKNOWN | matches recorded run |
| [vector_anisotropic](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | UNKNOWN | matches recorded run |
| [vector_overlap](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | UNKNOWN | matches recorded run |
| [vector_spiral](k3p.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | UNKNOWN | matches recorded run |
| [img_stripes2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | UNKNOWN | matches recorded run |
| [img_bars4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | UNKNOWN | matches recorded run |
| [img_blobs4](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | UNKNOWN | matches recorded run |
| [img_intensity2](k3p.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | UNKNOWN | matches recorded run |
| [grid100](k3p.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | UNKNOWN | matches recorded run |
| [rotated100](k3p.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | UNKNOWN | matches recorded run |
| [staggered100](k3p.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | UNKNOWN | matches recorded run |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](k3p.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

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

Used by: [adaptation / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| hold | 0.00350718 | <= 0.35 | PASS |
| recon_mse | 0.00443183 | <= 0.05 | PASS |

Recorded terminal passing observations: **21**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/b2e5ce07510342a3ae9b24f0e21449bb.json)

[Actual-training GIF](../tier1-completion/media/k3p/b2e5ce07510342a3ae9b24f0e21449bb/ae_gan_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: FAIL**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. step_label changed the update or common-prefix state

Actual task device: `0` (recorded execution receipt).

Used by: [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/eb7b95dbf5b34f69a0c41e97626df6cb.json)

[Actual-training GIF](../tier1-completion/media/k3p/eb7b95dbf5b34f69a0c41e97626df6cb/clockfree_audit_measurement_v1.gif); 4 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: UNKNOWN**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. no compatible result

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| mass_tv | 0.4 | <= 0.1 | FAIL |
| minimum_reconstruction_token_probability | 0 | >= 0.9 | FAIL |
| modes | 3 | == 5 | FAIL |
| quality_fraction | 1 | >= 0.95 | PASS |
| reconstruction_exact | 0 | == 1 | FAIL |
| sample_count | 1024 | >= 1024 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/2031cd7bb3a04d4090bdfbca8eaac088.json)

[Actual-training GIF](../tier1-completion/media/k3p/2031cd7bb3a04d4090bdfbca8eaac088/five_word_joint_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

<a name="cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1"></a>

### gaussian1d_acquisition_tier1_policy_selected_cloud_v1

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: UNKNOWN**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **recorded definition unavailable**. Recorded test definition unavailable; compatibility with today's declaration cannot be checked. No recorded result for this selected configuration and source.

Execution: **no recorded execution (*)**.

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [discriminator_stability / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

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

Used by: [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `1` (recorded execution receipt).

Used by: [discriminator_stability / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 0.621647 | <= 0.85 | PASS |
| component_min_eigen_ratio | 0.323789 | >= 0.15 | PASS |
| hq | 0.937256 | >= 0.85 | PASS |
| mass_tv | 0.0810547 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **5**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/d6d907c833524c1ea23374d90c548cf6.json)

[Actual-training GIF](../tier1-completion/media/k3p/d6d907c833524c1ea23374d90c548cf6/ring16_acquisition.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) · [clockfree_continuous / Tier 3](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `cpu` (recorded execution receipt).

Used by: [adaptation / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.185478 | <= 1 | PASS |
| mean_abs | 0.434736 | >= 0.3 | PASS |

Recorded terminal passing observations: **7**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/cecbce1a5db74a7bb9bc709231515998.json)

[Actual-training GIF](../tier1-completion/media/k3p/cecbce1a5db74a7bb9bc709231515998/two_pole.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [adaptation / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| concept_move | 0.994272 | >= 0.85 | PASS |
| unused_hold | 0.999029 | >= 0.85 | PASS |

Recorded terminal passing observations: **15**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/20d3eadff3724e6db327613becd577e7.json)

[Actual-training GIF](../tier1-completion/media/k3p/20d3eadff3724e6db327613becd577e7/unused_token_hold.gif); 24 saved observations; no new optimizer updates or sampling draws.

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

Used by: [tier1_policy_coverage / Tier 1](k3p.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](k3p.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

- [Standalone API evidence: k3p · 1-D Gaussian: histogram matching](../../toy_audit/api_contract/gaussian1d/README.md) · [actual-training GIF](../../toy_audit/api_contract/gaussian1d/goal.gif)
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
