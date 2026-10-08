<!-- Generated Forge family report -->

# Atlas

[← Family leaderboard](../technique-inventory.md)

**Tags:** [adaptive-training-policy](../technique-inventory.md#tag-adaptive-training-policy) · [adversarial-training](../technique-inventory.md#tag-adversarial-training) · [critic-gradient-penalty](../technique-inventory.md#tag-critic-gradient-penalty) · [optimizer-interventions](../technique-inventory.md#tag-optimizer-interventions)

## Technique overview

Atlas is the E22 independent-row policy with two added controls: automatic selection of a bounded critic-feature birth/death backend and a settled-state guard on optimizer-surprise reopening. It retains E22's KA2 critic, adaptive per-group rates, row evidence, learned output noise and state-selected serving. Its formulation is not a new adversarial loss.

## Mathematical formulation

$G$ is the generator, $D$ the raw-score critic, $P$ the task prior, $x$ a real batch, $z\sim P$ a latent batch and $y$ the generated batch, including any declared output noise. $C$ is the critic actually evaluated: $C=D$ without input noise; otherwise every forward evaluates $D(u+\epsilon_{\mathrm{in}})$ with a fresh draw. $\mathbb E$ means a sample mean, $\lVert\cdot\rVert_2$ is the Euclidean norm, $(a)_+=\max(a,0)$ and $\operatorname{softplus}(a)=\log(1+e^a)$. The scalar sketch detaches $y$ during the critic update and resamples differentiable $y$ after that update; joint and auxiliary hosts retain task-owned objectives. $g_r=\nabla_x C(x)$ and $g_f=\nabla_y C(y)$ are per-sample input gradients; $d$ counts critic-input coordinates. $\lambda$ is penalty strength and $\kappa$ the cap threshold. $\bar C$ evaluates the critic parameter EMA with the same fresh-input-noise law as $C$. $A$ is the early RMS-scaled penalty, $B_{\mathrm{cap}}$ the late L2 cap term and $P_{\mathrm{anchor}}$ the dimension-normalized gradient proximity. $w$ is the configured anchor weight. $n$ counts applied penalty calls, $W\in\{0,1\}$ releases or restores the anchor, and $\alpha\in[0,1]$ is the moment-surprise tracking state. $\delta$ is critic EMA decay and $\theta$ the live critic parameters. Moment surprise compares critic gradient RMS with completed Adam second-moment RMS. $j$ indexes optimizer groups, $\eta_j^{(0)}$ is a group's base LR, $s_j$ its adaptive multiplier and $b$ the stationarity block size in intrinsic time. Intrinsic time accumulates applied LR divided by base LR. Game trust is policy confidence derived from optimizer surprise; drift means statistically detected change in update behavior. KA2 anchors critic input gradients; A2 separately dampens sparse latent-row responses. $h_i$ is latent row $i$'s current gradient, $h_i^{\mathrm{prev}}$ its last observed gradient, $\rho_i$ the A2 response multiplier and $\Delta z_i$ the corresponding row update. $\operatorname{cos}$ denotes the implementation's bounded cosine similarity, including its handling of zero-length history.

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

The first $799$ applied penalty calls use $A$ alone. Starting at call $800$, the equal blend operates even at constant LR. Moment-surprise ratios release $W$ above $3$ and restore it below $1.75$ once reference history exists. Lazy skips do not increment $n$; an applied lazy penalty includes its cadence multiplier. The independent-row preset uses $\lambda=3$, $\kappa=1$, $w=1$; game trust and drift evidence also modulate anchor release/tracking.

**Stationarity learning-rate ladder**

$$
\eta_j=\eta_j^{(0)}s_j,\qquad s_j^{\mathrm{next}}=\begin{cases}s_j/2,&\text{stationary},\\\min(1,2s_j),&\text{drift},\\s_j,&\text{no accepted rate change}.\end{cases}
$$

This is the stationarity ladder before policy corrections. Displacement tests use scales $b$ and $2b$; mixed or inconclusive evidence enlarges the observation window. Row-evidence holds, critic payoff damping and its relative table-rate floor can modify the applied rate. Optimizer surprise can reopen a ladder; Atlas additionally requires its settled-state guard.

## Simplified pseudocode

```text
For each training iteration:
  Select a supported feature backend automatically; the Atlas preset allows 128 feature cells.
  Restore the fast training state if the previous inference state used averages.
  Set stationarity LRs; permit optimizer-surprise reopen only when the settled guard admits the excursion.
  Damp only the critic's rate using payoff error; retain the policy's floor relative to table motion.
  Draw real x and prior z; generate detached fake with the learned output-noise law.
  Compute the paired critic loss and add KA2 regularization shown above.
    g_r and g_f are gradients of the critic with respect to its detached real/fake inputs.
    Compute the early RMS-scaled real-gradient/fake-cap term A shown above.
    Compute the late unnormalized real/fake cap term B_caps shown above.
    Measure real-input gradient proximity to the critic average; use zero when the anchor first starts.
    Use the early-only KA2 penalty for calls 1..799, then the equal early/late blend shown above.
    Its gradient anchor release/tracking also reads the policy's game-trust and drift evidence.
  Backpropagate; apply the Adam-state critic spike guard, take its Adam/AMSGrad step, record surprise.
  Freeze D parameters; draw fresh latent rows, regenerate fake and recompute both scores.
  Compute the paired generator adversarial loss from the updated critic; backpropagate through G, latent rows and learned log output-noise sigma.
  Observe row gradients. A2 applies when some rows have zero gradient, cumulative observed fraction < .5, and Adam state exists:
    Scale observed-row Adam responses using the A2 cosine-agreement rule; leave rows without history unchanged.
  Apply the policy corrections to statistically flagged hot rows and exclude those rows from settling tests.
  Update G/prior; record applied displacements for the next stationarity decision.
  Update G/prior averages over the table tester's declared serving window.
  Use support evidence to clone/replace eligible particle rows; copy optimizer/history and rebase affected tests.
  Serve the averaged state only when the policy's settlement condition allows it; otherwise serve the fast state.
  Preserve the declared served sampling noise and state-selection identity in evaluation.
```

## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Paired relativistic logistic with KA2 input-gradient regularization. Scalar GANTrainer jointly learns the generated-output noise scale through the generator loss. Caller-owned joint/conditional objectives must explicitly bind their roles and guard contexts. |
| Optimizer | KA2 critic Adam and shared K3P generator/prior Adam, with AMSGrad enabled and $\beta_1=0$, $\beta_2=0.999$ in the independent-row preset. AMSGrad keeps a running maximum second moment; the policy also owns per-group stationarity, row evidence and restart state. |
| Learning rates and annealing | Base LR $0.00425$, critic multiplier $1$, latent-prior multiplier $2$. No total training horizon drives the LR: per-group two-scale displacement tests halve settled rates, restore rates on drift and enlarge inconclusive observation windows. Optimizer surprise can reopen the ladders and restart generator moments. The critic also receives payoff-error damping; external update/time limits are budgets, not LR schedules. |
| Parameter-gradient clipping | The critic spike guard remains: after $200$ Adam steps, clip each tensor relative to $5$ times its bias-corrected second-moment RMS. With AMSGrad it uses the running maximum denominator that Adam will apply. Row-evidence corrections and A2 modify other responses separately; there is no added fixed global norm clip. |
| Critic penalties and anchors | KA2 $\lambda=3$ and $\kappa=1$, using early real R1/fake RMS caps followed by an equal early/late blend and adaptive critic-gradient anchoring. The applied-call warmup at $800$ remains, even though the LR controller has no fixed horizon. The policy modulates anchor release and EMA tracking using its own evidence. |
| Damping and update guards | A2 dampens eligible sparse latent-row Adam responses by gradient agreement, with scale $0.75+0.25\operatorname{cos}(h_i,h_i^{\mathrm{prev}})$ in $[0.5,1]$. It activates only when some rows have zero gradient, cumulative observed-row fraction is below $0.5$, and Adam state exists; missing row history gives scale $1$. RowEvidence identifies unusually inconsistent transport, can hold table LR descent, excludes flagged rows from settling tests and corrects hot-row motion. Birth/death uses real-support evidence in standardized critic features and copies/rebases state after a move. These controls require the independent-row policy contract. Atlas adds reopen_guard='settled': an optimizer-surprise excursion must begin after network contraction, with known KA2 loss epochs rebased. Automatic feature selection may choose a feature-cell or kNN path; the chosen backend and its sampling law are recorded. |
| Training and sampling noise | Critic-input noise is zero. Generated-output noise is learnable, initialized at standard deviation $0.029$, with a policy settlement/mobility-dependent floor. DV12 also perturbs latent rows from support statistics. These sampled laws are part of served evidence; a clean diagnostic is a different cohort. |
| Parameter averaging and serving | serve_average=$4$ maintains generator/prior averages over four table-tester blocks. The update rate is table LR scale / ($4$ times block size), with the $0.995$ EMA fallback when no tester rate is available. Serving selects fast or averaged state according to settlement evidence; the critic's adaptive gradient anchor is a separate average. A selected feature-cell backend has its own paired-average eligibility check; the recorded backend decides when average serving is valid. |

## Configuration differences

- Each result retains its executed source, recipe, task prior, initialization, budget and sampling law. These descriptions do not change or requalify recorded measurements.
- Task-owned objectives and active components matter: direct sample particles, learned latent rows and a generator network are different parameter roles. A declared recipe switch does not imply that every host can apply it.
- Selected Atlas is a historical incumbent at source 928b485ffbe6e17d79b41d307ae8b6275489b37a, with explicit independent particle-cloud and served/state-selected identity. Original Atlas passes and clean diagnostics retain separate sampling cohorts.
- Atlas differs from E22 through birth_death_backend='auto', birth_death_cells=128 and reopen_guard='settled'. Automatic selection is capability/evidence constrained; 128 is a preset cell count, not a promise that every host uses feature cells.
- The settled reopen guard suppresses acquisition-time and known-loss-epoch excursions from looking like a moved target. It does not remove KA2's applied-call warmup or declare the whole trainer clock-free.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [configs/forge/ideas/atlas.json](../../../configs/forge/ideas/atlas.json)
- [particlegan/recipes.py](../../../particlegan/recipes.py)
- [particlegan/policy.py](../../../particlegan/policy.py)
- [particlegan/continuous.py](../../../particlegan/continuous.py)
- [particlegan/feature_policy.py](../../../particlegan/feature_policy.py)
- [particlegan/birth_death.py](../../../particlegan/birth_death.py)
- [particlegan/row_evidence.py](../../../particlegan/row_evidence.py)
- [particlegan/ka2.py](../../../particlegan/ka2.py)
- [reports/forge/family-winner-round1/E22_POLICY_READOUT.md](../family-winner-round1/E22_POLICY_READOUT.md)
- [https://github.com/255BITS/ParticleGAN/blob/928b485ffbe6e17d79b41d307ae8b6275489b37a/particlegan/recipes.py](https://github.com/255BITS/ParticleGAN/blob/928b485ffbe6e17d79b41d307ae8b6275489b37a/particlegan/recipes.py)

</details>

Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific contracts; grouping them under current views grants no new qualification.

**Atlas measured evidence — separate configurations and contracts.** [Retained live-clean common-26 diagnostic: HALTED_INCOMPLETE](../common26-full-original-diagnostic-20261005/README.md) · [Restored native Atlas — raw particles, public selected/noisy: grid100 PASS, rotated100 PASS, staggered100 PASS](../atlas-native-restoration-checkpoint-20261005/README.md) · [Repaired Atlas AE — auxiliary MoG, live scheduled-noise: ae_gan_hold PASS](../atlas-ae-sourceguard-checkpoint-20261005/README.md) · [Atlas inventory gaps and bounded next steps](../atlas-inventory-next-steps-20261005.md). These records do not add cells to the selected Atlas row or grant prerequisite credit, default adoption or speed ranking.

**Archived first Full Atlas case: two_pole FAIL.** mean_abs 0.00244565 >= 0.3 (FAIL); grad_med 0.010897 <= 1 (PASS). This accepted first case completed 80 updates and 24 ordinary live observations at seed 0. [Archived first-case result and goal GIF](../common26-first-two-pole-full-atlas-20261004/README.md) · [Pinned result, full Recipe and source](../common26-first-two-pole-full-atlas-20261004/results.json). Its first-case status does not describe the later diagnostic scopes or the canonical Atlas configuration selected in the recorded table. No selected-table cells, prerequisite credit, default adoption or speed ranking are awarded.

<a name="cohort-cuda-1bf9d7d34422"></a>

## Current benchmark

Runtime: **cuda**. Selected configuration: [atlas](../../../configs/forge/ideas/atlas.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `6269a18ac4f82564cb16ba19afa4b3dd2f836a2b4085fbe5aeb81a35a453e895`. Candidate revision: `674f43fbb22ae3df45c666d3290ec8ad40e990d0883c036729d8e56ad097752b`. Runtime cohort: `e84c5e7910559b15078d38f8f3320f1e02605b16a365f6883beaa060224129ac`.

[Frozen numerical evidence](../technique-evidence/7c4e188f1ee495526c2decf6f02a45f392123aaf4f33f1f6fb9a398fe3ac05e7.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Current-source pre-run whole candidate choice is partially BLOCKED or unmeasured; retain exact display evidence with no complete current-measurement, qualification or default-adoption credit.

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation) | [0(*)/3](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [0(*)/23](atlas.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0(*)/4](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [0(*)/30](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0(*)/6](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [0(*)/29](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0(*)/3](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [0(*)/24](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0(*)/3](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [0(*)/24](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0(*)/3](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [0(*)/22](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) | [0(*)/7](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1) | [0/0](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-2) | [0/0](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-3) | [0(*)/7](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-1bf9d7d34422-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | BLOCKED | matches recorded run |
| [clockfree_audit_measurement_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | BLOCKED | matches recorded run |
| [five_word_joint_smoke](atlas.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](atlas.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | BLOCKED | matches recorded run |
| [ring16_acquisition](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | BLOCKED | matches recorded run |
| [two_pole](atlas.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](atlas.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [five_word_joint_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](atlas.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [trajectory](atlas.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](atlas.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | BLOCKED | matches recorded run |
| [ring_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3), [discriminator_stability](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3), [formulation_comparison](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3), [host_profile_transfer](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | BLOCKED | matches recorded run |
| [rotated100_14k](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](atlas.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | [adaptation](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3), [clockfree_continuous](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](atlas.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [two_pole](atlas.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |
| [clockfree_audit_measurement_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](atlas.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | BLOCKED | matches recorded run |
| [grid100_14k](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](atlas.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 8**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](atlas.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_smoke) | required | BLOCKED | matches recorded run |
| [two_pole](atlas.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |
| [ring16_acquisition](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) | required | BLOCKED | matches recorded run |
| [five_word_joint_smoke](atlas.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) | diagnostic | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](atlas.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) | required | BLOCKED | matches recorded run |
| [five_word_joint_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](atlas.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | BLOCKED | matches recorded run |
| [img_intensity2_residual16](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | BLOCKED | matches recorded run |
| [img_intensity2_residual16](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](atlas.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-1bf9d7d34422-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-1bf9d7d34422-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-1bf9d7d34422-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-1bf9d7d34422-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-1bf9d7d34422-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) | required | BLOCKED | matches recorded run |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

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

**ae_gan_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ae_gan_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. clockfree_audit_measurement_v1: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-1bf9d7d34422-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cover_leftover: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

Used by: [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2).

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

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

**gaussian1d_smoke: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. gaussian1d_smoke: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

**gaussian1d_stability: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. gaussian1d_stability: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2).

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

**grid100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. grid100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2).

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

**img_bars4: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_bars4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**img_blobs4: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_blobs4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**img_intensity2: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_intensity2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**img_stripes2: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_stripes2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**mid_scale_identity: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. mid_scale_identity: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

**mode_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. mode_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

**residual_student: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. residual_student: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

**ring16_acquisition: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring16_acquisition: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

**ring_extension: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring_extension: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3).

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

**ring_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring_hold: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3).

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

**rotated100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. rotated100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**staggered100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. staggered100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) · [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-1bf9d7d34422-experiment-trajectory"></a>

### trajectory

**trajectory: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. trajectory: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

**two_pole: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

**unipolar: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. unipolar: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

**unused_token_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. unused_token_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-1bf9d7d34422-tier1_policy_coverage-tier-1).

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

**vector_anisotropic: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_anisotropic: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**vector_overlap: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_overlap: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**vector_spiral: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_spiral: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**vector_two_broad: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_two_broad: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**vector_unequal_mass: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_unequal_mass: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

**vector_unequal_width: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_unequal_width: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2).

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

## Archived runtime cohort

Runtime: **cuda**. Selected configuration: [atlas](../../../configs/forge/ideas/atlas.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `d276c5a7344fab6ec5de7b314d3982af5b0ef8027c8b89c01376ae366844a9cb`. Candidate revision: `e6e54eac004e29f826627be2410bf64b59475443340387f6d0deee667178bb6c`. Runtime cohort: `8dc147191bfc2772cb2f23970506a979ad355b7801d69db357b50b1863f525fa`.

[Frozen numerical evidence](../technique-evidence/83f549889bf36b44bda98c8fe4c0f1600ef0cdd1d12c529f9e480ffcfdc93512.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Current-source pre-run whole candidate choice is partially BLOCKED or unmeasured; retain exact display evidence with no complete current-measurement, qualification or default-adoption credit.

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation) | [0(*)/3](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-1) | [0(*)/19](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) | [0(*)/1](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-3) | [0(*)/23](atlas.md#cohort-cuda-c195899a64af-adaptation) |
| [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous) | [0(*)/4](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) | [0(*)/19](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) | [0(*)/7](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | [0(*)/30](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous) |
| [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability) | [0(*)/6](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | [0(*)/21](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | [0(*)/2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) | [0(*)/29](atlas.md#cohort-cuda-c195899a64af-discriminator_stability) |
| [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison) | [0(*)/3](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) | [0(*)/19](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) | [0(*)/2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) | [0(*)/24](atlas.md#cohort-cuda-c195899a64af-formulation_comparison) |
| [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer) | [0(*)/3](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) | [0(*)/19](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) | [0(*)/2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | [0(*)/24](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer) |
| [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage) | [0(*)/3](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | [0(*)/19](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | [0/0](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-3) | [0(*)/22](atlas.md#cohort-cuda-c195899a64af-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage) | [0(*)/7](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1) | [0/0](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-2) | [0/0](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-3) | [0(*)/7](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage) |

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-c195899a64af-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | BLOCKED | matches recorded run |
| [clockfree_audit_measurement_v1](atlas.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) | BLOCKED | matches recorded run |
| [five_word_joint_smoke](atlas.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](atlas.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | BLOCKED | matches recorded run |
| [ring16_acquisition](atlas.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) | BLOCKED | matches recorded run |
| [two_pole](atlas.md#cohort-cuda-c195899a64af-experiment-two_pole) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-1) | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](atlas.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [five_word_joint_hold](atlas.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](atlas.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-c195899a64af-experiment-grid100) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-c195899a64af-experiment-img_bars4) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-c195899a64af-experiment-mode_hold) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-c195899a64af-experiment-residual_student) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [trajectory](atlas.md#cohort-cuda-c195899a64af-experiment-trajectory) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-c195899a64af-experiment-unipolar) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2) | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](atlas.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](atlas.md#cohort-cuda-c195899a64af-experiment-grid100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](atlas.md#cohort-cuda-c195899a64af-experiment-ring_extension) | [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | BLOCKED | matches recorded run |
| [ring_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ring_hold) | [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3), [discriminator_stability](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3), [formulation_comparison](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3), [host_profile_transfer](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3) | BLOCKED | matches recorded run |
| [rotated100_14k](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](atlas.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | [adaptation](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-3), [clockfree_continuous](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-c195899a64af-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-c195899a64af-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](atlas.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [two_pole](atlas.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |
| [clockfree_audit_measurement_v1](atlas.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-c195899a64af-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](atlas.md#cohort-cuda-c195899a64af-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | BLOCKED | matches recorded run |
| [grid100_14k](atlas.md#cohort-cuda-c195899a64af-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](atlas.md#cohort-cuda-c195899a64af-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 8**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](atlas.md#cohort-cuda-c195899a64af-experiment-gaussian1d_smoke) | required | BLOCKED | matches recorded run |
| [two_pole](atlas.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |
| [ring16_acquisition](atlas.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition) | required | BLOCKED | matches recorded run |
| [five_word_joint_smoke](atlas.md#cohort-cuda-c195899a64af-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](atlas.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1) | diagnostic | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](atlas.md#cohort-cuda-c195899a64af-experiment-gaussian1d_stability) | required | BLOCKED | matches recorded run |
| [five_word_joint_hold](atlas.md#cohort-cuda-c195899a64af-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](atlas.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-c195899a64af-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-c195899a64af-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | BLOCKED | matches recorded run |
| [img_intensity2_residual16](atlas.md#cohort-cuda-c195899a64af-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](atlas.md#cohort-cuda-c195899a64af-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](atlas.md#cohort-cuda-c195899a64af-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](atlas.md#cohort-cuda-c195899a64af-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](atlas.md#cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](atlas.md#cohort-cuda-c195899a64af-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](atlas.md#cohort-cuda-c195899a64af-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-c195899a64af-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | BLOCKED | matches recorded run |
| [img_intensity2_residual16](atlas.md#cohort-cuda-c195899a64af-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](atlas.md#cohort-cuda-c195899a64af-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](atlas.md#cohort-cuda-c195899a64af-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](atlas.md#cohort-cuda-c195899a64af-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](atlas.md#cohort-cuda-c195899a64af-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](atlas.md#cohort-cuda-c195899a64af-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-c195899a64af-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-c195899a64af-experiment-ring_extension) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](atlas.md#cohort-cuda-c195899a64af-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](atlas.md#cohort-cuda-c195899a64af-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](atlas.md#cohort-cuda-c195899a64af-experiment-two_pole) | required | BLOCKED | matches recorded run |
| [unused_token_hold](atlas.md#cohort-cuda-c195899a64af-experiment-unused_token_hold) | required | BLOCKED | matches recorded run |
| [ae_gan_hold](atlas.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-c195899a64af-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-c195899a64af-experiment-trajectory) | required | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-c195899a64af-experiment-residual_student) | required | BLOCKED | matches recorded run |
| [unipolar](atlas.md#cohort-cuda-c195899a64af-experiment-unipolar) | required | BLOCKED | matches recorded run |
| [cover_leftover](atlas.md#cohort-cuda-c195899a64af-experiment-cover_leftover) | required | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-c195899a64af-experiment-mid_scale_identity) | required | BLOCKED | matches recorded run |
| [mode_hold](atlas.md#cohort-cuda-c195899a64af-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-c195899a64af-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-c195899a64af-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-c195899a64af-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-c195899a64af-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-c195899a64af-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-c195899a64af-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-c195899a64af-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-c195899a64af-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-c195899a64af-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-c195899a64af-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-c195899a64af-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-c195899a64af-experiment-staggered100) | required | BLOCKED | matches recorded run |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-c195899a64af-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [two_pole_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-c195899a64af-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [unused_token_hold_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-c195899a64af-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-c195899a64af-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-c195899a64af-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-c195899a64af-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-c195899a64af-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | UNKNOWN | recorded definition unavailable |

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

**ae_gan_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ae_gan_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. clockfree_audit_measurement_v1: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-c195899a64af-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. cover_leftover: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

Used by: [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2).

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

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

**gaussian1d_smoke: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_smoke.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. gaussian1d_smoke: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

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

**gaussian1d_stability: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/gaussian1d_stability.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. gaussian1d_stability: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2).

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

**grid100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. grid100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2).

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

**img_bars4: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_bars4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

**img_blobs4: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_blobs4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

**img_intensity2: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_intensity2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

**img_stripes2: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_stripes2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

**mid_scale_identity: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. mid_scale_identity: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

**mode_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. mode_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

**residual_student: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. residual_student: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

**ring16_acquisition: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring16_acquisition: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

**ring_extension: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring_extension: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3).

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

**ring_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring_hold: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-3).

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

**rotated100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. rotated100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

**staggered100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. staggered100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-3) · [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-c195899a64af-experiment-trajectory"></a>

### trajectory

**trajectory: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. trajectory: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

**two_pole: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](atlas.md#cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](atlas.md#cohort-cuda-c195899a64af-k3p_two_pole_horizon-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

**unipolar: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. unipolar: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

**unused_token_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. unused_token_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-1).

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

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-c195899a64af-tier1_policy_coverage-tier-1).

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

**vector_anisotropic: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_anisotropic: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

**vector_overlap: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_overlap: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

**vector_spiral: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_spiral: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

**vector_two_broad: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_two_broad: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

**vector_unequal_mass: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_unequal_mass: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

**vector_unequal_width: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_unequal_width: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-c195899a64af-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-c195899a64af-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-c195899a64af-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-c195899a64af-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-c195899a64af-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-c195899a64af-host_profile_transfer-tier-2).

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

## Archived runtime cohort

Runtime: **cuda**. Selected configuration: [atlas](../../../configs/forge/ideas/atlas.json).

Recorded qualification: **tier 0**, discriminator_stability revision 7. Other view rows below are navigation over recorded task evidence, not recomputed qualification.

<details>
<summary>Configuration, source and runtime provenance</summary>

Source digest: `21ec7e3f89402b5d0c77669d5fba0e36bcf4e690069cfdceab383520006c525f`. Candidate revision: `2f2806b54846656bc87e008e616f8fc7eae9b1f24e080f4eeaaf88bf3fb4b098`. Runtime cohort: `01b933360e24e554b4965d036ad47f6b011d8bd6cdb15d7f6d56d8822192b6aa`.

[Frozen numerical evidence](../technique-evidence/ddde64ee936114becac42863847a51e88ec0301bad9f7f69767d0a2fbc3f3d69.json) · [Complete recipe, prior, initialization and sampling bindings](../technique-inventory.json)

Selection: historical_incumbent. Exact executed-source parent cohort remains blocked. Separate selected-policy/cloud measurements grant no parent clean qualification.

</details>

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation) | [0(*)/3](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) | [0(*)/19](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) | [0(*)/1](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) | [0(*)/23](atlas.md#cohort-cuda-0d83d78027c5-adaptation) |
| [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous) | [0(*)/4](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | [0(*)/19](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) | [0(*)/7](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | [0(*)/30](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous) |
| [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability) | [0(*)/6](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | [0(*)/21](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | [0(*)/2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) | [0(*)/29](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability) |
| [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison) | [0(*)/3](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) | [0(*)/19](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) | [0(*)/2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) | [0(*)/24](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison) |
| [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer) | [0(*)/3](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) | [0(*)/19](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) | [0(*)/2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | [0(*)/24](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer) |
| [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage) | [0(*)/3](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | [0(*)/19](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | [0/0](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-3) | [0(*)/22](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage) |

Separate cohort coverage (excluded from family totals):

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| [tier1_policy_coverage](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) | [0(*)/7](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1) | [0/0](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-2) | [0/0](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-3) | [0(*)/7](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage) |

[Frozen separate-cohort numerical evidence](../tier1-completion/scoped-evidence.json); source commit `928b485ffbe6e17d79b41d307ae8b6275489b37a`; exact scoped cohort `df620730e3d310e590a2841155c9a78fb3cacbb57a97be2e4f9a3799ff0e243c` (runtime SHA256 `957e45749c6edd39d55ad206e3911d627884a6e8d49bfb3d3b803093cbf7eaf7`). These cells give no parent-cohort credit.

(*) means at least one required experiment has no recorded execution, including preflight blockers. PASS and FAIL both count as executed. Attempted errors retain their status and cause; test-definition compatibility is shown separately and does not add (*).

<a name="cohort-cuda-0d83d78027c5-tier-1"></a>

### Tier 1 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ae_gan_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | BLOCKED | changed since run |
| [clockfree_audit_measurement_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) | BLOCKED | matches recorded run |
| [five_word_joint_smoke](atlas.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_smoke](atlas.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | UNKNOWN | recorded definition unavailable |
| [ring16_acquisition](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) | BLOCKED | changed since run |
| [two_pole](atlas.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | BLOCKED | changed since run |
| [unused_token_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-1), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1) | BLOCKED | changed since run |

<a name="cohort-cuda-0d83d78027c5-tier-2"></a>

### Tier 2 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [cover_leftover](atlas.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | changed since run |
| [five_word_joint_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [gaussian1d_stability](atlas.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) | UNKNOWN | recorded definition unavailable |
| [grid100](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [mid_scale_identity](atlas.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | changed since run |
| [mode_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [residual_student](atlas.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | changed since run |
| [rotated100](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [trajectory](atlas.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | changed since run |
| [unipolar](atlas.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | changed since run |
| [vector_anisotropic](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2), [quality_coverage](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2) | BLOCKED | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-tier-3"></a>

### Tier 3 across views

Shared experiments appear once in this list; the family numerator/denominator count their view requirements.

| Experiment | Required by | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](atlas.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [grid100_14k](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [ring_extension](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | BLOCKED | matches recorded run |
| [ring_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3), [discriminator_stability](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3), [formulation_comparison](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3), [host_profile_transfer](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3) | BLOCKED | matches recorded run |
| [rotated100_14k](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](atlas.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | [adaptation](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-3), [clockfree_continuous](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-adaptation"></a>

## adaptation

**adaptation — revision 2**. [View declaration](../../../configs/forge/views/adaptation.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 1**.

Calibration: **provisional**. Phase D historical calibration remains required

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | BLOCKED | changed since run |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](atlas.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](atlas.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](atlas.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](atlas.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-adaptation-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [target_shift_recovery](atlas.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

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
| [two_pole](atlas.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | BLOCKED | changed since run |
| [clockfree_audit_measurement_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](atlas.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](atlas.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](atlas.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](atlas.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [clockfree_audit](atlas.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit) | required | UNKNOWN | recorded definition unavailable |
| [ring_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | BLOCKED | matches recorded run |
| [grid100_14k](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100_14k) | required | UNKNOWN | recorded definition unavailable |
| [rotated100_14k](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100_14k) | required | UNKNOWN | recorded definition unavailable |
| [staggered100_14k](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100_14k) | required | UNKNOWN | recorded definition unavailable |
| [target_shift_recovery](atlas.md#cohort-cuda-0d83d78027c5-experiment-target_shift_recovery) | required | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability"></a>

## discriminator_stability

**discriminator_stability — revision 8**. [View declaration](../../../configs/forge/views/discriminator_stability.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **6 / 21 / 2**.

Calibration: **provisional**. Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_smoke](atlas.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_smoke) | required | UNKNOWN | recorded definition unavailable |
| [two_pole](atlas.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | BLOCKED | changed since run |
| [ring16_acquisition](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition) | required | BLOCKED | changed since run |
| [five_word_joint_smoke](atlas.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_smoke) | required | UNKNOWN | recorded definition unavailable |
| [clockfree_audit_measurement_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1) | diagnostic | BLOCKED | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [gaussian1d_stability](atlas.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_stability) | required | UNKNOWN | recorded definition unavailable |
| [five_word_joint_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_hold) | required | UNKNOWN | recorded definition unavailable |
| [trajectory](atlas.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](atlas.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](atlas.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](atlas.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](atlas.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-discriminator_stability-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison"></a>

## formulation_comparison

**formulation_comparison — revision 1**. [View declaration](../../../configs/forge/views/formulation_comparison.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | BLOCKED | changed since run |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](atlas.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](atlas.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](atlas.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](atlas.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | BLOCKED | matches recorded run |
| [img_intensity2_residual16](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_paired_laws_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_paired_laws_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_release07_cloud_named_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100_release07_cloud_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-formulation_comparison-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer"></a>

## host_profile_transfer

**host_profile_transfer — revision 4**. [View declaration](../../../configs/forge/views/host_profile_transfer.json).

Qualification requires every required experiment to pass, with all lower tiers and task dependencies passed first. Required counts (Tier 1 / 2 / 3): **3 / 19 / 2**.

Calibration: **provisional**. Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole](atlas.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | BLOCKED | changed since run |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](atlas.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](atlas.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](atlas.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](atlas.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | BLOCKED | matches recorded run |
| [img_intensity2_residual16](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_two_broad_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_mass_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_unequal_width_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_anisotropic_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_overlap_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [vector_spiral_published](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral_published) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_stripes2_residual16](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_bars4_residual16](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_bars4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [img_blobs4_residual16](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4_residual16) | diagnostic | UNKNOWN | recorded definition unavailable |
| [grid100_affine_square_named_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [rotated100_affine_square_named_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [staggered100_affine_square_named_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100_affine_square_named_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

<a name="cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3"></a>

### Tier 3

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [ring_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring_hold) | required | BLOCKED | matches recorded run |
| [ring_extension](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring_extension) | required | BLOCKED | matches recorded run |

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon"></a>

## k3p_two_pole_horizon

**k3p_two_pole_horizon — revision 1**. [View declaration](../../../configs/forge/views/k3p_two_pole_horizon.json).

Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.

Calibration: **provisional**. Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

<a name="cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1"></a>

### Tier 1

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule80_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |
| [two_pole_800_schedule800_diagnostic_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-two_pole_800_schedule800_diagnostic_v1) | diagnostic | UNKNOWN | recorded definition unavailable |

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
| [two_pole](atlas.md#cohort-cuda-0d83d78027c5-experiment-two_pole) | required | BLOCKED | changed since run |
| [unused_token_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold) | required | BLOCKED | changed since run |
| [ae_gan_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold) | required | BLOCKED | changed since run |

<a name="cohort-cuda-0d83d78027c5-quality_coverage-tier-2"></a>

### Tier 2

| Experiment | Role | Recorded result | Test definition |
| --- | --- | --- | --- |
| [trajectory](atlas.md#cohort-cuda-0d83d78027c5-experiment-trajectory) | required | BLOCKED | changed since run |
| [residual_student](atlas.md#cohort-cuda-0d83d78027c5-experiment-residual_student) | required | BLOCKED | changed since run |
| [unipolar](atlas.md#cohort-cuda-0d83d78027c5-experiment-unipolar) | required | BLOCKED | changed since run |
| [cover_leftover](atlas.md#cohort-cuda-0d83d78027c5-experiment-cover_leftover) | required | BLOCKED | changed since run |
| [mid_scale_identity](atlas.md#cohort-cuda-0d83d78027c5-experiment-mid_scale_identity) | required | BLOCKED | changed since run |
| [mode_hold](atlas.md#cohort-cuda-0d83d78027c5-experiment-mode_hold) | required | BLOCKED | matches recorded run |
| [vector_two_broad](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_two_broad) | required | BLOCKED | matches recorded run |
| [vector_unequal_mass](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_mass) | required | BLOCKED | matches recorded run |
| [vector_unequal_width](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_unequal_width) | required | BLOCKED | matches recorded run |
| [vector_anisotropic](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_anisotropic) | required | BLOCKED | matches recorded run |
| [vector_overlap](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_overlap) | required | BLOCKED | matches recorded run |
| [vector_spiral](atlas.md#cohort-cuda-0d83d78027c5-experiment-vector_spiral) | required | BLOCKED | matches recorded run |
| [img_stripes2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_stripes2) | required | BLOCKED | matches recorded run |
| [img_bars4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_bars4) | required | BLOCKED | matches recorded run |
| [img_blobs4](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_blobs4) | required | BLOCKED | matches recorded run |
| [img_intensity2](atlas.md#cohort-cuda-0d83d78027c5-experiment-img_intensity2) | required | BLOCKED | matches recorded run |
| [grid100](atlas.md#cohort-cuda-0d83d78027c5-experiment-grid100) | required | BLOCKED | matches recorded run |
| [rotated100](atlas.md#cohort-cuda-0d83d78027c5-experiment-rotated100) | required | BLOCKED | matches recorded run |
| [staggered100](atlas.md#cohort-cuda-0d83d78027c5-experiment-staggered100) | required | BLOCKED | matches recorded run |

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
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-gaussian1d_acquisition_tier1_policy_selected_cloud_v1) | required | FAIL | changed since run |
| [two_pole_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-two_pole_tier1_policy_selected_cloud_v1) | required | FAIL | changed since run |
| [unused_token_hold_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-unused_token_hold_tier1_policy_selected_cloud_v1) | required | BLOCKED | changed since run |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-ae_gan_hold_tier1_policy_selected_cloud_v1) | required | BLOCKED | changed since run |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-ring16_acquisition_tier1_policy_selected_cloud_v1) | required | FAIL | changed since run |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-five_word_joint_acquisition_tier1_policy_selected_cloud_v1) | required | BLOCKED | changed since run |
| [clockfree_audit_tier1_policy_selected_cloud_v1](atlas.md#cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1) | required | FAIL | changed since run |

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

**ae_gan_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ae_gan_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. ae_gan_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

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

**ae_gan_hold_tier1_policy_selected_cloud_v1: BLOCKED**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. ae_gan_hold_tier1_policy_selected_cloud_v1: the frozen posterior/encoder MoG law needs a routed AE formulation; an independent cloud policy is not that law

Execution: **no recorded execution (*)**.

Policy-cohort variant of [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json); parent task SHA256 `53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Recorded conditions: particle_cloud prior (sigma 0); tier1_selected_generated_and_reconstructed_prior_with_scheduled_output_noise; weights state_selected; output noise public_recipe_schedule.

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_measurement_v1"></a>

### clockfree_audit_measurement_v1

**clockfree_audit_measurement_v1: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. clockfree_audit_measurement_v1: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

Recorded conditions: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-clockfree_audit_tier1_policy_selected_cloud_v1"></a>

### clockfree_audit_tier1_policy_selected_cloud_v1

**clockfree_audit_tier1_policy_selected_cloud_v1: FAIL**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. step_label changed the update or common-prefix state

Actual task device: `1` (recorded execution receipt).

Policy-cohort variant of [clockfree_audit](../../../configs/forge/tasks/clockfree_audit.json); parent task SHA256 `d7748d04db85633e5c678622486b94b2a44f0e462ffb9c4b0179216db7840258`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

[Compact metrics and receipt provenance](../technique-receipts/6b44efa5cbac4cce91192d9a3ac33cde.json)

[Actual-training GIF](../tier1-completion/media/atlas/6b44efa5cbac4cce91192d9a3ac33cde/clockfree_audit_tier1_policy_selected_cloud_v1.gif); 4 saved observations; no new optimizer updates or sampling draws.

Recorded clock parity diagnostics:

| Condition | Exact state digest equality |
| --- | --- |
| evaluation_cadence | equal |
| horizon | equal |
| restart | different |
| step_label | different |

Recorded unexplained clock dependencies: **3**.

- continuous policy lifecycle needs a separate reviewed clock/state audit
- KA2 switches from pure A to blended penalty at call 800
- critic guard releases at a fixed minimum update count

[Certified parity digests and source audit](../tier1-completion/scoped-evidence.json). These display diagnostics preserve the recorded gate FAIL.

Recorded conditions: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

Current pass criteria:

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

Declared budget: 24 updates; timeout 300 seconds.

Current measurement: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

<a name="cohort-cuda-0d83d78027c5-experiment-cover_leftover"></a>

### cover_leftover

**cover_leftover: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/cover_leftover.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. cover_leftover: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

**five_word_joint_acquisition_tier1_policy_selected_cloud_v1: BLOCKED**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. five_word_joint_acquisition_tier1_policy_selected_cloud_v1: the frozen five-row joint BiGAN needs complete generator/encoder/policy ownership and a separately scoped minimum-population contract

Execution: **no recorded execution (*)**.

Policy-cohort variant of [five_word_joint_acquisition](../../../configs/forge/tasks/five_word_joint_acquisition.json); parent task SHA256 `26875d18d2d8572a479fe8170894bb8cde0798de38e639514fb077c88344d1f8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Recorded conditions: particle_cloud prior (sigma 0); tier1_selected_generated_and_paired_reconstructed_prior_without_output_noise; weights state_selected; output noise clean.

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

Used by: [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2).

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

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

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

**gaussian1d_acquisition_tier1_policy_selected_cloud_v1: FAIL**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Policy-cohort variant of [gaussian1d_acquisition](../../../configs/forge/tasks/gaussian1d_acquisition.json); parent task SHA256 `b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| cdf_ks | 0.0301488 | <= 0.05 | PASS |
| finite_fraction | 1 | == 1 | PASS |
| mean_error_sigma | 0.0275741 | <= 0.2 | PASS |
| sample_count | 4096 | >= 4096 | PASS |
| std_ratio | 1.0346 | <= 1.2 | PASS |

Recorded terminal passing observations: **3**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/6fc1a81be1ba4664993a0af5b2dec088.json)

[Actual-training GIF](../tier1-completion/media/atlas/6fc1a81be1ba4664993a0af5b2dec088/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.gif); 24 saved observations; no new optimizer updates or sampling draws.

Recorded conditions: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

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

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

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

Used by: [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2).

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

**grid100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/grid100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. grid100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2).

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

**img_bars4: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_bars4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_bars4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

**img_blobs4: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_blobs4.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_blobs4: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

**img_intensity2: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_intensity2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_intensity2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

**img_stripes2: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/img_stripes2.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. img_stripes2: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

**mid_scale_identity: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/mid_scale_identity.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. mid_scale_identity: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

**mode_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/mode_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. mode_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

**residual_student: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/residual_student.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. residual_student: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

**ring16_acquisition: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring16_acquisition.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. ring16_acquisition: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [discriminator_stability / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1).

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

**ring16_acquisition_tier1_policy_selected_cloud_v1: FAIL**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `0` (recorded execution receipt).

Policy-cohort variant of [ring16_acquisition](../../../configs/forge/tasks/ring16_acquisition.json); parent task SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| component_covariance_error | 4.3922 | <= 0.85 | FAIL |
| component_min_eigen_ratio | 1.99837 | >= 0.15 | PASS |
| hq | 0.637695 | >= 0.85 | FAIL |
| mass_tv | 0.0429688 | <= 0.15 | PASS |
| modes | 16 | >= 16 | PASS |
| sample_count | 4096 | >= 4096 | PASS |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/fa5b20a8f092437b9ce53b4573e6ea8b.json)

[Actual-training GIF](../tier1-completion/media/atlas/fa5b20a8f092437b9ce53b4573e6ea8b/ring16_acquisition_tier1_policy_selected_cloud_v1.gif); 24 saved observations; no new optimizer updates or sampling draws.

Recorded conditions: particle_cloud prior (sigma 0); tier1_selected_public_prior_without_output_noise; weights state_selected; output noise clean.

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

**ring_extension: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring_extension.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring_extension: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

**ring_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/ring_hold.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. ring_hold: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3) · [discriminator_stability / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-3) · [formulation_comparison / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-3) · [host_profile_transfer / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-3).

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

**rotated100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/rotated100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. rotated100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

**staggered100: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/staggered100.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. staggered100: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

Used by: [adaptation / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-3) · [clockfree_continuous / Tier 3](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-3).

Current pass criteria:

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

Declared budget: 3600 updates; timeout 3600 seconds.

Current measurement: mog prior (sigma 0.025); public_prior_without_output_noise; weights live; output noise clean.

Dependencies: mode_hold (gate).

<a name="cohort-cuda-0d83d78027c5-experiment-trajectory"></a>

### trajectory

**trajectory: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/trajectory.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. trajectory: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

**two_pole: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/two_pole.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

Used by: [k3p_two_pole_horizon / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-k3p_two_pole_horizon-tier-1).

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

**two_pole_tier1_policy_selected_cloud_v1: FAIL**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. recomputed complete live curve and terminal suffix

Actual task device: `cuda:0` (recorded execution receipt).

Policy-cohort variant of [two_pole](../../../configs/forge/tasks/two_pole.json); parent task SHA256 `55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Recorded final metric checks:

| Metric | Measured | Recorded bound | Recorded check |
| --- | ---: | --- | --- |
| grad_med | 0.0105167 | <= 1 | PASS |
| mean_abs | 0.00258616 | >= 0.3 | FAIL |

Recorded terminal passing observations: **0**; required: 5.

[Compact metrics and receipt provenance](../technique-receipts/e28ab59ac9204a258cc60d431a8d3f3c.json)

[Actual-training GIF](../tier1-completion/media/atlas/e28ab59ac9204a258cc60d431a8d3f3c/two_pole_tier1_policy_selected_cloud_v1.gif); 24 saved observations; no new optimizer updates or sampling draws.

Recorded conditions: particle_cloud prior (sigma 0); tier1_selected_learned_particles_and_critic_gradient; weights state_selected; output noise not_applied_to_measurement.

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

**unipolar: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/unipolar.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. unipolar: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

**unused_token_hold: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/unused_token_hold.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget). Earlier verdict preserved. unused_token_hold: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-1) · [clockfree_continuous / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-1) · [discriminator_stability / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-1) · [formulation_comparison / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-1) · [host_profile_transfer / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-1) · [quality_coverage / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-1).

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

**unused_token_hold_tier1_policy_selected_cloud_v1: BLOCKED**. [Current experiment declaration](../../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Test definition: **changed since run**. Test definition changed since this run: execution (host, recipe binding, prior, initialization or budget); evaluation (gates or sampling law). Earlier verdict preserved. unused_token_hold_tier1_policy_selected_cloud_v1: shared/slot parameters require a conditional RoutedRows mechanism; independent Atlas/E22 cannot claim its protected-context law

Execution: **no recorded execution (*)**.

Policy-cohort variant of [unused_token_hold](../../../configs/forge/tasks/unused_token_hold.json); parent task SHA256 `ef8ccde8d1fa54af8bfce01c044e3671de8131c980eb4e8022d12ffc8caf51d8`. This measurement supplies no cells to the parent clean cohort.

Used by: [tier1_policy_coverage / Tier 1](atlas.md#cohort-cuda-0d83d78027c5-tier1_policy_coverage-tier-1).

Recorded conditions: particle_cloud prior (sigma 0); tier1_selected_learned_parameter_measurement; weights state_selected; output noise not_applied_to_measurement.

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

**vector_anisotropic: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_anisotropic.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_anisotropic: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

**vector_overlap: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_overlap.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_overlap: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

**vector_spiral: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_spiral.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_spiral: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

**vector_two_broad: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_two_broad.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_two_broad: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

**vector_unequal_mass: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_mass.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_unequal_mass: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

**vector_unequal_width: BLOCKED**. [Current experiment declaration](../../../configs/forge/tasks/vector_unequal_width.json).

Test definition: **matches recorded run**. Test definition matches the recorded conditions. vector_unequal_width: clean/live scoring has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation

Execution: **no recorded execution (*)**.

Used by: [adaptation / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-adaptation-tier-2) · [clockfree_continuous / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-clockfree_continuous-tier-2) · [discriminator_stability / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-discriminator_stability-tier-2) · [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2) · [quality_coverage / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-quality_coverage-tier-2).

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

Used by: [formulation_comparison / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-formulation_comparison-tier-2) · [host_profile_transfer / Tier 2](atlas.md#cohort-cuda-0d83d78027c5-host_profile_transfer-tier-2).

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

## References

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). Base optimizer. Guard, damping and controller rules are repository additions.
- [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734). Related paired relativistic adversarial objective.
- [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406). Zero-centered input-gradient regularization; not the full repository formulation.
- [Using Statistics to Automate Stochastic Optimization](https://papers.neurips.cc/paper_files/paper/2019/hash/e1054bf2d703bca1e8fe101d3ac5efcd-Abstract.html). Related statistical learning-rate adaptation. The repository uses its own displacement tests and policy.
