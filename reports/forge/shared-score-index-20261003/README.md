# Shared ParticleGAN score index

One selected configuration per family; full tier denominators stay fixed. Representation is read from its declared prior and task-owned hosts.

| Model/configuration | Representation | Measured scores |
| --- | --- | --- |
| Atlas · [atlas](../../../configs/forge/ideas/atlas.json)<br>cuda | Particles | Tier 1: BLOCKED (5 required)<br>Tier 2: BLOCKED (19 required)<br>Tier 3: BLOCKED (2 required)<br>Recorded tier: 0 |
| BCap · [bcap · 08689a73c551](../../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json)<br>cuda | MoG + particles (per task) | Tier 1: 3/5<br>FAIL 1 · UNKNOWN 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| E22 · [e22](../../../configs/forge/ideas/e22.json)<br>cuda | Particles | Tier 1: BLOCKED (5 required)<br>Tier 2: BLOCKED (19 required)<br>Tier 3: BLOCKED (2 required)<br>Recorded tier: 0 |
| GAN v3 release 0.7 (MoG) · [release07-gan-v3-mog · 1e266b5a2986](../../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json)<br>cuda | MoG + particles (per task) | Tier 1: 3/5<br>FAIL 1 · UNKNOWN 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| GAN v3 release 0.7 (cloud) · [release07-gan-v3-cloud-v1](../../../configs/forge/ideas/release07-gan-v3-cloud-v1.json)<br>cuda | MoG + particles (per task) | Tier 1: BLOCKED (5 required)<br>Tier 2: BLOCKED (19 required)<br>Tier 3: BLOCKED (2 required)<br>Recorded tier: 0 |
| K3P · [k3p · 0b37e98a01e3](../../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json)<br>cuda | MoG + particles (per task) | Tier 1: 4/5<br>FAIL 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| K3P without A2 · [k3p-a2-off-native-diagnostic](../../../configs/forge/ideas/k3p-a2-off-native-diagnostic.json)<br>cuda | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| K3P without critic anchor · [forge-onboarding-anchor-ablation](../../../configs/forge/ideas/forge-onboarding-anchor-ablation.json)<br>cuda | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| K3P without critic penalty · [forge-no-critic-penalty](../../../configs/forge/ideas/forge-no-critic-penalty.json)<br>cuda | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| K3P without training output noise · [k3p-no-output-noise-diagnostic](../../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json)<br>cuda | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| KA2 · [ka2 · 093c6f2bd417](../../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json)<br>cuda | MoG + particles (per task) | Tier 1: 4/5<br>FAIL 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| R1/R2 · [r1r2 · 302b6baa44f6](../../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json)<br>cuda | MoG + particles (per task) | Tier 1: 3/5<br>FAIL 1 · UNKNOWN 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |

## Selected-policy GPU scores

Each row keeps its own configuration, execution source and 26 required slots. These diagnostic scores remain separate from ordinary qualification.

| Model/configuration | Representation | Measured scores |
| --- | --- | --- |
| [Atlas C6 · `9563dea5`](../atlas-current-gpu-diagnostics-native-v2-20261003/README.md) | Particles | **7/26 PASS** · FAIL 11 · BLOCKED 8 · [source and 18 goal GIFs](../atlas-current-gpu-diagnostics-native-v2-20261003/README.md) |
| [`atlas_conditional` · C6 · `ff94453b`](../atlas-named-gpu-diagnostics-native-v3-20261003/README.md) | Particles (conditional clouds / role banks) | **4/26 PASS** · NOT_RUN 22<br>goal GIFs: [mid_scale_identity](../atlas-named-gpu-diagnostics-native-v3-20261003/gifs/mid_scale_identity_conditional_policy_selected_cloud_v1.gif) · [residual_student](../atlas-named-gpu-diagnostics-native-v3-20261003/gifs/residual_student_conditional_policy_selected_cloud_v1.gif) · [trajectory](../atlas-named-gpu-diagnostics-native-v3-20261003/gifs/trajectory_conditional_policy_selected_cloud_v1.gif) · [unipolar](../atlas-named-gpu-diagnostics-native-v3-20261003/gifs/unipolar_conditional_policy_selected_cloud_v1.gif) |
| [`atlas_ae_routed` · C6 · `ff94453b`](../atlas-named-gpu-diagnostics-native-v3-20261003/README.md) | MoG (fixed σ .025; routed AE) | **1/26 PASS** · NOT_RUN 25<br>goal GIFs: [ae_gan_hold](../atlas-named-gpu-diagnostics-native-v3-20261003/gifs/ae_gan_hold_ae_routed_policy_v1.gif) |
| [`atlas_routed` · C6 · `fb7acc77`](../atlas-named-gpu-diagnostics-native-v4b-20261004/README.md) | Particles (shared / slot parameter bank) | **1/26 PASS** · NOT_RUN 25<br>goal GIFs: [unused_token_hold](../atlas-named-gpu-diagnostics-native-v4b-20261004/gifs/unused_token_hold_routed_policy_selected_cloud_v1.gif) |
| [`atlas_multibank` · C6 · `fb7acc77`](../atlas-named-gpu-diagnostics-native-v4b-20261004/README.md) | Particles (two routed clouds) | **1/26 PASS** · NOT_RUN 25<br>goal GIFs: [cover_leftover](../atlas-named-gpu-diagnostics-native-v4b-20261004/gifs/cover_leftover_multibank_policy_v1.gif) |
| [`atlas_word_joint_min11` · C6 · `fb7acc77`](../atlas-word-retained-context-20261004/README.md) | Particles (N11 joint cloud; free encoder) | **INVALID 1** · NOT_RUN 25 · numerical gate UNAVAILABLE<br>retained INVALID illustration: [five_word_joint_acquisition](../atlas-word-retained-context-20261004/word-retained-goal.gif) |

C6 is the fixed LR .0053125 / prior-rate 1.5 configuration with declared host-specific Recipe fields. Representation labels come from the pinned applied priors, Recipes and routed table owners; a parameter bank is not a sampled MoG. [Full source and representation bindings](../technique-inventory.json). Original N5 word execution remains BLOCKED. The N11 illustration has no accepted numerical grade.

## Qualification and accounting

Each model/configuration keeps its own source, law and full 26-slot denominator. MoG + particles (per task) is a mixture of separate suite hosts, not a hybrid-model claim. Named diagnostic passes do not fill ordinary cells or combine into a family score. INVALID supplies no accepted numerical grade; original N5 word execution remains BLOCKED. No family default or comparable speed winner is established.

The named campaign's predecessor-inclusive charge remains **910.2391431590077 / 10500 seconds**, reserve zero: GPU0 **234.82608077581972 / 7500**, GPU1 **675.4130623831879 / 3000**. Prospective rate proposals are unexecuted and add no measurement to these tables.

The linked original report sections and their index.json below retain their separate snapshots.

## Completed current Atlas GPU diagnostic

The [native-v2 result and original goal GIFs](../atlas-current-gpu-diagnostics-native-v2-20261003/README.md)
cover all 26 current required questions: **7 PASS, 11 executed FAIL and 8 BLOCKED**;
Tier 1 is 0/5, Tier 2 is 7/19 and Tier 3 is 0/2. All 17 physical jobs completed,
costing 2,996.7973392466083 measured paid seconds with zero interruption reserve.
The two ring slots share one job and their cost is counted once.

The [portable diagnostic JSON](../atlas-current-gpu-diagnostics-native-v2-20261003/summary.json)
binds execution commit `9563dea57bb150f2a0275bbe8d785bf76210fca3`, source digest
`db5492df4aa5ef60ce9e7d5869b2b6be8d492f19191a2889967274555f8d0037`,
the seed-0 C6 rate pair, actual Recipes, GPU/runtime, original task/evaluator
identities, numerical grades, costs and all 18 original GIFs. It explains each
passing question and records the failed numerical bounds. Raw checkpoints and
populations remain outside Git.

[Retained goal-view supplements](../atlas-current-gpu-publication-views-20261003/README.md)
make TwoPole's near-zero failure and the three native tests' missing Gaussian
width visible. They use the original saved arrays, nine actual states and
analytic reference contours, with zero new draws, updates or scoring. Their
separately recorded CPU display cost supplies no convergence-speed credit.

This source/cohort has diagnostic credit only. Its passes cannot fill the
ordinary leaderboard, historical Atlas19 or a different family's cells, and
no shipping default or fair convergence-speed winner is established. The eight
source/API blockers are being evaluated as separately named family variants;
the original blocked identities retain their original results. The older
comparison scopes below and their `index.json` remain their original snapshots.

## Retained acquisition and stability

The [saved-observation audit](../atlas-retained-acquisition-gaps-20261003/README.md) distinguishes a first passing read from five-read acquisition. Overlap first passes at 150 and confirms at 850; intensity first passes at 350 and confirms at 525. Both have early gaps and no observed loss after confirmation. Their original terminal-five PASS results stand. The ring fails precision at 1407 (HQ .89453125 below .9), while retaining all eight modes and cover 1. This independently replayed chronology changes no gates or grades and adds zero training, scoring or qualification credit.

## Named-family startup attempts

The [native-v1 engineering report](../atlas-named-gpu-diagnostics-invalid-20261003/README.md)
and [portable receipt](../atlas-named-gpu-diagnostics-invalid-20261003/summary.json)
retain two **INVALID** attempts before training and six adapted questions
**NOT_RUN**. Each of the five separately named families keeps all 26 required
questions: 2 INVALID and 128 NOT_RUN cells across their separate ledgers.
The measured engineering charge is **25.322955040959641 seconds**, reserve zero.
There are no numerical PASS/FAIL outcomes, observations, checkpoints or GIFs.

The exact failure was typed planner metadata reaching strict producer
constructors; all scientific fields still match their frozen definitions.
It supplies no capacity or optimizer conclusion. A correction must use a new
frozen execution source and preserve these failed attempts and costs. This
report cannot fill another family, original source or ordinary ranking.

## Completed named-family v3 cut

The [source-bound results and five actual goal GIFs](../atlas-named-gpu-diagnostics-native-v3-20261003/README.md)
record **5 PASS, 1 INVALID and 2 NOT_RUN** among the eight adapted questions.
Trajectory, residual-student, unipolar, mid-scale identity and routed AE passed
their full original horizons and terminal-five gates at the fixed C6 rate pair.
Unused-token hit a CUDA/CPU evaluation-boundary defect; cover and min11 word
were never admitted. This is not a numerical failure of the routed family.

The [portable results](../atlas-named-gpu-diagnostics-native-v3-20261003/results.json)
retain **5 PASS, 1 INVALID and 124 NOT_RUN** across five separate 26-slot views,
with exact Recipes, runtime, selected owners, original grades, cost and source
pins. Current paid cost is **235.7073353389278 seconds**, reserve zero; including
the earlier **25.32295504095964-second** engineering debit, the campaign charge
is **261.03029037988745/10500 seconds**. These are supervised costs, not fair
convergence-speed measurements. All original N5 blocks and other source-scoped
results remain separate. No complete family, shipping default or speed winner
is established. The five completed passes are retained for the repaired
continuation, which may execute only the three remaining questions.

## Completed named-family continuation

The [v4b portable results and original goal GIFs](../atlas-named-gpu-diagnostics-native-v4b-20261004/README.md)
record **2 PASS and 1 INVALID** among the three remaining adaptations.
Unused-token passes its full 200 updates and cover its full 800 updates, both
with 24 recorded reads and the original terminal-five bounds. Their questions
are preservation of concept geometry under nuisance variation, and separation
of residual poles while preserving content and identity. The min11 word
producer is rejected by the original health guard; INVALID carries no numerical
grade. Five v3 passes remain separate source-bound references, with no reruns.

The [word retained-evidence companion](../atlas-word-retained-context-20261004/README.md)
supplies its original nine-frame illustration, recorded 20,001 G/E/prior/D
updates and 24 finite/pure reads. It corrects the frozen publisher's clock,
error and GIF-availability wording while retaining INVALID and numerical
UNAVAILABLE. The [separate sentinel diagnosis](../atlas-word-dimension-health-20261004/README.md)
traces the rejection to a safely skipped undefined reference-dimension
diagnostic. Saved model tensors are finite; the recorded word goals are missed.

The [passive word-goal analysis](../atlas-word-goal-diagnosis-20261004/WORD_GOALS.md)
shows that the encoder was active. The original joint objective has no explicit
inverse reconstruction term, and public DV12 perturbation remains in the
reconstruction question. A zero all-five flag still allowed one to three
individually correct words. Lower shared motion and slower prior rows are two
proposed rate-only contrasts; neither is executed or a predicted repair.
Critic damping and stochastic joint coupling remain competing explanations.

The [independent word-law review](../atlas-word-joint-law-review-20261004/WORD_LAW.md)
reproduces 24,576 generated-code perturbations and 120 inverse-code
perturbations from saved arrays. At those observed states, the ideal continuous
fake-code law and the five deterministic real-code atoms cannot match exactly.
This does not prove that the finite word gates are impossible or identify the
cause of training failure. The original INVALID and costs remain unchanged;
the proposed rate contrasts retain the objective and measurement law.

All **130 current-source cells** remain visible: **2 PASS, 1 INVALID and 127
NOT_RUN**. Execution source is `fb7acc775b3a1a6184d36b55e035b9da04531492` /
`f380eed990931bacb205e6537beaf387fdbb676ffc97b6f32f19ff903ae1cfed`.
Current paid cost is **649.2088527791202 seconds**, reserve zero. Including the
unchanged v1 and v3 debits once, the campaign is **910.2391431590077 / 10500
seconds**; GPU1 is **675.4130623831879 / 3000**. The original N5 word question
remains BLOCKED. No complete family, default or speed winner is established.

The [two actual CUDA scorer controls](../routed-cuda-scorer-control-20261004/README.md)
passed on GPU1 before this continuation. Their two earlier collection failures
stay INVALID. Inclusive engineering cost is **17.023810611106455 / 120 seconds**,
separate from the named campaign; these controls supply structural evidence.

## Grid100 output-kernel diagnosis

The [corrected saved-endpoint diagnostic](../grid100-output-kernel-endpoint-20261004/README.md)
reproduces the old clean 100,000-sample holdout exactly. Clean samples retain
all 100 modes but lack Gaussian width; the existing public output kernel
restores width and passes the native and accuracy bounds at this endpoint.
The old clean FAIL remains unchanged. The two endpoint engineering attempts
cost **11.62249431014061 / 180 seconds**, with zero new optimizer updates and
zero convergence/default/speed credit. Full noisy-law acquisition and hold
remain unexecuted.

## Additional Gaussian density question

The [source-bound Gaussian report and original goal GIF](../gaussian2d-current-c6-gpu-20261004/README.md)
cover the existing `api-gaussian2d` / source-family-16 question. It asks whether
the public selected policy recovers the mean, covariance and distribution of
`N((1,1), .04 I)`, beyond reaching the target center. The full 1,000 G/D/prior
updates and 25 original 4,096-sample reads were exported, including all nine
goal frames. The retained raw verdict is FAIL: each of the last five reads
misses the covariance lower bound; the endpoint also misses radial and
projected KS. Mean error at the endpoint is .0148192 sigma, while covariance
eigenvalues are .606076 and .666635 instead of the required [.85, 1.15].

The supervisor exceeded its inclusive 180-second allowance before final
attestation, so the accepted status is **BUDGET_EXCEEDED** and the numerical
score is **UNAVAILABLE**. The byte-original raw FAIL badge is an unaccepted
illustration and is captioned accordingly. Measured paid cost is
**180.21387464087456 seconds**, reserve zero, overrun .21387464087456465.
This is a separate additional-question scope, supplies no current 26-slot,
default or speed credit, and is not added to the named 10,500-second campaign.
There is no retry. The [independent publication review](../gaussian2d-current-c6-gpu-20261004/root-publication-review.json)
verifies 1,231 pinned inputs and the original nine-frame media bytes without
restoring a model, drawing samples or recomputing a numerical score.

| Scope | Source / declared law | Comparable score | Status and cost | Team report |
|---|---|---|---|---|
| Ordinary MoG, current Forge | Origin `28990990`; live, scheduled, learned MoG prior σ = 0.025 | KA2 and K3P tie at 4/5 Tier 1; 19 Tier 2 and 2 Tier 3 requirements remain unknown | Both FAIL; qualified tier 0; no default or speed credit | [Primary ordinary board](https://github.com/255BITS/ParticleGAN/blob/4749b2780add539df4bd8d2dd1d3cc9f002f77ad/reports/forge/technique-inventory.md) |
| Public-policy search, four separate cohorts | `eb2d77fb`, `0335ecf0`, `53102174`, `8021a1c5`; each keeps its own source/spec/runtime lane | 32 whole configs × 8 cases = 256 cells; 42 reached full cases, 214 UNKNOWN | No fully qualified config; 42 original + 10 reviewed GIFs | [Primary policy board](https://github.com/255BITS/ParticleGAN/blob/4749b2780add539df4bd8d2dd1d3cc9f002f77ad/reports/forge/policy-family-inventory.md) |
| Original Atlas19 replay | `a0d6d89f`; original config / observers / 48,800 updates | 19/19 original questions PASS; clean native diagnostics FAIL | Paid 5210.638847 s, reserve 0; original-scope evidence only | [PR266 report](https://github.com/255BITS/ParticleGAN/blob/edc9d2e6e1227f8d7a1dd52e6e363b03ef871eb8/reports/forge/continuous-baseline-20261003/README.md) |
| C6 broad hold extension | Original `8021a1c5` parents; H2 runner `82c85cc3`; 150 appended updates per family | 0/2 extension PASS; both FAIL; old original PASS / old study INCOMPLETE retained | New paid 31.184628 s + H1 engineering 6.818678 s once | [PR266 hold readout](https://github.com/255BITS/ParticleGAN/blob/edc9d2e6e1227f8d7a1dd52e6e363b03ef871eb8/reports/forge/continuous-baseline-20261003/README.md) |
| Critic-rate contrast (D2.25) | `a956c6fc`; LR .0053125, prior 1.5, D 2.25 | 2 full intensity FAIL, 14 UNKNOWN; original 2 FAIL and study 2 FAIL; Q1 16/16 SUPPORTED | Science 54.358772 s + engineering 4.757909 s = 59.116680 s; reserve 0 | [PR267 readout](https://github.com/255BITS/ParticleGAN/blob/dc3256b61042a5aa2193bc5dfc16799c15521521/reports/forge/critic-balance-20261003/README.md) |
| Generator-half contrast | `488b792e`; LR .00265625, prior 3, D 4.5 | 2 full intensity FAIL, 14 UNKNOWN; original 2 FAIL and study 2 FAIL; Q1 16/16 SUPPORTED | New science 54.877571 s; prior 59.116680 s once; campaign 113.994252 s; reserve 0 | [PR268 readout](https://github.com/255BITS/ParticleGAN/blob/95bc3480/reports/forge/generator-step-20261003/README.md) |
| Next KA2/K3P named-family cohort | LR .006375, prior 1, D 1; fast / no DV12 / fixed σ warmup / scheduled / AMSGrad false | UNEXECUTED at this boundary; 16 scientific cells UNKNOWN; no new Q1 or learned credit here | Remaining pair 15,246.005748 s; cost slots inherited, evidence never inherited | Authors preparing a separate frozen cohort |

The original Atlas19 image checks require full mode recovery and HQ ≥ 0.9; TV is diagnostic there. Current API image checks also gate finite-template TV. Original Atlas19 native evidence has 34 observations, final-five 20k checks and an independent 100k holdout. The current API native cohort has 24 full-count 20k observations and final-five checks, with noisy selected-policy primary samples and output-noise-off samples from that same selected policy as diagnostics. It has no separately forced EMA branch or independent 100k holdout. Those distinctions prevent cross-filling requirements.

## Ordinary MoG selected rows

Only rows inside the same declared comparison cohort can be ordered by their required PASS counts. These are imported selected rows, not a new sweep. All 47 configuration rows remain in the linked primary JSON. A failure or unknown downstream case is not erased by a higher pass count.

| Compatible selected rows | Tier 1 PASS / required | Tier 1 other status | Tier 2 / Tier 3 | Overall |
|---|---|---|---|---|
| ka2, k3p | 4/5 | 1 FAIL | 0/19 and 0/2; all UNKNOWN | FAIL, qualified tier 0 |
| bcap, r1r2, release07-gan-v3-mog | 3/5 | 1 FAIL, 1 UNKNOWN | 0/19 and 0/2; all UNKNOWN | FAIL, qualified tier 0 |
| Atlas, E22 canonical ordinary rows | 0/5 | BLOCKED | BLOCKED | No measured ordinary qualification |
| Other registered controls / source-only rows | See primary row | FAIL or BLOCKED; not independently ranked here | Unknown or blocked | No qualification |

## Whole-config policy selections within each source cohort

The primary board selects on study PASS counts, not endpoint appearance or speed. A display tie-break is only a display choice. C6 LR .0053125 / prior 1.5 is the useful original-gate baseline for the hold question, although the imported Atlas display tie chooses a different tuple. Both C6 smoke originals PASS; broad acquisition is too late for five later checks, so the whole config remains INCOMPLETE with six later domains UNKNOWN.

| Source cohort | Family | Imported best-observed tuple (LR / prior) | Original / 8 | Study / 8 | Unknown / 8 | Whole-config status |
|---|---|---|---|---|---|---|
| `eb2d77fb` | atlas | 0.006375 / 2.0 | 1/8 | 1/8 | 6/8 | FAIL |
| `eb2d77fb` | e22 | 0.006375 / 2.0 | 1/8 | 1/8 | 6/8 | FAIL |
| `0335ecf0` | atlas | 0.002125 / 1.0 | 1/8 | 0/8 | 7/8 | INCOMPLETE |
| `0335ecf0` | e22 | 0.002125 / 2.0 | 1/8 | 0/8 | 7/8 | INCOMPLETE |
| `53102174` | atlas | 0.0031875 / 1.0 | 1/8 | 1/8 | 6/8 | FAIL |
| `53102174` | e22 | 0.0053125 / 1.0 | 1/8 | 1/8 | 6/8 | FAIL |
| `8021a1c5` | atlas | 0.006375 / 1.5 | 1/8 | 1/8 | 6/8 | FAIL |
| `8021a1c5` | e22 | 0.0053125 / 1.5 | 2/8 | 1/8 | 6/8 | INCOMPLETE |

The JSON retains every one of the 32 tuples and every required denominator. Neither the four policy cohorts nor the newer two contrasts are pooled into a synthetic eight-case success. Atlas/E22 numerical matches on some small hosts do not establish complete algorithm equivalence; selected density paths and controller state remain source-owned.

## Evidence availability and accounting

The original/policy primary boards above are pinned to the last verified develop boundary `4749b278`. PR266/267/268 compact results are linked to their report branches and are not certified as merged to develop by this index. No fresh GitHub state or CI query was made. The bounded open-PR inventory covered only the newest 100 updated open rows; older open PRs and PR255’s current production diff remain unverified. PR255 was untouched.

| Raw archive | Availability | Bytes | SHA-256 |
|---|---|---|---|
| critic_rate | LOCAL_ONLY; not remotely replicated | 64,385,103 | `292061adad594577d9186207234929afa42437cdf9e5539128d9ca4865a7b533` |
| generator_half | LOCAL_ONLY; not remotely replicated | 111,262,244 | `271cf1c21a341d33d520311160c3d9c03f3ffcb6094dd544602e2084089c8949` |

Reported paid seconds are supervisory attempt intervals. Reservation is separate, and CPU capacity construction/replay/verification is diagnostic work outside ordinary learning credit. The D2.25 engineering startup is charged once in its own readout and once as prior cost within the generator-half cumulative campaign; do not add that cumulative figure to the earlier total again. Ordinary selected-row wall times and historical replay costs belong to different declared studies and are not summed into this campaign or used as a fair speed ranking.

The older comparison projection's inputs and hashes remain in [index.json](index.json).
Each newer report has its own linked input index and scope. The navigation
index itself adds no training updates, sampler calls, metric rescoring or gate
changes. Primary reports and their source-bound receipts remain authoritative.
