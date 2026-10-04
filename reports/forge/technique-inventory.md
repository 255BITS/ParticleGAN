# Current model/configuration scores

Each row keeps its own configuration, source, representation and measured scope. Required denominators stay fixed; diagnostic and ordinary qualification scores remain separate.

| Model/configuration | Actual representation | Measured score/status and scope |
| --- | --- | --- |
| [Original PR223 Atlas FULL · LR .00425 / prior2](continuous-baseline-20261003/README.md) | Particles | **19/19 PASS** · full original recipe/law replay · [19 original goal GIFs](continuous-baseline-20261003/README.md)<br>Policy-selected; averaging enabled; learned output kernel (init .029) · native/moving seed1234, portability seed0 · [fresh retest](pr223-original-full-retest-stopped17-20261004/README.md): 16/19 PASS · 0 FAIL · 1 INVALID, 2 NOT_RUN · overall INCOMPLETE · source `2068a661`/`00cadbfd` · [final charge 3210.214214 / 10800 s](pr223-original-full-retest-stopped17-20261004/FINAL_COST.json)<br>[Native3 continuation](pr223-native3-first-invalid-20261004/README.md): 0/3 PASS · 0 FAIL · 3 unavailable · 1 INVALID, 2 NOT_RUN · overall INCOMPLETE · source `0fa92c5b`/`b2d0e980` · [inclusive campaign charge 3227.941695 / 10800 s](pr223-native3-first-invalid-20261004/FINAL_COST.json); prior cases and cumulative metadata counted once; no current-26/default credit |
| [Atlas C6 CHANGED-rate / noise-OFF · LR .0053125 / prior1.5 · source `9563dea5`](atlas-current-gpu-diagnostics-native-v2-20261003/README.md) | Particles | **7/26 PASS** · FAIL 11 · BLOCKED 8 <br>Selected-policy diagnostic/reference · seed0 · [source and 18 goal GIFs](atlas-current-gpu-diagnostics-native-v2-20261003/README.md) |
| [`atlas_conditional` · C6 · `ff94453b`](atlas-named-gpu-diagnostics-native-v3-20261003/README.md) | Particles (conditional clouds / role banks) | **4/26 PASS** · NOT_RUN 22<br>Named-family diagnostic · goal GIFs: [mid_scale_identity](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/mid_scale_identity_conditional_policy_selected_cloud_v1.gif) · [residual_student](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/residual_student_conditional_policy_selected_cloud_v1.gif) · [trajectory](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/trajectory_conditional_policy_selected_cloud_v1.gif) · [unipolar](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/unipolar_conditional_policy_selected_cloud_v1.gif) |
| [`atlas_ae_routed` · C6 · `ff94453b`](atlas-named-gpu-diagnostics-native-v3-20261003/README.md) | MoG (fixed σ .025; routed AE) | **1/26 PASS** · NOT_RUN 25<br>Named-family diagnostic · goal GIFs: [ae_gan_hold](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/ae_gan_hold_ae_routed_policy_v1.gif) |
| [`atlas_routed` · C6 · `fb7acc77`](atlas-named-gpu-diagnostics-native-v4b-20261004/README.md) | Particles (shared / slot parameter bank) | **1/26 PASS** · NOT_RUN 25<br>Named-family diagnostic · goal GIFs: [unused_token_hold](atlas-named-gpu-diagnostics-native-v4b-20261004/gifs/unused_token_hold_routed_policy_selected_cloud_v1.gif) |
| [`atlas_multibank` · C6 · `fb7acc77`](atlas-named-gpu-diagnostics-native-v4b-20261004/README.md) | Particles (two routed clouds) | **1/26 PASS** · NOT_RUN 25<br>Named-family diagnostic · goal GIFs: [cover_leftover](atlas-named-gpu-diagnostics-native-v4b-20261004/gifs/cover_leftover_multibank_policy_v1.gif) |
| [`atlas_word_joint_min11` · C6 · `fb7acc77`](atlas-word-retained-context-20261004/README.md) | Particles (N11 joint cloud; free encoder) | **INVALID 1** · NOT_RUN 25 · numerical gate UNAVAILABLE<br>Named-family diagnostic · retained INVALID illustration: [five_word_joint_acquisition](atlas-word-retained-context-20261004/word-retained-goal.gif) |
| [half_base LR .00265625 / prior1.5 / D1 · source `f9f7ed9d`](word-half-base-20261004/README.md) | Particles (N11 joint cloud; free encoder) | **0/26 PASS** · FAIL 1 · NOT_RUN 25 · COMPLETE<br>Final quality 0.546875 · modes 2/5 · mass TV 0.622266 · all-five exact 0<br>[accepted 20,001-update goal GIF](word-half-base-20261004/media/goal.gif) |
| BCap · [bcap · 08689a73c551](../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json)<br>cuda · ordinary qualification | MoG + particles (per task) | Tier 1: 3/5<br>FAIL 1 · UNKNOWN 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |
| E22 · [e22](../../configs/forge/ideas/e22.json)<br>cuda · ordinary qualification | Particles | Tier 1: BLOCKED (5 required)<br>Tier 2: BLOCKED (19 required)<br>Tier 3: BLOCKED (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |
| GAN v3 release 0.7 (MoG) · [release07-gan-v3-mog · 1e266b5a2986](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json)<br>cuda · ordinary qualification | MoG + particles (per task) | Tier 1: 3/5<br>FAIL 1 · UNKNOWN 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |
| GAN v3 release 0.7 (cloud) · [release07-gan-v3-cloud-v1](../../configs/forge/ideas/release07-gan-v3-cloud-v1.json)<br>cuda · ordinary qualification | MoG + particles (per task) | Tier 1: BLOCKED (5 required)<br>Tier 2: BLOCKED (19 required)<br>Tier 3: BLOCKED (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |
| K3P · [k3p · 0b37e98a01e3](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json)<br>cuda · ordinary qualification | MoG + particles (per task) | Tier 1: 4/5<br>FAIL 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |
| K3P without A2 · [k3p-a2-off-native-diagnostic](../../configs/forge/ideas/k3p-a2-off-native-diagnostic.json)<br>cuda · ordinary qualification | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |
| K3P without critic anchor · [forge-onboarding-anchor-ablation](../../configs/forge/ideas/forge-onboarding-anchor-ablation.json)<br>cuda · ordinary qualification | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |
| K3P without critic penalty · [forge-no-critic-penalty](../../configs/forge/ideas/forge-no-critic-penalty.json)<br>cuda · ordinary qualification | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |
| K3P without training output noise · [k3p-no-output-noise-diagnostic](../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json)<br>cuda · ordinary qualification | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |
| KA2 · [ka2 · 093c6f2bd417](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json)<br>cuda · ordinary qualification | MoG + particles (per task) | Tier 1: 4/5<br>FAIL 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |
| R1/R2 · [r1r2 · 302b6baa44f6](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json)<br>cuda · ordinary qualification | MoG + particles (per task) | Tier 1: 3/5<br>FAIL 1 · UNKNOWN 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 · [source](technique-inventory.json) |

The C6 diagnostic/reference changes rates to LR .0053125 / prior-rate 1.5 and disables evaluation output noise, with seed0 and declared host-specific Recipe fields. Representation labels come from the pinned applied priors, Recipes and routed table owners; a parameter bank is not a sampled MoG. [Full source and representation bindings](technique-inventory.json). Original N5 word execution remains BLOCKED. The original C6 N11 illustration has no accepted numerical grade. The separate half_base run completed all 20,001 updates and 24 reads, with zero passing reads and an accepted numerical FAIL.

## Qualification and scope

MoG + particles (per task) denotes separate declared host laws within a suite; it does not assert a hybrid model. The release-0.7 cloud-labelled row retains its archived MoG declaration. BLOCKED rows show requested representations, not successful execution.

Before shipping defaults, one unchanged family/configuration tuple must satisfy all 26 required 5/19/2 gates with compatible source, runtime and serving evidence, followed by the separate calibration and robustness requirements. Named diagnostic rows do not pool passing cells across families or sources and grant no ordinary tier, default or speed credit.

Atlas's historical study has **19/19 original PASS**. The later C6 broad hold has **2/2 hold FAIL**, with six other domains UNKNOWN per family. See [completed source-bound studies](#completed-source-bound-studies) for the exact protocols and original goal GIFs. These separate results do not fill ordinary qualification.

Recorded results remain bound to their actual recipes, priors, initialization, budgets, sampling laws and hardware. They do not pool qualification across sources or qualify the latest checkout. Selection never combines passing tasks or tiers from different configurations. A failed best-observed configuration is not a qualified winner. Search qualification covers only its declared tuning tiers; later-tier outcomes are reported separately. The screening profile remains provisional and does not confer calibrated robustness or public-default adoption.

BLOCKED means execution was incompatible or a prerequisite was unavailable. NOT RUN and UNKNOWN mean unexecuted or unmeasured; FAIL records an executed gate failure. Failed or blocked prerequisites stop later work; required denominators stay fixed. The Atlas-history and C6-hold links retain their separate protocols and supply no current tier credit.

[All configuration alternatives, trials, task statuses and exact bindings](technique-inventory.json) · [Evidence and archived publication identities](technique-evidence/manifest.json)

Regenerate this same leaderboard from committed evidence, without training or raw-log hydration:

```sh
python reports/forge/regenerate_technique_inventory.py
```

For ordinary Forge qualification, use `--source-commit <executed-commit>` to independently regrade hydrated original receipts and update this leaderboard. Reusable candidates use the same complete task ladder. Historical task-only diagnostics remain motivation and reproduction evidence. Source snapshots are provenance, not additional leaderboards.

Publication input digest `7f1dffc70e7115ce42ec432945146ab66475faf87b64f8b94c6ed965b051a60d`.

The [whole-family repair readout](family-wide-word-repairs/README.md) records the ordinary candidate attempts and bounded global configuration search. Their complete rows remain unranked alternatives below and in the companion JSON; a failed replacement does not make its historical incumbent a qualified standard.

Earlier view policies retain their exact numerical snapshots and receipt proofs in the companion JSON. Their outcomes do not fill current requirements:

- `discriminator_stability` revision 2: recorded denominators 3/19/2; 5 source cohorts.

Unranked alternatives retain their original outcomes and exact source/runtime bindings:

- `k3p-bcap-matched-v1` (bcap), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `release07-gan-v3-mog-v1` (release07-gan-v3-mog), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `release07-gan-v3-task-adapted-v1` (release07-gan-v3-mog), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p` (k3p), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2` (ka2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p-r1r2-matched-v1` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r3gan-stacked-training-toy-v1` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `bcap--0fc5834c2f114d83b8b7ace471dbc8271d0afddd8c92ba1b84d5db8325a8ecd5` (bcap), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `bcap--46afd334b69b00edb05f5487d7e00d2c5f52d82ab2d3adaa6d7c6d9158e73604` (bcap), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `bcap--4cb8358e1e3eefdb6744ae63e84ce724bf73a47e471a12817aaf3a14466c2dd3` (bcap), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `bcap--66881ffa01364264272e87b084a09b8a00ce0c6d7a49904cc73d7c60238e8ce0` (bcap), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `bcap--67963edee7209ea85f9c07ec1f1195301e0ac40a975a44ef9891c5241a4ba2bc` (bcap), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `bcap--c5f3aebf586c1da32ddfbaad77fab841eb2c43c7da652f56e9ffe622382d803e` (bcap), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `bcap--f65f1922d02456d3a4b7135bb3aa6483d1c76184d74608479b2634094b039383` (bcap), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `five-word-joint-ka2-v1` (ka2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--00b69f56de29e1968b9d84f9b532afc535fa44555e235eb6c14cfc58b7baed9b` (k3p), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--370894f9284090fc9771145a869f2d7fa617283c3736c8e461a47553c9668ce6` (k3p), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--9f4bc2973d95cf1545e009ec044ed582a318e6d869d62f4d99f6eee529d1d93d` (k3p), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--a4b78c36d9faacd837e079c3fd3b9623fc141f42737e03312053a86c560abb8a` (k3p), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--caa08ebf04c2b95ed480dd9c23e39a8624e39ae27b2b80a9af1ac74f13a86b9f` (k3p), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p-global-repair-v1` (k3p), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--4496b859e066d7a4b6279f2d34fbd58d083e0f18c2a22ade5ba12c964d01a7ca` (ka2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--6183426d7810463d5461ad936a42c67a38516fce8b32b23065da215fed8d145d` (ka2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--8d6bb101262f49dcef37111b37ab781ca09652b47ad95e783ccea4dfbec680b4` (ka2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--d02d7932342e1fa1db0a7a91537394297601e55516882d20ec033b4992bb527f` (ka2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--e425fc18e432bbec6c1a1e2fc0b2f8024c7ad104fd5acf8908b6cfbbb5a3da03` (ka2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2-global-repair-v1` (ka2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--083deac11d390b7895cca37396e9caf2adf5706d4ce560ae70a089e305ab3b4b` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--14eeafe7d865bdc5f6886ae8db64afd22bac3fd20a82e5ddb59188ab4fa30cc6` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--151d3b970d66ccb2956e2b914d518a0e5652e7164dccf000a29bbc82831b2d46` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--4363c95b3f6f2739729c7a104bd080a427e3efbef1f915ce14c191009bfa3a27` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--724b3d52fbb2172a985d1bf29890b2ef9ba2598c9f1724d93a9afa7c872cbdad` (r1r2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--7728cf93188b9fe122536244667789941da52d435cfb0a3ae42b90db200d7595` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--9481826257709128f1c864bac2950906d0ad940b8f75fbfa3d47317e7b26007d` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--abf642c42c5346ad096c29202e4716db535c393c113478552133c1c22761ddbd` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--b58a087cabab6b43997ec233416294cfc2786bab59026c07a1a4f2565970f5ff` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--dad6fdf50034381c106102df008e26b7687b04918cab85c13721e2a2dbbbe743` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--e033ba04e5b9609b896a084a182d785394869fcad19d4a15cffe86e9a973f3fe` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--f398f6ea70a2d101c520a1a01db90ea968bbb16736e6cec7ffc887e1dd7e241a` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--f78d5d603be1611025389c4ca0f7f485f3d2caa0a25fb6bc38b7c96da3509a3e` (r1r2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2-global-repair-v1` (r1r2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `release07-gan-v3-mog--7311bde77895ceac2d493a6b102672cab0973b6685383339a2867d80956aa11f` (release07-gan-v3-mog), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `release07-gan-v3-mog--8bac47a479c2bb8dea2edee5378b642fbe5e6dd8c5fceab2d4e8077d6371345e` (release07-gan-v3-mog), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `release07-gan-v3-mog--c1b747662d851c7c1ee0d8572e34c0438f28d5e0920c572b74622a25479f9e26` (release07-gan-v3-mog), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.

## Completed source-bound studies

Historical protocols and distinct contrasts retain their exact scores, source and GIFs in their original readouts. They do not fill cells in the table above.

- [Historical Atlas19 original protocols](continuous-baseline-20261003/README.md) · own 19-cell scope.
- [Named C6 broad hold extension](continuous-baseline-20261003/README.md) · own 2-cell scope.
- [Critic-rate contrast](critic-balance-20261003/README.md) · own 16-cell scope.
- [Generator-half contrast](generator-step-20261003/README.md) · own 16-cell scope.

[Exact C6 baseline and retained persistence diagnosis](c6-baseline-debug-20261003/README.md): two original smoke passes per family, six later domains UNKNOWN; the added broad hold fails on projected-CDF shape excursions. The source-bound diagnostic preserves all original gates and supplies no default or speed credit.
