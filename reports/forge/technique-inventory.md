# Current model/configuration scores

One selected configuration per family; full tier denominators stay fixed. Representation is read from its declared prior and task-owned hosts.

| Model/configuration | Representation | Measured scores |
| --- | --- | --- |
| Atlas · [atlas](../../configs/forge/ideas/atlas.json)<br>cuda | Particles | Tier 1: BLOCKED (5 required)<br>Tier 2: BLOCKED (19 required)<br>Tier 3: BLOCKED (2 required)<br>Recorded tier: 0 |
| BCap · [bcap · 08689a73c551](../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json)<br>cuda | MoG + particles (per task) | Tier 1: 3/5<br>FAIL 1 · UNKNOWN 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| E22 · [e22](../../configs/forge/ideas/e22.json)<br>cuda | Particles | Tier 1: BLOCKED (5 required)<br>Tier 2: BLOCKED (19 required)<br>Tier 3: BLOCKED (2 required)<br>Recorded tier: 0 |
| GAN v3 release 0.7 (MoG) · [release07-gan-v3-mog · 1e266b5a2986](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json)<br>cuda | MoG + particles (per task) | Tier 1: 3/5<br>FAIL 1 · UNKNOWN 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| GAN v3 release 0.7 (cloud) · [release07-gan-v3-cloud-v1](../../configs/forge/ideas/release07-gan-v3-cloud-v1.json)<br>cuda | MoG + particles (per task) | Tier 1: BLOCKED (5 required)<br>Tier 2: BLOCKED (19 required)<br>Tier 3: BLOCKED (2 required)<br>Recorded tier: 0 |
| K3P · [k3p · 0b37e98a01e3](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json)<br>cuda | MoG + particles (per task) | Tier 1: 4/5<br>FAIL 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| K3P without A2 · [k3p-a2-off-native-diagnostic](../../configs/forge/ideas/k3p-a2-off-native-diagnostic.json)<br>cuda | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| K3P without critic anchor · [forge-onboarding-anchor-ablation](../../configs/forge/ideas/forge-onboarding-anchor-ablation.json)<br>cuda | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| K3P without critic penalty · [forge-no-critic-penalty](../../configs/forge/ideas/forge-no-critic-penalty.json)<br>cuda | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| K3P without training output noise · [k3p-no-output-noise-diagnostic](../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json)<br>cuda | MoG + particles (per task) | Tier 1: 0/5<br>FAIL 1 · UNKNOWN 4<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| KA2 · [ka2 · 093c6f2bd417](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json)<br>cuda | MoG + particles (per task) | Tier 1: 4/5<br>FAIL 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |
| R1/R2 · [r1r2 · 302b6baa44f6](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json)<br>cuda | MoG + particles (per task) | Tier 1: 3/5<br>FAIL 1 · UNKNOWN 1<br>Tier 2: UNKNOWN (19 required)<br>Tier 3: UNKNOWN (2 required)<br>Recorded tier: 0 |

## Selected-policy GPU scores

Each row keeps its own configuration, execution source and 26 required slots. These diagnostic scores remain separate from ordinary qualification.

| Model/configuration | Representation | Measured scores |
| --- | --- | --- |
| [Atlas C6 · `9563dea5`](atlas-current-gpu-diagnostics-native-v2-20261003/README.md) | Particles | **7/26 PASS** · FAIL 11 · BLOCKED 8 · [source and 18 goal GIFs](atlas-current-gpu-diagnostics-native-v2-20261003/README.md) |
| [`atlas_conditional` · C6 · `ff94453b`](atlas-named-gpu-diagnostics-native-v3-20261003/README.md) | Particles (conditional clouds / role banks) | **4/26 PASS** · NOT_RUN 22<br>goal GIFs: [mid_scale_identity](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/mid_scale_identity_conditional_policy_selected_cloud_v1.gif) · [residual_student](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/residual_student_conditional_policy_selected_cloud_v1.gif) · [trajectory](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/trajectory_conditional_policy_selected_cloud_v1.gif) · [unipolar](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/unipolar_conditional_policy_selected_cloud_v1.gif) |
| [`atlas_ae_routed` · C6 · `ff94453b`](atlas-named-gpu-diagnostics-native-v3-20261003/README.md) | MoG (fixed σ .025; routed AE) | **1/26 PASS** · NOT_RUN 25<br>goal GIFs: [ae_gan_hold](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/ae_gan_hold_ae_routed_policy_v1.gif) |
| [`atlas_routed` · C6 · `fb7acc77`](atlas-named-gpu-diagnostics-native-v4b-20261004/README.md) | Particles (shared / slot parameter bank) | **1/26 PASS** · NOT_RUN 25<br>goal GIFs: [unused_token_hold](atlas-named-gpu-diagnostics-native-v4b-20261004/gifs/unused_token_hold_routed_policy_selected_cloud_v1.gif) |
| [`atlas_multibank` · C6 · `fb7acc77`](atlas-named-gpu-diagnostics-native-v4b-20261004/README.md) | Particles (two routed clouds) | **1/26 PASS** · NOT_RUN 25<br>goal GIFs: [cover_leftover](atlas-named-gpu-diagnostics-native-v4b-20261004/gifs/cover_leftover_multibank_policy_v1.gif) |
| [`atlas_word_joint_min11` · C6 · `fb7acc77`](atlas-word-retained-context-20261004/README.md) | Particles (N11 joint cloud; free encoder) | **INVALID 1** · NOT_RUN 25 · numerical gate UNAVAILABLE<br>retained INVALID illustration: [five_word_joint_acquisition](atlas-word-retained-context-20261004/word-retained-goal.gif) |
| [half_base LR .00265625 / prior1.5 / D1 · source `f9f7ed9d`](word-half-base-20261004/README.md) | Particles (N11 joint cloud; free encoder) | **0/26 PASS** · FAIL 1 · NOT_RUN 25 · COMPLETE<br>Final quality 0.546875 · modes 2/5 · mass TV 0.622266 · paired exact 0<br>[accepted 20,001-update goal GIF](word-half-base-20261004/media/goal.gif) |

C6 is the fixed LR .0053125 / prior-rate 1.5 configuration with declared host-specific Recipe fields. Representation labels come from the pinned applied priors, Recipes and routed table owners; a parameter bank is not a sampled MoG. [Full source and representation bindings](technique-inventory.json). Original N5 word execution remains BLOCKED. The original C6 N11 illustration has no accepted numerical grade. The separate half_base run completed all 20,001 updates and 24 reads, with zero passing reads and an accepted numerical FAIL.

## Qualification and scope

MoG + particles (per task) denotes separate declared host laws within a suite; it does not assert a hybrid model. The release-0.7 cloud-labelled row retains its archived MoG declaration. BLOCKED rows show requested representations, not successful execution.

Before shipping defaults, one unchanged family/configuration tuple must satisfy all 26 required 5/19/2 gates with compatible source, runtime and serving evidence, followed by the separate calibration and robustness requirements. Named diagnostic rows do not pool passing cells across families or sources and grant no ordinary tier, default or speed credit.

Atlas's historical study has **19/19 original PASS**. The later C6 broad hold has **2/2 hold FAIL**, with six other domains UNKNOWN per family. See [completed source-bound studies](#completed-source-bound-studies) for the exact protocols and original goal GIFs. These separate results do not fill ordinary qualification.

<details>
<summary>Selected configurations and provenance</summary>

### Atlas (cuda)

- **Selected configuration:** [atlas](../../configs/forge/ideas/atlas.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 506319d4621e / e198762f715a
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** BLOCKED 26
- **Paid seconds:** unknown

- **Separate baseline evidence:** [Atlas history: 19/19 PASS](continuous-baseline-20261003/README.md) · [C6 Atlas hold FAIL](c6-baseline-debug-20261003/README.md); no current tier credit.

### BCap (cuda)

- **Selected configuration:** [bcap · 08689a73c551](../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 8d2031fe492f / bf859ff60898
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 22
- **Paid seconds:** 30.559

### E22 (cuda)

- **Selected configuration:** [e22](../../configs/forge/ideas/e22.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** cf5d3fdcbb1d / cab0cdc81bdb
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** BLOCKED 26
- **Paid seconds:** unknown

- **Separate baseline evidence:** [Atlas history: 19/19 PASS](continuous-baseline-20261003/README.md) · [C6 E22 hold FAIL](c6-baseline-debug-20261003/README.md); no current tier credit.

### GAN v3 release 0.7 (MoG) (cuda)

- **Selected configuration:** [release07-gan-v3-mog · 1e266b5a2986](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 47630167ae31 / f99998f9b0fa
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 22
- **Paid seconds:** 28.757

### GAN v3 release 0.7 (cloud) (cuda)

- **Selected configuration:** [release07-gan-v3-cloud-v1](../../configs/forge/ideas/release07-gan-v3-cloud-v1.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 95e403581ebf / 6cb4d268b94e
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** BLOCKED 26
- **Paid seconds:** unknown

### K3P (cuda)

- **Selected configuration:** [k3p · 0b37e98a01e3](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 412dbcba9246 / 7ebaeb278a75
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 21
- **Paid seconds:** 210.977

### K3P without A2 (cuda)

- **Selected configuration:** [k3p-a2-off-native-diagnostic](../../configs/forge/ideas/k3p-a2-off-native-diagnostic.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** b51787a935f0 / 31b4d36cc007
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 25
- **Paid seconds:** 5.318

### K3P without critic anchor (cuda)

- **Selected configuration:** [forge-onboarding-anchor-ablation](../../configs/forge/ideas/forge-onboarding-anchor-ablation.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 0fa611b1ebcc / 08a552799580
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 25
- **Paid seconds:** 5.419

### K3P without critic penalty (cuda)

- **Selected configuration:** [forge-no-critic-penalty](../../configs/forge/ideas/forge-no-critic-penalty.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** ea1e0bdda2f8 / 866cef3392e0
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 25
- **Paid seconds:** 5.517

### K3P without training output noise (cuda)

- **Selected configuration:** [k3p-no-output-noise-diagnostic](../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** b66c72f805f4 / edc93bce30da
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 25
- **Paid seconds:** 5.657

### KA2 (cuda)

- **Selected configuration:** [ka2 · 093c6f2bd417](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** d5ebfc250aac / 7be3028bd4fc
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 21
- **Paid seconds:** 222.721

### R1/R2 (cuda)

- **Selected configuration:** [r1r2 · 302b6baa44f6](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json)
- **Selection:** historical incumbent; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 2727e20c5a10 / 4ff453a8a3ed
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 22
- **Paid seconds:** 27.627

</details>

Recorded results remain bound to their actual recipes, priors, initialization, budgets, sampling laws and hardware. They do not pool qualification across sources or qualify the latest checkout. Selection never combines passing tasks or tiers from different configurations. A failed best-observed configuration is not a qualified winner. Search qualification covers only its declared tuning tiers; later-tier outcomes are reported separately. The screening profile remains provisional and does not confer calibrated robustness or public-default adoption.

BLOCKED means execution was incompatible or a prerequisite was unavailable. NOT RUN and UNKNOWN mean unexecuted or unmeasured; FAIL records an executed gate failure. Failed or blocked prerequisites stop later work; required denominators stay fixed. The Atlas-history and C6-hold links retain their separate protocols and supply no current tier credit.

[All configuration alternatives, trials, task statuses and exact bindings](technique-inventory.json) · [Evidence and archived publication identities](technique-evidence/manifest.json)

Regenerate this same leaderboard from committed evidence, without training or raw-log hydration:

```sh
python reports/forge/regenerate_technique_inventory.py
```

For ordinary Forge qualification, use `--source-commit <executed-commit>` to independently regrade hydrated original receipts and update this leaderboard. Reusable candidates use the same complete task ladder. Historical task-only diagnostics remain motivation and reproduction evidence. Source snapshots are provenance, not additional leaderboards.

Publication input digest `9f5c3700dad2c6be9e8becd106ab27a549492e9ae7ca5f56b9ec9ef1ec32ed6a`.

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

These separate publications add evidence navigation. They do not change the ordinary table, ranks, tiers or qualification. Historical and current cells are not pooled; no default or fair speed winner is established.

| Study | Result | Goal and readout |
| --- | --- | --- |
| Historical Atlas19 original protocols | 19/19 original PASS | [actual goal GIF](continuous-baseline-20261003/media/atlas-original19-portability-img_intensity2.gif); [readout + exact gates](continuous-baseline-20261003/README.md) |
| Named C6 broad hold extension | 2/2 new hold FAIL | [actual goal GIF](continuous-baseline-20261003/media/c6-atlas-broad-hold-1200-to-1350.gif); [readout + exact gates](continuous-baseline-20261003/README.md) |
| Critic-rate contrast | 2 FAIL; 14 UNKNOWN | [actual goal GIF](critic-balance-20261003/media/atlas/image-develop-img_intensity2-source-transpose12/goal.gif); [readout + exact gates](critic-balance-20261003/README.md) |
| Generator-half contrast | 2 FAIL; 14 UNKNOWN | [actual goal GIF](generator-step-20261003/media/atlas/image-develop-img_intensity2-source-transpose12/goal.gif); [readout + exact gates](generator-step-20261003/README.md) |

<details>
<summary>Exact protocols, sources, costs and archive availability</summary>

### Historical Atlas19 original protocols

- **Required evidence and result:** 19/19 original PASS; 48,800 original updates; native final-five 20k + independent 100k; clean native diagnostics separate
- **Source and recipe:** `a0d6d89f; original noisy law`
- **Serving law:** Original noisy selected-policy law; clean diagnostics separate; native joint coverage+accuracy includes independent 100k
- **Paid / reserve (seconds):** 5210.638847 / 0.000000
- **Raw availability:** [archive card](continuous-baseline-20261003/archive.json); [raw resolver (LOCAL_ONLY)](continuous-baseline-20261003/ARCHIVE.md)

### Named C6 broad hold extension

- **Required evidence and result:** 2/2 new hold FAIL; old 1200 PASS/study INCOMPLETE retained; 150 appended each; 3/5 later checks
- **Source and recipe:** `8021a1c5 → 82c85cc3`
- **Serving law:** Original 8021 public served sampler; output_noise=False; 150 appended updates, no ordinary eight-case credit
- **Paid / reserve (seconds):** 31.184628 new + 6.818678 startup / 0.000000
- **Raw availability:** [archive card](continuous-baseline-20261003/archive.json); [raw resolver (LOCAL_ONLY)](continuous-baseline-20261003/ARCHIVE.md)

### Critic-rate contrast

- **Required evidence and result:** 2 configs × 8 = 16; 16 cold SUPPORTED; 2 FAIL, 14 UNKNOWN; original 2 FAIL, 14 UNAVAILABLE; hold 2 FAIL, 14 UNAVAILABLE; 24 observations; native 20k, no 100k
- **Source and recipe:** `a956c6fc; selected-policy; LR/prior/D 0.0053125/1.5/2.25`
- **Serving law:** Public selected-policy serving; image/vector output-noise-off primary, native noisy primary; 24×20k native checks, no independent 100k
- **Paid / reserve (seconds):** 54.358772 new / 0.000000; cumulative 59.116680
- **Raw availability:** [archive card](critic-balance-20261003/archive.json); [raw resolver (LOCAL_ONLY)](critic-balance-20261003/ARCHIVE.md)

### Generator-half contrast

- **Required evidence and result:** 2 configs × 8 = 16; 16 cold SUPPORTED; 2 FAIL, 14 UNKNOWN; original 2 FAIL, 14 UNAVAILABLE; hold 2 FAIL, 14 UNAVAILABLE; 24 observations; native 20k, no 100k
- **Source and recipe:** `488b792e; selected-policy; LR/prior/D 0.00265625/3.0/4.5`
- **Serving law:** Public selected-policy serving; image/vector output-noise-off primary, native noisy primary; 24×20k native checks, no independent 100k
- **Paid / reserve (seconds):** 54.877571 new / 0.000000; cumulative 113.994252
- **Raw availability:** [archive card](generator-step-20261003/archive.json); [raw resolver (LOCAL_ONLY)](generator-step-20261003/ARCHIVE.md)

</details>

Original GIF verdicts and added hold verdicts remain separate in each readout. Capacity uses zero optimizer updates and provides no learned PASS. UNKNOWN means unreached; engineering failures remain distinct from numerical FAIL.

The historical intensity host uses residual16, seed 0 and enumeration of 32 rows without latent perturbation; the new intensity host uses transpose12, seed 24002 and 1024 public noise-off selected-policy draws with latent perturbation. These are different protocols. The C6 extension preserves only the original broad checkpoint's PASS/INCOMPLETE grades, not a full eight-case qualification.

Costs are summed supervised child intervals, with conservative reserves separate. Generator cumulative 113.994252 seconds already includes critic 59.116680 seconds once. Historical Atlas19+holds 5248.642153 seconds is a disjoint study; these figures are not elapsed-time or FLOPs rankings. Raw archives remain LOCAL_ONLY; this projection does not hydrate or independently recertify them.


[Exact C6 baseline and retained persistence diagnosis](c6-baseline-debug-20261003/README.md): two original smoke passes per family, six later domains UNKNOWN; the added broad hold fails on projected-CDF shape excursions. The source-bound diagnostic preserves all original gates and supplies no default or speed credit.
