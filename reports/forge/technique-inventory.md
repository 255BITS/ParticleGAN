# Current model/configuration scores

The declared view is revision **4 (6/19/2)**. Ordinary scores below retain their measured revision **3 (5/19/2)**. Additional required cells are **UNKNOWN** for every ordinary family: `gaussian1d_acquisition`. Standalone API results keep their own source, recipe and runtime; they do not fill these cells.

Each row keeps its own configuration, source, representation and measured scope. Required denominators stay fixed; diagnostic and ordinary qualification scores remain separate.

| Model/configuration | Actual representation | Measured score/status and scope |
| --- | --- | --- |
| [Original PR223 Atlas FULL · LR .00425 / prior2](continuous-baseline-20261003/README.md) | Particles | **19/19 original cases PASS — verified across two source-bound runs** · historical **19/19 PASS** · full original recipe/law replay · [19 original goal GIFs](continuous-baseline-20261003/README.md)<br>Policy-selected; averaging enabled; learned output kernel (init .029) · native/moving seed1234, portability seed0 · [fresh retest](pr223-original-full-retest-stopped17-20261004/README.md): 16/19 PASS · 0 FAIL · 1 INVALID, 2 NOT_RUN · overall INCOMPLETE · source `2068a661`/`00cadbfd` · [final charge 3210.214214 / 10800 s](pr223-original-full-retest-stopped17-20261004/FINAL_COST.json)<br>[Native3 continuation](pr223-native3-first-invalid-20261004/README.md): 0/3 PASS · 0 FAIL · 3 unavailable · 1 INVALID, 2 NOT_RUN · overall INCOMPLETE · source `0fa92c5b`/`b2d0e980` · [inclusive campaign charge 3227.941695 / 10800 s](pr223-native3-first-invalid-20261004/FINAL_COST.json); prior cases and cumulative metadata counted once<br>[Repaired native3 retest](pr223-native3-repaired-20261004/README.md): 3/3 PASS · 0 FAIL · 0 unavailable · overall PASS · source `2557cfc1`/`a2088619` · [inclusive campaign charge 5778.858768 / 10800 s](pr223-native3-repaired-20261004/FINAL_COST.json); original19 and pretraining-invalid case debits separately once, SAME cumulative metadata once; separate source, no pooled19 credit; no current-26/default credit |
| [Atlas C6 CHANGED-rate / noise-OFF · LR .0053125 / prior1.5 · source `9563dea5`](atlas-current-gpu-diagnostics-native-v2-20261003/README.md) | Particles | **7/26 PASS** · FAIL 11 · BLOCKED 8 <br>Selected-policy diagnostic/reference · seed0 · [source and 18 goal GIFs](atlas-current-gpu-diagnostics-native-v2-20261003/README.md) |
| [`atlas_conditional` · C6 · `ff94453b`](atlas-named-gpu-diagnostics-native-v3-20261003/README.md) | Particles (conditional clouds / role banks) | **4/26 PASS** · NOT_RUN 22<br>Named-family diagnostic · goal GIFs: [mid_scale_identity](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/mid_scale_identity_conditional_policy_selected_cloud_v1.gif) · [residual_student](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/residual_student_conditional_policy_selected_cloud_v1.gif) · [trajectory](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/trajectory_conditional_policy_selected_cloud_v1.gif) · [unipolar](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/unipolar_conditional_policy_selected_cloud_v1.gif) |
| [`atlas_ae_routed` · C6 · `ff94453b`](atlas-named-gpu-diagnostics-native-v3-20261003/README.md) | MoG (fixed σ .025; routed AE) | **1/26 PASS** · NOT_RUN 25<br>Named-family diagnostic · goal GIFs: [ae_gan_hold](atlas-named-gpu-diagnostics-native-v3-20261003/gifs/ae_gan_hold_ae_routed_policy_v1.gif) |
| [`atlas_routed` · C6 · `fb7acc77`](atlas-named-gpu-diagnostics-native-v4b-20261004/README.md) | Particles (shared / slot parameter bank) | **1/26 PASS** · NOT_RUN 25<br>Named-family diagnostic · goal GIFs: [unused_token_hold](atlas-named-gpu-diagnostics-native-v4b-20261004/gifs/unused_token_hold_routed_policy_selected_cloud_v1.gif) |
| [`atlas_multibank` · C6 · `fb7acc77`](atlas-named-gpu-diagnostics-native-v4b-20261004/README.md) | Particles (two routed clouds) | **1/26 PASS** · NOT_RUN 25<br>Named-family diagnostic · goal GIFs: [cover_leftover](atlas-named-gpu-diagnostics-native-v4b-20261004/gifs/cover_leftover_multibank_policy_v1.gif) |
| [`atlas_word_joint_min11` · C6 · `fb7acc77`](atlas-word-retained-context-20261004/README.md) | Particles (N11 joint cloud; free encoder) | **INVALID 1** · NOT_RUN 25 · numerical gate UNAVAILABLE<br>Named-family diagnostic · retained INVALID illustration: [five_word_joint_acquisition](atlas-word-retained-context-20261004/word-retained-goal.gif) |
| [half_base LR .00265625 / prior1.5 / D1 · source `f9f7ed9d`](word-half-base-20261004/README.md) | Particles (N11 joint cloud; free encoder) | **0/26 PASS** · FAIL 1 · NOT_RUN 25 · COMPLETE<br>Final quality 0.546875 · modes 2/5 · mass TV 0.622266 · all-five exact 0<br>[accepted 20,001-update goal GIF](word-half-base-20261004/media/goal.gif) |
| [K3P · 1-D Gaussian: histogram matching](../toy_audit/api_contract/gaussian1d/README.md) | MoG (fixed σ 0.025) | **0/1 PASS · FAIL 1** · standalone API · cpu · 1,000 updates · terminal 3/5 · KS 0.05674 / ≤ 0.05<br>[Actual-training histogram GIF](../toy_audit/api_contract/gaussian1d/goal.gif) · no ordinary tier credit |
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

Before shipping defaults, one unchanged family/configuration tuple must satisfy all 27 required 6/19/2 gates with compatible source, runtime and serving evidence, followed by the separate calibration and robustness requirements. Named diagnostic rows do not pool passing cells across families or sources and grant no ordinary tier, default or speed credit.

Atlas's historical study has **19/19 original PASS**. The later C6 broad hold has **2/2 hold FAIL**, with six other domains UNKNOWN per family. See [completed source-bound studies](#completed-source-bound-studies) for the exact protocols and original goal GIFs. These separate results do not fill ordinary qualification.

Recorded results remain bound to their actual recipes, priors, initialization, budgets, sampling laws and hardware. They do not pool qualification across sources or qualify the latest checkout. Selection never combines passing tasks or tiers from different configurations. A failed best-observed configuration is not a qualified winner. Search qualification covers only its declared tuning tiers; later-tier outcomes are reported separately. The screening profile remains provisional and does not confer calibrated robustness or public-default adoption.

BLOCKED means execution was incompatible or a prerequisite was unavailable. NOT RUN and UNKNOWN mean unexecuted or unmeasured; FAIL records an executed gate failure. Failed or blocked prerequisites stop later work; required denominators stay fixed. The Atlas-history and C6-hold links retain their separate protocols and supply no current tier credit.

[All configuration alternatives, trials, task statuses and exact bindings](technique-inventory.json) · [Evidence and archived publication identities](technique-evidence/manifest.json)

Regenerate this same leaderboard from committed evidence, without training or raw-log hydration:

```sh
python reports/forge/regenerate_technique_inventory.py --refresh-publication
```

For ordinary Forge qualification, use `--source-commit <executed-commit>` to independently regrade hydrated original receipts and update this leaderboard. Reusable candidates use the same complete task ladder. Historical task-only diagnostics remain motivation and reproduction evidence. Source snapshots are provenance, not additional leaderboards.

Publication input digest `3f2c4c4a914c6559d8e989836a412633f695295771c29a4a7bf6d6bd5889eb57`.

The [whole-family repair readout](family-wide-word-repairs/README.md) records the ordinary candidate attempts and bounded global configuration search. Their complete rows remain unranked alternatives below and in the companion JSON; a failed replacement does not make its historical incumbent a qualified standard.

Earlier view policies retain their exact numerical snapshots and receipt proofs in the companion JSON. Their outcomes do not fill current requirements:

- `discriminator_stability` revision 2: recorded denominators 3/19/2; 5 source cohorts.

Unranked alternatives retain their original outcomes and exact source/runtime bindings:

- `atlas` (atlas), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p-bcap-matched-v1` (bcap), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `e22` (e22), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `release07-gan-v3-mog-v1` (release07-gan-v3-mog), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `release07-gan-v3-mog-v1` (release07-gan-v3-mog), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `release07-gan-v3-cloud-v1` (release07-gan-v3-cloud), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
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
- `k3p--01eca360219ea5225a6e30a31800c7bbe70ca08c3800e259c17a67f0ea528ab5` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--0ababb7bec701731ff473ae925c5aaebc92aec05056db9f0d01e881aa1a451b5` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--15b66aa393dfb7f3e75c6241b985a5c19e6c3b6ee4122df3abe170cd5a5521b1` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--18af773f5d3cc22f96b26c7e5751bc73c058408043bd1f2cb30ff0b2ccc611a9` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--1c04efed544cabaab50ebd306711bfe83db8615beede38114fe5d3a6f337a9d0` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--1d5fb8e62263de7b98df3fe6ae33c669bac9dfd5207651734e2ae38fe2b9fc8c` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--209c3438a7f0f4576d1dc07746ed2c202ab382ba1c90f1ac58c559ce466d4c11` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--239cdcf7cbe2739c63a0abc5cd27d38adfe73f1e09103ed5e1be2c0df1073176` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--2f125618edc4e05825b879fd34cccf20f4cb4f3c2ac43001c262f8ba580c3201` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--370894f9284090fc9771145a869f2d7fa617283c3736c8e461a47553c9668ce6` (k3p), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--370894f9284090fc9771145a869f2d7fa617283c3736c8e461a47553c9668ce6` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--607bd2f2bf44f748b7acddcca0456c4cb92cda1371c9f1c5e93672f5d782046e` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--74219b54461b106a1f780a72ce43035f47611df4760a34a448292e6a40597503` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--8072e0db5de655465437a1de7db530340beb273340f909fcfc0e8c731e92bd6f` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--810e39e941cafd725e3a103f019a6b3f2d616c5b78256e0d7a20d7e61f415fae` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--862c7c23a730daa7bae2c2076b47e192483f4731012e3faa97e9881c708edb0f` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--8721c414567d7c056864f46e71b53c73b0848a91d6a384c69e4804c4d61cf696` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--8951909fddc06a4529927659598aa380f0368268ff1fd71115b0003eee5620b4` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--8c14d4a6a059d0954eb79a8d100d1c4de0eb13c039f49a8ac6f8f66ca544b8df` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--8e32578b69ab9a2d42d46002e22ee6bb3efb23fffee381067d8eacdc1e6f9b6d` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--922f620215f1e76351aeca2f68b79ddefeb7a44c1f824d9fdb867a27e5a03ade` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--9c6f1ae673fc7b51dbdeac0a826695ceb94c869ba886e750177b73c6727af687` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--9f4bc2973d95cf1545e009ec044ed582a318e6d869d62f4d99f6eee529d1d93d` (k3p), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--9f4bc2973d95cf1545e009ec044ed582a318e6d869d62f4d99f6eee529d1d93d` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--a338cc158d705aada992b1fdec6ba1d24b458371d5f715190496099e2f7d918d` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--a4b78c36d9faacd837e079c3fd3b9623fc141f42737e03312053a86c560abb8a` (k3p), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--a5ef4da177f58c3bd3aa6f3af5a6ffb6145889635d2a303ecb7534711b698993` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--a9f37c77c97987a31d1ae35ae01de536de29d59914640ebcdc06bfebaf1ceadc` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--adbcdd335837dcd7c340e2fe7f4b98ade162bf5a3e8f1e4f78e64e316d7a47b7` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--b188804c4f52e27a2c1524f44c7e6d10b637f36846f71bdf17efada7180448e2` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--b1bac237024d49da20182f1a15ce2298c18475fe6c7a286a569a0f145d6936f7` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--bf75899e6d6805fc9779dbbb56e2ed937ffcb838f70e7b64ca422e48cea37d82` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--c4458f4c26996e5e889dfabcae38099962017f164e1bbc1085d5f8084fcb39d7` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--caa08ebf04c2b95ed480dd9c23e39a8624e39ae27b2b80a9af1ac74f13a86b9f` (k3p), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--d9270948218dc559ecd52946f251ab755bc201f72aae5bcca8a11b7439965510` (k3p), source `730b77d2b4e1`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--e99f7210e7272e65ba80cd6793b63a073862cff90ce2de25ea1ea4726bc7442f` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--fcf67be558ec42242657e9b103c7698896a1021a7fc88325348058e7800bdfaa` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p--fd2c44a2f5c0e6194db14cc1c464a17e43dc2aa4340046bf76b47b3ad66019bf` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p-global-input-noise-v1` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p-global-repair-v1` (k3p), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `k3p-global-repair-v1` (k3p), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--4496b859e066d7a4b6279f2d34fbd58d083e0f18c2a22ade5ba12c964d01a7ca` (ka2), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--4496b859e066d7a4b6279f2d34fbd58d083e0f18c2a22ade5ba12c964d01a7ca` (ka2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--6183426d7810463d5461ad936a42c67a38516fce8b32b23065da215fed8d145d` (ka2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--8d6bb101262f49dcef37111b37ab781ca09652b47ad95e783ccea4dfbec680b4` (ka2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--8d6bb101262f49dcef37111b37ab781ca09652b47ad95e783ccea4dfbec680b4` (ka2), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--d02d7932342e1fa1db0a7a91537394297601e55516882d20ec033b4992bb527f` (ka2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2--e425fc18e432bbec6c1a1e2fc0b2f8024c7ad104fd5acf8908b6cfbbb5a3da03` (ka2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2-global-repair-v1` (ka2), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `ka2-global-repair-v1` (ka2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--083deac11d390b7895cca37396e9caf2adf5706d4ce560ae70a089e305ab3b4b` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--14eeafe7d865bdc5f6886ae8db64afd22bac3fd20a82e5ddb59188ab4fa30cc6` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--151d3b970d66ccb2956e2b914d518a0e5652e7164dccf000a29bbc82831b2d46` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--4363c95b3f6f2739729c7a104bd080a427e3efbef1f915ce14c191009bfa3a27` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--724b3d52fbb2172a985d1bf29890b2ef9ba2598c9f1724d93a9afa7c872cbdad` (r1r2), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--724b3d52fbb2172a985d1bf29890b2ef9ba2598c9f1724d93a9afa7c872cbdad` (r1r2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--7728cf93188b9fe122536244667789941da52d435cfb0a3ae42b90db200d7595` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--9481826257709128f1c864bac2950906d0ad940b8f75fbfa3d47317e7b26007d` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--abf642c42c5346ad096c29202e4716db535c393c113478552133c1c22761ddbd` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--b58a087cabab6b43997ec233416294cfc2786bab59026c07a1a4f2565970f5ff` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--dad6fdf50034381c106102df008e26b7687b04918cab85c13721e2a2dbbbe743` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--e033ba04e5b9609b896a084a182d785394869fcad19d4a15cffe86e9a973f3fe` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--f398f6ea70a2d101c520a1a01db90ea968bbb16736e6cec7ffc887e1dd7e241a` (r1r2), source `7306340bac0a`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--f78d5d603be1611025389c4ca0f7f485f3d2caa0a25fb6bc38b7c96da3509a3e` (r1r2), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2--f78d5d603be1611025389c4ca0f7f485f3d2caa0a25fb6bc38b7c96da3509a3e` (r1r2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2-global-repair-v1` (r1r2), source `2aea8c53d187`; recorded tier 0. Full evidence is in the companion JSON.
- `r1r2-global-repair-v1` (r1r2), source `964186eab95b`; recorded tier 0. Full evidence is in the companion JSON.
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
