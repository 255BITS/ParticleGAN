# Current Forge trainer-family leaderboard

Atlas's historical study has **19/19 original PASS**. The later C6 broad hold has **2/2 hold FAIL**, with six other domains UNKNOWN per family. See [completed source-bound studies](#completed-source-bound-studies) for the exact protocols and original goal GIFs. These separate results do not fill the ordinary qualification cells below.

Each cell is **passes / full required total** from one complete selected configuration. Each trainer family and runtime has one row; its alternatives remain recorded separately. Expand the configuration details below for selection, provenance, other outcomes and cost.

| Trainer family / runtime | Tier 1 | Tier 2 | Tier 3 | Recorded tier |
| --- | ---: | ---: | ---: | ---: |
| Atlas<br>cuda | 0/5 | 0/19 | 0/2 | 0 |
| BCap<br>cuda | 3/5 | 0/19 | 0/2 | 0 |
| E22<br>cuda | 0/5 | 0/19 | 0/2 | 0 |
| GAN v3 release 0.7 (MoG)<br>cuda | 3/5 | 0/19 | 0/2 | 0 |
| GAN v3 release 0.7 (cloud)<br>cuda | 0/5 | 0/19 | 0/2 | 0 |
| K3P<br>cuda | 4/5 | 0/19 | 0/2 | 0 |
| K3P without A2<br>cuda | 0/5 | 0/19 | 0/2 | 0 |
| K3P without critic anchor<br>cuda | 0/5 | 0/19 | 0/2 | 0 |
| K3P without critic penalty<br>cuda | 0/5 | 0/19 | 0/2 | 0 |
| K3P without training output noise<br>cuda | 0/5 | 0/19 | 0/2 | 0 |
| KA2<br>cuda | 4/5 | 0/19 | 0/2 | 0 |
| R1/R2<br>cuda | 3/5 | 0/19 | 0/2 | 0 |
| five-word-joint-ka2-v1<br>cuda | 0/5 | 0/19 | 0/2 | 0 |

<details>
<summary>Selected configurations and provenance</summary>

### Atlas (cuda)

- **Selected configuration:** [atlas](../../configs/forge/ideas/atlas.json)
- **Selection:** canonical fallback; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 506319d4621e / e198762f715a
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** BLOCKED 26
- **Paid seconds:** unknown

### BCap (cuda)

- **Selected configuration:** [bcap · 08689a73c551](../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json)
- **Selection:** best observed; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 8d2031fe492f / bf859ff60898
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 22
- **Paid seconds:** 30.559

### E22 (cuda)

- **Selected configuration:** [e22](../../configs/forge/ideas/e22.json)
- **Selection:** canonical fallback; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** cf5d3fdcbb1d / cab0cdc81bdb
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** BLOCKED 26
- **Paid seconds:** unknown

### GAN v3 release 0.7 (MoG) (cuda)

- **Selected configuration:** [release07-gan-v3-mog · 1e266b5a2986](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json)
- **Selection:** best observed; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 47630167ae31 / f99998f9b0fa
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 22
- **Paid seconds:** 28.757

### GAN v3 release 0.7 (cloud) (cuda)

- **Selected configuration:** [release07-gan-v3-cloud-v1](../../configs/forge/ideas/release07-gan-v3-cloud-v1.json)
- **Selection:** canonical fallback; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 95e403581ebf / 6cb4d268b94e
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** BLOCKED 26
- **Paid seconds:** unknown

### K3P (cuda)

- **Selected configuration:** [k3p · 0b37e98a01e3](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json)
- **Selection:** best observed; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 412dbcba9246 / 7ebaeb278a75
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 21
- **Paid seconds:** 210.977

### K3P without A2 (cuda)

- **Selected configuration:** [k3p-a2-off-native-diagnostic](../../configs/forge/ideas/k3p-a2-off-native-diagnostic.json)
- **Selection:** canonical fallback; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** b51787a935f0 / 31b4d36cc007
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 25
- **Paid seconds:** 5.318

### K3P without critic anchor (cuda)

- **Selected configuration:** [forge-onboarding-anchor-ablation](../../configs/forge/ideas/forge-onboarding-anchor-ablation.json)
- **Selection:** canonical fallback; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 0fa611b1ebcc / 08a552799580
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 25
- **Paid seconds:** 5.419

### K3P without critic penalty (cuda)

- **Selected configuration:** [forge-no-critic-penalty](../../configs/forge/ideas/forge-no-critic-penalty.json)
- **Selection:** canonical fallback; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** ea1e0bdda2f8 / 866cef3392e0
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 25
- **Paid seconds:** 5.517

### K3P without training output noise (cuda)

- **Selected configuration:** [k3p-no-output-noise-diagnostic](../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json)
- **Selection:** canonical fallback; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** b66c72f805f4 / edc93bce30da
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 25
- **Paid seconds:** 5.657

### KA2 (cuda)

- **Selected configuration:** [ka2 · 093c6f2bd417](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json)
- **Selection:** best observed; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** d5ebfc250aac / 7be3028bd4fc
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 21
- **Paid seconds:** 222.721

### R1/R2 (cuda)

- **Selected configuration:** [r1r2 · 302b6baa44f6](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json)
- **Selection:** best observed; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 2727e20c5a10 / 4ff453a8a3ed
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 22
- **Paid seconds:** 27.627

### five-word-joint-ka2-v1 (cuda)

- **Selected configuration:** [five-word-joint-ka2-v1](../../configs/forge/ideas/five-word-joint-ka2-v1.json)
- **Selection:** canonical fallback; no qualified winner
- **Evidence source:** [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json)
- **Exact revision / cohort:** 1b10bb148f0e / 471a74f59360
- **Compute:** cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000
- **Other outcomes:** FAIL 1, UNKNOWN 25
- **Paid seconds:** 5.637

</details>

Recorded results remain bound to their actual recipes, priors, initialization, budgets, sampling laws and hardware. They do not pool qualification across sources or qualify the latest checkout. Selection never combines passing tasks or tiers from different configurations. A failed best-observed configuration is not a qualified winner. Search qualification covers only its declared tuning tiers; later-tier outcomes are reported separately. The screening profile remains provisional and does not confer calibrated robustness or public-default adoption.

UNKNOWN means unmeasured. Failed or blocked prerequisites stop later work; required denominators stay fixed.

[All configuration alternatives, trials, task statuses and exact bindings](technique-inventory.json) · [Evidence and archived publication identities](technique-evidence/manifest.json)

Regenerate this same leaderboard from committed evidence, without training or raw-log hydration:

```sh
python reports/forge/regenerate_technique_inventory.py
```

For ordinary Forge qualification, use `--source-commit <executed-commit>` to independently regrade hydrated original receipts and update this leaderboard. Task-only word diagnostics are read from their committed recipe selections and compact receipts. Source snapshots are provenance, not additional leaderboards.

Publication input digest `ee77c534ac48d935c41d037df2b72cddc8b5907164c9f99e0c02db48d70fea4d`.

## Five-word joint task diagnostics

The bounded word-task study recommends the exact recipes below. These are **task-only diagnostics**: they do not replace the configurations above, fill Tier 1 cells, or qualify defaults. Each result retains its executed source, recipe, prior, initialization, budget and clean/live sampling law.

[Exact task-only recipes](../../configs/forge/selections/word-joint-task-v1.json) · [Root cause, all 18 runs and training GIFs](word-root-cause/README.md)

The passing suffix is the terminal passing observations / total observations. Min. inverse P is the minimum reconstruction token probability.

| Family | Result | Passing suffix | TV | Min. inverse P |
| --- | --- | ---: | ---: | ---: |
| K3P | PASS | 24/24 | 0.024219 | 0.997103 |
| KA2 | PASS | 21/24 | 0.024219 | 0.999942 |
| R1/R2 | PASS | 22/24 | 0.024219 | 0.996137 |

<details>
<summary>Diagnostic recipes, modes, quality and provenance</summary>

### K3P

- **Selected diagnostic / receipt:** [k3p-coeff170-cap1](word-root-cause/receipts/k3p-coeff170-cap1.json)
- **Modes / quality:** 5 / 1.000
- **Executed commit:** `575d485eedca`
- **Compute:** NVIDIA RTX A6000

### KA2

- **Selected diagnostic / receipt:** [ka2-slow-prior-0p1](word-root-cause/receipts/ka2-slow-prior-0p1.json)
- **Modes / quality:** 5 / 1.000
- **Executed commit:** `1a2c8d06de6b`
- **Compute:** NVIDIA RTX A6000

### R1/R2

- **Selected diagnostic / receipt:** [r1r2-mid-rate-fast-roles](word-root-cause/receipts/r1r2-mid-rate-fast-roles.json)
- **Modes / quality:** 5 / 1.000
- **Executed commit:** `ecc2dc8f48b7`
- **Compute:** NVIDIA RTX A6000

</details>

Use these recipes for this word host. Whole-configuration Tier 1 qualification still requires all required tasks under one compatible recipe and source cohort. No experiments were rerun for this publication.

Earlier view policies retain their exact numerical snapshots and receipt proofs in the companion JSON. Their outcomes do not fill current requirements:

- `discriminator_stability` revision 2: recorded denominators 3/19/2; 5 source cohorts.



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
