# Current Forge trainer-family leaderboard

Each cell is **passes / full required total** from one complete selected configuration. Each trainer family and runtime has one row; its alternatives remain recorded separately.

| Trainer family | Selected configuration | Selection | Evidence source | Exact revision / cohort | Compute | Tier 1 | Tier 2 | Tier 3 | Recorded tier | Other outcomes | Paid seconds |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Atlas | [`atlas`](../../configs/forge/ideas/atlas.json) | canonical fallback; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | 506319d4621e / e198762f715a | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/5 | 0/19 | 0/2 | 0 | BLOCKED 26 | unknown |
| BCap | [`bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212`](../../configs/forge/configurations/bcap--08689a73c551728cc82434ac9601a06d1a9f3efa1a3999d5a3ec9e69746cc212.json) | best observed; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | 8d2031fe492f / bf859ff60898 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 3/5 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 22 | 30.559 |
| E22 | [`e22`](../../configs/forge/ideas/e22.json) | canonical fallback; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | cf5d3fdcbb1d / cab0cdc81bdb | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/5 | 0/19 | 0/2 | 0 | BLOCKED 26 | unknown |
| GAN v3 release 0.7 (MoG) | [`release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c`](../../configs/forge/configurations/release07-gan-v3-mog--1e266b5a2986ee4cb2f2fdc46437cc82982bf1cf02707eeba95743f4890e8a0c.json) | best observed; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | 47630167ae31 / f99998f9b0fa | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 3/5 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 22 | 28.757 |
| GAN v3 release 0.7 (cloud) | [`release07-gan-v3-cloud-v1`](../../configs/forge/ideas/release07-gan-v3-cloud-v1.json) | canonical fallback; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | 95e403581ebf / 6cb4d268b94e | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/5 | 0/19 | 0/2 | 0 | BLOCKED 26 | unknown |
| K3P | [`k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c`](../../configs/forge/configurations/k3p--0b37e98a01e3cc7c0f4b3325b43f9e6d569de81e4f890f20942f0fc33221305c.json) | best observed; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | 412dbcba9246 / 7ebaeb278a75 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 4/5 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 21 | 210.977 |
| K3P without A2 | [`k3p-a2-off-native-diagnostic`](../../configs/forge/ideas/k3p-a2-off-native-diagnostic.json) | canonical fallback; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | b51787a935f0 / 31b4d36cc007 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/5 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 25 | 5.318 |
| K3P without critic anchor | [`forge-onboarding-anchor-ablation`](../../configs/forge/ideas/forge-onboarding-anchor-ablation.json) | canonical fallback; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | 0fa611b1ebcc / 08a552799580 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/5 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 25 | 5.419 |
| K3P without critic penalty | [`forge-no-critic-penalty`](../../configs/forge/ideas/forge-no-critic-penalty.json) | canonical fallback; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | ea1e0bdda2f8 / 866cef3392e0 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/5 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 25 | 5.517 |
| K3P without training output noise | [`k3p-no-output-noise-diagnostic`](../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json) | canonical fallback; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | b66c72f805f4 / edc93bce30da | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/5 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 25 | 5.657 |
| KA2 | [`ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f`](../../configs/forge/configurations/ka2--093c6f2bd41768a3f99e3470d24845f6a99ebc0bfbe9c794aff871a5a466770f.json) | best observed; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | d5ebfc250aac / 7be3028bd4fc | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 4/5 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 21 | 222.721 |
| R1/R2 | [`r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f`](../../configs/forge/configurations/r1r2--302b6baa44f629bfc97270c00a91f3cd6747585897bf43e105ab8aba2a276d6f.json) | best observed; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | 2727e20c5a10 / 4ff453a8a3ed | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 3/5 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 22 | 27.627 |
| five-word-joint-ka2-v1 | [`five-word-joint-ka2-v1`](../../configs/forge/ideas/five-word-joint-ka2-v1.json) | canonical fallback; no qualified winner | [`7306340bac0a`](technique-evidence/494153533dcedb578c944134a3b16f6d368d1439685dd7689934eabf858887e3.json) | 1b10bb148f0e / 471a74f59360 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/5 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 25 | 5.637 |

Recorded results remain bound to their actual recipes, priors, initialization, budgets, sampling laws and hardware. They do not pool qualification across sources or qualify the latest checkout. Selection never combines passing tasks or tiers from different configurations. A failed best-observed configuration is not a qualified winner. Search qualification covers only its declared tuning tiers; later-tier outcomes are reported separately. The screening profile remains provisional and does not confer calibrated robustness or public-default adoption.

UNKNOWN means unmeasured. Failed or blocked prerequisites stop later work; required denominators stay fixed.

[All configuration alternatives, trials, task statuses and exact bindings](technique-inventory.json) · [Evidence and archived publication identities](technique-evidence/manifest.json)

Regenerate this same leaderboard from committed evidence, without training or raw-log hydration:

```sh
python reports/forge/regenerate_technique_inventory.py
```

For ordinary Forge qualification, use `--source-commit <executed-commit>` to independently regrade hydrated original receipts and update this leaderboard. Task-only word diagnostics are read from their committed recipe selections and compact receipts. Source snapshots are provenance, not additional leaderboards.

Publication input digest `ce68822de9c442d3a6d3330ce308a39ed98dadc46e399d22120d1027a922867f`.

## Five-word joint task diagnostics

The bounded word-task study recommends the exact recipes below. These are **task-only diagnostics**: they do not replace the configurations above, fill Tier 1 cells, or qualify defaults. Each result retains its executed source, recipe, prior, initialization, budget and clean/live sampling law.

[Exact task-only recipes](../../configs/forge/selections/word-joint-task-v1.json) · [Root cause, all 18 runs and training GIFs](word-root-cause/README.md)

| Family | Selected diagnostic / receipt | Result | Terminal passing suffix | Modes / quality | TV | Minimum inverse probability | Executed commit | Compute |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| K3P | [`k3p-coeff170-cap1`](word-root-cause/receipts/k3p-coeff170-cap1.json) | PASS | 24/24 | 5 / 1.000 | 0.024219 | 0.997103 | `575d485eedca` | NVIDIA RTX A6000 |
| KA2 | [`ka2-slow-prior-0p1`](word-root-cause/receipts/ka2-slow-prior-0p1.json) | PASS | 21/24 | 5 / 1.000 | 0.024219 | 0.999942 | `1a2c8d06de6b` | NVIDIA RTX A6000 |
| R1/R2 | [`r1r2-mid-rate-fast-roles`](word-root-cause/receipts/r1r2-mid-rate-fast-roles.json) | PASS | 22/24 | 5 / 1.000 | 0.024219 | 0.996137 | `ecc2dc8f48b7` | NVIDIA RTX A6000 |

Use these recipes for this word host. Whole-configuration Tier 1 qualification still requires all required tasks under one compatible recipe and source cohort. No experiments were rerun for this publication.

Earlier view policies retain their exact numerical snapshots and receipt proofs in the companion JSON. Their outcomes do not fill current requirements:

- `discriminator_stability` revision 2: recorded denominators 3/19/2; 5 source cohorts.
