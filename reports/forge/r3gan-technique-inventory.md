# Forge technique inventory by tier

This report grades the **frozen source cohort `77a373648e3a7a5f1923015fa2097ec9e0ec877f`** reconstructed from Git and verified against original receipt manifests. Its outcomes do not qualify the latest checkout.

View `discriminator_stability` revision 2; requested device `cuda`. Calibration remains `provisional`.

Each cell is **passes / full required total** for that exact recorded cohort. Rows follow Forge's attained-tier order; no aggregate quality or speed ranking is declared.

| Technique | Cohort / exact revision | Compute | Tier 1 | Tier 2 | Tier 3 | Qualified tier | Other outcomes | Paid seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| [Atlas](../../configs/forge/ideas/atlas.json) | ab642835c6e3 / 472adad5135c | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 24 | unknown |
| [E22](../../configs/forge/ideas/e22.json) | 23a02a2da5e9 / 3f622df5abce | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 24 | unknown |
| [K3P without critic penalty](../../configs/forge/ideas/forge-no-critic-penalty.json) | a04487f4ea2a / 9126fee666e0 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [K3P without critic anchor](../../configs/forge/ideas/forge-onboarding-anchor-ablation.json) | c8a26e8f4543 / b77a0971be66 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [K3P](../../configs/forge/ideas/k3p.json) | 0a641cbb5be1 / 8d7a61d341b1 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [K3P without A2](../../configs/forge/ideas/k3p-a2-off-native-diagnostic.json) | 9a9d560ee713 / 40d358cc3220 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [BCap (matched K3P recipe)](../../configs/forge/ideas/k3p-bcap-matched-v1.json) | d05f6c372a0f / 98646f581b60 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [K3P without training output noise](../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json) | 75b2e8874a3e / 55aa44474188 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [R1/R2 standard penalty (matched K3P recipe)](../../configs/forge/ideas/k3p-r1r2-matched-v1.json) | 63a0cb06d04b / 87ec652e2c04 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [KA2](../../configs/forge/ideas/ka2.json) | ea7af0cea45b / 7ed7d9c898b9 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [R3GAN Stacked-MNIST recipe (toy-host adaptation)](../../configs/forge/ideas/r3gan-stacked-training-toy-v1.json) | 2114b25c4323 / 724a67246849 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 23 | 5.347 |
| [GAN v3 release 0.7 (cloud)](../../configs/forge/ideas/release07-gan-v3-cloud-v1.json) | 815dd846abde / fa7282acdfc1 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 21, UNKNOWN 3 | unknown |
| [GAN v3 release 0.7 (MoG adaptation)](../../configs/forge/ideas/release07-gan-v3-mog-v1.json) | 736dc3cdf284 / 06d7826e0ef3 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 21, UNKNOWN 3 | unknown |

UNKNOWN means missing or unrun evidence. A failed or blocked prerequisite stops later-tier spending while every declared task stays in its denominator. Nonrequired diagnostics never fill required cells.

Recorded rows bind the resolved recipe, prior, initialization, full budget, sampling law, source and runtime. Different clean/noisy sampling and hardware cohorts stay separate.

## Measured cells and first blockers

The earliest required blocker is shown for each cohort; full task statuses and reasons are in JSON. A smoke failure describes this frozen profile and recipe, without measuring unrun quality tiers.

- **Atlas / ab642835c6e3:** `two_pole` BLOCKED — missing capability live_sampling; two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation.
- **E22 / 23a02a2da5e9:** `two_pole` BLOCKED — missing capability live_sampling; two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation.
- **R3GAN Stacked-MNIST recipe (toy-host adaptation) / 2114b25c4323:** `two_pole` FAIL — recomputed complete live curve and terminal suffix. [receipt `ea66e44192c0`](../../reports/forge/technique-receipts/ea66e44192c048eaa97f8595c5e55508.json)
- **GAN v3 release 0.7 (cloud) / 815dd846abde:** `two_pole` BLOCKED — two_pole: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; two_pole: recipe override 'batch_size' is owned by the frozen host; revise its task specification; two_pole: recipe override 'conditioning' is owned by the frozen host; revise its task specification; two_pole: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; two_pole: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; two_pole: recipe override 'model' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_classes' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_particles' is owned by the frozen host; revise its task specification; two_pole: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; two_pole: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; two_pole: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; two_pole: recipe override 'total_steps' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'z_dim' is owned by the frozen host; revise its task specification.
- **GAN v3 release 0.7 (MoG adaptation) / 736dc3cdf284:** `two_pole` BLOCKED — two_pole: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; two_pole: recipe override 'batch_size' is owned by the frozen host; revise its task specification; two_pole: recipe override 'conditioning' is owned by the frozen host; revise its task specification; two_pole: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; two_pole: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; two_pole: recipe override 'model' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_classes' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_particles' is owned by the frozen host; revise its task specification; two_pole: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; two_pole: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; two_pole: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; two_pole: recipe override 'total_steps' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'z_dim' is owned by the frozen host; revise its task specification.

Pinned, calibration diagnostic and historical evidence remains separate and unranked: pinned 10 cohorts; calibration_diagnostic 22 cohorts; historical 143 cohorts. Overlapping historical summaries are not independent trials or a combined cost total. [Historical memory and original evidence](../../reports/forge/EXPERIMENT_MEMORY.md).

[Full task statuses, exact scientific bindings, archived cohorts and provenance](r3gan-technique-inventory.json).

Regenerate after new Forge receipts or declarations with:

```sh
python reports/forge/regenerate_technique_inventory.py --root . --goal discriminator_stability --device cuda --output-prefix reports/forge/r3gan-technique-inventory --source-commit 77a373648e3a7a5f1923015fa2097ec9e0ec877f
```

Reducer `forge-technique-board-v1`; input digest `0b2fb7bed46f050ccb3a9ab416c6942ffe5b6e60f3ea43145b96c19d7881967c`. Report generation launches no training.

Published receipt summaries retain final metrics, gate outcomes and original file hashes. They are display artifacts and supply no qualification input. Hydrate byte-exact original request/evidence/result receipts from the artifact archive before a full independent regrade.
