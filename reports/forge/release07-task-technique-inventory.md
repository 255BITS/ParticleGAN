# Forge technique inventory by tier

This report grades the **frozen source cohort `fc66d7c49259a8977cbb4478b20dad1ffcc6c1cb`** reconstructed from Git and verified against original receipt manifests. Its outcomes do not qualify the latest checkout.

View `discriminator_stability` revision 2; requested device `cuda`. Calibration remains `provisional`.

Each cell is **passes / full required total** for that exact recorded cohort. Rows follow Forge's attained-tier order; no aggregate quality or speed ranking is declared.

| Technique | Cohort / exact revision | Compute | Tier 1 | Tier 2 | Tier 3 | Qualified tier | Other outcomes | Paid seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| [Atlas](../../configs/forge/ideas/atlas.json) | 58b6ac7e299b / f6c97b44c9b6 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 24 | unknown |
| [E22](../../configs/forge/ideas/e22.json) | f5d8eafc3023 / 84974a8f73c4 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 24 | unknown |
| [K3P without critic penalty](../../configs/forge/ideas/forge-no-critic-penalty.json) | 60d8ba798dae / 8af1889a4d00 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [K3P without critic anchor](../../configs/forge/ideas/forge-onboarding-anchor-ablation.json) | c417e4492613 / 05cc06989363 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [K3P](../../configs/forge/ideas/k3p.json) | a08b6b11aa6f / c254808883fc | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [K3P without A2](../../configs/forge/ideas/k3p-a2-off-native-diagnostic.json) | 353d61bdd150 / 5019a7d5e1e9 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [BCap (matched K3P recipe)](../../configs/forge/ideas/k3p-bcap-matched-v1.json) | 03c5b521e357 / f6f043d83d4f | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [K3P without training output noise](../../configs/forge/ideas/k3p-no-output-noise-diagnostic.json) | f59b045909d4 / 28d5f8804df4 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [R1/R2 standard penalty (matched K3P recipe)](../../configs/forge/ideas/k3p-r1r2-matched-v1.json) | 5a02b88df9c0 / c3495bed49f8 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [KA2](../../configs/forge/ideas/ka2.json) | ad29088c173b / 4ae8005b559b | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [R3GAN Stacked-MNIST recipe (toy-host adaptation)](../../configs/forge/ideas/r3gan-stacked-training-toy-v1.json) | d19f6ea5b11c / f5c3d248148a | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | UNKNOWN 24 | unknown |
| [GAN v3 release 0.7 (cloud)](../../configs/forge/ideas/release07-gan-v3-cloud-v1.json) | e0b90fa02551 / 4f23ed49e40d | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 21, UNKNOWN 3 | unknown |
| [GAN v3 release 0.7 (MoG adaptation)](../../configs/forge/ideas/release07-gan-v3-mog-v1.json) | c0d330cde7dc / 9d3e04acbb27 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 21, UNKNOWN 3 | unknown |
| [GAN v3 release 0.7 (task adaptation)](../../configs/forge/ideas/release07-gan-v3-task-adapted-v1.json) | cbef159cbffb / d5341d8d1252 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 23 | 5.508 |

UNKNOWN means missing or unrun evidence. A failed or blocked prerequisite stops later-tier spending while every declared task stays in its denominator. Nonrequired diagnostics never fill required cells.

Recorded rows bind the resolved recipe, prior, initialization, full budget, sampling law, source and runtime. Different clean/noisy sampling and hardware cohorts stay separate.

## Measured cells and first blockers

The earliest required blocker is shown for each cohort; full task statuses and reasons are in JSON. A smoke failure describes this frozen profile and recipe, without measuring unrun quality tiers.

- **Atlas / 58b6ac7e299b:** `two_pole` BLOCKED — missing capability live_sampling; two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation.
- **E22 / f5d8eafc3023:** `two_pole` BLOCKED — missing capability live_sampling; two_pole: public components has no declared policy-control evidence and served-sampling contract; freeze a policy-aware task before reservation.
- **GAN v3 release 0.7 (cloud) / e0b90fa02551:** `two_pole` BLOCKED — two_pole: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; two_pole: recipe override 'batch_size' is owned by the frozen host; revise its task specification; two_pole: recipe override 'conditioning' is owned by the frozen host; revise its task specification; two_pole: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; two_pole: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; two_pole: recipe override 'model' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_classes' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_particles' is owned by the frozen host; revise its task specification; two_pole: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; two_pole: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; two_pole: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; two_pole: recipe override 'total_steps' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'z_dim' is owned by the frozen host; revise its task specification.
- **GAN v3 release 0.7 (MoG adaptation) / c0d330cde7dc:** `two_pole` BLOCKED — two_pole: recipe override 'alpha_bar' is owned by the frozen host; revise its task specification; two_pole: recipe override 'batch_size' is owned by the frozen host; revise its task specification; two_pole: recipe override 'conditioning' is owned by the frozen host; revise its task specification; two_pole: recipe override 'distance_reduction' is owned by the frozen host; revise its task specification; two_pole: recipe override 'encoder_mode' is owned by the frozen host; revise its task specification; two_pole: recipe override 'model' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_classes' is owned by the frozen host; revise its task specification; two_pole: recipe override 'num_particles' is owned by the frozen host; revise its task specification; two_pole: recipe override 'observation_sigma' is owned by the frozen host; revise its task specification; two_pole: recipe override 'prior_reg' is owned by the frozen host; revise its task specification; two_pole: recipe override 'reconstruction_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'routing_temperature' is owned by the frozen host; revise its task specification; two_pole: recipe override 'total_steps' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_target' is owned by the frozen host; revise its task specification; two_pole: recipe override 'ucd_weight' is owned by the frozen host; revise its task specification; two_pole: recipe override 'z_dim' is owned by the frozen host; revise its task specification.
- **GAN v3 release 0.7 (task adaptation) / cbef159cbffb:** `two_pole` FAIL — recomputed complete live curve and terminal suffix. [receipt `df1794ebea64`](../../reports/forge/technique-receipts/df1794ebea6444c5a10c2a953843257d.json)

Pinned, calibration diagnostic and historical evidence remains separate and unranked: pinned 11 cohorts; calibration_diagnostic 22 cohorts; historical 143 cohorts. Overlapping historical summaries are not independent trials or a combined cost total. [Historical memory and original evidence](../../reports/forge/EXPERIMENT_MEMORY.md).

[Full task statuses, exact scientific bindings, archived cohorts and provenance](release07-task-technique-inventory.json).

Regenerate after new Forge receipts or declarations with:

```sh
python reports/forge/regenerate_technique_inventory.py --root . --goal discriminator_stability --device cuda --output-prefix reports/forge/release07-task-technique-inventory --source-commit fc66d7c49259a8977cbb4478b20dad1ffcc6c1cb
```

Reducer `forge-technique-board-v1`; input digest `b1c0bf57fd0d99efd31cd98ed2c4e63c1e21821bc3e96ea539a3446aa3451924`. Report generation launches no training.

Published receipt summaries retain final metrics, gate outcomes and original file hashes. They are display artifacts and supply no qualification input. Hydrate byte-exact original request/evidence/result receipts from the artifact archive before a full independent regrade.
