# Forge technique inventory: recorded source cohorts

Each cell is **passes / full required total** in its row's recorded source and runtime. The original inventory and appended training baseline keep their separate evidence identities.

| Technique | Recorded source | Exact revision / cohort | Compute | Tier 1 | Tier 2 | Tier 3 | Recorded tier | Other outcomes | Paid seconds |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| BCap (matched K3P recipe) | [`bb31b3f77bee`](technique-inventory.md) | 18096544dede / 341957c4bd22 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 3/3 | 5/19 | 0/2 | 1 | FAIL 1, UNKNOWN 15 | 86.357 |
| Atlas | [`bb31b3f77bee`](technique-inventory.md) | 74ba5a828ce9 / 0ef26a812a67 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 24 | unknown |
| E22 | [`bb31b3f77bee`](technique-inventory.md) | f95c55691e55 / a8f36adca210 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 24 | unknown |
| K3P without critic penalty | [`bb31b3f77bee`](technique-inventory.md) | aa9e200be9f5 / 09c371803da6 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 23 | 7.827 |
| K3P without critic anchor | [`bb31b3f77bee`](technique-inventory.md) | e356a8bd7091 / 4bc2bdaad0bc | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 23 | 5.449 |
| K3P | [`bb31b3f77bee`](technique-inventory.md) | 2a97b74e933c / c5486fad1525 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 23 | 5.581 |
| K3P without A2 | [`bb31b3f77bee`](technique-inventory.md) | b5ae40d1428d / 1444cceca091 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 23 | 5.426 |
| K3P without training output noise | [`bb31b3f77bee`](technique-inventory.md) | 7bb7527f2fa6 / 1fbda76d4718 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 23 | 5.876 |
| R1/R2 standard penalty (matched K3P recipe) | [`bb31b3f77bee`](technique-inventory.md) | 0a2b2de1a8a7 / a937f51d8ee7 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 23 | 5.812 |
| KA2 | [`bb31b3f77bee`](technique-inventory.md) | 2a6cb4b3f19e / 378e64b07dbc | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 23 | 5.46 |
| GAN v3 release 0.7 (cloud) | [`bb31b3f77bee`](technique-inventory.md) | 52487891a5ae / fc24f561606f | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 21, UNKNOWN 3 | unknown |
| GAN v3 release 0.7 (MoG adaptation) | [`bb31b3f77bee`](technique-inventory.md) | f4dd6011a5d3 / cb45990f0d23 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | BLOCKED 21, UNKNOWN 3 | unknown |
| R3GAN Stacked-MNIST recipe (toy-host adaptation) | [`8bfb0b738e28`](r3gan-technique-inventory.md) | 724a67246849 / 2114b25c4323 | cuda / AMD Ryzen 9 5900X 12-Core Processor, NVIDIA RTX A6000 | 0/3 | 0/19 | 0/2 | 0 | FAIL 1, UNKNOWN 23 | 5.347 |

Recorded tiers describe each independently graded publication. This display does not pool passes, rank across source cohorts, or qualify the latest checkout. Recipes, priors, initialization, budgets, clean/noisy sampling and hardware remain bound to their original rows.

UNKNOWN means unmeasured or unrun evidence. Failed or blocked prerequisite gates stop further work; every declared task remains in its tier denominator.

[Original frozen inventory](technique-inventory.md) · [Full current inventory, including all technique denominator rows](r3gan-technique-inventory.md) · [Exact row bindings and publication hashes](technique-inventory-expanded.json)

Regenerate this display from the committed publications without launching training or requiring raw receipt hydration:

```sh
python reports/forge/regenerate_technique_inventory.py --root . --compose-original reports/forge/technique-inventory.json --compose-current reports/forge/r3gan-technique-inventory.json --append-candidate r3gan-stacked-training-toy-v1 --output-prefix reports/forge/technique-inventory-expanded
```

Publication input digest `06a4df19c880f422545a2e32bba329e3ecb186ff4240500b40e9b1a92965cfab`. Full independent regrading uses each linked publication's own regeneration command and byte-exact original receipts.
