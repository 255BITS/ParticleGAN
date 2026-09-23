# Fixed-formulation data toys: discriminator architecture search

Only discriminator width/depth/Fourier representation changes. Seed0, CPU thread1, original data/G/settings/budgets/gates. Six valid data toys; intentionally weak architecture/data diagnostics and forced optimizer stresses are excluded. EMA never qualifies a live failure. Missing cases remain unmeasured, not passes.

| Discriminator | Unequal mass | Unequal width | Overlap | Broad | Anisotropic | Spiral | Sustained / measured |
|---|---|---|---|---|---|---|---:|
| original_d64_l2_f2 | FAIL (0/24) | FAIL (0/24) | FAIL (4/24) | PASS (19/24) | PASS (15/24) | PASS (23/24) | 3/6 |
| existing_d64_l2_f4 | FAIL (0/24) | FAIL (0/24) | FAIL (4/24) | unmeasured | unmeasured | unmeasured | 0/3 |
| d128_l2_f2 | FAIL (0/24) | FAIL (0/24) | FAIL (2/24) | unmeasured | unmeasured | unmeasured | 0/3 |
| d256_l2_f2 | FAIL (0/24) | FAIL (0/24) | FAIL (2/24) | unmeasured | unmeasured | unmeasured | 0/3 |
| d64_l3_f2 | FAIL (0/24) | FAIL (0/24) | FAIL (3/24) | unmeasured | unmeasured | unmeasured | 0/3 |
| d128_l3_f2 | FAIL (0/24) | FAIL (2/24) | FAIL (1/24) | unmeasured | unmeasured | unmeasured | 0/3 |
| d64_l2_f3 | FAIL (0/24) | FAIL (0/24) | FAIL (1/24) | unmeasured | unmeasured | unmeasured | 0/3 |
| d128_l2_f3 | FAIL (0/24) | FAIL (0/24) | FAIL (1/24) | unmeasured | unmeasured | unmeasured | 0/3 |
| d128_l3_f3 | FAIL (0/24) | FAIL (0/24) | PASS (5/24) | PASS (21/24) | PASS (13/24) | PASS (24/24) | 4/6 |
| d256_l3_f3 | FAIL (0/24) | FAIL (0/24) | PASS (7/24) | PASS (23/24) | FAIL (0/24) | PASS (23/24) | 3/6 |
| d64_l2_f5 | FAIL (0/24) | FAIL (0/24) | PASS (10/24) | unmeasured | unmeasured | unmeasured | 1/3 |
| d128_l2_f4 | FAIL (0/24) | FAIL (0/24) | FAIL (1/24) | unmeasured | unmeasured | unmeasured | 0/3 |
| d128_l3_f4 | FAIL (0/24) | FAIL (0/24) | PASS (11/24) | PASS (22/24) | PASS (13/24) | PASS (22/24) | 4/6 |
| d64_l4_f3 | FAIL (0/24) | FAIL (0/24) | FAIL (4/24) | unmeasured | unmeasured | unmeasured | 0/3 |

Cells show sustained verdict and final passing suffix. A complete24-point curve and at least5 final passing observations are required; final bounds alone do not qualify. Baseline/Fourier4 rows are explicitly reused historical controls. The spiral retains its original1600-step budget; other valid vector cases use1200. No training extensions.

[Frozen plan](plan.json.gz), [runtime/source hashes](protocol.json.gz), [all results](index.json.gz), [exact driver](run.py), [tailable log](progress.log).

## Completed validation

All45 new episodes completed24 observations without errors. Source/episode hashes, unchanged thresholds/settings/budgets, and every sustained verdict were independently rechecked.

| Fully evaluated D architecture | D parameters | Sustained /6 | Final /6 | Episode seconds |
|---|---:|---:|---:|---:|
| d128_l3_f3 | 35073 | 4/6 | 4/6 | 72.53 |
| d256_l3_f3 | 135681 | 3/6 | 3/6 | 117.77 |
| d128_l3_f4 | 35585 | 4/6 | 4/6 | 66.02 |

Selection used only the three hard valid data toys, followed by full-six regression checks of the three best predeclared screen scores. Each row is one discriminator setting shared across tasks. Screening failures and incomplete candidates remain visible. This is architecture search on inspected development data, not a new formulation, a seed study, or held-out generalization evidence. Wall times include evaluations under concurrent machine load; they do not establish a speedup.

[Machine leaderboard](leaderboard.json.gz), [selected cards and screen scores](selected.json.gz).
