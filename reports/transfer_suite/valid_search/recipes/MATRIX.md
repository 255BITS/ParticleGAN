# Shared vector optimizer recipes

Rp logistic, b_cap coefficient 3 / kappa 1.25, prior regularization .05, no particle L2. Every card uses the original generator/critic architecture, 256 particles, batch 128, cosine schedule and original outer budgets (1,200; spiral 1,600). All runs use seed 0, 24 fixed live observations and a final passing suffix of at least five. EMA is separate.

The 18 new coordinated optimizer cards were declared before training; no task-specific recipe overrides. The first three hard data cases screen candidates. Up to three advance to all six valid data toys. Untested cells do not imply success. These are inspected development cases; this matrix does not include required or image validation.

| Card | Broad | Rare mass | Unequal width | Anisotropic | Overlap | Spiral | Sustained / attempted | Mean final shortfall | Wall seconds |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| b999_lr075_d2_p30 | PASS (16/24) | PASS (6/24) | FAIL (0/24) | FAIL (0/24) | PASS (7/24) | PASS (24/24) | 4/6 | 0.0429 | 44.36 |
| b995_lr075_d2_p20 | PASS (19/24) | FAIL (0/24) | PASS (7/24) | PASS (11/24) | Final only (3/24) | PASS (23/24) | 4/6 | 0.0833 | 44.71 |
| b999_lr075_d3_p5 | PASS (17/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | PASS (14/24) | PASS (23/24) | 3/6 | 0.1316 | 47.21 |
| g2_b999_p20 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | PASS (21/24) | Not run | 1/3 | 0.2444 | 23.99 |
| g3_b999_lr15_p20 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | PASS (15/24) | Not run | 1/3 | 0.2444 | 18.91 |
| g2_b999_lr15_d1 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | PASS (8/24) | Not run | 1/3 | 0.2444 | 20.48 |
| b999_lr075_d3_p20 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | PASS (5/24) | Not run | 1/3 | 0.2444 | 17.82 |
| b999_lr05_p20 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | PASS (20/24) | Not run | 1/3 | 0.2542 | 22.20 |
| g2_b999_lr2_d075 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | PASS (6/24) | Not run | 1/3 | 0.3000 | 18.69 |
| mom05_lr05_d2_p20 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | Final only (2/24) | Not run | 0/3 | 0.1443 | 21.36 |
| b999_lr15_p5 | Not run | FAIL (0/24) | Final only (2/24) | Not run | FAIL (0/24) | Not run | 0/3 | 0.1453 | 17.74 |
| b999_lr075_p15 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | Final only (3/24) | Not run | 0/3 | 0.1762 | 17.63 |
| mom05_d2_p10 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | Final only (3/24) | Not run | 0/3 | 0.2114 | 21.74 |
| b99_d3_p30 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | Final only (1/24) | Not run | 0/3 | 0.2321 | 18.15 |
| b999_d3_p10 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | FAIL (0/24) | Not run | 0/3 | 0.2774 | 19.63 |
| d2_b999_lr075_d3_p20 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | Final only (4/24) | Not run | 0/3 | 0.2995 | 16.12 |
| mom09_lr03_d3_p20 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | FAIL (0/24) | Not run | 0/3 | 0.3668 | 25.62 |
| b999_d075_p20 | Not run | FAIL (0/24) | FAIL (0/24) | Not run | FAIL (0/24) | Not run | 0/3 | 0.5628 | 17.89 |

## Exact shared cards

| Card | G LR | D / G LR | Prior / G LR | Adam betas | D every | G every |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| b999_lr05_p20 | 0.0005 | 1.5 | 20.0 | [0.0, 0.999] | 1 | 1 |
| b999_lr075_p15 | 0.00075 | 1.5 | 15.0 | [0.0, 0.999] | 1 | 1 |
| b999_lr15_p5 | 0.0015 | 1.5 | 5.0 | [0.0, 0.999] | 1 | 1 |
| b999_lr075_d3_p20 | 0.00075 | 3.0 | 20.0 | [0.0, 0.999] | 1 | 1 |
| b999_d3_p10 | 0.001 | 3.0 | 10.0 | [0.0, 0.999] | 1 | 1 |
| b999_d075_p20 | 0.001 | 0.75 | 20.0 | [0.0, 0.999] | 1 | 1 |
| b99_d3_p30 | 0.001 | 3.0 | 30.0 | [0.0, 0.99] | 1 | 1 |
| b995_lr075_d2_p20 | 0.00075 | 2.0 | 20.0 | [0.0, 0.995] | 1 | 1 |
| b999_lr075_d2_p30 | 0.00075 | 2.0 | 30.0 | [0.0, 0.999] | 1 | 1 |
| mom05_lr05_d2_p20 | 0.0005 | 2.0 | 20.0 | [0.5, 0.999] | 1 | 1 |
| mom05_d2_p10 | 0.001 | 2.0 | 10.0 | [0.5, 0.999] | 1 | 1 |
| mom09_lr03_d3_p20 | 0.0003 | 3.0 | 20.0 | [0.9, 0.999] | 1 | 1 |
| g2_b999_p20 | 0.001 | 1.5 | 20.0 | [0.0, 0.999] | 1 | 2 |
| g2_b999_lr15_d1 | 0.0015 | 1.0 | 10.0 | [0.0, 0.999] | 1 | 2 |
| g2_b999_lr2_d075 | 0.002 | 0.75 | 10.0 | [0.0, 0.999] | 1 | 2 |
| g3_b999_lr15_p20 | 0.0015 | 1.5 | 20.0 | [0.0, 0.999] | 1 | 3 |
| d2_b999_lr075_d3_p20 | 0.00075 | 3.0 | 20.0 | [0.0, 0.999] | 2 | 1 |
| b999_lr075_d3_p5 | 0.00075 | 3.0 | 5.0 | [0.0, 0.999] | 1 | 1 |

D/G cadence is measured on the same declared outer-step clock. Fewer role updates are explicit, not hidden extra compute. Actual counts and EMA metrics are retained in each full episode.

Actual episodes: **63**. Fixed live observations: **1512**. Recorded wall time: **434.24 seconds** on a shared CPU host.

Full per-episode artifacts and source hashes: [index.json](index.json.gz). Original phase logs, plans and exact source archives are retained.
