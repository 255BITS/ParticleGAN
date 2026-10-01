# Stability and mass diagnosis

The baseline frozen GPU suite finished with 9/13 portability passes and 0/3 native passes. All 16 mandatory fixture/source checks were valid, with no runtime error. Small-population actuation and large-population transport have separate causes.

## Small populations

| N | Heldout M | Minimum BH flagged rows | Maximum guard rows | Feature-cell actuation feasible |
|---:|---:|---:|---:|---|
| 12 | 6 | 35 | 0 | No |
| 32 | 16 | 38 | 1 | No |
| 256 | 128 | 40 | 12 | No |
| 799 | 399 | 40 | 39 | No |
| 800 | 400 | 40 | 40 | Yes |

Mode_hold had no ordinary or isolation move at N=12; blobs4 had 127 count discoveries and no move at N=32. Unequal_mass at N=256 ended within its point thresholds, but component-covariance error 1.13695 at step1050 broke the required final five-observation suffix; its terminal suffix was three. Infeasible isolation and a mismatched fixed feature-cell training/sampling kernel make a reference fallback appropriate; GPU efficacy of that correction remains unmeasured.

## Parent concentration and action resolution

Frozen cost fixtures at N=1024 use the single existing seed90229. Oracle labels are confined to evaluation. Removing the planted intervention gives exact correct macro masses (TV=0) but fine-cell count L1 discrepancies 730 nominal and 696 rare-hole, caused by narrower within-support geometry. Fine action quotas therefore confuse geometric width with support mass. An overlap-ball topology remained fragmented; the selected real-only centre graph coarsens both cases to eight groups with zero macro discrepancy.

| Fixture / exact-copy law | Ordinary copies | Isolation repairs | Unique / total parents | Maximum reuse | Mass TV | Rare / target mass | Intended-parent agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nominal original | 18 | 46 | 15 / 64 | 11 | .026367 | 1.0 | .695652 |
| Nominal proposed | 0 | 46 | 46 / 46 | 1 | 0 | 1.0 | .913043 |
| Rare-hole original | 20 | 46 | 9 / 66 | 15 | .043945 | 14.5 | 0 |
| Rare-hole proposed | 0 | 46 | 46 / 46 | 1 | 0 | 1.0 | .195652 |

All proposed exact-copy repairs are supported; original legitimate rare rows are retained. The nominal allocation uses each child's local real anchor and prioritizes scarce capacity by alternative-distance regret. This fixes the agreement regression from proportional quotas/absolute-distance tie handling while retaining exact macro mass. Rare-hole latent corruption overwrites common identities with the same near-rare hole, so its low individual provenance agreement remains a disclosed limit.

Eight focused CPU regressions pass: exact finite-resolution boundary, original small-host reference parity, small and active checkpoint reload/mismatch, ordinary count/budget/supply, supported-parent protection, isolation shortage/guard/unique supply, no rare inflation, and macro-balanced narrow-support conservation. No support/count p-value law or quality gate is relaxed. The original reference `birth_death.py` remains byte-identical.

## Integration and recommendation

Use the existing recipe fields unchanged: feature_cells, cells64, rank8, chunk256, real_anchor. Population routing is derived from the law and requires no threshold option. Compose the active kernel from geometry and fit/count/pool changes from performance. Preserve selected-operator sampling, the new mass settings and schema3, plus geometry cache invalidation. This private contribution still contains the old active fixed kernel solely because geometry owns its replacement; it must be composed before training.

Run the frozen-input GPU mass contract first, then full original GPU small-host screens and learned/native budgets through the root queue. CPU exact-copy improvements establish the repaired transport mechanism, not broad CUDA quality or training improvement. Keep the passing reference as the comparison until the complete composed candidate satisfies all original gates.
