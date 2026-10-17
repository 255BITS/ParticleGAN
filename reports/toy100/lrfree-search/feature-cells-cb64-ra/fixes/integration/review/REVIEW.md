# Integrated stability review

Status: focused contracts PASS; full CUDA quality validation pending.

The shared package passes 8 stability, 5 adaptive geometry and 2 flagged-surplus CPU regressions. The original reference backend remains byte-identical. Factory routing, small-population reference law, matching sampler, checkpoint rejection, caches, gradients, optimizer rows and unique parent supply are covered.

## Concrete integration correction

The original frozen rare-hole input exposed one cross-group ordinary move: flagged contamination was treated as legitimate population surplus. Excluding flagged rows from ordinary population counts preserves every supported survivor, including p≤Q rows that cannot seed births. Isolation accounts for the combined planned ordinary moves. Parents selected for deletion cannot seed ordinary or isolation births. Count evidence, Q, guard arithmetic and quality gates remain unchanged. Mass-policy v3 rejects incompatible v2 continuation.

The original GPU failure and its traceback remain preserved. Root applied NONFLAGGED-SURPLUS.patch and reran every unchanged named assertion on physical GPU0.

| Frozen case | Ordinary | Repairs | Distinct parents | Mass TV | Rare/target | Intended agreement |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 0 | 46 | 46 | 0.0 | 1.0 | 0.913043 |
| rare_hole | 0 | 46 | 46 | 0.0 | 1.0 | 0.195652 |

The mass contract reserves 34 MiB on GPU0. This is planning on fixed saved inputs, without training. Rare-hole intended identity is erased by the fixture; its agreement is disclosed and has no identity-recovery gate. Geometry and performance function ASTs match their separately verified implementations.

## Read-only validation monitor

monitor_validation.py waits for the future validation freeze, checks its immutable local/external maps, and imports that lane’s original canonical collector. Only collector JSON writes are redirected to integration/review; validation bytes stay read-only. Active jobs remain PENDING. Completed jobs retain exact original PASS/FAIL/ERROR and mandatory fixture validity, including official native clouds, schedules and holdout. The source-freeze identity is retained across polls. Final saved inputs receive a read-only hash manifest.

The monitor reproduced all 16 completed baseline verdicts (9/13 portability PASS and 0/3 native PASS). Every file in the baseline screen manifest and every declared original stability source/evidence hash remains unchanged.

Root may run:

```bash
/tmp/pr38-default-env/bin/python -u -B /ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/integration/review/monitor_validation.py --watch
```

Reports: integration/review/validation-monitor/summary.json and REPORT.md. The monitor creates no CUDA context or numerical job.

Reviewed feature_cells.py SHA: 080dccbc404e1fe4112cd18275eaf94b8f68e6086996b88b5f9b5718770573c1
Reviewed training.py SHA: 28f6cc3f01308829531d4fab655db4cfd1870354d9c81f9374175a8b324088d9

Recommendation: retain the recipe fields and perform the authorized complete CUDA validation after the support decision and final freeze. These focused contracts do not establish training quality.
