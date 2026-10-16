# Forge family leaderboard

A **family** can have several **recipes**. Each score belongs to one exact recipe and source. For example, BCAP's selected recipe uses DualNorm; Confidence LR, Finite critic cap and Positive evidence are separate BCAP research recipes, with their own scores.

Click a count for its gates, measurements and evidence. Required lower tiers must pass before advancing.

## Selected recipes

Passes / required-view instances for one selected recipe per family. Shared tests count once in every view that requires them: **152 means 152 view requirements, including repeated tests**. These totals retain the selected recipe's recorded qualification; separate diagnostic runs appear below.

| Family / selected recipe | Tier 1 requirements | Tier 2 requirements | Tier 3 requirements | Total view requirements |
| --- | ---: | ---: | ---: | ---: |
| **[BCAP](families/bcap-pure.md)** · [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)<br>Recorded source measurement · **stale**; no current qualification | **[22/22](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[49/116](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/14 (7 blocked, 7 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[71/152 (7 blocked, 7 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422)** |
| **[BISPF](families/bispf.md)**<br>Recorded source measurement · **stale**; no current qualification | **[22/22](families/bispf.md#cohort-cuda-6421ba55ca82-tier-1)** | **[55/116](families/bispf.md#cohort-cuda-6421ba55ca82-tier-2)** | **[0/14 (3 blocked, 11 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-tier-3)** | **[77/152 (3 blocked, 11 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82)** |
| **[Atlas](families/atlas.md)** | **[0/22 (22 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/116 (114 incompatible, 2 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/14 (7 incompatible, 7 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[0/152 (143 incompatible, 9 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422)** |
| **[E22](families/e22.md)** | **[0/22 (22 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/116 (114 incompatible, 2 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/14 (7 incompatible, 7 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[0/152 (143 incompatible, 9 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422)** |
| **[GAN v3 release 0.7](families/release07-gan-v3.md)**<br>Recorded source measurement · **stale**; no current qualification | **[20/22](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/116 (116 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/14 (7 blocked, 7 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20/152 (7 blocked, 123 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422)** |
| **[K3P](families/k3p.md)**<br>Recorded source measurement · **stale**; no current qualification | **[20/22](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/116 (116 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/14 (7 blocked, 7 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20/152 (7 blocked, 123 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422)** |
| **[KA2](families/ka2.md)**<br>Recorded source measurement · **stale**; no current qualification | **[20/22](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/116 (116 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/14 (7 blocked, 7 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20/152 (7 blocked, 123 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422)** |
| **[R1/R2](families/r1r2.md)**<br>Recorded source measurement · **stale**; no current qualification | **[20/22](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0/116 (116 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0/14 (7 blocked, 7 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20/152 (7 blocked, 123 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422)** |

Backfill coverage: **853/853 distinct recipe/test pairs attempted (selected and research)**; 0 pending and 0 running. [Outcomes, blocked attempts and setup](ATTEMPT_COVERAGE.md). This counts attempts, including start refusals, rather than passes or qualification.

## Research recipes and separate diagnostics

Each row names its owning family and its own recipe or diagnostic scope. Counts are **passes / distinct required tests within that row**: 29 means 29 different tests. These scores are excluded from the selected-recipe totals above. Compare rows only when their test, source and sampling contracts match.

Research recipes use their recorded revision 9 assignments. Atlas selected-cloud has a separate seven-question Tier 1 policy scope. Same-source backfill results fill previously unmeasured research cells while retaining original verdicts and qualification. Historical seed-only studies are retained in the expandable section below and excluded from this backfill.

| Family | Recipe / diagnostic scope | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests |
| --- | --- | ---: | ---: | ---: | ---: |
| [BCAP](families/bcap-pure.md) | [Confidence LR; seed 0](families/bcap-pure.md#confidence-lr-correction-six-task-screen-and-own-retention) | [6/6](families/bcap-pure.md#confidence-lr-correction-six-task-screen-and-own-retention) | [9/21](families/bcap-pure.md#confidence-lr-correction-six-task-screen-and-own-retention) | [0/2 (1 blocked)](families/bcap-pure.md#confidence-lr-correction-six-task-screen-and-own-retention) | [15/29 (1 blocked)](families/bcap-pure.md#confidence-lr-correction-six-task-screen-and-own-retention) |
| [BCAP](families/bcap-pure.md) | [Finite critic cap; seed 0](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [4/6 (1 timeout)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [7/21 (2 blocked)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [0/2 (1 blocked)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [11/29 (3 blocked, 1 timeout)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) |
| [BCAP](families/bcap-pure.md) | [Critic orientation; seed 0](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [1/6 (4 incompatible)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [1/21 (13 incompatible, 1 blocked)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [0/2 (1 incompatible, 1 blocked)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) | [2/29 (18 incompatible, 2 blocked)](families/bcap-pure.md#later-gaussian-corrections-one-failed-acquisition-one-failed-retention) |
| [BCAP](families/bcap-pure.md) | [Signed confidence; seed 0](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [5/6 (1 timeout)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [8/21 (1 blocked)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [0/2 (1 blocked)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [13/29 (2 blocked, 1 timeout)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) |
| [BCAP](families/bcap-pure.md) | [Positive evidence; seed 0](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [5/6 (1 timeout)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [8/21 (1 blocked)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [0/2 (1 blocked)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) | [13/29 (2 blocked, 1 timeout)](families/bcap-pure.md#signed-and-positive-evidence-responses-retention-remains-unsolved) |
| [BCAP](families/bcap-pure.md) | [G-only coherent mean (original seed0)](g-coherent-mean-seed0/README.md) | [1/6 (5 incompatible)](g-coherent-mean-seed0/README.md) | [0/21 (19 incompatible, 1 blocked)](g-coherent-mean-seed0/README.md) | [0/2 (1 incompatible, 1 blocked)](g-coherent-mean-seed0/README.md) | [1/29 (25 incompatible, 2 blocked)](g-coherent-mean-seed0/README.md) |
| [BCAP](families/bcap-pure.md) | [Fresh matched protected NS/transport control](g-coherent-mean-seed0/README.md) | [1/6 (5 incompatible)](g-coherent-mean-seed0/README.md) | [0/21 (19 incompatible, 1 blocked)](g-coherent-mean-seed0/README.md) | [0/2 (1 incompatible, 1 blocked)](g-coherent-mean-seed0/README.md) | [1/29 (25 incompatible, 2 blocked)](g-coherent-mean-seed0/README.md) |
| [BCAP](families/bcap-pure.md) | [kinetic_transport_local_v2; seed 0; b95b46ed](families/bcap-pure.md#pr399-retained-research) | [2/6 (1 timeout)](families/bcap-pure.md#pr399-retained-research) | [11/21 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [13/29 (2 blocked, 1 timeout)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-nsgda-cross-v1; seed 0; b95b46ed](families/bcap-pure.md#pr399-retained-research) | [1/6](families/bcap-pure.md#pr399-retained-research) | [7/21 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [8/29 (2 blocked)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [cell-quota-dualnorm-cross-v1; seed 0; b95b46ed](families/bcap-pure.md#pr399-retained-research) | [1/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [4/21 (5 incompatible, 1 blocked, 4 errors, 3 timeouts)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [5/29 (9 incompatible, 2 blocked, 4 errors, 3 timeouts)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-v1; seed 0; 9bc9e6ae](families/bcap-pure.md#pr399-retained-research) | [2/6 (1 timeout)](families/bcap-pure.md#pr399-retained-research) | [9/21 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [11/29 (2 blocked, 1 timeout)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-reference8-v1; seed 0; 9d618ca1](families/bcap-pure.md#pr399-retained-research) | [0/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [6/21 (5 incompatible, 2 blocked, 3 timeouts)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [6/29 (9 incompatible, 3 blocked, 3 timeouts)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-reference8-current-v1; seed 0; 9d618ca1](families/bcap-pure.md#pr399-retained-research) | [0/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [5/21 (5 incompatible, 2 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [5/29 (9 incompatible, 3 blocked)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-reference8-u-v1; seed 0; 79315f58](families/bcap-pure.md#pr399-retained-research) | [0/6 (4 incompatible, 1 timeout)](families/bcap-pure.md#pr399-retained-research) | [7/21 (5 incompatible, 2 blocked, 3 timeouts)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [7/29 (9 incompatible, 3 blocked, 4 timeouts)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-u-v1; seed 0; 79315f58](families/bcap-pure.md#pr399-retained-research) | [1/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [6/21 (5 incompatible, 1 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [7/29 (9 incompatible, 2 blocked)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-bank8-u-v1; seed 0; 89b0e03f](families/bcap-pure.md#pr399-retained-research) | [1/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [6/21 (5 incompatible, 1 blocked, 2 timeouts)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [7/29 (9 incompatible, 2 blocked, 2 timeouts)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [kinetic-local-dualnorm-g-history-bank8-v-v1; seed 0; 89b0e03f](families/bcap-pure.md#pr399-retained-research) | [1/6 (4 incompatible)](families/bcap-pure.md#pr399-retained-research) | [7/21 (5 incompatible, 1 blocked, 2 timeouts)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [8/29 (9 incompatible, 2 blocked, 2 timeouts)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [generator-idle-ema-bcap-direction-v1; seed 0; 27ab4cf9](families/bcap-pure.md#pr399-retained-research) | [4/6 (2 incomplete)](families/bcap-pure.md#pr399-retained-research) | [8/21 (2 blocked)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [12/29 (3 blocked, 2 incomplete)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [bcap-default-baseline-direction-v1; seed 0; 27ab4cf9](families/bcap-pure.md#pr399-retained-research) | [5/6 (1 incomplete)](families/bcap-pure.md#pr399-retained-research) | [8/21 (1 blocked, 1 error)](families/bcap-pure.md#pr399-retained-research) | [0/2 (1 blocked)](families/bcap-pure.md#pr399-retained-research) | [13/29 (2 blocked, 1 error, 1 incomplete)](families/bcap-pure.md#pr399-retained-research) |
| [BCAP](families/bcap-pure.md) | [Strict clock-free audit; separate Torch 2.13 cohort](tier-coverage-20261014/README.md) | — | — | [1/1](tier-coverage-20261014/README.md) | [1/1](tier-coverage-20261014/README.md) |
| [BCAP](families/bcap-pure.md) | [Unchanged selected recipe: deeper diagnostics (Torch 2.13)](actual-missing-diagnostics-20261014/README.md) | — | — | [1/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) | [1/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) |
| [BISPF](families/bispf.md) | [BISPF rebuilt checkpoints + matched-prior 14k diagnostics](ATTEMPT_COVERAGE.md#checkpoint-recovery) | — | [0/3](ATTEMPT_COVERAGE.md#checkpoint-recovery) | [0/3](ATTEMPT_COVERAGE.md#checkpoint-recovery) | [0/6](ATTEMPT_COVERAGE.md#checkpoint-recovery) |
| [BISPF](families/bispf.md) | [Verified extension from the existing uninterrupted run; source fafd66154108](grouped-evidence/4ee70c2cf634f45fbef5072e8689ed2a366f695603ad175fa4905dd19d9e04e8.json) | — | — | [1/1](grouped-evidence/4ee70c2cf634f45fbef5072e8689ed2a366f695603ad175fa4905dd19d9e04e8.json) | [1/1](grouped-evidence/4ee70c2cf634f45fbef5072e8689ed2a366f695603ad175fa4905dd19d9e04e8.json) |
| [Atlas](families/atlas.md) | [Selected-cloud policy; seed 0 (separate scope)](atlas-compatible-coverage-20261013/README.md) | [0/7 (4 incompatible)](atlas-compatible-coverage-20261013/README.md) | — | — | [0/7 (4 incompatible)](atlas-compatible-coverage-20261013/README.md) |
| [Atlas](families/atlas.md) | [Unchanged public policy: image diagnostics](actual-missing-diagnostics-20261014/README.md) | — | [1/4](actual-missing-diagnostics-20261014/README.md) | — | [1/4](actual-missing-diagnostics-20261014/README.md) |
| [Atlas](families/atlas.md) | [Selected-cloud execution after adapter investigation (pre-repair source)](ATTEMPT_COVERAGE.md#remaining-attempts) | [0/7 (3 incompatible, 2 errors, 1 incomplete)](ATTEMPT_COVERAGE.md#remaining-attempts) | — | — | [0/7 (3 incompatible, 2 errors, 1 incomplete)](ATTEMPT_COVERAGE.md#remaining-attempts) |
| [Atlas](families/atlas.md) | [ATLAS checkpoint-export repair (same recipe)](ATTEMPT_COVERAGE.md#checkpoint-recovery) | [0/2](ATTEMPT_COVERAGE.md#checkpoint-recovery) | — | — | [0/2](ATTEMPT_COVERAGE.md#checkpoint-recovery) |
| [E22](families/e22.md) | [Unchanged public policy: image diagnostics](actual-missing-diagnostics-20261014/README.md) | — | [1/4](actual-missing-diagnostics-20261014/README.md) | — | [1/4](actual-missing-diagnostics-20261014/README.md) |
| [E22](families/e22.md) | [Selected-cloud execution after adapter investigation (pre-repair source)](actual-missing-diagnostics-20261014/README.md) | [0/7 (3 incompatible)](actual-missing-diagnostics-20261014/README.md) | — | — | [0/7 (3 incompatible)](actual-missing-diagnostics-20261014/README.md) |
| [E22](families/e22.md) | [E22 checkpoint-export repair (same recipe)](ATTEMPT_COVERAGE.md#checkpoint-recovery) | [0/2](ATTEMPT_COVERAGE.md#checkpoint-recovery) | — | — | [0/2](ATTEMPT_COVERAGE.md#checkpoint-recovery) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | [Selected recipe Tier 2 diagnostics; seed 0](tier-coverage-20261014/README.md) | — | [9/21](tier-coverage-20261014/README.md) | — | [9/21](tier-coverage-20261014/README.md) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | [Unchanged selected recipe: deeper diagnostics (Torch 2.13)](actual-missing-diagnostics-20261014/README.md) | — | — | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) |
| [K3P](families/k3p.md) | [Selected recipe Tier 2 diagnostics; seed 0](tier-coverage-20261014/README.md) | — | [11/21](tier-coverage-20261014/README.md) | — | [11/21](tier-coverage-20261014/README.md) |
| [K3P](families/k3p.md) | [Unchanged selected recipe: deeper diagnostics (Torch 2.13)](actual-missing-diagnostics-20261014/README.md) | — | — | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) |
| [KA2](families/ka2.md) | [Selected recipe Tier 2 diagnostics; seed 0](tier-coverage-20261014/README.md) | — | [10/21](tier-coverage-20261014/README.md) | — | [10/21](tier-coverage-20261014/README.md) |
| [KA2](families/ka2.md) | [Unchanged selected recipe: deeper diagnostics (Torch 2.13)](actual-missing-diagnostics-20261014/README.md) | — | — | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) |
| [R1/R2](families/r1r2.md) | [Selected recipe Tier 2 diagnostics; seed 0](tier-coverage-20261014/README.md) | — | [9/21](tier-coverage-20261014/README.md) | — | [9/21](tier-coverage-20261014/README.md) |
| [R1/R2](families/r1r2.md) | [Unchanged selected recipe: deeper diagnostics (Torch 2.13)](actual-missing-diagnostics-20261014/README.md) | — | — | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) | [0/7 (4 blocked)](actual-missing-diagnostics-20261014/README.md) |

<details>
<summary>Historical seed-only studies (excluded from backfill)</summary>

These 12 archived seed 1–6 rows retain **324 unrun recipe/test pairs**. They are excluded by the repository instruction against seed-only experiments. Their original counts and evidence remain visible here; they are outside the completed backfill scope.

| Family | Historical seed-only recipe | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests |
| --- | --- | ---: | ---: | ---: | ---: |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 1](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 1](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 2](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 2](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 3](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 3](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 4](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 4](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 5](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 5](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence candidate; research seed 6](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |
| [BCAP](families/bcap-pure.md) | [G-coherence control; research seed 6](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/6 (5 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/21 (20 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [0/2 (2 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) | [1/29 (27 not run)](families/bcap-pure.md#g-only-coherence-registered-paired-research) |

</details>

<details>
<summary>Selected recipe breakdown by view (shared tests repeat across views)</summary>

Each view uses the selected recipe above. View scores are the requirements summed in its selected-recipe total.

| Family / selected recipe | View | Tier 1 | Tier 2 | Tier 3 | Total view requirements |
| --- | --- | ---: | ---: | ---: | ---: |
| [BCAP](families/bcap-pure.md) | ↳ [adaptation](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [8/19](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/1 (1 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [11/23 (1 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [BCAP](families/bcap-pure.md) | ↳ [clockfree_continuous](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [4/4](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [8/19](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/7 (4 blocked, 3 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [12/30 (4 blocked, 3 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [BCAP](families/bcap-pure.md) | ↳ [discriminator_stability](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [6/6](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [9/21](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [15/29 (1 blocked, 1 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [BCAP](families/bcap-pure.md) | ↳ [formulation_comparison](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [8/19](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [11/24 (1 blocked, 1 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [BCAP](families/bcap-pure.md) | ↳ [host_profile_transfer](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [8/19](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [11/24 (1 blocked, 1 diagnostic attempts)](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [BCAP](families/bcap-pure.md) | ↳ [quality_coverage](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [8/19](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [11/22](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| [BISPF](families/bispf.md) | ↳ [adaptation](families/bispf.md#cohort-cuda-6421ba55ca82-adaptation) | [3/3](families/bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-1) | [9/19](families/bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-2) | [0/1 (1 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-adaptation-tier-3) | [12/23 (1 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-adaptation) |
| [BISPF](families/bispf.md) | ↳ [clockfree_continuous](families/bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous) | [4/4](families/bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-1) | [9/19](families/bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-2) | [0/7 (3 blocked, 4 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous-tier-3) | [13/30 (3 blocked, 4 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-clockfree_continuous) |
| [BISPF](families/bispf.md) | ↳ [discriminator_stability](families/bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability) | [6/6](families/bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-1) | [10/21](families/bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-2) | [0/2 (2 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability-tier-3) | [16/29 (2 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-discriminator_stability) |
| [BISPF](families/bispf.md) | ↳ [formulation_comparison](families/bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison) | [3/3](families/bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-1) | [9/19](families/bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-2) | [0/2 (2 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison-tier-3) | [12/24 (2 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-formulation_comparison) |
| [BISPF](families/bispf.md) | ↳ [host_profile_transfer](families/bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer) | [3/3](families/bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-1) | [9/19](families/bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-2) | [0/2 (2 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer-tier-3) | [12/24 (2 diagnostic attempts)](families/bispf.md#cohort-cuda-6421ba55ca82-host_profile_transfer) |
| [BISPF](families/bispf.md) | ↳ [quality_coverage](families/bispf.md#cohort-cuda-6421ba55ca82-quality_coverage) | [3/3](families/bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-1) | [9/19](families/bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-2) | [0/0](families/bispf.md#cohort-cuda-6421ba55ca82-quality_coverage-tier-3) | [12/22](families/bispf.md#cohort-cuda-6421ba55ca82-quality_coverage) |
| [Atlas](families/atlas.md) | ↳ [adaptation](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/3 (3 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/19 (19 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/1 (1 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [0/23 (23 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [Atlas](families/atlas.md) | ↳ [clockfree_continuous](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/4 (4 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/19 (19 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/7 (3 incompatible, 4 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [0/30 (26 incompatible, 4 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [Atlas](families/atlas.md) | ↳ [discriminator_stability](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/6 (6 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/21 (19 incompatible, 2 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/2 (1 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [0/29 (26 incompatible, 3 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [Atlas](families/atlas.md) | ↳ [formulation_comparison](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/3 (3 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/19 (19 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/2 (1 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [0/24 (23 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [Atlas](families/atlas.md) | ↳ [host_profile_transfer](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/3 (3 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/19 (19 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/2 (1 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [0/24 (23 incompatible, 1 blocked)](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [Atlas](families/atlas.md) | ↳ [quality_coverage](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/3 (3 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/19 (19 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [0/22 (22 incompatible)](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| [E22](families/e22.md) | ↳ [adaptation](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation) | [0/3 (3 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/19 (19 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/1 (1 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [0/23 (23 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [E22](families/e22.md) | ↳ [clockfree_continuous](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0/4 (4 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/19 (19 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/7 (3 incompatible, 4 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [0/30 (26 incompatible, 4 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [E22](families/e22.md) | ↳ [discriminator_stability](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0/6 (6 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/21 (19 incompatible, 2 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/2 (1 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [0/29 (26 incompatible, 3 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [E22](families/e22.md) | ↳ [formulation_comparison](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0/3 (3 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/19 (19 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/2 (1 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [0/24 (23 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [E22](families/e22.md) | ↳ [host_profile_transfer](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0/3 (3 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/19 (19 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/2 (1 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [0/24 (23 incompatible, 1 blocked)](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [E22](families/e22.md) | ↳ [quality_coverage](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0/3 (3 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/19 (19 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [0/22 (22 incompatible)](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [adaptation](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/19 (19 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/1 (1 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3/23 (20 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [clockfree_continuous](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/19 (19 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/7 (4 blocked, 3 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3/30 (4 blocked, 22 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [discriminator_stability](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/21 (21 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5/29 (1 blocked, 22 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [formulation_comparison](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/19 (19 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3/24 (1 blocked, 20 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [host_profile_transfer](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/19 (19 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3/24 (1 blocked, 20 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [GAN v3 release 0.7](families/release07-gan-v3.md) | ↳ [quality_coverage](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/19 (19 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3/22 (19 diagnostic attempts)](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| [K3P](families/k3p.md) | ↳ [adaptation](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/19 (19 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/1 (1 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3/23 (20 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [K3P](families/k3p.md) | ↳ [clockfree_continuous](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/19 (19 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/7 (4 blocked, 3 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3/30 (4 blocked, 22 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [K3P](families/k3p.md) | ↳ [discriminator_stability](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/21 (21 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5/29 (1 blocked, 22 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [K3P](families/k3p.md) | ↳ [formulation_comparison](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/19 (19 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3/24 (1 blocked, 20 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [K3P](families/k3p.md) | ↳ [host_profile_transfer](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/19 (19 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3/24 (1 blocked, 20 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [K3P](families/k3p.md) | ↳ [quality_coverage](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/19 (19 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3/22 (19 diagnostic attempts)](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| [KA2](families/ka2.md) | ↳ [adaptation](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/19 (19 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/1 (1 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3/23 (20 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [KA2](families/ka2.md) | ↳ [clockfree_continuous](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/19 (19 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/7 (4 blocked, 3 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3/30 (4 blocked, 22 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [KA2](families/ka2.md) | ↳ [discriminator_stability](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/21 (21 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5/29 (1 blocked, 22 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [KA2](families/ka2.md) | ↳ [formulation_comparison](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/19 (19 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3/24 (1 blocked, 20 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [KA2](families/ka2.md) | ↳ [host_profile_transfer](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/19 (19 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3/24 (1 blocked, 20 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [KA2](families/ka2.md) | ↳ [quality_coverage](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/19 (19 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3/22 (19 diagnostic attempts)](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| [R1/R2](families/r1r2.md) | ↳ [adaptation](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0/19 (19 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0/1 (1 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3/23 (20 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation) |
| [R1/R2](families/r1r2.md) | ↳ [clockfree_continuous](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0/19 (19 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0/7 (4 blocked, 3 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3/30 (4 blocked, 22 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| [R1/R2](families/r1r2.md) | ↳ [discriminator_stability](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0/21 (21 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5/29 (1 blocked, 22 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| [R1/R2](families/r1r2.md) | ↳ [formulation_comparison](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0/19 (19 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3/24 (1 blocked, 20 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| [R1/R2](families/r1r2.md) | ↳ [host_profile_transfer](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0/19 (19 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0/2 (1 blocked, 1 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3/24 (1 blocked, 20 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| [R1/R2](families/r1r2.md) | ↳ [quality_coverage](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0/19 (19 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3/22 (19 diagnostic attempts)](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage) |

</details>

## Reading the results

Counts show passes / required tasks; selected-row pass counts remain unchanged. For rows with run-attempt coverage, the suffix counts genuinely unattempted tasks as not run, constructor refusals as incompatible, missing required artifacts as blocked, and other separate-runtime attempts as diagnostic attempts. A start refusal is not a numerical failure. Rows without that registry retain their recorded missing/preflight counts. No formulation is changed to fit a test. Recorded PASS and FAIL are both executed results. Attempted ERROR, INVALID or INCOMPLETE results show their cause and execution evidence on the family page and earn no pass credit. Zero recorded passes always displays as 0, including unrun families.

Counts retain each original recipe, prior, initialization, budget, serving law and source. Family pages retain exact metrics, phase checks, failures, blockers, historical comparisons and calibration requirements. Observation checks and catalog attempt counts add no benchmark passes; untiered historical studies remain on those detail pages.

[Actual attempts of unrun and blocked tests](actual-missing-diagnostics-20261014/README.md). The new diagnostic rows show executions beyond failed tier gates and Atlas/E22 public-policy tests. Atlas's original 19/19 CUDA passes remain valid under their original task and sampling contracts.

[Diagnostic completion, new strict audit and every skip reason](tier-coverage-20261014/README.md). The selected-recipe Tier 2 diagnostic rows retain their original 21-task scope. The new strict BCAP audit has a separate runtime and changes no qualification.

[Prior-search measurements and gates](bcap-prior-search/README.md) · [Prior-search findings](bcap-prior-search/findings.md)

## Browse by technique tag

Tags describe properties shared by a family's supported variants, not performance or qualification. Configuration-dependent behavior is explained on each family page. Historical pages retain their own scope.

<a name="tag-adaptive-training-policy"></a>

### `adaptive-training-policy`

Uses an observation-driven training policy rather than a fixed update schedule.

- [Atlas](families/atlas.md)
- [E22](families/e22.md)

<a name="tag-adversarial-training"></a>

### `adversarial-training`

Trains a generator against a critic; task-owned auxiliary objectives may also apply.

- [Atlas](families/atlas.md)
- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [BISPF](families/bispf.md)
- [E22](families/e22.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)
- [Halloween optimizer/loss transfer](families/halloween.md) (historical cohort)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without critic penalty](families/k3p-no-penalty.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)
- [KA2](families/ka2.md)
- [R1/R2](families/r1r2.md)

<a name="tag-capped-input-gradients"></a>

### `capped-input-gradients`

Penalizes critic input-gradient norms above a threshold on real and generated inputs; this is a soft loss penalty.

- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [BISPF](families/bispf.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)

<a name="tag-constant-learning-rate"></a>

### `constant-learning-rate`

Declared baseline uses constant learning rates; resolved per-role rates remain configuration-specific.

- [BISPF](families/bispf.md)
- [Halloween optimizer/loss transfer](families/halloween.md) (historical cohort)

<a name="tag-critic-gradient-penalty"></a>

### `critic-gradient-penalty`

Uses a penalty on critic input gradients; its formula and strength are configuration-specific.

- [Atlas](families/atlas.md)
- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [BISPF](families/bispf.md)
- [E22](families/e22.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)
- [KA2](families/ka2.md)
- [R1/R2](families/r1r2.md)

<a name="tag-historical-cohort"></a>

### `historical-cohort`

A retained historical family with its original source, prior and scoring contracts.

- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)

<a name="tag-learning-rate-annealing"></a>

### `learning-rate-annealing`

Declared baseline decreases learning rates during training; horizons and floors are configuration-specific.

- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)

<a name="tag-optimizer-interventions"></a>

### `optimizer-interventions`

Includes formulation-specific interventions around optimizer steps; availability depends on the host and resolved recipe.

- [Atlas](families/atlas.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [BISPF](families/bispf.md)
- [E22](families/e22.md)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without critic penalty](families/k3p-no-penalty.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)
- [KA2](families/ka2.md)

<a name="tag-selectable-loss"></a>

### `selectable-loss`

The family explicitly supports multiple adversarial loss formulations.

- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BISPF](families/bispf.md)

<a name="tag-structural-ablation"></a>

### `structural-ablation`

Removes a named mechanism from a parent technique and retains a separate evidence identity.

- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without critic penalty](families/k3p-no-penalty.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)

<a name="tag-tensorflow-v1-adam"></a>

### `tensorflow-v1-adam`

Uses the declared dense TensorFlow-v1 Adam update law with checkpointed application clocks and epsilon before second-moment bias correction.

- [Halloween optimizer/loss transfer](families/halloween.md) (historical cohort)

## Refresh

```sh
python reports/forge/regenerate_technique_inventory.py
```

This regenerates the leaderboard, family pages and experiments-by-tier report from committed evidence and declarations. Register newly measured evidence with `--source-commit <executed-commit>`; advancing the recorded view policy also requires `--advance-policy`.

[Experiments, criteria and tier assignments](EXPERIMENTS_BY_TIER.md) · [Complete numerical publication and provenance](technique-inventory.json)

Publication input digest `8faf5cc91713e381344ba834cd766f8cd7f1e9e80407c622c2d3c62d8e92bc58`.
[Performance by test and complete recipe](test-performance.md): compatible passes, failures, execution errors and missing witnesses, with sortable evidence drill-down.
