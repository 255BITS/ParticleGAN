# Toy and grid quality target

CB64-RA11 is the validated toy/grid winner and passes all three native tests plus exact CUDA replay. MNIST and five portability regressions prevent a general base-package recommendation.

| Candidate | Toy P | Modes | TV | Toy | Grid | Grid validity | Final center σ | Final max covariance | Holdout |
|---|---:|---:|---:|---|---|---|---:|---:|---|
| CB64-RA11 | 0.965332 | 25/25 | 0.052114 | PASS | PASS | VALID | 0.166607 | 1.347556 | PASS |
| CB64-RA9 | 0.965332 | 25/25 | 0.052114 | PASS | FAIL | VALID | 0.211927 | 1.612579 | PASS |
| CB64-RA10 | 0.965332 | 25/25 | 0.052114 | PASS | FAIL | VALID | 0.198250 | 1.773122 | FAIL |
| CB64-RA8 | 0.965332 | 25/25 | 0.052114 | PASS | FAIL | VALID | 0.203570 | 1.819186 | FAIL |
| CB64-RA4 | 0.758057 | 21/25 | 0.263540 | FAIL | FAIL | VALID | 0.313777 | 1.604172 | FAIL |
| E22 | 0.715454 | 25/25 | 0.284546 | FAIL | PENDING | pending | — | — | — |
| CB64-RA7 | 0.681641 | 25/25 | 0.318359 | FAIL | NOT_RUN_TOY_FAILED | pending | — | — | — |
| CB64-RA | 0.623291 | 24/25 | 0.377969 | FAIL | FAIL | pending | 0.233899 | 1.754370 | FAIL |
| CB64-RA6 | 0.516968 | 23/25 | 0.483032 | FAIL | NOT_RUN_TOY_FAILED | pending | — | — | — |
| CB64-RA3 | 0.409912 | 17/25 | 0.592935 | FAIL | PENDING | pending | — | — | — |
| CB64-RA2 | 0.156616 | 1/25 | 0.843384 | FAIL | FAIL | VALID | 0.191218 | 1.599540 | PASS |
| CB64-RA5 | — | — | — | ERROR | NOT_RUN_RUNTIME_ERROR | pending | — | — | — |

Complete toy entries use the final update 2000; runtime errors have no quality verdict. Grid requires all original terminal observations and the independent holdout.
Equal toy results are ordered by the original independent holdout pass. Final grid metrics are diagnostic: Grid still requires all five terminal checks and the holdout. Center limit is0.20σ; maximum covariance ratio limit is1.7.
A completed training process is separate from passing the quality gate. CPU mechanism tests do not establish GPU quality.

## Broader learned-model results

| Candidate | MNIST active feature distance ↓ | MNIST recall | CUDA replay |
|---|---:|---:|---|
| CB64-RA11 | 40.544410 | 0.00% | PASS |
| CB64-RA9 | — | — | PENDING |
| CB64-RA10 | — | — | PENDING |
| CB64-RA8 | — | — | PENDING |
| CB64-RA4 | 0.393402 | 78.61% | PASS |
| E22 | 0.544488 | 84.72% | PASS |
| CB64-RA7 | — | — | PENDING |
| CB64-RA | 1.488828 | 32.28% | PASS |
| CB64-RA6 | — | — | PENDING |
| CB64-RA3 | 0.476611 | 75.83% | PASS |
| CB64-RA2 | 0.814142 | 71.39% | PASS |
| CB64-RA5 | — | — | PENDING |

MNIST uses the same frozen 2000-update comparison. Replay PASS measures reproducible continuation, separately from sample quality.
RA11 is the joint toy/grid leader but its MNIST result regresses severely. It is not a general base-package recommendation.
