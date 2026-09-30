# Toy and grid quality target

No recommendation for this target until one candidate passes both unchanged gates and required validity/replay checks.

| Candidate | Toy P | Modes | TV | Toy | Grid | Grid validity |
|---|---:|---:|---:|---|---|---|
| CB64-RA8 | 0.965332 | 25/25 | 0.052114 | PASS | FAIL | VALID |
| CB64-RA9 | 0.965332 | 25/25 | 0.052114 | PASS | FAIL | VALID |
| CB64-RA4 | 0.758057 | 21/25 | 0.263540 | FAIL | FAIL | VALID |
| E22 | 0.715454 | 25/25 | 0.284546 | FAIL | PENDING | pending |
| CB64-RA7 | 0.681641 | 25/25 | 0.318359 | FAIL | NOT_RUN_TOY_FAILED | pending |
| CB64-RA | 0.623291 | 24/25 | 0.377969 | FAIL | FAIL | pending |
| CB64-RA6 | 0.516968 | 23/25 | 0.483032 | FAIL | NOT_RUN_TOY_FAILED | pending |
| CB64-RA3 | 0.409912 | 17/25 | 0.592935 | FAIL | PENDING | pending |
| CB64-RA2 | 0.156616 | 1/25 | 0.843384 | FAIL | FAIL | VALID |
| CB64-RA5 | — | — | — | ERROR | NOT_RUN_RUNTIME_ERROR | pending |

Complete toy entries use the final update 2000; runtime errors have no quality verdict. Grid requires all original terminal observations and the independent holdout.
A completed training process is separate from passing the quality gate. CPU mechanism tests do not establish GPU quality.
