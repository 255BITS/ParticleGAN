# RP5 vector wave1 audit

Completed runs were copied losslessly, including checkpoint files, source ZIPs, raw curves and all audit corrections. JSONL compression is reversible; archive manifests record original and archived hashes. Shared broader-results/README/state were not edited.

| Task | Status | Passing / 24 | Final suffix | Stable from |
| --- | --- | --- | --- | --- |
| vector_two_broad | PASS | 23/24 | 23 | 100 |
| vector_unequal_mass | PASS | 18/24 | 18 | 350 |
| vector_unequal_width | PASS | 19/24 | 19 | 300 |
| vector_anisotropic | NOT_RUN | — | — | — |
| vector_overlap | NOT_RUN | — | — | — |
| vector_spiral | NOT_RUN | — | — | — |

Each completed run retains the unchanged RP5 package, canonical parameter-only initialization, frozen task/model/scorer, CUDA streams and declared isolated output-noise seed2303. Scores above are independently derived from every observation and original thresholds; all24 checks and final-five suffix are required.

The original two_broad initialization alarm included the non-parameter D.freqs buffer. Its G6/prior1/D6 parameters actually matched. Both the false audit and later corrections are preserved. No rerun was used.

The JSON companion contains ready-to-append broader_entries and per-tensor/source/archive proofs. Pending or unrun tasks earn no pass. Historical K3P global-noise seed402 results are not a matched comparator. No model execution, GPU, training or tests were performed for this audit.

Supervisor scheduling note: anisotropic,overlap,spiral are deferred until the unchanged RP5 long30000 finishes. This audit is complete for the first three; monitoring has stopped, with no reruns.
