# Corrected learned CUDA saved-artifact audit

Completed jobs only. All tensor loads use CPU storages; original device tags are retained for the frozen GPU fingerprints.

| Record | Primary | Evidence | Toy gate | Peak reserved MiB |
|---|---|---|---|---:|
| training-toy | COMPLETE | VALID | FAIL | 76.0 |
| training-mnist | PENDING | PENDING | — | — |
| replay-toy | PENDING | PENDING | — | — |
| replay-mnist | PENDING | PENDING | — | — |

Checkpoint/endpoint backend, matching sampler, kernel, mass policy, parent counters and RNG placement are retained in summary.json. MNIST has no newly introduced numerical gate.
