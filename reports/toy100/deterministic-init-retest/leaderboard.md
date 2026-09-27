# New initialization: quick coverage screen

All runs use develop’s deterministic network and prior initialization, 1,200 public API updates, and the same 24 observations. A pass requires all eight modes and at least 90% high-quality samples for the final five observations. This screen does not select a release default.

| Configuration | New result | Passing observations | First arrival | Final passing streak | Final modes / quality | Old result on this screen |
|---|---|---:|---:|---:|---|---|
| [public-k3p-new-init](evidence/public-k3p-new-init/archive-manifest.json) | FAIL | 0/24 | — | 0 | 6 / 97.1% | FAIL 0/24 (released K3P; not a pure initialization ablation) |
| [public-ka2-new-init](evidence/public-ka2-new-init/archive-manifest.json) | FAIL | 0/24 | — | 0 | 6 / 99.9% | NOT_MEASURED_ON_THIS_EXACT_SCREEN |
| [public-ka2-constant-new-init](evidence/public-ka2-constant-new-init/archive-manifest.json) | FAIL | 0/24 | — | 0 | 4 / 100.0% | NOT_MEASURED_ON_THIS_EXACT_SCREEN |
| [API-DV16-new-init](evidence/api-dv16-new-init/archive-manifest.json) | FAIL | 0/24 | — | 0 | 7 / 91.0% | PASS 11/24; final streak 11 |
| [API-RP5-new-init](evidence/api-rp5-new-init/archive-manifest.json) | FAIL | 0/24 | — | 0 | 5 / 75.5% | FAIL 0/24; final 5/8 |
| [API-DV1-new-init](evidence/api-dv1-new-init/archive-manifest.json) | FAIL | 1/24 | 1200 | 1 | 8 / 91.1% | NOT_MEASURED_ON_THIS_EXACT_SCREEN |

Arrival means the first observation meeting both quality and coverage. All later departures and the complete observations are preserved in each linked record. Scheduled controls retain their declared horizon and noise settings; their quality results do not establish autonomous indefinite operation.
