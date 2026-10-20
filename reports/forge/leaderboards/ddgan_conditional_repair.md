# DDGAN conditional repair: current research leaderboard

GPU0 shared-load diagnostic only; no ordinary qualification or speed ranking. All runs use protocol seed0, clean live four-step sampling and unchanged scientific gates.

| Recipe | Task | Verdict | Updates | Last5 passes | Final CDF | Worst conditional CDF | GIF |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `bispf-optimized-quintic-v1` | `ddgan_repair_bimodal_chain_v1` | **INCOMPLETE** | >=3542/5000 | 0/5 | — | — | — |
| `bispf-ddgan-energy-v1` | `ddgan_repair_bimodal_chain_v1` | **INCOMPLETE** | >=3334/5000 | 0/5 | — | — | — |
| `bispf-optimized-quintic-v1` | `ddgan_repair_gaussian_chain_v1` | **FAIL** | 2500/2500 | 0/5 | 0.5905 | 0.6144 | [actual training](../../forge/ddgan-conditional-repair/media/bispf-optimized-quintic-v1-ddgan_repair_gaussian_chain_v1.gif) |
| `bispf-ddgan-energy-v1` | `ddgan_repair_gaussian_chain_v1` | **FAIL** | 2500/2500 | 0/5 | 0.0937 | 0.1289 | [actual training](../../forge/ddgan-conditional-repair/media/bispf-ddgan-energy-v1-ddgan_repair_gaussian_chain_v1.gif) |

CDF bounds: each final and conditional panel <=0.12. A PASS also requires all mean, covariance and bimodal mode/local-spread bounds at every final five scheduled checks. [Full failed bounds, provenance and exact final metrics](../ddgan-conditional-repair/results.json).
