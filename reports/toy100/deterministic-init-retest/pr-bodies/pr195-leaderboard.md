**KA2 is not qualified as the default. Search stopped. Draft, unmerged, targeting `develop`.**

**Public API leaderboard — new deterministic initialization**

All three tie at 0/24. Sorted by quick-screen result, passing checks, final streak and arrival; names order ties. Final clusters and HQ are diagnostics, not a tie-breaker.

| Rank | Entry | Screen | Passing checks | First arrival | Final streak | Final clusters / HQ | Why not the default |
|---:|---|---|---:|---:|---:|---|---|
| 1 | [Public k3p](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/evidence/public-k3p-new-init/archive-manifest.json) | FAIL | 0/24 | — | 0 | 6/8 · 97.09% | Quick screen fails; not qualified |
| 1 | [Public ka2-constant](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/evidence/public-ka2-constant-new-init/archive-manifest.json) | FAIL | 0/24 | — | 0 | 4/8 · 100.00% | Quick screen fails; not qualified |
| 1 | [Public ka2](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/evidence/public-ka2-new-init/archive-manifest.json) | FAIL | 0/24 | — | 0 | 6/8 · 99.93% | Quick screen fails; not qualified |

**Historical API stability — earlier initialization, sorted by pre-change retention**

| Rank | KA2 configuration | Pre-change retention | Updates to changed target | Passing checks after arrival | Why not the default |
|---:|---|---:|---:|---:|---|
| 1 | Decay diagnostic | 120/120 | 1,690 | 48/52 | No autonomous return from decay; later departures |
| 2 | Constant rates | 61/120 | 120 | 126/209 | Repeated loss of both distributions |

Recovery is arrival time followed by stability, without an 81/81 deadline requirement. Both historical runs continue through update 4,600.

| Qualification / validation | State |
|---|---|
| Continuous learner | Not demonstrated: no caller-selected end or phases; reversible autonomous rates allowed |
| Comparison with K3P | New quick screen fails for both; no KA2 release advantage established |
| New initialization | Develop `c720645e` merged; fresh networks and recipe-created priors use deterministic initialization; samples remain stochastic |
| Merge tests | 246 CPU tests passed; one CUDA-only test skipped in the authoritative CPU run |
| Old checkpoints | K3P schema 1–3 remain incompatible with KA2; use 0.8.0 for those checkpoints |
| Unfinished research lead | **SN3: PROMISING — UNFINISHED**; longer stability and public API checks unrun |
| Wider retest | 49 API + 75 research configurations complete; remaining work explicitly untested |
| Search / merge | Stopped; draft; unmerged; base `develop` |

[Sorted full leaderboard](https://github.com/255BITS/ParticleGAN/pull/155) · [Retest evidence and unrun checks](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/closeout.md) · [Earlier KA2 scores and rationale, preserved](https://github.com/255BITS/ParticleGAN/blob/codex/k3p-continuous-search/reports/toy100/deterministic-init-retest/pr-bodies/pr195-before-leaderboard.md).
