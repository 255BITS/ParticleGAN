# Shared ParticleGAN score index

No configuration is eligible to ship, and no fair speed winner is established. This is a read-only navigation and comparison index of pinned reports, not a new certifier. The local JSON preserves source, law, denominator, original grade, study grade, costs and availability separately.

| Scope | Source / declared law | Comparable score | Status and cost | Team report |
|---|---|---|---|---|
| Ordinary MoG, current Forge | Origin `28990990`; live, scheduled, learned MoG prior σ = 0.025 | KA2 and K3P tie at 4/5 Tier 1; 19 Tier 2 and 2 Tier 3 requirements remain unknown | Both FAIL; qualified tier 0; no default or speed credit | [Primary ordinary board](https://github.com/255BITS/ParticleGAN/blob/4749b2780add539df4bd8d2dd1d3cc9f002f77ad/reports/forge/technique-inventory.md) |
| Public-policy search, four separate cohorts | `eb2d77fb`, `0335ecf0`, `53102174`, `8021a1c5`; each keeps its own source/spec/runtime lane | 32 whole configs × 8 cases = 256 cells; 42 reached full cases, 214 UNKNOWN | No fully qualified config; 42 original + 10 reviewed GIFs | [Primary policy board](https://github.com/255BITS/ParticleGAN/blob/4749b2780add539df4bd8d2dd1d3cc9f002f77ad/reports/forge/policy-family-inventory.md) |
| Original Atlas19 replay | `a0d6d89f`; original config / observers / 48,800 updates | 19/19 original questions PASS; clean native diagnostics FAIL | Paid 5210.638847 s, reserve 0; original-scope evidence only | [PR266 report](https://github.com/255BITS/ParticleGAN/blob/edc9d2e6e1227f8d7a1dd52e6e363b03ef871eb8/reports/forge/continuous-baseline-20261003/README.md) |
| C6 broad hold extension | Original `8021a1c5` parents; H2 runner `82c85cc3`; 150 appended updates per family | 0/2 extension PASS; both FAIL; old original PASS / old study INCOMPLETE retained | New paid 31.184628 s + H1 engineering 6.818678 s once | [PR266 hold readout](https://github.com/255BITS/ParticleGAN/blob/edc9d2e6e1227f8d7a1dd52e6e363b03ef871eb8/reports/forge/continuous-baseline-20261003/README.md) |
| Critic-rate contrast (D2.25) | `a956c6fc`; LR .0053125, prior 1.5, D 2.25 | 2 full intensity FAIL, 14 UNKNOWN; original 2 FAIL and study 2 FAIL; Q1 16/16 SUPPORTED | Science 54.358772 s + engineering 4.757909 s = 59.116680 s; reserve 0 | [PR267 readout](https://github.com/255BITS/ParticleGAN/blob/dc3256b61042a5aa2193bc5dfc16799c15521521/reports/forge/critic-balance-20261003/README.md) |
| Generator-half contrast | `488b792e`; LR .00265625, prior 3, D 4.5 | 2 full intensity FAIL, 14 UNKNOWN; original 2 FAIL and study 2 FAIL; Q1 16/16 SUPPORTED | New science 54.877571 s; prior 59.116680 s once; campaign 113.994252 s; reserve 0 | [PR268 readout](https://github.com/255BITS/ParticleGAN/blob/95bc3480/reports/forge/generator-step-20261003/README.md) |
| Next KA2/K3P named-family cohort | LR .006375, prior 1, D 1; fast / no DV12 / fixed σ warmup / scheduled / AMSGrad false | UNEXECUTED at this boundary; 16 scientific cells UNKNOWN; no new Q1 or learned credit here | Remaining pair 15,246.005748 s; cost slots inherited, evidence never inherited | Authors preparing a separate frozen cohort |

The original Atlas19 image checks require full mode recovery and HQ ≥ 0.9; TV is diagnostic there. Current API image checks also gate finite-template TV. Original Atlas19 native evidence has 34 observations, final-five 20k checks and an independent 100k holdout. The current API native cohort has 24 full-count 20k observations and final-five checks, with noisy selected-policy primary samples and output-noise-off samples from that same selected policy as diagnostics. It has no separately forced EMA branch or independent 100k holdout. Those distinctions prevent cross-filling requirements.

## Ordinary MoG selected rows

Only rows inside the same declared comparison cohort can be ordered by their required PASS counts. These are imported selected rows, not a new sweep. All 47 configuration rows remain in the linked primary JSON. A failure or unknown downstream case is not erased by a higher pass count.

| Compatible selected rows | Tier 1 PASS / required | Tier 1 other status | Tier 2 / Tier 3 | Overall |
|---|---|---|---|---|
| ka2, k3p | 4/5 | 1 FAIL | 0/19 and 0/2; all UNKNOWN | FAIL, qualified tier 0 |
| bcap, r1r2, release07-gan-v3-mog | 3/5 | 1 FAIL, 1 UNKNOWN | 0/19 and 0/2; all UNKNOWN | FAIL, qualified tier 0 |
| Atlas, E22 canonical ordinary rows | 0/5 | BLOCKED | BLOCKED | No measured ordinary qualification |
| Other registered controls / source-only rows | See primary row | FAIL or BLOCKED; not independently ranked here | Unknown or blocked | No qualification |

## Whole-config policy selections within each source cohort

The primary board selects on study PASS counts, not endpoint appearance or speed. A display tie-break is only a display choice. C6 LR .0053125 / prior 1.5 is the useful original-gate baseline for the hold question, although the imported Atlas display tie chooses a different tuple. Both C6 smoke originals PASS; broad acquisition is too late for five later checks, so the whole config remains INCOMPLETE with six later domains UNKNOWN.

| Source cohort | Family | Imported best-observed tuple (LR / prior) | Original / 8 | Study / 8 | Unknown / 8 | Whole-config status |
|---|---|---|---|---|---|---|
| `eb2d77fb` | atlas | 0.006375 / 2.0 | 1/8 | 1/8 | 6/8 | FAIL |
| `eb2d77fb` | e22 | 0.006375 / 2.0 | 1/8 | 1/8 | 6/8 | FAIL |
| `0335ecf0` | atlas | 0.002125 / 1.0 | 1/8 | 0/8 | 7/8 | INCOMPLETE |
| `0335ecf0` | e22 | 0.002125 / 2.0 | 1/8 | 0/8 | 7/8 | INCOMPLETE |
| `53102174` | atlas | 0.0031875 / 1.0 | 1/8 | 1/8 | 6/8 | FAIL |
| `53102174` | e22 | 0.0053125 / 1.0 | 1/8 | 1/8 | 6/8 | FAIL |
| `8021a1c5` | atlas | 0.006375 / 1.5 | 1/8 | 1/8 | 6/8 | FAIL |
| `8021a1c5` | e22 | 0.0053125 / 1.5 | 2/8 | 1/8 | 6/8 | INCOMPLETE |

The JSON retains every one of the 32 tuples and every required denominator. Neither the four policy cohorts nor the newer two contrasts are pooled into a synthetic eight-case success. Atlas/E22 numerical matches on some small hosts do not establish complete algorithm equivalence; selected density paths and controller state remain source-owned.

## Evidence availability and accounting

The original/policy primary boards above are pinned to the last verified develop boundary `4749b278`. PR266/267/268 compact results are linked to their report branches and are not certified as merged to develop by this index. No fresh GitHub state or CI query was made. The bounded open-PR inventory covered only the newest 100 updated open rows; older open PRs and PR255’s current production diff remain unverified. PR255 was untouched.

| Raw archive | Availability | Bytes | SHA-256 |
|---|---|---|---|
| critic_rate | LOCAL_ONLY; not remotely replicated | 64,385,103 | `292061adad594577d9186207234929afa42437cdf9e5539128d9ca4865a7b533` |
| generator_half | LOCAL_ONLY; not remotely replicated | 111,262,244 | `271cf1c21a341d33d520311160c3d9c03f3ffcb6094dd544602e2084089c8949` |

Reported paid seconds are supervisory attempt intervals. Reservation is separate, and CPU capacity construction/replay/verification is diagnostic work outside ordinary learning credit. The D2.25 engineering startup is charged once in its own readout and once as prior cost within the generator-half cumulative campaign; do not add that cumulative figure to the earlier total again. Ordinary selected-row wall times and historical replay costs belong to different declared studies and are not summed into this campaign or used as a fair speed ranking.

All inputs and their hashes are in [index.json](index.json). This projection performed zero training updates, sampler calls, metric rescoring, gate changes or remote queries. The existing primary reports and their source-bound receipts remain authoritative.
