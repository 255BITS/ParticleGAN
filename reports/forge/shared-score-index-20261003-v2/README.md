# Shared ParticleGAN score index

No configuration is eligible to ship, and no fair speed winner is established. This is a read-only navigation and comparison index of pinned reports, not a new certifier. The local JSON preserves source, law, denominator, original grade, study grade, costs and availability separately. V2 retains all eight earlier input hashes and adds the completed, certified KA2/K3P named-family publication; the predecessor index remains unchanged.

| Scope | Source / declared law | Comparable score | Status and cost | Team report |
|---|---|---|---|---|
| Ordinary MoG, current Forge | Origin `28990990`; live, scheduled, learned MoG prior σ = 0.025 | KA2 and K3P tie at 4/5 Tier 1; 19 Tier 2 and 2 Tier 3 requirements remain unknown | Both FAIL; qualified tier 0; no default or speed credit | [Primary ordinary board](https://github.com/255BITS/ParticleGAN/blob/4749b2780add539df4bd8d2dd1d3cc9f002f77ad/reports/forge/technique-inventory.md) |
| Public-policy search, four separate cohorts | `eb2d77fb`, `0335ecf0`, `53102174`, `8021a1c5`; each keeps its own source/spec/runtime lane | 32 whole configs × 8 cases = 256 cells; 42 reached full cases, 214 UNKNOWN | No fully qualified config; 42 original + 10 reviewed GIFs | [Primary policy board](https://github.com/255BITS/ParticleGAN/blob/4749b2780add539df4bd8d2dd1d3cc9f002f77ad/reports/forge/policy-family-inventory.md) |
| Original Atlas19 replay | `a0d6d89f`; original config / observers / 48,800 updates | 19/19 original questions PASS; clean native diagnostics FAIL | Paid 5210.638847 s, reserve 0; original-scope evidence only | [PR266 report](https://github.com/255BITS/ParticleGAN/blob/edc9d2e6e1227f8d7a1dd52e6e363b03ef871eb8/reports/forge/continuous-baseline-20261003/README.md) |
| C6 broad hold extension | Original `8021a1c5` parents; H2 runner `82c85cc3`; 150 appended updates per family | 0/2 extension PASS; both FAIL; old original PASS / old study INCOMPLETE retained | New paid 31.184628 s + H1 engineering 6.818678 s once | [PR266 hold readout](https://github.com/255BITS/ParticleGAN/blob/edc9d2e6e1227f8d7a1dd52e6e363b03ef871eb8/reports/forge/continuous-baseline-20261003/README.md) |
| Critic-rate contrast (D2.25) | `a956c6fc`; LR .0053125, prior 1.5, D 2.25 | 2 full intensity FAIL, 14 UNKNOWN; original 2 FAIL and study 2 FAIL; Q1 16/16 SUPPORTED | Science 54.358772 s + engineering 4.757909 s = 59.116680 s; reserve 0 | [PR267 readout](https://github.com/255BITS/ParticleGAN/blob/dc3256b61042a5aa2193bc5dfc16799c15521521/reports/forge/critic-balance-20261003/README.md) |
| Generator-half contrast | `488b792e`; LR .00265625, prior 3, D 4.5 | 2 full intensity FAIL, 14 UNKNOWN; original 2 FAIL and study 2 FAIL; Q1 16/16 SUPPORTED | New science 54.877571 s; prior 59.116680 s once; campaign 113.994252 s; reserve 0 | [PR268 readout](https://github.com/255BITS/ParticleGAN/blob/95bc3480/reports/forge/generator-step-20261003/README.md) |
| Closed KA2/K3P named-family defaults | `26ff278c`; LR .006375, prior 1, D 1; fast / no DV12 / fixed σ warmup / scheduled / AMSGrad false | 2 full intensity FAIL, 14 UNKNOWN; original 2 FAIL and study 2 FAIL; cold Q1 16/16 SUPPORTED | New science 27.193673 s; prior 113.994252 s once; campaign 141.187925 s; reserve 0 | [Named-family report](../ka2-k3p-defaults-20261003/README.md) |
| Next named-family prior2 candidate | LR .006375, prior 2, D 1; separate future frozen cohort | ACCEPTED, UNEXECUTED; 16 scientific cells UNKNOWN; no new Q1 or learned credit | No new paid attempt; 15,218.812075 s unspent campaign ceiling | Source-law review and freezing still pending |

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

The JSON retains every one of the 32 tuples and every required denominator. Neither the four policy cohorts nor the newer contrasts are pooled into a synthetic eight-case success. Atlas/E22 numerical matches on some small hosts do not establish complete algorithm equivalence; selected density paths and controller state remain source-owned.

## Closed named-family comparison within source 26ff

This is a separate finite comparison of KA2 and K3P's own public-API laws, not an Atlas/E22 or ordinary-MoG row. Both use fast-only serving, no DV12, AMSGrad false and fixed output sigma warmed from zero to 0.029 over 20% of the original per-host 600/1200/7000 schedule. Image/vector primary evaluation has output noise off; native primary evaluation would retain output noise. The current image finite-template TV gate remains unchanged.

| Whole configuration | Original / 8 | Added study / 8 | Remaining / 8 | Capacity | Measured science | Original goal GIF |
|---|---|---|---|---|---|---|
| KA2, .006375 / 1 / 1 | 0 PASS, 1 FAIL | 0 PASS, 1 FAIL | 7 UNKNOWN | 8 SUPPORTED, clock-zero σ = 0 | 13.617583 s | [Intensity recovery](../ka2-k3p-defaults-20261003/media/ka2/image-develop-img_intensity2-source-transpose12/goal.gif) |
| K3P, .006375 / 1 / 1 | 0 PASS, 1 FAIL | 0 PASS, 1 FAIL | 7 UNKNOWN | 8 SUPPORTED, clock-zero σ = 0 | 13.576090 s | [Intensity recovery](../ka2-k3p-defaults-20261003/media/k3p/image-develop-img_intensity2-source-transpose12/goal.gif) |

Both ran the complete 600-update intensity protocol; no passing five-check acquisition window formed. They tie at 0/8 added-study PASS. The subsequent broad, grid100, fixed-rotated100, staggered100, unequal-mass, anisotropic and bars cases remain UNKNOWN for each tuple. A concluded finite request or imported `comparison_complete` flag does not certify these unknown cells. The partial template recovery visible in the two actual GIFs does not qualify either configuration.

At the terminal check, both raw nearest-template frequency TV values are 0.098633. KA2's strict finite-template TV is 0.128906 (rejected mass 0.030273), so its original gate FAILs despite HQ 0.969727. K3P has finite-template TV 0.202148 and HQ 0.896484, failing both the TV ≤ 0.1 and HQ ≥ 0.9 requirements. These are imported recorded metrics, not rescored images. The original inherited GIF caption mentions retained latent perturbation; the actual named-family law is **fast-only/no DV12**, with recorded perturbation and added output-noise flags zero in this image cohort.

All 16 Q1 records are clock-zero, sigma-zero capacity witnesses. They grant no learned PASS, terminal-noise solvability, independent confirmation, robustness or current ordinary-MoG credit. The accepted next .006375 / 2 / 1 tuple remains unexecuted in this index and inherits cost bookkeeping only.

The parent independently completed CPU capacity replay and retained numeric certification. The [published report](../ka2-k3p-defaults-20261003/README.md) and [results JSON](../ka2-k3p-defaults-20261003/results.json) bind 13,945 input identities, two original GIFs, scientific source `26ff278c`, publisher `5a74fc24`, combined SHA `f55d2e75…` and certification SHA `e10765e3…`. Full hashes are preserved in [index.json](index.json). This index only checks pins and projects stored outcomes.

## Evidence availability and accounting

The original/policy primary boards above are pinned to the last verified develop boundary `4749b278`. PR266/267/268 compact results are linked to their report branches and are not certified as merged to develop by this index. The parent is preparing a new PR for the closed named-family report; its relative links target that sibling report directory. This index makes no fresh CI or merge claim for that report, and no current named-family archive card has been supplied at this boundary. No fresh GitHub state or CI query was made. The bounded open-PR inventory covered only the newest 100 updated open rows; older open PRs and PR255’s current production diff remain unverified. PR255 was untouched.

| Raw archive | Availability | Bytes | SHA-256 |
|---|---|---|---|
| critic_rate | LOCAL_ONLY; not remotely replicated | 64,385,103 | `292061adad594577d9186207234929afa42437cdf9e5539128d9ca4865a7b533` |
| generator_half | LOCAL_ONLY; not remotely replicated | 111,262,244 | `271cf1c21a341d33d520311160c3d9c03f3ffcb6094dd544602e2084089c8949` |

Reported paid seconds are supervisory attempt intervals. Reservation is separate, and CPU capacity construction/replay/verification is diagnostic work outside ordinary learning credit. The current named-family science adds 27.193673191126436 s to the prior 113.99425188452005 s once, for a cumulative 141.1879250756465 s and zero interruption reserve within the original 15,360 s ceiling. The prior cost already contains 109.23634317959659 s of earlier science and the 4.757908704923466 s D2.25 startup error. Do not add earlier cumulative campaign rows again. No named-family raw archive location, digest or remote availability is inferred until its archive card is supplied. Ordinary selected-row wall times and historical replay costs belong to different declared studies and are not summed into this campaign or used as a fair speed ranking.

All nine report/input pins and predecessor hashes are in [index.json](index.json). The eight older pins were rechecked byte-exact, and the new results JSON is 136,259 bytes with SHA-256 `a35cccc85c1774dca3e301ba73e493943ba31f3562ea52a97ec25fc7a7b97047`. This projection performed zero training updates, sampler calls, metric rescoring, gate changes or remote queries. The existing primary reports and their source-bound receipts remain authoritative.
