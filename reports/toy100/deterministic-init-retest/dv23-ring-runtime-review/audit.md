# DV2 / DV3 completed ring followups

Both new-init runs finish all 4,600 updates. Neither reaches eight modes and HQ≥.90 before the target changes at 2,400. After the change, DV2 arrives in 380 updates and retains 183/183 observations; DV3 arrives in 500 and retains 171/171. No post-arrival departure occurs. These finite observations establish later adaptation and retention, but leave original-distribution acquisition unverified. No 81-observation deadline or claim of impossible eventual convergence is used.

| Candidate | Initialization | Original arrival | Original retention since arrival | Shifted arrival delay | Shifted retention | Minimum shifted HQ |
|---|---|---:|---:|---:|---:|---:|
| API-DV2-new-init | Historical initialization | 790 | 159/162 | 400 | 181/181 | 0.902344 |
| API-DV2-new-init | New deterministic initialization | not reached | 0/0 | 380 | 183/183 | 0.907959 |
| API-DV3-new-init | Historical initialization | 790 | 153/162 | 520 | 169/169 | 0.907471 |
| API-DV3-new-init | New deterministic initialization | not reached | 0/0 | 500 | 171/171 | 0.917725 |

Each own historical run arrived on the original distribution at 790: DV2 then missed 1390, 1420, 1430; DV3 missed 1390–1440 and 1470–1490. Their old shifted arrivals were 400/520 updates, each followed by uninterrupted retained quality. New initialization preserves the sampled streams while changing the initial learned weights; the 20-update earlier shifted arrivals do not establish an overall win because initial acquisition regressed. Both frozen-at-change controls fail all 220 shifted observations.

All 24 isolated package files and the reviewed worker seal match. Actual raw initial model/EMA/prior and controller/optimizer state matches each own CPU proof; all 17 raw Adam clocks at 2,400/4,600 remain CPU scalars with CUDA moments. The main run is never reloaded; the separate frozen copy restores constructor-modified global RNG and is guarded against main/caller state changes. All 460 scores,4,600 adaptive-rate/noise rows, 46 original sampling boundaries and initial/change/final raw state receipts verify. Detector evidence traces also match each own historical run at all 4,600 steps.

Rates vary with data evidence and gradient alignment; total_steps=None and constant output noise. These are horizon-free adaptive rates, not literal constant rates. The source-only audit does not regenerate samples, invent missing per-update batch receipts or prove fresh-process continuation. Stationary/long/full22 qualification remains absent on this initialization. No additional runs are authorized by this report.

All terminal files, including the three raw checkpoints, are archived losslessly under supplementary-evidence/api-dv{2,3}-single-shift-new-init. [Standalone manifest](../ring-followup-results.json) and [full audit](audit.json) retain exact hashes and limits. Existing score manifests and original results remain unchanged.
