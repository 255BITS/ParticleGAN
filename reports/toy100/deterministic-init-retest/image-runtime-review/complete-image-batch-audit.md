All12 completed image runs passed independent source, initialized-state, retained-sampling, optimizer and score audits. RP12, RP14 and RP15 each pass intensity, blobs and stripes, and each fails bars with0/24 observations. They do not qualify as a default.

| Candidate | Task | Result | Passing | First arrival | Departures | Final suffix |
|---|---|---|---:|---:|---|---:|
| API-RP12-new-init | img_bars4 | FAIL | 0/24 | — | None | 0 |
| API-RP12-new-init | img_blobs4 | PASS | 17/24 | 200 | None | 17 |
| API-RP12-new-init | img_intensity2 | PASS | 12/24 | 325 | None | 12 |
| API-RP12-new-init | img_stripes2 | PASS | 17/24 | 200 | None | 17 |
| API-RP14-new-init | img_bars4 | FAIL | 0/24 | — | None | 0 |
| API-RP14-new-init | img_blobs4 | PASS | 6/24 | 475 | None | 6 |
| API-RP14-new-init | img_intensity2 | PASS | 9/24 | 375 | 475 | 5 |
| API-RP14-new-init | img_stripes2 | PASS | 13/24 | 300 | None | 13 |
| API-RP15-new-init | img_bars4 | FAIL | 0/24 | — | None | 0 |
| API-RP15-new-init | img_blobs4 | PASS | 17/24 | 200 | None | 17 |
| API-RP15-new-init | img_intensity2 | PASS | 5/24 | 500 | None | 5 |
| API-RP15-new-init | img_stripes2 | PASS | 21/24 | 75 | 225 | 15 |

All runs retain exactly600 accepted public updates and24 observations, unchanged candidate packages, frozen task thresholds and five-check suffix. Each update uses two game fields. Costs, rates, controller state, live/EMA measurements and eager parameter-device optimizer counters are preserved. An additional complete-state comparison confirms actual initialization equals each own CPU proof for model tensors/buffers, precision reference/state, optimizer moments and all non-RNG values, ignoring declared device placement only; independent RNG/cursor checks remain in individual audits.

The source asserts every actual batch/cursor against the retained600-row dry stream. This artifact audit did not rerun models or sampling. Final checkpoints are preserved/pinned but no fresh-process continuation is claimed. No older initialization scores are replaced or inherited. Existing nine-case batch is complete; no further precision GPU qualification is warranted after the measured bars failures.
