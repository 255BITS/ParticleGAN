Original 1,200-step scores remain FAIL. These are unchanged checkpoint continuations, not replacement screen scores.

| Candidate | First arrival | Retained since arrival | Departures | Minimum HQ | Final suffix |
|---|---:|---:|---|---:|---:|
| API-DV1-new-init | 1200 | 24/25 | 1750 | 0.854980 | 13 from 1800 |
| API-DV2-new-init | 1200 | 25/25 | None | 0.906982 | 25 from 1200 |
| API-DV3-new-init | 1150 | 26/26 | None | 0.909668 | 26 from 1150 |
| API-DV4-new-init | 1100 | 26/27 | 2000 | 0.888916 | 8 from 2050 |

All four retain eight modes throughout the post-arrival window. DV2 and DV3 have no failing observation through 2,400; DV1 and DV4 have one HQ departure each and subsequently recover.

Independent CPU-only deserialization verified saved tensor bytes and original placements against restore receipts, including native CPU Adam clocks and CUDA moments. Complete old endpoint live/EMA values, 1,200 extension batch receipts, 24 observations, unchanged recipes, source-derived adaptive rates and noise and source hashes match. No model construction/training or GPU initialization occurred. Original module-mode omission and lack of an uninterrupted future update-control comparison remain explicit limits.
