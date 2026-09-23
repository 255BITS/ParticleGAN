# Smart descent — stronger clock-free damping

Previously inspected toys are development data. Ring failures are screened before the remaining eight toys; screened rows are incomplete and cannot receive an overall PASS. Live weights determine every score; EMA stays separate.

| Candidate | Evaluated toys | Live bounds | Sustained toys | Ring modes / HQ | Ring confirmation | Seconds |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| constant_control | 9/9 | 29/29 | 8/9 | 8/8 / 100.00% | — | 21.34 |
| growth_0p3_g | 1/9 | 1/29 | 0/9 | 7/8 / 82.30% | — | 7.99 |
| growth_0p3_d | 1/9 | 0/29 | 0/9 | 5/8 / 82.89% | — | 6.77 |
| growth_0p3_both | 1/9 | 1/29 | 0/9 | 7/8 / 83.03% | — | 6.82 |
| growth_0p7_g | 1/9 | 2/29 | 0/9 | 8/8 / 90.99% | — | 6.67 |
| growth_0p7_d | 1/9 | 0/29 | 0/9 | 4/8 / 58.69% | — | 6.92 |
| growth_0p7_both | 1/9 | 0/29 | 0/9 | 4/8 / 49.27% | — | 6.68 |
| innovation_0p3_g | 1/9 | 0/29 | 0/9 | 5/8 / 67.97% | — | 6.77 |
| innovation_0p3_d | 1/9 | 2/29 | 0/9 | 7/8 / 91.67% | — | 6.81 |
| innovation_0p3_both | 1/9 | 2/29 | 0/9 | 8/8 / 100.00% | — | 6.76 |
| innovation_0p7_g | 1/9 | 0/29 | 0/9 | 5/8 / 67.70% | — | 6.87 |
| innovation_0p7_d | 1/9 | 0/29 | 0/9 | 3/8 / 49.10% | — | 6.65 |
| innovation_0p7_both | 1/9 | 2/29 | 0/9 | 8/8 / 91.75% | — | 6.86 |

All attempted policies, full curves, actions, errors and source hashes: [search.json.gz](search.json.gz).
Timing is one CPU observation including measurement and controller overhead.

No nonzero candidate sustained all nine development toys. The two newly reserved transfer cases were not evaluated. [Exact source](source.tar.gz).
