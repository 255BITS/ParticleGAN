# Smart descent — direct fit without a clock

Previously inspected toys are development data. Every candidate runs all nine toys. There is no external LR schedule or time feature. Live weights determine every score; EMA stays separate.

| Candidate | Evaluated toys | Live bounds | Sustained toys | Ring modes / HQ | Ring confirmation | Seconds |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| constant_control | 9/9 | 29/29 | 8/9 | 8/8 / 100.00% | — | 21.34 |
| standalone_g00_p01 | 9/9 | 27/29 | 8/9 | 5/8 / 66.77% | — | 20.66 |
| standalone_g00_p02 | 9/9 | 28/29 | 8/9 | 7/8 / 82.86% | — | 20.53 |
| standalone_g00_p03 | 9/9 | 26/29 | 7/9 | 5/8 / 56.91% | — | 20.00 |
| standalone_g00_p04 | 9/9 | 28/29 | 8/9 | 7/8 / 82.89% | — | 19.95 |
| standalone_g00_p05 | 9/9 | 27/29 | 8/9 | 5/8 / 73.85% | — | 20.29 |
| standalone_g00_p06 | 9/9 | 27/29 | 8/9 | 5/8 / 66.70% | — | 19.93 |
| standalone_g00_p07 | 9/9 | 28/29 | 8/9 | 6/8 / 90.99% | — | 20.18 |
| standalone_g01_p00 | 9/9 | 27/29 | 8/9 | 5/8 / 58.96% | — | 20.10 |
| standalone_g01_p02 | 9/9 | 26/29 | 7/9 | 4/8 / 67.55% | — | 20.24 |
| standalone_g01_p03 | 9/9 | 29/29 | 8/9 | 7/8 / 90.99% | — | 20.15 |
| standalone_g01_p04 | 9/9 | 27/29 | 7/9 | 8/8 / 100.00% | — | 20.43 |
| standalone_g01_p05 | 9/9 | 26/29 | 7/9 | 4/8 / 49.07% | — | 20.37 |
| standalone_g01_p06 | 9/9 | 27/29 | 8/9 | 4/8 / 56.91% | — | 20.31 |
| standalone_g01_p07 | 9/9 | 26/29 | 7/9 | 5/8 / 58.01% | — | 20.10 |
| standalone_g02_p00 | 9/9 | 27/29 | 8/9 | 6/8 / 65.82% | — | 20.15 |
| standalone_g02_p02 | 9/9 | 27/29 | 8/9 | 3/8 / 34.38% | — | 20.01 |
| standalone_g02_p03 | 9/9 | 27/29 | 8/9 | 3/8 / 33.28% | — | 19.88 |
| standalone_g02_p04 | 9/9 | 27/29 | 7/9 | 6/8 / 100.00% | — | 20.24 |
| standalone_g02_p05 | 9/9 | 27/29 | 8/9 | 5/8 / 58.42% | — | 20.27 |
| standalone_g02_p06 | 9/9 | 26/29 | 6/9 | 3/8 / 41.58% | — | 20.30 |
| standalone_g02_p07 | 9/9 | 29/29 | 8/9 | 7/8 / 100.00% | — | 20.17 |

All attempted policies, full curves, actions, errors and source hashes: [search.json.gz](search.json.gz).
Timing is one CPU observation including measurement and controller overhead.

No nonzero candidate sustained all nine development toys. The two newly reserved transfer cases were not evaluated. [Exact source](source.tar.gz).
