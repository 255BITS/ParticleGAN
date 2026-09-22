# Smart descent v2 — full-suite LR-only search

Previously inspected toys are development data. Every new policy runs all nine toys. The cosine control is reused from the first stage with a pinned parent snapshot. Live weights determine every score; EMA stays separate.

| Candidate | Evaluated toys | Live bounds | Sustained toys | Ring modes / HQ | Ring confirmation | Seconds |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| cosine_control | 9/9 | 29/29 | 9/9 | 8/8 / 100.00% | 1050 | 25.62 |
| lr_g00_p02 | 9/9 | 27/29 | 8/9 | 6/8 / 82.71% | — | 30.39 |
| lr_g00_p03 | 9/9 | 28/29 | 8/9 | 6/8 / 90.99% | — | 28.71 |
| lr_g00_p04 | 9/9 | 28/29 | 8/9 | 7/8 / 82.96% | — | 27.07 |
| lr_g00_p05 | 9/9 | 27/29 | 8/9 | 5/8 / 75.29% | — | 26.68 |
| lr_g00_p06 | 9/9 | 27/29 | 8/9 | 4/8 / 59.18% | — | 29.05 |
| lr_g00_p07 | 9/9 | 28/29 | 7/9 | 8/8 / 100.00% | — | 28.74 |
| lr_g00_p08 | 9/9 | 29/29 | 9/9 | 8/8 / 90.99% | 1150 | 25.91 |
| lr_g00_p09 | 9/9 | 27/29 | 8/9 | 6/8 / 74.73% | — | 27.41 |
| lr_g00_p10 | 9/9 | 27/29 | 8/9 | 6/8 / 66.92% | — | 28.73 |
| lr_g00_p11 | 9/9 | 29/29 | 9/9 | 8/8 / 91.67% | 1000 | 27.37 |
| lr_g01_p00 | 9/9 | 28/29 | 8/9 | 6/8 / 91.58% | — | 25.72 |
| lr_g01_p02 | 9/9 | 27/29 | 8/9 | 3/8 / 68.14% | — | 26.20 |
| lr_g01_p03 | 9/9 | 27/29 | 8/9 | 4/8 / 65.62% | — | 22.81 |
| lr_g01_p04 | 9/9 | 27/29 | 8/9 | 6/8 / 83.06% | — | 25.98 |
| lr_g01_p05 | 9/9 | 29/29 | 9/9 | 8/8 / 100.00% | 1150 | 23.06 |
| lr_g01_p06 | 9/9 | 27/29 | 8/9 | 5/8 / 41.21% | — | 22.58 |
| lr_g01_p07 | 9/9 | 27/29 | 8/9 | 5/8 / 65.60% | — | 22.40 |
| lr_g01_p08 | 9/9 | 28/29 | 8/9 | 5/8 / 91.67% | — | 21.62 |
| lr_g01_p09 | 9/9 | 28/29 | 8/9 | 7/8 / 83.64% | — | 21.36 |
| lr_g01_p10 | 9/9 | 29/29 | 8/9 | 7/8 / 100.00% | — | 21.04 |
| lr_g01_p11 | 9/9 | 27/29 | 8/9 | 3/8 / 41.41% | — | 20.91 |

All attempted policies, full curves, actions, errors and source hashes: [search.json.gz](search.json.gz).
Timing is one CPU observation including measurement and controller overhead.

Exact fingerprinted source files: [source.tar.gz](source.tar.gz). Raw JSON is compressed without changing its bytes.
