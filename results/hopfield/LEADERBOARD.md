# Hopfield read leaderboard

One seed per arm; no seed averaging or uncertainty estimate. Ranked by final TV within each dataset and particle count.

| Dataset | M | Read | Seed | Modes | HQ | σ ratio | TV ↓ | KL ↓ | Steps to TV | max_w | eff_n | dead_frac | interp_hq | log β | Quality floor |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| imbalanced | 100 | uniform | 1234 | 45 | 0.8316 | 0.0580 | 0.2121 | 1.2231 | >7000 | N/A | N/A | N/A | N/A | N/A | fail |
| imbalanced | 100 | Hopfield β=16 | 1234 | 38 | 0.6352 | 6.1938 | 0.4229 | 2.4481 | >7000 | 0.9407 | 1.2012 | 0.7000 | 0.5908 | 2.7726 | fail |
