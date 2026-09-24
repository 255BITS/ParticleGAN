# Saved1325 finite-bank activation branches

The [read-only audit](pr84_finite_bank_activation_audit.py) reuses the source-bound 16-bank float64 field, captured Adam diagonal, and G-only captured direction from `pr84_finite_bank_vr_diagnostic.py`. Its [receipt](continuous-evidence/finite-bank-vr1325/activation-branches/result.json) binds the exact captured state, frozen-bank tensors, and source hashes. It performs no optimizer or host update and restores the global torch RNG.

| Centered secant fraction of captured G update | LeakyReLU sign changes | Activation inputs checked |
| --- | ---: | ---: |
| 1e-4 | 7 | 9,437,184 |
| 5e-5 | 3 | 9,437,184 |
| 1e-7 | 0 | 9,437,184 |
| 5e-8 | 0 | 9,437,184 |

The separate fine-scale audit already measured nearly equal G-only metric Rayleigh quotients (−0.137772409, −0.137772413) and fixed-metric alternating-map directional amplification (1.086325077, 1.086325064) at 1e-7 and 5e-8. This activation check explains why the coarse secants could cross different piecewise-linear branches; it does not identify which particular branch is responsible for the whole map's expansion. At the fine scales the recorded D and G own-bound factors also remain on the same branches. These are local derivatives of a **frozen finite-bank, fixed-metric** map, not of the live Adam moment-state map or the population game. The base fixed-bank map itself grades 8 modes with HQ 0.997314.
