# Frozen PR84 open-cap diagnosis

Diagnostic only. The stencil, bounds .25/3, and alternating host are unchanged. No gate was relaxed.

**Verdict B.** open-cap rho is below every useful acquisition step; one closed-cap exit sits just above the acquisition minimum.

Not a production result. The stencil and the .25/3 bounds are unchanged, and the continuation gate still fails.

## Environment

- torch `2.14.0+cpu` python `3.12.3`
- warm-state sha256 `57c10b571a2538075ea842cd0c9ea6f0ff677062ae9287e8e3c21d1241c48861`
- candidate adapter sha256 `9c17b1b02f79201ae5392b2b04e11570e08bbbf0d620794861e436b105f0111f`
- first continuation failure: `{'step': 1129, 'modes': 8, 'hq': 0.898682}`
- warm local `{'checks': 200, 'passing_checks': 196, 'failing_steps': [1129, 1130, 1131, 1132], 'min_modes': 8, 'min_hq': 0.8662109375, 'passing_suffix': 68, 'stable_from_step': 1133, 'pass_all': False, 'pass_suffix': True}`
- warm hold `{'checks': 120, 'passing_checks': 114, 'failing_steps': [1720, 1840, 1900, 2110, 2290, 2370], 'min_modes': 7, 'min_hq': 0.829345703125, 'passing_suffix': 3, 'stable_from_step': None, 'pass_all': False, 'pass_suffix': False}`
- cold ring on this build: `PASS` (terminal metrics [{'metric': 'modes', 'value': 8, 'op': '>=', 'threshold': 8, 'status': 'PASS', 'margin': 0}, {'metric': 'hq', 'value': 0.998779296875, 'op': '>=', 'threshold': 0.9, 'status': 'PASS', 'margin': 0.09877929687499998}]). That pass is this torch build only. It does not replace the archived seven-mode ring.

## Rho bands

Rho is the scalar same-sample G own-curvature. The cap is open when factor = 1 (rho ≤ .25). A destructive exit is a clean particle that is inside the HQ ball before the generator step and outside after it. A useful acquisition step still has fewer than 8 clean modes and moves at least one particle closer to an uncovered mode.

Separator: useful rho minimum is 0.5381. 33 of 34 exits are strictly below that minimum. Steps with rho ≤ .25: 23 exits and 0 useful acquisition steps.

| Population | Steps | Cap open | Rho |
| --- | ---: | ---: | --- |
| All clean HQ exits | 34 | 23 | n=34 min=0.0956 p10=0.1224 p50=0.2169 p90=0.4234 max=0.5538 |
| Open-cap exits only | 23 | 23 | n=23 min=0.0956 p10=0.1082 p50=0.1728 p90=0.2330 max=0.2499 |
| Closed-cap exits | 11 | 0 | n=11 min=0.2697 p10=0.2738 p50=0.3323 p90=0.4963 max=0.5538 |
| Useful cold-ring steps toward an uncovered mode | 674 | 0 | n=674 min=0.5381 p10=0.9470 p50=1.8263 p90=3.7422 max=11.2920 |

### Histogram (bin width .05)

| Bin | Destructive exits | Useful acquisition |
| --- | ---: | ---: |
| [0.05, 0.10) | 1 | 0 |
| [0.10, 0.15) | 6 | 0 |
| [0.15, 0.20) | 8 | 0 |
| [0.20, 0.25) | 8 | 0 |
| [0.25, 0.30) | 5 | 0 |
| [0.30, 0.35) | 1 | 0 |
| [0.40, 0.45) | 3 | 0 |
| [0.45, 0.50) | 1 | 0 |
| [0.50, 0.55) | 0 | 1 |
| [0.55, 0.60) | 1 | 1 |
| [0.60, 0.65) | 0 | 2 |
| [0.65, 0.70) | 0 | 5 |
| [0.70, 0.75) | 0 | 3 |
| [0.75, 0.80) | 0 | 9 |
| [0.80, 0.85) | 0 | 15 |
| [0.85, 0.90) | 0 | 17 |
| [0.90, 0.95) | 0 | 16 |
| [0.95, 1.00) | 0 | 11 |
| [1.00, inf) | 0 | 594 |

## First graded failure

Update 1129: eval modes 8, eval HQ 0.898682.
Scalar rho 0.170922, factor 1.0, cap_open True.
Clean before 8 modes / HQ 1.0; proposal 8 / 1.0; after 8 / 1.0.
Max proposed output move 0.079497, max applied 0.079497, closest margin before 0.010166.
Clean particles that leave HQ: 0. Particles whose unbounded proposal would leave: 0.

Proposal and applied output motion are equal on this step because the cap is open. The largest applied move is not the particle nearest the boundary.

| i | margin before | proposal margin | after margin | proposed | applied | left | cap |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 0.0469 | 0.0182 | 0.0182 | 0.0302 | 0.0302 | False | True |
| 1 | 0.0102 | 0.0000 | 0.0000 | 0.0108 | 0.0108 | False | True |
| 2 | 0.1872 | 0.1243 | 0.1243 | 0.0651 | 0.0651 | False | True |
| 3 | 0.0708 | 0.0724 | 0.0724 | 0.0031 | 0.0031 | False | True |
| 4 | 0.1793 | 0.1527 | 0.1527 | 0.0795 | 0.0795 | False | True |
| 5 | 0.0901 | 0.0365 | 0.0365 | 0.0537 | 0.0537 | False | True |
| 6 | 0.0486 | 0.0228 | 0.0228 | 0.0264 | 0.0264 | False | True |
| 7 | 0.0867 | 0.0874 | 0.0874 | 0.0015 | 0.0015 | False | True |
| 8 | 0.0670 | 0.0480 | 0.0480 | 0.0194 | 0.0194 | False | True |
| 9 | 0.1434 | 0.1353 | 0.1353 | 0.0100 | 0.0100 | False | True |
| 10 | 0.1603 | 0.1560 | 0.1560 | 0.0072 | 0.0072 | False | True |
| 11 | 0.1351 | 0.1274 | 0.1274 | 0.0109 | 0.0109 | False | True |

The next clean exit is update 1130: rho 0.421452, factor 0.593187, cap_open False, margin before 3.3e-05, proposed max 0.043936, applied max 0.026067. Clean HQ 1.0 → 0.9166666865348816 (8 modes).

| i | margin before | proposal margin | after margin | proposed | applied |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0000 | -0.0088 | -0.0052 | 0.0106 | 0.0063 |

## Gate failures

| Step | Modes | HQ |
| ---: | ---: | ---: |
| 1129 | 8 | 0.898682 |
| 1130 | 8 | 0.866211 |
| 1131 | 8 | 0.883301 |
| 1132 | 8 | 0.885742 |
| 1720 | 8 | 0.886963 |
| 1840 | 8 | 0.893066 |
| 1900 | 7 | 0.915039 |
| 2110 | 8 | 0.833252 |
| 2290 | 7 | 0.829346 |
| 2370 | 7 | 0.914795 |

