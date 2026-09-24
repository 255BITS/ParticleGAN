# Open-cap rho tighten on frozen PR84

PR #92 (verdict B) on this torch build: useful acquisition is 674 steps, all with the cap closed, minimum rho .538, median 1.83. Clean HQ exits have median rho .217. Every open-cap exit has rho ≤ .25. A cut at rho < .538 catches 33/34 exits and 0/674 acquisition steps. The warm fail is 1129–1132 (196/200). The .080 step at 1129 is not the boundary particle. Update 1130 leaves with the cap already closed and margin about 0. Same-machine hold is 114/120.

The live rule does not use a margin proxy. PR84 still applies `min(1, 0.25 / rho)`. When rho ≤ 0.25 that factor is 1, so the extra scale is `rho / 0.25`. Steps with rho above .25, including the whole acquisition band, are unchanged. No rest-slope damp and no new threshold.

## Same-machine gates

torch 2.14.0+cpu, one prefix, constant rates .00425 / .00425 / .0085 after update 1000.

| Rule | Warm 1001–1200 | Min warm HQ | Hold every 10 to 2400 | Hold failures | Worst hold |
| --- | ---: | ---: | ---: | --- | --- |
| Scheduled identity | 200/200 | .990 | 120/120 | none | 8 / .998 |
| PR84, clip off | 196/200 | .866 | 114/120 | 1720, 1840, 1900, 2110, 2290, 2370 | 7 / .829 |
| Nearest-real exit clip | 200/200 | .939 | 117/120 | 1490, 1590, 2280 | 8 / .808 |
| Open-cap rho tighten | 200/200 | .959 | 115/120 | 1630, 1960, 2160, 2200, 2280 | 8 / .784 |

The nearest-real clip clears warm 200 and beats 114/120, so it stays on the record. It is not 120/120. The rho tighten also clears warm 200 and beats 114/120 (115), with a higher warm floor than the margin clip, then misses five later HQ checks. All five stay at 8 modes. 787 open-cap steps were scaled. Cold acquisition was not run.

## Kill

Neither rule holds 120/120. The rho band is the mechanism #92 justifies, and it still loses HQ after the warm window. Stop. No threshold sweep and no cold gate.
