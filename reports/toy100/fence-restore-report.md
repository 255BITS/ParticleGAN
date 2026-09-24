# Fence restore on the nearest-real exit clip

Starts from the PR #91 nearest-real clip, not the open-cap rho tighten. Frozen PR84 stencil, bounds .25/3, alternating host. Rho is not an input. No rest-slope damp and no mode center.

PR #91 froze an outward step once nearest-real distance was already past the fence. Update 2280 still clipped (`min_scale` 0.0007) and graded HQ .866, because the particle stayed outside the HQ ball (radius 0.21; fence about 0.08). A closed cap does not exempt that particle.

After the unchanged .25 cap:

- A particle still inside whose step would cross the fence is shortened to the fence.
- A particle already past the fence whose step would increase nearest-real distance is placed on the fence along the ray to its nearest real.

## Same-machine gates

torch 2.14.0+cpu, one prefix, constant rates .00425 / .00425 / .0085 after update 1000.

| Rule | Warm 1001–1200 | Min warm HQ | Hold every 10 to 2400 | Hold failures | Worst hold |
| --- | ---: | ---: | ---: | --- | --- |
| Scheduled identity | 200/200 | .990 | 120/120 | none | 8 / .998 |
| PR84 | 196/200 | .866 | 114/120 | 1720, 1840, 1900, 2110, 2290, 2370 | 7 / .829 |
| Nearest-real exit clip (#91) | 200/200 | .939 | 117/120 | 1490, 1590, 2280 | 8 / .808 |
| Open-cap rho tighten (#91) | 200/200 | .959 | 115/120 | 1630, 1960, 2160, 2200, 2280 | 8 / .784 |
| Fence restore | 200/200 | .965 | 119/120 | 2160 | 8 / .891 |

Warm holds. Hold beats 117/120 and is not 120/120. Update 2160 has no exit-clip event. 2159 and 2161 do. The graded miss is not an outward crossing of the minibatch fence. Cold was not run.

## Stop

One extension. No threshold sweep. The remaining miss needs a read-only replay of update 2160 (nearest-real margin against the HQ radius) before any further rule.
