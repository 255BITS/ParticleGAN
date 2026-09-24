# Clump HQ ball only in the open-cap exit band

One change on the PR #94 fence-restore clip. The spacing fence is unchanged. The minibatch clump HQ ball runs only when generator rho is at most `0.25`. High-rho steps stay on PR84. No rest-slope damp, no mode catalog, no gate change.

## Why this cut

PR #92: open-cap exits have rho ≤ 0.25, and useful acquisition has rho ≥ 0.538 (median about 1.83). A cut at rho < 0.538 separates those two archived populations, but the known hold miss that the always-on ball was built for is update 2160 at rho 0.672, which is above both cuts. Widening the ball to 0.538 would not cover that step, so the single choice is the open-cap band. No second threshold was run.

The always-on ball (PR #94) held 120/120 and then fired on 1195 of 1200 cold-ring updates, leaving 6 modes. Those acquisition updates are the high-rho band this gate leaves alone.

## Same-machine gates

torch 2.14.0+cpu, one thread, AVX2. Scheduled prefix through update 1000 (8 modes, HQ .997), then constant rates `.00425 / .00425 / .0085`. Fail-fast: stop at the first missed warm or hold check. Cold was not run.

| Rule | Warm | Min HQ | Hold to 2400 | Worst hold |
| --- | ---: | ---: | ---: | --- |
| PR84 (#84) | 196/200 | .866 | 114/120 | 7 / .829 |
| Fence restore (#93) | 200/200 | .965 | 119/120 | 8 / .891 |
| Always-on clump HQ ball (#94) | 200/200 | .965 | **120/120** | 8 / .911 |
| Clump ball only if rho ≤ .25 | 200/200 | .965 | **stopped** | 8 / .882 at 2210 |

Warm is 200/200. Every-10 checks 1210 through 2200 pass (100 checks, minimum HQ .911 at 8 modes). Update 2210 is 8 modes, HQ .882. The run stops there. 2220–2400 were not graded.

Of 204 exit clips in this continuation, 119 had the clump ball on (rho .065–.249). The clips on 2208–2210 are rho .257 to .499, so the ball is off. That is the closed-cap side of the PR #92 exit band, not the acquisition band. It is not a reason to sweep another cut: the mission stops when hold regresses.

Not a production candidate.
