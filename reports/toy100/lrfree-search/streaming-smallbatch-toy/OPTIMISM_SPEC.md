# Prospective critic-lag diagnostic (no new controller)

Written after the continuous mass-flow result and Astra's review, before
computing the diagnostic. Replay the existing `learned_payoff` mass-flow arm
with the same N=32, batch two, 1,600 updates, cases, random streams, D/G
updates, and mass rate. Assert its saved final and cumulative TV are exactly
reproduced. Do not change the controller in this replay.

At the pre-D-update point, record each row's online two-real payoff `L_t`.
Also compute its exact-real expectation under the *current learned D*, `P_t`,
and the optimal toy critic's exact-real payoff `O_t`. The latter two read the
true toy law for evaluation only; no candidate can use them. Use the previous
step's corresponding vectors without realigning rows, because this arm has no
cloning. Compare the current vectors with a one-step linear extrapolation:

`online: L_t versus 2 L_t - L_(t-1)`;
`population: P_t versus 2 P_t - P_(t-1)`.

For each vector, subtract its current-mass-weighted mean. The direction error
at each step is the current-mass-weighted mean squared difference from the
centred `O_t`. Sum this error over steps 2..1,600. The population comparison
tests predictable D/G lag without the two-real Monte Carlo error; the online
comparison measures the actual available signal. Report per-case ratios of
extrapolated/current error and the online versus population discrepancy.
The exact null must be included; its oracle direction is zero.

**Go/no-go fixed now:** Test an optimistic mass controller only if online
extrapolation lowers integrated direction error in at least three of the four
learned-D moving cases (overmass, rare, shift, gauge), including rare, while
not increasing error on the exact null. Population-only gains do not qualify.
The frozen-bad-D case is a diagnostic but cannot qualify the controller.
If this gate fails, stop this branch of the toy investigation and report the
negative result. This is one mechanism test, no seed or parameter sweep.
