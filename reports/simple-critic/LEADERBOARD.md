# Simple-critic leaderboard (rounds 1 and 2)

Protocol: ring of 8, 20k particles, public-trainer construction, constant LRs (G/D 0.00425, prior 0.0085), target shift (1,0) after update 2400, run to 4600, observe every 10 updates. Pass means 8 modes and HQ >= 0.90. Simple arms have no noise, no LR annealing, no KA2 controller and no spike guard. Seed 0, one run per formulation. Launchers: `round1.sh`, `round2.sh` (`a` = first five round-2 arms, `b` = `secant_r1_b2`, chosen after `a`). Scores: `python3 summarize.py`; diagnostics: `python3 summarize.py --diag`.

Round 1 defaults: lam-real 0.1 (r1 arm: 1), lam-path 10 (target 1), lam-cap 10 (c=1). Round 2 base: wgan + r1(1) + interior path-lower (lam 10, target 0.3, u in [0.1, 0.9]) + cap-all(10). Generator loss is RpGAN in every arm.

New round-2 worker flags (defaults reproduce round 1 exactly):
- `--path-u lo,hi`: path points are drawn at u in [lo, hi] (same random stream, remapped).
- `--path secant`: `lam * mean relu(t*|r_nn - f| - (D(r_nn) - D(f)))^2`, where r_nn is each fake's nearest real in the batch. It needs no input gradient and says nothing about the slope at real, and it vanishes when fakes sit on reals.
- `--d-beta2`: a constant critic Adam beta2 (recipe default 0.999, beta1 is 0). This is not a schedule.

## Leaderboard

| # | arm | formulation | prehold | arrival | post-arrival | departures | longest fail streak | final suffix | final HQ | max abs D(real) | max grad-norm | fails outside transit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | secant_r1_b2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] | 83/120 | 630 | 118/158 | 33 | 22 | 37 (from 4240) | 0.981 | 3.17 | 2.82 | 77 |
| ref | ka2_stock_ref | REF: stock KA2 worker rerun (RpGAN+KA2 penalty/controller, noise on) | 61/120 | 120 | 126/209 | 10 | 58 | 15 (from 4460) | 0.986 | n/a | n/a | 142 |
| ref | ref:ka2-constant | RpGAN+KA2 penalty/controller, noise on (archived) | 61/120 | 120 | 126/209 | 10 | 58 | 15 (from 4460) | 0.986 | n/a | n/a | 142 |
| 2 | int_r1_b2 | wgan + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) [Dβ2=0.9] | 28/120 | 60 | 58/215 | 63 | 25 | 0 | 0.679 | 1.64 | 1.92 | 249 |
| 3 | secant_r1 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) | 51/120 | none | 0/0 | 7 | 37 | 0 | 0.259 | 161.28 | 394.54 | 69 |
| 4 | no_path | wgan + drift(0.1) + cap-all(10,c=1) | 35/120 | none | 0/0 | 14 | 42 | 0 | 0.000 | 46.09 | 201.68 | 85 |
| 5 | int_r1_hinge | hinge + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 19/120 | none | 0/0 | 18 | 20 | 0 | 0.010 | 27.99 | 181.99 | 101 |
| 6 | int_r1 | wgan + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 9/120 | none | 0/0 | 9 | 64 | 0 | 0.043 | 30.33 | 138.60 | 111 |
| 7 | full | wgan + drift(0.1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.016 | 120.32 | 685.01 | 120 |
| 8 | full_hinge | hinge + drift(0.1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.005 | 50.71 | 211.99 | 120 |
| 9 | full_r1 | wgan + r1(1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.094 | 6.93 | 29.86 | 120 |
| 10 | int_r1_rp | rplogistic + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.045 | 6.40 | 34.30 | 120 |
| 11 | no_cap | wgan + drift(0.1) + path-lower(10,t=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.000 | 60288.24 | 4629336.00 | 120 |
| 12 | no_real | wgan + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.389 | 15.88 | 25.52 | 120 |
| 13 | wgangp_ref | wgan + drift(0.1) + path-two_sided(10,t=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.631 | 8.34 | 52.73 | 120 |

`*` means still running. Departures and streaks exclude the shift transit (2400 to first arrival). "ref" rows are not simple arms (noise on, KA2 controller): `ref:ka2-constant` is the archived KA2 constant-LR run and `ka2_stock_ref` is the same worker rerun here, which reproduces it exactly; they are not ranked.

## D well-behavedness diagnostics

The table below gives medians over all 460 observations of the probe (4096 eval fakes, 4096 real points, and the path points between them). "pass 2410-4600" counts passing observations after the shift.

| arm | best HQ (step) | mean HQ 1210-2400 | mean HQ 2410-3600 | mean HQ 3600-4600 | pass 2410-4600 | g(real) | g(path) | gmax | obs gmax>2 | max gmax |
|---|---|---|---|---|---|---|---|---|---|---|
| secant_r1_b2 | 0.991 (4470) | 0.837 | 0.650 | 0.854 | 118/220 | 0.10 | 0.49 | 1.35 | 17/460 | 2.82 |
| int_r1_b2 | 0.994 (4270) | 0.708 | 0.704 | 0.765 | 58/220 | 0.16 | 0.49 | 1.28 | 0/460 | 1.92 |
| wgangp_ref | 0.631 (4600) | 0.212 | 0.185 | 0.411 | 0/220 | 0.93 | 0.92 | 1.90 | 157/460 | 52.7 |
| no_real | 0.667 (4550) | 0.070 | 0.152 | 0.321 | 0/220 | 0.90 | 0.97 | 2.98 | 401/460 | 25.5 |
| secant_r1 | 0.994 (1960) | 0.541 | 0.032 | 0.117 | 0/220 | 1.35 | 2.32 | 14.37 | 276/460 | 395 |
| int_r1_hinge | 0.975 (2210) | 0.690 | 0.045 | 0.079 | 0/220 | 0.24 | 0.64 | 1.69 | 221/460 | 182 |
| int_r1 | 0.964 (1750) | 0.350 | 0.039 | 0.079 | 0/220 | 0.60 | 0.68 | 3.93 | 277/460 | 139 |
| full_r1 | 0.377 (2360) | 0.128 | 0.064 | 0.043 | 0/220 | 0.64 | 0.90 | 2.78 | 432/460 | 29.9 |
| full | 0.215 (710) | 0.013 | 0.037 | 0.055 | 0/220 | 0.85 | 0.94 | 3.81 | 413/460 | 685 |
| int_r1_rp | 0.209 (450) | 0.031 | 0.025 | 0.049 | 0/220 | 0.31 | 0.48 | 2.22 | 302/460 | 34.3 |
| no_cap | 0.315 (4050) | 0.012 | 0.000 | 0.017 | 0/220 | 1025.13 | 457778.78 | 2749986.38 | 460/460 | 4.63e+06 |
| full_hinge | 0.787 (2160) | 0.357 | 0.003 | 0.005 | 0/220 | 0.99 | 1.14 | 2.30 | 268/460 | 212 |
| no_path | 0.991 (2320) | 0.777 | 0.005 | 0.000 | 0/220 | 0.74 | 0.68 | 1.67 | 221/460 | 202 |

## Findings (both rounds)

1. **The spikes came from the critic optimizer, not the penalty.** The critic uses Adam with betas (0, 0.999). After a quiet stretch (the hold, where D's gradients are tiny), a sudden gradient gives a step of up to about lr/sqrt(1-beta2), which is 31.6×lr per parameter. With beta2 = 0.9 the bound drops to 3.2×lr. This one constant took `int_r1` from max grad-norm 139 to 1.92 (0 of 460 observations above 2), and `secant_r1` from 395 to 2.82 (max |D(real)| from 161 to 3.2). No soft cap can stop a single-step jump, so this is what makes "slope max 1" hold in practice. It keeps the LR constant and adds no controller.
2. **The path term must not act at the real endpoint, and it should be a rise in D from fake, not a slope floor.**
   - An interior-only slope floor (u in [0.1, 0.9], target 0.3) already lets reals become D peaks: median g(real) fell from 0.64 (`full_r1`) to 0.16.
   - The secant form is better. It asks only that D rise by t×distance from each fake to its nearest real, so it is zero at equilibrium and never sets a slope at real (median g(real) 0.10). It gives the best hold of any arm (`secant_r1_b2` prehold 83/120 versus 61/120 for KA2).
   - Without a path term, D inverts at the shift and never recovers (`no_path`). With the secant path and no spikes, the arm tracks the shift: 8 modes about 270 updates after it, and passing by update 630 after it.
3. **R1 is the right real term.** Drift did almost nothing in round 1. With a path term that no longer pulls g(real) up, R1(1) keeps g(real) at 0.1 to 0.2 and |D(real)| at 3 or less when there are no spikes.
4. **Cap-all(10) is still required, but only as the soft bound.** It holds median gmax at 1.3 in the beta2 = 0.9 arms. Without it, the one-sided path diverges (`no_cap`).
5. **Base loss: wgan is best, and a bounded or saturating base does not help.**
   - `int_r1_hinge` sharpened faster than `int_r1` (mean HQ 0.69 versus 0.35 from 1210 to 2400) but still spiked at the shift.
   - `int_r1_rp` (RpGAN, the KA2 base) never got above 0.21 HQ: the saturating logistic gives D too little push under the cap. A likely reason KA2 needed its controller is to compensate for this.
6. **Remaining weakness:**
   - `secant_r1_b2` arrives after the shift slower than KA2 (630 versus 120 updates), though it holds better once there (118/158 versus 126/209, 77 versus 142 fails outside transit, longest streak 22 versus 58).
   - D's level is unanchored: D(real) mean drifts from about 0.5 to 1.1 after the shift, because wgan plus R1 is invariant to adding a constant.
   - `int_r1_b2` is the calmest critic, but it flickers (63 departures): the slope floor of 0.3 in the interior, which is nonzero at equilibrium, keeps jostling the generator.

## Best formulation

`secant_r1_b2`: **wgan + R1(1) at real + secant path from fake to nearest real (lam 10, t 0.5) + cap-all(10, c=1)**, with critic Adam beta2 0.9. That is three penalty terms, one optimizer constant, no noise, no annealing and no controller. Each term maps to one requirement: R1 keeps D from spiking on real, the secant gives a path from fake, the cap bounds slope at 1, and beta2 makes the cap hold between steps.

## Next experiments (distinct formulations)

1. **Faster arrival:** try a secant target of 1.0, the largest rise the cap allows. This is a feasibility change, not a nudge: it changes whether the path and the cap can both be met.
2. **Pin the D level:** add a tiny drift (1e-3) or subtract the batch mean of D(real) to anchor the level. The goal is to stop the post-shift D offset drift without adding a spike term.
3. **Replace the soft cap with a hard Lipschitz bound** (spectral norm on D's hidden layers, with the Fourier scale fixed). If it holds, drop the cap term, leaving two terms.
4. **Promotion check:** run `secant_r1_b2` on the other benchmark targets (100 Gaussians, the sparse-UCD toy) before treating it as a KA2 replacement. It has only been run on ring-8 shift.
