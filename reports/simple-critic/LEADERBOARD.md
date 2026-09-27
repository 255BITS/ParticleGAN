# Simple-critic leaderboard (rounds 1 to 3b)

Protocol: ring of 8, 20k particles, public-trainer construction, constant LRs (G/D 0.00425, prior 0.0085), target shift (1,0) after update 2400, run to 4600, observe every 10 updates. Pass means 8 modes and HQ >= 0.90. Simple arms have no noise, no LR annealing, no KA2 controller and no spike guard. Seed 0, one run per formulation. Launchers: `round1.sh`, `round2.sh` (`a` = first five round-2 arms, `b` = `secant_r1_b2`, chosen after `a`). Scores: `python3 summarize.py`; diagnostics: `python3 summarize.py --diag`.

Round 1 defaults: lam-real 0.1 (r1 arm: 1), lam-path 10 (target 1), lam-cap 10 (c=1). Round 2 base: wgan + r1(1) + interior path-lower (lam 10, target 0.3, u in [0.1, 0.9]) + cap-all(10). Generator loss is RpGAN in every arm.

New round-2 worker flags (defaults reproduce round 1 exactly):
- `--path-u lo,hi`: path points are drawn at u in [lo, hi] (same random stream, remapped).
- `--path secant`: `lam * mean relu(t*|r_nn - f| - (D(r_nn) - D(f)))^2`, where r_nn is each fake's nearest real in the batch. It needs no input gradient and says nothing about the slope at real, and it vanishes when fakes sit on reals.
- `--d-beta2`: a constant critic Adam beta2 (recipe default 0.999, beta1 is 0). This is not a schedule.

New round-3 worker flags (defaults reproduce round 2: a 100-update replay of `secant_r1_b2` matched its stored `metrics.jsonl` at all 10 observations, bit for bit):
- `--lam-center lam`: adds `lam * (E_r D(r))^2`, an explicit level pin on the batch mean of D over reals. Subtracting the batch mean from D's outputs was not used: every term here (wgan base, R1, secant difference, cap) and the RpGAN generator loss is unchanged by adding a constant to D, so output centering would change no gradient (it would only relabel the logged level). The penalty is the version that acts. Unlike drift it does not penalize the spread of D over reals.
- `--lazy-k k`: every non-base term (R1, secant, cap, and drift/center when on) is applied only on critic steps with step % k == 0, with its weight multiplied by k; other steps take the base loss alone. This matches `lazy_k` in `particlegan/grad_regularizers.py`.
- `--latent-damping 0` (existing flag, first use): A2 particle-row damping off, so the prior table gets a plain Adam step.

Round 3 (`round3.sh`): five single changes on top of `secant_r1_b2`: `sec_t1` (secant t 1.0), `sec_drift` (+ drift 1e-3), `sec_center` (+ center(1)), `sec_nodamp` (A2 off), `sec_lazy4` (lazy_k 4). Spectral-norm D was dropped (too limiting).

Round 3b (appended to `round3.sh`): combinations of the changes that improved both fails-outside-transit and arrival (nodamp, center, t1; drift and lazy_k 4 excluded): `combo_a` = base + `--latent-damping 0 --lam-center 1`, `combo_b` = base + `--latent-damping 0 --lam-center 1 --path-target 1.0`.

## Leaderboard

| # | arm | formulation | prehold | arrival | post-arrival | departures | longest fail streak | final suffix | final HQ | max abs D(real) | max grad-norm | fails outside transit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | k3p_stock_ref | REF: K3P v0.8.0 as released (RpGAN+K3P penalty/anchor/guard, own noise + LR schedules) | 120/120 | 1960 | 25/25 | 3 | 24 | 25 (from 4360) | 0.922 | 2.48 | 7.45 | 0 |
| ref | k3p_constant | REF: K3P v0.8.0 critic/penalty, noise off, constant LRs (floors 1) | 88/120 | 360 | 167/185 | 5 | 32 | 123 (from 3380) | 0.989 | 2.63 | 5.81 | 50 |
| 1 | sec_nodamp | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] | 97/120 | 230 | 171/198 | 21 | 24 | 3 (from 4580) | 0.989 | 2.26 | 2.64 | 50 |
| 2 | combo_a | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) + center(1) [Dβ2=0.9, A2=0] | 94/120 | 170 | 162/204 | 20 | 26 | 8 (from 4530) | 0.987 | 2.10 | 2.54 | 68 |
| 3 | sec_t1 | wgan + r1(1) + path-secant(10,t=1) + cap-all(10,c=1) [Dβ2=0.9] | 76/120 | 190 | 178/202 | 24 | 18 | 0 | 0.210 | 4.85 | 5.43 | 68 |
| 4 | sec_center | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) + center(1) [Dβ2=0.9] | 78/120 | 180 | 174/203 | 27 | 28 | 73 (from 3880) | 0.949 | 1.79 | 3.13 | 71 |
| 5 | secant_r1_b2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] | 83/120 | 630 | 118/158 | 33 | 22 | 37 (from 4240) | 0.981 | 3.17 | 2.82 | 77 |
| 6 | combo_b | wgan + r1(1) + path-secant(10,t=1) + cap-all(10,c=1) + center(1) [Dβ2=0.9, A2=0] | 58/120 | 180 | 158/203 | 26 | 39 | 58 (from 4030) | 0.992 | 3.20 | 3.24 | 107 |
| 7 | sec_drift | wgan + drift(0.001)+r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] | 79/120 | 150 | 131/206 | 35 | 52 | 1 (from 4600) | 0.957 | 3.90 | 3.48 | 116 |
| ref | ka2_stock_ref | REF: stock KA2 worker rerun (RpGAN+KA2 penalty/controller, noise on) | 61/120 | 120 | 126/209 | 10 | 58 | 15 (from 4460) | 0.986 | n/a | n/a | 142 |
| ref | ref:ka2-constant | RpGAN+KA2 penalty/controller, noise on (archived) | 61/120 | 120 | 126/209 | 10 | 58 | 15 (from 4460) | 0.986 | n/a | n/a | 142 |
| 8 | int_r1_b2 | wgan + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) [Dβ2=0.9] | 28/120 | 60 | 58/215 | 63 | 25 | 0 | 0.679 | 1.64 | 1.92 | 249 |
| 9 | secant_r1 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) | 51/120 | none | 0/0 | 7 | 37 | 0 | 0.259 | 161.28 | 394.54 | 69 |
| 10 | no_path | wgan + drift(0.1) + cap-all(10,c=1) | 35/120 | none | 0/0 | 14 | 42 | 0 | 0.000 | 46.09 | 201.68 | 85 |
| 11 | int_r1_hinge | hinge + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 19/120 | none | 0/0 | 18 | 20 | 0 | 0.010 | 27.99 | 181.99 | 101 |
| 12 | int_r1 | wgan + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 9/120 | none | 0/0 | 9 | 64 | 0 | 0.043 | 30.33 | 138.60 | 111 |
| 13 | full | wgan + drift(0.1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.016 | 120.32 | 685.01 | 120 |
| 14 | full_hinge | hinge + drift(0.1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.005 | 50.71 | 211.99 | 120 |
| 15 | full_r1 | wgan + r1(1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.094 | 6.93 | 29.86 | 120 |
| 16 | int_r1_rp | rplogistic + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.045 | 6.40 | 34.30 | 120 |
| 17 | no_cap | wgan + drift(0.1) + path-lower(10,t=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.000 | 60288.24 | 4629336.00 | 120 |
| 18 | no_real | wgan + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.389 | 15.88 | 25.52 | 120 |
| 19 | sec_lazy4 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, lazy_k=4] | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.054 | 4.02 | 4.20 | 120 |
| 20 | wgangp_ref | wgan + drift(0.1) + path-two_sided(10,t=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.631 | 8.34 | 52.73 | 120 |

`*` means still running. Departures and streaks exclude the shift transit (2400 to first arrival). "ref" rows are not simple arms (noise on, KA2 controller): `ref:ka2-constant` is the archived KA2 constant-LR run and `ka2_stock_ref` is the same worker rerun here, which reproduces it exactly; they are not ranked.

## D well-behavedness diagnostics

The table below gives medians over all 460 observations of the probe (4096 eval fakes, 4096 real points, and the path points between them). "pass 2410-4600" counts passing observations after the shift.

| arm | best HQ (step) | mean HQ 1210-2400 | mean HQ 2410-3600 | mean HQ 3600-4600 | pass 2410-4600 | g(real) | g(path) | gmax | obs gmax>2 | max gmax |
|---|---|---|---|---|---|---|---|---|---|---|
| k3p_constant | 0.994 (3940) | 0.768 | 0.775 | 0.990 | 167/220 | 0.04 | 0.56 | 3.14 | 458/460 | 5.81 |
| sec_center | 0.993 (4590) | 0.760 | 0.851 | 0.877 | 174/220 | 0.06 | 0.37 | 1.25 | 12/460 | 3.13 |
| combo_b | 0.997 (4000) | 0.689 | 0.824 | 0.905 | 158/220 | 0.12 | 0.54 | 1.43 | 62/460 | 3.24 |
| sec_nodamp | 0.995 (4310) | 0.865 | 0.800 | 0.897 | 171/220 | 0.06 | 0.31 | 1.13 | 14/460 | 2.64 |
| sec_t1 | 0.995 (4370) | 0.828 | 0.844 | 0.820 | 178/220 | 0.11 | 0.51 | 1.42 | 85/460 | 5.43 |
| combo_a | 0.993 (3080) | 0.847 | 0.848 | 0.754 | 162/220 | 0.07 | 0.45 | 1.26 | 12/460 | 2.54 |
| sec_drift | 0.994 (2960) | 0.786 | 0.846 | 0.654 | 131/220 | 0.09 | 0.44 | 1.35 | 31/460 | 3.48 |
| secant_r1_b2 | 0.991 (4470) | 0.837 | 0.650 | 0.854 | 118/220 | 0.10 | 0.49 | 1.35 | 17/460 | 2.82 |
| int_r1_b2 | 0.994 (4270) | 0.708 | 0.704 | 0.765 | 58/220 | 0.16 | 0.49 | 1.28 | 0/460 | 1.92 |
| k3p_stock_ref | 0.983 (880) | 0.974 | 0.526 | 0.815 | 25/220 | 0.27 | 0.71 | 3.42 | 418/460 | 7.45 |
| sec_lazy4 | 0.916 (4060) | 0.409 | 0.409 | 0.436 | 0/220 | 0.24 | 0.51 | 1.65 | 97/460 | 4.2 |
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

## Findings (rounds 1 and 2)

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

## Round 3 findings (single changes on `secant_r1_b2`)

| arm | change | prehold | arrival | departures | fails outside transit | max grad-norm | verdict |
|---|---|---|---|---|---|---|---|
| secant_r1_b2 | (base) | 83/120 | 630 | 33 | 77 | 2.82 | |
| sec_nodamp | A2 latent damping 0 | **97/120** | 230 | **21** | **50** | **2.64** | helps on all five |
| sec_center | + center(1) | 78/120 | 180 | 27 | 71 | 3.13 | helps arrival, departures, fails; slightly worse hold and max grad |
| sec_t1 | secant t 1.0 | 76/120 | 190 | 24 | 68 | 5.43 | helps arrival and departures, but 85/460 obs gmax>2 and a late collapse at about 4450 (final HQ 0.21) |
| sec_drift | + drift 1e-3 | 79/120 | **150** | 35 | 116 | 3.48 | only arrival helps; post-arrival worse (131/206, streak 52) |
| sec_lazy4 | lazy_k 4 | 0/120 | none | 0 | 120 | 4.20 | fails outright |
| combo_a | nodamp + center(1) | 94/120 | 170 | **20** | 68 | **2.54** | no gain over nodamp alone: fails 50 -> 68, late mean HQ 0.897 -> 0.754 |
| combo_b | nodamp + center(1) + t1 | 58/120 | 180 | 26 | 107 | 3.24 | worse than every single change; t1 cost dominates |

1. **Turning off A2 damping is the best single change.** `sec_nodamp` has the best hold (97/120, mean HQ 0.865), fewest fails outside transit (50) and fewest departures (21), and the calmest critic (median gmax 1.13, g(real) 0.06). The generator/prior step is now plain Adam, so the whole run has no adaptive element beyond Adam itself.
2. **The level pin works as intended and helps a little.** `sec_center` holds mean D(real) at 0.00 to 0.01 (base drifts 0.5 to 1.1); it reaches 8 modes 10 updates after the shift, has the longest final passing run of round 3 (73 checks, from 3880) and median gmax 1.25. Tiny drift (1e-3) also pins the level (mean D(real) about 0) through the output bias, which otherwise gets no gradient, but it worsens post-arrival flicker.
3. **Arrival 630 in the base looks like the outlier, not a property of the formulation.** Every non-lazy change, including the near-null drift 1e-3, arrived in 150 to 230 updates, and every non-lazy arm (base included) had 8 modes within 10 to 100 updates of the shift. Read arrival differences within 150 to 230 as trajectory sensitivity, not as ranking evidence.
4. **Secant t = 1 is too strong at equilibrium.** When the rise target equals the cap, the secant and the cap can only both hold when D is at its slope limit along every fake-to-real segment. Late in the run the critic went flat (g(real) about 0.02), then kicked (|D(real)| 3.25, gmax 2.9) and the generator collapsed to 2 modes at 4500.
5. **Lazy regularization breaks this critic.** With k = 4, three of every four critic steps are pure wgan with no secant or cap, and Adam (beta1 0, beta2 0.9) normalizes the 4x hit on the fourth step, so the average pressure is not preserved as it is under SGD. The secant and cap are the terms that shape D here, not a mild add-on like StyleGAN2's R1. `sec_lazy4` never passes (best HQ 0.916 once, 97/460 obs gmax>2).

6. **The helpful changes do not stack.** `combo_a` (nodamp + center) matches nodamp's hold (94 vs 97) and critic calm (max gmax 2.54, median 1.26), but post-arrival holding is worse (162/204, 68 fails outside transit vs 50; mean HQ 3600-4600 0.754 vs 0.897). Center's benefit on the damped base (fails 77 -> 71) was small and does not survive once A2 is off, so the level pin is not needed. `combo_b` adds t1 and is clearly worse (prehold 58, fails 107, 62/460 obs gmax>2): t = 1 is harmful in combination too. Drift, center, t1 and lazy regularization are all closed; `sec_nodamp` stays the best simple arm.

## Best formulation

`sec_nodamp`: **wgan + R1(1) at real + secant path from fake to nearest real (lam 10, t 0.5) + cap-all(10, c=1)**, critic Adam beta2 0.9, and A2 latent damping off. It is `secant_r1_b2` with the one generator-side adaptive element removed. It has three penalty terms, one optimizer constant, no noise, no annealing and no controller on either side. Each term maps to one requirement: R1 keeps D from spiking on real, the secant gives a path from fake, the cap bounds slope at 1, and beta2 makes the cap hold between steps.

## Next experiments (distinct formulations)

1. **Promotion check (transfer candidate `sec_nodamp`):** run it on 100 Gaussians and the sparse-UCD toy before treating it as a KA2 replacement. It has only been run on ring-8 shift.
2. **Do not pursue:** center / drift level pins (no gain once A2 is off), secant t = 1 (late collapse, worse in combination), lazy regularization under Adam (never passes), spectral norm (dropped, too limiting).
