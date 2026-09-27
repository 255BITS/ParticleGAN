# Simple-critic leaderboard (all rounds, develop's batch_feature_zero init)

> Refreshed from `runs/` after merging origin/develop c720645e (#194). Every ring arm was rerun under the package default `initialization='batch_feature_zero'`; each `result.json` carries an `init_receipt`. The old random-init board is `leaderboard_runs_oldinit.md` (`summarize.py --runs-dir runs_oldinit`), and the per-arm old-vs-new comparison is `INIT_RERUN.md`. The findings below the diagnostics are from the old-init runs (rounds 1 to 3b); see the README's "Rerun with develop's QR initialization" for which still hold. Rank gaps under about 30 fails outside transit are within init noise.

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

## Leaderboard (new init, `runs/`)

| # | arm | formulation | prehold | arrival | post-arrival | departures | longest fail streak | final suffix | final HQ | max abs D(real) | max grad-norm | fails outside transit | init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | k3p_stock_ref | REF: K3P k3p-develop@c720645e as released (RpGAN+K3P penalty/anchor/guard, own noise + LR schedules) | 120/120 | 1220 | 99/99 | 1 | 1 | 99 (from 3620) | 0.954 | 1.16 | 3.55 | 0 | bfz |
| ref | k3p_constant | REF: K3P k3p-develop@c720645e critic/penalty, noise off, constant LRs (floors 1) | 95/120 | 360 | 185/185 | 1 | 30 | 185 (from 2760) | 0.993 | 2.35 | 4.90 | 25 | bfz |
| 1 | B_cap3 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=3) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 80/120 | 250 | 195/196 | 11 | 21 | 67 (from 3940) | 0.995 | 5.02 | 5.43 | 41 | bfz |
| 2 | lr_c0.5_g0.5 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×0.5 g×0.5} | 86/120 | 240 | 181/197 | 15 | 16 | 79 (from 3820) | 0.993 | 2.55 | 2.40 | 50 | bfz |
| 3 | sec_nodamp | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] | 97/120 | 160 | 176/205 | 17 | 20 | 26 (from 4350) | 0.982 | 2.40 | 2.73 | 52 | bfz |
| 4 | lr_c0.5_g2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×0.5 g×2} | 99/120 | 400 | 148/181 | 9 | 26 | 0 | 0.008 | 6.62 | 3.48 | 54 | bfz |
| ref | ka2_stock_ref | REF: stock KA2 worker rerun (RpGAN+KA2 penalty/controller, noise on) | 78/120 | 390 | 168/182 | 12 | 48 | 0 | 0.724 | n/a | n/a | 56 | bfz |
| 5 | lr_c1_g0.5 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×1 g×0.5} | 86/120 | 90 | 185/212 | 25 | 17 | 45 (from 4160) | 0.991 | 2.98 | 2.35 | 61 | bfz |
| 6 | lr_c0.25_g0.25 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×0.25 g×0.25} | 75/120 | 100 | 192/211 | 6 | 24 | 0 | 0.139 | 3.19 | 2.80 | 64 | bfz |
| 7 | sec_drift | wgan + drift(0.001)+r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] | 83/120 | 250 | 168/196 | 24 | 31 | 26 (from 4350) | 0.995 | 3.51 | 2.82 | 65 | bfz |
| 8 | combo_a | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) + center(1) [Dβ2=0.9, A2=0] | 90/120 | 200 | 165/201 | 19 | 20 | 20 (from 4410) | 0.966 | 1.99 | 3.35 | 66 | bfz |
| 9 | lr_c0.5_g1 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 81/120 | 380 | 153/183 | 17 | 29 | 91 (from 3700) | 0.976 | 2.79 | 2.66 | 69 | bfz |
| 10 | c3_r1w | wgan + r1(0.1) + path-secant(10,t=0.5) + cap-all(10,c=3) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 83/120 | 270 | 161/194 | 10 | 33 | 34 (from 4270) | 0.983 | 5.85 | 5.53 | 70 | bfz |
| 11 | sec_center | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) + center(1) [Dβ2=0.9] | 93/120 | 150 | 163/206 | 15 | 24 | 1 (from 4600) | 0.985 | 1.81 | 2.73 | 70 | bfz |
| 12 | B_margin | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) + margin(10,κ=7.14) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 97/120 | 420 | 129/179 | 12 | 45 | 15 (from 4460) | 0.919 | 3.66 | 2.82 | 73 | bfz |
| 13 | c3_capinterp | wgan + r1(1) + path-secant(10,t=0.5) + cap-interp(10,c=3) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 109/120 | 50 | 153/216 | 18 | 28 | 22 (from 4390) | 0.985 | 4.34 | 5.41 | 74 | bfz |
| 14 | sec_guard_anchor | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] + K3P guard(ratio=5,min=200) + K3P EMA-anchor(0.5*prox,decay=0.999) | 83/120 | 280 | 156/193 | 32 | 18 | 1 (from 4600) | 0.991 | 2.87 | 2.54 | 74 | bfz |
| 15 | sec_anchor | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] + K3P EMA-anchor(0.5*prox,decay=0.999) | 82/120 | 560 | 117/165 | 37 | 18 | 1 (from 4600) | 0.941 | 3.22 | 2.54 | 86 | bfz |
| 16 | B_cap2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=2) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 57/120 | 130 | 180/208 | 18 | 40 | 29 (from 4320) | 0.976 | 4.39 | 4.14 | 91 | bfz |
| 17 | c3_capnopath | wgan + r1(1) + path-secant(10,t=0.5) + cap-ends(10,c=3) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 84/120 | 40 | 155/217 | 30 | 42 | 0 | 0.135 | 10.67 | 9.30 | 98 | bfz |
| 18 | B_nnpair | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) + nnpair [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 75/120 | 200 | 144/201 | 31 | 26 | 33 (from 4280) | 0.945 | 5.38 | 5.11 | 102 | bfz |
| 19 | secant_r1_b2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] | 48/120 | 280 | 159/193 | 27 | 47 | 10 (from 4510) | 0.960 | 2.79 | 2.70 | 106 | bfz |
| 20 | sec_guard | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] + K3P guard(ratio=5,min=200) | 63/120 | 230 | 149/198 | 32 | 26 | 0 | 0.102 | 2.64 | 2.70 | 106 | bfz |
| 21 | rp_center | rplogistic + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=3) + pair-center(1) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 53/120 | 170 | 153/204 | 6 | 48 | 100 (from 3610) | 0.987 | 3.02 | 5.63 | 118 | bfz |
| 22 | sec_t1 | wgan + r1(1) + path-secant(10,t=1) + cap-all(10,c=1) [Dβ2=0.9] | 68/120 | 150 | 129/206 | 28 | 50 | 0 | 0.545 | 5.29 | 4.31 | 129 | bfz |
| 23 | lr_c1_g2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×1 g×2} | 84/120 | 220 | 98/199 | 24 | 81 | 0 | 0.346 | 4.78 | 3.60 | 137 | bfz |
| ref | ref:ka2-constant | RpGAN+KA2 penalty/controller, noise on (archived) | 61/120 | 120 | 126/209 | 10 | 58 | 15 (from 4460) | 0.986 | n/a | n/a | 142 | old |
| 24 | lr_c2_g1 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×2 g×1} | 75/120 | 190 | 102/202 | 44 | 58 | 0 | 0.794 | 3.64 | 2.98 | 145 | bfz |
| 25 | combo_b | wgan + r1(1) + path-secant(10,t=1) + cap-all(10,c=1) + center(1) [Dβ2=0.9, A2=0] | 79/120 | 220 | 84/199 | 24 | 95 | 0 | 0.358 | 3.26 | 5.11 | 156 | bfz |
| 26 | lr_c2_g0.5 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×2 g×0.5} | 67/120 | 120 | 105/209 | 61 | 22 | 11 (from 4500) | 0.974 | 4.27 | 4.76 | 157 | bfz |
| 27 | lr_c2_g2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×2 g×2} | 8/120 | 350 | 84/186 | 34 | 106 | 1 (from 4600) | 0.964 | 4.82 | 7.20 | 214 | bfz |
| 28 | rp_center_min | rplogistic + cap-all(10,c=3) + pair-center(1) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 1/120 | 950 | 26/126 | 17 | 29 | 3 (from 4580) | 0.906 | 2.49 | 4.06 | 219 | bfz |
| 29 | int_r1_b2 | wgan + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) [Dβ2=0.9] | 21/120 | 160 | 78/205 | 63 | 20 | 0 | 0.733 | 2.60 | 2.21 | 226 | bfz |
| 30 | wgan_huber | wgan + huber(10,κ=7.14) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 0/120 | 1020 | 5/119 | 2 | 104 | 0 | 0.709 | 7.38 | 16.16 | 234 | bfz |
| 31 | secant_r1 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) | 77/120 | none | 0/0 | 11 | 31 | 0 | 0.475 | 92.12 | 270.27 | 43 | bfz |
| 32 | int_r1_hinge | hinge + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 21/120 | none | 0/0 | 19 | 27 | 0 | 0.080 | 26.48 | 219.04 | 99 | bfz |
| 33 | no_path | wgan + drift(0.1) + cap-all(10,c=1) | 7/120 | none | 0/0 | 5 | 42 | 0 | 0.031 | 203.42 | 1201.09 | 113 | bfz |
| 34 | int_r1 | wgan + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 5/120 | none | 0/0 | 7 | 106 | 0 | 0.399 | 19.33 | 72.43 | 115 | bfz |
| 35 | sec_lazy4 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, lazy_k=4] | 1/120 | none | 0/0 | 1 | 24 | 0 | 0.635 | 3.96 | 3.91 | 119 | bfz |
| 36 | c3_rate | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=3) + rate(10) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.402 | 5.30 | 7.28 | 120 | bfz |
| 37 | full | wgan + drift(0.1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.136 | 22.63 | 95.06 | 120 | bfz |
| 38 | full_hinge | hinge + drift(0.1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.224 | 2.27 | 14.51 | 120 | bfz |
| 39 | full_r1 | wgan + r1(1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.706 | 2.09 | 5.41 | 120 | bfz |
| 40 | lr_c4_g4 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×4 g×4} | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.000 | 61.21 | 61.75 | 120 | bfz |
| 41 | no_cap | wgan + drift(0.1) + path-lower(10,t=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.000 | 61942.81 | 6715439.00 | 120 | bfz |
| 42 | no_real | wgan + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.093 | 63.88 | 343.71 | 120 | bfz |
| 43 | wgan_margin | wgan + margin(10,κ=7.14) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.028 | 56526.21 | 1704162.75 | 120 | bfz |
| 44 | wgangp_ref | wgan + drift(0.1) + path-two_sided(10,t=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.593 | 0.88 | 6.59 | 120 | bfz |
| 45 | int_r1_rp | rplogistic + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 0/120 | none | 0/0 | 2 | 126 | 0 | 0.010 | 42.55 | 345.12 | 120 | bfz |

`*` = still running (scored through its last observation). Departures and streaks exclude the shift transit (2400 to first arrival). ref = archived KA2 constant-LR run (not a simple arm). init: bfz = batch_feature_zero (#194) with G/D/prior all different from the old random draw; old = pre-#194 random init (no receipt).

## D well-behavedness diagnostics (new init)

The table below gives medians over all 460 observations of the probe (4096 eval fakes, 4096 real points, and the path points between them). "pass 2410-4600" counts passing observations after the shift.

| arm | best HQ (step) | mean HQ 1210-2400 | mean HQ 2410-3600 | mean HQ 3600-4600 | pass 2410-4600 | g(real) | g(path) | gmax | obs gmax>2 | max gmax |
|---|---|---|---|---|---|---|---|---|---|---|
| B_cap3 | 0.995 (4600) | 0.741 | 0.875 | 0.980 | 195/220 | 0.11 | 1.00 | 3.28 | 425/460 | 5.43 |
| lr_c0.25_g0.25 | 0.990 (4350) | 0.782 | 0.896 | 0.910 | 192/220 | 0.07 | 0.47 | 1.32 | 14/460 | 2.8 |
| lr_c1_g0.5 | 0.995 (3850) | 0.791 | 0.858 | 0.947 | 185/220 | 0.07 | 0.39 | 1.21 | 13/460 | 2.35 |
| B_cap2 | 0.989 (4430) | 0.719 | 0.891 | 0.893 | 180/220 | 0.10 | 0.75 | 2.36 | 388/460 | 4.14 |
| k3p_constant | 0.998 (4320) | 0.841 | 0.798 | 0.995 | 185/220 | 0.02 | 0.44 | 2.22 | 330/460 | 4.9 |
| sec_drift | 0.996 (4560) | 0.787 | 0.793 | 0.974 | 168/220 | 0.07 | 0.38 | 1.20 | 22/460 | 2.82 |
| lr_c0.5_g0.5 | 0.995 (4590) | 0.815 | 0.850 | 0.874 | 181/220 | 0.07 | 0.46 | 1.28 | 21/460 | 2.4 |
| sec_center | 0.994 (1720) | 0.834 | 0.788 | 0.934 | 163/220 | 0.06 | 0.38 | 1.27 | 9/460 | 2.73 |
| sec_nodamp | 0.992 (3550) | 0.862 | 0.872 | 0.822 | 176/220 | 0.07 | 0.40 | 1.23 | 10/460 | 2.73 |
| rp_center | 0.990 (4070) | 0.765 | 0.726 | 0.983 | 153/220 | 0.09 | 0.76 | 3.29 | 458/460 | 5.63 |
| combo_a | 0.994 (4280) | 0.801 | 0.777 | 0.897 | 165/220 | 0.07 | 0.43 | 1.28 | 17/460 | 3.35 |
| secant_r1_b2 | 0.990 (4560) | 0.732 | 0.810 | 0.845 | 159/220 | 0.10 | 0.43 | 1.34 | 22/460 | 2.7 |
| B_nnpair | 0.993 (3420) | 0.767 | 0.832 | 0.799 | 144/220 | 0.09 | 0.60 | 2.47 | 341/460 | 5.11 |
| c3_r1w | 0.998 (3580) | 0.804 | 0.861 | 0.762 | 161/220 | 0.25 | 1.26 | 3.42 | 454/460 | 5.53 |
| sec_guard_anchor | 0.994 (4510) | 0.822 | 0.770 | 0.861 | 156/220 | 0.09 | 0.43 | 1.32 | 17/460 | 2.54 |
| c3_capnopath | 0.991 (2820) | 0.865 | 0.802 | 0.826 | 155/220 | 0.15 | 1.35 | 5.15 | 460/460 | 9.3 |
| lr_c0.5_g1 | 0.996 (4430) | 0.817 | 0.689 | 0.926 | 153/220 | 0.07 | 0.42 | 1.29 | 17/460 | 2.66 |
| sec_guard | 0.995 (4490) | 0.776 | 0.771 | 0.825 | 149/220 | 0.10 | 0.42 | 1.33 | 29/460 | 2.7 |
| c3_capinterp | 0.988 (2020) | 0.908 | 0.771 | 0.809 | 153/220 | 0.10 | 0.88 | 3.30 | 459/460 | 5.41 |
| lr_c2_g0.5 | 0.990 (4450) | 0.803 | 0.711 | 0.870 | 105/220 | 0.12 | 0.42 | 1.32 | 24/460 | 4.76 |
| rp_center_min | 0.955 (3510) | 0.713 | 0.700 | 0.830 | 26/220 | 1.77 | 1.42 | 3.25 | 460/460 | 4.06 |
| sec_t1 | 0.988 (4090) | 0.767 | 0.825 | 0.674 | 129/220 | 0.14 | 0.55 | 1.48 | 71/460 | 4.31 |
| sec_anchor | 0.993 (4160) | 0.840 | 0.686 | 0.837 | 117/220 | 0.09 | 0.42 | 1.33 | 18/460 | 2.54 |
| k3p_stock_ref | 0.990 (1320) | 0.988 | 0.595 | 0.941 | 99/220 | 0.10 | 0.42 | 1.71 | 193/460 | 3.55 |
| int_r1_b2 | 0.989 (3630) | 0.693 | 0.722 | 0.787 | 78/220 | 0.17 | 0.50 | 1.30 | 1/460 | 2.21 |
| lr_c0.5_g2 | 0.996 (4180) | 0.878 | 0.735 | 0.748 | 148/220 | 0.07 | 0.44 | 1.29 | 39/460 | 3.48 |
| lr_c2_g2 | 0.974 (4540) | 0.328 | 0.672 | 0.776 | 84/220 | 0.17 | 0.49 | 1.53 | 79/460 | 7.2 |
| B_margin | 0.993 (1840) | 0.859 | 0.762 | 0.652 | 129/220 | 0.08 | 0.47 | 1.32 | 24/460 | 2.82 |
| lr_c2_g1 | 0.987 (2270) | 0.823 | 0.777 | 0.614 | 102/220 | 0.12 | 0.40 | 1.35 | 31/460 | 2.98 |
| wgan_huber | 0.974 (3540) | 0.494 | 0.642 | 0.701 | 5/220 | 1.94 | 1.71 | 6.57 | 460/460 | 16.2 |
| combo_b | 0.991 (3340) | 0.811 | 0.749 | 0.546 | 84/220 | 0.15 | 0.57 | 1.51 | 87/460 | 5.11 |
| lr_c1_g2 | 0.990 (2380) | 0.796 | 0.701 | 0.526 | 98/220 | 0.09 | 0.39 | 1.31 | 42/460 | 3.6 |
| c3_rate | 0.979 (4140) | 0.356 | 0.392 | 0.537 | 0/220 | 0.21 | 1.07 | 4.10 | 459/460 | 7.28 |
| sec_lazy4 | 0.912 (2160) | 0.394 | 0.398 | 0.494 | 0/220 | 0.25 | 0.52 | 1.68 | 114/460 | 3.91 |
| full_r1 | 0.840 (4140) | 0.133 | 0.258 | 0.640 | 0/220 | 0.57 | 0.90 | 1.84 | 78/460 | 5.41 |
| wgangp_ref | 0.626 (4440) | 0.193 | 0.319 | 0.387 | 0/220 | 0.93 | 0.94 | 1.74 | 88/460 | 6.59 |
| int_r1 | 0.965 (1000) | 0.158 | 0.087 | 0.248 | 0/220 | 0.44 | 0.64 | 3.48 | 323/460 | 72.4 |
| full_hinge | 0.337 (4520) | 0.038 | 0.096 | 0.173 | 0/220 | 0.84 | 0.92 | 2.58 | 437/460 | 14.5 |
| lr_c4_g4 | 0.645 (4180) | 0.133 | 0.109 | 0.134 | 0/220 | 0.39 | 0.64 | 2.08 | 257/460 | 61.7 |
| no_real | 0.337 (4510) | 0.024 | 0.045 | 0.102 | 0/220 | 0.91 | 1.02 | 4.34 | 398/460 | 344 |
| full | 0.288 (4580) | 0.012 | 0.033 | 0.089 | 0/220 | 0.83 | 0.93 | 3.53 | 400/460 | 95.1 |
| secant_r1 | 0.987 (2260) | 0.830 | 0.000 | 0.102 | 0/220 | 0.35 | 0.69 | 2.32 | 241/460 | 270 |
| int_r1_rp | 0.935 (1080) | 0.260 | 0.045 | 0.040 | 0/220 | 0.51 | 0.64 | 2.92 | 283/460 | 345 |
| int_r1_hinge | 0.983 (1690) | 0.695 | 0.020 | 0.055 | 0/220 | 0.26 | 0.66 | 1.72 | 222/460 | 219 |
| wgan_margin | 0.395 (3230) | 0.033 | 0.027 | 0.009 | 0/220 | 426.81 | 129683.57 | 716445.59 | 460/460 | 1.7e+06 |
| no_path | 0.949 (2340) | 0.675 | 0.007 | 0.009 | 0/220 | 0.79 | 0.71 | 1.65 | 221/460 | 1.2e+03 |
| no_cap | 0.274 (2010) | 0.014 | 0.000 | 0.000 | 0/220 | 2021.14 | 631103.28 | 3519207.88 | 460/460 | 6.72e+06 |

## Archived findings (old random init, rounds 1 to 3b)

Numbers in these sections refer to `runs_oldinit/`.

### Findings (rounds 1 and 2)

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

### Round 3 findings (single changes on `secant_r1_b2`)

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

### Best formulation

`sec_nodamp`: **wgan + R1(1) at real + secant path from fake to nearest real (lam 10, t 0.5) + cap-all(10, c=1)**, critic Adam beta2 0.9, and A2 latent damping off. It is `secant_r1_b2` with the one generator-side adaptive element removed. It has three penalty terms, one optimizer constant, no noise, no annealing and no controller on either side. Each term maps to one requirement: R1 keeps D from spiking on real, the secant gives a path from fake, the cap bounds slope at 1, and beta2 makes the cap hold between steps.

### Next experiments (distinct formulations)

1. **Promotion check (transfer candidate `sec_nodamp`):** run it on 100 Gaussians and the sparse-UCD toy before treating it as a KA2 replacement. It has only been run on ring-8 shift.
2. **Do not pursue:** center / drift level pins (no gain once A2 is off), secant t = 1 (late collapse, worse in combination), lazy regularization under Adam (never passes), spectral norm (dropped, too limiting).
