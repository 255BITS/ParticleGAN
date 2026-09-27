# Init rerun: batch_feature_zero (#194) vs old random init

All 48 ring arms were rerun on cuda:0, three at a time, after merging origin/develop c720645e ("Default to deterministic QR initialization"). The setup is otherwise unchanged: seed 0, 4600 updates, the ring shift at update 2400, each arm's recorded worker and flags, and no seed variants. Reference rows keep their stated setups. `k3p_*` now runs from the `k3p-develop@c720645e` checkout.

- **Init check:** every `runs/*/result.json` carries an `init_receipt` with `initialization=batch_feature_zero` and `external_init_hook=None`. G, D and the prior all differ from the old random draw, no tensor is left equal to the old init, and the initial hashes match a fresh build through the public recipe path. Wherever an EMA critic exists (sec_anchor, sec_guard_anchor, k3p_*, ka2_stock_ref), `ema_D_equals_D=True` at the start.
- **Crashes:** none. All 48 finished with rc=0. The `.err` files hold only each worker's final LR-check line.
- **Old results:** `runs_oldinit/`, with the leaderboard in `leaderboard_runs_oldinit.md`. New results are in `runs/`, with the leaderboard in `leaderboard.md`. Regenerate this file with `python compare_init.py`.
- **Scoring:** as in summarize.py. A check passes when it finds 8 modes and HQ >= .90. Arms are ranked by arrival, then fails outside transit, then departures, then arrival time. `final HQ` is the single last observation, so treat it as noisy.
- **Not rerun here:** toy100 (see the README note).

## Summary

1. **The top of the board is stable.** B_cap3 stays #1: fails outside transit 38 -> 41, arrival 140 -> 250. sec_nodamp stays #3: fails 50 -> 52, arrival 230 -> 160. The whole failure group (full*, no_cap, no_real, wgangp_ref, wgan_margin, c3_rate, lr_c4_g4, int_r1*, secant_r1, sec_lazy4, no_path) still never arrives after the shift. None of the conclusions about which components are needed changes.
2. **The middle of the board moves a lot.** Changing only the initial weights moves the median arm 2 places, but it moves 13 of 45 arms 7 or more places: rp_center 4 -> 21, B_cap2 5 -> 16, sec_t1 10 -> 22, lr_c0.5_g1 2 -> 9, lr_c0.5_g0.5 15 -> 2, sec_drift 22 -> 7. Fails outside transit swing by 30 to 65 on arms that look structurally similar. This acts like a seed probe that the no-seed-variants rule never ran. Any gap of fewer than about 30 fails between arms ranked roughly 2 to 25 is within init noise, so the fine-grained ranking there should not be trusted.
3. **The reference rows improve the most under the new init.** k3p_constant: fails 50 -> 25 and departures 5 -> 1, which puts it ahead of every simple arm. k3p_stock_ref: still 0 fails, but arrival 1960 -> 1220 and max grad 7.45 -> 3.55. ka2_stock_ref: fails 142 -> 56, although arrival worsens 120 -> 390. The new init helps the noise/controller/anchor formulations more than the constant-LR simple critics.
4. **The new init lowers the blow-up in failing arms but does not rescue them.** Max grad-norm falls for full (685 -> 95), full_hinge (212 -> 15), wgangp_ref (53 -> 6.6) and full_r1 (30 -> 5.4). Those arms still fail every check, so the problem is the formulation, not the starting point.

## Recommendations

- Keep B_cap3, the secant + R1 + cap-all(c=3) recipe at lr c×0.5 g×1, as the simple-critic lead. It is the only simple arm ranked in the top 3 under both inits.
- Close the remaining gap to k3p_constant (25 fails vs 41) by looking at its anchor/guard components on top of B_cap3. The component tests sec_anchor, sec_guard and sec_guard_anchor were run on the c=1 base, not on B_cap3.
- Rank on a margin, not on rank order: treat arms as tied unless their fails outside transit differ by 30 or more, since that is how far the init change alone moved them. To separate B_cap3, lr_c0.5_g0.5 and sec_nodamp for real, change the objective rather than the seed. One option is a second shift or a longer post-shift window, which gives more checks per run.
- Drop rp_center as a finalist. Its old #4 came from a lucky init: fails 54 -> 118, prehold 66 -> 53.

## New-init leaderboard

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

## Old vs new (every arm, new-init order)

| arm | place old -> new | prehold old -> new | arrival old -> new | departures old -> new | fails outside transit old -> new (Δ) | max grad-norm old -> new | final HQ old -> new |
|---|---|---|---|---|---|---|---|
| k3p_stock_ref | ref@1 -> ref@1 | 120 -> 120 | 1960 -> 1220 | 3 -> 1 | 0 -> 0 (+0) | 7.45 -> 3.55 | 0.922 -> 0.954 |
| k3p_constant | ref@3 -> ref@1 | 88 -> 95 | 360 -> 360 | 5 -> 1 | 50 -> 25 (-25) | 5.81 -> 4.90 | 0.989 -> 0.993 |
| B_cap3 | 1 -> 1 | 82 -> 80 | 140 -> 250 | 10 -> 11 | 38 -> 41 (+3) | 5.94 -> 5.43 | 0.990 -> 0.995 |
| lr_c0.5_g0.5 | 15 -> 2 | 97 -> 86 | 560 -> 240 | 29 -> 15 | 81 -> 50 (-31) | 3.05 -> 2.40 | 0.992 -> 0.993 |
| sec_nodamp | 3 -> 3 | 97 -> 97 | 230 -> 160 | 21 -> 17 | 50 -> 52 (+2) | 2.64 -> 2.73 | 0.989 -> 0.982 |
| lr_c0.5_g2 | 8 -> 4 | 100 -> 99 | 740 -> 400 | 9 -> 9 | 61 -> 54 (-7) | 3.32 -> 3.48 | 0.991 -> 0.008 |
| ka2_stock_ref | ref@26 -> ref@5 | 61 -> 78 | 120 -> 390 | 10 -> 12 | 142 -> 56 (-86) | n/a -> n/a | 0.986 -> 0.724 |
| lr_c1_g0.5 | 13 -> 5 | 85 -> 86 | 20 -> 90 | 26 -> 25 | 78 -> 61 (-17) | 2.57 -> 2.35 | 0.979 -> 0.991 |
| lr_c0.25_g0.25 | 6 -> 6 | 87 -> 75 | 240 -> 100 | 16 -> 6 | 56 -> 64 (+8) | 2.55 -> 2.80 | 0.980 -> 0.139 |
| sec_drift | 22 -> 7 | 79 -> 83 | 150 -> 250 | 35 -> 24 | 116 -> 65 (-51) | 3.48 -> 2.82 | 0.957 -> 0.995 |
| combo_a | 9 -> 8 | 94 -> 90 | 170 -> 200 | 20 -> 19 | 68 -> 66 (-2) | 2.54 -> 3.35 | 0.987 -> 0.966 |
| lr_c0.5_g1 | 2 -> 9 | 96 -> 81 | 260 -> 380 | 17 -> 17 | 42 -> 69 (+27) | 2.83 -> 2.66 | 0.993 -> 0.976 |
| c3_r1w | 14 -> 10 | 78 -> 83 | 170 -> 270 | 15 -> 10 | 79 -> 70 (-9) | 6.42 -> 5.53 | 0.098 -> 0.983 |
| sec_center | 11 -> 11 | 78 -> 93 | 180 -> 150 | 27 -> 15 | 71 -> 70 (-1) | 3.13 -> 2.73 | 0.949 -> 0.985 |
| B_margin | 23 -> 12 | 85 -> 97 | 240 -> 420 | 13 -> 12 | 118 -> 73 (-45) | 3.05 -> 2.82 | 0.935 -> 0.919 |
| c3_capinterp | 7 -> 13 | 94 -> 109 | 250 -> 50 | 20 -> 18 | 58 -> 74 (+16) | 5.07 -> 5.41 | 0.959 -> 0.985 |
| sec_guard_anchor | 24 -> 14 | 55 -> 83 | 320 -> 280 | 24 -> 32 | 128 -> 74 (-54) | 2.89 -> 2.54 | 0.912 -> 0.991 |
| sec_anchor | 21 -> 15 | 54 -> 82 | 220 -> 560 | 24 -> 37 | 115 -> 86 (-29) | 2.89 -> 2.54 | 0.472 -> 0.941 |
| B_cap2 | 5 -> 16 | 89 -> 57 | 280 -> 130 | 13 -> 18 | 54 -> 91 (+37) | 4.31 -> 4.14 | 0.992 -> 0.976 |
| c3_capnopath | 16 -> 17 | 78 -> 84 | 110 -> 40 | 33 -> 30 | 90 -> 98 (+8) | 15.67 -> 9.30 | 0.917 -> 0.135 |
| B_nnpair | 20 -> 18 | 81 -> 75 | 250 -> 200 | 31 -> 31 | 111 -> 102 (-9) | 5.79 -> 5.11 | 0.967 -> 0.945 |
| secant_r1_b2 | 12 -> 19 | 83 -> 48 | 630 -> 280 | 33 -> 27 | 77 -> 106 (+29) | 2.82 -> 2.70 | 0.981 -> 0.960 |
| sec_guard | 17 -> 20 | 74 -> 63 | 400 -> 230 | 23 -> 32 | 98 -> 106 (+8) | 2.82 -> 2.70 | 0.985 -> 0.102 |
| rp_center | 4 -> 21 | 66 -> 53 | 300 -> 170 | 5 -> 6 | 54 -> 118 (+64) | 4.88 -> 5.63 | 0.989 -> 0.987 |
| sec_t1 | 10 -> 22 | 76 -> 68 | 190 -> 150 | 24 -> 28 | 68 -> 129 (+61) | 5.43 -> 4.31 | 0.210 -> 0.545 |
| lr_c1_g2 | 18 -> 23 | 72 -> 84 | 340 -> 220 | 20 -> 24 | 106 -> 137 (+31) | 4.41 -> 3.60 | 0.226 -> 0.346 |
| ref:ka2-constant | ref@26 -> ref@24 | 61 -> 61 | 120 -> 120 | 10 -> 10 | 142 -> 142 (+0) | n/a -> n/a | 0.986 -> 0.986 |
| lr_c2_g1 | 26 -> 24 | 34 -> 75 | 240 -> 190 | 36 -> 44 | 194 -> 145 (-49) | 3.99 -> 2.98 | 0.967 -> 0.794 |
| combo_b | 19 -> 25 | 58 -> 79 | 180 -> 220 | 26 -> 24 | 107 -> 156 (+49) | 3.24 -> 5.11 | 0.992 -> 0.358 |
| lr_c2_g0.5 | 25 -> 26 | 49 -> 67 | 140 -> 120 | 67 -> 61 | 133 -> 157 (+24) | 3.15 -> 4.76 | 0.961 -> 0.974 |
| lr_c2_g2 | 27 -> 27 | 6 -> 8 | 350 -> 350 | 28 -> 34 | 225 -> 214 (-11) | 3.93 -> 7.20 | 0.974 -> 0.964 |
| rp_center_min | 28 -> 28 | 8 -> 1 | 610 -> 950 | 32 -> 17 | 236 -> 219 (-17) | 5.44 -> 4.06 | 0.945 -> 0.906 |
| int_r1_b2 | 29 -> 29 | 28 -> 21 | 60 -> 160 | 63 -> 63 | 249 -> 226 (-23) | 1.92 -> 2.21 | 0.679 -> 0.733 |
| wgan_huber | 43 -> 30 | 0 -> 0 | none -> 1020 | 0 -> 2 | 120 -> 234 (+114) | 29.72 -> 16.16 | 0.792 -> 0.709 |
| secant_r1 | 30 -> 31 | 51 -> 77 | none -> none | 7 -> 11 | 69 -> 43 (-26) | 394.54 -> 270.27 | 0.259 -> 0.475 |
| int_r1_hinge | 32 -> 32 | 19 -> 21 | none -> none | 18 -> 19 | 101 -> 99 (-2) | 181.99 -> 219.04 | 0.010 -> 0.080 |
| no_path | 31 -> 33 | 35 -> 7 | none -> none | 14 -> 5 | 85 -> 113 (+28) | 201.68 -> 1201.09 | 0.000 -> 0.031 |
| int_r1 | 33 -> 34 | 9 -> 5 | none -> none | 9 -> 7 | 111 -> 115 (+4) | 138.60 -> 72.43 | 0.043 -> 0.399 |
| sec_lazy4 | 42 -> 35 | 0 -> 1 | none -> none | 0 -> 1 | 120 -> 119 (-1) | 4.20 -> 3.91 | 0.054 -> 0.635 |
| c3_rate | 34 -> 36 | 0 -> 0 | none -> none | 0 -> 0 | 120 -> 120 (+0) | 6.10 -> 7.28 | 0.000 -> 0.402 |
| full | 35 -> 37 | 0 -> 0 | none -> none | 0 -> 0 | 120 -> 120 (+0) | 685.01 -> 95.06 | 0.016 -> 0.136 |
| full_hinge | 36 -> 38 | 0 -> 0 | none -> none | 0 -> 0 | 120 -> 120 (+0) | 211.99 -> 14.51 | 0.005 -> 0.224 |
| full_r1 | 37 -> 39 | 0 -> 0 | none -> none | 0 -> 0 | 120 -> 120 (+0) | 29.86 -> 5.41 | 0.094 -> 0.706 |
| lr_c4_g4 | 39 -> 40 | 0 -> 0 | none -> none | 0 -> 0 | 120 -> 120 (+0) | 30.68 -> 61.75 | 0.048 -> 0.000 |
| no_cap | 40 -> 41 | 0 -> 0 | none -> none | 0 -> 0 | 120 -> 120 (+0) | 4629336.00 -> 6715439.00 | 0.000 -> 0.000 |
| no_real | 41 -> 42 | 0 -> 0 | none -> none | 0 -> 0 | 120 -> 120 (+0) | 25.52 -> 343.71 | 0.389 -> 0.093 |
| wgan_margin | 44 -> 43 | 0 -> 0 | none -> none | 0 -> 0 | 120 -> 120 (+0) | 1796275.12 -> 1704162.75 | 0.011 -> 0.028 |
| wgangp_ref | 45 -> 44 | 0 -> 0 | none -> none | 0 -> 0 | 120 -> 120 (+0) | 52.73 -> 6.59 | 0.631 -> 0.593 |
| int_r1_rp | 38 -> 45 | 0 -> 0 | none -> none | 0 -> 2 | 120 -> 120 (+0) | 34.30 -> 345.12 | 0.045 -> 0.010 |

## Rank changes (positive = moved up)

- rp_center: 4 -> 21 (-17)
- sec_drift: 22 -> 7 (+15)
- lr_c0.5_g0.5: 15 -> 2 (+13)
- wgan_huber: 43 -> 30 (+13)
- sec_t1: 10 -> 22 (-12)
- B_cap2: 5 -> 16 (-11)
- B_margin: 23 -> 12 (+11)
- sec_guard_anchor: 24 -> 14 (+10)
- lr_c1_g0.5: 13 -> 5 (+8)
- int_r1_rp: 38 -> 45 (-7)
- lr_c0.5_g1: 2 -> 9 (-7)
- secant_r1_b2: 12 -> 19 (-7)
- sec_lazy4: 42 -> 35 (+7)
- c3_capinterp: 7 -> 13 (-6)
- combo_b: 19 -> 25 (-6)
- sec_anchor: 21 -> 15 (+6)
- lr_c1_g2: 18 -> 23 (-5)
- c3_r1w: 14 -> 10 (+4)
- lr_c0.5_g2: 8 -> 4 (+4)
- sec_guard: 17 -> 20 (-3)
- c3_rate: 34 -> 36 (-2)
- full: 35 -> 37 (-2)
- full_hinge: 36 -> 38 (-2)
- full_r1: 37 -> 39 (-2)
- no_path: 31 -> 33 (-2)
- B_nnpair: 20 -> 18 (+2)
- lr_c2_g1: 26 -> 24 (+2)
- c3_capnopath: 16 -> 17 (-1)
- int_r1: 33 -> 34 (-1)
- lr_c2_g0.5: 25 -> 26 (-1)
- lr_c4_g4: 39 -> 40 (-1)
- no_cap: 40 -> 41 (-1)
- no_real: 41 -> 42 (-1)
- secant_r1: 30 -> 31 (-1)
- combo_a: 9 -> 8 (+1)
- wgan_margin: 44 -> 43 (+1)
- wgangp_ref: 45 -> 44 (+1)
- B_cap3: 1 -> 1 (+0)
- int_r1_b2: 29 -> 29 (+0)
- int_r1_hinge: 32 -> 32 (+0)
- lr_c0.25_g0.25: 6 -> 6 (+0)
- lr_c2_g2: 27 -> 27 (+0)
- rp_center_min: 28 -> 28 (+0)
- sec_center: 11 -> 11 (+0)
- sec_nodamp: 3 -> 3 (+0)
