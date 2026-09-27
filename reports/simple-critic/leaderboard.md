# Simple-critic leaderboard (runs)

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
