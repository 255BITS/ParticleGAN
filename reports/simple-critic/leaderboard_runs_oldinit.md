# Simple-critic leaderboard (runs_oldinit)

| # | arm | formulation | prehold | arrival | post-arrival | departures | longest fail streak | final suffix | final HQ | max abs D(real) | max grad-norm | fails outside transit | init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | k3p_stock_ref | REF: K3P v0.8.0 as released (RpGAN+K3P penalty/anchor/guard, own noise + LR schedules) | 120/120 | 1960 | 25/25 | 3 | 24 | 25 (from 4360) | 0.922 | 2.48 | 7.45 | 0 | old |
| 1 | B_cap3 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=3) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 82/120 | 140 | 207/207 | 10 | 23 | 207 (from 2540) | 0.990 | 4.59 | 5.94 | 38 | old |
| 2 | lr_c0.5_g1 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 96/120 | 260 | 177/195 | 17 | 21 | 123 (from 3380) | 0.993 | 1.87 | 2.83 | 42 | old |
| ref | k3p_constant | REF: K3P v0.8.0 critic/penalty, noise off, constant LRs (floors 1) | 88/120 | 360 | 167/185 | 5 | 32 | 123 (from 3380) | 0.989 | 2.63 | 5.81 | 50 | old |
| 3 | sec_nodamp | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] | 97/120 | 230 | 171/198 | 21 | 24 | 3 (from 4580) | 0.989 | 2.26 | 2.64 | 50 | old |
| 4 | rp_center | rplogistic + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=3) + pair-center(1) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 66/120 | 300 | 191/191 | 5 | 50 | 191 (from 2700) | 0.989 | 3.21 | 4.88 | 54 | old |
| 5 | B_cap2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=2) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 89/120 | 280 | 170/193 | 13 | 23 | 36 (from 4250) | 0.992 | 4.03 | 4.31 | 54 | old |
| 6 | lr_c0.25_g0.25 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×0.25 g×0.25} | 87/120 | 240 | 174/197 | 16 | 15 | 86 (from 3750) | 0.980 | 3.05 | 2.55 | 56 | old |
| 7 | c3_capinterp | wgan + r1(1) + path-secant(10,t=0.5) + cap-interp(10,c=3) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 94/120 | 250 | 164/196 | 20 | 23 | 10 (from 4510) | 0.959 | 5.44 | 5.07 | 58 | old |
| 8 | lr_c0.5_g2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×0.5 g×2} | 100/120 | 740 | 106/147 | 9 | 36 | 23 (from 4380) | 0.991 | 4.20 | 3.32 | 61 | old |
| 9 | combo_a | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) + center(1) [Dβ2=0.9, A2=0] | 94/120 | 170 | 162/204 | 20 | 26 | 8 (from 4530) | 0.987 | 2.10 | 2.54 | 68 | old |
| 10 | sec_t1 | wgan + r1(1) + path-secant(10,t=1) + cap-all(10,c=1) [Dβ2=0.9] | 76/120 | 190 | 178/202 | 24 | 18 | 0 | 0.210 | 4.85 | 5.43 | 68 | old |
| 11 | sec_center | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) + center(1) [Dβ2=0.9] | 78/120 | 180 | 174/203 | 27 | 28 | 73 (from 3880) | 0.949 | 1.79 | 3.13 | 71 | old |
| 12 | secant_r1_b2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] | 83/120 | 630 | 118/158 | 33 | 22 | 37 (from 4240) | 0.981 | 3.17 | 2.82 | 77 | old |
| 13 | lr_c1_g0.5 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×1 g×0.5} | 85/120 | 20 | 176/219 | 26 | 20 | 5 (from 4560) | 0.979 | 2.02 | 2.57 | 78 | old |
| 14 | c3_r1w | wgan + r1(0.1) + path-secant(10,t=0.5) + cap-all(10,c=3) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 78/120 | 170 | 167/204 | 15 | 28 | 0 | 0.098 | 6.04 | 6.42 | 79 | old |
| 15 | lr_c0.5_g0.5 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×0.5 g×0.5} | 97/120 | 560 | 107/165 | 29 | 22 | 33 (from 4280) | 0.992 | 2.99 | 3.05 | 81 | old |
| 16 | c3_capnopath | wgan + r1(1) + path-secant(10,t=0.5) + cap-ends(10,c=3) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 78/120 | 110 | 162/210 | 33 | 34 | 1 (from 4600) | 0.917 | 9.61 | 15.67 | 90 | old |
| 17 | sec_guard | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] + K3P guard(ratio=5,min=200) | 74/120 | 400 | 129/181 | 23 | 20 | 98 (from 3630) | 0.985 | 2.20 | 2.82 | 98 | old |
| 18 | lr_c1_g2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×1 g×2} | 72/120 | 340 | 129/187 | 20 | 51 | 0 | 0.226 | 6.00 | 4.41 | 106 | old |
| 19 | combo_b | wgan + r1(1) + path-secant(10,t=1) + cap-all(10,c=1) + center(1) [Dβ2=0.9, A2=0] | 58/120 | 180 | 158/203 | 26 | 39 | 58 (from 4030) | 0.992 | 3.20 | 3.24 | 107 | old |
| 20 | B_nnpair | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) + nnpair [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 81/120 | 250 | 124/196 | 31 | 47 | 6 (from 4550) | 0.967 | 4.43 | 5.79 | 111 | old |
| 21 | sec_anchor | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] + K3P EMA-anchor(0.5*prox,decay=0.999) | 54/120 | 220 | 150/199 | 24 | 39 | 0 | 0.472 | 3.08 | 2.89 | 115 | old |
| 22 | sec_drift | wgan + drift(0.001)+r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] | 79/120 | 150 | 131/206 | 35 | 52 | 1 (from 4600) | 0.957 | 3.90 | 3.48 | 116 | old |
| 23 | B_margin | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) + margin(10,κ=7.14) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 85/120 | 240 | 114/197 | 13 | 60 | 2 (from 4590) | 0.935 | 3.07 | 3.05 | 118 | old |
| 24 | sec_guard_anchor | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] + K3P guard(ratio=5,min=200) + K3P EMA-anchor(0.5*prox,decay=0.999) | 55/120 | 320 | 126/189 | 24 | 77 | 7 (from 4540) | 0.912 | 3.14 | 2.89 | 128 | old |
| 25 | lr_c2_g0.5 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×2 g×0.5} | 49/120 | 140 | 145/207 | 67 | 23 | 10 (from 4510) | 0.961 | 2.65 | 3.15 | 133 | old |
| ref | ka2_stock_ref | REF: stock KA2 worker rerun (RpGAN+KA2 penalty/controller, noise on) | 61/120 | 120 | 126/209 | 10 | 58 | 15 (from 4460) | 0.986 | n/a | n/a | 142 | old |
| ref | ref:ka2-constant | RpGAN+KA2 penalty/controller, noise on (archived) | 61/120 | 120 | 126/209 | 10 | 58 | 15 (from 4460) | 0.986 | n/a | n/a | 142 | old |
| 26 | lr_c2_g1 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×2 g×1} | 34/120 | 240 | 89/197 | 36 | 76 | 2 (from 4590) | 0.967 | 4.11 | 3.99 | 194 | old |
| 27 | lr_c2_g2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×2 g×2} | 6/120 | 350 | 75/186 | 28 | 102 | 6 (from 4550) | 0.974 | 7.85 | 3.93 | 225 | old |
| 28 | rp_center_min | rplogistic + cap-all(10,c=3) + pair-center(1) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 8/120 | 610 | 36/160 | 32 | 41 | 1 (from 4600) | 0.945 | 3.01 | 5.44 | 236 | old |
| 29 | int_r1_b2 | wgan + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) [Dβ2=0.9] | 28/120 | 60 | 58/215 | 63 | 25 | 0 | 0.679 | 1.64 | 1.92 | 249 | old |
| 30 | secant_r1 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) | 51/120 | none | 0/0 | 7 | 37 | 0 | 0.259 | 161.28 | 394.54 | 69 | old |
| 31 | no_path | wgan + drift(0.1) + cap-all(10,c=1) | 35/120 | none | 0/0 | 14 | 42 | 0 | 0.000 | 46.09 | 201.68 | 85 | old |
| 32 | int_r1_hinge | hinge + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 19/120 | none | 0/0 | 18 | 20 | 0 | 0.010 | 27.99 | 181.99 | 101 | old |
| 33 | int_r1 | wgan + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 9/120 | none | 0/0 | 9 | 64 | 0 | 0.043 | 30.33 | 138.60 | 111 | old |
| 34 | c3_rate | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=3) + rate(10) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.000 | 4.76 | 6.10 | 120 | old |
| 35 | full | wgan + drift(0.1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.016 | 120.32 | 685.01 | 120 | old |
| 36 | full_hinge | hinge + drift(0.1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.005 | 50.71 | 211.99 | 120 | old |
| 37 | full_r1 | wgan + r1(1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.094 | 6.93 | 29.86 | 120 | old |
| 38 | int_r1_rp | rplogistic + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.045 | 6.40 | 34.30 | 120 | old |
| 39 | lr_c4_g4 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, A2=0] {lr c×4 g×4} | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.048 | 32.60 | 30.68 | 120 | old |
| 40 | no_cap | wgan + drift(0.1) + path-lower(10,t=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.000 | 60288.24 | 4629336.00 | 120 | old |
| 41 | no_real | wgan + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.389 | 15.88 | 25.52 | 120 | old |
| 42 | sec_lazy4 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9, lazy_k=4] | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.054 | 4.02 | 4.20 | 120 | old |
| 43 | wgan_huber | wgan + huber(10,κ=7.14) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.792 | 12.56 | 29.72 | 120 | old |
| 44 | wgan_margin | wgan + margin(10,κ=7.14) [Dβ2=0.9, A2=0] {lr c×0.5 g×1} | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.011 | 97405.10 | 1796275.12 | 120 | old |
| 45 | wgangp_ref | wgan + drift(0.1) + path-two_sided(10,t=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.631 | 8.34 | 52.73 | 120 | old |

`*` = still running (scored through its last observation). Departures and streaks exclude the shift transit (2400 to first arrival). ref = archived KA2 constant-LR run (not a simple arm). init: bfz = batch_feature_zero (#194) with G/D/prior all different from the old random draw; old = pre-#194 random init (no receipt).
