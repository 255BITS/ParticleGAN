# K3P constant-LR suite leaderboard

Pass per task as defined in run_suite.py; ring cells: prehold x/120, arrival per shift (x = never), f = fails outside transit, d = departures. "copied" rows were not trained separately: result.json copied from k3p_const_ams after a 400-update bit-exact check, or from runs_smoke/a where the hashes differed (ab_nodirect two_pole); k3p_simple (both removals) is the full-budget run that supports them.

| # | arm | passes | native | transfer 19 | hold | shift | ring8-shift (prehold arrival fails dep) | ring8-multishift (prehold arrivals fails dep) | formulation |
|---|---|---|---|---|---|---|---|---|---|
| 1 | k3p_stock | 18/26 | 3/3 | 14/19 | FAIL | FAIL | 120/120 +1220 f0 d1 | 120/120 +1990/x/+1640 f2 d4 | K3P as released on develop: RpGAN + K3P penalty/EMA anchor/spike guard + A2, input noise .5->0 and output noise .029, cosine LR decay (net floor .01 over a 1600 horizon, prior floor .05), Adam (0,.999) |
| 2 | k3p_stock_ams | 17/26 | 2/3 | 14/19 | FAIL | FAIL | 120/120 +1460 f0 d0 | 110/120 x/x/x f10 d1 | k3p + ams: k3p_stock with AMSGrad on every optimizer |
| 3 | k3p_nonoise | 15/26 | 0/3 | 15/19 | FAIL | FAIL | 110/120 +1020 f10 d2 | 110/120 +1010/x/x f18 d4 | k3p_stock (LR decay, Adam) with no instance noise |
| 4 | ab_noanchor (copied: 26/26 results) | 14/26 | 0/3 | 12/19 | FAIL | FAIL | 120/120 +430 f0 d0 | 120/120 +430/+190/+20 f0 d0 | ablation of k3p_const_ams (constant LR + AMSGrad, no noise): EMA anchor term off (reg_anchor_weight 0); inert at constant LR (s==1), bit-exactness check |
| 5 | ab_nodirect (copied: 26/26 results) | 14/26 | 0/3 | 12/19 | FAIL | FAIL | 120/120 +430 f0 d0 | 120/120 +430/+190/+20 f0 d0 | ablation of k3p_const_ams (constant LR + AMSGrad, no noise): direct-particle response off (gain False, betas = G's (0,.999)) |
| 6 | k3p_const_ams | 14/26 | 0/3 | 12/19 | FAIL | FAIL | 120/120 +430 f0 d0 | 120/120 +430/+190/+20 f0 d0 | k3p constant ams: constant LR (floors 1) + AMSGrad, no instance noise |
| 7 | k3p_simple | 14/26 | 0/3 | 12/19 | FAIL | FAIL | 120/120 +430 f0 d0 | 120/120 +430/+190/+20 f0 d0 | k3p_simple: k3p_const_ams minus every component whose removal cost no suite pass or ring stability: EMA anchor/EMA critic term off (reg_anchor_weight 0; inert at constant LR), direct-particle response off (gain False, betas = G's). Keeps AMSGrad, constant LR, no noise, R1+fake one-sided RMS cap, spike guard, A2, prior LR x2 |
| 8 | ab_absunits | 13/26 | 0/3 | 11/19 | FAIL | FAIL | 120/120 +460 f0 d1 | 120/120 +460/+190/+60 f0 d1 | ablation of k3p_const_ams (constant LR + AMSGrad, no noise): penalty in L2 units: ||g_r||^2 + relu(||g_f|| - kappa)^2 (no 1/d, 1/sqrt d) |
| 9 | k3p_const_ams_c3 | 13/26 | 0/3 | 11/19 | FAIL | FAIL | 120/120 +420 f0 d0 | 120/120 +420/+80/+50 f0 d0 | k3p_const_ams + penalty coefficient x3 (reg_coeff 3) |
| 10 | k3p_nonoise_ams | 13/26 | 0/3 | 13/19 | FAIL | FAIL | 120/120 x f0 d0 | 120/120 x/x/+1840 f0 d0 | k3p_stock LR decay + AMSGrad, no instance noise |
| 11 | ab_prior1 | 13/26 | 0/3 | 13/19 | FAIL | FAIL | 120/120 +370 f5 d3 | 120/120 +370/+430/+30 f5 d3 | ablation of k3p_const_ams (constant LR + AMSGrad, no noise): prior LR multiplier 2 -> 1 (prior LR = G's) |
| 12 | ab_noA2 | 13/26 | 0/3 | 12/19 | FAIL | FAIL | 120/120 +420 f0 d0 | 120/120 +420/+120/+310 f22 d2 | ablation of k3p_const_ams (constant LR + AMSGrad, no noise): A2 latent-row damping off (latent_damping_max_rate 0) |
| 13 | k3p_const | 13/26 | 0/3 | 13/19 | FAIL | FAIL | 95/120 +360 f25 d1 | 95/120 +360/+440/+70 f52 d3 | K3P constant LR (floors 1), Adam, no instance noise (baseline for the constant problem) |
| 14 | simple_B_cap3 | 13/26 | 0/3 | 13/19 | FAIL | FAIL | 98/120 +440 f27 d13 | 98/120 +440/+360/+90 f93 d23 | simple-critic winner B_cap3: wgan + R1(1) + secant to nearest real (10, t .5) + cap-all (10, c=3); critic Adam (0,.9), critic LR x.5, no guard/EMA anchor, A2 off; RpGAN G loss; no noise, constant LR |
| 15 | ab_noguard | 12/26 | 0/3 | 12/19 | FAIL | FAIL | 120/120 x f0 d0 | 120/120 x/x/x f0 d0 | ablation of k3p_const_ams (constant LR + AMSGrad, no noise): critic spike guard off (d_guard_ratio 0) |
| 16 | ab_r1only | 12/26 | 0/3 | 12/19 | FAIL | FAIL | 0/120 x f120 d0 | 0/120 x/x/+1420 f123 d1 | ablation of k3p_const_ams (constant LR + AMSGrad, no noise): penalty = R1 on reals only (RMS units), fake one-sided cap dropped |
| 17 | ab_caponly | 10/26 | 0/3 | 10/19 | FAIL | FAIL | 6/120 +240 f188 d23 | 6/120 +240/+160/+110 f231 d42 | ablation of k3p_const_ams (constant LR + AMSGrad, no noise): penalty = one-sided RMS cap (kappa 1) on reals and fakes, no R1 |
| 18 | k3p_stock_oldinit | 1/2 | – | 0/0 | – | – | 120/120 +1960 f0 d3 | 120/120 x/x/+1550 f0 d1 | calibration: k3p_stock with no particlegan.init call anywhere (constructor init on ring too). native/toy/hold/shift already use constructor init in this suite, so only the ring tasks differ from k3p_stock |

## Per task (status, key metric)

| task | k3p_stock | k3p_stock_ams | k3p_nonoise | ab_noanchor | ab_nodirect | k3p_const_ams | k3p_simple | ab_absunits | k3p_const_ams_c3 | k3p_nonoise_ams | ab_prior1 | ab_noA2 | k3p_const | simple_B_cap3 | ab_noguard | ab_r1only | ab_caponly | k3p_stock_oldinit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid100 | P 100/0.987 | F 88/0.978 | F 99/0.983 | F 99/0.973 | F 99/0.973 | F 99/0.973 | F 99/0.973 | F 99/0.969 | F 99/0.961 | F 99/0.978 | F 96/0.922 | F 99/0.977 | F 86/0.958 | F 100/0.933 | F 99/0.973 | F 98/0.982 | F 100/0.977 | – |
| rotated100 | P 100/0.982 | P 100/0.977 | F 100/0.975 | F 100/0.971 | F 100/0.971 | F 100/0.971 | F 100/0.971 | F 100/0.958 | F 100/0.950 | F 100/0.974 | F 100/0.966 | F 100/0.964 | F 100/0.915 | F 100/0.930 | F 100/0.971 | F 100/0.971 | F 92/0.981 | – |
| staggered100 | P 100/0.987 | P 100/0.978 | F 100/0.982 | F 100/0.970 | F 100/0.970 | F 100/0.970 | F 100/0.970 | F 100/0.959 | F 100/0.947 | F 100/0.972 | F 100/0.970 | F 100/0.965 | F 100/0.970 | F 100/0.948 | F 100/0.970 | F 100/0.970 | F 100/0.985 | – |
| hold | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 299/1200+0/0 | F 10/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | – |
| ring8-multishift | F 2 | F 10 | F 18 | P 0 | P 0 | P 0 | P 0 | P 0 | P 0 | F 0 | F 5 | F 22 | F 52 | F 93 | F 0 | F 123 | F 231 | F 0 |
| shift | F 0/81 | F 0/81 | F 0/81 | F 0/81 | F 0/81 | F 0/81 | F 0/81 | F 71/81 | F 0/81 | F 0/81 | F 8/81 | F 0/81 | F 0/81 | F 3/81 | F 0/81 | F 0/81 | F 18/81 | – |
| ring8-shift | P 0 | P 0 | F 10 | P 0 | P 0 | P 0 | P 0 | P 0 | P 0 | F 0 | F 5 | P 0 | F 25 | F 27 | F 0 | F 120 | F 188 | P 0 |
| two_pole | P 9/24 | F 0/24 | P 10/24 | F 2/24 | F 0/24 | F 2/24 | F 0/24 | F 2/24 | F 0/24 | F 0/24 | F 0/24 | F 2/24 | P 10/24 | P 20/24 | F 2/24 | F 2/24 | P 13/24 | – |
| trajectory | P 23/24 | P 23/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | F 0/24 | F 0/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 5/24 | P 22/24 | P 22/24 | P 23/24 | – |
| residual_student | P 20/24 | P 20/24 | P 20/24 | P 8/24 | P 8/24 | P 8/24 | P 8/24 | F 0/24 | P 23/24 | P 18/24 | P 23/24 | P 8/24 | P 20/24 | F 0/24 | P 8/24 | P 8/24 | P 14/24 | – |
| unipolar | P 19/24 | P 19/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 14/24 | P 15/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 19/24 | – |
| ae_gan_hold | P 22/24 | P 13/24 | P 15/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 15/24 | P 11/24 | P 22/24 | P 22/24 | P 22/24 | – |
| cover_leftover | P 13/24 | P 13/24 | P 11/24 | P 11/24 | P 11/24 | P 11/24 | P 11/24 | P 13/24 | P 12/24 | P 11/24 | P 9/24 | P 11/24 | P 11/24 | P 14/24 | P 11/24 | P 10/24 | P 9/24 | – |
| unused_token_hold | P 11/24 | P 11/24 | P 10/24 | P 10/24 | P 10/24 | P 10/24 | P 10/24 | P 7/24 | P 6/24 | P 10/24 | P 10/24 | P 10/24 | P 10/24 | P 12/24 | P 10/24 | P 10/24 | P 7/24 | – |
| mid_scale_identity | P 17/24 | P 17/24 | P 16/24 | P 16/24 | P 16/24 | P 16/24 | P 16/24 | P 13/24 | P 14/24 | P 16/24 | P 16/24 | P 16/24 | P 16/24 | P 15/24 | P 16/24 | P 16/24 | P 17/24 | – |
| mode_hold | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 1/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | – |
| v_two_broad | P 23/24 | P 23/24 | P 21/24 | P 21/24 | P 21/24 | P 21/24 | P 21/24 | P 22/24 | P 22/24 | P 21/24 | P 20/24 | P 23/24 | P 21/24 | P 23/24 | P 21/24 | P 23/24 | P 23/24 | – |
| v_unequal_mass | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 1/24 | F 3/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 2/24 | F 0/24 | F 0/24 | F 0/24 | – |
| v_unequal_width | F 0/24 | P 11/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | P 21/24 | F 0/24 | F 0/24 | P 8/24 | F 0/24 | F 0/24 | F 0/24 | – |
| v_anisotropic | P 6/24 | P 6/24 | P 7/24 | P 5/24 | P 5/24 | P 5/24 | P 5/24 | P 21/24 | P 22/24 | P 21/24 | P 21/24 | P 20/24 | P 7/24 | F 2/24 | P 5/24 | P 7/24 | F 0/24 | – |
| v_overlap | P 11/24 | P 9/24 | P 13/24 | P 9/24 | P 9/24 | P 9/24 | P 9/24 | F 0/24 | F 0/24 | P 9/24 | F 3/24 | P 11/24 | F 0/24 | P 7/24 | P 9/24 | P 9/24 | F 0/24 | – |
| v_spiral | P 21/24 | P 23/24 | P 23/24 | P 24/24 | P 24/24 | P 24/24 | P 24/24 | P 22/24 | P 24/24 | P 24/24 | P 23/24 | P 23/24 | P 23/24 | P 23/24 | P 24/24 | P 24/24 | P 23/24 | – |
| i_stripes2 | P 22/24 | P 8/24 | F 0/24 | P 5/24 | P 5/24 | P 5/24 | P 5/24 | P 7/24 | P 13/24 | P 17/24 | F 0/24 | P 5/24 | F 0/24 | P 12/24 | P 5/24 | P 5/24 | F 1/24 | – |
| i_bars4 | P 10/24 | P 5/24 | P 18/24 | F 3/24 | F 3/24 | F 3/24 | F 3/24 | P 20/24 | F 0/24 | P 15/24 | F 0/24 | F 3/24 | P 18/24 | F 0/24 | F 3/24 | F 3/24 | F 0/24 | – |
| i_blobs4 | F 0/24 | F 0/24 | P 18/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | P 9/24 | P 21/24 | F 0/24 | P 18/24 | F 0/24 | P 18/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | – |
| i_intensity2 | F 0/24 | F 0/24 | P 6/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 4/24 | F 0/24 | F 0/24 | P 5/24 | F 0/24 | F 1/24 | P 11/24 | F 0/24 | F 0/24 | F 3/24 | – |

## Runtime per task (s, wall under concurrency; * = result copied from another arm, see result.json 'inferred')

| task | k3p_stock | k3p_stock_ams | k3p_nonoise | ab_noanchor | ab_nodirect | k3p_const_ams | k3p_simple | ab_absunits | k3p_const_ams_c3 | k3p_nonoise_ams | ab_prior1 | ab_noA2 | k3p_const | simple_B_cap3 | ab_noguard | ab_r1only | ab_caponly | k3p_stock_oldinit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid100 | 159 | 144 | 138 | 121* | 121* | 121 | 138 | 242 | 232 | 187 | 258 | 203 | 153 | 110 | 232 | 225 | 246 | – |
| rotated100 | 163 | 147 | 140 | 125* | 125* | 125 | 145 | 267 | 247 | 193 | 282 | 232 | 156 | 112 | 249 | 250 | 273 | – |
| staggered100 | 176 | 160 | 151 | 132* | 132* | 132 | 164 | 298 | 261 | 206 | 331 | 280 | 162 | 117 | 300 | 280 | 295 | – |
| hold | 171 | 166 | 190 | 136* | 136* | 136 | 187 | 146 | 78 | 172 | 407 | 418 | 138 | 132 | 400 | 323 | 337 | – |
| ring8-multishift | 152 | 151 | 186 | 118* | 118* | 118 | 147 | 235 | 128 | 154 | 264 | 190 | 123 | 109 | 247 | 201 | 236 | 152 |
| shift | 73 | 67 | 78 | 51* | 51* | 51 | 65 | 105 | 58 | 62 | 113 | 106 | 54 | 52 | 101 | 94 | 99 | – |
| ring8-shift | 68 | 78 | 93 | 64* | 64* | 64 | 70 | 113 | 64 | 87 | 124 | 92 | 64 | 57 | 121 | 106 | 118 | 78 |
| two_pole | 2 | 2 | 2 | 2* | 2* | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 3 | 4 | 3 | – |
| trajectory | 3 | 3 | 3 | 3* | 3* | 3 | 2 | 4 | 3 | 4 | 5 | 4 | 3 | 3 | 5 | 4 | 6 | – |
| residual_student | 3 | 4 | 3 | 3* | 3* | 3 | 2 | 4 | 3 | 4 | 6 | 4 | 3 | 2 | 4 | 5 | 6 | – |
| unipolar | 4 | 4 | 3 | 3* | 3* | 3 | 3 | 4 | 3 | 4 | 5 | 4 | 3 | 3 | 6 | 5 | 6 | – |
| ae_gan_hold | 2 | 3 | 2 | 2* | 2* | 2 | 2 | 4 | 2 | 3 | 4 | 3 | 2 | 2 | 4 | 4 | 5 | – |
| cover_leftover | 5 | 5 | 5 | 5* | 5* | 5 | 5 | 7 | 6 | 7 | 10 | 10 | 5 | 6 | 9 | 9 | 8 | – |
| unused_token_hold | 2 | 2 | 2 | 2* | 2* | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 2 | 2 | 3 | 4 | 3 | – |
| mid_scale_identity | 8 | 9 | 8 | 8* | 8* | 8 | 7 | 15 | 8 | 12 | 15 | 14 | 7 | 9 | 13 | 10 | 15 | – |
| mode_hold | 9 | 10 | 9 | 9* | 9* | 9 | 8 | 17 | 9 | 18 | 18 | 19 | 9 | 10 | 16 | 15 | 18 | – |
| v_two_broad | 21 | 21 | 21 | 22* | 22* | 22 | 21 | 42 | 25 | 31 | 56 | 54 | 18 | 20 | 48 | 39 | 47 | – |
| v_unequal_mass | 30 | 30 | 31 | 32* | 32* | 32 | 29 | 57 | 33 | 42 | 71 | 66 | 29 | 32 | 70 | 53 | 64 | – |
| v_unequal_width | 23 | 23 | 22 | 24* | 24* | 24 | 21 | 44 | 23 | 29 | 60 | 49 | 22 | 21 | 54 | 41 | 49 | – |
| v_anisotropic | 34 | 29 | 31 | 31* | 31* | 31 | 28 | 56 | 32 | 37 | 67 | 62 | 29 | 28 | 64 | 53 | 50 | – |
| v_overlap | 29 | 21 | 24 | 22* | 22* | 22 | 20 | 47 | 24 | 26 | 49 | 50 | 21 | 21 | 53 | 45 | 33 | – |
| v_spiral | 34 | 25 | 29 | 25* | 25* | 25 | 24 | 48 | 27 | 29 | 41 | 43 | 24 | 23 | 49 | 41 | 38 | – |
| i_stripes2 | 14 | 10 | 13 | 10* | 10* | 10 | 18 | 22 | 11 | 10 | 17 | 18 | 10 | 10 | 18 | 17 | 14 | – |
| i_bars4 | 14 | 10 | 14 | 10* | 10* | 10 | 31 | 19 | 11 | 10 | 16 | 20 | 9 | 10 | 19 | 16 | 12 | – |
| i_blobs4 | 15 | 10 | 13 | 10* | 10* | 10 | 29 | 18 | 11 | 10 | 17 | 19 | 10 | 9 | 18 | 15 | 11 | – |
| i_intensity2 | 14 | 10 | 13 | 9* | 9* | 9 | 30 | 17 | 11 | 10 | 16 | 18 | 9 | 9 | 15 | 14 | 9 | – |
| **sum (s)** | 1226 | 1142 | 1224 | 976 | 977 | 976 | 1199 | 1834 | 1313 | 1351 | 2257 | 1984 | 1064 | 910 | 2120 | 1872 | 2001 | 230 |

## Receipt checks

- ring8-multishift: k3p_stock_oldinit uses constructor init by declaration (calibration); all other arms share init 6a095898
- ring8-shift: k3p_stock_oldinit uses constructor init by declaration (calibration); all other arms share init 6a095898
