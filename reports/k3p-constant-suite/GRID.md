# k3p_simple hyperparameter grid (constant LR, AMSGrad, no noise)

## Question

k3p_simple scores 14/26 and shipped K3P (k3p_stock) scores 18/26. The gap is 5 tasks: grid100, rotated100 and
staggered100 (native), plus toy-img_bars4 and toy-two_pole. Can tuning k3p_simple's existing knobs close that gap
without losing ring stability? The rules: no new components, no noise, no LR schedule, no EMA anchor, no seed repeats.

## Result

**Winner: `gs2_c03_lr2_d05`. It scores 17/26: +3 over k3p_simple (14), −1 against k3p_stock (18).**

Its diff from k3p_simple is three config values:
- `reg_coeff` 1 → **0.3**
- `lr` .00425 → **.0085**
- `d_lr_mult` 1 → **0.5**

So the applied LRs are G .0085, prior .017 (still ×2 of G) and critic .00425 (unchanged). In effect G and the prior
run twice as fast relative to the critic, and the penalty is 0.3×.

- **Transfer: 15/19.** That is k3p_simple 12 and k3p_stock 14.
  - Gains: two_pole 13/24, img_bars4 19/24, img_blobs4 19/24, img_intensity2 9/24. k3p_stock fails blobs4 and
    intensity2.
  - Loss: vector_anisotropic (5 → 0/24).
- **Ring: both tasks still pass with 0 fails and 0 departures.**
  - Arrival is slower than k3p_simple: +940 against +430. It is still faster than stock's +1220.
  - Multishift re-arrivals: +940, +310, +110.
- **Native: still 0/3.**
  - No tested value of any knob reaches the coverage gate's per-mode covariance limit (eig ratio ≥ .4 on every mode).
  - The whole 1-pass gap to stock is the 3 native tasks. On the other 23 tasks the winner scores 17 and stock 15.

Receipts on all 26 tasks: the LR was constant, AMSGrad was on every group, noise was 0, and the applied LRs were
.00425 / .0085 / .017.

## Knobs (real field names)

Config keys (benchmark config → host recipes, and ring recipe via `RECIPE_KEYS`):
- `reg_coeff`: penalty coefficient.
- `reg_kappa`: RMS threshold of the one-sided fake cap.
- `lr`: G/D base LR. The prior LR is `lr × prior_lr_mult`.
- `d_lr_mult`
- `prior_lr_mult`
- `betas`, `prior_betas`
- `batch_size`: native only. Transfer hosts use their own batch, and the ring protocol stays at 2048.

K3P recipe fields:
- `d_guard_ratio`
- `latent_damping_max_rate` (A2)

Dropped:
- **R1 weight separate from the cap.** Not a knob: one `reg_coeff` scales both terms (`particlegan/grad_regularizers.py`).
- `reg_every`, `ema_decay` (diagnostic only), `prior_reg` (this would add a component).

Harness changes:
- `run_suite.py`: `--logs-dir`. The ring's `RECIPE_KEYS` now also passes `lr`, `betas`, `prior_betas` and `reg_kappa`.
  No earlier arm sets these keys, so earlier results are unchanged.
- New files: `run_grid.sh` (arm queue) and `grid_summarize.py` (all tables below).

## Metrics

Each native cell reads `status cov/acc HQ eigmin-eigmax cWORST`:
- **cov**: terminal coverage-gate checks passed, /5. Needs HQ ≥ .97 and every mode's covariance eigenvalue ratio in
  [.4, 1.7], among other conditions.
- **acc**: terminal accuracy checks passed, /5. Needs center RMS ≤ .2σ, mass TV ≤ .06 and a few others.
- **HQ**: final HQ.
- **eigmin–eigmax**: the final worst per-mode covariance eigenvalue ratio.
- **cWORST**: the worst terminal center RMS/σ.

Toy cells show the passing suffix /24. Ring cells read `status f<fails outside transit> arrivals d<departures>`. The
"mean min eig" column averages the worst-mode ratio over the 3 native tasks. It is the binding native metric.

## Stage 1: one knob at a time (5 target tasks; ring added for the candidates)

| arm | 5-task passes | native checks cov+acc /30 | mean HQ | mean min eig | mean worst center | grid100 | rotated100 | staggered100 | img_bars4 | two_pole | ring8-shift | ring8-multishift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3p_stock | 5/5 | 30 | 0.985 | 0.61 | 0.17 | P 5/5 0.987 0.61-1.19 c0.18 | P 5/5 0.982 0.65-1.20 c0.16 | P 5/5 0.987 0.57-1.21 c0.16 | P 10/24 | P 9/24 | P f0 +1220 d1 | F f2 +1990/x/+1640 d4 |
| gs_lr_2x | 2/5 | 0 | 0.965 | 0.08 | 0.42 | F 0/0 0.971 0.09-2.59 c0.38 | F 0/0 0.960 0.04-2.12 c0.50 | F 0/0 0.963 0.10-2.12 c0.37 | P 7/24 | P 11/24 | F f42 +600 d2 | F f42 +600/+370/+30 d2 |
| gs_coeff_0.3 | 1/5 | 0 | 0.980 | 0.19 | 0.28 | F 0/0 0.984 0.21-2.23 c0.23 | F 0/0 0.977 0.10-1.60 c0.33 | F 0/0 0.977 0.24-1.69 c0.29 | F 0/24 | P 6/24 | P f0 +610 d0 | F f1 +610/+190/+280 d1 |
| gs_dlr_0.5 | 1/5 | 0 | 0.969 | 0.15 | 0.32 | F 0/0 0.975 0.20-2.64 c0.28 | F 0/0 0.971 0.08-1.96 c0.39 | F 0/0 0.962 0.17-1.84 c0.28 | P 16/24 | F 0/24 | P f0 +470 d0 | F f1 +470/+390/+210 d1 |
| gs_guard_3 | 1/5 | 0 | 0.972 | 0.13 | 0.32 | F 0/0 0.973 0.15-1.91 c0.27 | F 0/0 0.971 0.14-1.81 c0.41 | F 0/0 0.971 0.10-2.09 c0.28 | P 15/24 | F 0/24 | P f0 +510 d0 | P f0 +510/+320/+180 d0 |
| gs_dlr_0.25 | 1/5 | 0 | 0.966 | 0.10 | 0.32 | F 0/0 0.970 0.16-3.05 c0.28 | F 0/0 0.965 0.04-2.49 c0.40 | F 0/0 0.962 0.10-2.08 c0.28 | P 13/24 | F 0/24 | – | – |
| gs_a2_1 | 1/5 | 0 | 0.971 | 0.16 | 0.33 | F 0/0 0.973 0.15-1.91 c0.27 | F 0/0 0.971 0.14-1.81 c0.41 | F 0/0 0.970 0.20-1.78 c0.32 | P 6/24 | F 0/24 | – | – |
| gs_prior_4 | 1/5 | 0 | 0.964 | 0.08 | 0.47 | F 0/0 0.968 0.06-2.40 c0.37 | F 0/0 0.970 0.11-2.03 c0.46 | F 0/0 0.953 0.06-2.19 c0.56 | F 2/24 | P 7/24 | – | – |
| gs_b1_0.5 | 1/5 | 0 | 0.973 | 0.09 | 0.66 | F 0/0 0.983 0.14-1.76 c0.80 | F 0/0 0.966 0.05-1.70 c0.62 | F 0/0 0.971 0.08-1.74 c0.55 | P 6/24 | F 0/24 | – | – |
| gs_coeff_0.03 | 0/5 | 1 | 0.982 | 0.32 | 0.30 | F 0/0 0.983 0.28-1.67 c0.29 | F 0/0 0.975 0.26-1.72 c0.36 | F 1/0 0.988 0.41-1.67 c0.26 | F 0/24 | F 0/24 | F f70 +690 d6 | F f70 +690/+100/+60 d6 |
| gs_b2_0.99 | 0/5 | 0 | 0.968 | 0.14 | 0.26 | F 0/0 0.974 0.17-2.00 c0.24 | F 0/0 0.963 0.08-1.93 c0.32 | F 0/0 0.967 0.16-1.95 c0.24 | F 0/24 | F 0/24 | – | – |
| gs_lr_0.5x | 0/5 | 0 | 0.970 | 0.17 | 0.27 | F 0/0 0.974 0.17-2.35 c0.23 | F 0/0 0.966 0.17-1.98 c0.29 | F 0/0 0.971 0.17-1.79 c0.27 | F 0/24 | F 0/24 | – | – |
| gs_batch_4096 | 0/5 | 0 | 0.975 | 0.15 | 0.31 | F 0/0 0.979 0.16-2.26 c0.26 | F 0/0 0.972 0.12-1.99 c0.33 | F 0/0 0.973 0.17-2.10 c0.33 | F 3/24 | F 0/24 | – | – |
| gs_coeff_0.1 | 0/5 | 0 | 0.983 | 0.25 | 0.31 | F 0/0 0.986 0.24-1.95 c0.25 | F 0/0 0.978 0.19-1.70 c0.36 | F 0/0 0.985 0.31-1.94 c0.32 | F 0/24 | F 1/24 | F f5 +570 d1 | F f9 +570/+600/+380 d4 |
| k3p_simple | 0/5 | 0 | 0.971 | 0.16 | 0.33 | F 0/0 0.973 0.15-1.91 c0.27 | F 0/0 0.971 0.14-1.81 c0.41 | F 0/0 0.970 0.20-1.78 c0.32 | F 3/24 | F 0/24 | P f0 +430 d0 | P f0 +430/+190/+20 d0 |
| gs_a2_0.25 | 0/5 | 0 | 0.971 | 0.16 | 0.33 | F 0/0 0.973 0.15-1.91 c0.27 | F 0/0 0.971 0.14-1.81 c0.41 | F 0/0 0.970 0.20-1.78 c0.32 | F 3/24 | F 0/24 | – | – |
| gs_guard_10 | 0/5 | 0 | 0.971 | 0.16 | 0.33 | F 0/0 0.973 0.15-1.91 c0.27 | F 0/0 0.971 0.14-1.81 c0.41 | F 0/0 0.970 0.20-1.78 c0.32 | F 3/24 | F 0/24 | – | – |
| gs_kappa_2 | 0/5 | 0 | 0.972 | 0.14 | 0.33 | F 0/0 0.978 0.16-2.20 c0.26 | F 0/0 0.969 0.14-2.01 c0.43 | F 0/0 0.971 0.11-2.00 c0.30 | F 3/24 | F 0/24 | – | – |
| gs_batch_1024 | 0/5 | 0 | 0.969 | 0.18 | 0.37 | F 0/0 0.975 0.22-2.31 c0.31 | F 0/0 0.965 0.10-1.84 c0.42 | F 0/0 0.967 0.20-2.66 c0.37 | F 3/24 | F 0/24 | – | – |
| gs_dlr_2 | 0/5 | 0 | 0.972 | 0.14 | 0.37 | F 0/0 0.977 0.15-2.25 c0.29 | F 0/0 0.967 0.11-2.14 c0.48 | F 0/0 0.971 0.17-1.84 c0.35 | F 0/24 | F 1/24 | – | – |
| gs_coeff_3 | 0/5 | 0 | 0.953 | 0.10 | 0.38 | F 0/0 0.961 0.18-2.94 c0.31 | F 0/0 0.950 0.05-1.84 c0.49 | F 0/0 0.947 0.07-2.77 c0.33 | F 0/24 | F 0/24 | – | – |
| gs_kappa_0.5 | 0/5 | 0 | 0.971 | 0.08 | 0.38 | F 0/0 0.971 0.11-2.20 c0.31 | F 0/0 0.970 0.05-1.80 c0.47 | F 0/0 0.973 0.07-2.01 c0.35 | F 3/24 | F 0/24 | – | – |
| gs_b2_0.9999 | 0/5 | 0 | 0.968 | 0.12 | 0.38 | F 0/0 0.973 0.14-2.20 c0.32 | F 0/0 0.968 0.08-1.86 c0.45 | F 0/0 0.964 0.13-2.74 c0.38 | F 0/24 | F 0/24 | – | – |
| gs_prior_1 | 0/5 | 0 | 0.953 | 0.14 | 0.49 | F 0/0 0.922 0.04-3.76 c0.79 | F 0/0 0.966 0.18-1.38 c0.36 | F 0/0 0.970 0.22-1.90 c0.31 | F 0/24 | F 0/24 | – | – |

Stage 1 was extended along the one monotone native trend: `reg_coeff` .1 and .03, and `d_lr_mult` .25.

## Stage 2: small factorial of the helpful knobs

The helpful stage-1 values were:
- `reg_coeff` .3: two_pole, native HQ/eig.
- `lr` ×2: two_pole and bars4, but ring f42.
- `d_lr_mult` .5: bars4 16/24.
- `d_guard_ratio` 3: bars4 15/24, ring clean.

| arm | 5-task passes | native checks cov+acc /30 | mean HQ | mean min eig | mean worst center | grid100 | rotated100 | staggered100 | img_bars4 | two_pole | ring8-shift | ring8-multishift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3p_stock | 5/5 | 30 | 0.985 | 0.61 | 0.17 | P 5/5 0.987 0.61-1.19 c0.18 | P 5/5 0.982 0.65-1.20 c0.16 | P 5/5 0.987 0.57-1.21 c0.16 | P 10/24 | P 9/24 | P f0 +1220 d1 | F f2 +1990/x/+1640 d4 |
| gs2_c03_lr2_d05_p1 | 2/5 | 0 | 0.979 | 0.23 | 0.34 | F 0/0 0.986 0.24-2.03 c0.28 | F 0/0 0.974 0.25-1.80 c0.41 | F 0/0 0.978 0.21-1.82 c0.32 | P 20/24 | P 6/24 | P f0 +830 d0 | F f2 +830/+280/+150 d2 |
| gs2_c03_lr2_d05_g3 | 2/5 | 0 | 0.976 | 0.17 | 0.37 | F 0/0 0.980 0.14-2.62 c0.33 | F 0/0 0.971 0.16-1.96 c0.43 | F 0/0 0.977 0.20-1.77 c0.36 | P 19/24 | P 13/24 | F f1 +740 d1 | F f3 +740/+230/+300 d2 |
| gs2_c03_lr2_d05 | 2/5 | 0 | 0.972 | 0.18 | 0.38 | F 0/0 0.980 0.14-2.62 c0.33 | F 0/0 0.969 0.15-2.02 c0.42 | F 0/0 0.966 0.23-1.96 c0.39 | P 19/24 | P 13/24 | P f0 +940 d0 | P f0 +940/+310/+110 d0 |
| gs2_lr2_d05 | 2/5 | 0 | 0.967 | 0.12 | 0.41 | F 0/0 0.974 0.19-2.66 c0.38 | F 0/0 0.965 0.13-1.82 c0.48 | F 0/0 0.962 0.05-1.93 c0.38 | P 21/24 | P 7/24 | F f3 +510 d3 | F f3 +510/+130/+20 d3 |
| gs2_c01_lr2_d05 | 1/5 | 0 | 0.979 | 0.17 | 0.37 | F 0/0 0.985 0.20-1.84 c0.32 | F 0/0 0.974 0.11-1.61 c0.44 | F 0/0 0.979 0.21-1.78 c0.34 | F 0/24 | P 14/24 | F f67 +210 d1 | F f70 +210/+170/+90 d2 |
| gs2_c03_lr2 | 1/5 | 0 | 0.972 | 0.18 | 0.40 | F 0/0 0.980 0.17-2.10 c0.36 | F 0/0 0.965 0.17-1.99 c0.45 | F 0/0 0.972 0.19-1.98 c0.40 | F 0/24 | P 15/24 | – | – |
| gs2_c03_d05_g3 | 0/5 | 0 | 0.977 | 0.21 | 0.31 | F 0/0 0.981 0.25-1.87 c0.26 | F 0/0 0.974 0.14-1.78 c0.37 | F 0/0 0.977 0.22-2.12 c0.29 | F 1/24 | F 1/24 | – | – |
| gs2_c03_d05 | 0/5 | 0 | 0.978 | 0.19 | 0.31 | F 0/0 0.981 0.25-1.87 c0.26 | F 0/0 0.974 0.14-1.78 c0.37 | F 0/0 0.977 0.19-1.76 c0.30 | F 0/24 | F 1/24 | F f11 +610 d2 | F f27 +610/+360/+280 d4 |
| k3p_simple | 0/5 | 0 | 0.971 | 0.16 | 0.33 | F 0/0 0.973 0.15-1.91 c0.27 | F 0/0 0.971 0.14-1.81 c0.41 | F 0/0 0.970 0.20-1.78 c0.32 | F 3/24 | F 0/24 | P f0 +430 d0 | P f0 +430/+190/+20 d0 |

## Stage 3: full 26-task suite

Three arms ran the full suite:
- `gs2_c03_lr2_d05`: the only stage-2 combo that passes 2/5 with a clean ring.
- `gs_guard_3`: a ring-clean single knob.
- `gs_dlr_0.5`: this was queued for stage 3 by mistake. Its multishift has f1, as the stage-1 table shows, and it
  also loses ae_gan_hold.

| arm | passes | native | transfer /19 | ring8-shift | ring8-multishift |
|---|---|---|---|---|---|
| k3p_simple | 14/26 | 0/3 | 12 | P f0 +430 d0 | P f0 +430/+190/+20 d0 |
| k3p_stock | 18/26 | 3/3 | 14 | P f0 +1220 d1 | F f2 +1990/x/+1640 d4 |
| gs2_c03_lr2_d05 | 17/26 | 0/3 | 15 | P f0 +940 d0 | P f0 +940/+310/+110 d0 |
| gs_guard_3 | 14/26 | 0/3 | 12 | P f0 +510 d0 | P f0 +510/+320/+180 d0 |
| gs_dlr_0.5 | 12/26 | 0/3 | 11 | P f0 +470 d0 | F f1 +470/+390/+210 d1 |

| task | k3p_simple | k3p_stock | gs2_c03_lr2_d05 | gs_guard_3 | gs_dlr_0.5 |
|---|---|---|---|---|---|
| native-grid100 | F 99/0.973 | P 100/0.987 | F 100/0.980 | F 99/0.973 | F 99/0.975 |
| native-rotated100 | F 100/0.971 | P 100/0.982 | F 100/0.969 | F 100/0.971 | F 100/0.971 |
| native-staggered100 | F 100/0.970 | P 100/0.987 | F 100/0.966 | F 100/0.971 | F 100/0.962 |
| hold-mode_hold | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 |
| ring8-multishift | P f0 +430/+190/+20 d0 | F f2 +1990/x/+1640 d4 | P f0 +940/+310/+110 d0 | P f0 +510/+320/+180 d0 | F f1 +470/+390/+210 d1 |
| shift-mode_hold | F 0/81 | F 0/81 | F 0/81 | F 0/81 | F 0/81 |
| ring8-shift | P f0 +430 d0 | P f0 +1220 d1 | P f0 +940 d0 | P f0 +510 d0 | P f0 +470 d0 |
| toy-two_pole | F 0/24 | P 9/24 | P 13/24 | F 0/24 | F 0/24 |
| toy-trajectory | P 22/24 | P 23/24 | P 22/24 | P 22/24 | P 21/24 |
| toy-residual_student | P 8/24 | P 20/24 | P 18/24 | P 8/24 | P 21/24 |
| toy-unipolar | P 18/24 | P 19/24 | P 22/24 | P 18/24 | P 19/24 |
| toy-ae_gan_hold | P 22/24 | P 22/24 | P 24/24 | P 22/24 | F 1/24 |
| toy-cover_leftover | P 11/24 | P 13/24 | P 17/24 | P 11/24 | P 11/24 |
| toy-unused_token_hold | P 10/24 | P 11/24 | P 20/24 | P 10/24 | P 12/24 |
| toy-mid_scale_identity | P 16/24 | P 17/24 | P 21/24 | P 16/24 | P 17/24 |
| toy-mode_hold | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 |
| toy-vector_two_broad | P 21/24 | P 23/24 | P 21/24 | P 21/24 | P 21/24 |
| toy-vector_unequal_mass | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 |
| toy-vector_unequal_width | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 |
| toy-vector_anisotropic | P 5/24 | P 6/24 | F 0/24 | P 5/24 | F 0/24 |
| toy-vector_overlap | P 9/24 | P 11/24 | P 6/24 | F 0/24 | F 0/24 |
| toy-vector_spiral | P 24/24 | P 21/24 | P 22/24 | P 24/24 | P 23/24 |
| toy-img_stripes2 | P 5/24 | P 22/24 | P 8/24 | P 17/24 | P 18/24 |
| toy-img_bars4 | F 3/24 | P 10/24 | P 19/24 | P 15/24 | P 16/24 |
| toy-img_blobs4 | F 0/24 | F 0/24 | P 19/24 | F 0/24 | P 19/24 |
| toy-img_intensity2 | F 0/24 | F 0/24 | P 9/24 | F 0/24 | F 2/24 |

## Leaderboard (26 tasks)

| # | arm | passes | native | transfer /19 | ring8-shift | ring8-multishift |
|---|---|---|---|---|---|---|
| 1 | k3p_stock (reference) | **18** | **3/3** | 14 | P f0 +1220 | F f2, misses shift 2 |
| 2 | **gs2_c03_lr2_d05** | 17 | 0/3 | **15** | P f0 +940 | **P f0 +940/+310/+110** |
| 3 | k3p_simple (reference) | 14 | 0/3 | 12 | P f0 +430 | P f0 +430/+190/+20 |
| 3 | gs_guard_3 | 14 | 0/3 | 12 | P f0 +510 | P f0 +510/+320/+180 |
| 5 | gs_dlr_0.5 | 12 | 0/3 | 11 | P f0 +470 | F f1 |

Transfer gaps of 1–2 passes are within init noise (README, Audit caveats). The winner's +3 over k3p_simple is
4 toy gains and 1 loss, so it is beyond that noise. Its +1 over stock on transfer is not.

## What moves within-mode covariance, and why

The binding native failure is the **worst mode's covariance shape**. Across the three native tasks:
- k3p_simple's worst-mode eigenvalue ratio is .14–.20. The gate needs ≥ .4 on every mode; stock gets .57–.65.
- Center RMS comes second: .27–.41σ against a limit of .2σ.
- HQ sits just above .97. It is not the blocker.

**The penalty coefficient is the only knob that moves the worst-mode shape monotonically.** Mean min eig, with mean
HQ in parentheses:

| `reg_coeff` | mean min eig | mean HQ |
|---|---|---|
| 3 | .10 | .953 |
| 1 | .16 | .971 |
| .3 | .19 | .980 |
| .1 | .25 | .983 |
| .03 | .32 | .982 |

At .03, staggered100's worst ratio is .41, and it passes 1/5 coverage checks. That is the only native coverage check
passed in the whole grid.
- **Why.** R1 plus the fake cap (in RMS units) bounds ‖∇D‖. That makes the critic Lipschitz-smooth over roughly the
  mode width, so D cannot see how a mode is shaped inside that scale. G's per-mode ellipse is then only weakly
  constrained, and one or two modes end up flattened.
  - A weaker penalty lets D resolve within-mode shape.
  - The same weakness costs ring stability: `reg_coeff` .1 gives ring f5/f9, and .03 gives f70/f70 with prehold
    59/120. It also fails bars4.
  - The coefficient that fixes native shape is the one that breaks tracking.

**This is a critic-resolution bias, not optimizer variance.** Knobs that only reduce step noise don't help:
- **LR ×.5**: .17. Not better than base (.16).
- **Batch**: 4096 gives .15 and 1024 gives .18.
- **AMSGrad β2**: .99 gives .14 and .9999 gives .12.
- **LR decay.** In the earlier suite, k3p_nonoise (decay, no noise) reached only .01–.04. Annealing freezes the bias
  in harder. Instance noise fixes shape because it convolves real and fake at the noise scale, so D can only match
  mode shape at that scale.

**The other knobs make shape worse or don't touch it:**
- `reg_kappa` .5 → .08 and 2 → .14. κ = 1 is the sweet spot.
- `lr` ×2 → .08.
- `prior_lr_mult` 1 → .14 (grid100 falls to HQ .922). 4 → .08.
- `d_lr_mult` .25 / .5 / 2 → .10 / .15 / .14.
- β1 .5 (G/D momentum): the worst center error, .66σ, because the mode centers orbit.
- A2 rate .25 and 1: bit-identical on all 3 native tasks. The rate cap is never binding there. It changes only
  bars4 (rate 1 → 6/24).
- Guard ratio 10: identical to base. The guard never clips on native (0 clips), and ratio 3 changes only staggered100.

**What moves the two toys** (both flip in the winner):
- **two_pole** responds to a weaker penalty (`reg_coeff` .3 → 6/24) and a faster G/prior (`lr` ×2 → 11/24,
  `prior_lr_mult` 4 → 7/24).
- **img_bars4** responds to a slower critic relative to G:
  - `d_lr_mult` .5 → 16/24
  - `d_lr_mult` .25 → 13/24
  - guard ratio 3 → 15/24
  - `lr` ×2 with `d_lr_mult` .5 → 21/24
- **Together.** A faster G/prior with an unchanged critic LR and a 0.3× penalty flips both. It also flips img_blobs4
  and img_intensity2 on the full suite.

**Ring stability is not separable across knobs, so the winner sits in a narrow basin.** Each of its stage-2
neighbours fails the ring:
- `+g3` (guard ratio 3): f1/f3.
- `p1` (prior ×1): multishift f2.
- `c01` (`reg_coeff` .1): f67/f70.
- `c03_d05` (`lr` ×1): f11/f27.
- `lr2_d05` (`reg_coeff` 1): f3/f3.

The single knobs are just as fragile: `reg_coeff` .3 alone gives multishift f1, and `d_lr_mult` .5 alone gives f1.
Treat the winner's clean ring as a point result, not a robust margin.

## Recommendations

1. **To promote a constant-LR k3p_simple config, use `gs2_c03_lr2_d05`** (`reg_coeff` .3, `lr` .0085,
   `d_lr_mult` .5): 17/26, clean ring, 15/19 transfer. Keep k3p_stock as the default, because it still owns native.
2. **Don't tune further for native within this formulation.** No existing knob gets the worst-mode ratio to .4 while
   the ring stays clean. The trade-off runs through `reg_coeff`.
3. **Next experiments (no seed runs):**
   - **Winner + output noise only** (.029, no input anneal). This tests whether a small fixed noise restores native
     shape without costing tracking. README next step #1 is the same test on k3p_simple.
   - **Split the penalty weights.** Native wants a weak R1 on reals, and the ring needs the fake cap (ab_r1only
     loses the ring). A separate cap weight would need a new `Recipe` field, so it is out of scope here. Test
     `reg_coeff` ≈ .03 on R1 with the cap kept at .3–1.
   - **Pin transfer init first** (README next step 4) before trusting the +1 transfer gap over stock.
   - **Check the winner's lost task** (vector_anisotropic 5 → 0) after pinning init.

## Reproduce and tail

```
./run_grid.sh cuda:1 5 native-grid100,native-rotated100,native-staggered100,toy-img_bars4,toy-two_pole gs_coeff_0.3 ...
./run_grid.sh cuda:0 6 all gs2_c03_lr2_d05
python grid_summarize.py                 # stage table (all grid arms)
python grid_summarize.py --full --arms k3p_simple,k3p_stock,gs2_c03_lr2_d05
tail -f logs_grid/<arm>.log              # one line per task
tail -f logs_grid/<arm>/<task>.log       # one line per eval
```

Runs are in `runs_grid/<arm>/<task>/`. Only `result.json`, `config.json` and the native gate JSONs are committed.
k3p_simple and k3p_stock were not rerun, because the suite is deterministic; their results are read from `runs/`.
