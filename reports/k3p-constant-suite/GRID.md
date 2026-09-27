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

---

# Follow-up: split R1 from the fake cap, and a symmetric real-side cap (noise-free)

## Question

Stage 1 found that a weaker `reg_coeff` improves native within-mode shape but breaks the ring. `reg_coeff` scales
two terms together: R1 on reals and the one-sided RMS cap on fakes. Is the trade-off R1 against the cap?

Martyn also objects to R1 because it pulls the critic's slope at the reals to zero even when D is right. So a
second arm replaces R1 with the fakes' own cap applied to reals, making one symmetric cap. Instance noise stays at
0 in every arm.

## Result

**No R1/cap split and no real-side cap beats `gs2_c03_lr2_d05` (17/26), and none fixes native.**
- **The ring needs R1 at the reals.** Every arm that weakens R1 or replaces it with a cap loses the ring.
- **The only ring-clean new config is `gsX_r1_c0.03`** (R1 1, cap .03, k3p_simple LRs). On the full suite it
  scores **12/26**, below k3p_simple.
- **It is a single point.** Each of its one-knob neighbours fails the ring. So do the winner's R1/cap neighbours.
- **Native stays 0/3.** The worst-mode covariance ratio stays at .14–.23 in every arm, well short of the gate's .4.

## Code (opt-in; defaults bit-identical)

`particlegan.Recipe` gets three fields, each registered in `training._ADDED_RECIPE_FIELDS`:
- **`reg_real_weight`** (1.0) scales phase A's reals term against `reg_coeff`. The fake cap stays at `reg_coeff`.
  - Effective R1 = `reg_coeff × reg_real_weight`.
  - Effective cap = `reg_coeff`.
- **`reg_real_mode`** (`"r1"`) picks phase A's reals term. `"cap"` replaces R1 with
  `relu(‖∇D(real)‖/√d − reg_real_kappa)²`, the same op as the fake side.
- **`reg_real_kappa`** (None, meaning `reg_kappa`).

Checks:
- **Kernel:** `GradientPenalty(real_weight, real_mode, real_kappa)`. At the defaults the phase-A ops are unchanged.
- **Diagnostics:** with `collect_stats`, phase A also reports `real_rms_mean/max` and `fake_rms_mean/max`
  (detached, diagnostics only). The ring and GANTrainer logs print them as `gR=mean/max`.
- **Unit tests** (`tests/test_reg_real_weight.py`):
  - the default equals the old penalty exactly;
  - weight 0 leaves only the fake cap;
  - cap mode equals the symmetric cap;
  - a cap threshold above every real slope equals weight 0;
  - validation and checkpoint upgrade work.
  - 138 tests pass across the recipe, K3P, penalty, training and API test files.
- **Suite bit-identity:** gs2_c03_lr2_d05 was rerun after the change on ring8-shift, toy-two_pole and toy-img_bars4
  (one task per route). All three have identical final-parameter sha256 and metrics.
- **Diagnostic reruns match:** the `runs_grid_diag/` ring reruns of k3p_simple and gs2_c03_lr2_d05 reproduce every
  ring metric.
- **Consistency check:** `gsC_cap_s` (the recipe path) reproduces the suite's harness-level `ab_caponly` exactly on
  all 7 tasks.

Harness:
- `suite_adapter.K3P_FIELDS` carries the three fields to every route.
- `grid_gradstats.py` builds the gradient table below.

Two arms had to rerun their 3 native tasks with `--overwrite`: gsR_r0.03_c0.3 and gsRs_r0.1_c1.
- The toy100 runner hashes the package source. The `reg_real_mode` edit landed mid-run, so its gate recorded
  "source changed during training".
- Training itself had finished and was unaffected. The rerun is the same config, not a seed variant.

## Arms

"R1 x, cap y" means an effective R1 weight of x and a fake-cap coefficient of y.

- **gs2 base** (`gs2_c03_lr2_d05`: lr .0085, `d_lr_mult` .5; R1 .3, cap .3)
  - `gsR_r0.1_c0.3`, `gsR_r0.03_c0.3`, `gsR_r0_c0.3`: R1 swept down.
  - `gsR_r0.1_c1`, `gsR_r0.03_c1`: cap raised to 1.
  - `gsX_w_r0.3_c0.1`: reverse split.
- **k3p_simple base** (lr .00425, `d_lr_mult` 1; R1 1, cap 1)
  - `gsRs_r0.1_c1`, `gsRs_r0.03_c1`, `gsRs_r0_c1`: R1 down, cap held.
  - `gsX_r1_c0.1`, `gsX_r1_c0.03`: reverse split. R1 is held at 1 and the cap goes down, which the data pointed to.
- **Symmetric cap**
  - `gsC_cap_w` (gs2 base, cap .3 on both sides).
  - `gsC_cap_s` (k3p_simple base, cap 1 on both).
  - Real threshold on the k3p_simple base, the better native base: `gsC_cap_s_k0.5` and `gsC_cap_s_k2`, plus
    `gsC_cap_s_k0.25` to follow the trend.
  - `gsC_cap_w_k0.5`: the same threshold on the gs2 base.

## Stage 1: 5 target tasks and both ring tasks

| arm | 5-task passes | native checks cov+acc /30 | mean HQ | mean min eig | mean worst center | grid100 | rotated100 | staggered100 | img_bars4 | two_pole | ring8-shift | ring8-multishift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsX_w_r0.3_c0.1 | 2/5 | 0 | 0.975 | 0.19 | 0.36 | F 0/0 0.978 0.16-1.95 c0.31 | F 0/0 0.972 0.14-1.96 c0.40 | F 0/0 0.977 0.27-1.95 c0.36 | P 19/24 | P 13/24 | F f0 x d0 | F f0 x/x/x d0 |
| gs2_c03_lr2_d05 | 2/5 | 0 | 0.972 | 0.18 | 0.38 | F 0/0 0.980 0.14-2.62 c0.33 | F 0/0 0.969 0.15-2.02 c0.42 | F 0/0 0.966 0.23-1.96 c0.39 | P 19/24 | P 13/24 | P f0 +940 d0 | P f0 +940/+310/+110 d0 |
| gsR_r0.1_c0.3 | 1/5 | 0 | 0.979 | 0.16 | 0.38 | F 0/0 0.983 0.16-2.36 c0.34 | F 0/0 0.975 0.13-1.86 c0.43 | F 0/0 0.979 0.20-1.83 c0.37 | F 0/24 | P 14/24 | F f34 +550 d3 | F f37 +550/+270/+200 d5 |
| gsR_r0.1_c1 | 1/5 | 0 | 0.979 | 0.18 | 0.39 | F 0/0 0.980 0.19-2.67 c0.36 | F 0/0 0.977 0.17-1.83 c0.45 | F 0/0 0.980 0.17-1.74 c0.36 | F 0/24 | P 14/24 | F f72 +480 d2 | F f75 +480/+520/+70 d4 |
| gsR_r0.03_c0.3 | 1/5 | 0 | 0.979 | 0.20 | 0.39 | F 0/0 0.981 0.20-1.91 c0.42 | F 0/0 0.975 0.19-1.81 c0.39 | F 0/0 0.981 0.22-1.98 c0.36 | F 0/24 | P 7/24 | F f207 +210 d13 | F f236 +210/+160/+80 d27 |
| gsC_cap_s | 1/5 | 0 | 0.981 | 0.23 | 0.41 | F 0/0 0.977 0.22-1.85 c0.41 | F 0/0 0.981 0.21-1.80 c0.41 | F 0/0 0.985 0.25-2.17 c0.39 | F 0/24 | P 13/24 | F f188 +240 d23 | F f231 +240/+160/+110 d42 |
| gsR_r0.03_c1 | 1/5 | 0 | 0.981 | 0.19 | 0.42 | F 0/0 0.985 0.11-1.86 c0.39 | F 0/0 0.979 0.23-1.96 c0.46 | F 0/0 0.981 0.22-2.15 c0.41 | F 0/24 | P 8/24 | F f144 +1010 d21 | F f149 +1010/+1310/+530 d25 |
| gsC_cap_s_k0.5 | 1/5 | 0 | 0.980 | 0.18 | 0.46 | F 0/0 0.981 0.19-1.87 c0.42 | F 0/0 0.975 0.12-2.21 c0.52 | F 0/0 0.985 0.24-1.70 c0.45 | F 0/24 | P 9/24 | F f33 +590 d4 | F f75 +590/+310/+100 d13 |
| gsC_cap_w_k0.5 | 1/5 | 0 | 0.985 | 0.14 | 0.47 | F 0/0 0.985 0.14-2.17 c0.42 | F 0/0 0.983 0.14-2.06 c0.50 | F 0/0 0.986 0.14-1.91 c0.48 | F 1/24 | P 13/24 | F f70 +790 d11 | F f76 +790/+290/+390 d14 |
| gsC_cap_w | 1/5 | 0 | 0.956 | 0.16 | 0.59 | F 0/0 0.980 0.20-1.92 c0.43 | F 0/0 0.904 0.06-1.69 c0.93 | F 0/0 0.985 0.22-1.90 c0.40 | F 1/24 | P 7/24 | F f166 +1090 d17 | F f214 +1090/+60/+480 d26 |
| gsR_r0_c0.3 | 1/5 | 0 | 0.943 | 0.20 | 0.62 | F 0/0 0.949 0.21-2.15 c0.63 | F 0/0 0.892 0.17-3.34 c0.85 | F 0/0 0.989 0.22-2.27 c0.39 | F 1/24 | P 6/24 | F f169 +1170 d4 | F f211 +1170/+240/+610 d27 |
| gsRs_r0.1_c1 | 0/5 | 0 | 0.980 | 0.19 | 0.31 | F 0/0 0.984 0.23-2.12 c0.27 | F 0/0 0.975 0.14-2.07 c0.36 | F 0/0 0.982 0.19-2.08 c0.30 | F 0/24 | F 1/24 | F f55 +90 d1 | F f55 +90/+400/+60 d1 |
| gsX_r1_c0.03 | 0/5 | 0 | 0.969 | 0.16 | 0.31 | F 0/0 0.975 0.24-2.06 c0.25 | F 0/0 0.967 0.11-1.61 c0.41 | F 0/0 0.966 0.12-1.78 c0.27 | F 2/24 | F 0/24 | P f0 +490 d0 | P f0 +490/+220/+20 d0 |
| gsX_r1_c0.1 | 0/5 | 0 | 0.974 | 0.17 | 0.31 | F 0/0 0.979 0.23-1.89 c0.26 | F 0/0 0.971 0.08-1.65 c0.37 | F 0/0 0.972 0.19-2.08 c0.31 | F 3/24 | F 0/24 | F f26 +370 d1 | F f29 +370/x/+610 d4 |
| k3p_simple | 0/5 | 0 | 0.971 | 0.16 | 0.33 | F 0/0 0.973 0.15-1.91 c0.27 | F 0/0 0.971 0.14-1.81 c0.41 | F 0/0 0.970 0.20-1.78 c0.32 | F 3/24 | F 0/24 | P f0 +430 d0 | P f0 +430/+190/+20 d0 |
| gsRs_r0.03_c1 | 0/5 | 0 | 0.982 | 0.22 | 0.36 | F 0/0 0.984 0.24-2.12 c0.32 | F 0/0 0.979 0.17-1.61 c0.37 | F 0/0 0.984 0.24-1.81 c0.38 | F 0/24 | F 0/24 | F f217 +320 d13 | F f311 +320/+220/+60 d31 |
| gsC_cap_s_k2 | 0/5 | 0 | 0.985 | 0.20 | 0.44 | F 0/0 0.988 0.21-1.74 c0.44 | F 0/0 0.982 0.22-1.85 c0.46 | F 0/0 0.984 0.17-1.91 c0.41 | F 0/24 | F 0/24 | F f163 +530 d13 | F f193 +530/+120/+100 d28 |
| gsRs_r0_c1 | 0/5 | 0 | 0.982 | 0.22 | 0.45 | F 0/0 0.982 0.28-1.77 c0.38 | F 0/0 0.983 0.16-1.54 c0.49 | F 0/0 0.980 0.22-1.78 c0.47 | F 0/24 | F 0/24 | F f232 +540 d25 | F f277 +540/+390/+310 d42 |
| gsC_cap_s_k0.25 | 0/5 | 0 | 0.979 | 0.15 | 0.45 | F 0/0 0.978 0.21-2.20 c0.44 | F 0/0 0.975 0.15-1.80 c0.46 | F 0/0 0.984 0.10-1.69 c0.45 | F 0/24 | F 0/24 | F f45 +320 d1 | F f45 +320/+220/+60 d1 |

### R1 = 0 vs real-side cap vs R1 split (direct comparison)

| family | arm | mean min eig (gate ≥.4) | mean HQ | bars4 | two_pole | ring8-shift f | multishift f / arrivals |
|---|---|---|---|---|---|---|---|
| reference | k3p_simple (R1 1, cap 1) | .16 | .971 | 3 | 0 | 0 | 0 / +430 +190 +20 |
| reference | gs2_c03_lr2_d05 (R1 .3, cap .3) | .18 | .972 | **19** | **13** | 0 | 0 / +940 +310 +110 |
| R1 = 0 | gsRs_r0_c1 (cap 1) | .22 | .982 | 0 | 0 | 232 | 277 |
| R1 = 0 | gsR_r0_c0.3 (cap .3) | .20 | .943 | 1 | 6 | 169 | 211 |
| real cap | gsC_cap_s (κr 1, coeff 1) | **.23** | .981 | 0 | 13 | 188 | 231 |
| real cap | gsC_cap_s_k0.5 | .18 | .980 | 0 | 9 | 33 | 75 |
| real cap | gsC_cap_s_k0.25 | .15 | .979 | 0 | 0 | 45 | 45 |
| real cap | gsC_cap_s_k2 | .20 | **.985** | 0 | 0 | 163 | 193 |
| real cap | gsC_cap_w (κr 1, coeff .3) | .16 | .956 | 1 | 7 | 166 | 214 |
| real cap | gsC_cap_w_k0.5 | .14 | .985 | 1 | 13 | 70 | 76 |
| R1 split | gsRs_r0.03_c1 | .22 | .982 | 0 | 0 | 217 | 311 |
| R1 split | gsR_r0.1_c0.3 | .16 | .979 | 0 | 14 | 34 | 37 |
| R1 split | gsR_r0.1_c1 | .18 | .979 | 0 | 14 | 72 | 75 |
| reverse split | **gsX_r1_c0.03** | .16 | .969 | 2 | 0 | **0** | **0** / +490 +220 +20 |
| reverse split | gsX_r1_c0.1 | .17 | .974 | 3 | 0 | 26 | 29 |
| reverse split | gsX_w_r0.3_c0.1 | .19 | .975 | 19 | 13 | 0 (never arrives) | 0 (never arrives) |

### Is the result a region or a point? Ring one-knob neighbours

| base | arm | R1 / cap change | ring8-shift | ring8-multishift |
|---|---|---|---|---|
| k3p_simple | gsX_r1_c0.03 | R1 1, cap .03 (candidate) | P f0 +490 d0 (prehold 120/120) | P f0 +490/+220/+20 d0 (prehold 120/120) |
| k3p_simple | gsX_r1_c0.1 | cap .1 | F f26 +370 d1 (prehold 94/120) | F f29 +370/x/+610 d4 (prehold 94/120) |
| k3p_simple | gsX_r1_c0.01 | cap .01 | F f120 x d0 (prehold 0/120) | F f122 x/+220/+50 d2 (prehold 0/120) |
| k3p_simple | gsX_r0.3_c0.03 | R1 .3 | F f18 +400 d3 (prehold 102/120) | F f19 +400/+50/+30 d4 (prehold 102/120) |
| k3p_simple | gsX_r3_c0.03 | R1 3 | F f44 +300 d14 (prehold 85/120) | F f44 +300/+40/+50 d14 (prehold 85/120) |
| gs2_c03_lr2_d05 | gs2_c03_lr2_d05 | R1 .3, cap .3 (winner) | P f0 +940 d0 (prehold 120/120) | P f0 +940/+310/+110 d0 (prehold 120/120) |
| gs2_c03_lr2_d05 | gsR_r0.1_c0.3 | R1 .1 | F f34 +550 d3 (prehold 87/120) | F f37 +550/+270/+200 d5 (prehold 87/120) |
| gs2_c03_lr2_d05 | gsX_w_r1_c0.3 | R1 1 | P f0 +620 d0 (prehold 120/120) | F f16 +620/+480/+140 d1 (prehold 120/120) |
| gs2_c03_lr2_d05 | gsX_w_r0.3_c0.1 | cap .1 | F f0 x d0 (prehold 120/120) | F f0 x/x/x d0 (prehold 120/120) |
| gs2_c03_lr2_d05 | gsX_w_r0.3_c1 | cap 1 | F f2 +870 d2 (prehold 118/120) | F f2 +870/+460/+20 d2 (prehold 118/120) |

**Both ring-clean configs are isolated points.**
- Around gsX_r1_c0.03: cap .1 and cap .01 fail, and so do R1 .3 and R1 3.
- Around gs2_c03_lr2_d05: R1 .1, R1 1, cap .1 and cap 1 each fail at least one ring task. The first grid's
  LR/penalty neighbours of the winner failed too.

### Real-side input gradient on the ring (ring8-multishift, logged every 10 updates)

RMS = ‖∇ₓD‖/√d per sample. "Prehold" covers updates 1210–2400, before any shift.

| arm | prehold real RMS mean (median) | prehold real RMS max (median / max) | run real RMS max (max) | run fake RMS max (max) |
|---|---|---|---|---|
| k3p_simple | 0.016 | 0.30 / 0.98 | 1.82 | 2.58 |
| gs2_c03_lr2_d05 | 0.018 | 0.36 / 0.86 | 4.02 | 4.34 |
| gsX_r1_c0.03 | 0.021 | 0.40 / 1.38 | 3.43 | 6.08 |
| gsX_r1_c0.1 | 0.029 | 0.47 / 2.44 | 3.35 | 6.45 |
| gsX_w_r0.3_c0.1 | 0.027 | 0.58 / 1.96 | 4.17 | 7.32 |
| gsR_r0.1_c1 | 0.409 | 1.12 / 1.81 | 2.94 | 3.20 |
| gsRs_r0.03_c1 | 0.772 | 1.29 / 1.52 | 2.70 | 2.70 |
| gsRs_r0_c1 | 0.951 | 1.46 / 1.78 | 3.62 | 5.95 |
| gsR_r0_c0.3 | 1.089 | 2.51 / 3.57 | 6.23 | 6.12 |
| gsC_cap_s_k0.25 | 0.051 | 0.43 / 1.19 | 2.22 | 2.89 |
| gsC_cap_s_k0.5 | 0.344 | 0.64 / 1.12 | 2.26 | 2.52 |
| gsC_cap_s | 0.863 | 1.21 / 1.51 | 3.33 | 3.06 |
| gsC_cap_s_k2 | 0.895 | 1.94 / 3.30 | 4.46 | 4.11 |
| gsC_cap_w_k0.5 | 0.387 | 1.16 / 1.84 | 3.53 | 4.25 |
| gsC_cap_w | 0.927 | 1.38 / 1.78 | 4.77 | 5.29 |

With R1 at weight ≥ .3, the median real-side slope is about .02. The critic is flat at the data, which is the
equilibrium R1 enforces.

Without R1, or with R1 at .1 or lower, the typical real-side slope sits near the cap threshold (.4–1.1) or above it.

**The real-side cap does not stop spikes.**
- Run-max real RMS with a cap is 2.2–4.8, against 1.8 for k3p_simple.
- Its prehold max is 1.1–3.3, against 0.98.
- Lowering the threshold moves the cap toward R1. At κr = 0 the cap is exactly R1 in RMS units. That is why
  κr = .5 and .25 are the best cap arms on the ring (f33–75). Even κr = .25 has 45 fails.

## Stage 2: full 26-task suite

Only `gsX_r1_c0.03` keeps 0 ring fails and arrives after every shift. `gsX_w_r0.3_c0.1` also has 0 fails, but its
ring never re-forms 8 modes after the shift.

| arm | passes | native | transfer /19 | ring8-shift | ring8-multishift |
|---|---|---|---|---|---|
| k3p_simple | 14/26 | 0/3 | 12 | P f0 +430 d0 | P f0 +430/+190/+20 d0 |
| k3p_stock | 18/26 | 3/3 | 14 | P f0 +1220 d1 | F f2 +1990/x/+1640 d4 |
| gs2_c03_lr2_d05 | 17/26 | 0/3 | 15 | P f0 +940 d0 | P f0 +940/+310/+110 d0 |
| gsX_r1_c0.03 | 12/26 | 0/3 | 10 | P f0 +490 d0 | P f0 +490/+220/+20 d0 |

| task | k3p_simple | k3p_stock | gs2_c03_lr2_d05 | gsX_r1_c0.03 |
|---|---|---|---|---|
| native-grid100 | F 99/0.973 | P 100/0.987 | F 100/0.980 | F 99/0.975 |
| native-rotated100 | F 100/0.971 | P 100/0.982 | F 100/0.969 | F 100/0.967 |
| native-staggered100 | F 100/0.970 | P 100/0.987 | F 100/0.966 | F 100/0.966 |
| hold-mode_hold | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 |
| ring8-multishift | P f0 +430/+190/+20 d0 | F f2 +1990/x/+1640 d4 | P f0 +940/+310/+110 d0 | P f0 +490/+220/+20 d0 |
| shift-mode_hold | F 0/81 | F 0/81 | F 0/81 | F 0/81 |
| ring8-shift | P f0 +430 d0 | P f0 +1220 d1 | P f0 +940 d0 | P f0 +490 d0 |
| toy-two_pole | F 0/24 | P 9/24 | P 13/24 | F 0/24 |
| toy-trajectory | P 22/24 | P 23/24 | P 22/24 | P 22/24 |
| toy-residual_student | P 8/24 | P 20/24 | P 18/24 | P 8/24 |
| toy-unipolar | P 18/24 | P 19/24 | P 22/24 | P 18/24 |
| toy-ae_gan_hold | P 22/24 | P 22/24 | P 24/24 | F 0/24 |
| toy-cover_leftover | P 11/24 | P 13/24 | P 17/24 | P 10/24 |
| toy-unused_token_hold | P 10/24 | P 11/24 | P 20/24 | P 10/24 |
| toy-mid_scale_identity | P 16/24 | P 17/24 | P 21/24 | P 16/24 |
| toy-mode_hold | F 0/24 | F 0/24 | F 0/24 | F 0/24 |
| toy-vector_two_broad | P 21/24 | P 23/24 | P 21/24 | P 23/24 |
| toy-vector_unequal_mass | F 0/24 | F 0/24 | F 0/24 | F 0/24 |
| toy-vector_unequal_width | F 0/24 | F 0/24 | F 0/24 | F 0/24 |
| toy-vector_anisotropic | P 5/24 | P 6/24 | F 0/24 | P 5/24 |
| toy-vector_overlap | P 9/24 | P 11/24 | P 6/24 | P 16/24 |
| toy-vector_spiral | P 24/24 | P 21/24 | P 22/24 | P 23/24 |
| toy-img_stripes2 | P 5/24 | P 22/24 | P 8/24 | F 0/24 |
| toy-img_bars4 | F 3/24 | P 10/24 | P 19/24 | F 2/24 |
| toy-img_blobs4 | F 0/24 | F 0/24 | P 19/24 | F 0/24 |
| toy-img_intensity2 | F 0/24 | F 0/24 | P 9/24 | F 0/24 |

`gsX_r1_c0.03` scores 12/26:
- It reproduces k3p_simple's ring exactly in shape: +490/+220/+20 against +430/+190/+20.
- It loses 2 transfer tasks: ae_gan_hold (22 → 0) and img_stripes2 (5 → 0).
- It gains none of the target tasks.

## Leaderboard (26 tasks)

| # | arm | passes | native | transfer /19 | ring8-shift | ring8-multishift |
|---|---|---|---|---|---|---|
| 1 | k3p_stock | **18** | **3/3** | 14 | P f0 +1220 | F f2, misses shift 2 |
| 2 | **gs2_c03_lr2_d05** (still the constant-LR winner) | 17 | 0/3 | **15** | P f0 +940 | **P f0 +940/+310/+110** |
| 3 | k3p_simple | 14 | 0/3 | 12 | P f0 +430 | P f0 +430/+190/+20 |
| 4 | gsX_r1_c0.03 (R1 1, cap .03) | 12 | 0/3 | 10 | P f0 +490 | P f0 +490/+220/+20 |

No other split or cap arm passes both ring tasks.

## Explanation

**1. R1 is what holds the ring, not the fake cap.**

With the cap held or raised, every drop in R1 costs the ring:

| R1 | ring fails (shift / multishift) |
|---|---|
| .1 | 34–72 / 37–75 |
| .03 | 144–217 / 149–311 |
| 0 | 169–232 / 211–277 |

The cap alone does not hold it either: ab_r1only (R1 without the cap) also loses the ring, and the reverse split's
cap .01 has 120 fails. So both terms are load-bearing. The ring tolerates only narrow ratios between them:
- R1 1 with cap 1 or .03 is clean (k3p_simple, gsX_r1_c0.03).
- On the fast-G base, R1 .3 with cap .3 is clean.

**2. Why R1 is not penalising correct gradients here.**

At the RpGAN equilibrium p_g = p_data, the optimal critic is flat on the data. A slope at the reals is the
disequilibrium signal itself.
- R1 damps the critic's rotation around that equilibrium. This is the Dirac-GAN / "Which training methods for GANs
  do actually converge?" result.
- A one-sided cap leaves any slope below κ free. So the critic can hold a slope of size ~κ at the reals, which is
  the observed median of .4–1.1.
- G is then pushed across the modes and never settles. At κ ≥ 1 the prehold is 0–6/120, with many departures.
- The ring is a small-support problem where this rotation is visible. The gradient table shows the flat-at-reals
  signature (median slope about .02) in every ring-clean arm and in none of the failing ones.

**3. The native covariance bias is not an R1 effect.**

Removing R1 entirely raises mean min eig only from .16 to .22. `reg_coeff` .03 (both terms weak) reaches .32, and
the gate needs ≥ .4.
- Weakening only the fake cap does nothing: gsX_r1_c0.03 gives .16.
- The within-mode shape error comes from a critic smoothed by both terms together. The ring needs the same
  smoothing, so no split of the two weights resolves it.

**4. Why the winner's toy gains survive only with R1 ≥ .3.**
- two_pole mostly responds to a weak or absent R1: 13–14/24 in the gs2-base R1 .1 arms and the symmetric caps.
- img_bars4 needs the gs2 base's fast G/prior together with R1 at .3. It scores 19/24 in gs2_c03_lr2_d05 and
  gsX_w_r0.3_c0.1, and 0–3 everywhere else.

## Recommendations

1. **Keep `gs2_c03_lr2_d05` as the constant-LR, noise-free configuration** (17/26, clean ring). Keep k3p_stock as the
   default.
2. **Keep R1 on reals.** The symmetric real cap fails the ring on both bases and at every threshold tested, and it
   does not reduce slope spikes at the reals. Merge `reg_real_weight` / `reg_real_mode` / `reg_real_kappa` only as
   documented ablation switches (defaults bit-identical), or drop them from the PR.
3. **Treat native within-mode covariance as unsolved without noise.** Penalty re-weighting cannot reach it. The
   remaining noise-free levers change what the critic can resolve, not how hard it is penalised:
   - a critic with more within-mode resolution (Fourier features / width on the native hosts);
   - generator capacity (the worst modes are flattened ellipses).
   Both are architecture changes, outside "existing knobs".
4. **Every ring-clean config found so far is a point, not a region.** Before promoting gs2_c03_lr2_d05, check that
   it holds on another ring geometry (another mode count or radius), a new task rather than a seed repeat.
