# Simple 3-term critic on the KA2 shift protocol

> **Initialization change (merge of origin/develop c720645e, #194).** Every result in rounds
> 1-5 below used the old random PyTorch init; those runs and logs now live in
> `runs_oldinit/` and `logs_oldinit/` (`toy100/runs_oldinit/`, `toy100/logs_oldinit/`).
> Score them with `summarize.py --runs-dir runs_oldinit`. `rerun_all.sh` reruns every ring
> arm (48, incl. the K3P refs from `.claude/worktrees/k3p-develop` and `ka2_stock_ref`) under
> the package default `batch_feature_zero` into `runs/`. Every run carries an `init_receipt`
> (`init_receipt.py`): the mode, sha256 of the initial G/D/prior params, and whether they differ
> from the `initialization=None` draw with the same seed. On the ring, G, D (and the EMA/anchor
> critic) and the particle prior all change. On toy100 only D's weights change: the
> `affine_square_v1` identity generator, zero biases and the model policy's uniform particle
> square are kept by the public path. **Results: "Rerun with develop's QR initialization" below.**

**Question.** Can a plain critic, with no instance noise, constant learning
rates and no adaptive controller, hold the ring-8 target and follow it through
the (1,0) shift as well as KA2 does? The requirement was: "keep D from spiking
on real, a path from fake, and the slope of any gradient max 1". Each part of
that sentence becomes one penalty term. The arms below test which form of each
term works, and what happens when one is removed.

**Answer.** Yes, with one caveat. `secant_r1_b2` (wgan + R1 + secant path +
cap, with critic Adam beta2 0.9) is the only simple arm that passes. It holds
better than the KA2 constant-LR reference and has a lower max grad-norm, but it
arrives at the shifted target 630 updates after the shift versus KA2's 120.

**Round 3 update.** Turning off A2 latent damping (`sec_nodamp`) is the best
single change: prehold 97/120, arrival 230, 21 departures, 50 fails outside
transit, max grad-norm 2.64. It is the new best arm and has no adaptive element
on either side. Combining it with `center(1)` (`combo_a`) does not beat it.
Secant t = 1 collapses late, and lazy_k 4 never passes. See "Round 3".

**Transfer update.** `sec_nodamp` does not transfer. On the 100-Gaussian gate
it covers all 100 modes on all three problems with max grad-norm at most 2.2
after update 1000, but fails on within-mode shape (HQ 0.89 to 0.95, gate 0.97).
On sparse-UCD it ends at 58/64 modes against 62/64 for the constant-LR
champion. The simple critic carries its stability across, not its accuracy.
See "Transfer".

## Rerun with develop's QR initialization

**What changed.** origin/develop c720645e (#194) was merged. `Recipe.initialization` now
defaults to `batch_feature_zero`: `make_optimizers` initializes fresh G/D/E once and syncs
the EMA critic, and `make_prior` applies R2 init to learnable particle tables. Nothing else
changed. All 48 ring arms were rerun with their recorded worker and flags (seed 0, no noise,
constant LRs, no controller for simple arms; reference rows keep their setups; K3P refs from
`k3p-develop@c720645e`). All 8 toy100 arms were rerun on the full gate (7000 updates, seed 1234).
Old results are archived in `runs_oldinit/` and `logs_oldinit/` (toy100 likewise). Full
comparison: `INIT_RERUN.md` (regenerate with `compare_init.py`). New ring board: `leaderboard.md`
and `LEADERBOARD.md`. Toy100: `toy100/LEADERBOARD.md`.

**How the init was verified.** Every `result.json` carries an `init_receipt` (`init_receipt.py`).
- Ring: `initialization=batch_feature_zero` and `external_init_hook=None`. G, D and the prior all
  differ from the `initialization=None` draw, with no tensor left at its old value. The start
  hashes match a fresh public-recipe build, and the EMA critic equals D at the start
  (sec_anchor, sec_guard_anchor, k3p_*, ka2). An independent audit rebuilt the start weights:
  the new code gives the receipt hashes, and the pre-merge code (aa92ddf4, K3P 0ff9a7af)
  gives the recorded old hashes. For 32 arms the only declaration difference is
  `recipe.initialization`; the other arms differ only in keys added later at their no-op defaults.
- Toy100: only D's weights change. G is the identity generator, D's biases stay zero, and
  `prior.z` keeps the benchmark's own uniform draw. With `run_arm.py --init none`, ref_stock
  reproduces the archived result exactly.

**New ring leaderboard (top).** fails = fails outside transit. `ref` rows are not ranked.

| # | arm | prehold | arrival | post-arrival | departures | fails | max grad | final HQ |
|---|---|---|---|---|---|---|---|---|
| ref | k3p_stock_ref | 120/120 | 1220 | 99/99 | 1 | 0 | 3.55 | 0.954 |
| ref | k3p_constant | 95/120 | 360 | 185/185 | 1 | 25 | 4.90 | 0.993 |
| 1 | B_cap3 | 80/120 | 250 | 195/196 | 11 | **41** | 5.43 | 0.995 |
| 2 | lr_c0.5_g0.5 | 86/120 | 240 | 181/197 | 15 | 50 | 2.40 | 0.993 |
| 3 | sec_nodamp | 97/120 | 160 | 176/205 | 17 | 52 | 2.73 | 0.982 |
| 4 | lr_c0.5_g2 | 99/120 | 400 | 148/181 | 9 | 54 | 3.48 | 0.008 |
| ref | ka2_stock_ref | 78/120 | 390 | 168/182 | 12 | 56 | n/a | 0.724 |
| 5 | lr_c1_g0.5 | 86/120 | 90 | 185/212 | 25 | 61 | 2.35 | 0.991 |
| 6 | lr_c0.25_g0.25 | 75/120 | 100 | 192/211 | 6 | 64 | 2.80 | 0.139 |
| 7 | sec_drift | 83/120 | 250 | 168/196 | 24 | 65 | 2.82 | 0.995 |
| 8 | combo_a | 90/120 | 200 | 165/201 | 19 | 66 | 3.35 | 0.966 |

Final HQ is one observation; 0.008 and 0.139 are a single bad check, not a collapse.

**Old -> new for key arms (ring).**

| arm | rank | prehold | arrival | departures | fails | max grad |
|---|---|---|---|---|---|---|
| B_cap3 | 1 -> 1 | 82 -> 80 | 140 -> 250 | 10 -> 11 | 38 -> 41 | 5.94 -> 5.43 |
| sec_nodamp | 3 -> 3 | 97 -> 97 | 230 -> 160 | 21 -> 17 | 50 -> 52 | 2.64 -> 2.73 |
| lr_c0.5_g1 | 2 -> 9 | 96 -> 81 | 260 -> 380 | 17 -> 17 | 42 -> 69 | 2.83 -> 2.66 |
| rp_center | 4 -> 21 | 66 -> 53 | 300 -> 170 | 5 -> 6 | 54 -> 118 | 4.88 -> 5.63 |
| secant_r1_b2 | 12 -> 19 | 83 -> 48 | 630 -> 280 | 33 -> 27 | 77 -> 106 | – |
| lr_c0.5_g0.5 | 15 -> 2 | 97 -> 86 | 560 -> 240 | 29 -> 15 | 81 -> 50 | – |
| sec_drift | 22 -> 7 | 79 -> 83 | 150 -> 250 | 35 -> 24 | 116 -> 65 | – |
| k3p_constant | ref | 88 -> 95 | 360 -> 360 | 5 -> 1 | 50 -> 25 | 5.81 -> 4.90 |
| k3p_stock_ref | ref | 120 -> 120 | 1960 -> 1220 | 3 -> 1 | 0 -> 0 | 7.45 -> 3.55 |
| ka2_stock_ref | ref | 61 -> 78 | 120 -> 390 | 10 -> 12 | 142 -> 56 | n/a |

13 of 45 ranked arms moved 7 or more places; the median arm moved 2. Every arm in the failure
group (full*, no_cap, no_real, wgangp_ref, wgan_margin, c3_rate, lr_c4_g4, int_r1* except
int_r1_b2, secant_r1, sec_lazy4, no_path) still fails after the shift. The exception is
wgan_huber, which now arrives (at 1020) but has 234 fails. The new init lowers their blow-ups
(full max grad 685 -> 95) without rescuing them.

**Toy100 gate (new init).** No arm passes. The shipped `ref_stock` drops from PASS 3/3 to 0/3.

| # | arm | pass | modes (g/r/s) | final HQ (g/r/s) | ΔHQ sum vs old | max grad |
|---|---|---|---|---|---|---|
| 1 | rp_center | 0/3 | 100/98/100 | 0.980/0.948/0.972 | +0.001 | 2.93 |
| 2 | c3_r1w | 0/3 | 100/100/100 | 0.967/0.922/0.972 | +0.104 | 4.96 |
| 3 | sec_nodamp | 0/3 | 100/100/100 | 0.948/0.907/0.944 | +0.057 | 2.26 |
| 4 | secant_r1_b2 | 0/3 | 100/100/100 | 0.925/0.900/0.948 | −0.023 | 2.14 |
| 5 | c3_capinterp | 0/3 | 100/97/100 | 0.934/0.811/0.945 | −0.067 | 4.29 |
| 6 | sec_nodamp_lazy4 | 0/3 | 98/100/95 | 0.851/0.849/0.843 | −0.051 | 2.34 |
| ref | ref_stock | **0/3** (was 3/3) | 100/71/40 | 0.947/0.615/0.449 | **−0.949** | 9.62 |
| ref | ref_matched | 0/3 | 66/88/9 | 0.853/0.841/0.148 | +0.514 | 105.67 |

Controls: with `--init none` (old init, merged code), ref_stock gets PASS 3/3 again, identical to
the archive, so the init alone causes the regression. With `--init hook` (develop's own toy100
path, which also re-spaces the prior), ref_stock gets 2/3 and grid collapses to 44 modes.

**Do the round 1-5 conclusions still hold?**
- **Yes: which components are needed.** R1 + secant + cap-all is still required; every arm
  without one of them still fails. Lazy regularization, the rate penalty and secant t = 1 stay
  closed. B_cap3 stays ring #1 and sec_nodamp stays #3.
- **Yes: toy100 fails on within-mode shape, not coverage.** The simple arms still cover 95–100
  modes and fail on covariance (min–max ratio 0.13–2.9). Their HQ barely moves with the init.
- **Changed: rp_center as the next base (Round 5 recommendation 1).** Its ring #4 came from a
  lucky init: fails 54 -> 118 and prehold 66 -> 53. It is still toy100 #1, but the ring no longer
  supports promoting it.
- **Changed: the fine ranking in the middle of the ring board.** Changing only the start weights
  moved fails by 30 to 65 on similar arms. In ranks 2 to 25, gaps under about 30 fails are noise.
  This invalidates the order among lr-grid cells, B_cap2, c3_capinterp and sec_t1. It also weakens
  the round 3 claims that rest on small fails gaps, such as "combo_a does not beat sec_nodamp".
- **Weakened: "keep R1(1)".** R1 0.1 (c3_r1w) is now ring #10 (fails 79 -> 70, 29 behind
  B_cap3, inside the noise band) and toy100 #2 (ΔHQ +0.104). It still fails covariance, so the
  flat-D explanation stays refuted, but R1 0.1 is no longer clearly harmful on the ring.
- **Changed: the references.** k3p_constant (25 fails) now beats every simple arm; before it
  tied sec_nodamp at 50. On toy100 the shipped ref_stock no longer passes under the new D init,
  so it cannot serve as a passing baseline until that is resolved.
- **Changed: the headline answer above.** secant_r1_b2 is now ring #19 (106 fails), so it is no
  longer the arm to cite. B_cap3 is.

**Recommendations (no seed runs, no spectral norm).**
1. Keep **B_cap3** as the ring lead. It is the only simple arm in the top 3 under both inits.
2. Next, graft K3P's anchor/guard onto B_cap3; they were only tested on the older c=1 base. That
   is the most direct attempt at k3p_constant's 25 fails.
3. Treat ring arms as tied unless their fails outside transit differ by 30 or more. To separate
   the top three, change the test rather than the seed, for example a second shift or a longer
   post-shift window.
4. On toy100, the next tests are unchanged: the margin secant, and a lower constant particle LR
   (`prior_lr_mult` 1). Run them on B_cap3's critic settings as well as rp_center's, since the
   ring no longer favours rp_center.
5. Find out why ref_stock loses the toy100 gate under `batch_feature_zero` on
   `constraints_simple_regularization.json` (develop reports 22/22 on a different setup). This
   is a package-level question, not a simple-critic one.

## Formulation

The critic loss is `L_D = base + real + path + cap`, with base = wgan
`E D(f) - E D(r)`. The generator loss is RpGAN in every arm.

| user's phrase | term | equation (best arm) |
|---|---|---|
| keep D from spiking on real | R1 at real | `lam_r * E_r ‖∇D(r)‖²`, lam_r = 1 |
| a path from fake | secant path | `lam_p * E_f relu(t·‖r_nn(f) − f‖ − (D(r_nn(f)) − D(f)))²`, lam_p = 10, t = 0.5, where r_nn(f) is the nearest real in the batch |
| slope of any gradient max 1 | cap-all | `lam_c * E_{x ∈ real ∪ fake ∪ path} relu(‖∇D(x)‖ − 1)²`, lam_c = 10 |

The secant asks D to rise from each fake toward its nearest real by at least t
per unit distance. It needs no input gradient, sets no slope at the real
point, and is zero once fakes sit on reals.

Optimizer constant: critic Adam betas (0, 0.9) instead of the recipe's
(0, 0.999). This is fixed for the whole run. It is not a schedule. See finding 1.

Other forms tested: `drift` (`E D(r)²`), a `lower` slope floor
`E relu(t − ‖∇D(x̂)‖)²` on interpolates with u in [lo, hi], `two_sided`
(WGAN-GP), and the `hinge` and `rplogistic` base losses.

## Protocol (deltas from `reports/ka2-default-candidate/constant-lr-api`)

These match the reference: ring of 8, public `GANTrainer`/`get_recipe()`
construction, 20k particles, latent 2, batch 2048, ring arch 3x96, Fourier
scale 3, G/D LR 0.00425, prior LR 0.0085, target shift (1,0) after update 2400,
4600 updates, observe every 10 updates with 4096 samples, pass = 8 modes and
HQ >= 0.90, seed 0. The G, D, prior, ring means and data stream are the same
as the stock KA2 run (sha256 of the rebuilt initial state matches its
`raw/initial.json`).

These differ from the reference:
- **No instance noise.** Critic input noise and generator output noise are 0
  from step 0, and the worker never adds any. The prior's `sigma_rel` is 0.
- **Constant LRs.** Floors are 1, there is no `NetworkLRTransition` and no
  decay. Every group's `lr` is checked against its initial value on every
  update (`constant_lr_verified_updates = 4600` in every arm).
- **No KA2/K3P controller on the critic.** The critic's recipe optimizer is
  built with `ema_critic=None` and `d_guard_ratio=0`. The KA2 penalty is not
  built (`reg_anchor_weight=0`), so no Kalman surprise, anchor or memory
  reaches training.
- **Torch 2.14.0+cu130** on an RTX A6000. The archived reference used 2.13.0.
  `ka2_stock_ref` reruns the stock worker on 2.14 and matches the archive at
  all 460 observations (step, modes and HQ).

## Leaderboard

Columns:
- prehold: passing checks from 1210 to 2400.
- arrival: updates from the shift to the first passing check.
- post-arrival: passing checks from arrival to the end.
- departures, longest fail streak and fails outside transit: all exclude the
  shift transit.
- final suffix: the uninterrupted passing run that ends at 4600.

Rank order: arrived first, then fewest fails outside transit, fewest
departures, earliest arrival. `ref` rows are not simple arms (noise on, KA2
controller) and are not ranked.

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

The `k3p_*` reference rows come from a separate agent's `k3p_worker.py` runs in
this directory. That agent's LR grid (`lr_c*`) and K3P guard/anchor arms
(`sec_guard*`, `sec_anchor`) are not part of this commit and are left out of
this table; `summarize.py` lists them when their runs are present.

Critic diagnostics are medians over all 460 observations of the probe (real,
fake and path points). The full table is in `LEADERBOARD.md` and
`python3 summarize.py --diag`.
- `sec_nodamp`: g(real) 0.06, median gmax 1.13, 14 of 460 observations above
  2. Mean HQ is 0.865 from 1210 to 2400 and 0.897 from 3600 to 4600.
- `secant_r1_b2`: g(real) 0.10, median gmax 1.35, 17 of 460 observations above
  2. Mean HQ is 0.837 from 1210 to 2400 and 0.854 from 3600 to 4600.
- `int_r1_b2`: g(real) 0.16, median gmax 1.28, 0 of 460 above 2, max gmax 1.92.
- Every arm on the default critic beta2 (0.999) reaches a max gmax of 25 or more.

## What the ablations show

1. **The single-step spikes come from the critic optimizer, not from a missing
   penalty.** With betas (0, 0.999), a sudden gradient after the quiet hold
   gives a step of up to about lr/sqrt(1−beta2) = 31.6×lr per parameter. With
   beta2 0.9 the bound is 3.2×lr. That one constant takes `int_r1` from max
   grad-norm 139 to 1.92 (`int_r1_b2`), and `secant_r1` from 395 to 2.82
   (`secant_r1_b2`, where max |D(real)| also falls from 161 to 3.2). A soft
   cap cannot stop a jump that happens within one step, so this constant is
   what makes "slope max 1" hold.
2. **"A path from fake" is required, and it must not act at the real point.**
   - Without a path term (`no_path`), D inverts at the shift and never
     recovers.
   - The round-1 slope floor of 1 over u in [0, 1] (`full`, `full_r1`) holds
     g(real) at 0.64 to 0.99, so real modes are never peaks of D. None of these
     arms ever passes.
   - Moving to an interior window (`int_r1`: t 0.3, u in [0.1, 0.9]) drops
     g(real) to 0.16. That arm still flickers (63 departures with beta2 0.9),
     because the slope floor is nonzero at equilibrium.
   - The secant form asks for a rise in D, not a slope. It gives the best hold.
3. **"Don't spike on real" means R1.** Drift at 0.1 did nothing in round 1.
   Once the path stops pulling g(real) up, R1(1) keeps g(real) near 0.1 and
   |D(real)| at 3 or less. Removing the real term (`no_real`) never passes.
4. **Cap-all(10) is required.** Without it, the one-sided path diverges
   (`no_cap`: gmax 4.6e6). With beta2 0.9 it holds median gmax near 1.3.
5. **Base loss: wgan.**
   - Hinge sharpens faster before the shift but crashes at it.
   - RpGAN, the KA2 base, is too weak under the cap: its best HQ is 0.21. That
     plausibly explains why KA2 needs its controller.
6. **Remaining weaknesses of `secant_r1_b2`:**
   - Its arrival after the shift is slow: 630 updates versus 120 for KA2.
   - It has more short departures than KA2: 33 versus 10.
   - D's level is unanchored. The mean of D(real) drifts from about 0.5 to 1.1
     after the shift, because wgan + R1 is unchanged by adding a constant to D.

## Audit caveats

- **The zero-noise evidence is the code, not a per-update log.** Only the KA2
  reference logs noise values. For the simple arms, the evidence is the worker
  code and the recipe settings (all 0).
- **A2 latent-row damping was on in every arm through round 2, including the reference.**
  `latent_damping_max_rate=0.5` (`K3PGeneratorAdam`, `particlegan/k3p.py`)
  scales each particle row's step by 0.5 to 1, based on the cosine with that
  row's last gradient. It also sets that group's beta1 to 0.5 inside the step.
  - The prior's `lr` stays constant, but its effective step depends on
    gradient memory.
  - This is a confound shared by every arm, so it does not separate arms.
    Round 3 ran `--latent-damping 0` (`sec_nodamp`); it improved every metric.
- **Some claims come from rebuilds and replays, not stored run data.**
  - The runs do not store their own init hashes. Identical initialization is
    shown by rebuilding with the worker's code and matching the stock KA2
    hashes.
  - A 100-step replay of `full`, `no_real` and `secant_r1_b2` matched the
    stored `metrics.jsonl` bit for bit. So the round-2 flag defaults reproduce
    round 1.
- **`int_r1` changes two things relative to `full_r1`:** the target (1 to 0.3)
  and the u-window. Finding 2 mixes the two effects.
- **The run was chosen after seeing results.** `secant_r1_b2` was picked after
  the `a` batch of round 2.
- **The recorded code commit is not the code that ran.** Runs record commit
  `fa511ce0`, but `worker.py` and `summarize.py` were untracked at run time.
  This commit adds them unchanged except for the `summarize.py` ref-row
  ranking fix.
- **One seed per benchmark:** ring-8 shift (seed 0), 100 Gaussians (1234),
  sparse-UCD (1). Transfer ran only `secant_r1_b2`/`sec_nodamp` and lazy_k 4.

## Round 3: single changes and combinations on the ring

Each round-3a arm makes exactly one change to `secant_r1_b2`; round 3b combines
the changes that helped. All ran the full 4600 updates on seed 0 with constant
LRs verified on every update. Two new worker flags, both off by default (a
100-update replay of `secant_r1_b2` matches its stored metrics at all 10
observations):
- `--lam-center`: adds `lam * (E_r D(r))^2`. This is an explicit penalty.
  Subtracting the batch mean of D(real) would change no gradient, because
  wgan, R1, the secant difference, the cap and the RpGAN G loss are all
  unchanged when a constant is added to D.
- `--lazy-k`: every term except the base is applied only on critic steps with
  `step % k == 0`, weighted ×k (the `lazy_k` rule in
  `particlegan/grad_regularizers.py`).

| arm | change | prehold | arrival | departures | fails outside transit | max grad-norm | verdict |
|---|---|---|---|---|---|---|---|
| secant_r1_b2 | base | 83/120 | 630 | 33 | 77 | 2.82 | |
| **sec_nodamp** | A2 latent damping 0.5 -> 0 | **97/120** | 230 | **21** | **50** | 2.64 | best; improves all five metrics |
| sec_center | + center(1) | 78/120 | 180 | 27 | 71 | 3.13 | small help; pins mean D(real) at 0.00 to 0.01 |
| sec_t1 | secant t 0.5 -> 1 | 76/120 | 190 | 24 | 68 | 5.43 | collapses late (final HQ 0.21) |
| sec_drift | + drift(1e-3) | 79/120 | 150 | 35 | 116 | 3.48 | worse flicker |
| sec_lazy4 | lazy_k = 4 | 0/120 | none | 0 | 120 | 4.20 | never passes |
| combo_a | nodamp + center(1) | 94/120 | 170 | 20 | 68 | 2.54 | worse than nodamp alone |
| combo_b | nodamp + center(1) + t1 | 58/120 | 180 | 26 | 107 | 3.24 | worse than any single change |

What each result says:
1. **A2 off is the real improvement.** With latent damping off, nothing
   adaptive remains on either side beyond plain Adam, and every metric
   improves. Median g(real) falls to 0.06 and median gmax to 1.13.
2. **Arrival does not rank these arms.** Every non-lazy arm, the base
   included, reaches 8 modes within 10 to 100 updates of the shift, and even the
   near-null drift(1e-3) arrives at 150. The base's 630 is most likely the
   outlier. Rank by fails outside transit and departures instead.
3. **Centering is not needed once A2 is off.** It helped the base (fails 77 to
   71), but `combo_a` holds worse after arrival than `sec_nodamp`: mean HQ over
   3600 to 4600 drops from 0.897 to 0.754 and fails rise from 50 to 68.
4. **Secant t = 1 over-constrains.** With the rise target equal to the slope
   cap, D goes flat (g(real) about 0.02), kicks at about 4450 and falls to 2
   modes (0 modes on the EMA copy). It also hurts in `combo_b`.
5. **lazy_k = 4 fails, on every benchmark.** Three of every four critic steps
   are pure wgan with no secant or cap. With critic Adam beta2 0.9, the ×4 on
   the fourth step is largely normalized away, so the terms that shape D lose
   most of their average pressure. StyleGAN2 gets away with lazy R1 because
   its R1 is mild; here the regularizers carry the formulation. On the ring it
   never passes (mean HQ 0.41). On 100 Gaussians (`sec_nodamp_lazy4`) first
   full coverage slips from 1000-1500 to 2500 and final HQ drops to 0.81 to
   0.92. On sparse-UCD it ends at 54/64 modes against 58/64. Its only gain is
   speed (80 against 55 steps/s on sparse-UCD).
6. **Spectral norm was not run.** It was dropped as too limiting.

## Transfer: 100 Gaussians and sparse-UCD

Neither benchmark has a passing simple arm. Details and reproduction are in
`toy100/LEADERBOARD.md` and `sparse_ucd/LEADERBOARD.md`.

**100 Gaussians** (`benchmarks.toy100`: grid100, rotated100, staggered100; full
7000-update gate, the benchmark's seed 1234, cuda:0). The adapter
`toy100/run_arm.py` swaps in the ring study's critic loss from `worker.py` and
builds the critic optimizer with `ema_critic=None`, guard 0 and betas (0, 0.9);
nothing in `particlegan/` was edited. Matched arms have input and output noise 0
and LR floors 1. Gate: 100 modes, HQ >= 0.97, mass TV <= 0.10, covariance
eigenvalue ratio in [0.40, 1.70], radial median ratio in [0.65, 1.40], sustained
over 5 terminal checks plus a 100k holdout.

| # | arm | gate | final modes (g/r/s) | final HQ (g/r/s) | first 100 modes (g/r/s) | min-max cov ratio (g / r / s) | max abs D(real) | max grad-norm (all / >=1000) | g(real) median >=1000 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | secant_r1_b2 (A2 .5) | FAIL | 100/100/100 | 0.948/0.897/0.951 | 1250/2000/500 | 0.29-1.73 / 0.29-1.89 / 0.27-1.72 | 5.44 | 5.33 / 2.16 | 0.029/0.029/0.037 |
| 2 | sec_nodamp | FAIL | 100/100/100 | 0.903/0.886/0.954 | 1000/1500/500 | 0.17-1.83 / 0.36-2.12 / 0.21-1.76 | 4.85 | 5.33 / 1.95 | 0.029/0.030/0.035 |
| 3 | sec_nodamp_lazy4 | FAIL | 98/99/100 | 0.859/0.813/0.922 | 2500/2500/2500 | 0.20-2.05 / 0.42-2.22 / 0.24-2.36 | 3.93 | 7.85 / 1.85 | 0.066/0.041/0.056 |
| ref | ref_stock (shipped: RpGAN-logistic + b_cap, noise, cosine LR decay) | PASS | 100/100/100 | 0.985/0.986/0.989 | 750/750/750 | 0.62-1.18 / 0.68-1.18 / 0.59-1.29 | 0.81 | 5.93 / 2.98 | 0.777/0.579/0.491 |
| ref | ref_matched (shipped critic, noise off, constant LRs) | FAIL | 22/92/13 | 0.285/0.856/0.187 | 250/none/250 | 0.00-5.34 / 0.05-2.20 / 0.00-3.17 | 14.33 | 122.25 / 122.25 | 1.324/0.473/0.981 |

- The simple critic keeps its stability: all 100 modes on every problem, mass
  TV 0.035 to 0.042, max grad-norm at most 2.2 after update 1000 (the 5.3 peak
  is the untrained D at update 1). The shipped critic under the same
  constraints (`ref_matched`) collapses to 22 and 13 modes with grad-norm 122.
- It fails on within-mode shape. D is almost flat at the data (median g(real)
  about 0.03, against 0.5 to 0.8 for the passing `ref_stock`), and a flat D
  cannot shape a sigma = 0.03 mode. EMA samples reach HQ 0.96 to 0.99 but their
  modes are too narrow (min cov ratio 0.08 to 0.16), so the failure is not only
  live jitter from constant LRs.
- Two plausible causes: R1(1) flattens D at every real point, and the secant
  rewards D rising toward the nearest sampled real even inside a mode.
- A2 0.5 against 0 is within noise here; `ref_stock` reproduces the archived
  CPU run exactly (HQ 0.985/0.986/0.989, first 100 modes at 750).

**Sparse-UCD** (64 modes of 3-sparse data in R^24, 8 classes, symbol head;
champion config `champion/l0p02_gw_sp0p003`, seed 1, 5000 steps, cuda:1, all
LRs constant via `lr_anneal_start: 1.0`, no instance noise). Code is on branch
[`sparse-ucd-secant`](https://github.com/255BITS/ParticleGAN/tree/sparse-ucd-secant)
(commit 358e458d, `results/sparse-secant/`). It adds `h_secant_cap` to
`lib/grad_regularizers.py`, taking the nearest real from same-class reals only.
This harness has no latent damping, so `secant_r1_b2` and `sec_nodamp` are the
same formulation and ran once. Bar: at the end, modes 64, hq >= 0.9, cond,
sym and sp@1e-2 >= 0.95. No arm reached it at any eval.

| # | arm | formulation | modes (final) | min modes, 2nd half | hq | joint | sp@1e-2 | core | w1 | max abs D(real) | max grad-norm | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | champ_ref_anneal | champion as archived (RpGAN + g_interp_cap(1), cosine LR anneal) | 63 | 54 | 0.975 | 0.986 | 0.972 | 0.74 | 0.037 | n/a | n/a | n/a |
| 1 | champ_matched | RpGAN + g_interp_cap(1), constant LRs | 62 | 30 | 0.985 | 0.997 | 0.961 | 0.41 | 0.035 | 4.95 | 1.96 | 81.3 |
| 2 | sec_nodamp | wgan + r1(1) + secant(10, t=0.5) + cap-all(10), D beta2 0.9 | 58 | 32 | 0.884 | 0.967 | 0.964 | 0.74 | 0.049 | 5.41 | 2.62 | 54.7 |
| 3 | sec_lazy4 | sec_nodamp + lazy_k 4 | 54 | 21 | 0.755 | 0.975 | 0.962 | 0.85 | 0.065 | 5.51 | 2.44 | 79.9 |
| 4 | sec_rpbase | RpGAN base + r1(1) + secant(10, t=0.5) + cap-all(10), D beta2 0.9 | 45 | 19 | 0.801 | 0.970 | 0.984 | 1.01 | 0.128 | 4.08 | 1.92 | 52.7 |

- D is already well behaved in this harness (max |D(real)| 4 to 5.5, max
  grad-norm 1.9 to 2.6 in every arm, the champion included), so the ring
  study's main gain, stopping D spikes, has nothing to fix here.
- The secant term dominates the critic loss and is never satisfied: 82% of
  fakes still violate it at the end. In 24-D with 3-sparse modes the nearest
  same-class real in the batch is often a different mode, so the secant pulls
  fakes toward the wrong mode.
- Removing the anneal hurts the champion mid-run (modes fall to 30 at step
  3500, core width 0.41 against 0.74). Every arm, champion included, drops
  right after the gate turns on at step 2000.

## Round 4: K3P references, guard/anchor, LR grid, curvature

All ring, seed 0, no noise, constant LRs. Details: [LR_GRID.md](LR_GRID.md), [CURVATURE.md](CURVATURE.md).

| Entry | Prehold | Arrival | Departures | Fails outside transit |
|---|---|---|---|---|
| ref: K3P v0.8.0 as released (noise + LR decay) | 120/120 | +1960 | 3 | 0 |
| **B_cap3**: lr_c0.5_g1 with cap c=3 | 82/120 | +140 | 10 | **38** |
| lr_c0.5_g1 (sec_nodamp, critic LR x0.5) | 96/120 | +260 | 17 | 42 |
| ref: K3P v0.8.0, noise off, constant LRs | 88/120 | +360 | 5 | 50 |
| sec_nodamp | 97/120 | +230 | 21 | 50 |
| secant_r1_b2 + K3P guard / anchor / both | 74 / 54 / 55 | +400 / +220 / +320 | 23 / 24 / 24 | 98 / 115 / 128 |
| B + peak margin (κ=7.14) | 85/120 | +240 | 13 | 118 |
| wgan + Huber profile only / + peak margin only | 0/120 | none | - | 120 (margin-only diverges) |

- **K3P's EMA anchor never engages under constant LRs** (it waits for LR annealing), so K3P's constant-LR stability comes from its penalty plus spike guard. Grafting guard/anchor onto the secant critic hurts.
- **LR grid:** critic LR drives departures and grad-norm (x2 is bad everywhere); G LR drives arrival and D magnitude. Best cell critic x0.5, G x1.
- **Curvature hypothesis rejected.** Curvature at reals correlates *positively* with departures (Spearman +0.49) and fails (+0.77); within runs it rises before a pass->fail flip. Forcing a curvature floor (peak margin) makes D cone-like and hurts. Loosening the slope cap to 3 is the only curvature-adjacent change that helps.
- Without Lipschitz terms (plain wgan + one shape penalty) nothing holds 8 modes.

## Round 5: R1 weight, cap placement, rate penalty, relativistic + centering

Base **B_cap3** (wgan + R1(1) + secant(10, t 0.5, u in [0.1, 0.9]) + cap-all(10, c=3), critic
Adam β2 0.9, A2 off, LRs critic 0.002125 / G 0.00425 / prior 0.0085). Ring, seed 0, no noise,
constant LRs, no controller/guard/anchor; one run per formulation. Code: `round5_worker.py`
(wraps `lr_grid.py` + `worker.py` read-only and reuses `curvature_worker.py`'s diagnostics),
launcher `round5.sh`, table `round5_table.py`. With the new flags off it reproduces B_cap3
bit-for-bit: all 10 metrics.jsonl rows over updates 1–100 are identical (losses, LRs, curvature).

Exact forms of the new terms:
- **cap-ends**: `lam·mean relu(‖∇D(x)‖ − c)²` over the real and fake points only (no path points
  are built). `cap-interp` (existing) caps only the u ∈ [0.1, 0.9] interpolates.
- **rate(λ)**: at critic step k, D has θ_k and a frozen copy holds θ_{k−1} (the parameters from
  before the previous critic step). On the step's own batch,
  `λ·mean_i[(D_θk(r_i) − D_θk−1(r_i))² + (D_θk(f_i) − D_θk−1(f_i))²]`, gradient only through θ_k.
  The copy is then overwritten with θ_k. The term is 0 at step 1.
- **RpGAN** in this repo (`particlegan/gan_loss.py`): critic `mean_i softplus(−(D(r_i) − D(f_i)))`
  = softplus(D(f) − D(r)) on index-paired real/fake (worker `--loss rplogistic`); generator
  `mean_i softplus(−(D(f'_i) − D(r_i)))` on fresh fakes. Every simple arm, B_cap3 included, already
  used this RpGAN generator loss (`--g-loss rpgan` default); only the critic base changes.
- **pair-center(λ)**: `λ·(mean_i (D(r_i) + D(f_i))/2)²`. RpGAN sees only differences, so this pins
  D's level.

### Ring (seed 0)

| arm | change vs B_cap3 | prehold | arrival | post-arrival | departures | longest fail streak | fails outside transit | max abs D(real) | max grad | median curv (wide) | median g(real) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **B_cap3** | — | 82/120 | +140 | 207/207 | 10 | 23 | **38** | 4.59 | 5.94 | 0.19 (0.37) | 0.084 |
| ref: k3p_constant | K3P v0.8.0, noise off | 88/120 | +360 | 167/185 | 5 | 32 | 50 | 2.63 | 5.81 | 0.14 (0.28) | 0.039 |
| rp_center | RpGAN critic + pair-center(1) | 66/120 | +300 | **191/191** | **5** | 50 | 54 | 3.21 | 4.88 | 0.09 (0.26) | 0.064 |
| c3_capinterp | cap on interpolates only | 94/120 | +250 | 164/196 | 20 | 23 | 58 | 5.44 | 5.07 | 0.39 (0.63) | 0.127 |
| c3_r1w | R1 weight 0.1 | 78/120 | +170 | 167/204 | 15 | 28 | 79 | 6.04 | 6.42 | 1.07 (1.48) | 0.260 |
| c3_capnopath | cap on real+fake only | 78/120 | +110 | 162/210 | 33 | 34 | 90 | 9.61 | 15.67 | 0.37 (0.65) | 0.155 |
| c3_rate | + rate(10) | 0/120 | none | – | – | – | 120 | 4.76 | 6.10 | 0.98 (1.58) | 0.196 |
| rp_center_min | RpGAN + pair-center(1) + cap-all(c=3) only | 8/120 | +610 | 36/160 | 32 | 41 | 236 | 3.01 | 5.44 | 15.9 (13.7) | 1.804 |

k3p_constant curvature is from the observation-only rerun in `diag/`. `summarize.py` ranks
B_cap3 #1 (unchanged), rp_center #4 and c3_capinterp #7.

**No combo was run.** No arm lowered fails outside transit below 38. rp_center halves departures
(10 → 5) and has zero post-arrival fails, but it loses prehold (82 → 66) to a single collapse at
1910 to 2400 (HQ 0.15, 3 modes at 2000). c3_capinterp raises prehold (94) but doubles departures
(20). Both are clear losses under the combo rule.

### 100 Gaussians gate (full protocol: 7000 updates, seed 1234, grid/rotated/staggered)

New arms port the ring formulation, including critic LR ×0.5 (config `d_lr_mult` 0.5), to
`toy100/run_arm.py`. Top two ring arms (rp_center, c3_capinterp) plus c3_r1w, which targets the
flat-D explanation. Matched rows are from earlier runs, not rerun. Note that sec_nodamp has cap
c=1 and critic LR ×1, so it differs from the new arms in two settings besides the tested change.

| arm | gate | final modes (g/r/s) | final HQ (g/r/s) | cov-eig-ratio min–max (g / r / s) | mass TV (max) | g(real) median ≥1000 | max grad (all / ≥1000) |
|---|---|---|---|---|---|---|---|
| rp_center | FAIL | 98/100/100 | **0.973**/0.957/0.969 | 0.18–2.24 / 0.21–1.78 / 0.13–2.02 | 0.062 | 0.009/0.030/0.018 | 4.03 / 3.15 |
| c3_capinterp | FAIL | 100/100/100 | 0.909/0.908/0.940 | 0.14–3.22 / 0.26–1.89 / 0.20–2.44 | 0.033 | 0.039/0.035/0.051 | 5.53 / 5.53 |
| c3_r1w | FAIL | 100/100/98 | 0.950/0.938/0.869 | 0.16–1.87 / 0.17–1.74 / 0.29–3.00 | 0.037 | 0.133/0.163/0.155 | 4.85 / 4.27 |
| sec_nodamp (matched) | FAIL | 100/100/100 | 0.903/0.886/0.954 | 0.17–1.83 / 0.36–2.12 / 0.21–1.76 | 0.039 | 0.029/0.030/0.035 | 5.33 / 1.95 |
| ref_stock (shipped) | PASS | 100/100/100 | 0.985/0.986/0.989 | 0.62–1.18 / 0.68–1.18 / 0.59–1.29 | – | 0.777/0.579/0.491 | 5.93 / 2.98 |
| ref_matched | FAIL | 22/92/13 | 0.285/0.856/0.187 | 0.00–5.34 / 0.05–2.20 / 0.00–3.17 | – | 1.324/0.473/0.981 | 122 / 122 |

Gate needs 100 modes, HQ ≥ 0.97 and cov ratio in [0.40, 1.70] over the five terminal checks.
No arm has a passing live evaluation (0/25 on every problem).

### Which terms matter

- **R1 weight: the flat-D explanation is refuted.** R1 0.1 raises g(real) about 5× on toy100 (0.13
  to 0.16) but HQ and covariance do not improve, and staggered loses 2 modes. rp_center has the
  *flattest* D (g(real) 0.009 to 0.03) and the best HQ. On the ring R1 0.1 doubles fails (79), lets
  |D(real)| reach 6 and ends collapsed (final HQ 0.10). Keep R1(1).
- **Cap placement: the cap is needed on both kinds of points, for different jobs.** Without path
  points (c3_capnopath), grad-norm reaches 15.7 and departures triple (33). Without real/fake points
  (c3_capinterp), prehold improves (94) but departures double (20) and |D(real)| reaches 11 on
  toy100. cap-all stays the best placement.
- **Rate penalty: harmful.** At λ 10 the critic cannot track G, and the run never passes (0/120,
  modes 0 to 4 throughout). The penalty measured D's per-step output change: RMS about 0.3 per
  point under Adam β1 0, β2 0.9 at LR 0.002. That is large relative to the gap D(r) − D(f) (about
  0.1 to 0.8). Penalizing it strongly freezes D; the within-run sharpening signal from Round 4 is
  not fixed by slowing D down.
- **Relativistic + centering: the only idea with a real gain, but not a strict one.** On the ring,
  departures fall to k3p's 5 and there are no post-arrival fails. On toy100 it gives the best final
  HQ of any simple arm (grid 0.973 passes the HQ bar), and |D(real)| stays at 1.25 versus 4.9 to 15
  for the wgan arms. It still fails covariance shape (min ratio 0.13 to 0.21) and loses 2 modes on
  grid. The minimal version (no R1, no secant) does not hold: 236 fails and curvature 16. The secant
  and R1 terms are what make RpGAN work without noise.
- **Covariance shape is the common toy100 failure.** Every simple arm, and every EMA, has some modes
  squeezed to lines (min ratio 0.05 to 0.29), whatever D's slope at the reals. This points at the
  G/particle side (step noise, per-mode width), not at the critic's real term.

### Recommendations (no seed variants, no spectral norm)

1. **Next base: rp_center** (RpGAN critic + pair-center(1) + R1(1) + secant + cap-all c=3). B_cap3
   stays ring #1 on fails, but rp_center is the only formulation that improves departures on the
   ring and HQ on toy100 together, and it keeps D smallest. Its one weakness is a single prehold
   collapse.
2. On rp_center, attack the prehold collapse with a distinct term rather than weights: for example
   the margin secant (Next experiments 1), which stops the path term inside a mode and is also aimed
   at toy100 covariance shape.
3. Test the G/particle side on toy100 with a lower constant particle LR (prior_lr_mult 1), on
   rp_center. The EMA covariance failure says the critic alone will not fix shape.
4. Closed: R1 below 1, cap on only one point set, the temporal rate penalty, RpGAN without R1/secant.

## Next experiments (distinct formulations, same constraints)

Round 5 closed item 2 (weaker real term); see its recommendations for the current order.

No seed variants and no spectral norm. Closed: secant t = 1, drift, center,
lazy regularization (fails on all three benchmarks), and the centered combos.

1. **Margin secant.** `relu(t·(‖r_nn − f‖ − m) − (D(r_nn) − D(f)))²` with m about
   3 sigma (0.09 on 100 Gaussians), so the path term stops acting inside a mode.
   Tests the secant-noise explanation for the within-mode shape failure. On
   sparse-UCD, measure the secant to the nearest same-class mode center rather
   than the nearest batch real.
2. **Weaker real term.** R1 at 0.1, or replace R1 with center(1) and keep
   cap-all. Tests whether a flat D at the data is what breaks within-mode shape
   on 100 Gaussians. Re-check the ring hold afterwards, because R1 is what kept
   |D(real)| low there.
3. **Lower constant particle/G step.** For example prior_lr_mult 1 instead of 2
   on 100 Gaussians, or a lower constant LR / D beta2 0.9 on the sparse-UCD
   champion. A fixed constant, not a schedule. Tests whether the anneal's
   stabilizing effect (EMA/live HQ gap 0.98 against 0.91 on 100 Gaussians; the
   champion's mid-run dip on sparse-UCD) can be had without one.
4. **Sparse-UCD gate onset.** Every arm crashes right after the gate turns on at
   step 2000; that transition is worth attacking directly.

## Reproduce

Run from the repo root. Each run takes about 35 to 90 s on an A6000.

```bash
# confirm the worktree's particlegan is imported
PYTHONPATH=$PWD .venv/bin/python -c "import particlegan; print(particlegan.__file__)"

bash reports/simple-critic/round1.sh     # 8 round-1 arms + ka2_stock_ref
bash reports/simple-critic/round2.sh a   # int_r1, int_r1_hinge, int_r1_rp, int_r1_b2, secant_r1
bash reports/simple-critic/round2.sh b   # secant_r1_b2
bash reports/simple-critic/round3.sh     # sec_t1, sec_drift, sec_center, sec_nodamp, sec_lazy4, then combo_a, combo_b
bash reports/simple-critic/toy100/run.sh secant_r1_b2 sec_nodamp sec_nodamp_lazy4 ref_stock ref_matched   # 100-Gaussian gate
bash reports/simple-critic/round5.sh     # round 5 ring arms (tail logs/{c3_*,rp_*}.log)
bash reports/simple-critic/toy100/run.sh c3_r1w c3_capinterp rp_center   # round 5 toy100 arms
# sparse-UCD: on branch sparse-ucd-secant, experiments/sparse_secant.sh

# a single arm (best formulation)
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$PWD .venv/bin/python reports/simple-critic/worker.py \
  --arm sec_nodamp --device cuda:0 --loss wgan --real r1 --lam-real 1 \
  --path secant --lam-path 10 --path-target 0.5 --path-u 0.1,0.9 --cap all --lam-cap 10 \
  --d-beta2 0.9 --latent-damping 0

# tail: one flushed line per observation (step phase modes hq pass | D(real) | D(fake) | grad-norms | loss terms)
tail -f reports/simple-critic/logs/sec_nodamp.log
tail -f reports/simple-critic/toy100/logs/sec_nodamp.log
grep -h COMPLETE reports/simple-critic/logs/*.log     # one-line outcome per arm

cd reports/simple-critic && python3 summarize.py && python3 summarize.py --diag
```

Evidence in this commit:
- Logs: `logs/<arm>.log`.
- Results: `runs/<arm>/result.json`, which has the config, LR check, scores
  and every observation.
- Not included: model checkpoints, `metrics.jsonl` and the reference's raw
  state.
- 100 Gaussians: `toy100/` has the adapter, configs, logs, per-arm
  `result.json`, D probes and the benchmark's own gate files. Heavy evidence
  (npz samples, snapshots, per-step events) is gitignored and regenerable
  with `toy100/run.sh`.
- Sparse-UCD: `sparse_ucd/LEADERBOARD.md` (copy); runs and code are on
  branch `sparse-ucd-secant`.
