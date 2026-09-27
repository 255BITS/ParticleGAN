# Simple critic on the 100-Gaussian gate

**Question.** Does the ring-8 winner (`sec_nodamp`) and the original
`secant_r1_b2` transfer to the repo's 100-Gaussian gate, under the same
constraints: no instance noise, constant LRs, no KA2/K3P controller?

**Answer.** No arm passes. Both simple critics cover all 100 modes on all three
problems, and mass TV is fine (0.035 to 0.042; the limit is 0.10). They fail
on within-mode shape instead. Final live HQ is 0.89 to 0.95 (the gate needs
0.97), and per-mode covariance ratios reach 0.17 to 2.1 (the gate allows 0.40
to 1.70). Even so, they beat the matched reference by a wide margin. The
benchmark's own critic with noise and annealing removed (`ref_matched`)
collapses to 22 and 13 modes on grid and staggered, with a max grad-norm of 122.
Only the shipped recipe (`ref_stock`: noise plus LR decay) passes.

**Update, new default init (#194, `batch_feature_zero`).** Every arm was rerun on
`cuda:1` under the new init (tables directly below; the old-init runs are archived in
`runs_oldinit/`). Still no arm passes. The ranking of the simple arms barely moves:
`rp_center` is still first, and the other HQ changes are within about 0.1 summed over the three problems. The big
change is the reference: **`ref_stock` goes from PASS 3/3 to FAIL 0/3** (100/71/40
modes). An old-init control on the same GPU and code reproduces the old PASS exactly,
so the init alone causes the loss. develop's own toy100 mechanism (the `--init`
registry hook, which also re-spaces the prior) gives 2/3: grid collapses to 44 modes.

## Protocol

- **Benchmark harness, unchanged.** `benchmarks.toy100` `run`: grid100,
  rotated100 and staggered100.
  - Base config: `configs/toy100/constraints_simple_regularization.json`.
  - Affine G, 20k learned particles, D 128x3 with Fourier 3, batch 2048.
  - Full 7000-update budget, the benchmark's seed 1234, evaluation every 250
    updates on 20k draws.
  - The coverage gate plus the strict accuracy gate (five terminal checks and
    a 100k holdout) score live weights.
  - This is the full protocol, not a shortened one: about 60 to 100 s of
    training per problem on the A6000 (`cuda:0`; the new-init rerun used `cuda:1`, also an A6000).
- **Init (rerun).** `run_arm.py` replaces the legacy recipe's `initialization=None` pin
  with `batch_feature_zero` (the package default) through the recipe path, for every
  arm including the refs. No registry hook is installed (`K3P_INIT` is refused). Every
  `result.json` has an `init_receipt` per problem. All 24 (8 arms x 3 problems) show
  `init=batch_feature_zero hook=None public_path_match=G,D,prior`. On this benchmark
  **only D changes**: G is the identity `affine_square_v1` (constants), D's biases stay zero,
  and the prior is the benchmark's own uniform(-5, 5) draw, which the public path does not
  re-initialize.
- **Matched arms** (all except `ref_stock`): input and output noise 0,
  `lr_floor` 1 and `network_lr_floor` 1. Every LR is constant, as the
  trainer's recorded LR policy shows.
- **Simple arms.** The adapter `run_arm.py` swaps in the critic loss from
  `../worker.py` (`SimpleCriticLoss`, the same code as the ring study).
  - Critic optimizer: `recipe.make_critic_optimizer(ema_critic=None)` with
    guard 0 and betas (0, 0.9). The run asserts that no controller is present.
  - Generator/prior step, EMA and data streams are `GANTrainer.step`'s own.
    The generator loss is RpGAN.
  - Nothing in `particlegan/` or `benchmarks/` was edited.
- **A2 latent damping.** The benchmark's legacy recipe already has A2 = 0, so
  the reference and `sec_nodamp` both run without it. `secant_r1_b2` sets
  it back to 0.5 to match the ring original.
- **D probe.** Every 50 updates, on 2048 real, fake and interpolate points,
  using its own RNG streams. The probe does not affect training.
- **`ref_stock` checks the GPU harness.** It reproduces the archived CPU run
  (`reports/toy100/simpler22`): the same final HQ 0.985/0.986/0.989 and first
  100 modes at 750 on all three problems.

## Leaderboard: new init (`runs/`)

Same columns and rank order as the old-init table below.

| # | arm | formulation | gate | accuracy | final live pass | final modes (g/r/s) | final HQ (g/r/s) | first 100 modes | passing live evals >=1000 | final min-max cov eig ratio | max abs D(real) | max grad-norm (all / >=1000) | median gmax | g(real) median >=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rp_center | rplogistic + r1(1) + path-secant(10,t=.5) + cap-all(10,c=3) + pair-center(1) [Db2=.9, A2=0, critic LR x.5] | FAIL | FAIL | 0/3 | 100/98/100 | 0.980/0.948/0.972 | 500/None/750 | 0/25, 0/25, 0/25 | 0.18-1.85, 0.14-2.18, 0.19-1.64 | 1.28 | 2.93 / 2.93 | 1.81/0.77/0.96 | 0.009/0.030/0.013 |
| 2 | c3_r1w | wgan + r1(0.1) + path-secant(10,t=.5) + cap-all(10,c=3) [Db2=.9, A2=0, critic LR x.5] | FAIL | FAIL | 0/3 | 100/100/100 | 0.967/0.922/0.972 | 500/1750/250 | 0/25, 0/25, 0/25 | 0.15-1.94, 0.16-2.05, 0.16-2.19 | 15.12 | 4.96 / 4.96 | 3.12/1.85/2.87 | 0.125/0.165/0.135 |
| 3 | sec_nodamp | wgan + r1(1) + path-secant(10,t=.5) + cap-all(10,c=1) [Db2=.9, A2=0] | FAIL | FAIL | 0/3 | 100/100/100 | 0.948/0.907/0.944 | 500/1250/500 | 0/25, 0/25, 0/25 | 0.19-1.87, 0.31-1.91, 0.21-1.50 | 4.44 | 2.26 / 2.26 | 0.96/0.34/0.65 | 0.029/0.031/0.028 |
| 4 | secant_r1_b2 | wgan + r1(1) + path-secant(10,t=.5) + cap-all(10,c=1) [Db2=.9, A2=.5] | FAIL | FAIL | 0/3 | 100/100/100 | 0.925/0.900/0.948 | 500/1500/500 | 0/25, 0/25, 0/25 | 0.39-1.63, 0.36-2.03, 0.13-2.15 | 5.27 | 2.14 / 2.14 | 0.91/0.52/0.73 | 0.027/0.038/0.030 |
| 5 | c3_capinterp | wgan + r1(1) + path-secant(10,t=.5) + cap-interp(10,c=3) [Db2=.9, A2=0, critic LR x.5] | FAIL | FAIL | 0/3 | 100/97/100 | 0.934/0.811/0.945 | 1500/1250/750 | 0/25, 0/25, 0/25 | 0.25-2.92, 0.29-2.98, 0.26-2.17 | 10.47 | 4.29 / 4.29 | 2.70/0.69/1.87 | 0.045/0.037/0.042 |
| 6 | sec_nodamp_lazy4 | wgan + r1(1) + path-secant(10,t=.5) + cap-all(10,c=1) [Db2=.9, A2=0, lazy_k=4] | FAIL | FAIL | 0/3 | 98/100/95 | 0.851/0.849/0.843 | 1000/2500/2000 | 0/25, 0/25, 0/25 | 0.30-2.67, 0.31-2.24, 0.19-2.26 | 2.70 | 2.34 / 2.34 | 0.70/0.42/0.75 | 0.059/0.051/0.052 |
| ref | ref_stock | REF: benchmark default as shipped (RpGAN-logistic + b_cap(1,k=1), Adam b2 .999, input noise .5->0, output noise .029, cosine LR decay) | FAIL | FAIL | 0/3 | 100/71/40 | 0.947/0.615/0.449 | 1000/None/None | 0/25, 0/25, 0/25 | 0.64-1.57, 0.01-2.81, 0.00-2.96 | 0.96 | 9.62 / 3.20 | 2.32/1.77/2.65 | 0.396/0.374/0.490 |
| ref | ref_matched | REF: benchmark default critic (RpGAN-logistic + b_cap(1,k=1), Adam b2 .999), noise off, constant LRs | FAIL | FAIL | 0/3 | 66/88/9 | 0.853/0.841/0.148 | 250/None/500 | 0/25, 0/25, 0/25 | 0.00-2.33, 0.00-2.36, 0.00-3.52 | 32.48 | 105.67 / 105.67 | 4.02/2.69/16.71 | 0.489/0.429/1.925 |
| ref | ref:archived-cpu | REF: shipped default, archived CPU run (reports/toy100/simpler22) | PASS | PASS | 3/3 | 100/100/100 | 0.985/0.986/0.989 | 750/750/750 | n/a | 0.70-1.27, 0.66-1.25, 0.67-1.15 | n/a | n/a | n/a | n/a |

## Old init against new init

HQ is final live HQ. Delta is the change in HQ summed over grid, rotated and staggered. Only one
seed was run, so changes under about 0.05 are not distinguishable from run-to-run variation.

| arm | rank old -> new | final live pass old -> new | modes old -> new (g/r/s) | final HQ old (g/r/s) | final HQ new (g/r/s) | delta HQ (sum) | EMA HQ new (g/r/s) | first 100 modes old -> new | max grad-norm all old -> new |
|---|---|---|---|---|---|---|---|---|---|
| rp_center | 1 -> 1 | 0/3 -> 0/3 | 98/100/100 -> 100/98/100 | 0.973/0.957/0.969 | 0.980/0.948/0.972 | +0.001 | 0.996/0.976/0.987 | None/4750/1000 -> 500/None/750 | 4.03 -> 2.93 |
| c3_r1w | 4 -> 2 | 0/3 -> 0/3 | 100/100/98 -> 100/100/100 | 0.950/0.938/0.869 | 0.967/0.922/0.972 | +0.104 | 0.985/0.983/0.986 | 500/1500/250 -> 500/1750/250 | 4.85 -> 4.96 |
| sec_nodamp | 5 -> 3 | 0/3 -> 0/3 | 100/100/100 -> 100/100/100 | 0.903/0.886/0.954 | 0.948/0.907/0.944 | +0.057 | 0.984/0.972/0.983 | 1000/1500/500 -> 500/1250/500 | 5.33 -> 2.26 |
| secant_r1_b2 | 2 -> 4 | 0/3 -> 0/3 | 100/100/100 -> 100/100/100 | 0.948/0.897/0.951 | 0.925/0.900/0.948 | -0.023 | 0.986/0.960/0.981 | 1250/2000/500 -> 500/1500/500 | 5.33 -> 2.14 |
| c3_capinterp | 3 -> 5 | 0/3 -> 0/3 | 100/100/100 -> 100/97/100 | 0.909/0.908/0.940 | 0.934/0.811/0.945 | -0.067 | 0.981/0.974/0.980 | 1250/1500/500 -> 1500/1250/750 | 5.53 -> 4.29 |
| sec_nodamp_lazy4 | 6 -> 6 | 0/3 -> 0/3 | 98/99/100 -> 98/100/95 | 0.859/0.813/0.922 | 0.851/0.849/0.843 | -0.051 | 0.896/0.957/0.864 | 2500/2500/2500 -> 1000/2500/2000 | 7.85 -> 2.34 |
| ref_stock | ref | **3/3 -> 0/3** | 100/100/100 -> **100/71/40** | 0.985/0.986/0.989 | 0.947/0.615/0.449 | **-0.949** | 0.947/0.615/0.450 | 750/750/750 -> 1000/None/None | 5.93 -> 9.62 |
| ref_matched | ref | 0/3 -> 0/3 | 22/92/13 -> 66/88/9 | 0.285/0.856/0.187 | 0.853/0.841/0.148 | +0.514 | 0.881/0.842/0.152 | 250/None/250 -> 250/None/500 | 122.25 -> 105.67 |

### Controls for `ref_stock` (`runs_ctrl/`, `logs_ctrl/`, `run_arm.py --init`)

| run | init | final live pass | modes (g/r/s) | final HQ (g/r/s) |
|---|---|---|---|---|
| ref_stock `--init none` (cuda:1, merged code) | old random init (legacy pin kept) | **3/3** | 100/100/100 | 0.985/0.986/0.989 (identical to the old run) |
| ref_stock `--init hook` | develop's registry `use_init('batch_feature_zero')` (also re-spaces the prior) | 2/3 | 44/100/100 | 0.479/0.985/0.990 |
| sec_nodamp `--init hook` | same as above | 0/3 | 100/100/100 | 0.925/0.865/0.903 (EMA 0.988/0.964/0.985) |

### Findings (new init)

1. **The init breaks the shipped reference on this config.** The control shows that the merge,
   the GPU and concurrency change nothing: `--init none` gives the old PASS bit for bit. With the
   new D init, `ref_stock` loses 29 and 60 modes on rotated and staggered. The hook
   variant fails on grid instead (44 modes, max grad-norm 18). develop reports 22/22 on toy100,
   but that result used a different run setup than this one. Here, `constraints_simple_regularization.json`
   with the batch_feature_zero D is not robust: whether it collapses depends on the problem.
2. **The simple critics do not care about the init.** All six simple arms stay at 0/3, cover
   all or nearly all modes, and keep mass TV at 0.03 to 0.07. The within-mode shape failure is
   unchanged: the covariance ratio is still 0.13 to 2.9, and EMA modes are still squeezed to
   0.04 to 0.18. The new init removes the untrained-D gradient spike at update 1 (max
   grad-norm 5.3 -> 2.2 on the secant arms), and full coverage arrives the same or earlier.
3. **The ranking is stable where the gaps are large.** `rp_center` stays first (grid 0.980 clears
   0.97 again). `c3_r1w` and `sec_nodamp` move up, and `secant_r1_b2` and `c3_capinterp` move
   down. These shifts are 0.02 to 0.10 of summed HQ, about the size of the run-to-run noise.
   `lazy_k=4` is still last.
4. **`ref_matched` is still the least stable arm.** Its grid improves from 22 to 66 modes, but it
   collapses on staggered (9 modes) and its grad-norm still exceeds 100.

### Recommendations (new init)

- **Before trusting any toy100 reference under the new init,** check that the
  `batch_feature_zero` D is compatible with `b_cap` + noise-annealed RpGAN on this config. The
  shipped reference no longer passes, so it is not a valid baseline row until that is fixed.
- **The simple-critic plan is unchanged.** The init does not touch the within-mode shape failure.
  The margin secant and a lower constant particle LR (below) remain the next distinct tests.

## Old-init leaderboard (archived, `runs_oldinit/`)

Rank order: final live passes, then passing live evaluations from update 1000
on, then the sum of final HQ. `ref` rows are not ranked. g/r/s means grid,
rotated, staggered.

| # | arm | formulation | gate | accuracy | final live pass | final modes (g/r/s) | final HQ (g/r/s) | first 100 modes | passing live evals >=1000 | final min-max cov eig ratio | max abs D(real) | max grad-norm (all / >=1000) | median gmax | g(real) median >=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rp_center | rplogistic + r1(1) + path-secant(10,t=.5) + cap-all(10,c=3) + pair-center(1) [Db2=.9, A2=0, critic LR x.5] | FAIL | FAIL | 0/3 | 98/100/100 | 0.973/0.957/0.969 | None/4750/1000 | 0/25, 0/25, 0/25 | 0.18-2.24, 0.21-1.78, 0.13-2.02 | 1.25 | 4.03 / 3.15 | 1.91/0.78/1.14 | 0.009/0.030/0.018 |
| 2 | secant_r1_b2 | wgan + r1(1) + path-secant(10,t=.5) + cap-all(10,c=1) [Db2=.9, A2=.5] | FAIL | FAIL | 0/3 | 100/100/100 | 0.948/0.897/0.951 | 1250/2000/500 | 0/25, 0/25, 0/25 | 0.29-1.73, 0.29-1.89, 0.27-1.72 | 5.44 | 5.33 / 2.16 | 0.94/0.39/0.90 | 0.029/0.029/0.037 |
| 3 | c3_capinterp | wgan + r1(1) + path-secant(10,t=.5) + cap-interp(10,c=3) [Db2=.9, A2=0, critic LR x.5] | FAIL | FAIL | 0/3 | 100/100/100 | 0.909/0.908/0.940 | 1250/1500/500 | 0/25, 0/25, 0/25 | 0.14-3.22, 0.26-1.89, 0.20-2.44 | 11.01 | 5.53 / 5.53 | 2.72/0.53/2.37 | 0.039/0.035/0.051 |
| 4 | c3_r1w | wgan + r1(0.1) + path-secant(10,t=.5) + cap-all(10,c=3) [Db2=.9, A2=0, critic LR x.5] | FAIL | FAIL | 0/3 | 100/100/98 | 0.950/0.938/0.869 | 500/1500/250 | 0/25, 0/25, 0/25 | 0.16-1.87, 0.17-1.74, 0.29-3.00 | 14.73 | 4.85 / 4.27 | 3.18/1.80/2.96 | 0.133/0.163/0.155 |
| 5 | sec_nodamp | wgan + r1(1) + path-secant(10,t=.5) + cap-all(10,c=1) [Db2=.9, A2=0] | FAIL | FAIL | 0/3 | 100/100/100 | 0.903/0.886/0.954 | 1000/1500/500 | 0/25, 0/25, 0/25 | 0.17-1.83, 0.36-2.12, 0.21-1.76 | 4.85 | 5.33 / 1.95 | 0.88/0.25/0.89 | 0.029/0.030/0.035 |
| 6 | sec_nodamp_lazy4 | wgan + r1(1) + path-secant(10,t=.5) + cap-all(10,c=1) [Db2=.9, A2=0, lazy_k=4] | FAIL | FAIL | 0/3 | 98/99/100 | 0.859/0.813/0.922 | 2500/2500/2500 | 0/25, 0/25, 0/25 | 0.20-2.05, 0.42-2.22, 0.24-2.36 | 3.93 | 7.85 / 1.85 | 0.81/0.35/0.75 | 0.066/0.041/0.056 |
| ref | ref_stock | REF: benchmark default as shipped (RpGAN-logistic + b_cap(1,k=1), Adam b2 .999, input noise .5->0, output noise .029, cosine LR decay) | PASS | PASS | 3/3 | 100/100/100 | 0.985/0.986/0.989 | 750/750/750 | 8/25, 6/25, 8/25 | 0.62-1.18, 0.68-1.18, 0.59-1.29 | 0.81 | 5.93 / 2.98 | 1.25/1.54/2.13 | 0.777/0.579/0.491 |
| ref | ref_matched | REF: benchmark default critic (RpGAN-logistic + b_cap(1,k=1), Adam b2 .999), noise off, constant LRs | FAIL | FAIL | 0/3 | 22/92/13 | 0.285/0.856/0.187 | 250/None/250 | 0/25, 0/25, 0/25 | 0.00-5.34, 0.05-2.20, 0.00-3.17 | 14.33 | 122.25 / 122.25 | 11.95/3.08/7.56 | 1.324/0.473/0.981 |
| ref | ref:archived-cpu | REF: shipped default, archived CPU run (reports/toy100/simpler22) | PASS | PASS | 3/3 | 100/100/100 | 0.985/0.986/0.989 | 750/750/750 | n/a | 0.70-1.27, 0.66-1.25, 0.67-1.15 | n/a | n/a | n/a | n/a |

Gate thresholds: 100 modes, HQ >= 0.97, mass TV <= 0.10, max mode mass <= 2%,
covariance eigenvalue ratio in [0.40, 1.70] and radial median ratio in
[0.65, 1.40], sustained over the five terminal checks.

## Findings (old init)

1. **The simple critic transfers its stability, not its accuracy.**
   - D stays tame. After update 1000, max grad-norm is 1.95 (`sec_nodamp`)
     and 2.16 (`secant_r1_b2`), and max |D(real)| is about 5. The only
     larger value is the untrained D at update 1 (5.3).
   - Coverage is complete. All 100 modes are reached by update 500 to 2000,
     later than the stock 750.
   - The matched reference is the opposite: g(real) around 1, gmax up to 122,
     and a mode collapse. Without noise and decay, the benchmark's
     b_cap + RpGAN critic is the less stable of the two.
2. **The simple arms fail on within-mode shape.**
   - D is almost flat at the data. The median g(real) after update 1000 is
     0.03, against 0.5 to 0.8 for the passing `ref_stock`.
   - A flat D cannot shape the width of a σ = 0.03 mode. Live samples are too
     spread (HQ 0.89 to 0.95, covariance ratio up to 2.1), while some modes
     are thin lines (minimum ratio 0.17 to 0.36).
   - EMA samples reach HQ 0.96 to 0.99, but their modes are too narrow
     (minimum ratio 0.08 to 0.16). EMA fails too, so this is not only live
     jitter from constant LRs.
   - Two causes are plausible:
     - R1(1) flattens D at every real point.
     - The secant rewards D rising toward the nearest *sampled* real point,
       even inside a mode, where it acts as noise.
3. **A2 = 0.5 against 0: mixed.**
   - `secant_r1_b2` has the higher final HQ on grid (0.948 against 0.903).
     Staggered is about equal, and rotated is 0.897 against 0.886.
   - Both arms have 0 of 25 passing evaluations.
   - On the ring, A2 = 0 was the clear win. Here the difference is within
     noise, and one seed cannot separate the two.
4. **lazy_k = 4 is worse here, as it was on the ring.** First full coverage
   slips to 2500, final HQ drops to 0.81 to 0.92, and grid and rotated lose 1
   to 2 modes. Critic-step regularization every step is needed.

## Round 5 arms (ring B_cap3 family)

Three ring round-5 formulations, ported with the ring's cap c=3 and critic LR x0.5 (config
`d_lr_mult` 0.5): `rp_center` (RpGAN critic + pair-center(1) + R1 + secant + cap-all), `c3_capinterp`
(cap on interpolates only) and `c3_r1w` (R1 0.1). Term definitions are in `../README.md`, Round 5.
All fail. Final live HQ and EMA are below:

| arm | final HQ live (g/r/s) | final HQ EMA (g/r/s) | min-max cov ratio EMA (g / r / s) | g(real) median >=1000 |
|---|---|---|---|---|
| rp_center | 0.973/0.957/0.969 | 0.999/0.981/0.991 | 0.10-1.55 / 0.05-0.66 / 0.08-1.62 | 0.009/0.030/0.018 |
| c3_capinterp | 0.909/0.908/0.940 | 0.986/0.973/0.981 | 0.14-1.86 / 0.11-1.22 / 0.11-1.09 | 0.039/0.035/0.051 |
| c3_r1w | 0.950/0.938/0.869 | 0.988/0.984/0.986 | 0.07-1.20 / 0.09-0.93 / 0.06-0.89 | 0.133/0.163/0.155 |
| sec_nodamp | 0.903/0.886/0.954 | 0.987/0.963/0.981 | 0.11-0.94 / 0.16-1.25 / 0.11-0.86 | 0.029/0.030/0.035 |

- **Flat D is not the cause.** R1 0.1 raises g(real) about 5x but does not improve HQ or shape.
  `rp_center` has the flattest D and the best live HQ (grid 0.973 clears the 0.97 bar).
- **Shape is the shared failure.** Every arm's EMA has some modes squeezed to lines (min cov ratio
  0.05 to 0.16), so the critic's real term is not what sets per-mode width. Next test: a lower
  constant particle LR (`prior_lr_mult` 1) on `rp_center`.
- `rp_center` keeps |D(real)| at 1.25; the wgan arms reach 11 (`c3_capinterp`) and 15 (`c3_r1w`).
  It loses 2 modes on grid and its mass TV rises to 0.062 (limit 0.10).

## Recommendations (distinct formulations, same constraints)

1. **Margin secant:** `relu(t*(|r_nn - f| - m) - (D(r_nn) - D(f)))^2` with
   m about 3σ = 0.09. The path then acts only on fakes outside a mode and
   stops injecting per-sample bumps inside it. This tests finding 2's
   secant-noise explanation.
2. **Weaker real term:** R1 at 0.1, or R1 replaced by `center(1)` (level
   only) with cap-all left as the slope limit. This lets D keep enough slope
   at the data to shape within-mode width, and tests the R1-flattening
   explanation. Check the ring hold again afterwards: R1 was what kept
   |D(real)| low there.
3. **A lower constant particle/G LR** (for example `prior_lr_mult` 1 in
   place of 2). This is a fixed constant, not a schedule. It tests whether
   live HQ is capped by the step-size noise floor. The EMA/live gap (HQ 0.98
   against 0.91) suggests part of the loss is jitter.
4. **Drop lazy_k from the simple critic.** It failed on both benchmarks.

## Reproduce

```bash
bash reports/simple-critic/toy100/run.sh ref_matched sec_nodamp secant_r1_b2 sec_nodamp_lazy4 ref_stock
bash reports/simple-critic/toy100/run.sh c3_r1w c3_capinterp rp_center   # round 5
tail -f reports/simple-critic/toy100/logs/sec_nodamp.log      # PROBE/EVAL lines, one per observation
python3 reports/simple-critic/toy100/summarize.py [--detail] [--runs-dir runs_oldinit]
# new-init rerun on cuda:1 (all 8 arms concurrently, ~8 min):
GPU=1 bash reports/simple-critic/toy100/run.sh ref_stock ref_matched sec_nodamp secant_r1_b2 sec_nodamp_lazy4 c3_r1w c3_capinterp rp_center
# init controls: --init none (old init replay) | hook (develop's registry mechanism)
.venv/bin/python -u reports/simple-critic/toy100/run_arm.py --arm ref_stock --init none --runs-dir reports/simple-critic/toy100/runs_ctrl/oldinit
```
Artifacts: `runs/<arm>/result.json`, `runs/<arm>/diag/<problem>.jsonl` (D probe),
`runs/<arm>/bench/` (benchmark evidence: gate.json, accuracy-gate.json,
leaderboard.md, per-problem summary.json). Sample `.npz` files, snapshots,
source archives and per-step `events.jsonl` are gitignored and can be
regenerated.
