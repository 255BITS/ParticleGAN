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
    training per problem on the A6000 (`cuda:0`).
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

## Leaderboard

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

## Findings

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
python3 reports/simple-critic/toy100/summarize.py [--detail]
```
Artifacts: `runs/<arm>/result.json`, `runs/<arm>/diag/<problem>.jsonl` (D probe),
`runs/<arm>/bench/` (benchmark evidence: gate.json, accuracy-gate.json,
leaderboard.md, per-problem summary.json). Sample `.npz` files, snapshots,
source archives and per-step `events.jsonl` are gitignored and can be
regenerated.
