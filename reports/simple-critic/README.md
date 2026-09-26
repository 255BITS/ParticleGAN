# Simple 3-term critic on the KA2 shift protocol

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
| 1 | secant_r1_b2 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) [Dβ2=0.9] | 83/120 | 630 | 118/158 | 33 | 22 | 37 (from 4240) | 0.981 | 3.17 | 2.82 | 77 |
| ref | ka2_stock_ref | REF: stock KA2 worker rerun (RpGAN+KA2 penalty/controller, noise on) | 61/120 | 120 | 126/209 | 10 | 58 | 15 (from 4460) | 0.986 | n/a | n/a | 142 |
| ref | ref:ka2-constant | RpGAN+KA2 penalty/controller, noise on (archived) | 61/120 | 120 | 126/209 | 10 | 58 | 15 (from 4460) | 0.986 | n/a | n/a | 142 |
| 2 | int_r1_b2 | wgan + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) [Dβ2=0.9] | 28/120 | 60 | 58/215 | 63 | 25 | 0 | 0.679 | 1.64 | 1.92 | 249 |
| 3 | secant_r1 | wgan + r1(1) + path-secant(10,t=0.5) + cap-all(10,c=1) | 51/120 | none | 0/0 | 7 | 37 | 0 | 0.259 | 161.28 | 394.54 | 69 |
| 4 | no_path | wgan + drift(0.1) + cap-all(10,c=1) | 35/120 | none | 0/0 | 14 | 42 | 0 | 0.000 | 46.09 | 201.68 | 85 |
| 5 | int_r1_hinge | hinge + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 19/120 | none | 0/0 | 18 | 20 | 0 | 0.010 | 27.99 | 181.99 | 101 |
| 6 | int_r1 | wgan + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 9/120 | none | 0/0 | 9 | 64 | 0 | 0.043 | 30.33 | 138.60 | 111 |
| 7 | full | wgan + drift(0.1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.016 | 120.32 | 685.01 | 120 |
| 8 | full_hinge | hinge + drift(0.1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.005 | 50.71 | 211.99 | 120 |
| 9 | full_r1 | wgan + r1(1) + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.094 | 6.93 | 29.86 | 120 |
| 10 | int_r1_rp | rplogistic + r1(1) + path-lower(10,t=0.3,u=0.1,0.9) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.045 | 6.40 | 34.30 | 120 |
| 11 | no_cap | wgan + drift(0.1) + path-lower(10,t=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.000 | 60288.24 | 4629336.00 | 120 |
| 12 | no_real | wgan + path-lower(10,t=1) + cap-all(10,c=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.389 | 15.88 | 25.52 | 120 |
| 13 | wgangp_ref | wgan + drift(0.1) + path-two_sided(10,t=1) | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.631 | 8.34 | 52.73 | 120 |

Critic diagnostics are medians over all 460 observations of the probe (real,
fake and path points). The full table is in `LEADERBOARD.md` and
`python3 summarize.py --diag`.
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
- **A2 latent-row damping was on in every arm, including the reference.**
  `latent_damping_max_rate=0.5` (`K3PGeneratorAdam`, `particlegan/k3p.py`)
  scales each particle row's step by 0.5 to 1, based on the cosine with that
  row's last gradient. It also sets that group's beta1 to 0.5 inside the step.
  - The prior's `lr` stays constant, but its effective step depends on
    gradient memory.
  - This is a confound shared by every arm, so it does not separate arms.
    `--latent-damping 0` was never run.
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
- **Only one benchmark was tested:** ring-8 shift, one seed.

## Next experiments (distinct formulations, same constraints)

1. **Secant target t = 1.0**, the largest rise the cap allows, to speed
   arrival after the shift.
2. **Pin D's level** with a tiny drift (1e-3) or by subtracting the batch mean
   of D(real).
3. **Spectral-norm D in place of the cap term**, to get down to two terms.
4. **`--latent-damping 0`** on `secant_r1_b2`, to remove the A2 confound and
   leave a fully plain generator step.
5. **Promotion check:** run `secant_r1_b2` on 100 Gaussians and the sparse-UCD
   toy before treating it as a KA2 replacement.

## Reproduce

Run from the repo root. Each run takes about 35 to 90 s on an A6000.

```bash
# confirm the worktree's particlegan is imported
PYTHONPATH=$PWD .venv/bin/python -c "import particlegan; print(particlegan.__file__)"

bash reports/simple-critic/round1.sh     # 8 round-1 arms + ka2_stock_ref
bash reports/simple-critic/round2.sh a   # int_r1, int_r1_hinge, int_r1_rp, int_r1_b2, secant_r1
bash reports/simple-critic/round2.sh b   # secant_r1_b2

# a single arm (best formulation)
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$PWD .venv/bin/python reports/simple-critic/worker.py \
  --arm secant_r1_b2 --device cuda:0 --loss wgan --real r1 --lam-real 1 \
  --path secant --lam-path 10 --path-target 0.5 --cap all --lam-cap 10 --d-beta2 0.9

# tail: one flushed line per observation (step phase modes hq pass | D(real) | D(fake) | grad-norms | loss terms)
tail -f reports/simple-critic/logs/secant_r1_b2.log
grep -h COMPLETE reports/simple-critic/logs/*.log     # one-line outcome per arm

cd reports/simple-critic && python3 summarize.py && python3 summarize.py --diag
```

Evidence in this commit:
- Logs: `logs/<arm>.log`.
- Results: `runs/<arm>/result.json`, which has the config, LR check, scores
  and every observation.
- Not included: model checkpoints, `metrics.jsonl` and the reference's raw
  state.
