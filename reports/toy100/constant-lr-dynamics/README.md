# Constant-LR GAN dynamics

Three mechanisms for the toy100 probes at K3P's pre-anneal rates, held constant: network `0.00425`, prior `0.0085` (`prior_lr_mult` 2). `lr_anneal_start` is 0 and both floors are 1, so `policy_multipliers` stays 1 at steps 0, 720, 721, 1199, 2400, and 4000. The horizon cap stays 1600. No decay, anneal, or warm restart.

CPU numbers below do not rank against the A6000. `--init hid_q` fixes the weights. Offset 0 is the unshifted repo seed. The other offsets (`101 202 303 404 505 606 707`) change samples and noise only, via `K3P_SEED_OFFSET`.

Flag: `K3P_DYNAMICS` in `{unit_rms, pair_chord, shared_batch, unrolled}`. Unset is the constant-LR K3P baseline. `sitecustomize.py` in this directory installs the named mechanism before the probe captures `Adam.step`. The screen puts this directory on `PYTHONPATH`.

Each idea has one setting taken from the existing step, the existing penalty term, or the existing batch. No coefficient, clip, or ratio was swept. Not on this track: coverage, anchors, forward-KL as a training signal, mode quotas, Chamfer assignment.

## Why these three

PR #178: the ring/hold/stay pass is the cosine anneal. From update 721 the network LR falls to 1% and the prior LR to 5% by update 1200, and the stay window runs at that floor. Holding `0.00425` / `0.0085` fails ring, hold, and stay on 8/8 offsets. The early amplifier is a joint G–D kink: at update 4 a finite-difference probe of size `1e-5` reports a joint singular value ~566, and `1e-4` reports ~109. With Adam `β1 = 0` a sign flip moves a coordinate by about `2 lr`, so the reported gain is order `2 * 0.00425 / ε` (850 and 85). LeakyReLU's Hessian is zero almost everywhere, so an HVP limiter does not see the jump. A constant scale of the gradient is invisible to Adam.

PR #177: a failure is particles climbing onto a neighbor the critic scores higher. About 40 updates before the crossing, those gradients jump ~10× and the D/G gradient ratio reaches ~23. `β2 = 0.999` remembers `g²` for about 1000 updates, so that spike becomes a step about 10× `lr`. The emptied mode stays empty (real-vs-fake logit gap ~4; passes stay near 0). First-order ascent on `D` does not leave a local maximum toward a higher distant mode. The executed penalty is legacy `a_r1r2` at the real point and the fake point. It does not see the open segment between them.

PR #176: under `hid_q` the ring is decided in short windows of the critic's particle index and of the data batch.

Already failed at one written setting, and not repeated: optimistic `α = 1`, extragradient, simultaneous Adam, EMA-fake (decay 0.995), wired `k3p_pull`, dense A2 row damping.

### `unit_rms` — the kink and the lagged spike

```
Δ = -lr * g / (sqrt(mean(g²)) + ε)
```

`lr` and `ε` are the ones already on the Adam group (`0.00425` or `0.0085`, `ε = 1e-8`). Scaling `g` does not change `Δ`. A 10× spike does not change the RMS of `Δ`. A single near-zero coordinate cannot jump by `2 lr`. Moments are not read or written. The ring's A2 rewrite never fired, so dropping the moment buffer does not turn that rewrite off. Weight decay must be 0, which it is.

### `pair_chord` — the segment the hop climbs

Add the endpoint term that is already zero-centered, at the midpoint of the relativistic pair, with the coefficient already written for one endpoint.

- Legacy `a_r1r2` (the ring): `(c/2) mean ||∇D(mid)||²`, same expression as each of the real and fake terms. Not divided by dimension.
- K3P at `s == 1` (constant LR, `GANTrainer`): `(c/2) mean(||∇D(mid)||² / d)`, same expression as the real term.

No new coefficient, cap, or target slope. Other arms are unchanged. Early fakes sit near the origin and reals sit on the ring, so the same chord is where a small generator move changes `D`.

### `shared_batch` — one game on one sample

The ring host and `GANTrainer` draw a second latent batch and a second real batch after the critic step. This keeps critic-then-generator order. The generator re-forwards the latents the critic just scored and is paired with that same real batch. Output noise on the generator forward is still drawn: that draw is the observation of this forward, not a second minibatch. Not simultaneous, and not extragradient.

### `unrolled` — D, k steps ahead, on the generator's batch

Metz et al. 2017, Unrolled GANs, toy-mixture depth `k = 5` (`particlegan/dynamics/unrolled.py`, `UNROLL_K`). The real critic step is unchanged. The generator loss, and the particle step that flows through `D(G(z))`, is scored on a functional copy of D after five more steps of D's own Adam update (group lr, betas, epsilon, single-tensor rule, K3P spike guard when the critic optimizer has one). Those five steps minimize the same critic loss, including the penalty, on this generator batch: the live fake and the real batch from the critic step. No new minibatch, so the training RNG is unchanged and the reaction is to the samples G is stepping on.

The copy's parameter values match that Adam update. The second-moment scale is held constant in the generator derivative: `sqrt(v)` has no derivative on coordinates `v` has not seen, and `beta2 = 0.999` barely moves `v` in five steps. The reaction G differentiates is the first-moment step. `k = 1` was the allowed fallback if 5 was too slow; a 20-step CPU timing was 0.064 s/step, so the screen uses 5.

## Patches

Each file applies alone on develop `407c2f7a` (`git apply`). They are not meant to be stacked: `__init__.py` is shared, and each patch's `sitecustomize.py` installs only that mechanism. This branch's `sitecustomize.py` installs whichever of `unit_rms`, `pair_chord`, `shared_batch`, or `unrolled` is set.

- `patches/unit_rms.patch`
- `patches/pair_chord.patch`
- `patches/shared_batch.patch` (also the `mode_hold` host and `GANTrainer.step`)

## GPU commands

A6000, eight offsets, constant LR. One process per mechanism so a failure does not share a job slot with the others. `unequal` is included. See the note under it before spending a GPU on that gate.

```
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics baseline --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics unit_rms --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics pair_chord --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics shared_batch --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics unrolled --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics unrolled_after_acquire --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
```

Pass rules the screen prints: ring and unequal use the probe verdict (`modes`, `hq`, `passing_suffix`). Hold passes only when `status` is `PASS` and `hold_checks == 1200`. Stay passes only when shift's continued hold is `120/120` and `pass_all`. The shift terminal status is not the stay gate.

`vector_unequal_mass` used to crash before any step. The probe imports `mechanism.py`, which replaces `GradientPenalty.penalty` with `scaled_penalty`. `CriticPenalty` always passes `ema_critic`, and the K3P penalty class has no `arm`, so the call raised `TypeError` before the body ran. `scaled_penalty` now accepts that keyword and, when the instance has no `arm`, delegates to the original K3P penalty. Ring, hold, and stay use the legacy penalty and are not on this path. Unequal runs.

## CPU sanity

96 jobs, 4 at a time, `OMP_NUM_THREADS=1`. Ring ~15 s, hold ~90–100 s (`NOT_CONVERGED` at the 4800-check settling budget), stay ~40–48 s. Group LRs stayed `0.00425` and `[0.00425, 0.0085]` through the logged steps. These pass rates are not an A6000 ranking.

| dynamics | ring | hold 1200 | stay 120/120 |
| --- | ---: | ---: | ---: |
| baseline | 0/8 | 0/8 | 0/8 |
| unit_rms | 0/8 | 0/8 | 0/8 |
| pair_chord | 0/8 | 0/8 | 0/8 |
| shared_batch | 1/8 | 0/8 | 0/8 |

The only pass is `shared_batch` ring at offset 606 (8 modes, hq 1.0, passing suffix 8). Hold never started a 1200-check window on any dynamic (`hold_checks` 0, `NOT_CONVERGED`). Stay never reached 120/120. Nothing here acquires all modes and then stays at a constant learning rate.

Ring modes / hq / passing suffix:

| offset | baseline | unit_rms | pair_chord | shared_batch |
| ---: | --- | --- | --- | --- |
| 0 | 8 / 0.878 / 0 | 2 / 0.156 / 0 | 4 / 0.297 / 0 | 0 / 0.000 / 0 |
| 101 | 7 / 0.659 / 0 | 2 / 0.221 / 0 | 3 / 0.038 / 0 | 0 / 0.000 / 0 |
| 202 | 7 / 0.985 / 0 | 3 / 0.206 / 0 | 5 / 0.340 / 0 | 8 / 0.990 / 3 |
| 303 | 6 / 0.670 / 0 | 3 / 0.334 / 0 | 5 / 0.581 / 0 | 8 / 0.952 / 1 |
| 404 | 6 / 0.400 / 0 | 2 / 0.173 / 0 | 7 / 0.739 / 0 | 6 / 0.484 / 0 |
| 505 | 4 / 0.263 / 0 | 4 / 0.236 / 0 | 3 / 0.299 / 0 | 8 / 1.000 / 1 |
| 606 | 7 / 0.694 / 0 | 5 / 0.265 / 0 | 7 / 0.648 / 0 | 8 / 1.000 / 8 PASS |
| 707 | 0 / 0.000 / 0 | 2 / 0.163 / 0 | 4 / 0.392 / 0 | 1 / 0.078 / 0 |

Offset 0 baseline matches #178's constant-LR offset 0 (FAIL, 8 modes, hq 0.878). `unit_rms` and `pair_chord` covered fewer modes than that baseline on every offset. `shared_batch` reached 8 modes on four offsets and kept a passing suffix on one.

Best stay fraction inside the 120 checks (still a fail): baseline 53/120 (offset 606), pair_chord 17/120 (offset 0), shared_batch 64/120 (offset 202).

`unit_rms` stay is `ERROR` on all 8 offsets: `continuous_probe` requires Adam's moment counter to equal the update count, and this step does not write moments. The run itself finished (receipt `steps=7200` = 3600 updates × D and G; displacement RMS stayed ~0.00425). Every 100-step checkpoint from 1200 through 2400 misses 8 modes and hq 0.90 (13/13 points on each offset), so that window was not a stay either.

Receipts on the ring: `unit_rms` `steps=2400`, `pair_chord` `calls=1200`. At step 1200 the `unit_rms` critic displacement RMS was 0.00425 while its gradient RMS was 0.0125.

## Recommendation

Do not promote any of the three. The constant-LR baseline still fails ring, hold, and stay on 8/8, which repeats #178. `unit_rms` removes the sign kink and the lagged 10× step (displacement stays at `lr` while the gradient moves) and coverage gets worse, so that kink was not what was holding the modes. `pair_chord` penalizes the open segment and also covers fewer modes than the baseline. `shared_batch` is the only one that passes a CPU ring at all, and it does not hold. A GPU ranking, if one is run, should start from `shared_batch` against this same constant-LR baseline. It should not be read as a recipe that stays. `unequal` was blocked on the probe for that screen. The penalty fix and the `unrolled` screen are below.

## Unrolled CPU screen

Same constant LR, this build, 64 jobs, 4 at a time, `OMP_NUM_THREADS=1`, `k=5`. Baseline on this build matches the table above and #183: ring 0/8, hold 0/8, stay 0/8, best stay 53/120 (offset 606). `unrolled` ring is also 0/8, so it does not lose ring passes. It does not add a hold PASS or a stay 120/120. **Handoff bar missed.** CPU only; not an A6000 ranking.

| dynamics | ring | hold 1200 | stay 120/120 | unequal |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0/8 | 0/8 | 0/8 | 2/8 |
| unrolled | 0/8 | 0/8 | 0/8 | 0/8 |

Ring at step 1200, modes / hq / passing suffix. Both columns FAIL.

| offset | baseline | unrolled |
| ---: | --- | --- |
| 0 | 8 / 0.878 / 0 | 0 / 0.000 / 0 |
| 101 | 7 / 0.659 / 0 | 0 / 0.000 / 0 |
| 202 | 7 / 0.985 / 0 | 3 / 0.142 / 0 |
| 303 | 6 / 0.670 / 0 | 0 / 0.000 / 0 |
| 404 | 6 / 0.400 / 0 | 0 / 0.000 / 0 |
| 505 | 4 / 0.263 / 0 | 0 / 0.000 / 0 |
| 606 | 7 / 0.694 / 0 | 2 / 0.085 / 0 |
| 707 | 0 / 0.000 / 0 | 2 / 0.079 / 0 |

Hold. Baseline is `NOT_CONVERGED`, `hold_checks` 0, on every offset. Unrolled enters `HOLDING` on two offsets and then drops the window.

| offset | unrolled hold |
| ---: | --- |
| 0 | `POST_CONVERGENCE_FAIL`, 1151/1200, converged step 2109, first failure step 3260 (live modes 7, hq 0.906) |
| 101 | `NOT_CONVERGED`, 0 |
| 202 | `NOT_CONVERGED`, 0 |
| 303 | `NOT_CONVERGED`, 0 |
| 404 | `NOT_CONVERGED`, 0 |
| 505 | `NOT_CONVERGED`, 0 |
| 606 | `NOT_CONVERGED`, 0 |
| 707 | `POST_CONVERGENCE_FAIL`, 219/1200, converged step 3871, first failure step 4090 (live modes 8, hq 0.899) |

Stay, passing checks out of 120. None is 120/120.

| offset | baseline | unrolled |
| ---: | ---: | ---: |
| 0 | 33/120 | 50/120 |
| 101 | 0/120 | 0/120 |
| 202 | 0/120 | 0/120 |
| 303 | 9/120 | 0/120 |
| 404 | 17/120 | 0/120 |
| 505 | 0/120 | 0/120 |
| 606 | 53/120 | 59/120 |
| 707 | 0/120 | 0/120 |

Unequal has no mode count (vector mass metrics). Status, passing suffix, hq. Baseline passes offsets 101 and 505. Unrolled passes none; suffix is 0 everywhere.

| offset | baseline | unrolled |
| ---: | --- | --- |
| 0 | FAIL / 2 / 0.968 | FAIL / 0 / 0.012 |
| 101 | PASS / 5 / 0.971 | FAIL / 0 / 0.021 |
| 202 | FAIL / 0 / 0.984 | FAIL / 0 / 0.939 |
| 303 | FAIL / 0 / 0.977 | FAIL / 0 / 0.954 |
| 404 | FAIL / 3 / 0.912 | FAIL / 0 / 0.910 |
| 505 | PASS / 10 / 0.982 | FAIL / 0 / 0.822 |
| 606 | FAIL / 0 / 0.957 | FAIL / 0 / 0.384 |
| 707 | FAIL / 0 / 0.974 | FAIL / 0 / 0.798 |

Best partial: hold offset 0 kept 8 modes at hq ≥ 0.90 for 1151 checks, then one mode left. Stay never beat 59/120 (offset 606; baseline on that offset is 53/120). Ring modes at step 1200 are lower than baseline on seven offsets.

The generator is scored on a critic that has already taken five Adam steps on the batch it is about to move. That reaction does what the hop hypothesis says once a full cover exists: offset 0 held for 1151 checks and lost the hold when live modes went 8→7, and offset 707 lost it when hq slipped to 0.899 with all 8 modes still present. The same reaction is why the ring, which stops at step 1200, is worse. Baseline offset 0 already has 8 modes there (hq 0.878); unrolled offset 0 has 0 modes there and only qualifies at step 2109. Anticipating D makes the early jump onto an empty mode less attractive, so acquisition slips past the ring gate, and the hold that does form still ends before 1200 checks. Stay stays broken (best 59/120). Unequal loses both baseline passes. This does not hand off to the A6000.

### Proof

`python3 -m pytest tests/test_constant_lr_dynamics.py` — 13 passed. Flag unset: `test_flag_off_skips_unroll_and_hashes_match` points `_unroll` at a raiser, runs `train_mode_hold` twice for 2 steps, gets the same `(modes, hq)`, and the raiser is never called. `test_trainer_flag_off_hash_matches_and_unrolled_is_deterministic` hashes G, D, the particle prior, and both Adam states after two `GANTrainer` steps: two flag-off runs match, two `unrolled` runs match, and the two hashes differ. `test_unrolled_mode_hold_is_deterministic` matches two flagged 2-step runs. `test_functional_adam_matches_single_tensor_adam` is `torch.equal` to single-tensor Adam, including a second step with `β1 = 0.1`. `test_unroll_leaves_real_discriminator_and_changes_generator_grad` keeps the real D parameters and Adam state bit-identical across the generator backward, and the generator gradient is finite and differs from the plain critic score. `test_acquire_before_latch_matches_flag_off` matches two-step `train_mode_hold` and two-step `GANTrainer` hashes with the flag unset. `test_acquire_latches_once_on_the_probe_cadence` latches only when `modes == n_modes` on a multiple of 50, and keeps that step. `test_acquire_does_not_unroll_until_latched` does not call `_unroll` before the latch.

## Unrolled after acquire

`K3P_DYNAMICS=unrolled_after_acquire`. Learning rates stay at the constant screen schedule. Generator steps use the plain baseline critic until the latch, then unrolled `k=5` for every later step.

Condition, declared before the run (`particlegan/dynamics/unrolled_after_acquire.py`): the live mode/hq detector is `mode_hold.diversity`. A target mode is occupied when at least one sample lies inside its 3σ ball. The latch fires at the first probe checkpoint on the normal cadence where `modes == n_modes`. The cadence is completed update `step % 50 == 0` (the ring recorder's 24 points on a 1200-step run, and hold's `diagnostic_every`). The hq fraction is not a second threshold. The update that produced the checkpoint stays on baseline dynamics. No step index is a switch time, and the losses have no per-mode term.

CPU screen for this build is recorded after the run, including whether plain `unrolled` hold at offsets 0 and 707 matches the previous screen.
