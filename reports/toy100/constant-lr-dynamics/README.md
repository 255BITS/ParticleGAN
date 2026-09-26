# Constant-LR GAN dynamics

Three mechanisms for the toy100 probes at K3P's pre-anneal rates, held constant: network `0.00425`, prior `0.0085` (`prior_lr_mult` 2). `lr_anneal_start` is 0 and both floors are 1, so `policy_multipliers` stays 1 at steps 0, 720, 721, 1199, 2400, and 4000. The horizon cap stays 1600. No decay, anneal, or warm restart.

CPU numbers below do not rank against the A6000. `--init hid_q` fixes the weights. Offset 0 is the unshifted repo seed. The other offsets (`101 202 303 404 505 606 707`) change samples and noise only, via `K3P_SEED_OFFSET`.

Flag: `K3P_DYNAMICS` in `{unit_rms, pair_chord, shared_batch, ema_g, ema_g_fake}`. Unset is the constant-LR K3P baseline. `sitecustomize.py` in this directory installs the named mechanism before the probe captures `Adam.step`. The screen puts this directory on `PYTHONPATH`.

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

## Patches

Each file applies alone on develop `407c2f7a` (`git apply`). They are not meant to be stacked: `__init__.py` is shared, and each patch's `sitecustomize.py` installs only that mechanism. This branch's `sitecustomize.py` installs whichever of the three names is set.

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
```

Pass rules the screen prints: ring and unequal use the probe verdict (`modes`, `hq`, `passing_suffix`). Hold passes only when `status` is `PASS` and `hold_checks == 1200`. Stay passes only when shift's continued hold is `120/120` and `pass_all`. The shift terminal status is not the stay gate.

`vector_unequal_mass` does not run. The probe always imports `mechanism.py`, which replaces `GradientPenalty.penalty` with `scaled_penalty`. That function does not accept `ema_critic` and reads `self.arm`. The current penalty has no `arm`, and `CriticPenalty` always passes `ema_critic`. A direct call raises `TypeError: scaled_penalty() got an unexpected keyword argument 'ema_critic'`. The same error hits the constant-LR baseline. Ring, hold, and stay use the legacy penalty and are not on this path.

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

Do not promote any of the three. The constant-LR baseline still fails ring, hold, and stay on 8/8, which repeats #178. `unit_rms` removes the sign kink and the lagged 10× step (displacement stays at `lr` while the gradient moves) and coverage gets worse, so that kink was not what was holding the modes. `pair_chord` penalizes the open segment and also covers fewer modes than the baseline. `shared_batch` is the only one that passes a CPU ring at all, and it does not hold. A GPU ranking, if one is run, should start from `shared_batch` against this same constant-LR baseline. It should not be read as a recipe that stays. `unequal` is blocked on the probe before any of these dynamics matter.

## `ema_g` — average the generator and the particles

Yazici et al. 2019, "The Unusual Effectiveness of Averaging in GAN Training". At a constant learning rate the iterates can orbit a good equilibrium; the average is the point that sits on it. `K3P_DYNAMICS=ema_g` keeps an exponential moving average of the generator parameters and the particle table at decay 0.999 (the published figure used here, not swept). Live Adam updates are unchanged. The probe's own gate stays on the live weights. The screen scores the average from `ema_g_scores.json` and prints it in a separate column. A pass that the live model misses is labeled `EMA-scored`.

Declared alternative, stated before it was run: `K3P_DYNAMICS=ema_g_fake`. The same average is also the fake the critic trains on. If the critic only sees the orbiting iterate, it keeps scoring the moving point and never has to defend the center. The generator step stays on the live weights. Decay stays 0.999. This is not the already-failed critic-fake average at decay 0.995. No new coefficient.

`vector_unequal_mass` used to crash: the probe's `scaled_penalty` rejected `ema_critic` and read `self.arm` on the current K3P penalty, which has no arm. That call now goes to the K3P penalty unchanged, so unequal runs.

Flag off is the constant-LR baseline. An 8-step ring host with the screen's noise policy hashes G, D, and the particles to `558cbe0de8d87faf4269e1bfb67b6f0fc61277efeeb7c70f0c7eff909b2d405d` with the flag unset, again with the flag unset, and twice with `ema_g`. The two `ema_g` live/EMA curves match each other. `tests/test_ema_g.py` is that proof.

```
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics baseline --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics ema_g --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics ema_g_fake --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
```

CPU screen on this build, 8 offsets, 4 jobs. These numbers do not rank against the A6000. Baseline offset 0 ring is 8 modes, hq 0.878, suffix 0, the same constant-LR point as #178. Best baseline stay is 53/120 at offset 606, the same partial as #183.

`ema_g` live matches that baseline on every ring, hold, stay, and unequal cell. The averaged column does not. `ema_g_fake` is the declared alternative and is worse. No hold PASS and no stay 120/120. The handoff bar is missed.

Ring, modes / hq / passing suffix. No ring PASS.

| offset | baseline | ema_g live | ema_g EMA | ema_g_fake live | ema_g_fake EMA |
| ---: | --- | --- | --- | --- | --- |
| 0 | 8 / 0.878 / 0 | 8 / 0.878 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 |
| 101 | 7 / 0.659 / 0 | 7 / 0.659 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 |
| 202 | 7 / 0.985 / 0 | 7 / 0.985 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 |
| 303 | 6 / 0.670 / 0 | 6 / 0.670 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 |
| 404 | 6 / 0.400 / 0 | 6 / 0.400 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 |
| 505 | 4 / 0.263 / 0 | 4 / 0.263 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 |
| 606 | 7 / 0.694 / 0 | 7 / 0.694 / 0 | 0 / 0.000 / 0 | ERROR | 0 / 0.000 / 0 |
| 707 | 0 / 0.000 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 | 0 / 0.000 / 0 |

Hold. Every finished run is `NOT_CONVERGED` (`hold_checks` 0). `ema_g_fake` offset 606 is `ERROR` on the live record; the averaged column is still `NOT_CONVERGED`.

Stay, passing checks out of 120. None is 120/120.

| offset | baseline | ema_g live | ema_g EMA | ema_g_fake live | ema_g_fake EMA |
| ---: | --- | --- | --- | --- | --- |
| 0 | 33/120 | 33/120 | 0/120 | 0/120 | 0/120 |
| 101 | 0/120 | 0/120 | 0/120 | 0/120 | 0/120 |
| 202 | 0/120 | 0/120 | 0/120 | 0/120 | 0/120 |
| 303 | 9/120 | 9/120 | 0/120 | 0/120 | 0/120 |
| 404 | 17/120 | 17/120 | 0/120 | 0/120 | 0/120 |
| 505 | 0/120 | 0/120 | 0/120 | 0/120 | 0/120 |
| 606 | 53/120 | 53/120 | 0/120 | ERROR | 0/120 |
| 707 | 0/120 | 0/120 | 0/120 | 0/120 | 0/120 |

Unequal, hq / passing suffix. The bugfix lets the gate run. Passes are the live model only, at offsets 101 and 505, for baseline and for `ema_g` (same cells). The average does not pass. `ema_g_fake` does not pass.

| offset | baseline | ema_g live | ema_g EMA | ema_g_fake live | ema_g_fake EMA |
| ---: | --- | --- | --- | --- | --- |
| 0 | 0.968 / 2 | 0.968 / 2 | 0.014 / 0 | 0.000 / 0 | 0.054 / 0 |
| 101 | 0.971 / 5 PASS | 0.971 / 5 PASS | 0.019 / 0 | 0.000 / 0 | 0.098 / 0 |
| 202 | 0.984 / 0 | 0.984 / 0 | 0.002 / 0 | 0.000 / 0 | 0.045 / 0 |
| 303 | 0.977 / 0 | 0.977 / 0 | 0.014 / 0 | 0.000 / 0 | 0.260 / 0 |
| 404 | 0.912 / 3 | 0.912 / 3 | 0.016 / 0 | 1.000 / 0 | 0.000 / 0 |
| 505 | 0.982 / 10 PASS | 0.982 / 10 PASS | 0.005 / 0 | 0.000 / 0 | 0.221 / 0 |
| 606 | 0.957 / 0 | 0.957 / 0 | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 |
| 707 | 0.974 / 0 | 0.974 / 0 | 0.016 / 0 | 0.000 / 0 | 0.010 / 0 |

`ema_g_fake` offset 606 ring, hold, and stay write a non-finite value, so the probe rejects the JSON (`allow_nan=False`). By step 1000 on that ring the live generator displacement is 0 and the gradient RMS is ~1e-19. The scored curve up to the failure is 0 modes.

Best averaged coverage anywhere: `ema_g` hold, 4 modes, hq 0.238, offset 606, step 6220. No averaged checkpoint has 8 modes and hq 0.90 (streak 0 on ring, hold, and stay, both variants).

Decay 0.999 still weights the initial parameters by about 0.30 after 1200 steps and about 0.027 after 3600. The ring average is still mixed with the transient, which is why it has 0 modes while the live model still shows the baseline's late coverage. That is not the whole miss. At the stay and the hold the initial weight is gone, and the average still never qualifies. The live constant-LR path is not orbiting a covered equilibrium, so the average has no covered center to sit on. Feeding that average to the critic (`ema_g_fake`) does not create one; it drops the live coverage and can go non-finite.

Do not send either variant to the A6000 as a stay recipe.
