# Constant-LR GAN dynamics

Three mechanisms for the toy100 probes at K3P's pre-anneal rates, held constant: network `0.00425`, prior `0.0085` (`prior_lr_mult` 2). `lr_anneal_start` is 0 and both floors are 1, so `policy_multipliers` stays 1 at steps 0, 720, 721, 1199, 2400, and 4000. The horizon cap stays 1600. No decay, anneal, or warm restart.

CPU numbers below do not rank against the A6000. `--init hid_q` fixes the weights. Offset 0 is the unshifted repo seed. The other offsets (`101 202 303 404 505 606 707`) change samples and noise only, via `K3P_SEED_OFFSET`.

Flag: `K3P_DYNAMICS` in `{unit_rms, pair_chord, shared_batch, extragradient}`. Unset is the constant-LR K3P baseline. `sitecustomize.py` in this directory installs the named mechanism before the probe captures `Adam.step`. The screen puts this directory on `PYTHONPATH`.

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

Each file applies alone on develop `407c2f7a` (`git apply`). They are not meant to be stacked: `__init__.py` is shared, and each patch's `sitecustomize.py` installs only that mechanism. This branch's `sitecustomize.py` installs whichever of the three names, or `extragradient`, is set.

- `patches/unit_rms.patch`
- `patches/pair_chord.patch`
- `patches/shared_batch.patch` (also the `mode_hold` host and `GANTrainer.step`)

## GPU commands

A6000, eight offsets, constant LR. One process per mechanism so a failure does not share a job slot with the others. `unequal` is included. The extragradient CPU screen is the reason not to spend a GPU on that dynamic.

```
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics baseline --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics unit_rms --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics pair_chord --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics shared_batch --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics extragradient --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
```

Pass rules the screen prints: ring and unequal use the probe verdict (`modes`, `hq`, `passing_suffix`). Hold passes only when `status` is `PASS` and `hold_checks == 1200`. Stay passes only when shift's continued hold is `120/120` and `pass_all`. The shift terminal status is not the stay gate.

`scaled_penalty` now accepts `ema_critic`. A penalty with no `arm` (the current K3P penalty) delegates to the function this replacement captured, so `vector_unequal_mass` runs. Ring, hold, and stay still use the legacy penalty.

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

## extragradient

`K3P_DYNAMICS=extragradient`. Simultaneous Extra-Adam (Gidel et al. 2019): at the current point, differentiate the critic, the generator, and the particles together; take one Adam step at each group's current learning rate; differentiate again at that joint point on the same minibatch; apply those gradients with Adam from the original point. The extrapolation's moments are not kept. One committed Adam update per player per outer step, so the hold and stay probes still see one moment update per step. Both half-steps use the group rates already written for that update (0.00425 and 0.0085 on this screen). No new coefficient.

Declared before the run and not used: extrapolation from the past (reuse the previous gradient as the lookahead). That drops the extra backward. The hypothesis is that a fresh lookahead of the current field cancels the rotation that walks a mode onto its neighbor, so a stale gradient is a different method.

`vector_unequal_mass` was crashing before any of these dynamics: the probe assigns `scaled_penalty` onto the K3P penalty, which now receives `ema_critic` and has no `arm`. Calls with no `arm` go to the penalty that was replaced. At constant LR that penalty stays in its `s == 1` form.

### CPU screen (this build)

64 jobs, 4 at a time. Baseline ring ~16 s, hold ~100 s, stay ~45 s, unequal ~33 s. Extragradient is about twice that (ring ~36 s, hold ~200 s, stay ~105 s, unequal ~80 s), which is the extra backward. Group LRs stayed `0.00425` and `[0.00425, 0.0085]`. These numbers are not an A6000 ranking. The handoff bar is missed: no hold PASS and no stay 120/120. Ring passes are 0/8, the same count as the baseline, so the miss is the stay, not a lost ring.

| offset | baseline ring | EG ring | baseline hold | EG hold | baseline stay | EG stay | baseline unequal | EG unequal |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 8 / 0.878 / 0 | 0 / 0.000 / 0 | 0 | 0 | 33/120 | 0/120 | FAIL 0.968 / 2 | FAIL 0.001 / 0 |
| 101 | 7 / 0.659 / 0 | 0 / 0.000 / 0 | 0 | 0 | 0/120 | 0/120 | PASS 0.971 / 5 | FAIL 0.000 / 0 |
| 202 | 7 / 0.985 / 0 | 0 / 0.000 / 0 | 0 | 0 | 0/120 | 0/120 | FAIL 0.984 / 0 | FAIL 0.000 / 0 |
| 303 | 6 / 0.670 / 0 | 0 / 0.000 / 0 | 0 | 0 | 9/120 | 0/120 | FAIL 0.977 / 0 | FAIL 0.007 / 0 |
| 404 | 6 / 0.400 / 0 | 0 / 0.000 / 0 | 0 | 0 | 17/120 | 0/120 | FAIL 0.912 / 3 | FAIL 0.000 / 0 |
| 505 | 4 / 0.263 / 0 | 0 / 0.000 / 0 | 0 | 0 | 0/120 | 0/120 | PASS 0.982 / 10 | FAIL 0.000 / 0 |
| 606 | 7 / 0.694 / 0 | 0 / 0.000 / 0 | 0 | 0 | 53/120 | 0/120 | FAIL 0.957 / 0 | FAIL 0.000 / 0 |
| 707 | 0 / 0.000 / 0 | 0 / 0.000 / 0 | 0 | 0 | 0/120 | 0/120 | FAIL 0.974 / 0 | FAIL 0.000 / 0 |

Ring cells are modes / hq / passing suffix. Hold cells are `hold_checks` (status `NOT_CONVERGED` on every row). Unequal cells are status, hq, passing suffix. Baseline ring and stay repeat the earlier constant-LR sanity, including the best stay of 53/120 at offset 606. Extragradient's best stay is 0/120. Every EG ring checkpoint from step 50 through 1200 is 0 modes.

On offset 0 the committed critic gradient RMS is 0.00785 at step 1 (baseline local field 0.00358) and 0.027 at step 100, with displacement RMS 0.009, larger than the learning rate. At step 1200 that gradient RMS is 0.99 and the displacement has fallen to 0.00037.

## Recommendation

Do not send `extragradient` to the A6000. The lookahead did not cancel a neighbor hop: no mode was ever occupied, so there was nothing to rotate off of. Extra-Adam extrapolates by a full Adam step at 0.00425 / 0.0085. The gradient applied from the origin is the field at that point, which is already a different game (step-1 critic gradient about 2× the local field, and a step larger than `lr` by update 100). The hop in #177 is a late crossing between two occupied modes. This step never places a particle on a mode. The late collapse of the displacement is the lagged second moment absorbing that blow-up, the same memory #177 described, reached with an empty ring.

Of the earlier three, still do not promote any. `shared_batch` is the only CPU ring pass, and it does not hold. Baseline unequal, now that the penalty signature is fixed, passes 2/8 offsets. That gate is not the handoff.
