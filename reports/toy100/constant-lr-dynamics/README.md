# Constant-LR GAN dynamics

Three mechanisms for the toy100 probes at K3P's pre-anneal rates, held constant: network `0.00425`, prior `0.0085` (`prior_lr_mult` 2). `lr_anneal_start` is 0 and both floors are 1, so `policy_multipliers` stays 1 at steps 0, 720, 721, 1199, 2400, and 4000. The horizon cap stays 1600. No decay, anneal, or warm restart.

CPU numbers below do not rank against the A6000. `--init hid_q` fixes the weights. Offset 0 is the unshifted repo seed. The other offsets (`101 202 303 404 505 606 707`) change samples and noise only, via `K3P_SEED_OFFSET`.

Flag: `K3P_DYNAMICS` in `{unit_rms, pair_chord, shared_batch, optimistic}`. Unset is the constant-LR K3P baseline. `sitecustomize.py` in this directory installs the named mechanism before the probe captures `Adam.step`. The screen puts this directory on `PYTHONPATH`.

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

## `optimistic` — Daskalakis Algorithm 1 at constant LR

Hypothesis, written before the run: the neighbor hop in #177 is a rotational game cycle (the critic scores a neighbor higher, particles climb, the emptied mode stays empty). Optimism damps that cycle at a constant learning rate. The annealed hid_q screen already ran this rule and it was worse there; this run is the same published update with no anneal.

The update is `-lr * (2 m_t - m_{t-1})` on D, G, and the particle prior. `m_t` is the bias-corrected Adam direction, `m_0 = 0`, and `lr` / betas stay the group values (network `0.00425`, prior `0.0085`, betas `(0, 0.999)`). No other coefficient.

`vector_unequal_mass` was crashing in `scaled_penalty` (`ema_critic` TypeError, and `self.arm` on a penalty that has no arm). The signature accepts `ema_critic`. At constant LR, `s` stays 1, so the penalty value matches K3P's full-LR term.

```
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics baseline --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics optimistic --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
```

CPU only, same build as the baseline above. Offset 0 baseline ring is again FAIL, 8 modes, hq 0.878173828125. Step 1 critic displacement RMS is 0.00425 on the baseline and 0.00850 with optimism (the paper's first step, `m_0 = 0`). Group LRs stay 0.00425 and `[0.00425, 0.0085]` through step 1200. Handoff bar is missed: no hold PASS and no stay 120/120. Ring passes stay 0/8, the same count as the baseline.

| offset | baseline ring | optimistic ring | baseline hold | optimistic hold | baseline stay | optimistic stay | baseline unequal | optimistic unequal |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 8 / 0.878 / 0 | 0 / 0.000 / 0 | 0, NOT_CONVERGED | 0, NOT_CONVERGED | 33/120 | 0/120 | FAIL suffix 2 | FAIL suffix 0 |
| 101 | 7 / 0.659 / 0 | 1 / 0.148 / 0 | 0, NOT_CONVERGED | 0, NOT_CONVERGED | 0/120 | 1/120 | PASS suffix 5 | FAIL suffix 1 |
| 202 | 7 / 0.985 / 0 | 4 / 0.295 / 0 | 0, NOT_CONVERGED | 0, NOT_CONVERGED | 0/120 | 0/120 | FAIL suffix 0 | FAIL suffix 2 |
| 303 | 6 / 0.670 / 0 | 8 / 0.537 / 0 | 0, NOT_CONVERGED | 0, NOT_CONVERGED | 9/120 | 0/120 | FAIL suffix 0 | FAIL suffix 0 |
| 404 | 6 / 0.400 / 0 | 1 / 0.008 / 0 | 0, NOT_CONVERGED | 0, NOT_CONVERGED | 17/120 | 5/120 | FAIL suffix 3 | FAIL suffix 3 |
| 505 | 4 / 0.263 / 0 | 2 / 0.115 / 0 | 0, NOT_CONVERGED | 0, NOT_CONVERGED | 0/120 | 0/120 | PASS suffix 10 | FAIL suffix 0 |
| 606 | 7 / 0.694 / 0 | 5 / 0.444 / 0 | 0, NOT_CONVERGED | 0, NOT_CONVERGED | 53/120 | 0/120 | FAIL suffix 0 | FAIL suffix 0 |
| 707 | 0 / 0.000 / 0 | 3 / 0.099 / 0 | 0, NOT_CONVERGED | 0, NOT_CONVERGED | 0/120 | 0/120 | FAIL suffix 0 | FAIL suffix 0 |

Ring column is modes / hq / passing suffix. Every ring row is FAIL. Hold checks are 0 on all 16 runs. Best stay fraction is still the baseline's 53/120 at offset 606. Optimistic's best stay is 5/120 at offset 404. Unequal modes are not a ring count; the probe verdict is PASS/FAIL plus `passing_suffix`. The bugfix lets unequal finish: baseline 2/8, optimistic 0/8.

Two isolated reruns of optimistic ring offset 0 match on all 106 checkpoint tensors (parameters, Adam moments, the saved direction, RNG). `result.json` differs in wall-clock seconds and the probe's optimizer object ids. A three-step parameter hash at lr 0.00425, betas `(0, 0.999)` is `3e2f5ed9aa0d5e826e457911deb1024e2e7bd9ceeb2d93b544e477dde93e3063` with the flag unset (twice, ordinary Adam keys only) and `dc67f3cad4611f84e33f4ee61162451828571086ec5c2e7d111c55a378ccbd83` with the flag on (twice, including `optimistic_prev_direction`).

Do not send this to the A6000. The first step is `-2 lr m_1`, so the sign step the early kink amplifies is twice as large before a previous direction exists to cancel. Offset 0 is already at 0 modes, and the late gradient cosine is +0.93 rather than a cancelled rotation. The hop this was aimed at is not the failure that fires first.

## Recommendation

Do not promote any of these, including `optimistic`. The constant-LR baseline still fails ring, hold, and stay on 8/8, which repeats #178. `unit_rms` removes the sign kink and the lagged 10× step (displacement stays at `lr` while the gradient moves) and coverage gets worse, so that kink was not what was holding the modes. `pair_chord` penalizes the open segment and also covers fewer modes than the baseline. `shared_batch` is the only one that passes a CPU ring at all, and it does not hold. A GPU ranking, if one is run, should start from `shared_batch` against this same constant-LR baseline. It should not be read as a recipe that stays. `optimistic` misses the handoff bar on this CPU screen (no hold PASS, no stay 120/120) and covers fewer modes than the baseline on 6 of 8 offsets. Offset 303 reaches 8 modes at hq 0.537, and offset 707 reaches 3 modes against the baseline's 0; neither passes. `vector_unequal_mass` now runs; on this build the constant-LR baseline passes 2/8 and optimism passes 0/8.
