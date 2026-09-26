# Constant-LR GAN dynamics

Four mechanisms for the toy100 probes at K3P's pre-anneal rates, held constant: network `0.00425`, prior `0.0085` (`prior_lr_mult` 2). `lr_anneal_start` is 0 and both floors are 1, so `policy_multipliers` stays 1 at steps 0, 720, 721, 1199, 2400, and 4000. The horizon cap stays 1600. No decay, anneal, or warm restart.

CPU numbers below do not rank against the A6000. `--init hid_q` fixes the weights. Offset 0 is the unshifted repo seed. The other offsets (`101 202 303 404 505 606 707`) change samples and noise only, via `K3P_SEED_OFFSET`.

Flag: `K3P_DYNAMICS` in `{unit_rms, pair_chord, shared_batch, sga}`. Unset is the constant-LR K3P baseline. `sitecustomize.py` in this directory installs the named mechanism before the probe captures `Adam.step`. The screen puts this directory on `PYTHONPATH`.

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

### `sga` — drop the rotational part of the D/G/particle game

Balduzzi et al. 2018, Algorithm 1. One published setting, written before the screen: `|λ| = 1` and the paper's sign rule

```
λ = sign( (1/d) ⟨ξ, ∇H⟩ ⟨Aᵀ ξ, ∇H⟩ + 1e-10 )
```

`ξ` stacks `∇_D L_D`, `∇_G L_G`, and `∇_z L_G` at one parameter point (the generator loss is built before the critic step, on the host's sample order). `Aᵀ ξ = (Jᵀ ξ − J ξ) / 2` from Hessian-vector products. The adjusted vector is what the existing Adam consumes, at network `0.00425` and prior `0.0085`. No clip and no second coefficient.

`vector_unequal_mass` was crashing because the probe's `scaled_penalty` rejected `ema_critic` and read `self.arm` on the K3P penalty, which has neither. That call now delegates to the original K3P penalty. Ring, hold, and stay still use the legacy penalty.

## Patches

Each file applies alone on develop `407c2f7a` (`git apply`). They are not meant to be stacked: `__init__.py` is shared, and each patch's `sitecustomize.py` installs only that mechanism. This branch's `sitecustomize.py` installs whichever of the four names is set.

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
python -u reports/toy100/constant-lr-dynamics/screen.py --backend cuda --dynamics sga --gates ring hold shift unequal --offsets 0 101 202 303 404 505 606 707
```

Pass rules the screen prints: ring and unequal use the probe verdict (`modes`, `hq`, `passing_suffix`). Hold passes only when `status` is `PASS` and `hold_checks == 1200`. Stay passes only when shift's continued hold is `120/120` and `pass_all`. The shift terminal status is not the stay gate.

`vector_unequal_mass` runs on this branch. The probe's `scaled_penalty` accepts `ema_critic`. On the K3P penalty (no `.arm`) it delegates to the original penalty, so the penalty value is unchanged. Legacy penalties that still have `.arm` keep their formula. Ring, hold, and stay use the legacy penalty and never call this function.

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

Do not promote any of the three. The constant-LR baseline still fails ring, hold, and stay on 8/8, which repeats #178. `unit_rms` removes the sign kink and the lagged 10× step (displacement stays at `lr` while the gradient moves) and coverage gets worse, so that kink was not what was holding the modes. `pair_chord` penalizes the open segment and also covers fewer modes than the baseline. `shared_batch` is the only one that passes a CPU ring at all, and it does not hold. A GPU ranking of those three, if one is run, should start from `shared_batch` against this same constant-LR baseline. It should not be read as a recipe that stays.

## CPU screen: `sga`

Same build, baseline and `sga`, 64 jobs, 2 at a time. Baseline ring, hold, and stay match the table above (ring 0/8, hold 0/8, stay 0/8, best stay 53/120 at offset 606). Unequal now runs. The screen's unequal row leaves `modes` null; status, hq, and passing suffix are the fields it prints.

Handoff bar missed. The bar is one hold PASS or one stay 120/120, with at least as many ring passes as the baseline. `sga` ring passes 0/8, hold passes 0/8, stay 120/120 is 0/8. Best recorded `sga` stay is 0/120. Completed rings are 0 modes and hq 0.

| offset | baseline ring | sga ring | baseline hold | sga hold | baseline stay | sga stay | baseline unequal | sga unequal |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 8 / 0.878 / 0 FAIL | 0 / 0.000 / 0 FAIL | NOT_CONVERGED, 0 | NOT_CONVERGED, 0 | 33/120 | 0/120 | FAIL 0.968 / 2 | FAIL 0.027 / 0 |
| 101 | 7 / 0.659 / 0 FAIL | 0 / 0.000 / 0 FAIL | NOT_CONVERGED, 0 | NOT_CONVERGED, 0 | 0/120 | 0/120 | PASS 0.971 / 5 | FAIL 0.000 / 0 |
| 202 | 7 / 0.985 / 0 FAIL | 0 / 0.000 / 0 FAIL | NOT_CONVERGED, 0 | ERROR | 0/120 | 0/120 | FAIL 0.984 / 0 | FAIL 0.000 / 0 |
| 303 | 6 / 0.670 / 0 FAIL | 0 / 0.000 / 0 FAIL | NOT_CONVERGED, 0 | NOT_CONVERGED, 0 | 9/120 | 0/120 | FAIL 0.977 / 0 | FAIL 0.000 / 0 |
| 404 | 6 / 0.400 / 0 FAIL | 0 / 0.000 / 0 FAIL | NOT_CONVERGED, 0 | NOT_CONVERGED, 0 | 17/120 | 0/120 | FAIL 0.912 / 3 | FAIL 0.003 / 0 |
| 505 | 4 / 0.263 / 0 FAIL | ERROR | NOT_CONVERGED, 0 | ERROR | 0/120 | ERROR | PASS 0.982 / 10 | FAIL 0.000 / 0 |
| 606 | 7 / 0.694 / 0 FAIL | 0 / 0.000 / 0 FAIL | NOT_CONVERGED, 0 | NOT_CONVERGED, 0 | 53/120 | 0/120 | FAIL 0.957 / 0 | FAIL 0.000 / 0 |
| 707 | 0 / 0.000 / 0 FAIL | 0 / 0.000 / 0 FAIL | NOT_CONVERGED, 0 | ERROR | 0/120 | ERROR | FAIL 0.974 / 0 | FAIL 0.000 / 0 |

Ring cells are modes / hq / passing suffix. Hold cells are status and `hold_checks`. Unequal cells are status, hq, passing suffix. ERROR is the probe refusing to write `result.json` because a metric was NaN, after training had already finished (`sga` receipt `steps` 1200 on the ring, 6300 on the hold, 3600 on the shift). Those hold logs end at step 6300 with `NOT_CONVERGED`, `hold_checks` 0, live modes 0. The two shift logs checkpoint 0 modes from step 2500 through 3600, so they are not a 120/120 stay. Ring offset 505 finished 1200 steps and then hit the same NaN write; it is not a ring pass.

`|λ| = 1` is what the paper plugs into gradient descent. Under this Adam (`β2 = 0.999`) the correction dominates `ξ`. On the offset-0 hold, step 200 already has generator gradient RMS ~178 and an alignment product ~1e20, while both learning rates are still 0.00425 and 0.0085. Displacement RMS falls to ~1e-4 and live modes stay 0 for the rest of the budget. Later alignments go to ±inf, which is the NaN the probe then refuses to serialize. The rotational term never gets a chance to stop a neighbor hop: the particles do not acquire the modes. No second coefficient was run.

Do not send `sga` to the A6000 on this bar.
