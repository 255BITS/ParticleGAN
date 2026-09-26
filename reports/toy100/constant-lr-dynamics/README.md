# Constant-LR GAN dynamics

Three mechanisms for the toy100 probes at K3P's pre-anneal rates, held constant: network `0.00425`, prior `0.0085` (`prior_lr_mult` 2). `lr_anneal_start` is 0 and both floors are 1, so `policy_multipliers` stays 1 at steps 0, 720, 721, 1199, 2400, and 4000. The horizon cap stays 1600. No decay, anneal, or warm restart.

CPU numbers below do not rank against the A6000. `--init hid_q` fixes the weights. Offset 0 is the unshifted repo seed. The other offsets (`101 202 303 404 505 606 707`) change samples and noise only, via `K3P_SEED_OFFSET`.

Flag: `K3P_DYNAMICS` in `{unit_rms, pair_chord, shared_batch}`. Unset is the constant-LR K3P baseline. `sitecustomize.py` in this directory installs the named mechanism before the probe captures `Adam.step`. The screen puts this directory on `PYTHONPATH`.

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

Machine: CPU torch, 4 threads, `OMP_NUM_THREADS=1`. Ring ~14–16 s. These pass rates are a sanity check that the flag runs and the schedule stays at the pre-anneal LR. They are not an A6000 ranking.

### Offset 0 ring (full 1200 updates)

Group LRs at step 1200 were `0.00425` and `0.0085` on every run. `unit_rms` receipt `steps=2400` (D and G). Its critic displacement RMS at step 1200 was `0.00425` while gradient RMS was `0.0125` (the RMS step, not Adam). `pair_chord` receipt `calls=1200`.

| dynamics | status | modes | hq | passing suffix |
| --- | --- | ---: | ---: | ---: |
| baseline | FAIL | 8 | 0.878 | 0 |
| unit_rms | FAIL | 2 | 0.156 | 0 |
| pair_chord | FAIL | 4 | 0.297 | 0 |
| shared_batch | FAIL | 0 | 0.000 | 0 |

The baseline matches #178's constant-LR offset 0 (FAIL, 8 modes, hq 0.878). All three mechanisms were worse on this one offset.

### Eight offsets, ring / hold / stay

`FULL_GRID` in `/tmp/k3p-clr` (96 jobs). The table is filled when that screen prints `ALL_DONE`.
