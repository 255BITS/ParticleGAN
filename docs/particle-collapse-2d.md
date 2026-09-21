# CPU 2D particle collapse repro

**Not the default.** This toy shows that pure RpGAN + sample-point `b_cap`
**FAILS** after imitation. The default particle gym recipe is YuE2 paired-error
RpGAN at `adv_weight=1`
([fine-tune note](gym-particle-finetune.md),
[PR #18](https://github.com/255BITS/ParticleGAN/pull/18)).

The parent imitates a linear expert. The baseline then drops that paired loss
and trains with the collapsed particle game: Rp logistic GAN, sample-point
`b_cap`, a shared action head, and a particle cloud at 100× the head learning
rate. Action MSE stays high and landings drop.

Closed PR #15 continued from the same toy with adversarial weight 0. That
setting configures the GAN and does not apply it. It is not shipped here.

```bash
python -u examples/particle_control_2d.py
```

CPU gate (asserts the baseline fails):

```bash
python -m unittest tests.test_particle_control_2d
```

Lines use the prefix `[particle-2d]`. A reproduced collapse prints
`GATE COLLAPSE` and exits 0. That exit 0 is the test passing, and it means
the baseline **FAIL**ed. `GATE NO_COLLAPSE` exits 1.
