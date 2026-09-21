# CPU 2D particle collapse repro

This toy reproduces the particle-control collapse. It is not a Lunar Lander
solution, and it does not change the gym particle trainer.

The parent imitates a linear expert. The baseline then drops that paired loss
and trains with the particle arm's game: Rp logistic GAN, sample-point `b_cap`,
a shared action head, and a particle cloud at 100× the head learning rate.
That baseline is expected to **FAIL** (action MSE stays high, landings drop).

Closed PR #15 continued from the same toy with adversarial weight 0. That
setting configures the GAN and does not apply it. It is not shipped here, and
`adv_weight: 0` is not the particle-arm default. The gym path stays the
adversarial recipe on `particle-finetune/base`.

```bash
python -u examples/particle_control_2d.py
```

CPU gate (asserts the baseline fails):

```bash
python -m unittest tests.test_particle_control_2d
```

Lines use the prefix `[particle-2d]`. A reproduced collapse prints
`GATE COLLAPSE` and exits 0. `GATE NO_COLLAPSE` exits 1.
