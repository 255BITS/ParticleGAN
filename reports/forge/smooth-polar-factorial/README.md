# Fixed smoothing across the four runtime combinations

Preregistered user-requested diagnostic, based on develop
`83b099d1e4330dda953d5fce4f68ce00f75fa9a6`. This study tests the
[smoothed matrix-polar rule](https://arxiv.org/html/2608.01911v1) on the
Gaussian and word regression tasks, with Ring16 as a protection check.
The [protocol](protocol.json) freezes every budget, scale, runtime combination,
prediction and original control identity before training.

One absolute scale `lambda=1e-4` is used across all tasks, roles and updates;
the paper's `epsilon=lambda**2=1e-8`. Matrix gradients use singular weights
`s / sqrt(s*s + lambda*lambda)`. Bias gradients and actually sampled prior
rows use the corresponding vector rule. Truncated arms additionally retain
the existing numerical rank mask. Full arms smooth every singular direction.
The prior/bias extensions are an explicit adaptation; the paper studies
continuous-time minimization rather than discrete stochastic adversarial games.

The four existing truncation/autograd combinations retain their original
settings. Only smoothing changes relative to each matching unsmoothed arm.
Recipe rates, initialization, task law, actual batch sequence, prior, sampling,
update budgets, observations and gates are held fixed. Archived controls are
reused under their original identities and are not retrained.

This is a nonqualifying task diagnostic. The word task retains its original
five-terminal-check gate. The qualification inventory, task declarations,
project scheduling policy and default optimizer behavior remain unchanged.
`Recipe.optimizer_smoothing` is opt-in and defaults to zero; checkpoint packets
record nonzero scales and reject mismatches. Historical scheduling/truncation
overrides exist only within the diagnostic runner process.

## Execution

The two independent controllers reserve 5,280 seconds in total for at most
12 runs / 90,404 logical training updates. Each arm executes once. No scale
search, seed study, retry, continuation or gate change is included.

```sh
mkdir -p runs/forge/smooth-polar-factorial-v1
/home/martyn/dev/ParticleGAN/.venv/bin/python -u \
  reports/forge/smooth-polar-factorial/controller.py --slot 0 \
  > runs/forge/smooth-polar-factorial-v1/controller-0.log 2>&1
/home/martyn/dev/ParticleGAN/.venv/bin/python -u \
  reports/forge/smooth-polar-factorial/controller.py --slot 1 \
  > runs/forge/smooth-polar-factorial-v1/controller-1.log 2>&1
tail -f runs/forge/smooth-polar-factorial-v1/controller-1.log
```

Controllers run in separate terminals, one worker per A6000. Per-arm logs emit
all scheduled numerical observations and can be tailed directly. Neural
training and numerical software checks require CUDA. Target generation and
scoring retain each task's original implementation and sampling law.
