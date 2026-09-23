# Smooth critic research architectures

`SmoothFourierCritic` changes the vector discriminator's activation and optionally its Fourier projection. It accepts no target statistics, mode centers or data-derived features. The original generator, GAN loss, particle prior and training recipe can be used unchanged.

The `axis_softplus5` architecture provides an unequal-width training witness with the original vector recipe and budget: live PASS for the final 7 of 24 observations, HQ .98706, component covariance error .41357 and minimum eigenvalue ratio .28064. It is not a universal discriminator: its complete six-toy profile passes broad modes, unequal width and spiral, and fails rare mass, anisotropic and sustained overlap.

```python
from benchmarks.transfer_suite.smooth_critic_research import (
    ARCHITECTURES, SmoothFourierCritic,
)

architecture = next(c for c in ARCHITECTURES if c["name"] == "axis_softplus5")
critic = SmoothFourierCritic(
    in_dim=2, hidden_dim=64, n_hidden=2, fourier=2,
    architecture=architecture,
)
```

The axis version retains the original feature order, frequencies (`pi`, `2*pi`), layer initialization order and parameter count (4,929). Only the activation changes. Oriented versions replace the projection with predeclared radial frequencies and fixed random directions from a local seed-0 generator; they do not consume the global initialization RNG. Four projections retain 4,929 parameters; eight projections use 5,441. Buffers, including the projection, are saved in `state_dict`.

For the existing **serial** vector runner, the constructor factory supplies the discriminator with its usual signature:

```python
from benchmarks.transfer_suite import vector_tasks
from benchmarks.transfer_suite.smooth_critic_research import constructor

spec = next(s for s in vector_tasks.TASKS if s["name"] == "vector_unequal_width")
spec = dict(spec, research_discriminator=architecture)
original = vector_tasks.SimpleMLPDiscriminator
vector_tasks.SimpleMLPDiscriminator = constructor(architecture)
try:
    result = vector_tasks.run_episode(
        spec, vector_tasks.fixed_policy("cosine"), fixed=True,
    )
finally:
    vector_tasks.SimpleMLPDiscriminator = original
```

This research adapter temporarily replaces a module-level constructor; run one episode at a time in that process. The episode spec records the architecture explicitly. The audited screen used seed 0, original 256 particles, batch 128, Adam(0,.99), G/D/prior LRs .001/.0015/.01, Rp logistic, b_cap coefficient 3 / kappa 1.25, prior regularization .05, no particle L2 and unchanged behavioral thresholds. Width uses its original 1,200 outer steps; spiral uses 1,600. EMA is reported separately.

Focused tests verify compatibility with an active gradient penalty, original axis feature/initialization parity, local projection RNG isolation and state-dict restoration:

```sh
python -m pytest -q tests/test_smooth_critic_research.py
```
