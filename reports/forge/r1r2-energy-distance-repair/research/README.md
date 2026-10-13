# Saved Ring shape diagnosis and one subsequent hypothesis

Raw energy-U weight 1 restores Ring16 radial spread but worsens the original
covariance and high-quality gates. Both fresh roles retain 16 modes; neither has
a covariance pass among 96 saved observations. This analysis does not change
their grades or introduce another leaderboard. The independent
[density readout](../density-readout.md) retains the full gate interpretation.

| Saved endpoint / proxy | Fresh NS control | Energy-U 1 |
|---|---:|---:|
| Original component covariance error, bound ≤ .85 | 1.556546 | 6.371506 |
| Original minimum eigen ratio, bound ≥ .15 | .094075 | 1.027582 |
| Original HQ, bound ≥ .85 | .888672 | .747559 |
| Mean radial output variance / target variance | .297019 | 8.026238 |
| Mean tangential output variance / target variance | 3.053394 | 4.393926 |
| Mean mapped-row radial variance / target variance | .185024 | 6.617546 |
| Mean mapped-row tangential variance / target variance | .156595 | 1.309586 |
| Mean local-noise radial variance proxy / target variance | .096483 | 1.539857 |
| Mean local-noise tangential variance proxy / target variance | 2.642178 | 2.889288 |

[saved_shape.py](saved_shape.py) uses only certified saved endpoint panels,
G/prior tensors and declared target axes. It computes the exact affine/LeakyReLU
Jacobian of saved G at each saved row and the first-order proxy
`sigma² J Jᵀ`, with the unchanged sigma .1. Component summaries are unweighted
means over nearest-target assignments. Mapped rows and noisy outputs can receive
different assignments; these quantities are **not an exact additive covariance
decomposition**. They reveal endpoint geometry, not historical forces or
reconstructed initialization. Eight mapped energy rows exceed three target
sigmas versus one control row; clipping alone cannot fix the core covariance
errors (1.156282 and 1.051236, both above .85).

The inference is that the extra output objective produced useful radial width
alongside excessive mapped-center dispersion and persistent tangential stretch.
It does not identify whether prior motion or G caused that dispersion. The
single selected next hypothesis is therefore an uncertain ownership contrast:
**freeze initialized latent locations on the exact energy parent**, retaining
energy on G and all original G/E/D objectives. See the new
[finite comparison plan](../../r1r2-energy-prior-freeze-repair/PLAN.md) for the
sole recipe delta, unchanged initialization law, word risk, historical BCAP
counterevidence and strict remaining-budget prerequisite.

[diagnosis.json](diagnosis.json) pins reproduction source, checkpoint/panel,
registration/publication and archived request hashes. It verifies all **1216**
current and frozen-snapshot scientific files against source
`2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12`,
origin commit `67920d9090727bd74adbd331187f0e6f22533e48`. Seven read-only public
contexts accept exactly `prior_update: learned → frozen` after removing the
false `learned_locations` candidate claim. The probe constructs no generative
model or optimizer, performs no draws/backward/updates and preserves ambient
Python/NumPy/Torch CPU RNG; Recipe's parameter-free validator is allowed.
This PASS concerns saved evidence and binding feasibility, not training or
checkpoint integration. The new plan separately requires bounded software
restore fixtures before admission.

Reproduce from the report worktree:

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-energy-distance-repair/research/saved_shape.py
```

Raw stdout and stderr are archived at
`/mnt/ml7tb/ParticleGAN-forge/r1r2-energy-distance-repair-20261012/saved-shape.stdout.json`
and `saved-shape.stderr.log`; the compact JSON is identical to stdout.
