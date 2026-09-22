# Reproduce the rare-mode fix

The successful 2D discriminator is a 96-wide, two-hidden-layer MLP with
Softplus(beta5), the original two axis Fourier bands, and a learned linear score
on the raw coordinates:

```python
score = smooth_fourier_mlp(x) + linear_without_bias(x)
```

The linear branch starts at zero and adds two parameters. The discriminator has
10,467 parameters in total; the original generator, 256 particles, batch128,
loss, regularization, optimizer settings and 1,200-update budget are unchanged.
The gradient cap differentiates the complete discriminator with respect to its
original inputs. There are no labels, target-derived features or auxiliary losses.

Run just the rare-mixture toy:

```bash
python -u -m benchmarks.transfer_suite.run_linear_skip \
  --tasks vector_unequal_mass --output /tmp/linear-skip-rare \
  > /tmp/linear-skip-rare.log 2>&1
tail -f /tmp/linear-skip-rare.log
```

Omit `--tasks` to run all six data toys. Use a new output directory for each run.
The command saves exact configurations, source hashes and archive, all 24 live
and EMA observations, actions and recomputed verdicts. It logs each completion.

The rare toy passes its final six live checks, confirms at step 1,150 and ends
with sample quality 99.71%, covariance error .4413 and minimum normalized variance
.2698 (required ≥ .15). EMA has four final passing checks and fails the sustained
rule; this is a live-weight success. This same D passes only 3/6 data toys: broad and spiral also
pass; anisotropic, overlap and unequal width fail. Appropriate architectures
for those cases bring the unchanged formulation to 19/19 across the main suite.
These are fixed-seed development results, not a universal architecture or
evidence of real-network transfer. The winning formulation is now the default
`get_recipe()`. The matching optional public discriminator is
`particlegan.LinearSkipDiscriminator()`; architectures and resources stay caller-owned.
[Public-default replay](../../reports/transfer_suite/default_promotion/README.md)
verifies the default recipe, trainer and critic together.

[Implementation](linear_skip_refinement_research.py) ·
[Leaderboard and full profiles](../../reports/transfer_suite/rare_focus/README.md) ·
[Main formulation comparison](../../reports/transfer_suite/formulations/README.md).
