# Read-only MMD witness preflight on critic-floor native100 states

`probe.py` loads the frozen update-7000 grid100, rotated100, and staggered100
checkpoints. It changes no candidate source or checkpoint. Each of eight
blocks per task draws 2,048 sampled prior rows, uses those same row indices in
fit and heldout contexts, independently draws DV12 latent jitter and unlabelled
real samples for the two contexts, and evaluates Gaussian-kernel unbiased MMD².
The kernel width is the median 10th other-neighbor distance in 8,192 separate
real calibration draws (about `.0213`–`.0216`). The generated Gaussian output
noise is integrated analytically, rather than resampled. A separate
real-vs-real MMD² estimates the finite-batch null.

For `k_h(x,y)=exp(-||x-y||²/(2h²))`, the integrated generated/generated
term uses width `h²+2σ²` and amplitude `(h²/(h²+2σ²))^(d/2)`; the
real/generated term uses width `h²+σ²` and amplitude
`(h²/(h²+σ²))^(d/2)`. The self diagonal is excluded in generated/generated
and real/real U-statistics. A central difference of log sigma on block 0
agrees with autograd on all three tasks to within `1.8e-7`.

| Task | Heldout MMD² minus real/real null, mean [95% CI] | G+prior local sign-step heldout ΔMMD², observed [95% CI] | Of that, q-q self term | Of that, real-q attraction |
| --- | ---: | ---: | ---: | ---: |
| grid100 | `+3.94e-5 [-0.39e-5, +8.27e-5]` | `-2.46e-6 [-3.14e-6, -1.77e-6]` | `-1.70e-6` | `-0.75e-6` |
| rotated100 | `+14.41e-5 [+8.81e-5, +20.00e-5]` | `-2.84e-6 [-3.07e-6, -2.61e-6]` | `-2.86e-6` | `+0.02e-6` |
| staggered100 | `+8.51e-5 [+3.23e-5, +13.79e-5]` | `-2.40e-6 [-3.25e-6, -1.55e-6]` | `-2.05e-6` | `-0.35e-6` |

The exact heldout real-q term's 95% interval contains zero for rotated100
and staggered100. Thus the reliable local decrease mainly spreads generated
samples; this preflight does not show that it corrects the target mismatch on
the two failing tasks. The integrated MMD derivative with respect to log sigma
is negative in all eight blocks of **all three** tasks, including the already
passing grid100. Unconstrained sigma descent would widen output noise there.

**Decision:** This local-width MMD witness alone does not justify a full online
candidate. A better data-driven attraction signal or null/settling mechanism
needs a separate preflight. No evaluator centers, component labels, accuracy
scores, or task-specific schedule were used by the objective. The task names
only select each frozen training sampler/checkpoint. The tested G/prior step is
`-current_lr * sign(MMD_gradient)`; it uses current applied LR magnitudes but
does not replay Adam moments or A2 latent damping, so it is a directional
diagnostic, not an exact optimizer update. Sigma is frozen at `.020` in all
three checkpoints; its measured direction is hypothetical.

Detailed independent rows, confidence intervals, checkpoint SHA-256 hashes,
seeds, and bandwidths are in `result.json`. `probe.log` records run progress.

## Paired attribution control

`attribution_control.py` compares a full-MMD sign proposal against a
generated/generated (`q-q`) term-only sign proposal on the **same** frozen
checkpoint samples and prior row indices. The two proposals have exactly
matched Euclidean norms separately in the G and prior parameter groups; both
are scored on the same independent heldout **total** MMD². Each task has eight
paired blocks. The exact zero translation derivative of the q-q term is set
to zero to avoid floating-point sign noise in G's bias.

| Task | Full minus q-q-only heldout ΔMMD², mean [95% CI] | Pairs favoring full | q-q-only Δreal-q term |
| --- | ---: | ---: | ---: |
| grid100 | `-1.56e-6 [-2.69e-6, -0.42e-6]` | `7/8` | `+8.08e-6` |
| rotated100 | `-0.52e-6 [-1.36e-6, +0.32e-6]` | `7/8` | `+6.70e-6` |
| staggered100 | `-0.94e-6 [-1.96e-6, +0.08e-6]` | `7/8` | `+7.67e-6` |

The real term does constrain the full-MMD direction: q-q-only repulsion lowers
its own term by about `9e-6` but worsens real/generated similarity by
`6.7e-6` to `8.1e-6`. The full direction largely avoids that penalty.
Nevertheless, its **paired advantage on total heldout MMD² is unresolved on
both failing tasks** at the prespecified eight-block limit. Under the
required criterion, the local-width witness does not justify an online
candidate. Full rows and matched proposal norms are in
`attribution_result.json`; `attribution.log` records progress. No further
tests were run.
