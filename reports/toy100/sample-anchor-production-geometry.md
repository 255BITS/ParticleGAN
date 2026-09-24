# First-bank grouping on the frozen 100-Gaussian tasks

The [offline first-bank check](sample_anchor_production_geometry.py) uses the production [sampler](../../benchmarks/toy100/problems.py) and [configuration](../../configs/toy100/constraints_simple_regularization.json): seed 1234, one native 2,048-real minibatch, 20,000 learned particles, and target standard deviation 0.03. The production [training loop](../../benchmarks/toy100/train.py) creates an independent data generator at that seed and draws its first real batch before the first update. Evaluator centers enter this diagnostic only after the unlabeled bank has been grouped.

| Task | True groups observed | MST-inferred groups | Fewest / most draws per group | MST within / between edge | Diagnostic purity |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid100 | 100/100 | 100 | 11 / 31 | 0.0749 / 0.7943 | 1.0 |
| rotated100 | 100/100 | 100 | 11 / 31 | 0.0749 / 0.8030 | 1.0 |
| staggered100 | 100/100 | 100 | 11 / 31 | 0.0749 / 0.8069 | 1.0 |

The hypothesis that the **first** production batch omits target groups is false for these three fixed tasks. The exact draws, post-bank RNG hashes, source hashes, MST gaps, and counts are in the [receipt](continuous-evidence/round6-sample-anchor-production-geometry/first-bank.json). This one-bank result does not certify grouping on every later stochastic batch or any full training behavior.

The current neural realization cannot be transplanted unchanged. Production uses an affine generator with six parameters and a 20,000×2 learned prior. `fit_output_targets` explicitly constructs a dense clean-output Jacobian of shape 40,000×40,006 before a thin SVD: the double-precision Jacobian alone needs 12.8 GB, with further comparably sized SVD factors. The 12-particle ring helper is therefore a feasibility demonstration, not a practical 20,000-particle implementation. A structured solver would be a separate method with its own parity and stability checks.

There is also a fidelity gap independent of compute. The sample-anchor objective is zero whenever each inferred group has at least one particle at its centroid and all other particles sit on any centroid. It does not constrain the remaining 19,900 particles' mode weights. For example, counts 19,901 in one group and one in each other group have zero anchor loss but nearest-mode total variation about 0.985, far beyond the production accuracy gate's 0.10 bound. The rule can prove centroid coverage in its fixed-center free-output setting without proving the [production mass and spread gates](../../benchmarks/toy100/metrics.py) or indefinite GAN stability.
