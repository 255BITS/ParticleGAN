# Round four: projection and local-v2 transport composition

This preregistered four-arm diagnostic compares the exact BCAP winner, the measured
strict-progress projection package, measured transport local-v2, and BOTH on one
frozen scientific source. Full unchanged sustained gates are authoritative. No
ordinary qualification, calibration acceptance or public-default promotion follows.

## Motivation and mechanism

[Projection PR361](https://github.com/255BITS/ParticleGAN/pull/361) repairs trajectory
and residual under their full gates; the corrected nonascent primary control does
not. [Transport PR358 round2](https://github.com/255BITS/ParticleGAN/blob/f9f6419ac251033e9076f7ba4bab87e715d321bd/reports/forge/bcap-physics/kinetic_transport/round2/README.md)
repairs sustained unequal mass, whereas its round3 backtracking successor loses
that pass. These are distinct source/recipe cohorts. [Original saved evidence](prior-evidence.json)
records inspected checkpoint hashes and the original report identities; it grants
no matched new-source credit.

All arms retain nonsaturating loss, full DualNorm momentum0, smoothing .001,
per-offset convolution layout, G/E .012, D .018, prior .030, constant floors1,
BCAP coefficient1/cap1/every update, zero extra prior regularization, additive
noise and EMA. The historical bare BCAP preset is resolved with every winning
override rather than mistaken for the winning trained configuration.

The projection switch is exactly `constraint_geometry_mode=strict_progress`.
For a rounded proposal d conflicting with an existing protected gradient a,
it blends the nonascent projection with bounded common descent and backtracks
on those same protected losses using the existing finite Armijo check. It is
inactive bitwise when no original conflict occurs. Its implementation is unchanged
from trained source `13450f4cae33748240ca5425814f3ff6fcc1bc41`.

Transport uses the exact public module from trained source
`5c2a64682b8900c5b42de94a6c27502e41d500f2`: normalized empirical sliced W2 with
32 deterministic directions, plus the relative real-anchor feature residual,
each global weight1. Fourth-neighbor real widths and scales1/2/4 are detached.
For real-anchor features phi, L_local = mean[(E_fake phi - E_real phi)^2 /
(E_real phi)^2]. No labels, evaluator centers/sigma, new draws or serving changes
enter. The finite feature form has a positive semidefinite kernel interpretation;
it is not a characteristic population guarantee. See
[Gretton et al.](https://www.jmlr.org/papers/v13/gretton12a.html) and
[Li et al.](https://proceedings.mlr.press/v37/li15.html) for MMD and differentiable
generator moment matching. [Sener and Koltun](https://arxiv.org/abs/1810.04650)
provide the multiobjective context for gradient conflict. These sources do not
prove this stochastic GAN composition's density or retention gates.

BOTH adds transport to the generator/prior total objective while retaining
projection's original protected losses. Scalar hosts protect adversarial loss
only; transport is not silently made a second protected loss. In adversarial-only
singletons, projection has no additional coverage objective. The G-phase uses the
same fake and real tensors already consumed by the host. Frozen vector/Gaussian
adapters reuse one actual real tensor for D/G, as before.

## Frozen scope and applicability

| Unchanged task | Winner | Projection | Transport local-v2 | BOTH |
| --- | --- | --- | --- | --- |
| Two-pole explicit fixed fixture | Supported | Supported | BLOCKED | BLOCKED |
| Gaussian smoke and own stability | Supported | Supported | Supported | Supported |
| Trajectory | Supported | Supported | BLOCKED | BLOCKED |
| Residual student | Supported | Supported | BLOCKED | BLOCKED |
| Mid-scale identity guardrail | Supported | Supported | BLOCKED | BLOCKED |
| Unequal mass | Supported | Supported | Supported | Supported |
| Two-broad guardrail | Supported | Supported | Supported | Supported |

The frozen conditional/fixed component hosts do not consume transport's
sample-space losses. Preserve these explicit BLOCKED cells with zero spend;
BOTH cannot claim preservation of the conditional repair in this comparison.
There is no task-specific routing or alternate initializer. Two-pole retains
its original identity/zero/stored-weight fixture cohort, separate from learned
initialization. The mid-scale task is parameter-only and its prior is not sampled.

Eight original task cards and numerical gates are unchanged. All arms use seed0,
public deterministic initialization, fixed architecture/data law, seen batches,
prior law, sampling, full committed-update budgets and scheduled evaluations.
Named constructor/data/training-noise/evaluation streams remain isolated and
checkpointed. Failed own smoke blocks its own stability continuation. Compatible
archived evidence is context only; new controls run on the same frozen source.

Four ready schema3 candidates/studies share campaign
`projection_transport-round4-v1`. Their total full task allowances are 40080
seconds (10020 per arm), within the 43200-second track ceiling. Unsupported cells
spend nothing. No sweep, seed experiment, second proposal, unchanged continuation
or failed transport-v3 controller is admitted. Software checks and read-only
saved-state analysis add no scientific qualification and are accounted separately.

The numerical preregistered signature is unequal-mass endpoint full component
covariance error <=.85; >.85 falsifies it. Complete sustained gates decide the
repair. The substantive composition question is whether BOTH retains local-v2's
complete rare-density PASS and the broad guardrail. A conditional union remains
unmeasured if the combined host is unsupported. Competing explanations include
adversarial protection suppressing useful transport, finite-anchor tail blindness,
changing critic objectives, and shared normalized map deformation. Stop after
this four-arm readout regardless of its outcome.

## Execution

Scientific source and ready declarations are pushed before enqueue. The independent
public Queue/drain runner disables the full-compile completion callback and uses
one shared GPU worker. Logs/checkpoints/JSONL remain outside Git. Publication
reconstructs metrics and actual-training GIFs from certified saved observations.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_transport/logs/driver.log
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/projection_transport/round4/run.py
```
