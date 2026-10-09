# Hydraulic round two: travel and local deformation

One successor adds **network-only antithetic finite-response regularization**
to hydraulic-v1. The primary matched control is the exact BCAP winner. Forge refuses hydraulic-v1
as an admission control because its two-pole binding is unsupported. This
comparison therefore measures the combined travel/deformation successor, not
the incremental causal effect of the added penalty. Round-one hydraulic-v1
measurements remain contextual; they are not a third measured arm. This bounded diagnostic changes no default
or ordinary qualification and preserves all round-one scientific identities.

## Saved-state motivation and mathematical mechanism

[Saved-state analysis](saved-deformation.json) uses the original round-one native
checkpoints without training or sampling. Swapping their complete latent tables
shows hydraulic G's mean local variance .008855 on its own table and .008822 on
the winner table; the winner G gives .005532 on its own table and .008831 on the
hydraulic table. Thus network shape and prior relocation both matter; hydraulic
G's width is insensitive to this particular table swap. Initial public G's
variance is only .000003906 on initial locations. Freezing its initial Jacobian
would prevent substantial necessary expansion. Endpoint network and prior
displacements partly oppose (normalized rowwise dot -.317988), with centered
energies 33.2904 and .605099. These endpoint swaps are sensitivity diagnostics,
not per-update causal decompositions. The analysis checks original checkpoint
hashes, preserves global RNG, and consumes neither target labels nor centers.

Let c be the consumed prior center and epsilon the **already sampled** training
jitter. Define q = (G(c+epsilon)-G(c-epsilon))/2 and
V = mean_i ||x_i-mean(x)||^2 for the consumed generator-side real batch.
For V>0 the successor trains with

`L_G = L_adversarial + existing_prior_term + |stopgrad(L_adversarial)| * mean(||q||^2) / V`.

The single global deformation coefficient is **1**, with unchanged travel
fraction **1**. Both center and jitter are detached in this additional term:
it changes only network gradients; the prior keeps its existing adversarial
gradient. For a locally linear G, E||q||^2 approximates sigma^2 ||J_z G||_F^2.
This discourages expansion of each noise neighborhood while the trainable
centers can supply population transport. The analogy is mechanical strain
alongside travel. It is a finite-response contraction preference, not an
incompressibility law, target-covariance estimate or bound on all Jacobians.

[Odena et al. (2018)](https://arxiv.org/abs/1802.08768) study generator Jacobian
conditioning and introduce Jacobian Clamping. This experiment penalizes the
energy of existing finite-jitter secants rather than clamping condition number
with newly drawn perturbations. The joint parameter proposal is still passed
through the original exact nonlinear hydraulic output-travel replay. No target
centers, component labels, evaluation metric or gate threshold enters training.

The successor requires a positive-width nonstandardized MoG, zero-momentum
DualNorm, clean deterministic training and a stateless supported generator.
It requires a materialized generator-real batch. Preflight computes finite
spacing and variance **before D steps**. Duplicates use nearest distinct positive
distances; a singleton or constant finite batch has radius0 and a complete
zero G/prior parameter ray. D trains normally and the zero-momentum optimizer
clocks advance once. V=0 contributes zero deformation loss. No epsilon floor
silently invents movement. Hydraulic-v1 keeps its original positive-spacing
refusal contract; positive continuous-law comparison batches share spacing.

## Frozen comparison, forecasts and budget

The ready studies are `hydraulic-deformation-candidate-round2-v1` and
`hydraulic-deformation-control-round2-v1`; campaign
`hydraulic-deformation-round2-v1` has a fresh **14,400-second ceiling** and
6,420 seconds per arm. Both arms use the unchanged
`hydraulic_motion_diagnostic` view: Gaussian smoke and its own-state stability,
grid100, vector_two_broad, plus two_pole genuinely BLOCKED for the candidate and measured on the unchanged
winner fixture. A failed own smoke blocks its stability continuation. No substituted
fixture, shortened schedule, new seed, third arm or automatic successor is allowed.
The control's original parent declaration is an untrained admission reference only.

Prespecified numerical prediction: native final receipt `precision >= .55`;
falsifier: `precision < .48`. Additional explanatory forecasts: served
`abs_cov_trace_bias <= .70`, uncensored saved-center median local variance
ratio below the matched winner control, and vector_two_broad PASS retained.
Strict Gaussian/native sustained gates are unchanged; forecasts grant no PASS.
Loss of the passing vector guardrail rejects this revision as a global repair.

Competing explanations: global real variance may underweight local native
density; contraction may prevent useful expansion or encourage prior
compensation; finite antithetic responses may miss activation crossings in
different directions; strict Gaussian mean/width drift may remain. Native
center metrics retain their within-radius population scope. Only the measured
current source/runtime pair supports causal comparison.

Protocol seed0, public deterministic initialization, architecture, prior widths,
mixture weights, component counts, sampling, seen batches, update allowances and
evaluation cadence remain task-owned and fixed. Frozen vector/Gaussian adapters
reuse one real tensor for D/G each update; that actual law is retained. Constructor,
data, prior, training-noise and evaluation streams remain isolated/checkpointed.
No extra RNG draws are added by antithetic secants or replay. Clean/live results
remain distinct from noisy or EMA cohorts. No finite-atom/image/conditional
transfer is claimed; two-pole remains its explicitly separate fixed fixture.

Raw logs and state dumps stay outside Git. Tail:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/hydraulic/logs/drain.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/hydraulic/queue/events.jsonl
```

The public Queue/drain runner uses one active worker on GPU0, authorized sharing,
`watch=False` and `on_completion=None`. Contended wall time is accounting only.
Readout/publication uses summaries-only compilation to preserve archived grades.
