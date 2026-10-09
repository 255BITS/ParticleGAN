# Blacksmith: evidence-responsive tempering of normalized strikes

This round tests one global trainer change against the exact winning BCAP recipe
on five unchanged tasks. Scientific results are pending. The finite comparison
reserves **9,240 worker seconds**, within the **14,400-second** round ceiling.
No ordinary qualification, default adoption or calibrated-screen claim follows.

The blacksmith analogy treats a normalized optimizer proposal as a hammer strike,
parameter displacement as plastic deformation, and persistent gradient evidence
as willingness to yield. Evidence disagreement tempers the strike. These terms
are a mathematical design prompt; no physical constitutive law or neural-game
convergence guarantee is claimed.

## Mechanism and scoped algebra

For each network parameter tensor, let `g_n` be the current raw gradient and use
one global evidence decay `beta=.95`. The observation count `n` increases only
when that tensor has a gradient:

```
m_n = beta*m_(n-1) + (1-beta)*g_n
v_n = beta*v_(n-1) + (1-beta)*(g_n elementwise squared)
z_n = 1 - beta^n
c_n = clip(||m_n||^2 / (z_n * sum(v_n)), 0, 1)
delta_theta = -eta * c_n * U(g_n)
```

`U` is the unchanged winning smoothed DualNorm rule, including aspect factors
and per-offset convolution polar factors. Exact zero denominator gives `c=0`.
A single scalar gain applies to all offsets of a kernel tensor. Biases use the
same tensor rule. All G/E/D players use it; their direction is the current
raw-gradient direction, rather than the moving average. For learned prior rows,
the identical formula runs independently on each **actually observed sampled
row**, using its own count and row norm. Unsampled rows neither move nor age.
A missing gradient consumes no history; an observed zero gradient consumes
history and supplies no current displacement. All moments and integer row
counts are checkpointed and bound to decay and schema; checkpoint restore is exact.

The bias-corrected moments are convex weighted averages over observed gradients.
Jensen's inequality gives `||m_n||² <= z_n*sum(v_n)`, so `0<=c<=1` in exact
arithmetic. A persistent identical nonzero gradient has `c=1`; two equally
sized opposing observations have `c=((1-beta)/(1+beta))²=.000657462`.
Thus this rule cannot amplify a parameter's current normalized strike. It can
recover after disagreement, because old evidence decays with observations.
Bias correction uses observation count and is not elapsed-time annealing.

This ratio resembles the signal/second-moment factor in
[Schaul, Zhang and LeCun, ICML 2013](https://proceedings.mlr.press/v28/schaul13.html).
Their separable noisy-quadratic learning-rate derivation also uses curvature;
this implementation has no curvature estimate and applies a tensor/row scalar
to a normalized direction. It is therefore a substantive adaptation, not their
optimal-SGD formula or a theorem for GANs. The finite-window mean has noise:
for independent stationary gradients its effective count tends to 39, and
zero population signal can still give a positive estimated gain. The ratio
of estimated moments is not an unbiased population signal estimate.

Unlike the failed coefficient1 network optimism, no previous normalized field
is added or subtracted and amplitude never exceeds the current proposal.
Unlike spectral-half, singular directions retain their original relative
weights. Unlike output-motion limiting, no sampled-output trust region is
measured. Small parameter motion can still deform output covariance; coherent
but harmful gradients and objective competition can remain unchanged.

## Existing saved-state evidence and falsifiable question

The original winner is
`bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36`,
revision `dfe88a2ee15fb9d83ffdc5c8a25d698686b73d4e7c63b6b0e35efb0d64e94359`,
executed digest `2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`.
Its archived 7/21 Tier 2 result retains that identity. The
[saved-state inspection receipt](inspected-evidence.json) verifies the actual
archived Gaussian checkpoint without training or random sampling.
[Published derivatives](../../bcap-tier2-search/failure-state-analysis.json)
and [failure analysis](../../bcap-tier2-search/FAILURE_ANALYSIS.md) show Gaussian
width oscillations, 2/72 stationary full passes and 0/24 shifted holds; mode
hold loses quality at update1,050 despite retaining all eight modes, leaving a
three-check passing suffix. These support investigating constant normalized
travel. They do **not** establish temporal raw-gradient coherence or its causal
role: that history was not saved by the old optimizer.

The completed [first-round comparison](../README.md) rejects unchanged optimism
and spectral-half; the individual
[optimism report](https://github.com/255BITS/ParticleGAN/blob/61b07b1c2769b5eb97bcbde70abc3cd4b4464ad9/reports/forge/bcap-physics/thermodynamic/README.md)
loses Gaussian confirmation and mode retention. No seed repeat or coefficient
sweep is authorized here.

Frozen forecasts for this sole candidate:

- Gaussian: retain independently confirmed smoke, raise stationary full passes
  from 2/72 to at least36/72 and shifted hold from0/24 to at least12/24, and
  final `cdf_ks<=.05`. Genuine task PASS still requires every original retention,
  deadline and shifted-hold criterion. A final `cdf_ks>.05` is the study's
  machine-readable falsifier; absent own-smoke evidence stays BLOCKED/incomplete.
- Mode hold: all eight modes, quality>=.90 and at least five terminal full
  passes; failure to extend the control's three-check suffix falsifies the
  proposed late-regression repair on this task.
- Broad-vector guard: preserve its complete five-check terminal gate. Two-pole
  remains its explicitly separate fixed identity/zero/stored-weight fixture.

Competing explanations include a coherent but wrong game field, harmful
objective balance, and useful fast-changing gradients being suppressed. Sparse
prior observations may retain stale evidence despite avoiding false zero
observations. A better endpoint without sustained passes is only a diagnostic
improvement. Any failure or missing evidence ends this exact revision; no second
candidate, automatic continuation, additional seed or sweep follows.

## Matched protocol, budget and reproduction

The public [candidate](../../../../configs/forge/ideas/blacksmith-evidence-tempering-r2-v1.json)
adds only `Recipe.optimizer_tempering=.95`. Both ready schema-v3 studies use
[one diagnostic view](../../../../configs/forge/views/blacksmith-tempering-r2-v1.json)
and one campaign: 4,620 seconds per arm, 14,400 total ceiling, one round.
Primary control is the saved winner declaration resolved in the same current
scientific source/runtime, rather than preset defaults alone. It retains
non-saturating BCAP coefficient1/cap1/every-update, zero-momentum full DualNorm,
smoothing.001/per-offset convolution, constant G/E.012 D.018 prior.030,
floors1, no additive input/output training noise, no EMA.

| Task | Full reservation per arm | Fixed task role |
| --- | ---: | --- |
| two_pole | 300 seconds | Explicit fixed direct-coordinate fixture |
| gaussian1d_smoke | 120 | 1,000 updates, independent same-state confirmation |
| gaussian1d_stability | 600 | Own passing smoke producer; full original6,000 horizon |
| mode_hold | 1,800 | Numerical retention failure |
| vector_two_broad | 1,800 | Real passing distribution guard |

Architecture, data/target law, prior widths/weights/components, sampled latent
law, seen batches, update allowance, evaluation cadence and full gates are
unchanged within each task. Protocol seed0 and public deterministic initializer
apply except the explicit two-pole fixed fixture. Constructor, data,
training-noise/prior and evaluation RNGs remain isolated and checkpointed.
The frozen vector/Gaussian adapters reuse one real tensor for D/G each update;
this actual law is preserved. Clean live results stay separate from noisy/EMA
and fixed-initialization diagnostics. Frozen scorers include oracle and
destructive controls. Actual-training GIFs will be exported from saved observations.
Native tasks do not fit the selected two-arm reservation with these mandatory
anchors; native covariance, images, conditional tasks and general transfer are
unmeasured. The parent owns the single current goal leaderboard.

Use the exact scientific commit and shared Python3.12 `.venv`, with `PYTHONPATH`
pointing to this worktree. Plan and enqueue both frozen arms before draining:

```sh
PYTHONPATH=$PWD .venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/blacksmith/queue plan blacksmith-evidence-tempering-r2-v1 --study blacksmith-tempering-r2-candidate-v1 --show-boundaries
PYTHONPATH=$PWD .venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/blacksmith/queue enqueue blacksmith-evidence-tempering-r2-v1 --study blacksmith-tempering-r2-candidate-v1
PYTHONPATH=$PWD .venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/blacksmith/queue enqueue bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36 --study blacksmith-tempering-r2-control-v1
```

One bounded public `Queue/drain` worker uses GPU1 with authorized sharing,
`watch=False`, and `on_completion=None` to preserve archived qualification.
No full compilation is run. Local stdout, JSONL and checkpoints remain outside
Git. Tail the dedicated logs:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/blacksmith/logs/drain.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/blacksmith/queue/events.jsonl
```

Export/verify compact metrics, source/RNG checks and actual-training GIFs without
new training or sampling:

```sh
PYTHONPATH=$PWD .venv/bin/python reports/forge/bcap-physics/blacksmith/publish.py --queue /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/blacksmith/queue --certificates reports/forge/attempts
.venv/bin/python -m experiments.forge compile --summaries-only
```
