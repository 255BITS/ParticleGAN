# Constraint geometry after full DualNorm

Status: declared, awaiting one bounded candidate/control experiment. This is an
explicit mechanism diagnostic, with unchanged numerical task gates and no
ordinary-tier qualification or default-promotion claim.

## Question and mechanism

Can useful descent directions survive competing set-coverage forces under full
DualNorm? The original winner's saved trajectory endpoint has coverage/identity
network-gradient cosine −0.899 and combined first-order identity-MSE change
+0.00579, despite adversarial-only change −0.02044. Residual student's cosine is
−0.831, with identity change −0.000426 versus −0.00805 without coverage. These
original-source endpoint derivatives motivate the hypothesis; they are not new
control passes or evidence about earlier training.

Let the unchanged host loss be `L = A + w_C C + w_R R + w_Z Z`, where A is
adversarial, C set coverage, R the existing paired loss when present, and Z prior
regularization. The full-DualNorm winner produces an actual displacement `d0`
across generator and learned-prior parameters, including its per-group rates and
sampled-row restrictions. The candidate computes existing protected gradients
`a_j = ∇ L_j` at the same pre-update state and solves

```
d* = argmin_d 1/2 ||d - d0||²
     subject to a_jᵀ d <= 0 for each existing protected objective.
```

The rule protects adversarial loss everywhere and the existing paired loss when
one is present: residual MSE in residual student and paired cover in mid-scale.
Trajectory's set coverage remains auxiliary because it does not bind identities.
All task coefficients, architectures, data, prior and training steps stay fixed.
For one active normal, `d* = d0 - max(aᵀd0,0) a / ||a||²`. For two, deterministic
active-set enumeration solves the tiny Gram system with a pseudoinverse. Zero
normals impose no constraint. Projection happens **after** the full DualNorm
transformation and includes joint prior/network displacement; no subsequent
normalization can undo it. Unsampled prior coordinates are excluded from both
displacement and constraint normals. The zero displacement is always feasible;
opposed gradients may remove all useful movement along their shared axis.

This is a parameter-space first-order non-ascent guarantee, subject to floating
point tolerance. The report will measure the actually applied displacement after
parameter recomposition. It does not guarantee finite-step loss reduction,
paired-identity descent from critic descent, feasible strict descent, or global
GAN convergence. Critic BCAP and update timing remain unchanged.

## Primary literature and competing explanation

[Yu et al., Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
project conflicting task gradients before aggregation. This candidate shares the
halfspace geometry but projects the actual normalized displacement, with fixed
protected objectives and no randomized order. It is not unchanged PCGrad.
[Sener and Koltun, Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)
frame conflicts through Pareto optimization. Here we seek the nearest feasible
non-ascent displacement rather than a minimum-norm convex gradient combination.
Neither paper establishes performance for BCAP or these conditional hosts.

A competing explanation is that the critic's local signal is insufficient to
escape a structured wrong permutation, or that coverage remains useful for
acquisition. Non-ascent constraints can stall at a Pareto boundary, and finite
normalized steps can still overshoot. Unconditional Gaussian instability and
finite MoG allocation do not supply an independent paired objective, so this
mechanism is expected to have little influence there. An unchanged Gaussian
failure is an anticipated scope limitation, not a reason to splice another rate.

The prior whole-rate [pacing search](../../dualnorm-pacing-v2/README.md) and
[saved Gaussian diagnosis](../../gaussian1d-diagnosis/README.md) do not test this
post-normalization directional constraint. Newer source-bound magnitude studies
[PR319](https://github.com/255BITS/ParticleGAN/pull/319),
[PR320](https://github.com/255BITS/ParticleGAN/pull/320), and
[PR321](https://github.com/255BITS/ParticleGAN/pull/321) fail continuous Gaussian
learning with fixed prior/network magnitude caps, including past extrapolation.
We retain those failures as context and do not repeat or reuse their cohorts.

## Frozen experiment

One global candidate `constraint_geometry-nonascent-v1` differs from
`constraint_geometry-control-v1` only by `Recipe.constraint_geometry_mode=nonascent`.
The control resolves the exact original winner recipe: non-saturating loss, full
DualNorm momentum0/smoothing .001/per-offset convolutions, constant G/E .012,
D .018 and prior .030, BCAP coeff1/cap1/every update, zero additive training noise
and no EMA. The original configuration identity is
`bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36`;
original revision `dfe88a2ee15fb9d83ffdc5c8a25d698686b73d4e7c63b6b0e35efb0d64e94359`,
executed-source digest `2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`.
Original outcomes 7/21 Tier2 remain archived, not credited to this matched source.

Both declarations use the same frozen current executed source and Python3.12 /
RTX A6000 compute cohort. Protocol seed0, public deterministic initializer,
named constructor/data/training-noise/evaluation streams, sampling and full task
contracts remain unchanged. Every consumed stream and optimizer constraint
history is checkpointed. Unsupported component hosts fail preflight, and an
enabled optimizer refuses a step without the protected-loss hook.

The diagnostic includes unchanged two_pole (80 updates, 300-second reservation),
Gaussian smoke (1,000/120), Gaussian stability (own passing-smoke prerequisite,
through 6,000/600), trajectory (400/1,800), residual student (400/1,800), and the
original passing mid-scale identity guardrail (800/1,800). Reservations total
6,420 seconds per recipe, **12,840 seconds combined**, below the 14,400 allowance.
The shared parent coordinator is the sole GPU launcher. No scientific retry,
seed search, continuation after failure or second candidate is authorized.

Numerical prediction: trajectory final identity MSE ≤ .02 and at least five
passing terminal checks; residual student retains its full MSE/success/wrong-pad
gates; mid-scale retains all concept and identity bounds. Falsifier: trajectory
endpoint MSE > .02, or failure of its sustained gate. Partial endpoint gains and
first-order protection alone do not establish a repair. Software controls test
intersecting/opposed halfspaces, disabled-path parity, exact winner resolution,
sampled-row ownership, and checkpoint/missing-hook rejection. Existing frozen
scorer controls and actual observed training media will be published on readout.

## Reproduction and logs

```
.venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue plan constraint_geometry-nonascent-v1 --study constraint_geometry-candidate-study-v1 --show-boundaries
.venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue enqueue constraint_geometry-nonascent-v1 --study constraint_geometry-candidate-study-v1
.venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue enqueue constraint_geometry-control-v1 --study constraint_geometry-control-study-v1
```

Local plans/tests/progress: `/tmp/bcap-physics-20261009/constraint_geometry/`.
Coordinator tail: `tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/logs/coordinator.log`.
Bulk logs, observation arrays and state tensors stay outside Git. Compact final
metrics/provenance and genuine training GIFs will be committed here. The parent
maintains the sole current cross-track leaderboard.
