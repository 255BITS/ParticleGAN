# Thermodynamic track: selective optimistic BCAP dynamics

This preregistered study asks whether correcting rotational network motion
improves the winning BCAP recipe's sustained distribution gates. Results are
pending. It is a six-task mechanism diagnostic, with no ordinary Tier 1/Tier 2
qualification, calibration, promotion or public-default claim.

## Local theory and its limits

Write the coupled descending game field as
`F(w)=(grad_D L_D, grad_G L_G, grad_z L_G)`; signs are absorbed into each
player's minimized loss. Locally `J=dF/dw=S+A`, with
`S=(J+J^T)/2`, `A=(J-J^T)/2`. The symmetric component supplies local
contraction/expansion; the antisymmetric component supplies circulation.
For the continuous field `dot(w)=-F(w)`, the residual energy
`R=||F||^2/2` obeys `dot(R)=-F^T S F`, because `F^T A F=0`.
This is a mathematical stability diagnostic, not physical heat or entropy.
There is no literal thermodynamic law for this GAN.

For a scoped bilinear rotation `F(w)=A w`, `A^T=-A`, ordinary simultaneous
Euler updates multiply energy in a frequency block by `1+(eta*omega)^2`.
Optimistic updates `w_(t+1)=w_t-2 eta F(w_t)+eta F(w_(t-1))` have roots
`lambda=(1-2 i a +/- sqrt(1-4a^2))/2`, `a=eta*omega`.
For `0<|a|<1/2`, their squared moduli are
`(1 +/- sqrt(1-4a^2))/2`, both below one. Thus a precisely scoped
prediction is asymptotic damping instead of Euler energy growth. The
initial ordinary step can expand; monotonic step-by-step energy is not claimed.
A dimensionless contraction surrogate is `-log(R_next/R_current)`;
its interpretation needs the fixed local model and excludes zero residual.

Neural non-saturating BCAP is not a bilinear monotone game. Its normalized
field is nonlinear, its updates alternate D then G/prior, and only networks
receive optimism. The theorem above does not carry over. Nor do scalar width
oscillations prove nonzero antisymmetric Jacobian flow. We test the resulting
trainer, while describing observed KS/width variation as stability proxies,
not measured entropy production or an identified game circulation tensor.

## One substantive trainer delta

The candidate [declaration](../../../../configs/forge/ideas/thermodynamic-bcap-optimism-v1.json)
adds public `Recipe.thermodynamic_optimism=1`. Compute each network's ordinary
unit-rate DualNorm direction `u_t`, including aspect/kernel factors and
smoothing .001; apply `delta=-eta*(2*u_t-u_previous)` thereafter. The first
active gradient uses `delta=-eta*u_t`. G/E/D share the same rule, coefficient
and fixed settings across all tasks. There is one cached normalized field per
network parameter; state validation binds it to the coefficient and checks its
shape, dtype and finiteness. A missing gradient consumes no history. A zero
current field allows the explicit previous-field correction.

Prior/table roles retain the exact sampled-row rule, including unsampled-row
immobility. No sampled prior cache is introduced: an asynchronously sampled
previous row field would be stale by a different interval on every row.
This is selective network optimism in the coupled network/prior game, not
an implementation of fully simultaneous three-player optimistic descent.

The control is the exact saved configuration
`bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36`
resolved in this same source/runtime: non-saturating BCAP coefficient1/cap1
on every update, full zero-momentum DualNorm smoothing .001 with per-offset
convolutions, constant G/E .012, D .018 and prior .030, zero input/output
training noise and zero EMA. Every other resolved winner setting is inherited.
Archived source digest `2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`
and revision `dfe88a2ee15fb9d83ffdc5c8a25d698686b73d4e7c63b6b0e35efb0d64e94359`
retain their original 7/21 Tier 2 result; those are not matched new-control credit.

## Prior evidence and primary literature

[The winner analysis](../../bcap-tier2-search/FAILURE_ANALYSIS.md) records Gaussian
stationary width .354–2.492, 2/72 stationary full passes and mode hold's
late quality dip to .8694. [Pacing](../../dualnorm-pacing-v2/README.md) and
[Gaussian diagnosis](../../gaussian1d-diagnosis/README.md) motivate a response
to direction history rather than another unchanged rate or seed experiment.
The saved-state analysis also finds conditional coverage opposing identity;
optimism might leave that competing objective unresolved.

[Daskalakis et al., Training GANs with Optimism](https://arxiv.org/abs/1711.00141)
propose optimistic game optimization. This track applies that family of
previous-direction correction to normalized network fields; it does not claim
novelty for optimism. [Gidel et al.](https://arxiv.org/abs/1802.10551) study
variational inequalities and extrapolation methods.
[Balduzzi et al.](https://proceedings.mlr.press/v80/balduzzi18a.html) motivate
the symmetric/antisymmetric game-Jacobian decomposition. Their assumptions
and algorithms do not certify this stochastic selectively corrected trainer.

The older [Optimistic Adam screen](../../../toy100/optimistic-adam-scratch.md)
failed all six declared rows under two different cores. The original
[game-dynamics attempts](../../../toy100/formulation-round-20260924/game_dynamics/conclusions.md)
also retain failures and different CPU-fixture initialization contracts.
Those are negative prior evidence, not reasons to rerun an unchanged method.
Our full smoothed DualNorm field, network-only correction, exact winning
non-saturating recipe and public initializer define a different operator/cohort.

Newer source-bound [network magnitude PR320](https://github.com/255BITS/ParticleGAN/pull/320),
[prior magnitude PR319](https://github.com/255BITS/ParticleGAN/pull/319), and
[combined magnitude PR321](https://github.com/255BITS/ParticleGAN/pull/321)
all failed the Gaussian continuous-learning gate and regressed ring. The combined
cached-past arm uses a temporary joint preview from its previous field, evaluates
a fresh field there, restores the base, then corrects. This candidate instead
evaluates the existing alternating field once at actual parameters and directly
combines current/previous normalized directions. It introduces neither fixed
spectral cap .1, prior cap .001, nor a preview evaluation. These studies differ
in source, prior and timing and supply no matched qualification reuse.

## Prediction, competing explanation and finite protocol

Before training, predict Gaussian stability terminal KS <= .05, improved
stationary full-pass count and narrower width variation; mode hold must acquire
five terminal passing checks, and the broad-vector/two-pole guards must retain
full passes. The study's machine signature is Gaussian terminal KS <= .05;
KS > .05 falsifies that signature. Every unchanged sustained task bound remains
necessary, including conditional trajectory identity MSE <= .02.

The competing explanation is continued sampled-prior motion, non-Gaussian
latent allocation or task-owned objective competition. Independent noisy
unit directions with coordinate variance `sigma^2` would give
`Var(2u_t-u_previous)=5 sigma^2` (or `(5-4rho)*sigma^2` with lag correlation
rho). Optimism may therefore amplify the disturbance it was meant to correct.
An improved endpoint alongside failed retention or guardrail regression does
not establish repair or identify circulation as the failure's cause.

[Diagnostic view](../../../../configs/forge/views/thermodynamic-circulation-v1.json)
contains unchanged two_pole, gaussian1d_smoke, gaussian1d_stability, mode_hold,
vector_two_broad and trajectory. All task budgets/gates remain intact; Gaussian
stability requires its own passing smoke checkpoint. A smoke failure must leave
that continuation BLOCKED. This is an explicit diagnostic scope, not a smaller
ordinary-tier denominator. Each recipe reserves 6,420 seconds; the two recipes
reserve 12,840 within the shared campaign ceiling 14,400, including any paid
repair/diagnostic. At most one substantive candidate and one matched control;
no scientific retries, parameter search, extra seed or continuation. Stop and
publish regardless of result.

Protocol seed0, public deterministic initialization, architectures, target/data
laws, prior widths/learnability, D/G seen batches, sampling, update budgets and
evaluation cadence remain fixed within each task. Public components and
GANTrainer consume/checkpoint their isolated constructor/data/prior/noise/eval
streams. Clean live sampling is retained. Full actual-training GIFs and numerical
readouts will accompany completion. The parent maintains the sole campaign
leaderboard; this report only contains task comparisons.

## Reproduction and logs

Use the pinned scientific source and Python3.12 `.venv`; the public package must
resolve to this worktree. Source/runtime requests freeze at enqueue. Planning:

```sh
.venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue plan thermodynamic-bcap-optimism-v1 --study thermodynamic-circulation-candidate-v1 --show-boundaries
```

The same dedicated queue receives both declarations; only the parent's shared
coordinator executes them. Do not start a second drain. Tail
`/mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue/events.jsonl` and
`/mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue/thermodynamic-circulation-v1/progress.jsonl`.
Bulk logs/JSONL/checkpoints stay in the artifact filesystem. Compact outcomes,
provenance and media will be committed here after readout.
