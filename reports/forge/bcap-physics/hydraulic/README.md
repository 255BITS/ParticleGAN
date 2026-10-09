# Hydraulic output travel: bounded mechanism study

Status: declarations frozen before training; results pending. This explicitly
scoped diagnostic does not qualify an ordinary tier or change public defaults.

The candidate applies one global joint-output travel rule to the exact winning
BCAP recipe. A pressure analogy motivates measuring displacement after mechanical
leverage: useful forces can produce excessive output motion when normalized
layer updates add coherently. It does not imply an optimizer conservation law.

For parameters θ, sampled prior locations z and existing normalized proposal v,
write the first-order output displacement as
`u_i = J_θ G(z_i) v_θ + J_z G(z_i) v_z,i`.
Its mean is shared network/prior motion; the residual describes individual motion.
Coherent layer terms amplify `||sum_l J_l v_l||`, even when every layer has a
bounded parameter step. Gradient direction and realized travel are distinct.

The proposed rule computes only from the consumed generator-side real batch:
`r = median_i min_(j != i) ||x_i - x_j||`, with global coefficient 1.
It proposes the original joint G/prior update, measures
`d(s) = sqrt(mean_i ||G_(θ+s vθ)(z_i+s vz,i+ε_i) - G_θ(z_i+ε_i)||²)`,
then uses `s=min(1,r/d(1))` and at most six further halvings until `d(s)<=r`.
If the bounded replay cannot satisfy the constraint, it rejects parameter motion.
The same latent indices and Gaussian jitters are replayed; there are no extra RNG
draws, critic updates, evaluations, target centers or quality-aware decisions.
Unsampled prior rows stay fixed. Zero-momentum optimizer counters advance once.
The nominal constant G/E/D/prior steps remain .012/.012/.018/.030; loss,
BCAP cap/coefficient1/every-update, smoothing .001, per-offset convolution,
no input/output noise, no EMA and task-owned settings remain the winner's.

The implementation observes proposed network-only and additional prior motion,
shared-mean energy fraction, accepted travel, backtracking scale and first-order
parameter work `-sum_p grad_p · accepted_delta_p`. These are training diagnostics,
not measures of objective improvement. The direction can still be wrong.

A training-batch output trust bound is related to function-space damping and
Gauss-Newton trust geometry discussed by [Martens (2020)](https://jmlr.org/papers/v21/17-678.html).
It is a scalar ray projection, without curvature inversion or natural-gradient
claim. [TRPO](https://proceedings.mlr.press/v37/schulman15.html) constrains policy
distribution change; this study bounds empirical sample displacement and inherits
none of its improvement guarantees. [TrasMuon (2026)](https://arxiv.org/abs/2602.13498)
restores magnitude control to orthogonalized optimizers through RMS calibration
and relative energy clipping. Here the measured quantity is joint generator/prior
output motion, including cross-layer coherence, and the radius comes from current
training-data spacing rather than gradient energy or an absolute spectral scale.

## Prior evidence, prediction and falsifier

The [original saved-state analysis](../../bcap-tier2-search/FAILURE_ANALYSIS.md)
finds native full-step median motion 7.26–7.92 target sigmas and 51–66% shared
motion energy. Half-step probes improve a geometric surrogate but do not select
an optimal rate. Gaussian width oscillation and distorted fitted-normal shape
also rule out a universal permanent-contraction account.
[Earlier pacing](../../dualnorm-pacing-v2/README.md) found task tradeoffs.
[PR319](https://github.com/255BITS/ParticleGAN/pull/319),
[PR320](https://github.com/255BITS/ParticleGAN/pull/320) and
[PR321](https://github.com/255BITS/ParticleGAN/pull/321) respectively test fixed
prior cap .001, network spectral cap .1 and their combination with several timing
laws. All fail strict Gaussian continuous learning; ring regresses. Those distinct
source/prior/loss/timing cohorts are motivation, never current control credit.
This candidate measures actual joint output displacement and changes neither the
winner's pressure field nor D, rather than repeating those fixed gradient caps.

Prespecified prediction: grid100 final precision at least .48 (approximately twice
the archived .24072 endpoint), while the matched vector_two_broad guardrail passes.
The complete native gate still requires precision >=.97, all100 genuine modes,
full density/shape/accuracy bounds and five terminal passing checks. All original
Gaussian gates and its own passing-smoke checkpoint prerequisite remain intact.
The quantitative grid falsifier is final precision <.48. Losing the broad-vector
pass also rejects the candidate as a global repair. Lower motion alone is not success.

Competing explanation: the batch-spacing radius is too small for timely initial
transport; limiting motion may leave wrong mass allocation, distorted shape and
critic oscillation untouched. A sample mean bound does not bound every individual
sample or unseen latent, and spacing depends on batch size/data dimension.

## Frozen scope and allowance

| Unchanged task | Updates | Full reservation per recipe | Role |
| --- | ---: | ---: | --- |
| two_pole |80|300 sec|Shared anchor; candidate explicitly BLOCKED on public_components |
| gaussian1d_smoke |1,000|120 sec|Shared acquisition anchor |
| gaussian1d_stability |6,000 total|600 sec|Own passing smoke prerequisite; retention/shift failure |
| grid100 |7,000|3,600 sec|Native overshoot failure |
| vector_two_broad |1,200|1,800 sec|Passing regression guardrail |

At most one substantive candidate plus one matched winner control: 12,840 sec
maximum full job reservation, within the 14,400 sec ceiling including software
checks. The candidate's two-pole blocker is explicit: its caller-owned component
API cannot replay joint output travel. There is no substituted initializer or
silently inactive limiter. Independent diagnostic peers run; Gaussian continuation
runs only after its own full smoke passes. Zero scientific retries, new seeds,
extra candidates, gate relaxation or promotion runs are allowed.

Both recipes execute in the same frozen source/runtime cohort, protocol seed0,
public deterministic initializer and isolated/checkpointed constructor, data,
training-noise and evaluation streams. The saved original winner at revision
`dfe88a2ee15fb9d83ffdc5c8a25d698686b73d4e7c63b6b0e35efb0d64e94359`
/source digest `2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`
remains archived original evidence, not a matched current control.

All submissions use the dedicated shared queue; the parent owns the only drain.
Tail `/mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/logs/coordinator.log`.
Compact task tables will be published here; the parent maintains the one current
cross-track comparison. Bulk logs, checkpoints and event streams remain local.
