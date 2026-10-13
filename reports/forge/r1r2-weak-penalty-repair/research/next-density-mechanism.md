# Next density mechanism: energy-MMD software probe only

**One bandwidth-free energy-distance U-statistic merits a cheap saved-output
gradient/resource probe. No paid arm, coefficient or API change is selected.**
Read [compiled memory](../../EXPERIMENT_MEMORY.md), the
[full-transport review](../../r1r2-objective-memory-repair/research/full-transport.md)
and [Jacobian review](../../r1r2-objective-memory-repair/research/jacobian-conditioning.md).
The closed [weak-penalty publication](../publication/results.json), execution
`3cb46b91121f42f8b171a77f17776365aebad1d8`, preserves 2/6 ordinary passes plus
clock; Ring covariance is 4.625256 and words never acquire. Neither another
gamma point nor an optimizer grid follows. A future structural comparison would
start from the measured **gamma .1 non-saturating AMS m110 parent**, retaining
both R1/R2 terms, rather than composing with the failed gamma .01 revision.

## Exact candidate law for diagnosis

Flatten the existing G-phase fake output rows X and detached real rows Y. For
n,m≥2, define d(a,b)=||a-b||₂, **not squared distance**, and compute

```text
U(X,Y) = 2/(n*m)       * sum_i,j     d(x_i,y_j)
       - 1/(n*(n-1))  * sum_i!=j    d(x_i,x_j)
       - 1/(m*(m-1))  * sum_i!=j    d(y_i,y_j).
```

This is squared MMD for the distance kernel
`k(a,b)=||a||+||b||-||a-b||`; Euclidean energy distinguishes distributions with
finite first moments. Squaring the norm instead would identify only mean
differences. [Sejdinovic et al., Theorem 22, Proposition 29 and Remark 30](https://arxiv.org/pdf/1207.6076),
[Bellemare et al., §4 and Appendix D](https://arxiv.org/pdf/1705.10743).

Exclude diagonal **indices**, retain duplicate-valued off-diagonal observations,
and differentiate both fake arguments. Keep signed U values; clamping negatives
changes the estimator and gradient. With a fixed kernel and independent IID
panels, the U-statistic estimates population discrepancy without minibatch
coupling; gradient/expectation exchange additionally needs suitable regularity.
These claims do not certify nonsmooth atomic gradients or the deterministic
pole panel. [Gretton et al., Lemma 6](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf),
[Dziugaite et al., §3](https://arxiv.org/pdf/1505.03906).

Specify the norm derivative as v/||v|| when v≠0 and **zero at exact collisions**.
Do not inject jitter, perturbations or a denominator floor. The real/real term
is constant for training gradients but should remain in diagnostic scalar values.
Unit coefficient is useful only for inspecting this law's gradient; it is not a
selected training strength. Raw energy scales linearly with output units: absence
of a bandwidth does **not** supply a dimensionless global coefficient. A
same-batch adaptive kernel/normalizer would change the fixed-kernel unbiasedness
claim and would need separate analysis.

## Comparison with exact squared OT

Exact balanced OT minimizes `mean_i ||x_i-y_pi(i)||²` over permutations for the
unchanged equal-cardinality panels. Its selected-plan gradient is
`2*(x_i-y_pi(i))/n` away from assignment transitions. A deterministic optimal
assignment can give distinct pole rows opposing directions at the initial tie;
averaging equally optimal plans instead gives zero. Solver/tie semantics are
therefore substantive. [SciPy's exact assignment API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).

OT has no entropy bandwidth, but it fits empirical batch coupling. Even correct
independent 128-row Ring panels have expected mode-count TV .1917003214, as
derived in the linked full-transport review; balanced matching must transport
that count mismatch between modes. Unbiased gradients **of an expected
minibatch-OT objective** do not imply unbiased gradients of population W₂².
[Fatras et al., §§3.3–3.5](https://proceedings.mlr.press/v108/fatras20a/fatras20a.pdf).
Energy-U removes that forced coupling and kernel bandwidth choice, so it is the
single probe priority. It still has finite-batch variance and uncertain force
strength. Neither theory establishes a density repair.

## All seven unchanged hosts

| Host; existing output panel | Binding and limiting evidence |
| --- | --- |
| Ring; 128×2 | Output-only G/sampled-prior gradients; retain actual rows/jitters. No mode IDs, centers, evaluation samples or covariance oracle enter training. Small local-width forces can be overwhelmed by distant-mode terms. |
| Gaussian; 128×1 | Same G/prior ownership; width and acquisition can still regress. |
| Words; 256×168 | Whole probability rows only; retain sampled prior and native joint G/E objective. Latent coordinates and encoder outputs stay outside U. Marginal matching does not establish paired inverse reconstruction; softmax saturation can suppress parameter gradients. Retain duplicate words with original counts. |
| AE; 64×2 | Existing generated decoder branch and prior; original reconstruction/encoder loss remains. Shared decoder parameters can create conflicts. |
| Direct pole; 12×1 | Original direct coordinates. With the specified collision derivative, fake rows at zero against symmetric ±(.95…1.05) targets get zero auxiliary force. Common collapsed translations remain flat inside the target gap. No separation claim; the original gate tests travel/slope and relies on the existing adversarial force. |
| Token; 8×2 | Existing repeated trained-concept output and target; unused slot/hold objective excluded. At collapsed fake a and repeated target t≠a, U gives a pull toward t; at a=t use zero derivative. The cusp and nonvanishing nearby force can cause finite-step oscillation. No variance division. |
| Own clock; 8×2 | Same output law, no step labels/horizon/cadence dependence; preserve every consumed stream and checkpoint. |

These bindings are feasibility requirements, not completed public API support.
The current recipe exposes only sliced/global and local-v2 output transport.
Its `OutputMarginalTransport` checkpoint loader requires nonnegative accumulated
losses; a signed-U consumer needs an explicit compatible state law.
Both failed their tested R1/R2 revisions; local-v2 is a data-dependent relative
feature residual, not this fixed distance-kernel U-statistic. Their failures
weigh against a generic auxiliary-loss optimism. Bellemare et al.'s successful
GAN uses learned features; its Appendix D reports poor raw-image energy training,
which supplies further caution for the 168-coordinate word output.

## Probe requirements and stopping rule

Use existing saved outputs and explicitly separate deterministic fixtures; if
the actual G-phase target tensor was not archived, report that absence rather
than synthesize its training identity. Check the analytic gradient against
finite differences away from collisions, exact zero/collision conventions,
permutation symmetry, duplicate multiplicities, signed-U handling and raw
gradient scale at the actual batch/dimension sizes. For Ring, measure whether
the auxiliary force distinguishes excessive tangent width from radial
compression; output forces alone cannot establish parameter/AMS update repair.
Inspect word softmax and reconstruction conflicts without constructing or
updating models. Do not infer instantaneous G/D gradients from EMA moments.

Quadratic pair matrices are practical in size: three 256² float32 matrices total
768 KiB before backward/workspace. Avoid an n×m×d broadcast; measure actual
forward/backward time and peak memory. This arithmetic is no runtime guarantee
for 20,001 word updates. Preserve the full **900-second word allowance** and
all other original budgets. Remaining shared paid headroom is
**1413.8375628870854 seconds**; no cap increase or shortened task follows.

Admission still requires a justified single global coefficient/normalization,
an explicit reusable public objective and consumers/checkpoints on all hosts,
and legitimate matched controls under the newly frozen scientific source.
Old-source certificates cannot fill fresh-source control cells automatically.
Any required task or own-clock failure rejects the complete recipe; Ring retains
all six bounds and five terminal checks. This note adds no implementation,
declaration, model, draw, update, grade, queue mutation or paid reservation.
