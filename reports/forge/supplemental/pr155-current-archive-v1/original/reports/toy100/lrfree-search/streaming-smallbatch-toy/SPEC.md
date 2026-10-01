# Frozen N=32, batch-two critic-memory toy

Written before executing `toy.py`. Astra reviewed the mechanism and limits.
This is a deliberately tiny symbolic GAN, not an E19 package or native gate.
No algorithm may use category labels, true probabilities, or real-sample
geometry; those are used only to construct/evaluate the toy. One NumPy
`default_rng(20260929)` stream, batch size two, 1,600 updates, N=32 rows,
four observable categories, float64, common random quantiles across arms.
No seed or parameter search after results.

Each particle has trainable logits over the four categories; a row is sampled
and then a category is sampled from it. The critic has a learned scalar score
per category, represented as a linear head over learned-feature coordinates.
D trains with E19's paired relativistic logistic loss using Adam lr `.00425`,
betas `(0,.999)`. Sampled rows update their categorical generator logits by
the exact conditional generator-loss gradient over just four symbols, using
Adam lr `.0085`, betas `(0,.999)`. No raw-sample or feature-space distances
enter the candidate algorithms. Initial rows have probability `.97` on their
assigned category and `.01` on each other category, except the fixed exact
null control, whose rows are one-hot and frozen. All algorithms see the same
two real observations per update and the same random quantiles for fake-row
and category draws. The random streams are private to this toy.

Before updating D on a fresh pair, evaluate every current row's paired
generator payoff:

`L_i,t = mean_{r in batch2} sum_c P_i(c) softplus(D_t(r)-D_t(c))`.

This is a stochastic estimate of the actual RpGAN generator objective for
row i. It reads the current D and each current particle distribution. The
rolling arm retains only an O(N) scalar exponential average, with update
`M_i <- (1-1/16) M_i + (1/16) L_i,t`. Sixteen steps is the N/B fake-draw
turnover for N=32, B=2; it is fixed before execution, not tuned on results.
When a row is replaced, its memory is set to the *current* payoff of the
cloned row, discarding the old row's history. Current-payoff birth/death
uses `L_i,t` without memory. Every 16 updates after step 16, either method
replaces at most one highest-payoff row with the lowest-payoff row, copying
its generator logits and resetting that row's Adam state. A zero payoff gap
causes no move. This deterministic extreme-pair discretization is a probe of
the score's behavior, not a proposed final reaction law.

A prequential gate evaluates `2 sigmoid(D_t(real)-D_t(fake))` for each of
the two pairs before D trains on them. Under conditional real/fake equality,
fresh independent draws, and predictable D, its conditional expectation is
one even while D changes. Maintain its cumulative log e-value. The guarded
rolling arm moves only after the current segment crosses the threshold
`1 / alpha_j`, with `alpha_j=.05/[j(j+1)]` for the j-th move; reset the
segment after a move. This alpha-spending construction controls the chance
of *any first false move* under an exact equality null, given the stated
assumptions. It does not validate row direction, local support, or post-move
behavior. The unguarded arms use the same test for diagnostics only.

Compare five arms:

1. ordinary GAN, no row transport;
2. immediate-payoff birth/death;
3. rolling-payoff birth/death;
4. rolling-payoff birth/death with the global prequential gate;
5. direct row-probability learning using the same paired generator payoff:
   update all 32 softmax row-probability logits from their exact conditional
   gradient, while D samples fake rows from the learned distribution. It
   changes public sampling probabilities and does no cloning.

Fixed cases, all 1,600 updates:

- **Exact null:** real law `(1/2,1/4,1/4,0)`, frozen one-hot rows with
  counts `(16,8,8,0)`. Any transport is false churn.
- **Overmass/outlier:** same real law, initial rows `(12,8,8,4)`;
  G and D both learn.
- **Legitimate rare component:** real law `(1/2,1/4,7/32,1/32)`,
  initial rows `(14,8,9,1)`; losing the rare row is a failure mode.
- **Real-law shift:** initial rows `(16,8,8,0)`, real law changes at step
  800 from `(1/2,1/4,1/4,0)` to `(1/4,1/2,1/4,0)`; G and D learn.
- **Impaired critic:** overmass/outlier start but freeze D at scores
  `(-.4,0,0,+.4)`, favoring the unsupported category.
- **Feature gauge:** overmass/outlier case, with critic feature channels
  rescaled by `(16,1/4,1,1)` at step 800 and its head inversely transformed.
  Assert critic outputs and all pre-update payoff values are unchanged at
  that instant. Subsequent optimizer trajectories may differ.

For each arm/case record exact model-vs-real total variation, cumulative and
last-quarter TV, exact population critic regret for the current model law,
prequential D loss against `log(2)`, e-value trajectory, number and direction
of clones, rare-row retention, learned row-weight concentration, and elapsed
time. Report whether a true shift is detected and how quickly, plus false
moves under exact equality. No absolute pass threshold is chosen from these
cases. A rolling-payoff claim requires lower TV and less null/rare churn than
immediate payoff; a usable controller also needs to beat ordinary GAN on the
non-null cases. Failure remains a valid result. This toy cannot establish
native performance, A2 compliance, or support/FDR guarantees.

Separately, the E19 source's split-conformal isolation test is analytically
unable to execute an isolation move for N=16, 32, or 64: with N/2 calibration
examples its minimum p-value is `1/(N/2+1)`, while BH over N tests at Q=.05
requires about 40 minimum-p rows and its action guard allows only `.05 N`
flagged rows. This claim is checked from source bounds, not inferred from the
toy outcome.
