# Deferred RA10 clean-feature versus emitted-law diagnostic

Status: source-only design. The original full grid has completed VALID/FAIL. No PT
object, chart, model or forward has been evaluated for this design. Candidate
selection and numerical execution remain the parent's decision after the
original final verdict. The existing frozen JSON acceptance watcher is separate.

## Source findings

1. `freeze_moment` computes the target from all even real rows in each learned
   real-only topology group. Its comparator is all clean EMA table rows. Neither
   `capture_features` nor this target/comparator call adds latent perturbation
   or observation noise. `run_mean_phase` refreshes the all-row EMA comparator
   after the count prefix and uses that same population for its objective.

2. The original native `draw` calls indexed `_generate(..., sigma=0)`, which
   still perturbs sampled latents, then adds the current output noise to obtain
   the noisy cloud. Therefore raw table outputs, saved clean draws and saved
   noisy draws are three different distributions. The original noise remains
   part of the target task and the accepted serving API.

3. Critic features are the learned final scalar-head inputs, not raw 2D output.
   The native discriminator uses Fourier sin/cos followed by three LeakyReLU
   layers; the fitted rank8 chart is not an affine output-coordinate map. Thus
   E[phi(real noisy)] and E[phi(clean anchors)] can differ at identical physical
   centers because of variance, activation crossings or Fourier curvature.
   Matching their means can expand/shift anchors and then double-count the
   stochastic path's contribution in emitted feature moments. As an algebraic
   example, phi(x)=x² gives E[phi(c+epsilon)]=c²+variance(epsilon); this is a law
   mismatch illustration, not a measurement of the actual critic.

4. Legal copies are restricted to jointly supported, inside, same-group rows
   and preserve each view's cell/category. The objective includes the complement
   of that movable cohort. It can therefore move supported rows to compensate
   an immovable complement. Exact whole-group mean decrease does not imply a
   decrease of the supported cohort's physical centroid error.

5. Unit directions are frozen before the prefix and the action means/counts
   are refreshed afterward. Category/group preservation makes the denominator
   update internally correct. D, projections, topology and scaling change at
   the next reaction, so logged objective values across checkpoints are not a
   single fixed longitudinal loss. These source properties are not a detected
   packet/ledger/cache bug or a causal attribution of the running grid.

## One fixed saved-state diagnostic, if selected after failure

Use only the final RA10 Grid1007000 state, its original clean/noisy100k EMA
holdout pair and saved real target. Bind complete hashes before interpretation.
No RA9 refit, alternate chart, action planning, counterfactual cloud, new draw,
training step, oracle decision, rescoring or threshold search.

Reconstruct the native G/EMA affine maps and current D feature head functionally
from saved tensors. Use the established immutable `head_features` equations;
avoid constructors and their initialization RNG. Fit exactly one requested128,
rank8 reference chart from the saved FIFO with a private clone of saved CPU RNG.
All primary grouping and clipping precede any oracle annotation. This is a
new descriptive chart at the final state, not the exact vanished pre-action
chart and not a replay or statistical certificate.

Apply the original even-reference group centers, scales and radial clipping to:

- even/odd real FIFO (all rows and the unchanged inside+p>Q subset), all raw EMA
  anchors A, and the current legal cohort L;
- the complementary raw anchor cohort U;
- the original saved EMA clean C and paired noisy Y holdout rows.

L uses the exact original inside+p>Q predicates in both FAST/EMA views and
same learned group. It is only a descriptive movable cohort; no discovery,
null, FWER or per-group confidence claim is made from this selection.

Report one bounded table per learned group and one aggregate, with no variants:

1. Counts and the exact mixture identity mu_A=f_L*mu_L+f_U*mu_U in clipped
   feature coordinates. Report target residuals for A/L/U, their weighted
   contributions and alignment. Keep the all-real and supported-inside real
   target means separate. This reveals whether the complement can make an
   all-row objective pull L in a different direction from the corresponding
   inside target. The double-view legal cohort has no identical real sampling
   counterpart, so this is an empirical conditioning contrast, not equivalence.

2. Paired stochastic increment delta_noise=mean(psi(Y)-psi(C)), conditioning
   BOTH members on C's learned group. Report the paired raw output mean
   increment alongside it. A nonzero feature increment with a small raw mean
   increment exposes nonlinear/clip effects without conflating them with
   noisy reassignment. Record the original C-to-Y group-transition fraction.
   The A-to-C difference is unpaired and is reported only as a distribution
   difference containing latent perturbation and row-sampling effects.

3. The original-style clipped feature residual energy for A/C/Y, plus raw
   output means and covariance traces for real/A/L/U/C/Y using the same
   learned groups. Contrast feature residual change with physical mean and
   variance direction. Oracle purity and physical within-mode centroids may
   be appended only after fixed grouping as annotations using unchanged
   original equations; they never define chart, cohort, target or action.

The two explanations are falsifiable: the stochastic-law hypothesis needs a
paired feature increment or A-to-C mismatch aligned with the target residual
despite small paired raw mean change; the cohort hypothesis needs a complement
contribution or opposed all-row/inside residual directions. Report the actual
values, weighted norms and alignments without introducing a tuned significance
cutoff. A negligible or oppositely directed result does not support the
corresponding explanation. Even a positive decomposition does not identify
historical causality or qualify a production correction.

## Limits and prospective scope

Use one fixed final snapshot/chart only. No claim that rank8 implies curvature,
that100 learned groups imply perfect physical purity, or that a fresh final
chart reconstructs an earlier reaction. Finite samples, clipping, group
assignment, trained D/shared FIFO and emitted-versus-anchor conditioning remain
explicit. The original toy/grid gates, noise, seeds, horizon, action budget,
serving lease, population gate, sources and saved results remain frozen.
