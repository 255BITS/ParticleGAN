# Allocation, neural landing, and validation of continuing updates

**Superseded status:** [Round7](continuous-round7.md) completes the corrected gates,
selects pre-start fitting, and isolates a remaining minibatch-omission failure.
The investigation below retains its earlier sequence and evidence.

No complete replacement for LR decay is qualified. The latest candidate is
**sample-derived distinct-group anchors with joint G/prior fitting**, restoring
the pre-G state if the bounded fit does not converge. It passes the 44-check
saved-state gate and warm200. Dense continuation and cold acquisition remain
required. Split-batch reallocation was rejected by a longer, cheaper
free-output test before additional GAN training.
The earlier full critic-refinement method fails cold acquisition at three
modes despite passing the borrowed-state hold.

| Method or diagnostic | Result | Consequence |
|---|---|---|
| One virtual penalized-D step, full G derivative | 28/44 saved checks | Rejected before warm |
| Exact free-output whole-map C+Q update | Warm 100/100; both cold clouds 0/100, ending at four modes | Better neural landing alone cannot fix allocation |
| Distinct sample-group anchors, joint neural fitting | Saved44 and neural warm200 all HQ1; free-output warm1200 all HQ1, cold first passes at update3 | Dense neural hold and cold acquisition required |
| Global sampled-data reallocation, joint fitting, whole-map C+Q acceptance | 43/44 saved checks | Rejected before warm |
| Same reallocation with separate proposal/confirmation batch halves | Saved44 pass, but free-output warm1198/1200 and cold1197/1200 | Withhold further neural training |

All host comparisons retain nominal G/D LR .00425 and prior LR .0085, the
original noise horizon, and one Adam moment update per player. They use the
same captured initial states and random streams. No seed or coefficient
sweep, benchmark threshold change, or GitHub CI wait occurs.

## What changed in the diagnosis

The [audited one-step total derivative](pr84-one-step-unroll.md) is valid:
small double-precision perturbations without activation-boundary crossings
agree with autodiff. Nevertheless the actual continuation fails. Its first
saved update improves HQ .8242 to .9707, but that does not establish stable
feedback; the late branch still loses a mode. [Independent implementation
audit](pr84-unrolled-candidate-independent-audit.md).

An [ideal density-ratio calculation](ideal-ratio-three-mode-diagnosis.md)
shows a separate local-transport problem: missing mass can be obvious from
critic scores while the local gradient near occupied clusters is almost
uninformative about it. That unrestricted population comparator is not the
finite capped critic and does not prove host impossibility.

The old C+Q objective also has an allocation barrier. Removing both neural
pullback error and the preceding GAN update leaves two cold free-output
clouds at four modes. Some particles serve two distant real-data clusters
by sitting between them, while surplus particles remain elsewhere. A
data-only search over donor-particle/real-sample replacements escapes these
two fixed-bank basins in four strict objective decreases. Four is an observed
result, not a selected iteration count. [Exact replay and evidence](continuous-round6-independent-audit.md).

For fixed distinct sample-derived group centers and more particles than
groups, a separate [anchor objective has a finite-step output-space proof](anchor-coverage-principle.md).
It allows surplus particles to rest and prevents uncovered-group local
minima. The group-estimation assumption and the finite-capacity neural map
remain separate requirements. The generic C+Q reallocation method does not
inherit this theorem.

## Neural realization and the surviving failure

A bounded joint G/prior Gauss–Newton solver now checks the actual nonlinear
output map. It holds each requested target fixed, factors the full output
Jacobian, and accepts steps by actual versus predicted error reduction.
Cold fixed-bank anchor targets converge in 6/4/3 and 5/3/3 iterations over
three rounds; warm needs two iterations and then rests. Maximum final row
error is below 1.1e-6. Four numerical helper tests plus an independent
nonfinite-trial test cover landing, rest, rollback, rank deficiency, and
backtracking. [Frozen landing evidence](continuous-evidence/round6-anchor-joint-landing/manifest.json).

The full-data reallocation trainer still fails update 1391: HQ .855225 with
eight modes. Its nonlinear landing error is only 2.23e-6. Two surplus
particles fit rare real-data tails supported by one and two assigned samples;
actual C+Q improves .012241 to .008677. Thus the solver faithfully realizes a
training target that harms the required quality measure. This is not a
numerical convergence failure.

The separately frozen split-batch rule constructs its target from the first
64 existing D real samples and checks proposed updates against both that
half and the remaining 64. Both C+Q values must strictly improve relative to
the entire pre-G update; otherwise G/prior may rest. Among eligible fitted
and native GAN proposals it chooses the lower summed cost. The second half
does not construct the target or enter the fixed-target fit. It still enters
the native discriminator update, so this is not a theorem about a wholly
independent holdout or an unbiased generalization estimate.

Its saved-window passes are 12/12, 16/16, and 16/16, minimum HQ .936523,
.937500, and .987305. Original full-state parity is checked wherever the
capture contains a full post-update state; all 44 original supports and
update records match. Candidate and original RNG/noise endpoints match.
[Frozen split-batch gate](continuous-evidence/round6-crossfit-saved44/manifest.json).

The subsequent [free-output 1200-update test](continuous-round6-split-output-stability.md)
rejects this rule as the next candidate. Even exact target realization has
two warm and three cold quality failures. Four newly accepted proposals
improve both halves yet harm the same-noise quality check; a fifth failure
inherits the previous bad cloud while resting. A passing saved window did
not imply stability over fresh batches.

## Current distinct-group candidate

The anchor rule infers groups from the current native real128 batch using
the largest additive gap in a minimum spanning tree. Neither the true mode
centers nor a configured group count enters training. It computes one
distinct-anchor quadratic target from the pre-G cloud, jointly fits G and
prior, and selects by the actual anchor objective. A nonconverged numerical
fit restores the pre-G/prior parameters; D and the once-advanced Adam state
remain intact. This is an additional data objective, not an unchanged GAN
objective or an R1/R2 zero pull.

The [paired free-output experiment](sample-anchor-free1200.md) uses exactly
the preceding split test's initial clouds and real-data streams. Warm passes
1200/1200 with minimum HQ1; cold first passes at update3 and passes every
remaining check. The [independent neural saved-state audit](continuous-round6-sample-anchor-independent-audit.md)
passes44/44 with HQ1, verifies original full-state parity where available,
and confirms normal native-host integration in one unobserved cold update.
The rest-on-nonconvergence variant likewise passes44/44. Its actual neural
warm200 passes all checks with HQ1 and exact scheduled/original controls.
The subsequent hold has HQ1 on all1200 later checks, but its full-snapshot
comparison at1200 rejects promotion: diagnostics and all200 update/correction
records match, while the hash including complete noise histories differs.
This is being isolated before cold training. The [warm archive](continuous-evidence/round6-sample-anchor-neural/warm/manifest.json)
and [incomplete hold archive](continuous-evidence/round6-sample-anchor-neural/hold/manifest.json)
retain the frozen code, raw results, complete final states and logs.

A [conditional invariant-region derivation](anchor-invariant-region.md)
separates the mechanism from its unproved premises. With correctly inferred
groups, bounded centroid error, sufficient separation and bounded numerical
fit error, every selected move remains in a covered neighborhood; failed
fits may rest. Correct group inference on every future Gaussian sample,
parameter-state boundedness, acquisition and general distribution fidelity
do not follow from that argument. Current perfect HQ must not be presented
as an unconditional all-time theorem.

Internal diagnostics are also recorded: by warm1200, G parameter norm is
15.75 versus13.27 in the disabled control, and first-fit Jacobian sensitivity
increases. Output quality alone does not rule out growing cancellation
between the native GAN move and its subsequent correction.

This method explicitly adds a sampled-data objective and extra optimization.
The native G curvature bound controls only the GAN proposal; it does not
bound the subsequent neural fit. The receipt records every extra Jacobian
and nonlinear trial. A saved-state pass is not evidence of cold acquisition,
responsiveness after error, or indefinite stability.

## Research informing the next tests

[Dumont, Lacombe and Vialard, September 15, 2026](https://arxiv.org/abs/2609.17167)
connect Gauss–Newton fitting of an output drift with natural-gradient
optimization. Their convergence result uses an implicit proximal scheme,
convex function-space assumptions, and increasing model capacity; it is not
a fixed-network, noisy GAN stability theorem. This motivates separating
output geometry from its neural realization, without supplying our guarantee.
[King et al., 2026](https://arxiv.org/abs/2602.00099) similarly study
function-space versus parameter-space conditioning for shape-learning
residuals. Applying this numerical principle to the current generator is an
experimental inference, not a result established for ParticleGAN by either
paper.

Next gates remain ordered: exact controls and warm 200; dense continuation
through 2400; unchanged cold trajectory and ring; then continuation of the
candidate's own acquired state and response to model error on the same fixed
dataset. A failed gate stops promotion. No unavoidable impossibility has
been established.
