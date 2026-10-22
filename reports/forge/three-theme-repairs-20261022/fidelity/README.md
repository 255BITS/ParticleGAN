# BISPF fidelity repair: tail-sensitive projected transport

**Candidate `bispf-fidelity-quartic-v1` is a proposed global recipe; training is unmeasured.**
The common three-theme comparison owns registration, budget, source freezing,
the single comparison table and subsequent measured readout. This proposal does
not alter the archived leader or any ordinary qualification/default.

## Evidence and hypothesis

The source-bound selected BISPF evidence is the ordinary-quality readout at
`/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json`,
SHA-256 `50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe`,
source `d9d72d53761c898bce20102e1995fc3d899c1447`, seed0. Original tensor trees
are unavailable; this does not authorize an unchanged historical rerun. The
new comparison executes both its declared control and changed recipes on its
own common frozen source.

Selected unequal-width full covariance error9.68690 coexists with a .66325
four-sigma core error in its dominant narrow failing component (full37.34861).
Intensity has balanced [16,16] nearest assignments but only19/32 clean images.
Native grid holdout precision is .30497 against .97; conditional center error
and shape also fail. These observations motivate a joint output-fidelity
question; none identifies the underlying gradient cause.

[Round3](../../bcap-physics/round3/README.md) already rejected global Cauchy MMD:
unequal-width core improved .444178→.259211 while full error worsened
6.287559→13.431491. Its bounded discrepancy can underweight small remote
contamination. [Round5](../../bcap-physics/round5/README.md) balanced assignment
repairs sustained width while regressing rare/native; prior-only routing and
output-force filtering also fail globally. The
[core/full negative controls](../../bcap-three-phase/README.md) prohibit treating
aggregate/core gains or active counters as a density repair.

Hypothesis: retaining adversarial learning while adding an unbounded projected
quartic output cost will strengthen correction of remote contamination relative
to small local errors. It might improve full component density and image
fidelity without changing the latent sampling/regularization law. This is a
new empirical contrast, not a claim that fourth-order transport must succeed.

## Exact trainer delta

The [schema3 declaration](../../../../configs/forge/ideas/bispf-fidelity-quartic-v1.json)
inherits `bispf-optimized-quintic-v1`, changes only
`fidelity_transport_weight=1`, and adds
[sliced_quartic_transport](../../../../particlegan/fidelity_repair.py) once to the
total G/prior objective before the existing constraint-geometry backward call.
It is not an additional protected objective. The same already-generated fake
batch and already-consumed generator real batch supply every term. No new
network forward, random draw, geometry label, target-component oracle or
evaluation sample enters. The prior remains learned with unchanged SBNP .01;
the critic and optimizer laws remain unchanged.

Flatten sample coordinates. Sort equal-mass projections of fake and real rows,
then average `(projected_fake - projected_real)^4 / scale^2`, where scale is
detached mean real coordinate variance. Exactly constant real batches use
scale1 in original output units, rather than a precision denominator. Dimension1
uses the identity; dimension2 uses32 evenly spaced angles in [0,π); higher
dimensions use up to32 evenly spaced coordinate axes plus32 deterministic unit
trigonometric directions. Construction and evaluation consume no random stream
or persistent state. Memory grows linearly with output dimension.

This is exact empirical scalar W4⁴ in dimension1 and a finite-direction
surrogate elsewhere. For nonconstant batches it is invariant to common
translation and positive global rescaling, up to floating arithmetic. Constant
batches retain their declared physical units. It is neither full multivariate
transport nor rotation invariant. A single sorted residual doubled has16times
the loss and8times the restoring gradient. No component-moment guarantee follows.

## Prediction, falsifier and stopping rules

The shared matrix must compare one control and this one global configuration
on all11 selected BISPF failures, with ring16 acquisition, mode hold, stripes
and Gaussian smoke guardrails. Preserve protocol seed0, public deterministic
initialization, each task's architecture/target/prior/capacity/sampling, exact
seen batches, named checkpointed streams, update allowances and evaluation
cadence. No identity/zero fixture substitutes for the shared initializer.

Primary prediction: unequal-width improves its complete five-check terminal
gate (including full covariance≤.85, mass, HQ and minimum eigenvalue), rather
than only the four-sigma core. Fidelity predictions: intensity reaches its
unchanged≥29/32 clean images and two populated levels for five terminal checks;
native improves its original full precision/placement/shape gates. These are
separate tasks with distinct sampling laws, not pooled covariance scores.

Falsifier: lower quartic/core residual without a new sustained full-task PASS,
or any regression on the declared passing guardrails, rejects this exact global
replacement. Quantile pairing noise, globally scaled weak narrow-component
forces, stronger remote forces and shared-network distortion are explicit
competing explanations. The mechanism cannot guarantee correction of all
image pixels or rare populations from32/64 projections. No coefficient/seed
sweep, extra continuation, best-checkpoint substitution or gate change follows.
Any one permitted substantive successor requires actual measured support and
a new frozen declaration.

## Software verification and artifacts

`python -m pytest -q tests/test_bispf_fidelity_repair.py` checks scalar numerical
loss/gradient, finite zero force, row permutation/translation/scale invariance,
quartic tail sensitivity, real-target detachment, RNG preservation, constant
targets, finite differences and malformed input. These are mechanism unit
controls, not public-API toy quality claims. The coordinator must additionally
verify the Recipe/trainer consumer, inactive baseline compatibility and named
stream pairing in the common source before paid execution.

All real training must use the public ParticleGAN API and retain numerical
PASS/FAIL receipts plus actual-training GIFs. Bulk stdout, update traces,
checkpoints, JUnit and tensors belong under
`/mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022`; only compact source-
bound summaries and goal GIFs belong in Git. Tail paths will be supplied by the
common coordinator. No execution or trained result is claimed in this commit.
