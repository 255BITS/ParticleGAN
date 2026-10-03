# Reserved local mean network-step guard

**Source-only, conditional proposal.** Do not implement or test unless the native
attribution diagnostic shows G movement worsening local means on fixed latent
coordinates. If that direction is absent, reject this alternative. No checkpoint
load, forward, numeric test, training or production source edit was performed.
The earlier inactive post-ra6 function-motion test is retained and not repeated.

There is no DeltaStepGuard class in the frozen package. The usable reference is
the private guard_design.py bounded parameter interpolation; population
stationary_undo_s concerns a different table scheduling decision. New network
rollback, clock and state/cache plumbing would be required.

## Minimal law

After D has stepped, before joint opt_g.step(), freeze the current D feature
epoch, detached panel latent codes and real-only chart for this comparison. The
baseline measurement can run after backward so both candidates use identical
post-training-forward registered buffers. Do not compare G+prior joint motion or
different D epochs as if it were G-only motion.

Fit a bounded current chart from chronological/evenly spaced real FIFO rows,
even rows for geometry/local means/covariance, rank<=8. A scalable deterministic
work bound is M<=min(fill,2*requested_cells*requested_rank) real references and
P<=min(N,4*actual_cells) evenly spaced detached prior row IDs. It uses no mode
labels or full-table Jacobian/forward per trial. A cloned existing birth-death
stream may construct the chart without advancing any production stream or
introducing a new seed. If finite chart/panel support covers less than 1-Q of
its real topology's mass, retain the original step: thin/missing supports do
not authorize blocking exploration. This remains bounded empirical coverage,
not proof that every data component is represented.

For each represented real topology group, compare its fixed weighted panel
mean with its even-real mean in resolved real covariance axes. Use

    E(theta) = sum_g real_weight_g *
               ||C_real,g^(-1/2)(panel_mean_g(theta)-real_mean_g)||^2

Freeze membership and weights during the comparison. Sum squared local errors
so opposite shifts do not cancel. Finite resolved covariance axes alone set the
normalization; omit singular axes/groups without enough fit rows. A candidate
must keep originally eligible probes supported/inside and in their original
real topology group. No group/cell/evaluator can be chosen from an oracle label.

Run the original joint optimizer once. Try only existing fractions
1,.5,.25,.125 of its G parameter delta; select the first finite candidate with
E(candidate)<=E(before) and the unchanged support/group checks. Otherwise restore
G parameters exactly (alpha=0). This is empirical nonincrease, not a statistical
negative-evidence test, positive distribution certificate or arbitrary .20sigma
quality cutoff. It can respond to adverse small mean drift even when the old
sqrt(Q) displacement guard would allow the step. It cannot force convergence
to real means or repair bias already present in a frozen G/table.

## Explicit optimizer, API and clock consequences

Keep Adam/A2 moments, scalar steps, prior coordinates/history and learnable sigma
advanced once. Contract only G network parameters. This changes the realized
network update; it intentionally is not optimizer rewind or unchanged training.
Use exact original full/zero parameter endpoints. All trial reads must preserve
global/owned RNG, registered buffers, modes, gradients and hooks.

Choose alpha **before** the current _settle_observe(0) and EMA update. Pure-G
groups must observe alpha*nominal_lr/base_lr; prior and sigma retain their own
nominal/base increments. roles labels sigma as generator too, so checking role
alone would incorrectly damp its clock. Identify actual G parameter membership;
a mixed G/non-G optimizer group needs separate bookkeeping or an explicit
unsupported fallback. Do not persist a reduced group LR that multiplies the
next schedule/controller step a second time. EMA_G averages accepted G exactly
once; EMA_prior, serving, noise, table population law and 5% birth/death rules
keep their existing order and semantics. No new row moves are authorized.

Fresh deterministic chart reconstruction at each guarded step avoids stale
current-D caches and allows replay from the existing FIFO/weights/streams.
Persist typed guard policy, bounds, alpha/counters and last diagnostic plus any
added group clock state, with atomic schema/config rejection before model load.
If a chart or frozen D epoch is cached across steps instead, those exact inputs
become semantic state: serialize them or deterministically rebuild and match
them on load. Clearing an active semantic cache to a neutral guard on resume
would silently alter continuation. No inherited newborn/history certificate.

## Cost, limits and condition for selecting a test

Per guarded update: one bounded real-D query, one P-row baseline G+D query and
at most four P-row trial G+D queries; O(parameters_G) backup/delta storage,
O(M*head_width*rank+M*K*rank) chart work and bounded O(K*rank^3) covariance
work. Native K128/rank8 gives at most2048 real rows/512 probes. Image-model
forward cost may dominate and could make per-update guarding impractical.
No CUDA cost has been measured. Guarding only reaction steps would reduce cost
but would not bound unguarded intervening network motion; it is not an equivalent
control law.

Panel means can differ from population means, topology can alias supports and
even-real means are noisy/adaptively reused. G changes can also alter kernel
response, covariance and noisy head bias while improving clean local means.
The joint prior update is outside this G-only comparison. This engineering
descent check therefore supplies no emitted quality or finite-sample inference
guarantee. No additional hypotheses, count cutoffs, seeds or quality gates change.

Only if the native fixed-coordinate attribution identifies a material adverse
G direction should root authorize one new bounded saved-input guard-activity
probe. It must first freeze current-D/FIFO/panel inputs and verify an active
nonincrease decision with explicit optimizer/clock/state neutrality scope.
Do not repeat the old displacement test, increase its limits until it activates,
or use oracle center measurements as production acceptance. Without adverse
native G attribution, this added cost/state complexity is not justified.
