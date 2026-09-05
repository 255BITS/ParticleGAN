# Corrections to the sparse study

The tables and plots checked into this directory record the original runs.
They have **not** been rerun with the corrections below. Other metrics from
those runs remain historical observations; the affected comparisons and
particle-purity values must not be treated as corrected results.

## X-only gradient penalty

The original `gp_on_y: false` branch applied the penalty separately to
`(x_real, x_real)` and `(x_fake, x_fake)`. For `g_interp_cap` and `e_interp`,
this removed interpolation as well as the y derivative. The `ucd_gpx` row in
`TABLE_ucd.md` and its curves therefore compare two changes at once. The old
claim that the penalty should include the symbol input is withdrawn; the
effect of excluding only that derivative requires new runs.

The corrected implementation passes the joint real and fake inputs to the
same regularizer for both settings. It detaches y inside the critic when
`gp_on_y` is false, preserving each interpolated (x, y) value while excluding
the y derivative from the norm. Endpoint arms continue to evaluate the real
and fake endpoints. New summaries record `implementation_versions.x_only_gp: 2`;
the analyzer keeps older affected runs in a separate legacy group.

## Particle class purity

All numerical `ppur` entries in the original tables are invalid: the metric
associated final outputs with freshly sampled, unrelated particle IDs.
Those historical values are retained and labeled here, not recalculated.
New runs retain the actual IDs in `final_samples.npz` as `particle_idx` and
record `implementation_versions.particle_class_purity: 2`. The analyzer
omits purity values from older summaries.

Corrected purity is the mean dominant-class fraction for each particle,
counting only draws whose generated nearest-mode class matches the requested
class, and requiring at least four such draws per particle. It measures
specialization, not generation quality. A class-conditioned G can use one
shared particle successfully across all classes, giving low purity; a
particle that always generates one class can have purity one. Small sample
counts bias the dominant-class fraction upward. The metric is not reported
for class-partitioned tables, which are class-specific by construction.

## Categorical hardness

`G.eval()` always emits an argmax one-hot symbol. Consequently `y_hardness`
is always one at evaluation, even for a completely uncertain softmax head.
It does not establish that training-time soft outputs saturated. The
categorical accuracy results describe the hard evaluation output; use the
logged `sym_conf` or measure training-time hardness to assess confidence.
