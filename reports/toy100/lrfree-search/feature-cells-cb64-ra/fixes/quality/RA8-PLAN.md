# Prospective paired averaging rule

RA8 keeps RA7's exact configuration and training mechanisms. It changes the
condition for serving the matched EMA generator and particle table when the
feature-cell backend is active. The small-population reference path retains
its existing rule.

After each real-FIFO reaction and all copy/birth actions, use one current
critic/chart for the current live and EMA pairs. A row qualifies when its EMA
anchor has p > Q, lies inside the existing count partition, and agrees with
its live counterpart on the real-only support group. Require at least
N - floor(Q*N) qualifying rows in a valid, nonduplicated chart. Q remains .05.
Save a typed decision and expire it by the next real-FIFO turnover; a failed
refresh clears eligibility. The ephemeral chart is not a required resume
cache. Checkpoint law compatibility and atomic rejection must be reviewed.

This is an empirical geometry rule for averaging. It does not claim temporal
stationarity, distribution equivalence, correct learned groups, support for
all latent perturbations, or accurate emitted samples between refreshes.
The existing noisy generation path remains the primary quality protocol.

The fixed saved CPU diagnostic gives 504, 649 and 985 qualifying rows at
updates 500, 1000 and 2000, respectively, versus the predeclared 973 threshold.
It uses observed real data and learned features; no oracle or new emissions
enter the decision. Clean EMA count mismatches remain in the evidence.

The extra work is one EMA population forward in existing chunks per reaction
and O(N*K) fixed-rank chart queries. There is no new all-row Jacobian or
N-by-N graph. Amortization follows the existing FIFO turnover; practical
shared-GPU timings will be reported separately.

Freeze the source/config and independent CPU contracts before the first
numerical quality run. Run the unchanged 2000-update toy first. A strict
final toy PASS receives the full canonical 7000-update grid schedule, all
terminal checks and the independent holdout. Both quality targets and the
required validity, replay and portability checks are required for a
recommendation. Preserve every negative result.
