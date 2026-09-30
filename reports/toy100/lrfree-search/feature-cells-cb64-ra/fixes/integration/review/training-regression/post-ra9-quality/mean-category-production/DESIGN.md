# RA10 bounded mean copies

Prospective private package, backend9 and trainer5. Configuration is the exact
RA9 byte stream. Of 29 original modules, feature_cells.py and birth_phase.py
change; mean_transport.py is added. The other 27, including training.py, stay
byte RA9. No model optimizer, noise, population, serving, scoring or horizon
law changes. Root's frozen RA10 plan and selection bind this implementation.

## Fixed witness and refreshed action context

The actual fitted K gives one common Q/(3K+3) cutoff, Q=.05: K categorical
mass, 2K inside/outside, two global support counts, and one scalar witness.
Existing conditional count pvalues are freshly thresholded. birth_phase
validates the same family. The even-only resolution cap and topology stay RA9.

Before the count-driven prefix, fit real group center/RMS scales from even
references. Radially clip standardized projected features at R=sqrt(rank/Q),
rank<=8. Even group means and current clean EMA offsets fix unit-or-zero u.
Every odd row supplies X=u_g dot (psi(odd)-mEMA_g), including zero-direction
rows. No importance weights or score-selected group omission are used.
The known range is 4R, variance uses ddof1, and alpha=Q/(3K+3):

    lower = mean(X) - sqrt(2*variance*log(2/alpha)/n)
            - (7/3)*4R*log(2/alpha)/(n-1)

Strict lower>0 is the only trigger. n<=1, nonfinite values, degenerate geometry,
duplicate references, missing/insufficient even groups, zero scales or missing
EMA groups veto. Unit directions/offsets and the score are never recomputed
after odd-count-driven actions. This is conditional iid bounded-score algebra
and empirical negative evidence with trained D/shared FIFO, not a population,
stationarity, equivalence or quality certificate. Nonlinear learned features
of clean anchors do not certify noisy emitted means. The witness's true-mixture
weighting differs from the action objective's even empirical group masses.

Apply the original mass/support/global/novel/isolation phases, then refresh
FAST cache. For a firing witness with budget remaining, query current EMA and
recompute its group means AND counts using the fixed chart/clip/targets. These
are actual replacement denominators; the ranking u remains the pre-prefix u.
The objective is sum(even_mass_g * ||even_mean_g-current_EMA_mean_g||^2).

## Fourth ordinary copy phase

Reserve all prior ordinary/isolation children and parents, new latent children
and source seeds. Unique mean parents/children are mutually disjoint and
disjoint from the full reservation union. Preserve one jointly eligible row
in each occupied fine cell in each view. Pools have at most 64 parents and 64
children per joint category signature, with stable fixed-u ranking. Attempt
no more than floor(.05*N)-earlier_ordinary pairs; do not retry a failed preview.
Original child and parent must be finite, p>Q, inside in BOTH FAST and EMA,
same own fine category per view, and same even-real topology group across views.

Draw the original dedicated copy jitter matrix once. Both priors use their
own bounded geometry and common lineage. Query exact proposed latent
coordinates directly through G/EMA_G and the learned D head, without a latent
or output-noise draw. Require finite coordinates/features and actual same
inside category/fine cell/real group with p>Q in both views. Bounded chart rows
transfer in bulk to CPU. Stable sequential float64 objective arithmetic is
the declared policy, not a claim of GPU arithmetic bit equivalence. Every
accepted actual offspring strictly reduces that current EMA objective. Before
write, verify actual device-produced features against the final virtual ledger.

The ephemeral prepared packet contains exact paired coordinates, features,
parent row optimizer tensors/history and an owned epoch. Validate it fully
before any write. Epoch pins parameters/prior/lineage/chart/cache, exact
buffer mappings/bytes, bandwidth and stream identity/state AFTER the draw.
Commit reuses stored bytes, registers lineage once, refreshes FAST cache and
draws nothing. Old copy moment/history inheritance is retained; novel zeroed
history semantics are separate. Rejected previews consume the one declared
draw. All accepted mean children join the original moved_rows vector and
unchanged caller RowEvidence reset and population rebase. Kind4 is mean; kind3
remains novel. All ordinary copies/new latents share the unchanged 5% budget.

All extra forwards preserve global CPU/device RNG, all dedicated streams,
module modes, registered buffer mappings/values/nonpersistent metadata and
gradient objects/values. Arbitrary unregistered external state is outside
this contract. Buffer copy restoration can increment versions; epoch checks
buffer bytes/mappings instead. Parameter identities/versions remain pinned.

## Bounded work and state

Pre-prefix witness adds one full EMA query per valid reaction. A firing phase
with remaining budget adds a second full EMA query and at most two preview
queries of the residual budget. The original fresh final paired-average query
remains explicit after all actions; no cached EMA lease reuse is claimed.
Chunks=256, rank<=8, pools=64; chart work is O(NK+N*rank), no N^2 distances or
full-population Jacobians. Geometry keeps the original bounded neighborhood.
mean_forward_rows/mean_preview_rows count attempted forward-row upper bounds;
nonfinite coordinates may veto before their proposed features are evaluated.

Only scalar last.mean_transport and typed counters/action ID lists persist.
Charts, packets, CPU ledgers and timing do not become semantic state. Typed
metadata binds the actual K/rank/snapshot/step/odd n/alpha, scalar range and
bound arithmetic, residual attempt cap, phase sums and unique reservations.
Initial/invalid/veto/firing paths are explicit. Backend8 rejects before model,
optimizer or RNG load. Trainer5 stays byte RA9. A positive stamp is still the
original anti-blur serving lease, not this mean witness.
