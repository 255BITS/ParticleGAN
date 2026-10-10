# Paired average: qualified CPU proposal

The proposal changes large feasible feature-cell serving eligibility to a
fixed current clean geometry check. At least N-floor(QN) paired rows must
have EMA p>Q, EMA inside the existing even-fitted region, and the same real-only
topology group as FAST. The chart must vary, remain below the existing duplicate
limit, and all observed coordinates/features must be finite. This is an
empirical anti-blur control, not stationarity, distribution equivalence or a
quality certificate. It adds no statistical count test or new Q budget.

| Saved RA7 update | Joint coherent /1024 | Required | Serving eligibility |
| ---: | ---: | ---: | --- |
| 500 | 504 | 973 | false |
| 1000 | 649 | 973 | false |
| 2000 | 985 | 973 | true |

The fitted geometry and check have no oracle labels or new emissions. Saved
EMA clean performance motivated investigation; it is not a serving quality
result. Source diagnostics separately document that final EMA is narrower
than real references and does not establish count equivalence.

## Placement, work and observational state

One additional EMA G/D feature query runs in existing chunks after all
ordinary copies, isolation copies and new-latent births. FAST uses its already
refreshed metric cache. The fixed K graph and chunked assignments remain
bounded: one N-row forward, O(NK) distance work and O(N*width*rank) projection,
with no N-by-N graph, Jacobian, extra reaction draw or count-family change.
The extra forward has its own deterministic `paired_average_forward_rows`
diagnostic; distance/projection work is included in existing work counters.

The new measurement forks global CPU/current-device RNG and restores every
trainer-owned stream and the reaction stream. It restores D/EMA registered
buffer mappings, values, nonpersistent flags, individual modes and parameter
gradient objects/values, including exceptional exits. It does not promise to
restore arbitrary unregistered Python state or external RNG objects.

## Semantic lease and load

Backend schema7 stores typed eligibility, intersection counts, geometry sizes,
snapshot serial and reaction step. The lease is valid only while fewer than N
new real rows have arrived and its snapshot remains current. Between reactions
it is explicitly a bounded stale view; G/D updates are not proven to preserve
the predicate. There is no independent optimizer-step TTL. Loading leaves the
ephemeral chart absent and preserves the scalar decision exactly.

The loader validates required counts, both intersection bounds, finite early
branch zeros, chart rank/validity, same-reaction geometry/duplicate metadata,
and future reaction steps before mutation. Old backend6 rejects atomically.
Trainer schema5 suffices because all new semantic state is validated in the
backend before model/optimizer loads. The trainer adds only serving dispatch
and cross-check of saved reaction step. Small infeasible/reference backends
retain their legacy gate and reaction law.

## Evidence

`gate-receipt.json` passes 42 fixed controls: saved early/final gates, supported
wrong-group cases, exact 51/52 shared exception boundary, finite overshoot,
nonfinite/duplicate/invalid charts, FIFO boundary/overshoot, chart-free load,
backend6 rejection, 16 malformed typed/consistent stamps and odd801 fitted
ceil401/calibration floor400. No large odd401-cell fit was needed.

`reaction-RA7.json` and `reaction-proposal.json` pass matched forced-ready
CPU reactions on saved1250/2000 states. Plans, original event fields, live/EMA
latent writes, moments/history, graph, gradients, models/buffers/modes, FAST
cache and reaction stream have identical hashes. Both use51 ordinary actions;
1250 includes4 novel births. Post-action coherent counts are931 (veto) and988
(pass). Additional gate metadata and geometric work counters are declared
comparison exclusions. These are fixed mechanical current-device reactions,
not optimizer steps or historical CUDA replay.

`neutrality-receipt.json` passes12 controls with tiny eval hooks that draw global
and all owned streams, mutate/rebind registered buffers, register a temporary
nonpersistent buffer, and mutate gradients. Owned state is exact after normal,
exceptional and supported nonfinite-output paths. The nonfinite captured head
produces a false stamp; unrelated custom forward exceptions propagate after
state restoration.

All original RA7 sources/config/checkpoints remain unchanged. Full trainer
serving/replay and prospective CUDA toy/grid contracts are root-owned after
composition. No CUDA, training, optimizer steps, new emissions, oracle policy,
noise changes, forced average, quality claim or default promotion occurred.
