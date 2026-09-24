# Why the first200 full snapshots differed

The version1 hold completed its training, but its final gate rejected unequal
first200 snapshot hashes. Quality observations and all first200 update and
correction records matched. The failure was retained; no cold promotion used
that failed gate.

The [passive capture](allocation_prefix_hash_diagnosis.py) reran only the
scheduled 1,000-update prefix and 200 active updates for each declared host
horizon, aborting before update1200's EMA/checkpoint. Both captures reproduce
their original raw snapshot hash exactly, and all200 records/corrections are
exact. They differ in precisely two fields:

| `noise_policy._counts` field | Warm1200 | Hold2400 | Difference |
| --- | ---: | ---: | ---: |
| `output_eval_calls` | 510 | 480 | 30 |
| `output_eval_elements` | 2,095,080 | 1,971,840 | 123,240 |

There are **15 extra native measurements**, independently enumerated from
the frozen host's two observation paths before the capture boundary:

* The original 24-check recorder uses steps50,100,…,1200 for a1200 horizon,
  but100,200,…,2400 for2400. Warm therefore has12 extra checks before1200:
  50,150,…,1150.
* The host's native live tail adds1050,1100,1150 for the1200 horizon. Its
  ordinary every200 live evaluations occur in both runs.

Each measurement calls the noisy generator twice:4096 sampled points plus
12 support points, each with2 coordinates. Thus the exact differences are
`15×2=30` calls and `15×2×(4096+12)=123240` elements. The additional dense
observer has identical cadence before this boundary and contributes no
difference. Evaluation restores the training RNG.

The initial hypothesis counted only the three native-tail evaluations and
predicted6 calls/24648 elements. Its assertion correctly failed. That initial
source/result is archived alongside both full captured snapshots; the12
extra original recorder measurements account for the remaining difference.
No training field, optimizer state, EMA tensor, RNG state, noise clock,
training counter, or noise history differs.

[Version2](allocation_continuous_probe_v2.py) preserves both raw first200
snapshot sidecars and their file/content hashes. Its
[reconciler](allocation_snapshot_reconciliation.py) independently enumerates
the observation schedules, requires exactly the two stated counter deltas,
subtracts them from a copy of the warm snapshot, and requires the **entire**
adjusted snapshot hash to equal the hold snapshot. No field is ignored.
The summary names this `first200_training_parity` and retains the full
`first200_snapshot_reconciliation` receipt; it does not label raw snapshots
identical. The version1 driver remains unchanged.

The changed source epoch requires new warm/hold gates. Original PR84 controls,
the original24 observations, every-update quality checks, and all thresholds
remain unchanged. This is an observer-accounting repair, not a training or
quality-controller change. Eleven focused tests cover the captured snapshots,
the v2 cold gate, and rejection of altered models, moments, RNG, clocks,
histories, training counters, unexpected metadata, or a wrong allowed delta.

Evidence is in
[the reconciliation receipt](continuous-evidence/anchor-prefix-hash-diagnosis/reconciliation.json)
and [manifest](continuous-evidence/anchor-prefix-hash-diagnosis/manifest.json).
