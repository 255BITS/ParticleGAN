# Review before refilling the constant-rate lane

No winner from the first three public API proposals. Keep every source and
measurement. This review authorizes a new attempt within the existing search;
all COMMON.md requirements and the three-proposal/one-worker limit still apply.

Previous attempt:
`/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T211209Z-1398466/constant_rate_stability/20260926T211209Z-1398479`

Read its `result.md`, `tests.jsonl`, declarations, source ZIPs and update traces
in `repo/experiments/constant_memory/` before implementing a successor. The
supervisor's independently derived scores are in `first-results.json` beside
this review. API-C3's local assessment is the final failed proposal.

## Measured failures and useful ideas

- API-C1 replaces KA2's frozen/reseeded reference with a continuously moving
  .99 EMA, keeping all public rates constant. Initial arrival 580; retention
  172/183 before the change, with eleven departures. Changed-target arrival
  after 570 updates; 164/164 afterward. Reference motion helps recovery but
  does not prevent self-instability.
- API-C2 bounds each applied Adam coordinate displacement by its nominal group
  LR, preserving moments and sparse prior behavior. Initial arrival 1960;
  45/45 retained before the change. Changed-target arrival after 1890 updates;
  32/32 afterward. The 7500-update stationary run then retains only 296/555
  observations after arrival, with minimum HQ .0344 and five modes. The first
  collapse begins at 2940. Many bounded coordinates move together: generator
  update L2 grows from .037 at 2930 to .344 at 2950. Coordinate bounds do not
  control the collective adversarial dynamics.
- API-C3 adds a bounded optimistic displacement correction. Neither original
  nor changed-target arrival is observed through the declared 4600-update
  window. Do not repeat it unchanged or reinterpret non-arrival as a pass.

The historical prehold window is not an acquisition deadline. C2's 45/120
prehold count is explained by late acquisition; its later stationary collapse
is the meaningful rejection. Preserve this distinction in all new reports.

## Next mechanism direction

Keep the constant-rate lane distinct from autonomous rate control and real-data
drift sensing. Investigate the coupled adversarial update itself, using the
measured collective excursion as the failure to address. A justified
predictor/corrector or proximal game update is a useful next direction; inspect
retained related experiments first and select one coherent mechanism. Simply
shrinking a fixed LR, another clipping coefficient, or repeating C3's optimism
is not a new hypothesis. Internal numerical substeps are compatible with the
user's requirement; there must still be no caller-managed acquisition/maintenance
phase or predetermined ending.

You may reuse the predecessor's continuous-API plumbing and evaluator after
recording exact copied source hashes/diffs. Work in your own new checkout;
never modify the predecessor. Reuse tests and evidence when the code is
unchanged, but a changed learner must earn its own quality scores. Start a
new real candidate test promptly; do not rebuild the benchmark.

## Shared work and remaining obligations

The reversible_precision lane's API-RP2 is the current single-shift survivor
and owns the matched K3P comparator if it survives longer tests. Do not duplicate
that comparison. Its success is not a score for this lane.

The data_drift_mobility lane owns a common CUDA checkpoint investigation:
immediate restored state is equal, yet file-resumed updates differ slightly
from original uninterrupted ones. All three approaches have seen this; two
restored branches agree, and short same-process restore checks pass. No root
cause is established. Do not duplicate expensive localization or change a
mechanism to conceal it. Keep exact continuation unpassed until resolved.
Ordinary non-fused Adam CPU scalar step counters are valid metadata; relocating
them changes KA2 tensor arithmetic and must be an explicitly distinct policy.

Use unique candidate names (for example API-C4 onward). No seed sweeps, hidden
target information, research-host substitution, copied passes or PR merges.
Qualification still requires the unchanged whole learner and actual public API.
