# Separate overnight BCAP extension

The user requested additional capacity to keep both GPUs busy toward **6 a.m.
America/Denver on 2026-10-09**. At the operational snapshot, 44 original
candidates remained. Completed DualNorm word tasks averaged 701 worker-seconds;
future optimizer costs and gate advancement remain uncertain. The original
batch may finish before 6 a.m. This separately frozen **24-candidate** batch
provides additional useful work while preserving the original 72 declarations.

All 24 use positive BCAP, the unchanged public smoothed/convolution-enabled
DualNorm implementation, seed0, and the same frozen full Tier1/Tier2 view.
Twelve zero-momentum recipes vary positive smoothing `.0003,.003,.03`, paired
D/prior multipliers `(.5,2.5)` or `(2,1.25)`, and BCAP strength `1,4`, with
positive cosine floor `.15` from horizon fraction zero. Twelve recipes enable
the existing network momentum `.5,.9` under a separately declared structural
base, paired with smoothing `.0003,.003` and positive floors `.15,.5,1`.
G/E base rate is `.012`, cap is `1`, and penalty cadence is `1` throughout.
Sampled prior rows retain zero momentum. Every complete recipe is distinct
from the original 72; this is not a seed experiment or a repeat.

Hypothesis: intermediate/stronger positive smoothing, alternate player pacing
or network momentum can improve sustained retention without losing acquisition.
Falsification uses the same unchanged 6/6 Tier1 and at-least-10/21 Tier2 success
criterion. The domain comes from supported public settings and archived pacing
and smoothing evidence; runtime/accounting measurements trigger extra capacity.
No interim configuration ranking or scientific result chooses this domain.

The initial study remains immutable. This extension has its own queue, campaign,
manifest, 43,020-second candidate ceilings and **1,032,480-worker-second** shared
ceiling. It launches after the first campaign concludes, using both A6000 GPUs
with one worker each. All independent current-tier tasks finish, and every
Tier1 survivor advances through Tier2 automatically. No retries, adaptive
expansion, Tier3 or default adoption follows. The exact source and runtime
must match the original 72, and final selection uses the same PASS-count/hash
objective across the matched 96-candidate union. Publish to the existing single
goal leaderboard and retain both original campaign identities and costs.

The extra batch may finish after 6 a.m.; the requested finish policy is recorded
in its launch receipt. A stop-at-6 policy, if selected, pauses new launches and
lets active jobs finish; remaining cells stay unmeasured, without shortened
training or invented qualification.
