# BCAP-pure: owner-requested 10× training-budget diagnostic

The owner requested exactly two runs: scalar Gaussian acquisition for 10,000
updates and 16-mode ring acquisition for 4,000 updates. Both use the complete
winning dualnorm recipe: G/E step .012, D step .018, sampled prior row step .03,
momentum zero, epsilon 1e-8. Both rate floors stay 1; no effective annealing,
noise warmup, continuous controller, loss or BCAP cap/coefficient change.

This study tests whether extra updates resolve the two failures diagnosed in
the [gate audit](../bcap-pure-tier1-gap-audit/README.md). It is an explicit
research diagnostic, not an ordinary Tier 1 replacement. Original numerical
thresholds remain unchanged. A 10× pass does not rewrite the original
"within 1,000/400 updates" results, the four other successes, or calibration.
The [single current leaderboard](../technique-inventory.md) retains its original
source-bound 4/6 measurement until a separately reviewed task policy changes.

The only experimental changes are the allowance, additional original-spaced
observations, full final state retention and explicit diagnostic evidence scope.
Preserve architecture, target law, batch 128, 256 learned MoG locations at
sigma .025, public initialization, named streams, protocol seed 0 and clean
live scoring. Start from scratch because the original runs retained no resumable
checkpoint. There are no additional seeds, optimizer configurations or retries.

The new opt-in evaluator consumes 240 observations at
`ceil(i * original_budget / 24)`, i=1..240. It requires all metric conditions
and five terminal passing observations together. Report original, 2×, 4× and
10× prefixes from the same two trajectories, first jointly passing observation,
first confirmed stable interval, terminal suffix and late instability. These
prefixes are not independent attempts. Compare the first 24 saved samples and
metrics with the archived winner and explicitly report any discrepancy; old
final model/optimizer/RNG tensor equality cannot be checked from unavailable
states. Final new states include every consumed named stream.

Predictions are scoped and preregistered: ring's late improvement may become a
sustained pass; Gaussian's fluctuating CDF may remain a failure. A prediction
is observed only if its whole unchanged metric conjunction has five passing
terminal observations at the declared endpoint. Keep partial improvements and
early passes visible. Stop at 10× or the full task timeout, never extend the
budget after seeing results. The two complete reservations are 1,200 seconds
for Gaussian and 3,000 for ring, maximum 4,200 worker-seconds; use one job per
GPU. Capacity does not authorize further work.

```sh
PYTHON=/home/martyn/dev/ParticleGAN/.venv/bin/python
BUDGET_QUEUE=/home/martyn/dev/ParticleGAN/runs/forge/bcap-pure-budget10x-v1-queue
mkdir -p "$BUDGET_QUEUE"
$PYTHON -u reports/forge/bcap-pure-budget10x-v1/run.py plan > "$BUDGET_QUEUE/plan.log" 2>&1
# Commit the reviewed implementation and declarations before freezing execution.
$PYTHON -u reports/forge/bcap-pure-budget10x-v1/run.py run > "$BUDGET_QUEUE/driver.log" 2>&1
tail -F "$BUDGET_QUEUE/driver.log"
# Each attempt also has run.log with all 240 scored observations.
```

No threshold change is part of this study. After completion, recommend whether
to adjust acquisition allowances, keep searching the optimizer, or investigate
another explicitly scoped cause. A threshold revision must justify what the
experiment should detect and retain existing destructive/oracle controls;
moving a cutoff solely to turn a measured failure into a pass is not evidence
of better training.
