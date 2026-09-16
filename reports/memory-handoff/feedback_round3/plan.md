# Round 3: bounded generated feedback

Eight 2k scouts use the existing durable two-GPU pipeline, 10k learning-rate
schedule, fixed particles per episode, learned GRU32 D memory, feedforward G,
default API exact B-cap, no clipping or EMA. No seed sweeps. Long generated
trajectories are evaluation-only; there is no path critic or cold/warm path loss.

For a target x_t, encode real observations strictly before x_(t-1), propose
xhat_(t-1) = G(z, M_(t-1)), detach it, and replace the last observation with
(1-a)*observed_(t-1) + a*xhat_(t-1). D writes that point; G predicts x_t and D
scores real/fake x_t against the identical cached memory. Unselected contexts
remain teacher-forced. Empty contexts always remain empty. The same z is used
for proposal and scored prediction. D alone trains writer parameters. Rebuild
memories/proposals after the D update; sample selection is shared across phases.

This is two G calls per D/G phase, with at most one generated write before a
local target. Gradients do not pass through the proposal, but D gradients do pass
through the prefix and its replacement write. No gradient is clipped. Partial
replacement is a data interpolation, not a parameter-gradient operation.

The real x_t is a recovery target. For a badly wrong proposal, x_t may not be the
natural continuation of its generated prefix. Frequency, interpolation, ramp,
and minimum-prefix scouts test this assumption. This is not training on fully
autonomous state distributions and need not resolve accumulated long-term drift.

| Scout | Replacement probability | Strength | Minimum target prefix | Ramp | Max prefix |
|---|---:|---:|---:|---:|---:|
| feedback_p25 | .25 | 1 | 1 | none | 63 |
| feedback_p50 | .5 | 1 | 1 | none | 63 |
| feedback_p100 | 1 | 1 | 1 | none | 63 |
| feedback_p50_mix50 | .5 | .5 | 1 | none | 63 |
| feedback_p100_mix50 | 1 | .5 | 1 | none | 63 |
| feedback_p50_mature | .5 | 1 | 4 | none | 63 |
| feedback_p50_ramp | .5 | 1 | 1 | 1000 updates | 63 |
| feedback_p50_ctx16 | .5 | 1 | 1 | none | 16 |

Probability is conditional on eligibility; zero-prefix examples are excluded.
No auxiliary losses in these scouts. A separate RNG stream selects replacements;
data/particle/context streams and shared initialization match prior dense4.
Historical dense4 and predictive10 are included for comparison without retraining.

53 focused tests passed: active B-cap, causal context and replacement timing,
no gradient through G proposal, D-only writer training, G/prior gradients in the
G phase, exact resume, zero-strength equivalence, minimum-prefix masking, and
exact bounded G call counts. A batch128 GPU smoke with1024-point evaluation passed.

Queue `runs/memory_path/feedback_round3`; completed-only reports refresh before
completion notifications. Central log remains:

```sh
tail -F runs/memory_path/core_round1/train.log
```

Rank by full256/1024 cold circles, original-orbit continuation after prefixes8/32,
both directions, stopping and continuous errors. Do not equate local fit or a
late-emerging oscillator with a solved cold start or faithful continuation.
