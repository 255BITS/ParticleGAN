# Independent architecture replays

The overlap control, new unequal-width PASS and closest rare failure reproduce every live/EMA
checkpoint, optimizer receipt, recipe, architecture, schedule action and verdict
exactly. Only timing fields are excluded. Replays are validation evidence and do
not add leaderboard points.

| Replay | Live | Final passing observations | Evidence |
| --- | --- | ---: | --- |
| Existing raw Softplus96×3 overlap control | PASS | 10/24 | [Checks](control/checks.json) |
| Raw Softplus128×3, beta8, unequal widths | PASS | 5/24 | [Checks](unequal-width/checks.json) |
| LayerNorm Softplus96×3, beta4, rare component | FAIL | 0/24 | [Checks](rare-near-miss/checks.json) |

The width witness passes all metrics at the five observations from step 1000 through 1200. Its minimum
component eigen ratio across those five observations is .2561695, above .15;
HQ stays at least .93042, above .85; covariance error stays at most .628182,
below .85. This is a live-weight pass; EMA also passes independently.

Each folder retains the exact source archive, source hashes, replay payload,
reference byte hash and field-by-field comparison results. Run from the repo root:

```sh
python -m reports.transfer_suite.unadjusted.runs.remaining-replays.verify
```

Use `benchmarks.transfer_suite.replay_shared_architecture` to repeat a retained
architecture episode. The verifier requires all comparison fields and a complete
source manifest; independent mutation checks reject incomplete evidence.
