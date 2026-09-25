# Archived training-condition diagnostics

These observations have no effect on the current PR's eligibility or practical count. The candidate owns its training recipe, and architecture is a separate axis. They remain available for future robustness work; no numerical result has been erased or changed.

The old `stress_nominal_ring` label means the ordinary eight-Gaussian target at 1,200 updates. It is retained as a reference run, not an additional required toy. The established required ring regression is still required, and [longer training](LONG_TRAINING.md) has its own toy.

## rp_logistic_bcap3

| Archived condition | Live result |
| --- | --- |
| stress_fast_critic | FAIL |
| stress_slow_critic | FAIL |
| stress_small_batch | FAIL |
| stress_large_critic | FAIL |
| stress_nominal_ring | FAIL |

## rp_logistic_bcap10

| Archived condition | Live result |
| --- | --- |
| stress_fast_critic | FAIL |
| stress_slow_critic | FAIL |
| stress_small_batch | FAIL |
| stress_large_critic | FAIL |
| stress_nominal_ring | FAIL |

The original controller study's tiers and balanced scores remain historical records. This scope revision was requested for the formulation-default comparison after that study.

[Exact settings, metrics and artifact references](leaderboard.json).
