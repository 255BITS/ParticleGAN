# D Armijo plus positive own-secant G response

The one declared combination fails the cold trajectory gate. It passed the
passing-state warm filter, then scored 0/24 cold trajectory checkpoints and
final identity MSE `.035842` against the unchanged `.02` limit. No cold ring,
fixed-target long hold, or shared 22-task run followed.

The combination used the **frozen** D total-loss Armijo adapter at SHA-256
`8679fc5f00b78c17e20a93ed69404a69ea80c77e3657fa0c3c04f49008630be3`
with Armijo fraction `.1`, at most 12 halvings, and verified exact zero-step
rejection if every positive trial fails. Only the G rule was changed: after
D accepts, G uses the positive rank-one own-secant resolvent described in
[the standalone report](alternating-positive-secant-report.md), with PR #82's
`.25` scalar curvature bound as its deterministic nonpositive-curvature
fallback. D then G order, same-sample/noise replay, one Adam moment update per
player, the original noise horizon, and fixed nominal rates are preserved.
The combination has no target-label, quality-oracle, or elapsed-time input.

| Stage | Result | Main receipt |
| --- | --- | --- |
| Scheduled identity / constant warm controls | 200/200 / 6/200 | Exact identity full-state parity |
| Combined warm continuation | **PASS 200/200** | Minimum HQ .92944; all 200 D proposals accepted whole |
| Cold trajectory | **FAIL 0/24** | Final MSE .035842, sustained suffix 0 |
| Cold ring and later gates | Skipped | First failed host stops the screen |

The warm trace matches the standalone positive-secant experiment because
all D Armijo proposals were accepted at factor one. Cold trajectory differs:
D accepted 168 partial proposals and no zero-step rejections; the G rule used
its positive rank-one response on 396/400 updates. The final five trajectory
MSE values stay near `.036–.039`, so this is not a delayed one-checkpoint
near miss. The D line search required 1,412 gradient callbacks per player
over 400 outer updates; Adam moments advanced exactly 400 times per player.
Analytic tests verify D accepted-point order and the exact G secant result,
as well as the D exhausted-retry zero replay. A frozen-host smoke test checks
same-sample RNG replay and one moment update. These observations do not imply
either constituent method fails in other formulations; they reject this
specific combined fixed-rate recipe.

The [manifest](continuous-evidence/alternating-armijo-secant/manifest.json)
binds two compressed raw archives with source copies/hashes, full per-update
controller and rate/moment receipts, effective configurations, complete host
observations, and logs. Reproduction uses fresh paths and the warm result is
required before a cold run:

```bash
python -u reports/toy100/alternating_armijo_secant_probe.py --phase warm --output NEW_WARM
python -u reports/toy100/alternating_armijo_secant_probe.py --phase cold --previous NEW_WARM/summary.json --output NEW_COLD
```
