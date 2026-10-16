# Wall-clock outcomes

A run killed by its task's wall-clock limit was never graded. It is reported as **NOT MEASURED (wall-clock)**, not FAIL, and is excluded (with any task it blocks downstream) from paired recipe-versus-baseline deltas. A same-key solo rerun under the one-job-per-GPU lease replaces the hit only when it produced a real PASS/FAIL verdict under the unchanged declared limit.

Totals: 256 PASS, 296 FAIL, 17 not measured (wall-clock), 8 blocked by an unmeasured parent.

| Recipe | PASS | FAIL | Not measured (wall-clock) | Blocked by unmeasured parent |
| --- | ---: | ---: | ---: | ---: |
| bcap-default-baseline-direction-v1 | 14 | 12 | 0 | 1 |
| cell-quota-dualnorm-cross-v1 | 5 | 6 | 3 | 0 |
| confidence-lr-finite-cap-gaussian-v1 | 11 | 14 | 1 | 1 |
| generator-idle-ema-bcap-direction-v1 | 12 | 12 | 0 | 2 |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | 7 | 9 | 2 | 0 |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | 8 | 8 | 2 | 0 |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | 7 | 7 | 3 | 0 |
| kinetic-local-dualnorm-g-history-reference8-v1 | 6 | 8 | 3 | 0 |
| kinetic-local-dualnorm-g-history-v1 | 11 | 15 | 1 | 1 |
| kinetic_transport_local_v2 | 13 | 13 | 1 | 1 |
| positive-evidence-lr-gaussian-v1 | 13 | 13 | 1 | 1 |
| signed-confidence-lr-gaussian-v1 | 14 | 13 | 0 | 1 |

## Real verdicts from solo reruns

| Recipe | Task | Verdict | Seconds | Superseded |
| --- | --- | --- | ---: | --- |
| signed-confidence-lr-gaussian-v1 | five_word_joint_smoke | PASS | 789.3 | TIMEOUT |
| bcap-default-baseline-direction-v1 | five_word_joint_smoke | PASS | 795.6 | INCOMPLETE |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | ring16_acquisition | FAIL | 33.1 | TIMEOUT |
