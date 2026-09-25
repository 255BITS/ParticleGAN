# Adam response attempt — stopped after 9 of 12 allowed proposals

**No promotion. Best diagnostic: `eps_net_1m` passes cold ring and its own
200-update continuation (200/200, minimum HQ .9189453125), but fails two_pole
at .0298888311 spread versus .30 required.** Full own-state1200 and native100
are SKIPPED. `current-base.json` remains `g_threequarter_rate`.

First batch launched 2026-09-24 18:10:39 UTC, within five minutes. Nine proposals
in three batches of three; at most three CPU workers, one thread each. Twenty
experiment gates executed: 8 PASS, 12 FAIL, 0 ERROR. Stopped the family early:
epsilon improves short own-state retention without repairing mobility; tensor
moments fail warm; split moments miss both cold ring and two_pole.

| Candidate | Fixed proposal | Base-own warm200 | Cold ring1200 | two_pole80 | Same-policy own200 |
|---|---|---|---|---|---|
| **eps_net_1m** | G/D epsilon .001; prior 1e-8 | PASS 200/200; min HQ .92554 | PASS; suffix14 | FAIL .029889 | **PASS 200/200; min HQ .91895** |
| eps_all_1m | All epsilon .001 | PASS 200/200; min HQ .92383 | PASS; suffix9 | FAIL .027490 | FAIL at1373/HQ .88159; 172 passes |
| eps_gp_1m | G/prior epsilon .001; D 1e-8 | PASS 200/200; min HQ .97852 | FAIL; suffix4 despite final8/HQ1 | SKIPPED | SKIPPED |
| eps_d_1m | D epsilon .001; G/prior 1e-8 | FAIL at1235/HQ .89282; 34 passes | SKIPPED | SKIPPED | SKIPPED |
| tensor_g | Tensor RMS denominator for G | FAIL at1251; 50 passes; 7 modes | SKIPPED | SKIPPED | SKIPPED |
| tensor_gp | Tensor RMS denominator for G/prior | FAIL at1254/HQ .70142; 53 passes | SKIPPED | SKIPPED | SKIPPED |
| tensor_all | Tensor RMS denominator for all | FAIL at1203/HQ .87939; 2 passes | SKIPPED | SKIPPED | SKIPPED |
| split_prior | Centroid/relative particle moments | PASS 200/200; min HQ .91333 | FAIL; suffix3 despite final8/HQ .99951 | Diagnostic FAIL .245253 | SKIPPED |
| eps_net_split_prior | Split prior + G/D epsilon .001 | PASS 200/200; min HQ .95386 | FAIL; final7/HQ .91479 | Diagnostic FAIL .245168 | SKIPPED |

Historical reference, **not rerun or credited here**: selected base cold ring
PASS; own continuation failed at1284/HQ .761230469 after83 passes; two_pole
.028546154. H remains a comparator. No H checkpoint or borrowed-H warm result
was used. Warm200 above always starts the selected base's OWN acquired state;
it is not same-policy ownership for a changed proposal.

The two successful cold acquisitions were separately tested from their own
checkpoints with the same policy. These are explicit short diagnostics after
two_pole failure, not full stability qualification. Split-policy two_pole runs
are explicitly `diagnostic_two_pole80_after_ring_fail`, not promotion. No partial
ring endpoint is credited. Unipolar, mid_scale_identity, cover_leftover and the
other nine original gates are individually SKIPPED after earlier failures.
The original ten include ring, which was executed for five candidates.

All actual G/D/prior rates remain **.001125/.0015/.00225**, Adam betas (0,.999).
No new adversarial signal: inherited logistic relativistic GAN, D R1+R2 .6,
mixup .01 and input noise .05. G and particles use only discriminator adversarial
gradients. No metrics, centers, labels, fitting loss, schedule or freeze enters
training. Hosts, seeds, data, budgets and scoring are unchanged.

Tensor geometry uses `sqrt(mean_tensor(v_hat))+epsilon`. Split geometry uses
orthogonal particle-centroid and relative-gradient blocks with separate second
moments per latent coordinate. Both denominators stay positive; all subspaces
retain movement. Split moments start at zero cold, inherit the original mean
coordinate second moment for changed-policy warm diagnostics, and are saved in
optimizer state thereafter. Split geometry requires particle tensors with a
leading particle axis and beta1=0. Every variant remains scratch evidence.

**Tests: 58 passed executions, 34 distinct test cases, 0 failed/errors/skips.**
Runs: 22 existing epsilon/continuation/host tests; 16 geometry+epsilon tests;
20 split-moment+epsilon tests. Tests cover hand-computed updates, positive
movement, exact optimizer replay, scoped patch restoration, receipts and frozen
host behavior. Full suite SKIPPED because no candidate survived cold gates.
Regression rows use `candidate=regression`. `git diff --check` passed.

Code: [adam_response.py](adam_response.py),
[adam_response_probe.py](adam_response_probe.py),
[adam_response_cold.py](adam_response_cold.py),
[test_adam_response.py](../../../tests/test_adam_response.py).
The existing [stability_runner.py](stability_runner.py) now calls the existing
receipt validator instead of hardcoding epsilon; its update loop is unchanged.
Cold execution reuses the existing runner, receipt/episode validators and state
capture. No immutable H audit or structural AE/unused-token rerun was performed.

Artifacts: [leaderboard](adam-response/leaderboard.json),
[unit totals](adam-response/unit-test-totals.json),
[best own-state metrics/checkpoint](adam-response/batch3/eps_net_1m-own200/),
[best cold state](adam-response/batch3/cold/eps_net_1m/), and batch1–3 declarations,
source archives/manifests, logs, raw metrics and checkpoints under
[adam-response/](adam-response/). Ten inherited parent-archive labels were
corrected without changing results; `declaration-as-emitted.json` preserves each
original, and [corrections](adam-response/parent-archive-metadata-corrections.json)
records them. Actual executable source snapshots were retained throughout.

Recommendation: keep `eps_net_1m` as the next **diagnostic reference**, not a new
qualified base. Any further intervention must repair cold mobility while keeping
ring acquisition. Do not extend its hold or run native100 before those gates,
and do not continue tensor/split coefficient sweeps from these failures.

Exact commands are in [REPLAY.md](adam-response/REPLAY.md). Runtime result and
complete gate ledger are `../result.md` and `../tests.jsonl` relative to repo root.

```bash
tail -f reports/toy100/h_stability/adam-response/batch3/*.log
tail -f ../tests.jsonl
```
