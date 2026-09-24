# Current research base: g_threequarter_rate

Executable selection: [current-base.json](current-base.json). Exact declaration:
[selected-base/declaration.json](selected-base/declaration.json).
G LR .001125, D .0015, particles .00225 are constant from initialization onward.
The logistic relativistic GAN, Adam(0,.999), R1+R2 .6, mixup .01 and fixed critic
input noise .05 are inherited from H. No additional G fitting objective exists.

| Measured result | Current base | Earlier H control |
| --- | --- | --- |
| Cold ring | PASS:8 modes, HQ.999755859, suffix9 | PASS:8 modes, HQ.999267578, suffix5 |
| Own-state dense continuation | FAIL at1284:83 checks pass, then HQ.761230469 | FAIL at1255:54 checks pass, then HQ.780517578 |
| two_pole | FAIL:spread.028546154 <.30 | FAIL:spread.032636743 <.30 |
| Broader older suite | Other17 hosts UNRUN | 13/19 PASS,6FAIL |

This is a selected experiment starting point, not an overall leaderboard winner
or release-qualified replacement. Its earlier200/200 warm PASS used H's borrowed
checkpoint; that differs from the now measured failure on its own state.
Full1200 own-state hold and native100 tests are gated off after cheap failures.

Independent cold replay reproduces both measured live results and the entire
ring checkpoint byte-for-byte. See selected-base/promotion-checks.json and
selected-base/own-state-short/metrics.json. All125 original H training files
still match their archive. The default stability_runner.py retains the H control;
selected_base_probe.py explicitly selects the NEW base and its own checkpoint.

Run from the repository root with the launcher's pinned CPU/AVX2 environment:

```bash
python reports/toy100/h_stability/selected_base_probe.py --output NEW_OUTPUT
python reports/toy100/selected_h_remaining.py --declaration reports/toy100/h_stability/selected-base/declaration.json --output NEW_OUTPUT --ledger NEW_LEDGER --workers 1 --tasks two_pole mode_hold unipolar mid_scale_identity cover_leftover trajectory
```

Do not repeat finished failed families. The preceding search tried paired and
simultaneous updates, matched observation noise, bounded temporal/optimistic
corrections, rate interpolations and nine critic-regularizer/noise proposals.
Three proposals passed borrowed-H warm screens; only this one also passed cold
ring. None passed the shared cold suite or an own-state hold.

A separate cold-repair comparator h_g020_d005_p010_c01 passes two_pole, unipolar,
mid_scale_identity and cover_leftover, but fails sustained ring. Its positive
particle-centroid mobility preconditioner uses no target fitting. The exact
source snapshot/declaration are in cold-repair-reference/. Do not combine that
candidate's passes with this base's passes.

Pure AE reconstruction has no encoder adversarial gradient path; pure unused-
token preservation has an identical-gradient parameter constraint. These remain
explicit blockers. Do not sweep their parameters, restore supervised losses and
claim pure-GAN qualification, or change frozen scoring/architecture.

Preserved references: ../critic_signal_attempt/ (H),
selected-base/previous-attempt-results.md (finished dynamics), and
rejected-signal-results.md (finished discriminator-signal attempt).
