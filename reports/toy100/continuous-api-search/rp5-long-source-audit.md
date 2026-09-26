RP5 long-run source contract verifies; the uninterrupted run should continue unchanged. This is source/partial-prefix evidence, not a completed 30,000-update quality result.

All 34 immutable source entries verify. All 15 learner package files match the single-shift, stationary and four image runs exactly. The long recipe equals the earlier ring recipe; image differences are only frozen resource dimensions. The worker differs from single/stationary only by retaining long-declaration.md. Initial full state receipts and runtime match. Independent retained-log comparison matches single through 2400 and stationary through 6000 across metrics, rates and complete state receipts.

The learner receives total_steps=None, RP5 precision and secant updates. The 30,000 limit and changes after 6000, 7800 and 27000 stay in the evaluator. Fixed 360/720 noise initialization is unchanged. Isolated seed-9/4096-sample observations and seed-0 real data match the prior ring host. Actual noise and accepted/precision clocks agree through the recorded cutoff 8100. Full checkpoint state includes controller/reference, optimizer/Adam/A2/KA2, model/EMA, private/global RNG and caller data stream. Frozen controls load a separate trainer and restore construction-changed global RNG; the main learner is not reloaded.

The long declaration correctly calls for separate continuations at all three saved changes. Those are not implemented by the existing continuation_check.py, which still hardcodes 2400→2500, split2450, +[1,0] and openings==2. The minimal remaining evidence is the declared short replay for each long checkpoint, using actual saved state and correct targets:

| Checkpoint | Saved offset | New absolute offset | Delta from saved means | Initial endpoint |
|---|---|---|---|---:|
| 6000 | [0,0] | [1,0] | [1,0] | 6100 |
| 7800 | [1,0] | [1,1] | [0,1] | 7900 |
| 27000 | [1,1] | [0,1] | [-1,0] | 27100 |

Compare immediate restored state and full endpoint receipts with the uninterrupted original, including caller data RNG. Record precision state/counts relative to each checkpoint rather than hardcoding two openings. If reopening occurs later than the initial 100-update interval, that interval cannot claim to cover it; use the original trace to select the first later archived receipt in the declared continuation check. Include a resumed closed state and an interior split in a reopened state when observed. Existing early proof already covers closed at 2400, reopen at 2440 and open split at 2450; it does not establish all aged/repeated-change continuations.

Long reporting must use actual segment boundaries and group frozen observations by their generating checkpoint. Inherited recovery_at_3600/recovery_extended/frozen_control convenience fields remain tied to the old single-shift layout. The long declaration warns against interpreting them as new transitions. The RP2 policy label and old comparator-owner sentence are stale prose; actual recipe/source and later supervisor instructions are authoritative. Preserve those immutable records with clarification.

Correction of auditor false alarm: an earlier warning claimed segment was first imported at line 270 and would fail at prefix9000. Full AST/control-flow review found the earlier import at line 203, before the loop at 216; both uses are valid. The warning was retracted immediately. Root confirmed the worker received the correction before acting, so the original run continued. No restart, patch or host-error classification is warranted. This correction is retained explicitly rather than presenting the false claim as a blocker.

Read-only CPU/stdlib source and retained-file inspection only. No PyTorch import, GPU, model/test execution, training, active-source edits or interruption was performed by this audit. Detailed hashes, references and replay checks are in rp5-long-source-audit.json.

By the audit cutoff, the first two change windows were also present. Their recorded precision states are preserved in the JSON; this does not replace separate-process continuation comparisons.
