# Complete learner-state envelope for support memory

The existing `pr84_critic_refinement_capture.snapshot` serializes the live
generator, critic, prior, both Adam optimizers, EMA, all training RNG streams,
and fixed-scale noise-policy history. It does **not** serialize an adapter's
learned support memory. A memory-bearing adapter must never resume from that
host snapshot alone: it must use `learner_state_envelope.capture` after a
completed update, and restore the complete envelope before the next native
real bank.

The recorder API is `learner_state_dict()` and
`load_learner_state_dict(state)`. The selected versioned payload includes a
confirmation flag, an optional pending bank with its absolute ID/hash and
ordered float64 sums, positive integer counts and float64 squared-norm sums,
plus ordered confirmed sums/counts/squares and immutable float64 reference
centers. It also records the positive fixed identity separation, confirmation
evidence, and accepted/rejected sample totals. The envelope clones and hashes **every** field; unexpected fields
fail closed until a new schema validator is added. A recorder also exposes audit-only
`learner_bank_count` and `learner_last_observed_step`; the former counts one
distinct D real bank per outer update, not its three phase replays. Matching
must not read either counter. The complete envelope hashes all host and
learner bytes, ordered list entries, schema, method, source digest, bank
count, and absolute update. New statistics require a schema revision rather
than being silently dropped.

The `attach_before_next_bank` hook wraps the existing source-audited
`resume_mode_hold` pre-`set_step` restore. It first verifies the saved host
hash, lets the original hook restore host state, then loads learner state
and checks that no model, optimizer, noise history, or RNG state changed.
An absent or altered memory payload is a hard error. A snapshot produced by
an older, memoryless candidate can initialize a **new bootstrap diagnostic**;
it is not an exact continuation of a trained memory learner.

`learner_state_split_harness.run_short_split` prepares the short definitive
check once the recorder exists: one source-bound uninterrupted path and a
prefix/split-resume path, at most three updates, exact per-update observations
and exact final host-plus-learner envelope. Its callbacks must use the same
frozen factory, host inputs, noise clock, and source digest. The current tests
exercise only the helper contract and mocks; no memory adapter or host
training was run here. Eight no-training tests passed, including a
`torch.save`/weights-only-load roundtrip, host-before-memory restore order,
missing/changed/reordered memory rejection, source rejection, and split
comparison failure when only learner memory differs.
