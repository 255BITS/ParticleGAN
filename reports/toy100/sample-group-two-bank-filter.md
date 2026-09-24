# Two-bank fixed-support memory: pure data filter

The [source](sample_group_two_bank_memory.py) infers groups from two consecutive native D real banks. It confirms identities only when both MST partitions have the same count (at least two), reciprocal unique nearest-centroid pairing, and strictly positive separation margin `min(separation_A, separation_B) - 2 max(paired_distance)`. A mismatch replaces the pending bank and leaves the support unconfirmed. Once confirmed, the pooled reference centers and their half-minimum-separation radius freeze. Later individual real samples update the nearest reference only inside its open radius; no later bank can create or delete an identity. The running centroids can still move the *full* distance indicated by accumulated data, without a time-dependent correction gain.

This is a conditional fixed-target support policy, not a certificate that two finite banks contain every target component. Before confirmation a proposed host adapter should leave G and prior at rest while D still trains. That behavior has **not** been tested here; this filter makes no model or optimizer updates. Its checkpoint state includes the pending bank, confirmation receipt, fixed references, all sufficient statistics, and accepted/rejected sample counts. The native adapter must observe consecutive absolute bank IDs and restore this state before its next bank.

The [raw branch receipts](continuous-evidence/round6-sample-group-two-bank/branches.json) use a saved passing mode-hold state, its actual first native 128-real D bank, then independent sequential draws from the saved data stream with native prior-index and G-real draws between D banks. Conditioned omission and singleton banks are explicit diagnostic subsets. Fixed evaluation grades use the saved late-noise clock, not a training oracle.

| Controlled branch | Result |
| --- | --- |
| Two ordinary complete banks | 8 then 8 groups confirm; margin 2.13555; one free-output anchor target has 8 modes and HQ 1.0. |
| Confirmed support, then a bank without mode 0 | The absent group's count and all frozen references stay unchanged; target has 8 modes and HQ 1.0. |
| Complete then omitted bank | 8 then 7 groups do not confirm. |
| Omitted, then two complete banks | 7 then 8 remains unresolved, followed by 8 then 8 confirmation; a separate free-output repair reaches 8 modes after two MM targets. |
| Complete then a bank with one noisy mode-0 sample | 8 then 8 confirm; no false ninth identity; target has 8 modes and HQ 1.0. |
| Two agreeing omitted banks | **They confirm 7 groups.** This is the finite-bank blind spot and a rejection of any unconditional completeness claim. |

The microfilter also verifies pending and confirmed state round trips, rejects a modified confirmation certificate, rejects skipped native bank IDs and empty-memory late restart, and leaves global torch RNG unchanged. The [manifest](continuous-evidence/round6-sample-group-two-bank/manifest.json) binds the source, inputs, and full receipts. No neural warm, cold, or production result is claimed.

After confirmation the limiting centroid is the mean of real samples inside a *fixed* reference ball, not necessarily the latent Gaussian component mean: tails from other components can enter that ball, and samples outside every ball are ignored. A rare target component absent from both bootstrap banks cannot be recovered without a later policy for discovering new identities. This method is scoped to a stationary, well-separated target and does not yet preserve full target-distribution fidelity beyond support centers.
