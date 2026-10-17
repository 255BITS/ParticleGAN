# Preseal transport draft findings

This review read `transport.py` draft SHA
`24f785c7cb0ba60a8a8261c9cbb81d05b09217dd11d91d9cfb4ad7e3afe7f3a5`
under `integration/review/training-regression/post-ra9-quality/mean-category-transport/`.
It ran no imports from that helper, PT loading, model forward, random draw,
numerical test, CUDA, or production modification. Findings were sent to the
owner and root before their preseal or numerical run. This is not a PASS receipt.

## Correct mechanisms in the draft

The planner bounds per-signature pools to 64, retains unique and disjoint
source/destination sets, subtracts earlier ordinary expenditure, and freezes
candidate pairs before its one reaction-stream noise tensor. The two
displacements use their own prior and the same parent IDs and noise. The
observation wrapper surrounds forwards after that intended draw. Actual
offspring are checked against the original categories/cells/groups. The
objective advances a virtual EMA mean ledger without editing either table.
Accepted coordinates and exact-shape optimizer rows/history are cloned into
the packet. Commit has no random draw or `_move` call and registers lineage
once, then refreshes the moved FAST cache rows.

## Corrections requested before preseal

1. `observe_view.eligible` lacks an explicit even category bit. Supported
   `p>Q` need not establish membership below the separately fitted inside
   boundary. Root requires inside in both views. The state reviewer sent this
   same finding independently.
2. `commit_packet` checks the digest, epoch and row disjointness but does not
   validate the entire packet schema before the first prior write. It needs
   row range/dtype/device, finite FAST/EMA coordinate shape/device/dtype,
   feature/cache shape, and parent optimizer/history schema validation first.
   Otherwise a malformed self-consistent packet can write FAST before a later
   EMA, row-state, lineage or cache operation fails. No malformed packet was
   executed in this source-only review.
3. The epoch omits bandwidth and the reaction stream identity/state after the
   intended draw. Those inputs must be pinned for a reusable precommit API.
   A later draw or bandwidth change should reject before row writes. An
   immediate synchronous scratch-only limitation would need an explicit
   driver proof; it is not a later generic API guarantee.
4. `model_tensors` must not apply pre-preview tensor-version checks to restored
   registered buffers. The observation guard's `copy_` restoration increments
   their versions even when values are unchanged. Preserve the full buffer
   ownership boundary and check exact restored mappings/bytes, or pin the
   post-restoration versions after proving restoration. The owner announced
   this correction before preseal.

## Pending review scope

The final driver is not available in this draft review. Its raw FAST/EMA state
decoding, private saved-CPU stream cloning, full source/input freeze, trigger
gating, reservation completeness, callback roots and owned streams, and
scratch-only state mutations remain to be checked. Production counters,
population rebase, row evidence reset, final paired-average refresh and audit
extensions remain future obligations; the prototype is not a production
candidate.
