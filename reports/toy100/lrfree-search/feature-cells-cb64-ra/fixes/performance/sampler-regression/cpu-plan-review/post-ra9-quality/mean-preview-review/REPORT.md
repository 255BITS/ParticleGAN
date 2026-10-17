# Mean preview and scratch commit source review

**PASS for the one frozen saved grid/toy CPU prototype.** This is a source-only
preexecution review, not a numerical, statistical, production or quality result.
Owner preseal: `9818d0439d36b6801ee9c5c21554c5e9b504bf1f1494b1443ebcc89d1ee520ab`.

All 47 owner source/input byte guards match. The complete 29-file RA9 package
and its config are unchanged. The helper source parses. The driver checks
those guards before importing Torch or interpreting either PT file. It selects
exactly the final grid and toy states, uses raw saved FAST/EMA weights, and
clones their saved CPU RNG into one private current-chart stream. No historical
CUDA chart or CUDA stream is reconstructed.

## Preview and commit

- Per-signature parent and child pools are bounded by 64; globally unique,
  disjoint pairs exclude the supplied complete reservation union. Candidate
  count is bounded by the residual shared ordinary budget. The driver accepts
  only an empty previous action prefix or a fully exhausted ordinary budget;
  missing partially spent reservation IDs abort instead of being inferred.
- Original and actual proposed rows must be finite, supported with `p>Q`, and
  inside in both views. Each offspring retains its child's own fine cell,
  category and group; both views share the real-only group. The planner keeps
  protected survivors and never overwrites a selected source.
- One intended noise matrix is drawn outside the observation guard. Both
  displacements use that matrix and their own prior, with the shared lineage
  graph and the same parent IDs. Both tables remain unchanged during preview.
  Direct clean callbacks evaluate the prepared coordinates; there is no
  sampler call, second perturbation or output-noise draw.
- Sequential objective decisions update a virtual EMA mean ledger using the
  actual proposed features. The accepted packet clones exact coordinates,
  optimizer rows of exactly the table shape, and parent latent history. It
  pins table, model-parameter, lineage and chart versions; exact buffer bytes
  and bindings; bandwidth; and the stream state after the draw. Expected
  buffer restoration does not spuriously fail a version check.
- Packet digest, epoch, paired layout, row IDs, coordinate and feature schemas,
  cache initialization, parent optimizer/history bytes and lineage validity
  are checked before row writes. Commit uses those stored bytes, inherits the
  existing copy row-state/history law, registers copy lineage once, refreshes
  accepted FAST cache rows and marks the packet consumed. It contains no
  random draw or `_move` call. Novel birth's optimizer-zeroing law is not used.

The driver mutates only scratch prior, row-state, history and lineage clones.
It contains checks for exact source/state/RNG preservation, unchanged
category/group/support ledgers, optimizer/history inheritance, untouched source
rows, virtual versus actual objective, shared budget and rejection of a second
commit. These are planned owner-run checks; this reviewer did not execute them.

## Scope and remaining production obligations

The fixture callbacks are pure functions of saved tensors and have no stateful
module roots. Thus the driver's empty observation-root list is honest. The
generic helper includes restoration of recursive modes, registered buffer
mappings/values/nonpersistent sets, gradient objects/values, global RNG and
supplied owned streams, but this review is not a custom-model numerical test.
Arbitrary unregistered Python state and external RNG remain outside that
declared ownership boundary. Full graph validation is a scratch audit step,
not a proposed production performance claim.

The prototype does not implement production phase orchestration, row-evidence
reset, population participation rebase, counters, checkpoint law or final
paired-average lease. If selected later, mean copies need an explicit fourth
ordinary phase, the full all-phase reservation set and moved-row hook, current
FAST/cache refresh, and a lease measurement after all actions. Inherited
optimizer/history bytes must not carry positive participation evidence.
The old three-phase/`3K+2` artifact assumptions require a declared audit
extension before such a candidate runs; all original gate checks and old
receipts must remain intact.

The independent state reviewer owns the witness mathematics and common family.
The frozen prototype uses a descriptive clean-table objective and trained
D/shared FIFO. It supplies no emitted-distribution, stationarity or repeated
adaptive certificate. No source result supports changing the original toy or
grid acceptance thresholds.

Earlier draft defects and requirements are retained in `DRAFT-FINDINGS.md` and
`PRECOMMIT.md`. This review imported no Torch/helper package, interpreted no PT,
performed no forward, random draw, numerical test, CUDA work or source edit to
the owner/package. Only this new private review area was written.
