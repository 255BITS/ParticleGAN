# Selected experimental base

Candidate: g_threequarter_rate. The authoritative selection is ../current-base.json.
The default launcher reads the committed ../SEARCH.md and starts from this recipe.
The original H candidate remains an immutable comparison, with its own artifacts.

- declaration.json: exact single candidate and effective overrides.
- cold/: original measured ring PASS and two_pole FAIL, checkpoint and source archive.
  The original batch manifest also names network_half_rate; its artifacts are
  omitted from this selected subset.
- borrowed-H-warm200/: original200-check warm PASS from H's state, not an own-state hold.
- borrowed-H-replay/: corrected explicit-source replay with complete bitwise parity.
- independent-cold/: fresh cold ring/two_pole replay from the newly selected base.
  The complete ring checkpoint is byte-identical to the original selected checkpoint.
- own-state-short/: new diagnostic from this candidate's own acquired state.
  FAIL at1284 after83 passing checks, retaining the acquired optimizer/RNG and rates.
- promotion-checks.json: independent selection validation.

No full19/22 or full1200 continuation pass is claimed. Existing failing toys and
unrun gates remain blockers. This is the user's chosen next search base, not a
production-default change.
