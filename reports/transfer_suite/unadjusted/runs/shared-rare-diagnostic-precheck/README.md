# Retained diagnostic precheck

The first read-only replay produced the same complete numerical result,
optimizer receipts, actions and live/EMA verdicts as the archived episode.
Its `recipe` comparison alone reported false because the in-memory recipe uses
tuples while the archived JSON uses lists. The comparison was corrected to
JSON-normalize the replay payload in the subsequent
[exact diagnostic replay](../shared-rare-diagnostic/README.md). This precheck
retains its original [checks](checks.json), [diagnostics](diagnostics.json),
[protocol](protocol.json) and [source](source.tar.gz); it contributes no
architecture selection point.
