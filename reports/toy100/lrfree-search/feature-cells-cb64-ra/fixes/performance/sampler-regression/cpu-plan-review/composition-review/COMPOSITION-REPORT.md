# Independent RA4 composition audit

PASS. Read-only review found no concrete defect in `compose4.py` or the
composed RA4 source. The CPU audit completed in 0.61 seconds with one thread,
hidden CUDA, no gradient updates, no quality evaluation, and no alternate
seeds. Source hashes were checked before and after the audit.

The four planner AST splices exactly match their frozen proposal/base hashes,
including the three `@torch.no_grad()` decorators. Restoring the three original
count planning methods reconstructs the entire final count snapshot class.
The final count comparison, common 3K+2 wrapper, certificates, shared ledgers,
isolation planner, and original MST remain AST exact.

Removing the declared helper, restoring the AXIS snapshot, and undoing only
the count settings/diagnostic additions reconstructs the entire AXIS module
AST. The other 26 package source files, including training, are byte exact
with AXIS. This proves lineage topology, cached Python axis IDs, copy paths,
generation/sample API, and backend checkpoint methods are preserved.

Composed settings exactly combine AXIS lineage/kernel/degree/candidate fields
with the final count owner's mass policy, partition, and 3K+2 family. The nine
new count diagnostic fields match the count source. A single original fixed
identity/two-cluster CPU reaction exercises these fields and the 194-test
cutoff; no training update is performed.

The canonical frozen harness resolves the composed trainer as indexed and its
actual fifth positional call forwards row IDs. Live/EMA native calls and
trainer sampling match explicit indexed calls and consume identical RNG bits.
Warm selected axes are Python integers. A nonempty copy graph survives actual
serialization and trainer restore exactly; derived caches are discarded.
Trainer/backend schemas remain 4. RA3 and old 3K count settings reject before
any trainer state changes, so checkpoints identify the composed semantic law.

Artifacts: `composition-cpu-01.json`, its log, audit script, and
`COMPOSITION-FROZEN.json`. The GPU contracts, timing and quality results belong
to the root-owned immutable validation lane and are outside this receipt.
