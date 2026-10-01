# Saved RA9 versus RA8 training neutrality

Freeze this helper before reading any numerical RA9 checkpoint. After the root
READY and numerical lane freeze exist, read CPU copies of the ten original toy
updates: 0,100,250,500,750,1000,1250,1500,1750,2000. Require each variant's existing
post-save `training_checkpoint` event. Hash both files before and after parsing.

This is a CPU parse of serialized saved endpoints, not numerical replay. There
are no model constructors, feature forwards, sampling, optimizer steps, training,
restored or consumed streams, CUDA contexts, new seeds or quality emissions.

Reuse the original frozen comparator's five hash/typed-difference functions
exactly. Compare every serialized trainer leaf: tensor dtype/shape/bytes including
NaN and signed zero, scalar types and bit values, dictionary keys and sequence
structure. State dictionaries store FAST training weights during EMA serving.
Compare saved step and data position. Outer evaluator metrics, log timing and
variant/config provenance are outside serialized training-state parity.

The complete normalization allowance is:

- RA9 backend schema 8 becomes RA8 schema 7.
- RA9 requested settings.cells and recipe.birth_death_cells 128 become 64.
- RA9's new resolution_policy setting is verified exactly and removed.
- Original last.eval_seconds is zeroed in both variants.

All other values remain compared, including actual chart/count metadata, paired
serving stamps, all work counters, model/EMA tensors and buffers, optimizer moments,
controller and settlers, row evidence, streams and global RNG, FIFO, graph, actions,
birth/copy records and counters. No tolerance or broad ignored subtree is used.
Report real differences rather than presuming exact training parity.

Before normalization, use only the reviewed RA9 pure scalar metadata checker and
cap helper, extracted by AST without importing its package or instantiating the
backend. Require typed actual K, effective rank, even-fit partition, 2K categories,
3K+2 multiplicity, Q/(3K+2) cutoff and stamp/last consistency. Initial stamp zero
retains its reviewed initial convention. The existing RA8 settings also satisfy
the same scalar law when its requested 64 is substituted. No geometry is refitted.
This recognition complements the independent state reviewer; it is not a replay
or recalculation of historical chart geometry or features.

Seal helper and original inputs before numerical execution. Pin actual root
READY/package/config and numerical lane source maps once frozen. Future candidate
checkpoints are pinned individually only after their original post-save event.
The watcher writes exclusive per-step receipts and immutable per-step seals, then
a final closed summary and post-exit report/seal. Retain any failed helper attempt
separately. This audit establishes endpoint training neutrality, not quality
acceptance or intermediate-step causal equivalence.
