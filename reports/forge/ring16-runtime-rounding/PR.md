Ring16's uninterrupted and reloaded runs first diverge in the critic backward
gradient at update 401. This CUDA diagnostic reproduces both historical
gradients exactly, then disables autograd multithreading only for update 401.
The serialized live and reloaded runs produce bit-identical critic gradients
and entire trainer contexts, removing the measured boundary discrepancy.
The constraint covers the whole update, including higher-order derivative graph
construction and final backward; those two effects are not independently split.

The recorded graphs have identical topology. Their ordinary relative node
sequence priorities differ in 1,054 of 7,503 pairs; serialized priorities agree.
Installed PyTorch 2.14 headers explain thread-local sequence counters and the
engine's ready-node priority rule. This supports accumulation order as the
mechanism, while distinguishing graph priorities from actual execution traces.
The serialized result is a third trajectory, matching neither ordinary result;
full 1,600-update quality and continued stability remain unmeasured.

All four frozen arms complete on an RTX A6000: 804 new updates, 24.795 seconds
whole-process cost, seed 0, public deterministic initialization, unchanged MoG,
batch, architecture, data, evaluation cadence and constant learning rates.
Adds compact source-bound results, original receipts, a saved-prefix training
GIF, byte-verified raw archive receipt and reproducible comparison/rendering
sources. The earlier GPU preflight refusal remains preserved separately and
consumed zero scientific attempts.

Validation: complete CUDA receipts, full 400-update checkpoint identity,
historical gradient parity, all named streams, exact serialized full-state
equality, 24 prefix sample comparisons, executed-source hashes, archive member
hashes, GIF provenance and Forge metadata validation. Saved-only reporting
adds no model calls or random draws. No optimizer defaults, task gates,
qualification outcomes or technique inventory are changed.
