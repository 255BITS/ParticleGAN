# RA13 replay failure diagnosis

Both original 2×10 CUDA continuations remain **FAIL** for exact semantic state. Canonical restoration, losses, samples, model/optimizer/RNG state, detector and settled guard are exact. The first mismatch occurs at update1008 in both fixtures and is confined to `trainer.lr_settle.0.1.last_block`.

The saved checkpoint contains one displacement tensor referenced by both `last_block` and `blocks[-1]`. Native same-device loading preserves this alias. Recursive CPU-to-CUDA conversion calls `.to()` separately for each occurrence, creating distinct tensors. A later moved-row rebase invalidates `blocks`, updating `last_block` only in the native branch. Toy differs in5376coordinates (42rows×128); MNIST differs in128coordinates (row717). Every differing native coordinate isNaN, whereas the CPU-map counterpart remains finite. All actual blocks and stationarity evidence are identical.

This is a checkpoint restoration defect, not scorer/harness normalization. Source uses `last_block` for diagnostic row energy, so these10updates demonstrate no training discrepancy. Strict whole-state parity is still required and the failure must remain recorded. A prospective loader correction should preserve repeated tensor identity during device transfer without changing checkpoint fields. Source, frozen adapters and recorded runs were not edited. This diagnosis used CPU checkpoint reads only and left CUDA uninitialized.

Details, aliases, differing row IDs and immutable file hashes are in DIAGNOSIS.json.
