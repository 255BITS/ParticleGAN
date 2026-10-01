# Independent sampler cache and native API review

Status: PASS for four focused CPU contracts. The private AXIS-ID package differs from frozen RA3 in only `feature_cells.py` and `training.py`;25 other Python files match exactly.

The cache converts the selected coordinate axis tensor to Python integers once during cache construction. Cold/warm output bits, cache work, version invalidation and all-tied behavior match RA3 on fixed tensors, including known-copy candidates. This review provides no GPU speed claim.

The new `_generate(model, latent, sigma, stream, indices=None, *, rows=None)` signature is recognized as indexed by the exact frozen native harness resolver. The exact native draw argument fragment successfully calls its fifth positional argument and forwards those row IDs to the lineage geometry. A specified known-copy case uses radius.005 with indices and5 without them, demonstrating that the call reaches the graph.

Existing four-argument and rows-keyword calls preserve output and saved RNG bits. Positional indices and indices-keyword calls match the existing rows alias, for live/EMA priors and sigma0/.029. Providing both nonnull aliases raises before RNG, geometry work or graph changes. The indexed gather retains the identity derivative.

Frozen RA3 has only a keyword rows argument, so the frozen native harness would resolve plain and omit row IDs. Root stopped RA3 before native screens; its completed learned lane uses the indexed internal sample/training path. AXIS-ID is a separate candidate and does not change the frozen RA3 package or its saved results.

No CUDA context, optimizer updates, new seeds or quality gate changes were used. Evidence: `audit_axis_api.py`, `axis-api-review.json`, `axis-api-review-attempt1.log`, `AXIS-API-FROZEN.json`.
