# API-C13-R1-new-init initialization port audit

PASS for initialization-only integration. Constructor-only CPU preflight completed with zero learner steps and CUDA uninitialized. Two fresh constructions separated by extra RNG draws, without reset, produced identical full non-RNG checkpoint state and all independently enumerated named parameters/buffers. Ordinary constructor RNG consumption remains; the public prior/network initializer does not consume those streams.

All 27 package files match the declared package ZIP. The 14 unchanged candidate files remain byte-identical to the archived source; ten added initializer modules match API Git base25751c0864dd8259b00c5804f600cd41cce6e4cf exactly. Every trainer method except initializer metadata loading is AST-identical; all training/recipe module AST outside the three reviewed recipe methods, initialization field and load_state_dict is unchanged. The full declared mode-hold recipe changes only initialization. The complete narrow diffs were manually inspected before this report.

Prior initialization occurs through the actual public recipe factory before any trainer-derived geometry; network initialization precedes the original optimizer construction. Existing candidate eager/lazy optimizer state and its declared counter placement remain intact. Geometry result: {'geometry_check': 'not applicable: no declared bandwidth controller'}. Checkpoint loading adds the merged public initialization metadata behavior; continuation itself was not executed.

Declaration `e427ac5e930b1422639875cc2768a956f4b8e090406b6c2da352706612f08d37`. [Hash-bound independent audit](api-c13-r1-independent-init-audit.json) and [CPU receipt](api-c13-r1-cpu-preflight.json). Historical declaration copies may differ in JSON formatting; original and copied raw hashes are retained and parsed contents were independently verified equal.

Actual CUDA sampling must still pass the sealed harness dry preflight before any learner update. No previous quality evidence transfers to the new initialization epoch.
