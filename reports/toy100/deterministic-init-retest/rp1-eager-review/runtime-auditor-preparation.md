# RP1 eager diagnostic runtime auditor

A separate read-only auditor is prepared at audit_rp1_eager_screen.py. It accepts only --source for this exact historical diagnostic, pins its own declaration/seal/CPU proof, and reuses the complete standard-library source/artifact/raw-checkpoint,1200batch and24observation checks. It additionally verifies the historical setup proof, independently recomputes its raw non-optimizer/RNG digest, verifies17 zero initial eager states and CUDA clocks0/1/1200, and checks every applied rate against RP1 open/quiet scales plus original360/720 startup noise.

No terminal result has yet been audited by this preparation. Run the command in runtime-auditor-preparation.json after artifact completion; output is rp1-cuda-eager-diagnostic-runtime-audit.json. This remains an external-worker setup diagnostic rather than ordinary package-owned public API behavior. No frozen harness/source was edited and no Torch, training or GPU execution occurred.
