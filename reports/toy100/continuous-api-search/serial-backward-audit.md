# Serial backward source and evidence audit

Scoped serial implementation is sound by source inspection; captured fresh-process regression passes within its declared variant/runtime scope. Address provenance and claim-scope issues before broad guarantees.

No training, GPU work or worker checkout edits were performed.

The retained critic graph has 161 nodes with unchanged topology and 1360 changed pairwise sequence priorities. Local PyTorch headers document thread-local sequence numbers and sequence-priority ready queues. This supports the scheduling explanation; it is not an execution trace.

The serial context wraps the complete update, including nested create_graph gradient construction and both backward calls. PyTorch restores the prior thread-local flag in __exit__, including exceptions. Checkpoint mode checks precede state mutation.

Independent stdlib checks confirm the reference/resume final receipts and all 10 observations match. Every declared source hash matches its archive. Reference/resume source changes are limited to docs/load validation; training-method AST hashes match.

## Actionable findings

- **effective-mode (medium):** False preserves ambient caller multithreading mode; the saved opt-in flag does not capture every effective execution-mode change. Document False as legacy/inherited execution and narrow the cross-mode rejection claim to explicit serial flags. If actual-mode provenance is promised, persist/validate it without silently overriding legacy caller behavior.
- **scope-of-proof (medium):** The 1000→1100 fresh-process match uses dv1 only. Serial execution changes numerical trajectories relative to historical mode, so existing dv3 quality results do not qualify serial mode. Use one declared execution mode consistently for further authorized quality work and include variant-specific continuation evidence before claiming it. No extra experiment launched by this audit.
- **source-closure (medium):** Serial regression source archive omits imported worker.py and benchmark model/evaluator helpers; runtime version/device metadata is also absent. Retain hashes/snapshots for imported worker.py, model/evaluator files, exact PyTorch/CUDA version and GPU identity with the completed evidence.
- **direct-nested-test (low):** Current CPU tests inspect forward hooks and a forward exception, not nested create_graph execution or exceptions inside backward. Source scope is correct, but regression coverage could detect future scope narrowing more directly. Add a focused contract test that observes the flag within autograd.grad(create_graph=True) and a backward exception; retain both caller starting modes.
- **causal-wording (low):** Recorded graph ranks and successful serial intervention strongly support an autograd scheduling cause but do not record actual node execution order or isolate each changed priority as causal. Call this a strongly supported explanation/workaround, not a proof that all CUDA checkpoint divergence has this sole cause.

Exact reviewed source/evidence hashes and line references are in `serial-backward-audit.json`.
