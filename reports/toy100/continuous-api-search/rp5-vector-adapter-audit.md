# RP5 vector adapter audit

**The completed vector_two_broad run is a valid PASS under its predeclared isolated-noise adapter.** It has all24 expected observations,23 passing observations and terminal suffix23 from100 through1200. Final HQ.995361, normalizedSW1.026707, massTV.005615. Source/declaration/artifact hashes verify; all package files match RP5 ring,stationary,and all4images.

**The initialization mismatch report was a false alarm.** Independent raw-storage inspection of `initial-state.pt` confirms G6/6,prior1/1,D6/6 tensors exactly match the retained canonical fixture. G/prior EMAs also match. The worker compared `D.state_dict().values()` with an optimizer-parameter fixture, accidentally including the leading `freqs` buffer. That extra[2] tensor is not a parameter. Preserve the erroneous audit and append a corrected named-parameter comparison; a repeated two_broad run is not justified by this issue.

The CUDA data0/latent1/penalty2 backend is correct. Frozen sampler/scorer functions match canonical CUDA semantics; plan,profile and all four D cards are preserved. RP5's public `GANTrainer.step` owns the same precision,secant,eager-Adam behavior as ring/images, with total_steps=None. G-real is drawn once per accepted update and reused by both fields; its independent data stream is preserved. No legacy training fallback or external schedule is executed.

Fixed evaluation output seed2303 is declared explicitly and allowed for this isolated-noise protocol. Latent990,target991,projection992 are CUDA,4096samples; all24checks and original final-five/live thresholds remain. Full trainer/caller hashes assert observation isolation. Historical K3P's402 global-noise results are a different measurement namespace; a later publicK3P comparison must use this same explicit adapter.

Before the remaining five runs, verify or copy their exact retained fixtures using **ordered named parameters**, then record buffers and EMA separately. Copy before GANTrainer constructs optimizer/reference state. Declare/archive the fixture paths/hashes and any harness-only fixes before execution. Do not inherit the first host's initialization proof for other D cards. The current exception handler keeps only repr(e); retain full tracebacks for any future error.

Read-only source/archive/raw-byte audit; no models,GPU,tests,training or worker edits. The unchanged long ring need not wait on this preparation. Full hashes and tensor evidence are in the JSON companion.
