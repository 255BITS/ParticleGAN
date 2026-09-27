# Released K3P mode_hold completed audit

The exact released v0.8.0 reference completed all 1,200 updates and failed the frozen tiny mode-holding gate: **0/24 passing observations, suffix 0, at most 6/8 modes**. Final live HQ was 0.97607421875 (EMA HQ 0.926025390625); final live and EMA coverage were both 6/8. Reported runtime was 19.323610109 s.

The independent audit passed. All 14 retained artifact hashes match; the executed source ZIP contains the 56 sealed files plus the exact seal `2ae5dd02c6049121536d0d692b437cb19399721ff5889d35ebe8030a204acf96`. All 12 package files are the reviewed unpatched release. Imported module receipts identify that isolated package. No Torch imports, training, GPU work or sample regeneration were performed by this audit.

A restricted standard-library raw-storage reader verified both checkpoint envelopes, native schema 3, accepted clocks, initial model tensors, every initial private/global/caller RNG, and the final caller RNG against the retained RP5/RP7 reconstruction fixtures. Initial native Adam states were empty; the first and final receipts show 17 CPU scalar clocks with CUDA parameters and moments. Final checkpoint storage independently confirms CPU clocks at 1,200 and CUDA moments. All 1,200 data/index/cursor receipts match RP5/RP7 byte-for-byte. All 1,200 actual learning rates and noise values match the declared released schedule.

This is the separate canonical CUDA tiny task (12 particles, z4, batch128), with explicit benchmark `total_steps=1200`: input noise reaches zero at120; output noise reaches0.029 at240; LR decay begins720. The last update uses index1199. It is neither the literal released default7000 nor the candidate's360/720 initialization schedule. Observation latent seed9 and isolated global output-noise seed402+step are unchanged. The restoring serial context covers the complete public step; the CPU factory default prevents eager or migrated Adam counters.

The failure is a valid supporting baseline result. It establishes that recurring RP5/RP7/RP8/RP9 tiny-host failures are also seen in this released reference; it does not waive the candidates' declared gate or establish an indefinite default. No historical full checkpoint identity or new cross-process replay was inferred.

Archive: `evidence/public-k3p-mode_hold/`. The JSON audit contains one ready `broader_entries` item for root to append sequentially. Original JSONL is losslessly compressed; checkpoints remain at content-hash-verified external paths. Shared manifests and active sources were not edited.
