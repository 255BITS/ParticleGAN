# Original quarter-rate moving Grid100 observation

Completed original checkpoint500→535 and1000→1035, exactly70 updates, in16.64 seconds including process startup. No scoring, sampling, source edits or hyperparameter changes. The original CUDA real Generator(seed1234) was reconstructed by exactly two native sample_real(batch2048) calls per completed update. Rotation/data functions were copied AST-identically from the original saved runner. Reconstruction preserved global RNG; loading each native checkpoint preserved the external real generator and restored the full semantic training state bit-identically. Each capture checked all global/private/birth/external RNG states unchanged.

| Turn | Fire step (begin update) | G ratio | Table ratio | D ratio | G scale | D scale |
|---|---:|---:|---:|---:|---:|---:|
|30°|514 (515)|18.532877|10.752232|14.806933|.5|.25|
|60°|1021 (1022)|5.435961|4.033462|4.077749|1|.00390625|

Noise remains zero/omitted across both windows; table/noise tester scales remain1. The first turn occurs in pure A; the second has an already-started KA2 anchor with no phase transition in the window. A guard requiring at least one settled network owner preserves eligibility at both observed useful fires, whereas requiring a settled generator alone would reject the second turn. This validates the proposed eligibility distinction only; actual guarded-source trajectories and original quality gates still require fresh runs.

The original moving run and all quality values stay intact. Full role signals, pending q, old fast/slow, context and raw losses are in the two trace.jsonl files. Saved endpoints now also preserve the external real-stream state.
