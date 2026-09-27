Exact public K3P ring reference audit

VERIFIED_COMPLETE_REFERENCE

Source ZIP: 0d738e15f0ecbbeb9ef15a17efbce19c634daa8ae324df5adf8b2f598291fdb5
Sealed bundle: ec3a85663d09712e9a5e226b8f7824540da0b7558d5fbe24b3151e070d60f57f
No Torch import, model construction, GPU execution, or training performed by this audit.

Segment 0–2400: first arrival 670 (+670); 157/174 retained; departures [770, 800, 850, 960, 1110]; misses [770, 800, 810, 850, 960, 970, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1210]; final suffix 119 from 1220.
Segment 2400–4600: first arrival None (+None); 0/0 retained; departures []; misses []; final suffix 0 from None.

All 460 observations, 4,600 rate/noise rows, 46 state receipts and 220 frozen controls verified. Exact released imports and initial models/RNG match. Raw initial/change/final states match receipts; CPU scalar counters and CUDA moments verified from both runtime proofs and saved tensor storage metadata. Main learner remains uninterrupted; only the frozen control is restored.

Prechange fixed hold: 119/120. Changed observations passing: 0/220; maximum changed HQ 0.8388671875. Nonarrival is limited to this observed window.

Initial runtime manifest has 10 imported package files; later lazy K3P/gradient modules are covered by exact ZIP plus the successful final assert_public_imports guard, not a separately persisted final import map.

This is the declared 4,600-update released schedule, including noise warmup 460/920. It is a supporting reference, not a continuous-default candidate or a constant-LR result. No resumed process execution is claimed.
