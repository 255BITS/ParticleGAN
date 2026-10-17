# One detector-only reconstruction

Use the four completed observation windows: static toy 750–835, static MNIST 100–215, and the original moving quarter-rate windows beginning at 500 and 1000. Bind their exact bytes plus RA12's original detector and RA13's provisional detector/guard source before execution.

Extract only `OptimizerSurprise` and `SettledReopenGuard` class ASTs. Use CPU float32 scalar `q` values from the recorded pending observations. The unchanged original detector and the candidate's legacy default must reproduce every captured decision and scalar history exactly. No model, PT file, forward, draw, optimizer update, scorer, or CUDA context is used.

For the guarded prefix, use the exact candidate implementation: contracted explicit network role witnesses, last-calm/onset excursion latch, unchanged K/RISE/CALM and all-group aggregation, and its one-time reset at the actual KA2 false-to-true loss epoch. Seed the visible epoch, not a fabricated step number. The first moving window is before the epoch and the second is already after it; do not invent a transition at the second window's left boundary.

Initialize scalar history from each recorded left boundary and qualify this as a nonresumable diagnostic. Stop at the first original or proposed action: subsequent gradients could differ under changed actuation. Report failure if the instantaneous epoch reset does not suppress the toy event. Report both moving fires explicitly. No threshold, seed, task-label, rotation, target, model, rate, or package change is part of the reconstruction.

The result proves mechanics only on the recorded prefixes. Fresh guarded CPU/API and original CUDA quality are separate requirements.
