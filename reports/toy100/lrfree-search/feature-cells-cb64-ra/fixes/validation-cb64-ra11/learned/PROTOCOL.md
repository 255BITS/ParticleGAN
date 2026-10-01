# Corrected CB64-RA11 CUDA acceptance

The candidate is frozen before any quality execution. Use physical GPU 0 only,
one numerical process at a time, deterministic algorithms and no TF32. Keep the
completed baseline's data, evaluator, seeds, G/D/prior initialization and gates.
Learned toy and MNIST: N=1024, z=128, batch=128, seed=314159, 2000 updates and all
ten saved checkpoints. Replay checkpoint 1000 twice for ten updates; only
birth_death.last.eval_seconds may differ. Reference measurements come from the
completed matched-input E22/CB64-RA CUDA runs; do not rerun unchanged controls.
Canonical screen.py and its scorers are unmodified, live noisy output is primary.
All 16 tasks receive their original budgets. Each native task has 7000 updates,
34 observations, five terminal 20k evaluations and a separate 100k holdout.
No source, config, quality gate, fixture or seed may change during this lane.

The full 19-job plan is frozen once. The first invocation stops after learned toy (--through 1);
a passing toy proceeds to full grid (--through 2); a later invocation verifies saved
job hashes before resuming. No completed job is repeated or overwritten.
