# Validation scope

Both learning-rate arms use the same saved80k training state and seed. These short checks validate instrumentation and rates; they are not quality/seed experiments.

An initial16-step historical-versus-instrumented comparison did not match bitwise, including in one shared process. Training random streams matched. Enabling strict PyTorch determinism failed explicitly because `adaptive_avg_pool2d_backward_cuda` lacks a deterministic implementation. Therefore exact equality of complete independent training trajectories is not a supported validation criterion for this historical trainer. Failed attempts are retained in the launcher, paired_validation and deterministic_validation logs; no production job passed through the failed gate.

The replacement direct audit loads actual checkpoint weights and Adam moments, holds gradients equal, and proves that diagnostic observation preserves model/EMA/prior state (including prior exposure and child RNG), gradients, Adam state, module modes and global CPU/CUDA RNG bitwise. The subsequent Adam update and optimizer state also match bitwise. Fixed-input drift is exactly zero when weights are unchanged. See DIAGNOSTICS_AUDIT.json.

Certified16-update GPU1 smokes additionally verify actual G/D/E/prior updates, sigma preservation, exact configured optimizer rates, finite diagnostics, source hashes and full-state resume. Their results are written to VALIDATION.json before either20k production job starts. Smokes disable FID and are never leaderboard entries.

The direct audit isolates instrumentation side effects; it does not claim bitwise reproducibility of nondeterministic production training or establish the sole source of every cross-run numerical difference. Historical trainer and shared-library sources remain unchanged.
