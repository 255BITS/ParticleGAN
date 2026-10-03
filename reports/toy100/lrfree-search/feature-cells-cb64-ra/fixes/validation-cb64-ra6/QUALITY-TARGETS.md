# Toy and grid quality repair

The target is one prospective candidate passing both gates. Descriptive MNIST
gains do not qualify it. Previous RA2/RA3/RA4 results and all frozen inputs remain
separate evidence.

## Required acceptance

- Learned toy: unchanged saved data and G/D/prior initialization, seed 314159,
  N=1024, latent width 128, batch 128, 2000 updates and all ten checkpoints.
  The final emitted samples must have precision >=0.90, all 25 modes with at
  least 1% supported mass each, and mass TV <=0.10.
- Grid100: unmodified canonical CUDA host and scorers, seed 1234, N=20000,
  latent width 2, batch 2048 and 7000 updates. All original precision, coverage,
  mass, centroid, covariance, radial, KS and stability gates apply, including
  five terminal 20k evaluations and the independent 100k holdout.
- Only physical GPU0 is used, one owned numerical job at a time, deterministic
  algorithms, TF32 disabled, per-process memory fraction 0.2. Existing jobs and
  other users' processes are preserved.
- A candidate's package, config, commands, declared generation API and evidence
  checks are frozen before its first quality run. Source/runtime/fixture
  validity is audited independently of a quality PASS or FAIL.

## Diagnostics and candidate order

Three GPT6.1 max agents investigate distinct mechanisms on saved CPU inputs:
parent availability and accepted real-anchor proposals; latent/observation
sampling and controller geometry; training/transport/stationarity dynamics.
These diagnostics do not produce GPU quality verdicts.

Production repairs may use observed real data and learned features. Oracle toy
mode IDs, grid geometry, held-out evaluator metrics and benchmark-specific
branches are excluded from training decisions. Serving/noise behavior must
remain honest. New latent proposals must be identified as proposals, with
bounded work, unchanged support acceptance and the existing shared action
budget/quota/ledger. Copy-parent uniqueness remains an explicit copy contract.

Prefer the smallest mechanism or configuration change supported by a diagnostic.
Run the unchanged learned toy first. A failed toy is retained as a negative
result; do not spend a full grid run on it. A toy PASS receives the complete
grid schedule. Required state/replay and portability regression checks follow
a candidate passing both quality targets. No seed experiments, score/gate
changes, budget extensions or selection of an earlier checkpoint as final.

The already-running frozen RA4 grid continues to its original final checks.
Additional owned GPU jobs wait until its numerical child finishes; a supervisor
may be parked between jobs to allocate a serial candidate slot. A stopped
supervisor does not change its numerical child or any fixture.

## Reporting

Maintain a leaderboard of final precision, coverage and mass TV for the toy;
all native grid gate verdicts; source/fixture validity; and descriptive cost.
Record the mechanism tested, fixed-input diagnostic, frozen source/config
identity, full checkpoint path, exact status and remaining issue for every
candidate. Archive valid negative findings alongside passing results. A
configuration is recommended for this target only after both gates pass.
