# PR #155 handoff: prior EMA relaxation (diagnostic-host result)

## Verified state

`cf1-d2-prior-ema-relaxation-final` passes all three diagnostic-host native100 gates
with one SHA-256 package hash: `64f82d9edba2a1422206b8474867cfdd35a793e42f24727c4e08fb79d0d532fc`.
The tested source is `package/particlegan/`, with `overrides.json`; complete
scores and reproduction instructions are in [README.md](README.md).

| Task | Coverage | Accuracy | Terminal live checks | Holdout |
|---|---|---|---|---|
| grid100 | PASS | PASS | 5/5 | PASS |
| rotated100 | PASS | PASS | 5/5 | PASS |
| staggered100 | PASS | PASS | 5/5 | PASS |

**Canonical correction:** original frozen `screen.py` caches its first
real-batch callback result. The diagnostic host used for the table drew a
new batch on the second call. The first canonical staggered100 run is FAIL
(4/5 terminal accuracy; final centre .20266σ). The frozen 3/3 objective is
still open.

The broader 22-task matrix is 7 PASS, 2 FAIL, 13 ERROR. The native result is
a research result; the package is not a general replacement for the project
default. The standard tensor-only hosts need an API design for fresh second-D
real batches, and the eight custom hosts refuse particle birth/death parity.
Those errors were recorded as errors, without scorer changes.

## Mechanism

The earlier 2/3 candidate already used paired transport, two D updates, and
stationarity-triggered full prior EMA handoffs. Staggered centre error hovered
near .20σ after its prior reached a stationary verdict. The successful change
smooths the live prior continuously toward its existing EMA while that
verdict remains in force, with relaxation fraction `1-exp(-Δτ/b)` per update.
The rate comes from the tester's own intrinsic block length. The tester
anchor shifts by exactly the same non-gradient displacement, preserving
its gradient evidence. Optimizer moments, random streams, critic work,
output noise, and birth/death rule are unchanged.

The first full no-transport ablation failed staggered100. A block-boundary
version missed only the 6250 centre check. The continuous rule passed every
terminal check on all three native tasks. Fresh full runs from initialization
were used because a diagnostic checkpoint continuation did not replay exactly.

## Review points

- Read [training.py](package/particlegan/training.py) around the prior
  stationarity and relaxation branch. `continuous.py` contains the prior
  multiscale drift rule, and `birth_death.py` retains paired transport.
- The candidate is scoped to the frozen QR/noisy native100 protocol. The
  narrowest terminal margin is rotated precision, .9708 versus .9700.
- The optional diagnostic repair converts invalid tester rows to finite
  zero-contribution values only when logging block RMS; it does not feed
  training or scoring. All three final package runs have zero diagnostic
  serialization errors.
- Do not describe the full22 ERROR rows as model failures or passes. The two
  actual full22 failures are `vector_unequal_mass` and `vector_overlap`.

## Artifacts

- [manifest.json](manifest.json): package/harness hashes and native scores.
- [all22-results.jsonl](all22-results.jsonl): complete broad-suite status.
- [evidence/](evidence/): native result, job header, QR fixture, frozen
  verdicts, rates, and compressed diagnostics for each task.
- [all22/](all22/): per-task result JSON and logs for errors.
