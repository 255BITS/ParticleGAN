# CB64-RA failure diagnosis and fixes

The user requested small diagnostic tests, causal explanations and fixes after the first canonical CUDA failures. This work starts from the unchanged CB64-RA package and config in `BASELINE.json`. The original CUDA study continues as baseline evidence.

## Agent responsibilities

- `performance/`: locate CUDA synchronization/kernel overhead; implement and test count/feature batching with equivalent statistical semantics.
- `stability/`: isolate small-population actuation, integer allocation, parent supply, rare mass and terminal covariance instability.
- `geometry/`: separate center/mass failures from latent perturbation and serving effects, including folded support and MNIST embedding diversity.
- Root: integrate independent patches into `pkg-CB64-RA2`, run required GPU diagnostics serially and validate the corrected candidate.

All three diagnostic agents use GPT6.1 with maximum reasoning effort. Each edits its private package and records reproducible tests and a proposed patch. The shared candidate and original artifacts remain unchanged during diagnosis.

## Diagnostic and validation rules

Use existing fixtures and their original seeds. A diagnostic should distinguish a specific cause or verify a contract that failed; avoid seed sweeps and blind parameter searches. CPU inference and mathematical tests are diagnostic evidence. CUDA profiling, training and quality checks run on physical GPU0 under root's serialized scheduling. GPU1 remains occupied by other work.

Changes to statistical tests, sampling law, routing or config must be stated explicitly, justified by a reproduced failure, and checked for mass/support/state regressions. Preserve a common perturbation law across training, serving, fake-pool and row copies, or document and test any deliberate exception. Preserve checkpoint continuation and bounded compute.

The original quality gates remain unchanged. After integration, freeze the corrected source/config and commands, run targeted failing cases first, and run the full prescribed acceptance suite for a candidate that qualifies. Paired learned validation uses the saved data/evaluator and reference initialization; no quality conclusion is substituted from CPU results.

## Scheduling

The root may reserve a brief diagnostic GPU slot between original baseline jobs by parking only its launcher, allowing the current numerical child to finish, then running one diagnostic and resuming the launcher. Record reservations and actual child completion. Never pause an active numerical child. Baseline trainer/scorer timing stays in its own result; launcher wall time spanning a reservation is orchestration time.

## Current observations

The frozen CUDA learned runs show lower allocated memory but slower training and lower measured quality for CB64-RA. All four exact checkpoint replays pass. The initial original-harness failures include mode_hold, img_blobs4 and vector_unequal_mass; the last has passing final metrics but an insufficient passing suffix. Agent tests will determine mechanisms before fixes are combined.
