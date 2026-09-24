# Current task: make the known 22/22 CPU GAN recipe work on GPU

Start a fresh attempt from `configs/toy100/constraints_simple_regularization.json`.
Read AGENTS.md, START.md, ../current-research-base.json, and
../cpu-recipe-gpu-port/README.md. Implement and run small, measurable experiments.

The user has changed the immediate priority: preserve the original passing CPU
recipe and resolve GPU failures first. Its learning-rate decay, noise schedules,
ordinary Adam, and original auxiliary AE/token host terms belong to this base.
Earlier instructions forbidding decay or those host terms apply to the separate
continuous-rate research track, not this porting task. Keep GAN adversarial
training central; do not substitute a target fitter or introduce target labels,
centers, or evaluation feedback into training.

The original recipe scores 16/22 in the full native-CUDA audit, including all
three native 100-mode gates. All six failing hosts pass fresh CPU runs. Copying
CPU initial parameters before GPU training recovers trajectory and all three
image gates with identical native-CUDA random draws. Ring and unequal mass
remain failing. Moving all random draws to CPU recovers a different four; it
still fails ring and bars. All 24 runs are audited and retained. Do not rerun
these completed controls or treat any partial diagnostic as 20/22.

Use CPU initialization with CUDA training as the practical porting reference.
First gate on `mode_hold` (seven modes) and `vector_unequal_mass` (rare component
minimum covariance eigenvalue ratio below .15), then trajectory, intensity,
bars, and blobs. Test the complete 22 only after a promising proposal clears
that regression screen. Every scored run must use CUDA parameters, gradients,
and Adam moments. CPU initialization is allowed and must be declared; CPU model
training cannot be credited as a GPU pass.

Try at most three meaningful proposals at a time. Favor small changes motivated
by the measured initialization/random-stream/arithmetic sensitivity, not broad
coefficient grids. No seed sweeps. Keep architecture, data distribution, fixed
budgets, targets, scoring, and thresholds unchanged. Do not select a separate
randomness policy per toy or borrow passes from another candidate. Retain all
FAIL/ERROR results and repair harness errors before scoring them as failures.

We care most about stability after convergence. A finite-budget 22/22 pass is
not proof of indefinite stability. After complete GPU qualification, explicitly
measure a continuation from that candidate's own converged state, preserving
its declared schedule and RNG. Do not quietly force constant rates onto this
scheduled recipe or let convergence metrics control training.

Use the source-verified replay/prepare scripts in ../cpu-recipe-gpu-port for the
controls. Keep source/config hashes, backend environment, raw metrics, actual
optimizer rates, and device proofs. Write concise logs and a result table with
PASS/FAIL/ERROR/NOT_RUN separately. Run relevant regression checks after code
changes. Report the strongest measured candidate, remaining failures, and exact
replay commands. Do not claim release qualification before all required checks.

[Earlier continuous-rate brief](SEARCH-continuous-history.md) is historical.
