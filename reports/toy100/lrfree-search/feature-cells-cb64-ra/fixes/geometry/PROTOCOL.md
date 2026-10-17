# Geometry and diversity diagnosis

This lane owns this directory and its private `pkg-CB64-RA2` copy. All baseline
studies, fixtures, checkpoints, configs and the shared candidate are read only.
The user requested small causal tests and fixes after the canonical GPU failures.
These initial tests use CPU only; they cannot certify GPU quality acceptance.

## Fixed diagnostics

1. Read the final saved toy and MNIST CUDA checkpoints for E22 and CB64-RA on
   CPU. Derive the declared serving state from the saved table stationarity
   decision: EMA when the table's last decisive verdict is stationary, otherwise
   the fast iterate. Preserve each checkpoint's recorded output sigma.
2. For each final MNIST generator/prior, compare clean centers, repeated centers
   plus output noise, the frozen .025/.05 feature-cell kernel, and the unchanged
   DV12 adaptive bandwidth plus exact half-nearest-nonidentical-latent cap.
   Each sampled case uses 4096 draws, seed314259 and batches256. Reuse the same
   row IDs and Gaussian latent/output noise across kernels so a difference is
   attributable to perturbation. CPU RNG differs from CUDA; record this scope.
3. Use the original saved evaluator, real training first5000 normalization,
   original test reference first5000, original active-dimension rule and
   manifold k5 / first2048 protocol. Report class mass, per-class recall and
   class-balanced weighted Fréchet diagnostics to distinguish mass imbalance
   from within-class geometry. No diagnostic enters training or changes gates.
4. Reuse the existing trained600 folded2D and folded128D N2048 fixtures and
   seed20260929. Compare emission after exact center repair for the same fixed
   kernels, preserving the oracle support/mass definitions. No new critic
   training, seed search or acceptance threshold changes.

## Fix constraints

Implement only a correction justified by these diagnostics. Training,
fake-pool, serving and row copies must use the same declared latent geometry.
Select the corresponding live/EMA prior, keep gradients through latent inputs,
keep private stream ownership, copy optimizer/history/EMA rows and preserve
checkpoint continuation. Geometry caches must be invalidated or recomputed
after table motion and row copies. Large-N compute must remain bounded and any
approximation to reference DV12 must be disclosed and tested.

Small-population statistical fallback belongs to the stability lane. This lane
does not change calibration, support tests, count tests or actuation budgets.
Root integrates the private patch and schedules any CUDA commands serially.

The first bounded isotropic proposal preserved repaired-center emission gates,
but four actual folded2D copy turnovers retained only .88235 of rare mass,
below the unchanged .90 gate. Its sources/results are preserved separately.
The second correction bounds each coordinate's controller width by the RMS
pair difference among the nearest rank8 bounded candidates divided by sqrt(2).
This derives local anisotropy from the table, addressing narrow folded latent
coordinates without an oracle, tuned scalar cap or altered statistical gate.

Deliverables: diagnostic JSON/logs, focused failing-before/passing-after tests,
private implementation patch, source hashes, concise findings and a minimal
GPU diagnostic command. No fixed-kernel comparison alone establishes that a
retrained corrected candidate will meet the original quality gates.
