# Current task: improve the selected dimension RMS GAN base

Read AGENTS.md, ../current-research-base.json and ../dimension-rms-base/README.md.
The user explicitly selected dimension_rms_hybrid as the new research base.
Start from its exact config.json + mechanism.py + probe.py. The config by itself
is NOT the selected formulation; it requires the regularizer installed by the
probe. Use the archived checksum-verified source preparer and retained fixtures.
Do not drift back to the original CPU recipe as the starting candidate.

The selected base independently passes six GPU regression toys and fails two_pole
(movement .1044 < .30). Full22 and own-state stability are NOT_RUN. It remains
Rp logistic GAN training with a real dimension-normalized R1 term and a fake
RMS b-cap. No target-fitter substitution. Preserve model shapes, fixed seeds,
data, budgets, thresholds and evaluation. Keep original decay/noise schedules
and auxiliary host terms as defaults; declare any assigned formulation change.
All model training, gradients and Adam state are CUDA; CPU initialization is
allowed and pinned. Retain original Adam arithmetic (capturable changes it).

Use two_pole as the first cheap gate. If it fails, stop that proposal and adapt.
If it passes, run trajectory, then ring and unequal mass, followed by intensity,
bars and blobs. Passing candidates continue; measured failures stop qualification.
Only a candidate clearing all seven advances to its own remaining full22 gates.
Full22 must precede an own-state post-convergence continuation under its declared
schedule. Acquisition and retention are separate results. No borrowed passes.

Three fresh Astra/max attempts, one CUDA worker each, at most three distinct
proposals per attempt and 45minutes. These are caps, not quotas. No nested agents,
seed sweeps, coefficient grids or uncontrolled extra model/optimizer updates.
Start a real candidate within five minutes. Small evidence-driven formulations
and actual frozen tests take priority over theory or new harness infrastructure.
Do not rerun completed unchanged controls, failed caps/optimism variants, or the
PyTorch upgrade. R1-containing candidates remain eligible when their tests pass.

Use local candidate snapshots and fixtures. Prepare source once, verify hashes,
and save exact declarations before execution. Rebuild expected specs from the
candidate recipe: image gradient_penalty/penalty_coeff/kappa are aliases for the
regularizer fields. Keep frozen data/budgets/scoring unchanged. Retain actual
CUDA update counts, state devices, source/config hashes, raw metrics and all
FAIL/ERROR/SKIPPED entries. Do not count a repaired receipt audit as new training.
Run focused mechanism checks; keep tests.jsonl and concise logs easy to inspect.

Prior scale alternatives: dimension_rms_bcap and detached_fake_scale_cap both
lose ring coverage. Two_pole uses d=1, where the selected real term has its
original R1 strength. Low movement suggests suppressed early learning, but that
cause has not been isolated. The new lanes own dead-zone geometry, real-penalty
warmup, and adversarial particle-update response. No toy-name special cases or
target/metric inputs to training. Summarize measured results and exact replay.

[Previous formulation history](formulation-search-before-dimension-rms.md).
