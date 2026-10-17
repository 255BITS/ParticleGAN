# Matched original E22 reference supplement

The first sealed extraction retained the older original E22 at
scaling-portability-20260929/validation/runs/mnist/E22. Its flat FD uses all
dimensions with std.clamp_min(1e-4), so it is descriptive only and is not the
later active39 reference.

Root identified the matched original CUDA E22 at
feature-cells-cuda-retest-20260929/learned/training/mnist/E22. Extract its existing
ten JSON milestones separately, verify the exact active mask/mean/std/reference
and evaluator hashes against RA11, and preserve the first helper/seal/result.
Use the supplemental active39 metrics for the final E22 comparison. This adds
no PT/model/scorer/import/numerical experiment or new quality observation.
