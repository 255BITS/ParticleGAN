# Isolated output-noise RNG validation

The optional `output_noise_rng: "isolated"` policy is implemented and archived,
but it has **no 22-task pass**. It gives generator output noise its own
checkpointed stream, seeded at the declared seed plus 1,901. Scored native
evaluations and the 100,000-draw holdout restore that training stream; legacy
generated-sample receipts cover every frozen checkpoint. Omitting the selector
retains the previous RNG path. No `particlegan/` or `lib/` source changed.

After integrating the policy, the stricter combined regrader accepted both
completed CPU seed-0 transfer archives as valid evidence and retained their
failures:

| Shared configuration | Older toys | Failed hosts |
| :--- | ---: | :--- |
| `accuracy_shared_policy_isolated_rng.json` (κ=1.25, common LR floor .05) | 14/19 | residual student, mode hold, unequal mass, overlap, bars4 |
| `shared_candidate_isolated_rng.json` (κ=1.0, network floor .01) | 11/19 | trajectory, residual student, mode hold, unequal mass, unequal width, overlap, bars4, blobs4 |

The raw archives are retained locally under
`artifacts/toy100-accuracy/compatibility/isolated-rng-shared-policy-all19-v1/candidate19`
and `artifacts/toy100-accuracy/isolated-rng/full19-1c1a086`. The first
archive retains the final numerical misses. These are transfer results, not a
combined 22-task verdict.

At frozen source commit `1c1a086`, the full unit suite passed with **900 passed,
8 skipped, 1 expected failure, and 27 subtests**. The focused suite after the
legacy checkpoint-completeness fix (`982b58e`) passed **93/93**. Tests cover
old-mode parity, exact private-stream replay after checkpoint resume, malformed
checkpoint rejection before state mutation, evaluation restoration, native
holdout restoration, omitted/tampered receipts, and selector-only V2 archives.
A separate real 1,000-step native CPU run with the selector and no affine or
network-horizon option passed the portable archive and RNG receipt checks after
producing its 100,000-draw holdout. Old V1 and V2 native evidence and older
transfer archives still regrade under the integrated code.
