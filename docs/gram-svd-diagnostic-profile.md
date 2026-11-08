# Explicit SVD diagnostic fallback profile

`optimizer_dense_profile="gram_fp64_svd_diagnostic_v2"` is opt-in. Public recipe defaults and the original `gram_fp64_guarded_v1` profile are unchanged.

The original guarded FP64 Gram operator runs first. Only its exact ambiguous-rank exception permits device-local FP64 SVD (`gesvd` on CUDA, native SVD on CPU). Technical errors are never retried on another device. The original matrix dimensions and FP32 epsilon define the strict cutoff `s > max(m,n)*eps32*s_max`; absolute smoothing is 0.001 in the qualified Sana recipe. The weighted direction is cast to FP32 once before the existing public optimizer applies aspect scaling, rounded parameter writes and DirectionBlend. No singular value is silently zeroed or retained to evade a gate. Exact computed ties drop.

## Cutoff uncertainty is diagnostic

The finite computed SVD rank is a well-defined PG decision. A conservative floating-point error model can fail to certify that rank without showing a bad factorization or invalid training update. V2 records this as `uncertified_svd_rank_decisions`, `rank_certified=False` and `rank_decision_policy="public_strict_computed_svd"`; it does not make rank uncertainty fatal. Independent finite/sorted, orthogonality and reconstruction arithmetic-model checks remain fatal. Nonfinite results and technical solver errors still propagate. This is a versioned numerical profile, not a bitwise replacement for native FP32 SVD or the original Gram-only profile.

## Exact failing-case evidence

Sana G84, `transformer_blocks.1.attn2.to_q.weight`, was reconstructed from authenticated checkpoint64. All 19 updates65–83 matched the original saved metrics exactly before reproducing the frozen guard failure. The FP32 gradient raw SHA256 is `0d33e98f2b0dd7e7217cb393be5c5d74527636df718146c019f3fd463c91f8e4`. Same GPU FP64 SVD factors gave a normalized cutoff gap1.05985957253368e-8 and conservative error1.0966917634044585e-8. Reconstruction residual was7.639137273427134e-15; the blanket diagnostic-reduction term was9.014351633470673e-10 before the fixed8x margin. A separately reviewed residual-aware model gives3.755453013759486e-9 without changing the threshold; it is diagnostic research, not integrated here.

GPU FP64 SVD and independent CPU Torch, LAPACK gesvd and gesdd agree on rank67. The two LAPACK weighted directions differ by2.52e-14 relative Frobenius. Current native FP32 PG selects rank68; its direction differs from the CPU64 reference by0.789%. Toggling the single boundary mode changes this leaf direction by0.03506086 Frobenius, or0.00788816 relative, and its unrounded parameter increment by2.80487e-6 Frobenius at lr8e-5. This is not evidence of equal future learning or a whole-game branch guarantee.

V2 on the exact matrix returns rank67 with explicit uncertified status and4.09e-13 relative error versus the FP32-cast CPU64 reference. Fixed analytic public-optimizer regressions cover above/below/tied cutoffs, original cutoff dimensions, metadata/restore identity, unchanged primary passes, corrupted U/V/S, nonfinite/sortedness failures and error propagation. Native continuation and quality outcomes are separate evidence.

## Checkpoint and failure contract

V2 is serialized as a distinct profile and rejects mismatched optimizer metadata before mutation. Diagnostic counters are nonnumerical, excluded from checkpoint/deepcopy/pickle state and reset on successful load. This does not make a public game update transactional: a fatal failure can follow earlier D/G writes and advanced clocks; discard that failed game and restore only a complete authenticated checkpoint. Changing an existing checkpoint profile requires an explicit source-bound continuation manifest and whitelist validation; do not relabel historical results.
