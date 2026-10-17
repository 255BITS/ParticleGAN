# Optimized BISPF investigation

The [current technique inventory](../technique-inventory.md) remains the single
comparison board. This readout records component diagnosis and bounded optimizer
iterations. The [protocol](../../../docs/bispf-optimization.md) freezes comparison
rules and performance targets. The PR remains a draft while final performance
and quality measurements are pending.

## Measured bottleneck

Unwrapped public `GANTrainer.step` medians on the existing DCGAN mock workloads:

| Workload | Native BISPF | R1/R2 | Native BISPF SVD share of instrumented update |
| --- | ---: | ---: | ---: |
| 64×64, batch 16 | 605.73 ms | 25.66 ms | 87.5%, 145 SVD calls |
| 256×256, batch 16 | 3,067.20 ms | 252.25 ms | 88.5%, 209 SVD calls |

The SVD share comes from a separate synchronized component replay. It does not
replace plain complete-update latency. Both replays consume identical initial
models, public prior initialization, batches and RNG streams and produce
identical final learned parameter hashes. Generator optimizer timings include
its prior groups; only prior sampling and regularization have separate spans.
Sinkhorn coupling and prior regularization each take a few milliseconds in
these workloads. Neither explains the original latency gap.

All observations use the same physical RTX A6000. Contention telemetry is
unavailable, so ratios describe observed runs and confer no certified speed
ranking. The [compact profiling receipt](profiling.json) records exact source
maps, declarations, runtime, hardware, component measurements, replay checks,
artifact identities and cost scopes. The original baseline combines core
`b5d2372f2` with the separately hashed added profiling harness; it is explicitly
a mixed source snapshot. Subsequent profiles bind complete committed trees.

## Retained optimizer iterations

The initial coupled Newton–Schulz solver formed a regularized channel Gram
matrix, then checked the inverse-root residual against that original matrix.
Although random matrix microbenchmarks were fast, real training gradients
triggered native SVD fallback for about 80 of 144 convolution slices per update
at 64 pixels. Plain medians were 515.81 ms at 64 pixels and 2,962.35 ms at 256
pixels. These results rejected that implementation as a latency solution;
neither its quality study nor its full performance campaign was launched.

The second solver orthogonalizes the augmented matrix `[M; lambda I]`, keeping
the absolute smoothing scale without squaring the original condition number
before iteration. It checks both augmented orthogonality and the relation of
the result to the original gradient. This removes all observed late-window
convolution fallbacks at 64 pixels, with a plain median of 102.02 ms. Its
24-iteration limit still rejects large critic kernels at 256 pixels, giving
1,513.20 ms. Increasing the bound and reducing launch overhead are the next
measured engineering steps. This iteration also has no full quality or
performance campaign execution.

Every successful approximate branch omits native SVD's hard numerical-rank
cutoff. This is an explicit numerical trainer change. On one captured 512×512
critic gradient, the converged unmasked result differs from the masked native
direction by about 1.16% in relative Frobenius norm. Small residuals alone do
not establish equality to the masked update or preservation of image quality.

## Quality scope

CPU kernels, tiny convolution kernels, dense weights, vector updates and sampled
prior-row updates retain native behavior. All current image-quality hosts have
at most 24 channels, so a complete tie on the existing suite would preserve
known quality results while exercising no accelerated convolution there.

The separate width-64 stripes and bars diagnostics widen only the original
architecture and declare their shorter wall allowance explicitly. Within each
new task, native and optimized BISPF share architecture, data law, initial
prior, public initializer, seen batches, sampling, 600 updates, 24 observations
and the original numerical quality gates. They exercise the accelerated path
and retain actual-training GIFs. This portability scope provides no ordinary
qualification and does not establish quality on large natural images.

The selected BISPF recipe, historical evidence and public trainer default remain
unchanged until the final candidate's evidence has been reviewed.
