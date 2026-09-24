# Smoothed forward-KL, pure-output first-bank screen

This is a free-output diagnostic, not a ParticleGAN update. The target is the
next native real128 bank convolved with a fixed Gaussian width
`h=0.031286240422040236`, measured from the first cold bank. The generated
density has twelve equally weighted Gaussian atoms at the clean outputs,
with covariance `(actual_output_sigma²+h²)I`. Cold update 1 has output sigma
zero; warm update 1324 has output sigma `.029`. The update never reads target
ring centers, mode labels, or diagnostic quality. It globally replaces a
donor atom by an observed real coordinate at most twelve times, then makes
at most twenty fixed-weight/covariance EM centroid steps. Each accepted step
lowers the declared fixed 5×5 Gauss-Hermite cross-entropy. A separate 9×9
quadrature audits each accepted point and does not choose updates.

| Source epoch | Saved cloud and next real bank | Objective before → after | Late-noise quality before → after | 9×9 reversals |
| --- | --- | ---: | ---: | ---: |
| v2 | cold1, own bank | 4486.9520 → 1.71758 | 0 modes/HQ 0 → 8/HQ .999756 | 0 |
| v2 | warm1324, own bank | 2.21736 → .200653 | 8/HQ .999512 → 8/HQ .999512 | 0 |

The first archived source epoch v1 used the cold bank for both saved clouds.
Its cold result is unchanged; its warm result is a shared-bank diagnostic and
is superseded by v2 for the native-bank screen. The exact source bytes,
declarations, raw rows, and SHA-256 manifest for both epochs are in
[`round8-forward-kl-first-bank`](continuous-evidence/round8-forward-kl-first-bank/manifest.json).

GH5 and GH9 are finite numerical quadrature rules, not a certified integral
bound. In the cold final state, the GH5 gradient norm is `.00233` while GH9
is `.06897`; the maximum coordinate difference is `.03870`. Their accepted
step objective signs agree, but this does not certify a stationary point of
the continuous forward KL. The late-noise HQ score is an external diagnostic;
it is not the cold sigma-zero optimization objective. Output displacement
can be large (maximum warm particle movement `6.086`), so a neural pullback
may fail even if this free-output screen passes.

Validation: three independent numerical tests check Gaussian quadrature
normalization and second moments, the global donor against brute-force
replacement enumeration, and log-space EM monotonicity with a distant atom.
`tests/test_forward_kl_free_filter.py`: 3 passed. The next predeclared screen
is a frozen-width cumulative 16-bank free-output continuation, with warm
quality checked on every update and cold last-five quality checked only if
warm passes. No native training or optimizer change follows from this result.
