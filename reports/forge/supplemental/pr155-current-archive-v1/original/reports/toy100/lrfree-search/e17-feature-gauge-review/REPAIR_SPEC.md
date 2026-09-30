# Predeclared sidecar check of covariance-normalized learned features

This follow-up was specified after seeing the E17 counterexample, before
running `whiten_probe.py`. It is a metric feasibility check only. Keep the
same fixed raw clouds, particle table, critic transforms, and PBD private seed
as [SPEC.md](SPEC.md). Add the non-diagonal invertible transform
`[[2,.7],[.3,1.1]]`. For each critic, estimate a full-rank covariance `C`
using **only** its learned features of the even-indexed real reservoir half
`R1`. Replace E17's extracted features by `h @ L^-T`, where `LL^T=C`.
This is an instance-local wrapper; leave Claude's E17 source unchanged.

Run E17's actual `maybe_apply` with the wrapper, plus the same independent
sidecar reconstruction, on the null and shifted cases. Use both E17's usual
float32 shortlist and exhaustive float64 distances. Check covariance condition
number; reject this fixture if ill-conditioned or rank-deficient. Compare
neighbor radii, density-ratio statistics, isolation scores and p-values,
BH flags, chosen move pairs, and final table against the base critic. For a
well-conditioned full-rank fixture, continuous statistics should agree within
`1e-8`; discrete decisions should match exactly. Logits must still match
within `1e-12`. Failure to match would rule out this straightforward repair
in the fixture. Success would establish only this fixture's affine invariance,
not general statistical validity, scalability, or native100 performance.
