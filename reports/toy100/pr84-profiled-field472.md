# Empirical opponent-response secant at cold update 472

One same-budget critic refit at the exact accepted G endpoint produces a
much stronger restoring partial-field change than the frozen-critic secant.
This observation does **not** support simply increasing G's gain because its
accepted own-curvature estimate is small. It also does not prove a stable
equilibrium, a best response, or acquisition of the five missing modes.

The baseline is the already evaluated best finite critic from update 472's
failed fit. The bounded G proposal exactly reproduces the repaired live
replay's rho, factor, stencil width, clean-output RMS, noisy HQ and modes.
Its accepted factor is 0.07761247, clean support RMS movement 0.1211697,
and grade three modes/HQ 0.916992. All D/G/prior moments remain saved values
except inside detached cloned proposal construction; no live training runs.

For the endpoint fit, the initial D is the **same original accepted D***
that initialized the baseline fit. The 1024 real samples, particle indices
and output-noise tensors are identical. Only generated fake values change
when G/prior move to their accepted endpoint. This avoids conflating another
optimization attempt from the baseline best critic with a response to G.
One unchanged 40-iteration/80-attempted-closure safe fit uses 49 finite
closures, with no rejected trial. G's stencil remains fixed at 0.15.

Let `s` be the actual accepted G/prior parameter displacement, `P` the
post-base G Adam metric, and `g0` the baseline partial G field. For each
endpoint field `g1`, the reported quantities are

```
rho = ||sqrt(P) (g1-g0)|| / ||s / sqrt(P)||
k   = sᵀ(g1-g0) / (sᵀ P⁻¹ s)
```

The profiled endpoint field still holds its fitted D fixed when taking the G
gradient. It is **not** the total derivative of a composite G loss through
the fitting algorithm. Neither empirical critic fit is stationary; the
response is that of a particular bounded, nonconvex algorithm with a common
initialization, not a certified smooth best-response map.

| At the identical accepted G endpoint | Frozen baseline D | Refit D |
|---|---:|---:|
| Secant norm `rho` | 1.04304 | 8.16917 |
| Signed directional curvature `k` | 0.044325 | 7.84192 |
| Actual G factor times `rho` | 0.080953 | 0.634029 |
| Raw endpoint G/prior gradient norm | 3.17843 | 1.31406 |
| Adam-metric endpoint gradient norm | 0.973097 | 0.418309 |
| Endpoint partial work along `s` | −0.073264 | −0.028772 |
| Cosine with the initial partial field | 0.998855 | 0.944202 |

The negative endpoint work means this particular displacement remains a
descent direction for both endpoint partial fields. Refitting substantially
reduces the force along it. A larger displacement is untested; the positive
response curvature means a small frozen-D value alone is not permission to
increase it. The unbounded Adam proposal's own secant is 3.22113 with signed
curvature −1.91119, demonstrating why a single norm measured on the far
unbounded proposal need not describe the local accepted region.

The fitted baseline's D-gradient infinity norm is 0.007367. Moving G with D
held fixed raises it to 0.100601; the endpoint refit reduces it to 0.003008.
On an independent cloned eight-batch bank, endpoint penalized D loss improves
from 0.256096 to 0.249670 and gradient infinity norm falls from 0.116285 to
0.016710. These support meaningful local adaptation, with explicit residual
error. The endpoint solve took about 0.565 seconds in the pinned one-thread
PyTorch 2.13 CPU environment.

The [source](pr84_profiled_field472.py), [declaration](continuous-evidence/profiled-field472/declaration.json)
and [archive manifest](continuous-evidence/profiled-field472/manifest.json)
retain all fields, moments, paired banks and parameters. Input payload and
external RNG hashes are unchanged. No controller, loss, gain or frozen source
was modified; all evidence remains `shared_gate_eligible=False`.
