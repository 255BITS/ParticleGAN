# Fixed-state forward-KL numerical accuracy audit

The cold cumulative run's GH9-reversed moves were replayed exactly at
updates 7 and 14: native real-bank hashes, GH5/GH9 objectives after every
recorded donor/EM move, and final clean outputs matched the frozen receipts.
The audit made no optimizer or neural update. It integrated the **difference**
`E_{real*N_h} log(q_old/q_new)` directly, so shared Gaussian normalizers and
the quadratic input term cancel. The tensor Gauss-Hermite orders were fixed
before the run: 9, 13, 17, 25, 33, 41.

| Cold transition | GH9 delta | GH41 delta | |GH41−GH33| |
| --- | ---: | ---: | ---: |
| update 7, EM20 | +1.61e−7 | +6.87e−7 | 5.33e−9 |
| update 7, whole update | −.00734073 | −.00732635 | 6.61e−7 |
| update 14, donor2 | +4.13e−5 | +4.05e−5 | 4.58e−8 |
| update 14, EM14–20 | +1.67e−7…+4.67e−7 | +3.56e−7…+4.96e−7 | ≤9.45e−9 |
| update 14, whole update | −.00189047 | −.00186704 | 9.40e−8 |

Positive delta means the proposed move **increases** cross-entropy. All nine
flagged inner moves remain positive at order 41, while both complete output
updates remain negative by much larger margins. This supports evaluating the
*materialized whole map* for a future mechanism, with proposal search kept
separate from final acceptance. It does not retroactively pass the old
per-move GH9 gate.

For the log ratio of two equal-weight, equal-covariance indexed mixtures,
`|log(q_old/q_new)(x)| ≤ A|x|+B`, where
`A=max_i|μ_new,i−μ_old,i|/v` and
`B=max_i||μ_new,i|²−|μ_old,i|²|/(2v)`, `v=h²+σ²`.
Gaussian-tail identities then give a **rigorous** upper bound on the absolute
contribution outside `[-8,8]²` in standardized target-noise coordinates:
at most `2.06e−13` for the audited whole updates and at most `4.03e−14`
for any flagged inner move. GH order differences are **numerical convergence
estimates only**, not rigorous bounds on integration error inside the square.
The signs are credible given the observed order differences, but the
continuous-integral claim remains uncertified. Each order sequence took
about 1.5 seconds for update 7 or 3.0 seconds for update 14 on one CPU
thread, per transition.

Source, full pairwise values/clouds, transcripts, and verified SHA manifest:
[`round8-forward-kl-quadrature-audit`](continuous-evidence/round8-forward-kl-quadrature-audit/manifest.json).
Two tests verify the independent log-ratio integral against the original
GH5/GH9 objectives and check the pointwise log-ratio envelope used for the
tail bound; both pass. The older GH5-only first-bank and cumulative results
remain unchanged in their separate archives.
