# Support diagnosis

Status: **unqualified research**. No support patch was merged, no support READY
was issued, and this lane initialized no CUDA context. The privately staged
`SUPPORT-SCORE.diff` is the rejected prediction-only experiment, not an
integration instruction. Verified performance artifacts were untouched.

## Established causes

1. **Sparse cell uncertainty.** Original trained folded N2048 flags 85 rows for
   82 unsupported rows; false positives are 131, 1166 and rare1745. Row1745 has
   one even real row and zero odd rows in its nearest cell. The spherical scale
   omits uncertainty in the estimated center. The factor `1+1/n` removes that
   rare flag but adds rare1924 under the frozen critic. It is not a universal fix.
2. **Local shape, not just sparse counts.** The Fisher/Student frozen rare1773 has
   n=13 and nine odd partition rows; it is not a singleton. Full covariance
   uncertainty removes that row, but another rare row3262 fails at frozen N4096
   (n=8, 13 odd partition rows). These shifts reject a sparse-only explanation.
3. **Representation and centroid geometry.** All 128 highdim critic features
   vary on real rows; no real-dead feature is being dropped. Unweighted residual
   magnitude is smaller for anomalies than many real rows. A real-only bounded
   dictionary Fisher8 score gives 46/92/184/369 true positives with zero false
   positives at N1024/2048/4096/8192: the anomaly information was present but
   diluted by nuisance variation. Audit's actual Fisher real anchors also
   recover trained N1024 (41/41, zero FP) where centroids do not. Their raw
   unsupported minimum is 508.55 versus null maximum360.72; centroid minimum
   57.20 is below null maximum82.13. Local-radius normalization loses this gap.

## Finite-reference margins

Geometry head inputs are float32; inherited `_features` converts them to
float64. A rule using `finfo(real_features.dtype)` therefore measures converted
arithmetic precision. The first such attempted source-floor test was a no-op.
The corrected source-head precision test is recorded separately and still fails
the family; its result is not used as a silent precision change.

In the full covariance NIW trained N1024 diagnostic, unsupported scores span
16.0529–18.9526 while the two largest odd-real scores are18.4715 and18.4092.
Only six query rows exceed the largest null score. With M512 and Q=.05, at least
40 rows at empirical floor1/513 are needed for a BH rejection; the best ordered
p/threshold ratio is2.9211. The observed overlap prevents power for this score.
Actual-anchor posterior recovers all41 at unchanged gates, so this is not proof
that the critic or finite real data make detection impossible.

Row1745's semantic position is approximately(3.0307,-.1354), with nearest even
and odd real distances .0377 and .0721 (oracle analysis only, after scoring).
It is a supported rare tail row absent from its fitted cell's odd calibration.
In the centroid NIW diagnostic it has p=.002927 and is kept; in the actual-anchor
posterior it has p=.000976, score17.7195 versus null maximum15.8063, and fails
the unchanged zero-rare-FP gate. Frozen rare1773 is kept by the actual-anchor
posterior with p=.038049. A universal support score remains unresolved.

## One combined posterior qualification

The final diagnostic uses existing actual even-real representatives, bounded
Fisher8 directions, the same four-row pooled covariance prior and degrees n+6.
Even-row anchor scatter includes the representative-to-mean offset. Odd rows
alone calibrate the score; original cell assignments, count tests and gates stay
fixed. It passes **14/18 detector cases**. Rejections are trained N2048 rare1745,
frozen N4096 bulk FPR11/3932=.002798, and zero recall for cost nominal/highdim
N1024. It also flags two supported rare rows in N2048/rare_hole. This is not an
end-to-end result and is not recommended for integration.

## Implementation contracts

Two numerical defects were separated from quality failures. Audit's bounded
Fisher helper previously produced non-finite intermediate eigenvalues for zero
within covariance; its observed-real-scale floor and strict finite contracts
are preserved under `geometry/support-highdim/`. The anchor scatter expansion
initially assumed transformed residuals summed exactly to zero. Adding the
residual-sum cross terms and a scale-conditioned forward-error bound makes the
expanded/direct scatter agree: maximum absolute error6.82e-13,
maximum error/bound0.00673. Direct residual
scatter drives scores. The old 8e-9 assertion failure is retained in its log;
no quality threshold was altered. Helper SHA256: `ab67cbd221de191febe696207dfc088e3150c0feedc6c3ed79b62cdbef4b95d6`.

## E22 context and recommendation

The archived E22 detector also has zero recall on all four highdim costs and
trained N1024/fold_nuisance. At trained folded N2048 it recalls68/82=.8293 with
four false positives, including two rare rows; frozen N2048 recalls82/82 with
five FP including one rare. Matching these failures is not qualification.

Proceed with the separately verified performance, sampling, mass and small-N
corrections. Preserve this bounded metric and its contracts as research evidence.
Require one unchanged support law to pass all required trained/frozen/cost gates
before merging it. Do not tune priors, seeds, Q or the zero-rare-FP gate to these
individual rows.

Saved numeric evidence: `results.json`, `LEADERBOARD.md`, `margins.json`,
`diagnosis-student.json`, `diagnosis-niw.json`, and
`diagnosis-anchor-posterior.json`. Other `diagnosis-*.json` files retain rejected
causal decompositions. Every experiment records its source/input hashes; no
GPU or learned-training portability claim is made here.
