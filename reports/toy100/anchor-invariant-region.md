# A conditional invariant region for sampled distinct anchors

The fixed-center [coverage proof](anchor-coverage-principle.md) can be made
quantitative near a covered cloud. The result below controls clean output
locations despite arbitrary native GAN proposals. It requires correct,
bounded-error sampled groups and a sufficiently accurate eligible output
fit. A proposed **failed-fit → restore pre-G state** guard removes the need
for every fit to succeed. That guard is analyzed here, not implemented here.

No learning-rate decay, positive movement floor, critic zero pull, or known
host center enters the update. Known centers are used only in the offline
assessment below. This is our direct derivation, not a neural convergence
theorem borrowed from a paper.

## Region and matching margin

Let the fixed population centers be `μ₁,…,μ_K`, with minimum separation
`Δ>0`, and let `N>K`. Define the good region `S_R` by two requirements:
every clean generated row is within `R` of some population center, and each
population center has at least one such row. At a particular update, assume
the current data-derived groups can be bijectively labeled so that
`||c_k−μ_k||≤ε`. The controller does not need these labels.

Suppose

```
Δ − R − ε > √K (R + ε),
equivalently Δ > (1 + √K)(R + ε).                 (1)
```

This also implies `R<Δ/2`, so true-center labels of the input rows are unique.
A correct injective anchor assignment exists because all groups are covered.
Its unnormalized squared assignment cost is at most `K(R+ε)²`. Any assignment
with a wrong anchor contains a pair whose distance is at least `Δ−R−ε`.
That one pair costs more than the entire correct assignment by (1).
Consequently every minimizing Hungarian assignment is correct. The nearest
sample-center labels are also correct: their distances are at most `R+ε`,
while every competing distance is at least `Δ−R−ε>R+ε`.

For the actual unit-mean distinct-anchor objective

```
L_c(Y) = min_injective_a (1/K) Σ_k ||y_a(k)−c_k||²
         + (1/N) Σ_j min_k ||y_j−c_k||²,
```

the active quadratic MM target is therefore exactly `c_k` for every row
whose true label is `k`. Its anchor and nearest-center terms agree; their
weighted minimizer cannot fall between different groups.

## Approximate fitting and native-proposal selection

Suppose an eligible joint G/prior fit realizes each target with row error
at most `δ`. Include target conversion/rounding in `δ`. The fitted cloud has
`L_c≤2δ²`, all rows within `ε+δ` of their population center, and at least one
row for every center.

The implementation may select the native GAN proposal when its loss is
lower than the fit. Thus `ε+δ` is **not** a bound for every selected update.
Let `τ` be the actual strict-comparison tolerance. If a native proposal is
selected while the fit is eligible, the selection logic implies
`L_native≤L_fit+τ` in exact objective arithmetic. If each computed loss has
a known error bound `η`, the true native loss is instead bounded by

```
ℓ = 2δ² + τ + 2η.                                (2)
```

For any cloud with loss at most `ℓ`, each row is within `√(Nℓ)` of some
sample center: a single precision summand cannot exceed `Nℓ`. Separately,
the minimizing injective coverage assignment puts a distinct row within
`√(Kℓ)` of each sample center. Therefore the selected moving cloud satisfies

```
every row:        distance to a true center ≤ ε + √(Nℓ),
every true group: a distinct anchor at distance ≤ ε + √(Kℓ).       (3)
```

Both the fitted and native selections stay in `S_R` if

```
ε + √(N(2δ²+τ+2η)) ≤ R.                          (4)
```

Rest retains the previous good cloud exactly. If (1) and (4), correct group
recovery, and the stated numerical error bounds hold at every update, then
induction gives conditional output-region invariance. For the current rule
this statement additionally assumes every fit is eligible. For a variant
that restores pre-G parameters whenever fitting is not `CONVERGED`, fit
success is no longer needed for *invariance*. Acquisition and response still
need successful progress, which this local result does not establish.

This is a finite-computation statement. A fatal nonfinite native proposal or
exception before restoration is not a certified rest. Nor does an output
region bound parameter norms, Adam state, or Jacobian conditioning; internal
nullspace drift remains a separate diagnostic and long-run concern.

## Why both qualifications are necessary

Exact rational checks retain two counterexamples with centers `{-1,+1}`,
`N=3`, and `K=2`.

* Without an eligible fit, loss descent can leave a radius-0.1 region.
  The pre-cloud `(-0.9,1.1,-0.9)` has loss `1/50`. The native cloud
  `(-1,1,-0.85)` has lower loss `3/400`, but its maximum radius is `0.15`.
  The current nonconverged-fit fallback can accept this update.
* Even with an eligible fit, native rows need the normalization factor in
  (3). A fit `(-0.99,1.01,-0.99)` has error `δ=.01` and loss `.0002`.
  The native cloud `(-1,1,-0.98)` has smaller loss `1/7500`, but one row's
  error is `.02>δ`. The `√N` loss-to-radius bound correctly covers it.

The tests also enumerate 1,458 exact rational noisy-center/covered-cloud
combinations satisfying (1); every Hungarian/nearest label is correct and
every exact MM target is its current group center. Four tests pass in
0.28 seconds. No training or new random seed was used.

## Posthoc assessment of the actual saved 44 updates

The [read-only assessment](anchor_invariant_region.py) uses the frozen
sample-anchor saved44 receipt and its exact initial-cloud replay. The
[complete result](continuous-evidence/anchor-invariant-region/assessment.json)
binds both input hashes and the actual host geometry source. It measures:

| Quantity | Observed/bounded value |
| --- | ---: |
| Particles/groups | 12 / 8 |
| True minimum separation `Δ` | 2.296099608 |
| Initial maximum radii, three branches | .138899203 / .132722200 / .190164791 |
| Maximum matched centroid error `ε` | .083187836 |
| Declared converged fit error plus target rounding `δ` | .0000306874 |
| Maximum comparison tolerance `τ` | 1.4211e-14 |
| Matching margin at `R=.190164791` | 1.249588996 |
| Moving-row bound from (3), using `η=0` | .083338173 |
| Actual maximum accepted row radius | .083187817 |
| Fit outcomes / selected proposals | 44 converged / 44 joint fits |

All 44 pre/post clouds cover all eight true centers, all matching margins
are positive, and the MM target differs from its matched sample centroid
by at most `6.3e-16`. The declared numerical fit threshold, not just its
smaller observed residual, was used in the bound. Target rounding is
included; no verified bound for every double-precision objective operation
is claimed. These are retrospective margins on these 44 receipts, not a
proved future bound on sample groups or optimization.

For the host's 2-D isotropic output noise `σ=.029` and HQ radius `.21`, a
clean radius `r<.21` gives the conditional population lower bound

```
P(HQ) ≥ 1 − exp(−(.21−r)²/(2·.029²)).
```

At the moving-row bound above this is `.999927952`. It is a population
probability bound using the triangle inequality, not a guarantee that
every finite 4096-draw evaluation passes. Likewise finite Gaussian real
batches have no deterministic all-time centroid-error bound, and MST group
recovery has not been established for every future batch. A finite horizon
could combine explicit centroid/group-recovery and evaluation tail bounds;
no such all-time probability claim follows from the observed 44 passes.
