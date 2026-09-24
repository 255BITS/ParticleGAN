# Bidirectional sampled-data correction: warm pass, cold failure

The new rule passes all **200 warm checks** and the unchanged cold trajectory
test, then fails the cold ring: **7 modes/HQ .87012, 3/24 passing checks**.
It does not qualify for continuation or production. The original PR84
smoothed-only adapter remains the selected partial candidate because the new
objective adds complexity without clearing another required gate.

This is a completed, source-frozen experiment, not a bandwidth, coefficient,
learning-rate or seed sweep. Rates stay G/D .00425 and prior .0085. The
conditional path is unchanged PR84, so its trajectory pass does not validate
the new unconditional correction on conditional data.

## Why this rule was tested

The [one-sided projection](coverage-projection-report.md) fails three dense
warm checks. [Exact stage attribution](coverage-failure-diagnosis.md) shows
that empty cells strand generated particles and a single assigned real
outlier can set a harmful centroid. The next rule adds a generated-to-real
term so every generated particle receives a signal from the sampled data.

For B real samples and N generated particles, use the two unit-mean terms:

`C = mean_real min_particle squared_distance`

`Q = mean_particle min_real squared_distance`

The objective is `S = C + Q`, with both coefficients exactly one. Holding
both nearest-neighbor assignments fixed, the exact quadratic target is

`t_j = (sum_assigned_real/B + nearest_real_j/N) / (n_j/B + 1/N)`.

The same frozen-generator, updated-Adam-metric pseudoinverse maps this target
to a prior correction. The full correction or at most eight halvings must
strictly decrease actual nonlinear S; otherwise the prior is restored. This
explicitly adds an objective and a projection outside PR84's G curvature
bound. The scalar prior LR cancels from the pseudoinverse projection. There
is no target-center lookup, elapsed-time gain, new training draw or zero pull.

The [three-state filter](chamfer-state-filter-report.md) reproduces the exact
original 4,096 evaluation indices and output noise. It improves the three
failed warm states from HQ .84619/.89697/.89185 to 1/.97461/.97314, keeping
all eight modes. That justified the fresh warm run; it did not prove cold
acquisition or long-term stability.

## Full gate results

| Gate | Result |
| --- | --- |
| Scheduled identity | Exact full-state cold-control parity |
| Correction-disabled active PR84 | Exact archived warm state, final state, observations and diagnostics |
| Ordinary constant-rate warm control | 6/200 passing, reproducing the existing failure |
| Chamfer warm continuation | **200/200**, minimum 8 modes/HQ .921875; final 8/.934570 |
| Cold trajectory, 400 updates | **PASS**, MSE .000942662, passing suffix 18 |
| Cold ring, 1,200 updates | **FAIL**, final 7/.870117; 3/24 passing, suffix 0 |

The three passing cold checks are updates 150, 700 and 1150. The terminal
window is deliberately retained in full:

| Update | Live modes | Live HQ | Modes/HQ if the frozen target were reached exactly |
| --- | ---: | ---: | --- |
| 1000 | 3 | .41846 | 4 / .72559 |
| 1050 | 5 | .65796 | 5 / .74243 |
| 1100 | 5 | .37256 | 6 / .90234 |
| 1150 | 8 | .99927 | 8 / .99268 |
| 1200 | 7 | .87012 | 8 / .95337 |

The ideal-target column uses the same diagnostic noise and is an attribution
counterfactual, not an observed training result or replacement gate. The
cold trajectory took 4.00 seconds and ring 35.08 seconds on one CPU thread.
No longer hold or other acquisition host was run after the failure.

## Remaining issue and cheapest next filter

Every cold correction is accepted and decreases S: 1,165 full steps, 29
half-steps, four quarter-steps and two eighth-steps. All row Jacobians retain
rank two. Thus this failure is not rejected-search starvation or exact rest.

Cold pullbacks differ sharply from the warm ones in this parameterization:
median latent movement is 189.92 versus .04104, and the median largest
nonlinear target error is .8820 versus .00298. The first correction already
moves the prior by norm 380.19 and leaves a maximum target error of 3.868.
At update 1200, one actual particle misses its target by .34285 and the
observed ring misses mode 2. Decreasing the aggregate objective permits this
imperfect landing. Acceptance compares the correction with the **post-GAN**
state; it does not constrain the entire preceding GAN-plus-projection update.
These measurements motivate a nonlinear step/scale investigation, rather
than claiming the metric pseudoinverse is globally accurate.

A cheap next test is to capture exact cold states at updates 750 and 1200
and refine the *same frozen output target* with an iterative nonlinear solve,
then score the original evaluation draws. Their ideal targets pass while
their actual outputs fail. However, perfect target landing still misses
modes at 1000, 1050 and 1100. Better numerical solving alone cannot repair
those already-formed targets. A new policy must also preserve early
acquisition; another full training run should wait for that isolated evidence.
No refinement arm has been run or implicitly promoted here.

## Validation and reproduction

Eight helper tests and three adapter tests pass in a separate 2.30-second
invocation, following the integrated 230-test research suite. The helper
checks exact target weighting, empty cells, rest, rank deficiency, restoration
and nonlinear acceptance. Adapter checks cover exact disabled parity,
conditional skip, current D minibatch replay, three field evaluations with
one Adam moment update, no extra RNG, and correction before EMA. Independent
Sol and Astra reviews found no source, timing or objective-label defect.

The [host evidence manifest](continuous-evidence/chamfer-projection-round4/manifest.json)
archives 45 exact source/declaration/result/log/test/analysis files. The earlier
state-filter manifest separately preserves its failed input-key-schema
attempt and corrected read-only driver; no candidate formula changed.
The [read-only receipt analysis](chamfer_gate_analysis.py) reproduces the warm/cold
statistics and ideal-target counterfactual table from the archived outputs.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
/tmp/pr38-default-env/bin/python -u reports/toy100/chamfer_projection_probe.py \
  --phase warm --output NEW_WARM > warm.log 2>&1
# Only after every warm check and both exact controls pass:
/tmp/pr38-default-env/bin/python -u reports/toy100/chamfer_projection_probe.py \
  --phase cold --previous NEW_WARM/summary.json --output NEW_COLD > cold.log 2>&1
```

The [research notes](coverage-projection-report.md#research-checked-after-the-failure)
separate recent Chamfer, weighted MMD and KDE-drift results from this empirical
GAN experiment. None provides a convergence guarantee for this finite,
uniformly weighted, nonlinear host.
