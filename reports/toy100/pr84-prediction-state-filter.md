# Fixed-step opponent prediction: mixed saved-state evidence

The fixed coefficient-one extrapolation corrects the first captured excursion,
but **does not pass all three saved-state continuation filters**. The later
mode-loss episode remains, and another branch creates a new terminal failure.
These are local diagnostics, not cold acquisition or long-hold qualification.

The candidate is `pr84_opponent_prediction.py` at commit `6757ccf`. D takes its
unchanged accepted step from `D0` to `D*`; only G evaluates the temporary critic
`Dprobe = D* + (D* - D0)`. Every G callback restores actual D immediately. The
same predicted critic and detached stencil width serve G's base and proposal
queries. Disabled prediction exactly preserves original PR84, including its
width recomputation. The active width freeze is an explicit part of the cold
operator, although all saved warm widths are already at the `.15` cap.

This is related to established prediction methods, not a novelty claim.
[Alex-GDA](https://proceedings.mlr.press/v235/lee24e.html) studies alternating
gradients at extrapolated iterates and bilinear convergence; the earlier
[prediction-method project](https://www.cs.umd.edu/~tomg/projects/stable_gans/)
also applies prediction to adversarial networks. Their guarantees do not
certify this nonlinear, stochastic, Adam-preconditioned GAN.

For the fixed-metric scalar bilinear game, D minimizes `d*g` and G minimizes
`-d*g`. With positive steps `a,b`, the update matrix is
`[[1,-a],[b,1-2ab]]`. When `0 < ab < 1`, its eigenvalues have modulus
`sqrt(1-ab)`, whereas ordinary alternating updates have determinant one.
The implementation test verifies this sign and keeps an exact zero field
stationary. No additional parameter pull, target read, gain search, time decay,
or extra game evaluation is introduced.

## Frozen replay and observed outcomes

The input is validator capture-v2 from the unchanged 2,400-step PR84
continuation, with full saved-state file SHA256
`37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47`.
The filter changes only the outer loop start/end and restores the archived
pre-step models, both Adam states, EMA, noise scalars and all RNG streams.
Every one of the 13 ordinary one-update replays matches the entire archived
accepted-state hash exactly. All 44 ordinary short-continuation support arrays
also match exactly. Candidate branches finish with identical RNG states to
their paired ordinary branches. Nominal rates remain G/D `.00425`, prior
`.0085`, noise horizon `1200`, and seed `0`. Each player makes three existing
gradient evaluations and exactly one Adam moment update per outer update.

The declaration fixes three branch starts before results are consumed:

| Resumed updates | Original passing checks | Prediction passing checks | Original minimum HQ | Prediction minimum HQ |
| --- | ---: | ---: | ---: | ---: |
| 1324–1335 | 9/12 | 12/12 | .775391 | .928467 |
| 1380–1395 | 14/16 | 15/16 | .878662 | .863281 |
| 1530–1545 | 1/16 | 1/16 | .788574 | .824463 |

The first prediction branch passes every check. The second prevents the
original 1389–1390 failures but introduces a new failure at 1395. The third
still loses a mode at 1533 and remains at seven modes through 1545. No branch
has been removed or declared a global pass.

Single-update prediction improves HQ at 1325 from `.824219` to `.904053`,
at 1389 from `.892578` to `.907227`, and at 1390 from `.878662` to `.929443`.
It cannot restore the already missing mode at 1539–1541; 1530 and 1570 are
slightly worse than their ordinary counterparts. The old-opponent comparator
is diagnostic only and has no candidate eligibility. Its nested recorder
labels inherit the prediction scaffold; the top-level `opponent="old"`
and declaration explicitly identify the executed override `Dprobe=D0`.

## Directional attribution

Target centers appear only in this offline explanation, never in the update.
At 1324 both G factors equal one: prediction moves outputs 6.0% farther in RMS,
but changes summed squared distance to each base point's nearest center by
`-.00313` instead of `+.02747`; its output-motion cosine with ordinary is
`.642`. At 1326 prediction has factor one versus ordinary `.496` and moves
36.1% farther, while that distance change switches from `+.03317` to
`-.00962` (cosine `.467`). At 1389 its motion is 3.9% larger and the radial
first-order work switches from `+.00814` to `-.00090` (cosine `.578`).
Thus some benefits reflect a direction change, rather than only smaller
steps. At 1325 itself motion remains closely aligned (cosine `.989`), and the
own-curvature factor falls from one to `.554`, so that improvement is largely
damping. At 1530/1540 directions remain about `.97` aligned and the severe
mode-loss family persists.

This supports one discriminating earlier-intervention warm test, if declared
separately; it does not turn the failed saved branches into a pass. The root
agent owns that decision and any full host run. No cold or full warm training
was run by this filter.

## Evidence and verification

The raw declaration, all 13 one-update comparisons, all three continuations,
summary and exact executed sources are archived under
[`continuous-evidence/pr84-opponent-prediction-filter`](continuous-evidence/pr84-opponent-prediction-filter/manifest.json).
Input states remain in the separately committed exact failure-capture evidence.
The filter's `NoisePolicy` counters describe the local replay, with the saved
step clock restored; they are not a reconstructed prefix draw history.
The complete optimizer tensors and RNG are restored and hash-checked.

Eight focused adapter tests pass, covering bilinear signs, exact zero-field
rest, one-moment accounting, actual critic query order, disabled full-host
parity on both hosts, active noise/RNG behavior, fixed G operator, and
exception restoration. The replay is run with torch 2.13, one CPU thread and
the declared AVX2 environment.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_prediction_state_filter.py \
  --capture /path/to/stationary-failure-replay/capture-v2 \
  --output /tmp/new-prediction-state-filter
```
