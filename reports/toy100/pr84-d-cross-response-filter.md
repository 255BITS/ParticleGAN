# Actual-D cross-response: strict local filter failed

**No warm or cold follow-up is eligible.** The one declared method repairs the
1380 branch, but still fails the acute 1325 update and the later mode-loss
episode. All three predeclared branches and every live G checkpoint are retained.

After unchanged alternating PR84 accepts `D*` and `G+`, the method applies

```text
D+ = D* - P_D [F_D(D*, G+) - F_D(D*, G0)]
P_D = constant_role_lr / (sqrt(post-base bias-corrected Adam second moment) + eps)
```

G includes both network and trainable prior. This is one explicit response to
the actual accepted G movement, with coefficient one. The actual critic
changes after G; G's original current-update proposal remains untouched.
There is no implicit solve, gain search, new loss, target access, added clipping
or clock decay. The original curvature bounds cover the ordinary PR84 proposals
only: the extra D response is explicitly unbounded.

This tests a different response channel from G-side opponent prediction, which
leaves actual D unchanged. It also differs from the previous joint simultaneous
cross-only solver and the alternating full-J implicit solve, whose acquisition
failure remains valid. For the scalar bilinear game, its update matrix is
`[[1-ab,-a(1-ab)],[b,1-ab]]`; for `0<ab<1`, the eigenvalues have modulus
`sqrt(1-ab)`. This restricted calculation gives a sign check, not a nonlinear
GAN guarantee. The correction vanishes when G does not move and exact zero
fields remain stationary.

| Predeclared resumed updates | Original passing checks | D response passing checks | Original minimum HQ | D response minimum HQ |
| --- | ---: | ---: | ---: | ---: |
| 1324–1335 | 9/12 | 11/12 | .775391 | .889160 |
| 1380–1395 | 14/16 | 16/16 | .878662 | .990967 |
| 1530–1545 | 1/16 | 1/16 | .788574 | .822021 |

At 1325 the corrected branch still falls below the unchanged .90 HQ threshold.
In the severe branch both methods lose mode 2 at 1533 and remain at seven
modes through 1545. The declared eligibility condition is every check in all
three branches passing. Consequently the overall status is **FAIL**, with
`warm_eligible=False`; no earlier-intervention exception or full host run was
used for this method.

No nonfinite value, clipping fallback or critic explosion occurred. D correction
parameter norms ranged from `.00610` to `.05414`, and were `.0518` to `.5253`
times the corresponding ordinary D displacement norm. Each of the 44 active
updates had an inactive original D bound and required one extra D-gradient
query. The totals are 176 D fields, 132 G fields, and 44 moment updates/player.
Normal optimizer callbacks remain three/player/update; extra D queries capture
gradients without calling either optimizer. Receipts distinguish those costs.

## Exact paired execution

[`pr84_d_cross_response.py`](pr84_d_cross_response.py) freezes the post-base D
metric. It reuses the phase-1 D field only after a bitwise `D1 == D*` check;
otherwise it evaluates a new base field at the actually accepted D. It always
evaluates the moved field at accepted `G+`, rather than the unbounded G proposal.
The extra source-transformed phases execute only the sharp D block. Each query
restores the original pre-block data/global/input/output RNG and buffers, then
verifies its resulting RNG against the original D-block endpoint. Finalization
restores the full ordinary block's RNG, accepted G, and accepted module buffers.
Each Adam moment advances exactly once.

The ordinary branch exactly reproduces all 44 archived support arrays and
curvature records, plus the complete first accepted-state hash for each branch.
Candidate and ordinary RNG endpoints match exactly. Applied nominal rates are
G/D `.00425`, prior `.0085`; noise horizon stays `1200`; seed stays `0`.
The input is the **full** capture-v2 sidecar, SHA256
`37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47`.
The compact public state subset does not contain 1380 and is insufficient;
the exact capture script can regenerate the full sidecar.

The reusable replay utility's `accepted_states` entries are captured after
phase 2, before the new D correction; the separately reported final training
state includes the correction. This stage distinction is explicit in the
declaration and raw records. Local noise counters are not a reconstructed
history of the skipped prefix.

The frozen adapter SHA256 is
`4a2d6050f6dc9375cdadf3ab73bef7345d8b1999d7701584fbcfe415ce9a8971`.
Nine tests pass, independently rerun: bilinear signs, no-op/zero-field behavior,
D-bound-active exact base queries, moments/query accounting, disabled complete
host parity on both hosts, active isolated-noise RNG preservation, and error
restoration. Independent source review found no remaining scope/order blocker
for these two fixed CPU hosts.

The [evidence manifest](continuous-evidence/pr84-d-cross-response-filter/manifest.json)
archives the declaration, log, all six original/candidate branch records,
summary, and exact source bytes. The run is a saved-state diagnostic only and
cannot qualify a production optimizer.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_d_cross_state_filter.py \
  --capture /path/to/stationary-failure-replay/capture-v2 \
  --output /tmp/new-d-cross-filter
```
