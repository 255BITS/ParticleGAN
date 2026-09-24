# PR #84 smoothed-only scratch candidate on PR #82

`pr84_smoothed_candidate.py` extracts only the original PR #84 `40ecb67`
five-point G-critic stencil and places it on the canonical PR #82 alternating
adapter (`alternating_curvature_scratch.py`, SHA-256 `cb6278251cf04d321afe7109f5eb91b3a5ebb41a866abb03c9a694b8c9969450`).
The constants are fixed: G own-curvature bound .25, D bound 3, stencil width
`min(.15, .5 / critic sharpness)`. D sees its ordinary sharp critic. There is
no target-center, mode-count, elapsed-time boost, ratio, latent-nudge, or
rest-gate controller in this file. The method is scratch-scoped to the
deterministic `mode_hold` and `trajectory` hosts; it is not a production
optimizer or a passing shared-gate candidate.

This cleanup is **exactly equivalent** to the frozen `40ecb67` source on the
complete cold trajectory (400 updates) and ring (1200 updates), in the local
CPU/AVX2 environment. The parity runner compares final G/D/prior weights,
Adam moments, ring EMA, global/data/input/output RNG state, all host metrics
and observations apart from wall time, the applied-rate and noise receipts,
every own-curvature and stencil-width record, and the generated host source
hash. Both replayed hosts match exactly. Trajectory passes with MSE
.000942662 and an 18-check suffix; the ring fails with seven modes and HQ 1.0
at update 1200. Every terminal ring check has seven modes, so no continuation
of an acquired cold state or further host is qualified.

A later, separately declared [conditional warm hold](stationary-stability-status.md)
tests the user's same-dataset stability concern despite that acquisition
failure. It reproduces warm200/200, then fails8 of120 checks through2400;
minimum7 modes/HQ .78857, first observed failure1390. Final8/HQ .99805 hides
the interruptions. The candidate is therefore rejected for both acquisition
and longer conditional stability.

The original `40ec` source and parity result are stored under
[continuous-evidence/pr84-smoothed-candidate](continuous-evidence/pr84-smoothed-candidate/manifest.json)
with SHA-256 hashes. The separate independent audit commit `363fc77`
archives the four warm/cold runs and explains the archived PR #84 result's
different warm/ring outcomes. The original source read target centers every
step for inactive diagnostic/mode-latch code; removing those paths did not
alter the tested trajectory or ring training state.

To rerun parity, from this tree:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 DNNL_MAX_CPU_ISA=AVX2
/tmp/pr38-default-env/bin/python -u reports/toy100/pr84_smoothed_parity.py --output NEW_PARITY.json
```

The focused tests pass (8/8, including PR #82's unchanged adapter tests).
The cold parity run itself is the end-to-end check; its deterministic gzip
receipt is included in the evidence directory. Both host optimizer groups
retain constant base rates (D and G .00425, prior .0085) and one Adam moment
update per outer step; measured own-curvature scales the actual D/G proposals
without a clock-based LR schedule.
