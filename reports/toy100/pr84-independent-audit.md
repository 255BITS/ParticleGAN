# Independent PR #84 audit (fixed-target continuous learning)

PR #84 does not yet supply a qualifying constant-rate replacement. I replayed the
smoothed-critic precursor `40ecb675bcb4b33bb0ab93e08e63607d30044930` and
the hard-rest head `468ad2619a0536f8c69ef878b86bc514d296338a`, each from
its exact source tree. Both pass the warm 200-update fork and cold trajectory,
then fail the unchanged cold ring acquisition gate. No 2400-update continuation
or further host was run. This is a local CPU reproduction, not a claim that the
earlier archived measurements were fabricated: PR #84 itself notes its traces
are not bit-reproducible across VMs, and it does not commit the raw smoothed
warm/cold runs reported at `40ecb67`.

| Exact source | Warm, candidate | Cold trajectory | Cold ring at update 1200 | Verdict |
| --- | --- | --- | --- | --- |
| `40ecb67`, G sees five-point smoothed critic | 200/200, minimum HQ .9702, final 8/1 | PASS, MSE .000942662, 18-check suffix | FAIL, 7 modes, HQ 1.0, 0/24 passing checks | Stop |
| `468ad26`, same smoothing plus hard-rest gate | 200/200, minimum HQ .9941, final 8/1 | PASS, same MSE .000942662 and suffix | FAIL, 7 modes, HQ .994385, 0/24 passing checks | Stop |

The exact `40ec` cold ring has seven modes at all five terminal checks:
HQ .8840, .9731, .9995, 1.0, 1.0 at updates 1000–1200. Mode 6 is missing
throughout and no update reaches eight modes. The `468` hard-rest run has
3/6/7/7/7 modes and HQ .470/.681/1/.9785/.9944 at those checks; mode 1 is
missing at the end. Smoothing width was .15 at all cold-ring updates. The
unchanged trajectory host uses non-2D critic input, so the smoothing path is
inactive there. The 40ec report had instead recorded a cold-ring pass with
eight modes and HQ .995/.988/.988/.996/.999, but its warm fork failed 196/200.
The local result reverses which side of the screen misses; neither complete
evidence set passes both.

The local warm controls show identity versus uninterrupted cold state-hash
parity, with the same prefix hash for both source versions. The plain constant
Adam control passes 6/200 checks. Cold applied rates are constant for every
optimizer group: D and G .00425, prior .0085, with `lr_floor=1` and
`lr_anneal_start=0`. The adapter still scales individual D/G **parameter
proposals** by measured own-curvature, so this is a constant *base* rate rather
than a fixed effective step. Every cold outer update has three gradient
evaluations and one Adam moment update per player; two RNG replays per step are
verified. These are 2D `mode_hold` and trajectory scratch results, not a
22-host or production-policy claim.

## Code audit

The `40ec` mechanism does not use the ring target centers for an update under
these flags. `_occupied_modes()` nevertheless computes exact target-centered
coverage every step for a diagnostic/latch field; `mode_loosen=None`,
`boost_steps=0`, and the other optional mode/clock controllers are inactive.
Smoothing is armed after D's ordinary step and applied only to G's critic
forward. It changes G's game field, not merely its optimizer. The critic's
own loss remains sharp. The phase replay preserves D-then-G alternation and
verifies RNG consumption.

The `468` hard-rest slope is **twice stenciled**. In
`_scale_bounded_g_step`, `score()` manually averages five `module(...)`
calls, while `SimpleMLPDiscriminator.forward` has already been patched to
five-point smoothing when `_smooth_on=True`. It therefore measures the slope
of a 25-call convolution, rather than the once-smoothed critic used for G's
gradient. This issue does not apply to `40ec`, which has no rest gate. The
rest threshold is fixed, so it is not a hidden clock, but the measured signal
and the description do not match.

Both probe declarations record `curvature_bound`, `bound_d`, and
`d_curvature_bound` but omit `smooth_critic`, `smooth_cap`, and other optional
method flags. The source hashes plus the explicit commands below identify my
runs; the JSON declarations alone are insufficient. This is a receipt gap,
not evidence of a different method in these runs.

## Reproduction and evidence

Python 3.12.13, PyTorch 2.13.0+cu126, CPU with one thread and AVX2. From each
exact commit's worktree, run warm first and consume its result before cold:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
/tmp/pr38-default-env/bin/python -u reports/toy100/alternating_curvature_warm_probe.py --curvature-bound .25 --bound-d --d-curvature-bound 3 --smooth-critic --output NEW_WARM_PREFIX
/tmp/pr38-default-env/bin/python -u reports/toy100/alternating_curvature_probe.py --curvature-bound .25 --bound-d --d-curvature-bound 3 --smooth-critic --output NEW_COLD_DIRECTORY
```

The [evidence directory](continuous-evidence/pr84-independent-audit/manifest.json)
contains deterministic gzip copies of all four warm/cold raw runs, logs,
declarations, source files from both commits, and SHA-256 hashes of both the
stored and decompressed bytes. The archived JSON receipts support all local
metrics and update/accounting claims above. No seed or threshold grid was run.
