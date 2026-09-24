# PR88 exact-head cold acquisition

PR88 reaches eight modes by update50, but the pinned cold run does not
sustain the original quality gate. It passes14/24 checkpoints, with a longest
passing run of three, and finishes at **8 modes /HQ`.880371`**. The strict
cold verdict is FAIL with suffix0. This is more rapid support acquisition
than PR84, which finishes at7 modes /HQ1.0 with0/24 passing checkpoints,
but it is not stable acquisition.

The exact submitted head is
`7d7676c37ebd390836aa957ecb085367f1e88a35` of
[PR88](https://github.com/255BITS/ParticleGAN/pull/88). No algorithm source,
assignment-round count, GN budget, coefficient, threshold or seed is changed.
The pinned environment is torch`2.13.0+cu126` on CPU, one thread, AVX2. The
author reported the same version string while already noting different
warm-state hashes. Its reported cold pass does not reproduce here.

| Same-process cold run | Final result | Checkpoints | Wall time |
| --- | --- | ---: | ---: |
|PR84 trajectory400|MSE`.000942662`, PASS|18/24; suffix18|4.28s|
|PR88 trajectory400|Bitwise identical to PR84|18/24; suffix18|3.52s|
|PR84 ring1200|7 modes /HQ1.0|0/24|32.85s|
|PR88 ring, correction disabled|Bitwise identical to PR84|0/24|33.19s|
|PR88 ring, correction active|8 modes /HQ`.880371`|14/24; suffix0|45.79s|

The ring overhead is about1.39× in this paired run. This is a measured
single-run comparison, not a confidence interval. The three submitted
pullback tests pass; routine torch deprecation warnings do not affect them.

## Chronology and landing accuracy

Coverage losses recur after early acquisition: update100 has2 modes;
300 and950 have4;250,500,650,800,900 and1000 have6. Several intervening
checkpoints recover all8. The terminal sequence is:

| Update | Modes | HQ | Actual frozen-target fitting error |
| ---: | ---: | ---: | ---: |
|1000|6|.847168|3.81e-6|
|1050|8|.997559|3.53e-6|
|1100|8|.984375|5.01e-6|
|1150|8|.971436|6.10e-6|
|1200|8|.880371|3.37e-6|

The original four-refresh Chamfer target is landed accurately at these
terminal points. At1200 the actual C+Q objective falls from`1.454346` to
`.009347`, yet noisy HQ is below`.9`. These particular misses are therefore
not gross nonlinear landing failures. Earlier fit errors are less uniform:
97 updates have target error above`1e-4`, with maximum`6.75137`; maximum
accepted latent displacement norm is`3991.69`. No universal causal claim is
made from the endpoint alone.

The correction is accepted on1199/1200 updates and evaluates5026 batched
prior-output Jacobians (counted from the target-error trace before each
actual Jacobian call). The unchanged native game also uses3600 full gradient
blocks per player, with only1200 Adam moment steps. GN forward trials add
work beyond those counts. The added sampled C+Q objective and prior-only
parameter correction remain explicit; this is not an Rp optimizer-only rule.

## Exact controls and preserved state

Both correction-disabled host controls match PR84 in the complete saved
model/prior/Adam/EMA/RNG/noise-policy state, all metrics excluding time,
every native update record, rates and generated host source. Active PR88
trajectory is also bitwise identical. Active ring changes the model but
retains exactly the baseline's final global/data RNG and complete noise
policy, including streams and counters. All final model tensors are finite.

Nominal D/G rates are`.00425`, prior`.0085`; every recorded schedule multiplier
is1. Each host retains its original400/1200 noise horizon and step calls.
No compatibility patch is needed. The exact checkout is verified clean
before and after execution. Complete final states are saved after the host's
final evaluation, including all noise-policy variables and optimizer state.

The strict cold failure is preserved. Under the revised practical rubric,
the separately authorized next test is an unchanged continuation from this
specific failing endpoint through2400, after exact restart parity is checked.
It is a **recovery continuation**, not a qualified acquired-state hold; its
starting grade is explicitly8/HQ`.880371`. That continuation has not been
run as part of this cold result.

The [manifest](continuous-evidence/pr88-cold-independent-audit/manifest.json)
binds the full24-point traces, exact source and generated host, complete final
states, report metadata, declaration and log. The reported external cold
terminal sequence remains archived separately for transparent comparison.

```bash
git worktree add --detach PR88_CHECKOUT 7d7676c37ebd390836aa957ecb085367f1e88a35
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr88_cold_audit.py \
  --checkout PR88_CHECKOUT --output NEW_OUTPUT
```
