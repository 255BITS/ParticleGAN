# Cold472 consecutive profiled-value diagnostic

Six consecutive own-state moves have certified decreases of one fixed
finite-bank value, but all twelve particles remain allocated four each to
the same three modes. Attempt7 restores the exact preceding state after
all nine trials fail to separate the numerical value bounds. This is a
bounded acquisition diagnostic, not native training or a convergence proof.

The first move is the [previously archived independent472 assay](pr84-convex-profiled-value-independent.md).
The continuation starts at its exact accepted G/prior/readout endpoint.
The sixteen D banks, real/fake pairing, particle indices, output-noise draws,
critic nonlinear features, scalar bias and post-Adam G metric remain fixed.
Each accepted fitted readout becomes the next base point; no base solve,
new moment, host-clock update, resampling or return to the original
trajectory occurs. The maximum was eight moves total, including the first.

Every trial uses the original one-attempt100-iteration /200-closure solver
and at most nine fixed halvings. Acceptance is still
`new D-loss lower bound > previous evaluated upper bound + Armijo decrease`.
The loss is the original sharp Rp logistic D loss and cap in a frozen
96-dimensional readout class. The G partial derivative includes the
fake-input cap derivative. No target label or quality grade affects the
direction, acceptance or stopping.

| Move | Accepted scale | D-loss lower / upper | Paired HQ | Clean output RMS |
| --- | ---: | ---: | ---: | ---: |
|Before|—|.23301099 /.23438020|.752441|—|
|1|1/4|.24633963 /.24975213|.968018|.102189|
|2|1/64|.25006807 /.25021862|.968018|.004383|
|3|1/32|.25083776 /.25106066|.966309|.008182|
|4|1/16|.25119680 /.25208255|.957520|.013557|
|5|1/64|.25210414 /.25217770|.955322|.001591|
|6|1/32|.25229683 /.25233193|.950684|.002474|
|7|Rest|unchanged|.950684|0|

All quality observations use the **same** original4724096-draw evaluation;
these are diagnostic move indices, not new host update numbers. Every
state has three covered modes. The nearest-center allocation remains
`[4,4,0,0,0,0,0,4]`. Mean nearest-center distance falls from`.169682` to
`.109759`, while the final worst particle is`.211688` from its nearest
center. These center-based measurements are post-hoc only. HQ falls modestly
after the first improvement even while every accepted value decreases.

The summed per-move certified value decrease is at least`.01317136`.
Comparing only the final lower D-loss bound with the original upper bound
gives the stronger endpoint value decrease`.01791663`. Intermediate fitting
gaps make the sum conservative. The G raw gradient norm declines from
`.584826` at move2 to`.130810` at attempted7; metric predicted work declines
from`.0329102` to`.00382893`. Neither quantity is zero.

The final rest is **not certified stationarity**. At the smallest trial
scale1/256, evaluated D loss rises by`1.4791e-5`, but its remaining inner
gap is`3.4302e-5`. Its lower bound therefore misses the old upper bound by
`1.9511e-5`. This may hide a small real value improvement. All40 new trial
fits retain `NOT_CERTIFIED` status against the strict`1e-7` gap target.
The declared budget is not enlarged and no tolerance is relaxed. The run
shows that this bounded local rule has not acquired missing modes; it does
not establish that an exact local value flow cannot eventually leave the
subset or that a global allocation change is mathematically necessary.

Twelve finite-difference checks across the six attempted new moves pass,
with maximum relative error`1.51e-6` and zero recorded LeakyReLU/cap switches.
The final rest restores G, prior, readout, fit receipt and support bitwise.
Input payload, caller RNG, features, data and metric are unchanged. There
are4350 new cached readout-gradient closures and40 final read-only gradients;
the previously executed first move used433 closures and4 final gradients.
Cache construction and G/derivative evaluations add further work. These
counts are not native GAN update equivalents.

Three new tests check consecutive scalar-game bound accumulation, exact
restoration when a valid loose bound cannot certify any of nine trials,
and exact zero-field rest without trial calls. Together with the original
six convex-readout tests,9/9 pass. The [archive](continuous-evidence/convex-profiled-cold-continuation/manifest.json)
preserves the frozen source, declaration, result, log and complete own
states. No full native branch follows from this diagnostic alone.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_convex_profiled_cold_continuation.py \
  --output NEW_CONTINUATION_OUTPUT
```
