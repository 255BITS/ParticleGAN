# Neural landing of adversarial nonlocal proposals

All three fixed V2 proposals land through copied generator and prior
parameters, decrease the unchanged actual Rp generator loss, and decrease
that loss on each of eight heldout banks. The cold state gains one mode;
both warm controls keep eight. This is numerical realization, not a native
training pass or sustained acquisition.

The [V2 proposal assay](pr84-adversarial-reallocation-assay.md) fixed each
donor and destination using only the original adversarial objective.
Landing starts from those same pre-G parameters, using the frozen joint
output solver: at most20 Gauss–Newton iterations,12 halvings, SVD relative
threshold`1e-6`, and maximum-row error tolerance
`1e-5 * max(1,max target-row norm)`. Target distance is only the numerical
solver's error. Final acceptance requires strict decrease of the native
paired G objective with the same critic, five-point`.15` stencil, real
batch and output-noise draw. Heldout loss and quality do not select or accept.

| State / critic | GN iterations | Maximum target error | G parameter displacement | Prior displacement | Accepted modes / HQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warm1530 / native D | 4 | 1.30e-6 | .583785 | 1.636991 | 8 / .968994 |
| Warm1530 / archived refined D | 5 | 4.78e-7 | 1.118803 | 1.371494 | 8 / .961914 |
| Cold472 / best finite D | 6 | 5.33e-7 | 2.922640 | 1.192207 | 4 / .776123 |

All fits report `CONVERGED`. Actual neural native G losses are bitwise equal
to the selected free-output after losses, respectively`.70675367`,
`.71950531`, and`4.72195005`. All24 heldout comparisons decrease. Quality uses
the same original4096-draw diagnostic per saved step. The cold baseline has
three modes / HQ`.752441`; warm has eight /`.968994`. Four modes is not full
acquisition.

The cloned critic, complete D/G Adam states, persistent buffers, absent
gradient buffers and caller RNG remain unchanged. G Adam is materialized
only to audit state ownership; it never steps. Joint GN uses a Euclidean
metric, not saved Adam. Final G parameter norms are13.248952,13.276029,
11.569758; prior norms are3.924489,3.700190,3.936710. Finite one-step values
do not prove parameter boundedness under repeated moves.

Before landing, the driver independently audits V1→V2 on all three actual
model states. Direct paired `GANLoss.g_loss` calls reproduce before and
after exactly and verify unchanged donor/index/target, after loss, heldout
losses and quality. Invalid V1 baselines remain explicitly recorded. V2's
first direct-file invocation failed to import `reports` before declaration
or evaluation; unchanged source then ran with Python's module entrypoint.
Both logs are retained.

The paired repair audit and three landings take1.48 seconds on pinned
single-thread CPU, PyTorch2.13.0+cu126/AVX2. Five proposal tests plus four
existing joint-solver tests pass,9/9 in2.10 seconds. Full states, fit trials,
source hashes, declaration, audit and logs are in the
[archive](continuous-evidence/gan-nonlocal-landing/manifest.json).

The unresolved issue is the critic's response to repeated nonlocal moves.
A frozen critic can reward concentration at one high-score region. These
moves do not establish that alternating D updates recover fast enough or
that shared-network changes preserve future behavior. A short declared
native alternating continuation is the next test. None ran in this assay.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_adversarial_landing.py \
  --proposal V2_OUTPUT/result.json --output NEW_LANDING_OUTPUT
```
