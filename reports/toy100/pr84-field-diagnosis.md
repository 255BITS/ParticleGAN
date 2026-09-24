# Frozen PR84 critic-field diagnosis

This is a read-only diagnosis of the precursor commit `40ecb675bcb4b33bb0ab93e08e63607d30044930` (G-only five-point critic stencil, G own-curvature cap 0.25, D cap 3). The exact-source cold rerun passed trajectory but failed the strict eight-mode ring gate: terminal mode counts were 7/7/7/7/7 at updates 1000/1050/1100/1150/1200. The missing mode was 6 at `(0, -3)` throughout. This differs from PR84's reported ring pass, and the archived claim has no run receipt to reconcile the difference. The fresh warm rerun also passed 200/200, unlike the reported 196/200.

The [diagnostic script](pr84_field_diagnosis.py) replayed all 1,200 cold updates against the frozen audit and inspected only the final state. Its entire untimed host result and all 1,200 optimizer records matched the audit exactly. The [field receipt](continuous-evidence/pr84-field/smooth40-field.json) and [compressed cold audit](continuous-evidence/pr84-field/audit/manifest.json) preserve the measurements and source hashes.

| Final-state quantity | Result |
| --- | --- |
| Missing-center critic logit | Sharp `1.852`; five-point smooth `1.796` (the largest mode-center logit in both fields) |
| Observed fake critic logit range | Sharp `[-0.963, 0.094]`; smooth `[-0.976, 0.086]` |
| Two closest particles' input-gradient projection toward missing center | Sharp `-0.614/-0.579`; smooth `-0.321/-0.368` |
| Their ordinary Adam proposal projection toward missing center | `-0.959/-0.449`; after curvature cap `-0.093/-0.042` |
| G network versus prior contribution for closest particle | Network proposal `-0.921` toward missing center; prior-only `-0.041` |
| Same-batch G curvature factor | `0.09559` |

The critic distinguishes the unoccupied real region, but its *local* gradient at nearby generated particles points away from it. Along the direct path from the closest particle to the missing center, the smoothed critic rises from `-0.976` to `1.796`, while its derivative at the starting particle is `-0.321` toward the center and becomes positive farther along the path. This is a local gradient barrier. The actual full G proposal follows the wrong local direction, and the final curvature cap shrinks it without changing its sign. The large network-only displacement explains most of the move; the prior contributes little. The critic derivatives here are noise-free, clean-particle probes with final D, while the host's G proposal uses its sampled noisy batch. They corroborate the proposal direction rather than decompose its exact batch gradient. These observations support testing a data-derived nonlocal acquisition signal before trying another local step-size controller. They do not establish that such a signal will preserve warm stability or cold trajectory.

Reproduce with the matching PyTorch 2.13 environment and source worktree checked out at `40ecb675`:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2 \
  /tmp/pr38-default-env/bin/python reports/toy100/pr84_field_diagnosis.py \
  --source-root /ml2/hypergan/ParticleGAN-pr84-smooth-audit \
  --audit /tmp/pr84-audit-20260924/smooth40-cold/mode_hold.json \
  --output /tmp/pr84-field-replay.json
```

The audit JSON is stored in `continuous-evidence/pr84-field/audit/mode_hold.json.gz` for a portable replay; decompress it to pass as `--audit`.

## Fixed-cloud coverage proxy

The [proxy script](pr84_coverage_proxy.py) uses the audit's cold seven-mode and passing warm eight-mode **single noisy draw per particle** (`support_scope` in the host receipt). It draws one 256-example real minibatch, matching the transfer recipe's resolved batch size, then applies 20 exact nearest-real centroid (Lloyd) updates in **output coordinates only**. Each real point is assigned to its nearest generated support point; empty cells stay fixed. The mode centers are used solely to score the resulting clouds.

| Cloud | Initial | After 1 centroid update | After 2 | After 20 |
| --- | --- | --- | --- | --- |
| Cold | 7 modes, HQ 1.0, coverage loss .52549 | 7, .8333, .07230 | 8, .9167, .00923 | 8, .9167, .00909 |
| Warm | 8 modes, HQ 1.0, coverage loss .01279 | 8, 1.0, .00895 | 8, 1.0, .00879 | 8, 1.0, .00878 |

At the cold cloud the centroid directions for two particles project `+1.436` and `+0.532` toward the missing mode; the critic-guided proposal above projects away from it. This supports a one-sided coverage correction as a **candidate**, not a result in parameter space. The transient cold HQ dip at the first centroid step is real. The [full proxy receipt](continuous-evidence/pr84-field/coverage-proxy.json) fixes the single sampled minibatch and all 21 measurements; no seeds, widths, or gains were searched. The proxy starts from a noisy observation and does not account for generator Jacobian coupling, further noise, Adam moments, or interaction with the GAN update.
