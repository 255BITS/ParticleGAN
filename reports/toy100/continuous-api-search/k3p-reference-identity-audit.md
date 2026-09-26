# K3P reference identity audit

The prepared **matched-eager K3P** reference changes public K3P behavior and cannot be the sole evidence for competing with ordinary public K3P. Archived public factories at `25812851` return optimizers directly; the prepared adapter adds eager CUDA scalar states. K3P's spike guard itself computes `1-beta2**tensor_step`, so counter placement can affect its arithmetic. Eager state also changes first-update guard/A2 state visibility.

The pinned `/tmp/pr38-default-env` runtime is Torch 2.13.0+cu126; its `optim/adam.py:166–175` deliberately puts ordinary nonfused/noncapturable scalar steps on CPU. Keep those counters for the public reference. Relax the prepared worker's all-state-on-CUDA assertion accordingly; model/gradient/moment tensors still belong on CUDA.

Exact public source exists locally at `v0.8.0` (`0ff9a7af`). The `25812851` factory source differs only through an optional unused network-transition API. Current retained K3P code additionally changes floating EMA buffers from mul/add to lerp, so an archived package imported under the same pinned runtime gives the strongest public identity. Historical public ring scores used Torch 2.14.0+cu130 and do not establish a matched-runtime result.

Prepared fixed 360/720 noise milestones are another explicit configuration override; literal public default with budget N uses .1N/.2N. Preserve and label the baseline's own declared schedules. Match model initialization, data, evaluation and runtime; do not force the baseline to share RP2-specific optimizer changes.

**NOT_RUN:** RP2 failed its first broader gate. This audit recommends no additional run now and executed no models or tests.
