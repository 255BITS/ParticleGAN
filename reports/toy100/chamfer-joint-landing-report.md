# Generic C+Q reallocation can land on three saved states

The fixed-bank numerical filter **passes its declared saved-state gate**.
The copied G and prior reach every reallocated Chamfer target, the actual
sampled C+Q objective decreases at every round, and all eight fixed-eval
rounds pass eight modes/HQ≥.9 for cold update 1, cold update 100, and a warm
passing state. This is **not** a GAN training run or a continual-stability
result. No critic, optimizer state, EMA, learning rate, noise schedule, or
training RNG was advanced.

The [driver](chamfer_joint_landing_filter.py) uses the exact saved cold
`pre_step` at update 1, cold `post_bounded_g` at update 100, and warm
`pre_step` at update 1324 from the tracked compact replay archive. Its
three case JSON files are byte-for-byte identical to a run using the original
full 37 MB warm-state sidecar; both source archive hashes are recorded.
Each case draws one native 128-real-sample bank
from its saved data stream and reuses that bank for eight rounds. At each
round the frozen [data-only rule](continuous-evidence/chamfer-joint-landing/source/reports/toy100/chamfer_discrete_reallocation.py.gz)
chooses up to 12 strictly improving replacements of individual free output
points by observed real rows for
`C=mean_real min_particle ||x−y||²` plus
`Q=mean_particle min_real ||y−x||²`. One fixed-nearest-assignment quadratic
minimizer then supplies the output target. The joint G+prior Gauss–Newton
helper fits that target with bounded nonlinear acceptance. Target ring means
appear only in the unchanged data sampler and posthoc quality scoring; they
do not choose a relocation, target, or neural update.

| Saved state | First-round C+Q before→after actual neural landing | First-round relocations / distinct donors | First→last HQ; minimum of eight | Neural landing |
| --- | ---: | ---: | ---: | --- |
| Cold 1 | 16.66633→0.008276 | 12 / 12 | .95313→.91211; min .91138 | 8/8 targets |
| Cold 100 | 11.05857→0.007802 | 12 / 12 | .99634→1; min .99634 | 8/8 targets |
| Warm 1324 | .015222→.006750 | 12 / 10 | .99927→1; min .99927 | 8/8 targets |

All 24 targets converge within the declared 20 Gauss–Newton iterations,
12 halvings, SVD relative tolerance `1e−6`, and maximum row error
`3.58e−6`. The original C+Q is recomputed after the **actual neural** step;
every one of 24 rounds is nonincreasing, so this filter avoids the earlier
post-GAN-only acceptance ambiguity. Fixed diagnostic evaluation uses the
same 4096 particle indices and late `.029` output-noise draws at clocks
241–248 for each case. The cold-1 margin is narrow: HQ falls from .95313 in
round one to a minimum .91138, just above the .9 gate.

The passing warm state is **not at rest under this sampled objective**.
Its first bank triggers 12 relocation operations on 10 distinct particles;
the output target asks for a maximum 5.640-unit row move and changes joint
G/prior parameters by norm 3.672. Even matching particles before and after
by the best permutation leaves output RMS displacement 1.139. The fixed-bank
subsequent rounds settle, but new minibatches may request fresh large
reallocations. Passing quality for eight rounds on one bank does not test
stochastic stationarity, retention of Adam moments, or uninterrupted cold
acquisition. Unlike the distinct-anchor objective, generic C+Q has no
all-modes local-minimum guarantee from the anchor geometry proof. Those
limitations must remain explicit before considering a host controller.

The [manifest](continuous-evidence/chamfer-joint-landing/manifest.json)
contains deterministic gzip copies of all three raw cases, declaration,
summary, run and test logs, and eleven exact source/test dependencies.
Every entry records compressed and decompressed SHA-256 and size. The
driver also verifies the cold/warm state archive hashes, tests its source
hashes again after running, and checks the saved state and global CPU RNG
unchanged. Focused helper tests, including nonfinite-trial recovery, pass
**8/8**. The numerical helper is scoped to this deterministic buffer-free
MLP; its generic behavior for stochastic or mutable-buffer generators is
not established here.

Reproduce from the tracked cold and compact warm state archives named in the
declaration:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
/tmp/pr38-default-env/bin/python -u reports/toy100/chamfer_joint_landing_filter.py \
  --source-root . \
  --output NEW_SAVED_STATE_OUTPUT
```
