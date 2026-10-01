# Evidence identity and reproduction

## Scope and identities

- Develop code: `4b16312e56328a679b92da69a287c0c9490259d9`.
- Open PR snapshot: 134 exact heads and their bases in
  [pull_requests.json](pull_requests.json). Every row has a complete local
  Python-file diff, avoiding GitHub's first-100-files limitation. Deep inspection
  and fresh replays concern the 47 problem/counterexample PRs; the remaining
  rows are scope classification, not algorithm validation.
- Runtime: Python 3.12.13, Torch 2.13.0+cu126; NumPy, SciPy, Matplotlib and
  Pillow. CPU transfer/behavior hosts use one Torch/BLAS thread and
  `MKL_CBWR=AVX2`. GPU native/MisGAN runs use RTX A6000 devices; physical GPU
  indices are selected with `CUDA_VISIBLE_DEVICES`, so `cuda:0` in a receipt
  names the first **visible** GPU. Shared-device wall times are not a speed ranking.
- Per-case specs/recipe, prior, initialization, budget, clean/live/EMA/served
  law, final metrics and source result/capture hashes are in
  [catalog.json](catalog.json). Proposal source hashes are in
  [source-receipts.json](source-receipts.json); media hashes/frame counts are in
  [media/media.json](media/media.json).
- One frozen training seed per task. No seed-only training experiment or
  success-seed selection. Separate evaluator/visualization RNGs are draws,
  not additional training-seed experiments.

## How observations were captured

The transfer hook calls the original evaluator first and records its actual
clean live draw. Image hosts additionally enumerate the existing finite
particle table, under an RNG fork and with module modes preserved. Existing
live/EMA ordering and all 24 declared checkpoints are checked; neither the
training loop nor target/threshold definitions are replaced. The wrapper
preserves the original call signature, including reserved-case authorization.

Two parity tests compare complete image/vector numerical losses, final metrics
and observation sequences with and without capture; only timestamps and wall
times are excluded. A third verifies that a completed original script's
`SystemExit(1)` preserves its scientific failure and recorded observations,
rather than losing the receipt. The image/vector parity probes are short
software checks, not full-budget scientific positives. See
[test_toy_audit.py](../../tests/test_toy_audit.py).

Ten early original image scripts used `sys.exit` after writing all observations;
the initial wrapper did not finalize their replay receipt. Both completed
evaluator captures and the original contrast-exit rule reconstruct those
receipts explicitly, without rerunning or changing the measured arm. The
updated wrapper handles this case. Initial instrumentation-only errors
(reserved signature forwarding, native capture event naming, serialization
of infinite controller diagnostics) stay in the local artifact directory and
are not reported as defects in the toy/model.

The 13 vector proposals first produce their original removed-field error.
An explicit second replay changes only the script's recipe-factory import to
`benchmarks.gan_v3.gan_v3_recipe`. The original/adapted source hashes and
parameter identities are retained. This is a historical recipe cohort on
current numerical hosts, not public KA2/Atlas qualification.

Develop transfer reference GIFs use each host's frozen rates/recipe and the
published architecture where specified. An earlier separate **public-v2
optimizer comparison** has different identity and is preserved in
[separate-public-v2-comparison.json](separate-public-v2-comparison.json); it is
excluded from the frozen-reference GIF/ranking cohort. No failing result is
silently relabelled as a config or formulation failure.

Former reserved annulus, alternate-cadence and residual-image cases are now
**seen audit data**. Their passes are not unseen-family generalization credit
and must not be reused as an untouched holdout.

Atlas uses PR223's explicit affine native fixture: 20,000-row square-uniform
prior, identity linear G, the specified 128×3 Fourier critic with deterministic
orthogonal initialization, current Atlas policy, and external 7,000-update
stationary budget. Clean and independently noisy **served** samples are
separate; this is not the ordinary transfer hosts' EMA/live law. Diagnostic
clouds use 4,096 fixed draws (latent stream 77, output-noise stream 78); gates
use separate 20,000 draws (streams 1637/1636), every 250 updates. Fresh native
source hashes are retained with the local summaries. The moving 1,500-update
fixture shifts the target 30 degrees after updates 500 and 1,000 and also
records fixed-model target-jump frames. Its original weaker moving criterion
and the strict native gate remain separate.

MisGAN preserves the proposal's original 7,000-update three-pair loop with
the current public recipe. Captures hook its existing EMA evaluator every
500 updates. The conditional visual shows the same first 16 ambiguous test
rows, each with 16 imputation draws; selection uses the observation mask,
not learned success. Bayes/mean controls are analytic evaluations on the same
fixed 10,000 test rows, not trained arms. PR224 uses its specified deterministic
96-update native controller game and two original controls.

## GIF interpretation

The 88 training GIFs show actual recorded training checkpoints, with update
labels, sample/template/target views where available, numerical curves and
full-budget status. There is **no cloud or metric interpolation**. The nine
behavioral hosts have numerical curves rather than sample clouds; these are
labelled accordingly. Live and EMA are never pooled for a passing verdict.
The native mode zoom uses fixed mode index 0, not the best-looking component.
Static PNG posters are the final recorded frame.

Historic circle/sprite endpoint media is copied from the exact PR heads and
labelled `historical-rollout`. It does not count among the 88 training GIFs
and does not fill either proposal's missing current convergence evidence.

## Reproduction commands

Run from the audit checkout with the listed dependencies and an activated
Python 3.12 environment. Use an **external** artifact root; bulk tensors,
checkpoints, stdout, JSONL and per-update streams
must stay out of Git. For this execution the root was
`/ml2/hypergan/toy-audit-artifacts-20261001`, with tail-able logs
`/ml2/hypergan/toy-audit-*.log`. Full raw artifacts are local, not claimed as a
publicly hosted archive. The committed code, exact heads, final metrics and
media are sufficient to rerun; raw-source hashes bind the original execution.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_CBWR=AVX2
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Extract immutable source overlays. Fetches exact commits only when absent.
python -m benchmarks.toy_audit.fetch --output /tmp/toy-sources

# Each of the 43 counterexample PRs; e.g. one image problem:
CUDA_VISIBLE_DEVICES='' python -m benchmarks.toy_audit.replay \
  --source /tmp/toy-sources/170 --output /tmp/toy-artifacts/pr170

# Vector original error, followed by an explicitly labelled archived replay:
CUDA_VISIBLE_DEVICES='' python -m benchmarks.toy_audit.replay \
  --source /tmp/toy-sources/45 --output /tmp/toy-artifacts/pr45
CUDA_VISIBLE_DEVICES='' python -m benchmarks.toy_audit.replay \
  --source /tmp/toy-sources/45 --output /tmp/toy-artifacts/pr45-adapted \
  --archived-recipe

CUDA_VISIBLE_DEVICES='' python -m benchmarks.toy_audit.base \
  --reference frozen --output /tmp/toy-artifacts/develop-frozen-reference

# Repeat for rotated100 and staggered100, preserving budget/initialization.
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.toy_audit.native grid100 \
  --output /tmp/toy-artifacts/native-grid100
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.toy_audit.native rotated100 \
  --steps 1500 --shift --output /tmp/toy-artifacts/native-moving-rotated100

# Repeat for mcar_p20, mcar_p80 and block: different observation laws.
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.toy_audit.proposal --pr 196 \
  --mechanism mcar_p50 --source /tmp/toy-sources/196 \
  --output /tmp/toy-artifacts/pr196-mcar_p50
CUDA_VISIBLE_DEVICES='' python -m benchmarks.toy_audit.proposal --pr 224 \
  --source /tmp/toy-sources/224 --output /tmp/toy-artifacts/pr224-v2

# These preserve the current pre-training blockers; no config repair:
CUDA_VISIBLE_DEVICES='' python -m benchmarks.toy_audit.proposal --pr 22 \
  --source /tmp/toy-sources/22 --output /tmp/toy-artifacts/pr22-current
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.toy_audit.proposal --pr 153 \
  --device cuda:0 --source /tmp/toy-sources/153 \
  --output /tmp/toy-artifacts/pr153-current

CUDA_VISIBLE_DEVICES=1 python -m benchmarks.toy_audit.oracle \
  --source /tmp/toy-sources/196 --device cuda:0 \
  --output reports/toy_audit/misgan-oracles.json
CUDA_VISIBLE_DEVICES='' python -m benchmarks.toy_audit.scorers \
  --artifacts /tmp/toy-artifacts --output reports/toy_audit/scorer-controls.json
CUDA_VISIBLE_DEVICES='' python -m benchmarks.toy_audit.render \
  --artifacts /tmp/toy-artifacts --output reports/toy_audit/media --cohort all

CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_toy_audit.py
```

For the complete gallery, replay every problem PR listed in the inventory;
only the 13 vector PRs use the archived-recipe option. Rendering is read-only
with respect to training artifacts and never launches training. Scientific
ratings are an explicit review judgment, not an aggregate leaderboard of
model performance, robustness or compute efficiency.
