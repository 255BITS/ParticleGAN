# Evidence identity and reproduction

## Follow-up evidence

The [improvement ledger](improvements.json) hashes every input addendum and
preserves all original catalog/media/training receipts. Current sorted results
are in [IMPROVEMENTS.md](IMPROVEMENTS.md). The compact
[publication validation](improvement-validation.json) binds the 53-test log,
35 Python sources, immutable original receipts and 107 decoded training GIFs.
Full vector rescoring curves stay in
the external archive bound by [vector-quality-controls.json](vector-quality-controls.json);
only endpoints, suffixes, controls and provenance are committed.

Fresh source families use known develop `6ec7e578`, CPU1, original sources,
seeds and budgets. See [sign/landing/native](source_families/README.md),
[routed/ring](SOURCE_FAMILY_TRAINING.md), [word/Gaussian](SOURCE_DEMOS.md),
[sparse/denoising](conditional_sources/README.md),
[trajectory/transition](route_sources/README.md) and
[paired transport](paired_sources/README.md) for
their distinct recipes, actual priors, sampling laws, caps and effective
optimizer overrides. A recipe dictionary alone is not the effective optimizer
manifest: native pretraining uses 250 updates at LR .001 although its recipe
describes the 400-update finetuning arm at .0006, and scalar gains also override
default rates. Their exact source hashes bind those original choices.

The frozen evaluator bytes used by these runs are preserved in the durable
external archive, with original SHA `7e6b7f35…`, in
[evaluator-archive.json](source_families/evaluator-archive.json). AST comparison
confirms the used Gaussian, word, paired, landing, ring and signed-code scoring
functions are unchanged. Later calibration controls and the corrected
twelve-atom two-pole contract are separately bound; old runs are not rebound.
Demo supervision interruptions and engineering recoveries retain both costs
and exact matching prefixes. No endpoint/seed winner is selected.

All 17 source-reviewed entries have exact-source attempts. Sparse identity,
split symbols and both transitions hit their declared caps, retaining partial
states without a full-budget verdict. Both denoising and trajectory laws stop
at the original CUDA prerequisite before any model update. Affine/swirl each
complete the original 6,000-update baseline/movable validation job; the full
twelve-job protocol and unopened test split receive no qualification credit.
The ledger hashes every current actual-state GIF and lists missing entries;
original media receipts are unchanged.

The [canonical selection](canonical-selection.json) binds all 109 original
cases and five exact shared-law groups. Its 97 entries are a conservative
inventory after proved aliases, not 97 independent scientific families.
Data-law grouping can ignore uniform-template mode order; execution reuse
requires the original ordered pixels and every effective recipe/control.
Related conditional exposure and unit-change variants remain separate where
their complete laws differ or equality is unproved.

The [frozen bars alias proof](frozen-image-alias.json) binds the dispatcher,
43 training dependencies, actual runtime, seed/initialization, prior/optimizer,
target/noise/clamp, serving/EMA, complete budget/cadence and gates. The new
runner writes only alias summary metadata; its result/cloud/spec pointers
remain those of the actual canonical observation. It neither creates another
raw capture nor supplies independent qualification. Historical repeated costs
remain in their original receipts. Changed or missing identity restores
separate execution; the public-v2 cohort remains separate.
The [retention validation](retention-validation.json) binds the 99-test log,
current selection/dispatcher sources, original-byte checks and integrated
real-capture alias verification. No new scientific campaign ran for cleanup.

The [local merge receipt](merge_readiness/local-integration.json) is a
prospective offline Git tree with byte comparisons to the tested overlay;
**no remote PR was merged**. Fresh authenticated heads and required checks are
unknown while GitHub access is unavailable. No production config or library
repair belongs to this audit.

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

### Later PR226/227 cohort

The [addendum](PR226_PR227.md) uses develop
`6ec7e5788e14ea15ddc3e16ac71110458108b6a6` and the two exact proposal heads,
recorded in [convergence-addendum.json](convergence-addendum.json). Combined
inventory now contains the original 134 records plus these two supplemental
records. Original package hashes, training results and sampling laws are unchanged.

PR226 retains Python3.11.15/Torch2.11.0 CPU training. Its 17 bound artifact files,
48 metric observations and 1,200 caller-chain records per profile were checked.
The 16-frame GIF uses recorded clean reporting metrics; intermediate output
tensors were not retained. No cloud reconstruction is implied.

PR227 retains Python3.12.13/Torch2.13.0 CPU training. The observer restores all
136 saved states, independently scores 132 held-out rows under all four actual
critic weights, and verifies state/RNG purity. Both original/recovered artifact
identities and the sole fresh-policy scoring repair are checked. Its 33-frame
GIF displays fixed every-128th-coordinate teacher/model edit pairs. Metrics use
all test contexts with the original private Gaussian panels. The signed-gate
counterexample substitutes synthetic ablation scores and is separately labelled;
it is not a trained alternative.

Both were reviewed on Python3.12.13/Torch2.13.0. There were **zero new training
updates**, no seed change and no Atlas substitution into their routed host.
The full learned opt-in subprocess campaign was not rerun. Its existing
tensor-based assertion was executed on retained states, together with 39 passing
focused software tests and one explicitly skipped opt-in training test.

Observation-only reproduction, from this audit checkout with the two pinned
proposal checkouts and retained local artifacts:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m benchmarks.toy_audit.review_convergence 226 \
  --source /tmp/particlegan-toy-review-226 \
  --artifacts /ml2/hypergan/ParticleGAN-paired-residual-toy-20261001/artifacts/paired-residual-toy/convergence \
  --output /ml2/hypergan/toy-audit-artifacts-20261001/pr226-review \
  --media reports/toy_audit/media > /ml2/hypergan/toy-audit-pr226-review.log 2>&1
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m benchmarks.toy_audit.review_convergence 227 \
  --source /tmp/particlegan-toy-review-227 \
  --artifacts /ml2/hypergan/ParticleGAN-convergence-toy-develop/runs/routed-convergence-v1 \
  --candidate /ml2/hypergan/ParticleGAN-convergence-toy-develop/runs/routed-convergence-neutral-v1 \
  --output /ml2/hypergan/toy-audit-artifacts-20261001/pr227-review \
  --media reports/toy_audit/media > /ml2/hypergan/toy-audit-pr227-review.log 2>&1
```

The pinned proposal reproduction sources generate the declared evidence
on a fresh machine; raw checkpoints/JSONL remain outside Git. To regenerate
the combined catalog, add
`--addenda reports/toy_audit/convergence-addendum.json` to the catalog command
below. Media receipts now include these two GIFs.

### Original cohort capture

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

The 90 original training GIFs show actual recorded training checkpoints, with update
labels, sample/template/target views where available, numerical curves and
full-budget status. There is **no cloud or metric interpolation**. The nine
behavioral hosts have numerical curves rather than sample clouds; these are
labelled accordingly. Live and EMA are never pooled for a passing verdict.
The native mode zoom uses fixed mode index 0, not the best-looking component.
Static PNG posters are the final recorded frame.

Historic circle/sprite endpoint media is copied from the exact PR heads and
labelled `historical-rollout`. It does not count among the 90 training GIFs
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
  --reference frozen --separate-execution \
  --output /tmp/toy-artifacts/develop-frozen-reference

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
